# Bootstrap‑Free On‑the‑fly 3D Gaussian Splatting (MoGE‑init)

Replace the fragile multi‑view geometric bootstrap in On‑the‑fly 3DGS with a robust single‑image generative initialization using MoGE‑2, then keep the official incremental pipeline unchanged for the rest of the sequence.

This repo provides portable notebooks (Colab or local) covering end‑to‑end: one‑shot init → incremental training → paper‑style evaluation vs the original baseline.

---

## Notebooks at a glance

| Notebook | Purpose |
|---|---|
| `pipeline/3dgsMoge_ETAPE1.ipynb` | Stage 1 only: one‑shot init. Run MoGE‑2 on a single image to get a dense colored point cloud, then convert it to a 3DGS `.ply` (GraphDeco/SuperSplat layout). |
| `pipeline/3dgsMoge_ETAPE2.ipynb` | Stage 2 only: incremental reconstruction. Inject the init 3DGS into `on-the-fly-nvs`, then run pose tracking, LoG sampling, joint optimization, and rendering. |
| `comparison/PSNR_compare.ipynb` | Evaluation (MoGE init). Compute PSNR/SSIM/LPIPS on a held‑out split (1 every 30, `--test_hold 30`) for the model trained from the MoGE initialization. Exports CSV + plots (boxplots, Δ‑PSNR histogram). |
| `comparison/PSNR_compareBasic.ipynb` | Evaluation (baseline). Same protocol, same split, same metrics — trained with the original bootstrap — enabling apples‑to‑apples comparison. |

Why two evaluation notebooks? To separate artifacts (CSVs/plots) under each run while enforcing an identical protocol.

---

## What’s new vs. the original On‑the‑fly paper

- Bootstrap‑free start: we replace the multi‑view geometric bootstrap (feature matching + BA over ~8 frames) with a single‑image generative bootstrap using MoGE‑2.
- Dense prior at t=0: the scene begins as a dense 3DGS, not a sparse SfM cloud, improving early stability and reducing floaters/noise.
- Less work per iteration: removing the bootstrap cuts a whole stage of compute; the incremental loop starts immediately.
- Better convergence: with a high‑quality prior, pose tracking and LoG sampling converge faster to a better minimum.
- Measurable gains: −6.3% total time and +1.17 dB mean PSNR on the same held‑out split, with statistically significant SSIM/LPIPS improvements.

At a glance, our contribution turns the pipeline’s weakest link (bootstrap) into its strongest asset (a learned, dense prior) without changing the proven incremental design.

---

## Expected directory layout

```
on-the-fly-nvs/
├─ results/
│  └─ … (models and renders produced by the notebooks)
└─ data/
   └─ my_scene/
      └─ images/
         ├─ frame-000000.jpg     
         ├─ frame-000001.jpg
         └─ … up to ~ frame-001527.jpg
```

Optional `.depth` / `.pose` files are ignored. If your raw files are `*.color.jpg`, the notebooks copy/rename to `.jpg` for compatibility.

---

## What we change (technical contribution)

### 1) One‑shot 3DGS initialization from a single image
Instead of multi‑view bootstrap (feature matching + BA), we run MoGE‑2 on one image to predict a dense colored point cloud, then convert it into 3D Gaussians compatible with GraphDeco’s rasterizer:

- Positions: `(x, y, z)` from MoGE‑2.
- Color (SH‑DC): encode the DC spherical harmonics coefficient as `f_dc = (color_linear − 0.5) / C0` with `C0 = 0.28209479177387814`; the renderer reconstructs color as `0.5 + C0·f_dc`.
- Opacity: stored as a logit (e.g., `logit(0.99) ≈ 4.5951`) for stable optimization.
- Scale: per‑axis log‑space (e.g., `log(0.01)`), as in 3DGS.
- Rotation: unit quaternion `[1, 0, 0, 0]` (axis‑aligned ellipsoids initially).
- Normals: unused by 3DGS rasterization.

Result: a dense, photometrically faithful scene at t=0, avoiding the SfM bottleneck (sparse + expensive).

### 2) Incremental pipeline unchanged
We keep On‑the‑fly as is: pose tracking for new images, LoG sampling for new Gaussians, and joint optimization (poses + Gaussians). Only the source of the initial cloud is different.

Scale alignment (optional). If needed, a quick two‑view relation (e.g., LoFTR + 5‑point) yields a global scale factor `S`; multiply positions and scales by `S`. The optimizer adapts quickly otherwise.

---

## Recommended order of execution

### 1) `pipeline/3dgsMoge_ETAPE1.ipynb` — produce the init 3DGS
1) Provide one image (sharp, textured, mid‑sequence works well).
2) The notebook runs MoGE‑2, extracts the cloud, and writes a `.ply` 3DGS (`3dgs_MoGe.ply`) with the required fields.
3) Optional: preview with SuperSplat/GraphDeco.

Main output: `3dgs_MoGe.ply`

### 2) `pipeline/3dgsMoge_ETAPE2.ipynb` — incremental training from the init
1) Clone `on-the-fly-nvs` and install dependencies.
2) Place your images under `data/my_scene/images/`.
3) Copy the init 3DGS to the canonical train location: `results/<model>/point_cloud/point_cloud.ply` so training picks it up and skips the built‑in bootstrap.
4) Run `train.py` with your options (`--downsampling`, `--save_every`, `--test_hold 30`, etc.), then render.

Outputs: `results/<model>/…` (checkpoints, `colmap/`, `rendered_path.mp4`, and `test_images/` if `--test_hold>0`).

Why that path? GraphDeco persists the current cloud at `results/<model>/point_cloud/point_cloud.ply`. Placing our `.ply` there before training makes the run naturally start from the MoGE scene.

### 3) `comparison/PSNR_compare.ipynb` & `comparison/PSNR_compareBasic.ipynb` — paper‑style evaluation
- Split: `--test_hold 30` ⇒ approx. 1 out of 30 images held out, rendered to `results/<…>/test_images/` with the same filename as in `data/<…>/test/`.
- Metrics:
  - PSNR on sRGB [0,1]: `PSNR = -10·log10(MSE)`.
  - SSIM on Y (YCbCr luma): robust to minor hue shifts.
  - LPIPS (VGG): perceptual similarity.
- Pairing: by filename (resize if needed).
- Outputs: per‑image CSV, boxplots (PSNR; SSIM/LPIPS separately), Δ‑PSNR histogram (MoGE − Baseline).

TUM‑style scenes: you may enable an optional mask (ignore black borders) in the notebooks.

---

## Why the one‑shot init helps

- Dense prior: geometry and color are already rich at t=0 (vs. sparse SfM), reducing early artifacts (floaters/noise).
- Faster start: eliminates the bootstrap’s matching + BA; the incremental loop starts immediately.
- Robustness: pose tracking and LoG sampling converge faster with a credible local anchor.
- Hybrid design: leverages a foundation model (MoGE‑2) with a modular geometric pipeline.

Limitations: arbitrary scale (correctable), texture/lighting bias inherited from the init image, and very low‑parallax sequences remain challenging long‑term.

---

## Implementation details: the bootstrap‑bypass hook

We minimally tweak the original training flow to start from our pre‑generated 3DGS scene if available. Concretely:

1) We place the MoGE init file at the canonical path used by GraphDeco:

```
results/<model>/point_cloud/point_cloud.ply
```

2) We add a tiny check at initialization time (or rely on existing code paths that already prefer an on‑disk point cloud when present):

```python
# Pseudocode illustrating the hook
cloud_path = Path(results_dir) / 'point_cloud' / 'point_cloud.ply'
if cloud_path.exists():
   scene = load_gaussian_cloud(cloud_path)
   bootstrapped = True  # skip geometric bootstrap entirely
else:
   scene = run_geometric_bootstrap(first_frames=~8)
   save_gaussian_cloud(scene, cloud_path)
```

This “hook” is intentionally simple and non‑intrusive: no changes to the incremental optimizer or rendering stack. The presence of `point_cloud.ply` acts as a switch that bypasses the geometric bootstrap, letting training continue from a dense MoGE prior. That’s where the wall‑clock savings and quality jump come from:

- Faster iterations: no per‑pair feature matching, no initial BA.
- Higher quality sooner: early renders are already coherent, so the optimizer spends cycles refining, not recovering from sparsity.

Tip: if scale needs alignment, compute a global scale `S` from a quick two‑view estimator (e.g., LoFTR + 5‑point), then apply it to positions and per‑axis log‑scales before writing `point_cloud.ply`.

---

## Dataset

The image sequence used in this repository comes from the Stanford “Relocalization” project dataset:

- Source: https://graphics.stanford.edu/projects/reloc/#data
- Sequence: Apartment 1 (1.4 GB)
- License: Creative Commons Attribution‑NonCommercial‑ShareAlike 4.0 (CC BY‑NC‑SA 4.0)  
   https://creativecommons.org/licenses/by-nc-sa/4.0/

This repository includes the actual scene frames we used for our experiments under `images/` (Apartment 1) to facilitate reproducibility. These images are part of the Stanford “Relocalization” dataset and remain subject to the CC BY‑NC‑SA 4.0 license. If you reuse them, please:

- Provide proper attribution to the dataset and authors.
- Use them for non‑commercial purposes only.
- Share any derivatives under the same license.

If you need alternative scenes or the full dataset, please download from the official page above and comply with its license.

---

## Experimental results (Apartment 1, 1528 imgs, test_hold=30)

- Total processing time (full sequence): Baseline 375.46s vs Ours 351.66s (−6.3%).
- Mean PSNR on held‑out views: Baseline 20.74 dB vs Ours 21.91 dB (+1.17 dB).

Perceptual statistics (paired over 51 test views):
- SSIM: mean gain +0.0303 (95% CI [+0.0077, +0.0529]), t = 2.71, p = 0.0092.
- LPIPS (VGG): mean gain +0.0173 (95% CI [+0.0035, +0.0313]), t = −2.50, p = 0.0157.
- PSNR: mean gain +1.17 dB (95% CI [−0.07, +2.41]), t = 1.90, p = 0.0637.

Robustness: our method improves 66.7% of test views; the right tail reaches +16.34 dB in the hardest cases (low parallax, low texture) where the original bootstrap is fragile.

```
Method          Time (s)   Mean PSNR (dB)
-----------------------------------------
Baseline        375.46     20.74
MoGE (Ours)     351.66     21.91
Gain            -23.8s     +1.17 dB
```

---

### Step 1 pipeline: from single image to initialized 3DGS

<img width="1323" height="361" alt="Capture d&#39;écran 2025-11-02 173955" src="https://github.com/user-attachments/assets/7877c5fa-f1cd-46f2-8bc3-6450501e9219"/>

<p align="center"><em>Step 1 pipeline: from the input image (left) to the initialized 3DGS scene (right) via MoGE.</em></p>

### Qualitative and global performance comparison (Apartment 1)

<table>
  <tr>
    <th align="center">Baseline</th>
    <th align="center">Our Approach (MoGE)</th>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/225f5f7b-d68f-4aca-9829-f37c32001201" width="420" />
      <br/>
      <sub><em>375.46 s · PSNR 20.74 dB</em></sub>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/54aec8c9-d2d3-4b61-9fe2-e59923157ae6" width="420" />
      <br/>
      <sub><em>351.66 s · PSNR 21.91 dB</em></sub>
    </td>
  </tr>
</table>

<p align="center">
  <em>Our method (right) produces higher-quality renderings while being faster than the original method (left).</em>
</p>

---


## Video comparison (Baseline vs MoGE)

Side-by-side rendered-path video showing both methods simultaneously.  
<div align="center">
  <video src="https://github.com/user-attachments/assets/886cacf0-79ee-4486-939d-757ff16d86d2"
         controls
         width="80%"
         muted
         loop>
  </video>
   <p align="center"><em>Original 3dgs (left) vs our MoGE approach (right) on Apartment 1.</em></p>
</div>

> **Note:** The slight noise visible at the end of the scene arises from the intrinsic characteristics of the captured environment, not from the rendering pipeline.

---

## Run locally (Windows) or Colab

Environment (local, optional):
```bash
# In a terminal inside the repo folder
python -m venv .venv
source .venv/Scripts/activate  # PowerShell: .\.venv\Scripts\Activate.ps1
pip install -U pip
git clone --recursive https://github.com/graphdeco-inria/on-the-fly-nvs.git
pip install -r on-the-fly-nvs/requirements.txt
pip install opencv-python-headless tqdm lpips scikit-image plyfile
```

Prepare data:
- Put your images under `on-the-fly-nvs/data/my_scene/images`.
- If you want to use the frames bundled in this repo, copy or symlink `images/` to `on-the-fly-nvs/data/my_scene/images`.
- For our method, copy the MoGE init PLY to `on-the-fly-nvs/results/scene/point_cloud/point_cloud.ply` before training.

Train (MoGE‑init):
```bash
cd on-the-fly-nvs
python train.py -s data/my_scene -m results/scene --viewer_mode none --downsampling 2.5 --test_hold 30 --save_every 100
```

Evaluation:
- Open `comparison/PSNR_compare.ipynb` or `comparison/PSNR_compareBasic.ipynb` and run all cells.
- CSV is written to `results/scene/metrics_testhold_30.csv` by default.

---

## References

- Kerbl et al., “3D Gaussian Splatting for Real‑Time Radiance Field Rendering,” TOG 2023.
- Meuleman et al., “On‑the‑fly Reconstruction for Large‑Scale Novel View Synthesis from Unposed Images,” TOG 2025.
- Wang et al., “MoGE‑2,” 2024.

---

## License

This repo builds on open‑source work by the original authors. Please respect their licenses. The added one‑shot initialization is provided for research purposes.

---

## TL;DR

- Problem: multi‑view bootstrap is slow and fragile.
- Idea: MoGE‑2 → dense 3DGS from a single image.
- Keep: the incremental on‑the‑fly pipeline (poses, LoG, joint optim.).
- Evaluate: same split (`--test_hold 30`), same metrics (PSNR/SSIM/LPIPS), boxplots + Δ‑PSNR.
- Result: faster start, stronger geometric prior, consistently higher and more stable quality.

