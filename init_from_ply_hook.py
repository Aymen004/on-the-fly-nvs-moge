
import os

def maybe_use_init_ply(model_path: str) -> bool:
    """
    If results/<model>/point_cloud/point_cloud.ply exists, start training from it
    and skip the geometric bootstrap stage. Pure behavior toggle; no scene data here.
    """
    pc = os.path.join(model_path, "point_cloud", "point_cloud.ply")
    return os.path.exists(pc)
