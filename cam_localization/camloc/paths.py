"""Where the trained model lives. Everything the localizer reads at run time is under MODEL (default
<repo>/cam_localization/model, override with CAMLOC_MODEL=<dir>):

    features/db_<cam>.npz       keyframe map: per keyframe the SuperPoint keypoints, descriptors and 3D points
    features/global_<cam>.npy   DINOv2 global descriptor of every frame of the training run (keyframe rows used)
    masks/<cam>.png             car-body masks
    kf_poses.npz                pose of every map keyframe
    motion_calib.json           car state -> base_link motion calibration
    night_stats.json            median grey of the night keyframes per camera (exposure normalisation)
    day/day_<cam>.npy           day descriptor layer (Qwen day-styled training run), aligned with db_<cam>.npz
                                (other directory: CAMLOC_DAY_DIR=<dir>)
    track_bounds.npz            track / pit-lane borders (used when the bag has no /track_bounds)
"""
import os

PKG = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(PKG)
MODEL = os.path.abspath(os.environ.get('CAMLOC_MODEL', os.path.join(PROJECT, 'model')))
CONFIGS = os.path.join(PROJECT, 'configs')
FEAT = os.path.join(MODEL, 'features')
MASKS = os.path.join(MODEL, 'masks')
DAY = os.path.abspath(os.environ.get('CAMLOC_DAY_DIR', os.path.join(MODEL, 'day')))
