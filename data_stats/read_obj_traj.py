import pickle
import numpy as np
import sys
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

# Default to the 4c966 sequence used in training
anno_path = sys.argv[1] if len(sys.argv) > 1 else (
    "data/OakInk-v2/anno_preview/"
    "scene_01__A001++seq__4c966751912ef4687b87__2023-04-27-18-41-04.pkl"
)

anno = pickle.load(open(anno_path, "rb"))
print(f"Anno keys: {list(anno.keys())}")
print()

# Object transforms: dict mapping obj_id -> {frame_id -> 4x4 matrix}
obj_transf = anno["obj_transf"]
print(f"Objects in scene: {list(obj_transf.keys())}")
print()

for obj_id, frames in obj_transf.items():
    frame_ids = sorted(frames.keys())
    T = len(frame_ids)
    # Stack into (T, 4, 4)
    traj = np.stack([frames[f] for f in frame_ids])  # (T, 4, 4)
    pos = traj[:, :3, 3]  # (T, 3)

    # Compute angular velocity via finite differences (same as base.py)

    r = traj[:, :3, :3]  # (T, 3, 3)
    diff_r = r[1:] @ r[:-1].swapaxes(-1, -2)  # (T-1, 3, 3)
    diff_angle = Rotation.from_matrix(diff_r).magnitude()  # (T-1,) in radians

    skip = 2
    time_delta = skip / 120.0
    ang_vel_mag = diff_angle / time_delta  # rad/s between consecutive frames

    print(f"obj_id: {obj_id}  frames: {T}")
    print(f"  position range: x=[{pos[:,0].min():.3f}, {pos[:,0].max():.3f}]  "
          f"y=[{pos[:,1].min():.3f}, {pos[:,1].max():.3f}]  "
          f"z=[{pos[:,2].min():.3f}, {pos[:,2].max():.3f}]")
    print(f"  angular velocity (rad/s): min={ang_vel_mag.min():.2f}  max={ang_vel_mag.max():.2f}  "
          f"frame0={ang_vel_mag[0]:.2f}  frame1={ang_vel_mag[1]:.2f}")
    # Find frames with very high angular velocity
    high_vel_frames = np.where(ang_vel_mag > 100)[0]
    if len(high_vel_frames) > 0:
        print(f"  HIGH ANG VEL frames (>100 rad/s): {high_vel_frames[:10]}")
    print()
