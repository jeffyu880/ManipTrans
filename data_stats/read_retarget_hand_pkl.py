import pickle
import numpy as np
import sys

path = sys.argv[1] if len(sys.argv) > 1 else (
    "data/retargeting/OakInk-v2/mano2inspire_lh/"
    "scene_01__A001++seq__4c966751912ef4687b87__2023-04-27-18-41-04@0.pkl"
)

with open(path, "rb") as f:
    data = pickle.load(f)

print(f"Keys: {list(data.keys())}")
print()
for k, v in data.items():
    v = np.array(v)
    print(f"  {k}: shape={v.shape}  min={v.min():.4f}  max={v.max():.4f}  mean={v.mean():.4f}")

# Check object velocity if present
if "obj_velocity" in data:
    vel = np.linalg.norm(data["obj_velocity"], axis=-1)
    ang_vel = np.linalg.norm(data["obj_angular_velocity"], axis=-1)
    print(f"\nobj_vel   norm: min={vel.min():.2f}  max={vel.max():.2f}  frame0={vel[0]:.2f}")
    print(f"obj_angvel norm: min={ang_vel.min():.2f}  max={ang_vel.max():.2f}  frame0={ang_vel[0]:.2f}")
