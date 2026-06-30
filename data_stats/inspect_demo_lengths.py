"""Print the sequence length (in frames at 60Hz) for each demo index."""
import sys
import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir("/home/jsyu/ManipTrans")
sys.path.insert(0, "/home/jsyu/ManipTrans")
import isaacgym  # must be before torch
import numpy as np
import torch
from main.dataset.transform import aa_to_rotmat
from main.dataset.my_dataset_RH import MyDatasetRH
from main.dataset.my_dataset_LH import MyDatasetLH

TABLE_SURFACE_Z = 0.6  # default from env config
mujoco2gym = np.eye(4)
mujoco2gym[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(np.array([np.pi / 2, 0, 0]))
mujoco2gym[:3, 3] = np.array([0, 0, TABLE_SURFACE_Z])
mujoco2gym_transf = torch.tensor(mujoco2gym, dtype=torch.float32)

# MyDataset cap_* capping demos (mirrored), matching train_maniptrans_array.run
INDICES = [
    "m_161528", "m_161551", "m_161610",
    "m_170342", "m_170401", "m_170418", "m_170435", "m_170454", "m_170509",
    "m_170527", "m_170541", "m_170556", "m_170612",
    "m_170639", "m_170654", "m_170708",
    "m_170726", "m_170741", "m_170753", "m_170805",
]

from maniptrans_envs.lib.envs.dexhands.inspire import InspireRH, InspireLH
ds_rh = MyDatasetRH(device="cpu", mujoco2gym_transf=mujoco2gym_transf, dexhand=InspireRH())
ds_lh = MyDatasetLH(device="cpu", mujoco2gym_transf=mujoco2gym_transf, dexhand=InspireLH())

output_path = os.path.join(_SCRIPT_DIR, "demo_lengths.txt")
lines = []
lines.append(f"{'Index':<12}  {'RH frames':>10}  {'LH frames':>10}  {'RH secs':>8}  {'LH secs':>8}")
lines.append("-" * 55)
for idx in INDICES:
    rh = ds_rh[idx]
    lh = ds_lh[idx]
    rh_len = len(rh["obj_trajectory"])
    lh_len = len(lh["obj_trajectory"])
    lines.append(f"{idx:<12}  {rh_len:>10}  {lh_len:>10}  {rh_len/60:>8.2f}s  {lh_len/60:>8.2f}s")

output = "\n".join(lines)
print(output)
with open(output_path, "w") as f:
    f.write(output + "\n")
print(f"\nWritten to {output_path}")
