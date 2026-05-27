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
from main.dataset.oakink2_dataset_dexhand_rh import OakInk2DatasetDexHandRH
from main.dataset.oakink2_dataset_dexhand_lh import OakInk2DatasetDexHandLH

TABLE_SURFACE_Z = 0.6  # default from env config
mujoco2gym = np.eye(4)
mujoco2gym[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(np.array([np.pi / 2, 0, 0]))
mujoco2gym[:3, 3] = np.array([0, 0, TABLE_SURFACE_Z])
mujoco2gym_transf = torch.tensor(mujoco2gym, dtype=torch.float32)

INDICES = [
    "8a043@3",
    "9a028@13",
    "380a3@3",
    "82851@13",
    "e6d2b@3",
    "e9aab@2",
    "fbc74@2",
    "1b938@2",
    "59a87@7",
    "8e463@4",
    "ce343@2",
    "0fb0e@3",
    "4e42c@2",
    "9a028@9",
    "7a6dd@3",
    "c8fd4@11",
    "662b0@6",
    "959a0@8"
]

from maniptrans_envs.lib.envs.dexhands.inspire import InspireRH, InspireLH
DATA_DIR = "/home/jsyu/ManipTrans/data/OakInk-v2"
ds_rh = OakInk2DatasetDexHandRH(device="cpu", mujoco2gym_transf=mujoco2gym_transf, dexhand=InspireRH(), data_dir=DATA_DIR)
ds_lh = OakInk2DatasetDexHandLH(device="cpu", mujoco2gym_transf=mujoco2gym_transf, dexhand=InspireLH(), data_dir=DATA_DIR)

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
