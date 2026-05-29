"""
Dump the full contents of a rollouts.hdf5 file.
Usage: python analyze_rollouts.py <path/to/rollouts.hdf5>
"""
import sys
import h5py
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "rollouts.hdf5"

def print_item(name, obj):
    if isinstance(obj, h5py.Dataset):
        arr = np.array(obj)
        print(f"{name}  shape={arr.shape}  dtype={arr.dtype}")
        print(f"  {arr}")
    else:
        print(f"[group] {name}/")

with h5py.File(path, "r") as f:
    f.visititems(print_item)
