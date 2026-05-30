"""
Inspect a retargeted pkl file and write its contents to a txt file.
Usage: python inspect_retarget_pkl.py <path/to/file.pkl> [output.txt]
"""
import sys
import pickle
import numpy as np
import torch

pkl_path = sys.argv[1]
out_path = sys.argv[2] if len(sys.argv) > 2 else pkl_path.replace(".pkl", "_inspect.txt")

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

lines = []
lines.append(f"File: {pkl_path}")
lines.append(f"Type: {type(data)}")
lines.append("")

def describe(val):
    if isinstance(val, torch.Tensor):
        return f"Tensor  shape={tuple(val.shape)}  dtype={val.dtype}  min={val.float().min().item():.4f}  max={val.float().max().item():.4f}"
    elif isinstance(val, np.ndarray):
        return f"ndarray shape={val.shape}  dtype={val.dtype}  min={val.min():.4f}  max={val.max():.4f}"
    elif isinstance(val, (list, tuple)):
        return f"{type(val).__name__}  len={len(val)}  (first: {type(val[0]).__name__ if val else 'empty'})"
    else:
        return f"{type(val).__name__}  val={val}"

def dump(obj, indent=0):
    prefix = "  " * indent
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict):
                lines.append(f"{prefix}[{k}]:")
                dump(v, indent + 1)
            elif isinstance(v, (torch.Tensor, np.ndarray)):
                lines.append(f"{prefix}{k}: {describe(v)}")
                # print first few values
                arr = v.numpy() if isinstance(v, torch.Tensor) else v
                flat = arr.reshape(-1)
                preview = "  ".join(f"{x:.4f}" for x in flat[:8])
                if len(flat) > 8:
                    preview += "  ..."
                lines.append(f"{prefix}  [{preview}]")
            else:
                lines.append(f"{prefix}{k}: {describe(v)}")
    else:
        lines.append(f"{prefix}{describe(obj)}")

dump(data)

with open(out_path, "w") as f:
    f.write("\n".join(lines))

print(f"Written to {out_path}")
