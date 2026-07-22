"""Print the sequence length (frames + seconds) for raw capture demo pkl(s).

Reads the .pkl directly with pickle -- no isaacgym / torch / dataset loader.

Usage:
    python data_stats/inspect_demo_lengths.py data/my_dataset/cap_5_0626_m_170805.pkl
    python data_stats/inspect_demo_lengths.py data/my_dataset/*.pkl
"""
import argparse
import os
import pickle

FPS = 60.0  # native capture rate


def frame_count(raw):
    """Number of frames in a raw capture pkl, however it stores them."""
    if raw.get("obj_transf"):
        return len(next(iter(raw["obj_transf"].values())))
    if "n_frames" in raw:
        return int(raw["n_frames"])
    if raw.get("hands"):
        return len(next(iter(raw["hands"].values()))["wrist_pos"])
    raise KeyError("no obj_transf / n_frames / hands found to count frames")


parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("pkls", nargs="+", help="raw capture pkl path(s)")
args = parser.parse_args()

print(f"{'File':<44}  {'frames':>7}  {'seconds':>8}")
print("-" * 64)
for path in args.pkls:
    try:
        with open(path, "rb") as f:
            raw = pickle.load(f)
        n = frame_count(raw)
        print(f"{os.path.basename(path):<44}  {n:>7}  {n / FPS:>7.2f}s")
    except Exception as e:
        print(f"{os.path.basename(path):<44}  ERROR: {type(e).__name__}: {e}")
