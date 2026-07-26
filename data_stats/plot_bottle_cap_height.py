"""Plot bottle / cap / gap along the vertical axis for a MyDataset capture.

The pkl stores transforms in the OptiTrack (Motive) frame, which is Y-up:
Motive +Y is the height axis and maps to gym +Z (see main/dataset/my_dataset_RH.py).
So the "z axis" plotted here is column 1 of the translation.

The reported vertical gap is restricted to frames where the cap is horizontally
aligned with the bottle, i.e. the in-plane (gym x/y) distance is under XY_RADIUS.

Usage:
    python data_stats/plot_bottle_cap_height.py [pkl_path] [out_png] [xy_radius_m]
"""

import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve(path):
    """Relative paths are repo-root-relative, so the script runs from any cwd."""
    return path if os.path.isabs(path) or os.path.exists(path) else os.path.join(REPO, path)


pkl_path = resolve(sys.argv[1] if len(sys.argv) > 1 else "data/my_dataset/cap_5_0702_m_164640.pkl")
out_png = resolve(sys.argv[2] if len(sys.argv) > 2 else "data_stats/vis_traj_outputs/bottle_cap_height.png")
XY_RADIUS = float(sys.argv[3]) if len(sys.argv) > 3 else 0.03

VERT = 1  # Motive +Y == gym +Z (height)
HORIZ = [0, 2]  # the remaining Motive axes span the gym x/y ground plane

data = pickle.load(open(pkl_path, "rb"))
body = data["obj_transf"]["bottle_body"][:, :3, 3].astype(np.float64)
cap = data["obj_transf"]["bottle_cap"][:, :3, 3].astype(np.float64)

t = data["timestamps_s"] - data["timestamps_s"][0]
body_z, cap_z = body[:, VERT], cap[:, VERT]
gap = cap_z - body_z

# Horizontal (in-plane) offset, and the frames where the cap is over the bottle.
xy_dist = np.linalg.norm(cap[:, HORIZ] - body[:, HORIZ], axis=1)
aligned = xy_dist <= XY_RADIUS
if not aligned.any():
    sys.exit(f"cap is never within {XY_RADIUS * 100:.1f} cm horizontally (min {xy_dist.min() * 100:.2f} cm)")

fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

axes[0].plot(t, body_z, color="tab:blue")
axes[0].set_ylabel("bottle z [m]")
axes[0].set_title(f"{pkl_path}  —  vertical axis (Motive +Y = gym +Z)")

axes[1].plot(t, cap_z, color="tab:orange")
axes[1].set_ylabel("cap z [m]")

axes[2].plot(t, gap, color="tab:green")
axes[2].axhline(0.0, color="gray", lw=0.8, ls="--")
axes[2].set_ylabel("cap z - bottle z [m]")
axes[2].set_xlabel("time [s]")

# Closest in the vertical axis, among horizontally-aligned frames only.
masked = np.where(aligned, np.abs(gap), np.inf)
i_min = int(np.argmin(masked))
axes[2].plot(t[i_min], gap[i_min], "r*", ms=12)
axes[2].annotate(
    f"min |gap| = {abs(gap[i_min]) * 1000:.1f} mm\nframe {i_min}, t={t[i_min]:.2f}s",
    xy=(t[i_min], gap[i_min]),
    xytext=(8, 14),
    textcoords="offset points",
    fontsize=9,
    color="red",
)

# Contiguous stretches where the cap is horizontally over the bottle.
edges = np.diff(np.concatenate(([0], aligned.view(np.int8), [0])))
spans = list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0] - 1))

for ax in axes:
    ax.grid(alpha=0.3)
    for lo, hi in spans:
        ax.axvspan(t[lo], t[hi], color="tab:red", alpha=0.10, lw=0)
axes[2].plot([], [], color="tab:red", alpha=0.25, lw=8, label=f"xy dist $\\leq$ {XY_RADIUS * 100:.0f} cm")
axes[2].legend(loc="upper left", fontsize=8)

fig.tight_layout()
os.makedirs(os.path.dirname(out_png), exist_ok=True)
fig.savefig(out_png, dpi=140)
print(f"saved {out_png}")
print(f"bottle z: [{body_z.min():.4f}, {body_z.max():.4f}]")
print(f"cap    z: [{cap_z.min():.4f}, {cap_z.max():.4f}]")
print(f"xy dist : min {xy_dist.min():.4f}  aligned frames (<= {XY_RADIUS:.3f} m): {int(aligned.sum())}/{len(t)}")
print(f"gap     : start {gap[0]:.4f}  end {gap[-1]:.4f}")
print(f"min |gap| while xy-aligned: {abs(gap[i_min]):.4f} m @ frame {i_min}, t={t[i_min]:.2f}s")
