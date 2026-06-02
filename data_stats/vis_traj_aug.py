"""
Visualize original vs augmented demo trajectories.

Figure 1: Wrist positions  — left=LH, right=RH (with wrist orientation arrows)
Figure 2: Object positions — left=LH object, right=RH object

Usage (from ManipTrans root):
    python data_stats/vis_traj_aug.py --data_idx 3b1e6@12 --n_aug 4 --out aug_vis.png
"""
import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

import isaacgym  # must come before torch
import torch

from main.dataset.oakink2_dataset_dexhand_rh import OakInk2DatasetDexHandRH
from main.dataset.oakink2_dataset_dexhand_lh import OakInk2DatasetDexHandLH
from main.dataset.transform import aa_to_rotmat, rotmat_to_aa
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
from maniptrans_envs.lib.envs.tasks.dexhandmanip_bih import DexHandManipBiHEnv


def build_mujoco2gym(device="cpu"):
    _table_surface_z = 0.4 + 0.015
    T = np.eye(4)
    T[:3, :3] = aa_to_rotmat(torch.tensor([[0, 0, -np.pi / 2]])).squeeze(0).numpy() @ \
                aa_to_rotmat(torch.tensor([[np.pi / 2, 0, 0]])).squeeze(0).numpy()
    T[:3, 3] = [0, 0, _table_surface_z]
    return torch.tensor(T, dtype=torch.float32, device=device)


def load_demo(data_idx, side, device="cpu"):
    mujoco2gym = build_mujoco2gym(device)
    dexhand = DexHandFactory.create_hand("inspire", side)
    Cls = OakInk2DatasetDexHandRH if side == "right" else OakInk2DatasetDexHandLH
    dataset = Cls(device=device, mujoco2gym_transf=mujoco2gym, dexhand=dexhand)
    return dataset[data_idx]


def plot_3d_traj(ax, pos, rot_aa, label, color, is_original=False, arrow_len=0.02):
    """Plot trajectory as dotted line with wrist orientation arrows every arrow_step frames."""
    x, y, z = pos[:, 0].numpy(), pos[:, 1].numpy(), pos[:, 2].numpy()
    lw = 2.5 if is_original else 1.5
    ls = ":" if is_original else "--"
    c = "red" if is_original else color
    ax.plot(x, y, z, linestyle=ls, color=c, linewidth=lw, label=label)
    ax.scatter(x[0], y[0], z[0], color=c, s=30)

    # draw wrist z-axis (palm normal) as a single arrow at the first frame
    if rot_aa is not None:
        R0 = aa_to_rotmat(rot_aa[0:1]).numpy()[0]  # [3, 3]
        z_axis = R0[:, 2]                           # local z-axis in world frame
        ax.quiver(
            pos[0, 0].numpy(), pos[0, 1].numpy(), pos[0, 2].numpy(),
            z_axis[0], z_axis[1], z_axis[2],
            length=arrow_len, color=c, alpha=0.9, normalize=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_idx", default="3b1e6@12")
    parser.add_argument("--n_aug", type=int, default=4)
    parser.add_argument("--out", default="aug_vis.png")
    parser.add_argument("--arrow_len", type=float, default=0.02, help="Arrow length in metres")
    args = parser.parse_args()

    print(f"Loading {args.data_idx}...")
    data_rh = load_demo(args.data_idx, "right")
    data_lh = load_demo(args.data_idx, "left")
    print(f"RH frames: {len(data_rh['wrist_pos'])}  LH frames: {len(data_lh['wrist_pos'])}")

    table_width_offset = 0.2
    table_surface_z = 0.4 + 0.015
    center = torch.tensor([-table_width_offset / 2, 0.0, table_surface_z])

    transforms = []
    for i in range(args.n_aug):
        torch.manual_seed(i)
        R, t, c = DexHandManipBiHEnv._sample_aug_transform("cpu", center)
        transforms.append((R, t, c))

    print(f"\nAugmentations applied ({args.n_aug} total):")
    for i, (R, t, c) in enumerate(transforms):
        angle_deg = float(np.degrees(np.arctan2(R[1, 0].item(), R[0, 0].item())))
        print(f"  aug {i+1}: rot={angle_deg:+.2f}°  Δx={t[0].item()*100:+.1f}cm  Δy={t[1].item()*100:+.1f}cm")
    print()

    aug_rh = [DexHandManipBiHEnv._aug_demo(data_rh, R, t, center=c) for R, t, c in transforms]
    aug_lh = [DexHandManipBiHEnv._aug_demo(data_lh, R, t, center=c) for R, t, c in transforms]

    colors = [plt.cm.tab10(i / max(args.n_aug - 1, 1)) for i in range(args.n_aug)]

    # ── Figure 1: Wrist trajectories with orientation arrows ──────────────
    fig1, (ax_lw, ax_rw) = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize=(14, 6))
    fig1.suptitle(f"Wrist trajectories — {args.data_idx}\n(arrows = wrist z-axis)", fontsize=13)

    plot_3d_traj(ax_lw, data_lh["wrist_pos"], data_lh["wrist_rot"], "original", "black",
                 is_original=True, arrow_len=args.arrow_len)
    for i, aug in enumerate(aug_lh):
        plot_3d_traj(ax_lw, aug["wrist_pos"], aug["wrist_rot"], f"aug {i+1}", colors[i],
                     arrow_len=args.arrow_len)
    ax_lw.scatter(*center.tolist(), color="black", s=80, marker="x", zorder=10, label="rot center")
    ax_lw.set_title("Left hand wrist")
    ax_lw.set_xlabel("x"); ax_lw.set_ylabel("y"); ax_lw.set_zlabel("z")
    ax_lw.legend(fontsize=7)

    plot_3d_traj(ax_rw, data_rh["wrist_pos"], data_rh["wrist_rot"], "original", "black",
                 is_original=True, arrow_len=args.arrow_len)
    for i, aug in enumerate(aug_rh):
        plot_3d_traj(ax_rw, aug["wrist_pos"], aug["wrist_rot"], f"aug {i+1}", colors[i],
                     arrow_len=args.arrow_len)
    ax_rw.scatter(*center.tolist(), color="black", s=80, marker="x", zorder=10, label="rot center")
    ax_rw.set_title("Right hand wrist")
    ax_rw.set_xlabel("x"); ax_rw.set_ylabel("y"); ax_rw.set_zlabel("z")
    ax_rw.legend(fontsize=7)

    fig1.tight_layout()
    out1 = args.out.replace(".png", "_wrist.png")
    fig1.savefig(out1, dpi=150)
    print(f"Saved {out1}")

    # ── Figure 2: Object trajectories ─────────────────────────────────────
    fig2, (ax_lo, ax_ro) = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize=(14, 6))
    fig2.suptitle(f"Object trajectories — {args.data_idx}", fontsize=13)

    plot_3d_traj(ax_lo, data_lh["obj_trajectory"][:, :3, 3], None, "original", "black", is_original=True)
    for i, aug in enumerate(aug_lh):
        plot_3d_traj(ax_lo, aug["obj_trajectory"][:, :3, 3], None, f"aug {i+1}", colors[i])
    ax_lo.set_title("Left hand object")
    ax_lo.set_xlabel("x"); ax_lo.set_ylabel("y"); ax_lo.set_zlabel("z")
    ax_lo.legend(fontsize=7)

    plot_3d_traj(ax_ro, data_rh["obj_trajectory"][:, :3, 3], None, "original", "black", is_original=True)
    for i, aug in enumerate(aug_rh):
        plot_3d_traj(ax_ro, aug["obj_trajectory"][:, :3, 3], None, f"aug {i+1}", colors[i])
    ax_ro.set_title("Right hand object")
    ax_ro.set_xlabel("x"); ax_ro.set_ylabel("y"); ax_ro.set_zlabel("z")
    ax_ro.legend(fontsize=7)

    fig2.tight_layout()
    out2 = args.out.replace(".png", "_obj.png")
    fig2.savefig(out2, dpi=150)
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()
