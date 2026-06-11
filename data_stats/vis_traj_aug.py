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
from copy import copy

sys.path.insert(0, ".")

# Stub out all modules that import isaacgym so we don't need it for trajectory loading
from unittest.mock import MagicMock
for _mod in (
    "isaacgym",
    "isaacgym.gymapi", "isaacgym.gymtorch", "isaacgym.gymutil",
    "main.dataset.mano2dexhand", "main.dataset.mano2dexhand_pk",
    "maniptrans_envs.lib.envs.core.vec_task",
    "maniptrans_envs.lib.envs.core.sim_config",
    "maniptrans_envs.lib.envs.tasks.dexhandimitator",
    "maniptrans_envs.lib.envs.tasks.dexhandmanip_bih",
    "maniptrans_envs.lib.envs.tasks.dexhandmanip_sh",
):
    sys.modules[_mod] = MagicMock()

import torch

from main.dataset.oakink2_dataset_dexhand_rh import OakInk2DatasetDexHandRH
from main.dataset.oakink2_dataset_dexhand_lh import OakInk2DatasetDexHandLH
from main.dataset.transform import aa_to_rotmat, rotmat_to_aa
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory


# ── Augmentation helpers (inlined from dexhandmanip_bih to avoid isaacgym import) ──

def sample_aug_transform(device="cpu"):
    table_width_offset = 0.2
    table_surface_z = 0.4 + 0.015
    center = torch.tensor([-table_width_offset / 2, 0.0, table_surface_z], dtype=torch.float32)
    angle = (torch.rand(1).item() * 2 - 1) * (30.0 * np.pi / 180.0)
    cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
    R = torch.tensor([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    t = (torch.rand(2) * 2 - 1) * 0.05
    t = torch.cat([t, torch.zeros(1)])
    return R, t, center


def aug_demo(data, R, t, center, noise_std=0.0):
    d = copy(data)
    def rp(x):
        return (R @ (x - center).T).T + center + t
    def rv(x):
        return (R @ x.T).T
    def raa(x):
        return rotmat_to_aa(R.unsqueeze(0) @ aa_to_rotmat(x))
    d["wrist_pos"] = rp(data["wrist_pos"])
    d["opt_wrist_pos"] = rp(data["opt_wrist_pos"])
    d["mano_joints"] = {k: rp(v) + (torch.randn_like(v) * noise_std if noise_std > 0 else 0)
                        for k, v in data["mano_joints"].items()}
    obj = data["obj_trajectory"].clone()
    obj[:, :3, 3] = rp(obj[:, :3, 3])
    obj[:, :3, :3] = R.unsqueeze(0) @ obj[:, :3, :3]
    d["obj_trajectory"] = obj
    d["wrist_rot"] = raa(data["wrist_rot"])
    d["opt_wrist_rot"] = raa(data["opt_wrist_rot"])
    d["wrist_velocity"] = rv(data["wrist_velocity"])
    d["wrist_angular_velocity"] = rv(data["wrist_angular_velocity"])
    d["obj_velocity"] = rv(data["obj_velocity"])
    d["obj_angular_velocity"] = rv(data["obj_angular_velocity"])
    d["opt_wrist_velocity"] = rv(data["opt_wrist_velocity"])
    d["opt_wrist_angular_velocity"] = rv(data["opt_wrist_angular_velocity"])
    d["mano_joints_velocity"] = {k: rv(v) for k, v in data["mano_joints_velocity"].items()}
    return d


def aug_demo_lh_obj_center(data_rh, data_lh, R):
    d_rh = copy(data_rh)
    c_t   = data_lh["obj_trajectory"][:, :3, 3]
    c_dot = data_lh["obj_velocity"]
    def rp(x):
        return (R @ (x - c_t).T).T + c_t
    def rv(x):
        return (R @ (x - c_dot).T).T + c_dot
    def raa(x):
        return rotmat_to_aa(R.unsqueeze(0) @ aa_to_rotmat(x))
    d_rh["wrist_pos"] = rp(data_rh["wrist_pos"])
    d_rh["opt_wrist_pos"] = rp(data_rh["opt_wrist_pos"])
    d_rh["mano_joints"] = {k: rp(v) for k, v in data_rh["mano_joints"].items()}
    obj_rh = data_rh["obj_trajectory"].clone()
    obj_rh[:, :3, 3] = rp(obj_rh[:, :3, 3])
    obj_rh[:, :3, :3] = R.unsqueeze(0) @ obj_rh[:, :3, :3]
    d_rh["obj_trajectory"] = obj_rh
    d_rh["wrist_rot"] = raa(data_rh["wrist_rot"])
    d_rh["opt_wrist_rot"] = raa(data_rh["opt_wrist_rot"])
    d_rh["wrist_velocity"] = rv(data_rh["wrist_velocity"])
    d_rh["wrist_angular_velocity"] = rv(data_rh["wrist_angular_velocity"])
    d_rh["obj_velocity"] = rv(data_rh["obj_velocity"])
    d_rh["obj_angular_velocity"] = rv(data_rh["obj_angular_velocity"])
    d_rh["opt_wrist_velocity"] = rv(data_rh["opt_wrist_velocity"])
    d_rh["opt_wrist_angular_velocity"] = rv(data_rh["opt_wrist_angular_velocity"])
    d_rh["mano_joints_velocity"] = {k: rv(v) for k, v in data_rh["mano_joints_velocity"].items()}
    return d_rh, data_lh

def aug_demo_rh_obj_center(data_rh, R):
    """Rotate only the RH demo around the RH object center at each timestep.

    At each timestep t:
        p_rh_aug_t = R @ (p_rh_t - c_t) + c_t
    where c_t = RH object position at frame t.

    """
    from copy import copy
    d_rh = copy(data_rh)

    c_t   = data_rh["obj_trajectory"][:, :3, 3]  # [T, 3] - RH object center
    c_dot = data_rh["obj_velocity"]               # [T, 3] - RH object velocity

    def rp(x):
        return (R @ (x - c_t).T).T + c_t

    def rv(x):
        return (R @ (x - c_dot).T).T + c_dot

    def raa(x):
        return rotmat_to_aa(R.unsqueeze(0) @ aa_to_rotmat(x))

    d_rh["wrist_pos"] = rp(data_rh["wrist_pos"])
    d_rh["opt_wrist_pos"] = rp(data_rh["opt_wrist_pos"])
    d_rh["mano_joints"] = {k: rp(v) for k, v in data_rh["mano_joints"].items()}
    obj_rh = data_rh["obj_trajectory"].clone()
    obj_rh[:, :3, 3] = rp(obj_rh[:, :3, 3])
    obj_rh[:, :3, :3] = R.unsqueeze(0) @ obj_rh[:, :3, :3]
    d_rh["obj_trajectory"] = obj_rh
    d_rh["wrist_rot"] = raa(data_rh["wrist_rot"])
    d_rh["opt_wrist_rot"] = raa(data_rh["opt_wrist_rot"])
    d_rh["wrist_velocity"] = rv(data_rh["wrist_velocity"])
    d_rh["wrist_angular_velocity"] = rv(data_rh["wrist_angular_velocity"])
    d_rh["obj_velocity"] = rv(data_rh["obj_velocity"])
    d_rh["obj_angular_velocity"] = rv(data_rh["obj_angular_velocity"])
    d_rh["opt_wrist_velocity"] = rv(data_rh["opt_wrist_velocity"])
    d_rh["opt_wrist_angular_velocity"] = rv(data_rh["opt_wrist_angular_velocity"])
    d_rh["mano_joints_velocity"] = {k: rv(v) for k, v in data_rh["mano_joints_velocity"].items()}

    return d_rh

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
    c = "blue" if is_original else color
    ax.plot(x, y, z, linestyle=ls, color=c, linewidth=lw, label=label)
    ax.scatter(x[0], y[0], z[0], color=c, s=60, marker="s")  # square = wrist

    # draw wrist z-axis (palm normal) as a single arrow at the first frame
    if rot_aa is not None:
        R0 = aa_to_rotmat(rot_aa[0:1]).numpy()[0]  # [3, 3]
        z_axis = R0[:, 2]                           # local z-axis in world frame
        ax.quiver(
            pos[0, 0].numpy(), pos[0, 1].numpy(), pos[0, 2].numpy(),
            z_axis[0], z_axis[1], z_axis[2],
            length=arrow_len, color=c, alpha=0.9, normalize=True,
        )

def visualize_rh_obj_center_animation(data_rh, data_rh_aug, data_idx, out_path, step=5):
    """Side-by-side GIF: left=original RH, right=augmented RH.

    Each panel shows wrist + fingertip positions at each timestep,
    with the cap (RH object) as the pivot marker.
    step: render every Nth frame.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter
    from tqdm import tqdm

    T = data_rh["wrist_pos"].shape[0]
    frames = list(range(0, T, step))

    orig_wrist  = data_rh["wrist_pos"].numpy()
    aug_wrist   = data_rh_aug["wrist_pos"].numpy()
    cap_pos     = data_rh["obj_trajectory"][:, :3, 3].numpy()

    joint_keys  = list(data_rh["mano_joints"].keys())
    orig_joints = np.stack([data_rh["mano_joints"][k].numpy()     for k in joint_keys], axis=1)  # [T, J, 3]
    aug_joints  = np.stack([data_rh_aug["mano_joints"][k].numpy() for k in joint_keys], axis=1)

    all_pts = np.concatenate([orig_wrist, aug_wrist,
                               orig_joints.reshape(-1, 3), aug_joints.reshape(-1, 3),
                               cap_pos], axis=0)
    pad = 0.04
    xlim = (all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
    ylim = (all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)
    zlim = (all_pts[:, 2].min() - pad, all_pts[:, 2].max() + pad)

    fig = plt.figure(figsize=(14, 6))
    ax_orig = fig.add_subplot(121, projection="3d")
    ax_aug  = fig.add_subplot(122, projection="3d")
    fig.suptitle(f"RH grip rotated about cap — {data_idx}  (30°)", fontsize=11)
    fig.legend(handles=[
        plt.Line2D([0], [0], color="steelblue", lw=2, label="original"),
        plt.Line2D([0], [0], color="tomato",    lw=2, label="augmented"),
        plt.Line2D([0], [0], marker="x", color="green", lw=0, markersize=8, label="cap (pivot)"),
    ], fontsize=8, loc="lower center", ncol=3)

    # Wrist-only ghost trail drawn once as static background (fast)
    for ax, wrist, color in [(ax_orig, orig_wrist, "steelblue"), (ax_aug, aug_wrist, "tomato")]:
        ax.plot(*wrist.T,   color=color, alpha=0.10, lw=0.8)
        ax.plot(*cap_pos.T, color="green", alpha=0.10, lw=0.8)

    pbar = tqdm(total=len(frames), desc="Rendering frames")

    def draw_hand(ax, wrist_t, joints_t, cap_t, color, title):
        ax.cla()
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(title, fontsize=9)
        # Wrist ghost trail
        ax.plot(*orig_wrist.T if color == "steelblue" else aug_wrist.T,
                color=color, alpha=0.10, lw=0.8)
        ax.plot(*cap_pos.T, color="green", alpha=0.10, lw=0.8)
        # Current hand pose
        ax.scatter(*wrist_t, color=color, s=80, marker="s", zorder=5)
        for j in range(len(joint_keys)):
            ax.scatter(*joints_t[j], color=color, s=25, zorder=5)
            ax.plot([wrist_t[0], joints_t[j, 0]],
                    [wrist_t[1], joints_t[j, 1]],
                    [wrist_t[2], joints_t[j, 2]],
                    color=color, alpha=0.7, lw=1.2)
        ax.scatter(*cap_t, color="green", s=120, marker="x", linewidths=2.5, zorder=10)

    def draw(t):
        draw_hand(ax_orig, orig_wrist[t], orig_joints[t], cap_pos[t],
                  "steelblue", f"Original  (frame {t}/{T-1})")
        draw_hand(ax_aug,  aug_wrist[t],  aug_joints[t],  cap_pos[t],
                  "tomato",    f"Augmented (frame {t}/{T-1})")
        pbar.update(1)

    anim = FuncAnimation(fig, draw, frames=frames, interval=50)

    if out_path.endswith(".gif"):
        anim.save(out_path, writer=PillowWriter(fps=20))
    else:
        anim.save(out_path, writer=FFMpegWriter(fps=20))
    plt.close(fig)
    print(f"Saved animation to {out_path}")

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

    transforms = []
    for i in range(args.n_aug):
        torch.manual_seed(i)
        R, t, c = sample_aug_transform("cpu")
        transforms.append((R, t, c))

    print(f"\nAugmentations applied ({args.n_aug} total):")
    for i, (R, t, c) in enumerate(transforms):
        angle_deg = float(np.degrees(np.arctan2(R[1, 0].item(), R[0, 0].item())))
        print(f"  aug {i+1}: rot={angle_deg:+.2f}°  Δx={t[0].item()*100:+.1f}cm  Δy={t[1].item()*100:+.1f}cm")
    print()

    aug_rh = [aug_demo(data_rh, R, t, c) for R, t, c in transforms]
    aug_lh = [aug_demo(data_lh, R, t, c) for R, t, c in transforms]

    colors = [plt.cm.Reds(0.4 + 0.5 * i / max(args.n_aug - 1, 1)) for i in range(args.n_aug)]

    # ── Figure 1: Wrist trajectories with orientation arrows ──────────────
    fig1, (ax_lw, ax_rw) = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize=(14, 6))
    fig1.suptitle(f"Wrist trajectories — {args.data_idx}\n(arrows = wrist z-axis)", fontsize=13)

    plot_3d_traj(ax_lw, data_lh["wrist_pos"], data_lh["wrist_rot"], "original", "black",
                 is_original=True, arrow_len=args.arrow_len)
    for i, aug in enumerate(aug_lh):
        plot_3d_traj(ax_lw, aug["wrist_pos"], aug["wrist_rot"], f"aug {i+1}", colors[i],
                     arrow_len=args.arrow_len)
    ax_lw.scatter(*transforms[0][2].tolist(), color="black", s=80, marker="x", zorder=10, label="rot center")
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


def visualize_lh_obj_center_animation(data_rh, data_lh, data_rh_aug, data_idx, out_path, step=5):
    """Save animation of original vs LH-obj-center-augmented RH trajectory to file.

    step: render every Nth frame to keep the file small.
    """
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

    T = data_rh["wrist_pos"].shape[0]
    frames = list(range(0, T, step))

    orig_wrist = data_rh["wrist_pos"].numpy()
    orig_obj   = data_rh["obj_trajectory"][:, :3, 3].numpy()
    orig_wv    = data_rh["wrist_velocity"].numpy()
    orig_ov    = data_rh["obj_velocity"].numpy()

    aug_wrist  = data_rh_aug["wrist_pos"].numpy()
    aug_obj    = data_rh_aug["obj_trajectory"][:, :3, 3].numpy()
    aug_wv     = data_rh_aug["wrist_velocity"].numpy()
    aug_ov     = data_rh_aug["obj_velocity"].numpy()

    lh_obj     = data_lh["obj_trajectory"][:, :3, 3].numpy()

    vel_scale = 0.05

    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection="3d")
    fig.suptitle(f"LH-obj-center aug (30°) — {data_idx}\nblue=orig, orange=aug, green×=LH obj pivot", fontsize=10)

    def draw(t):
        ax.cla()
        ax.plot(*orig_wrist.T, color="blue",      alpha=0.10, lw=0.7)
        ax.plot(*orig_obj.T,   color="steelblue", alpha=0.10, lw=0.7)
        ax.plot(*aug_wrist.T,  color="red",       alpha=0.10, lw=0.7)
        ax.plot(*aug_obj.T,    color="darkred",   alpha=0.10, lw=0.7)
        ax.plot(*lh_obj.T,     color="green",  alpha=0.10, lw=0.7)

        ax.scatter(*orig_wrist[t], color="blue",      s=60, marker="s", label="orig wrist")
        ax.scatter(*orig_obj[t],   color="steelblue", s=50, label="orig obj")
        ax.scatter(*aug_wrist[t],  color="red",       s=60, marker="s", label="aug wrist")
        ax.scatter(*aug_obj[t],    color="darkred",   s=50, label="aug obj")
        ax.scatter(*lh_obj[t],     color="green",  s=80, marker="x", label="LH obj (pivot)")

        def arrow(pos, vel, color):
            n = np.linalg.norm(vel)
            if n > 1e-6:
                ax.quiver(*pos, *(vel / n * vel_scale), color=color, alpha=0.85, linewidth=1.5)

        arrow(orig_wrist[t], orig_wv[t], "blue")
        arrow(orig_obj[t],   orig_ov[t], "navy")
        arrow(aug_wrist[t],  aug_wv[t],  "orange")
        arrow(aug_obj[t],    aug_ov[t],  "red")

        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(f"frame {t}/{T-1}", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")

    anim = FuncAnimation(fig, draw, frames=frames, interval=50)

    if out_path.endswith(".gif"):
        anim.save(out_path, writer=PillowWriter(fps=20))
    else:
        anim.save(out_path, writer=FFMpegWriter(fps=20))
    plt.close(fig)
    print(f"Saved animation to {out_path}")


FINGERTIP_KEYS = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
FINGERTIP_COLORS = ["tomato", "steelblue", "forestgreen", "darkorange", "purple"]


def visualize_first_last_frames(data_rh, data_rh_aug, data_idx, out_path):
    """2×2 plot: rows=first/last frame, cols=original/augmented. Shows fingertip positions."""
    frame_labels = ["First frame (t=0)", f"Last frame (t={data_rh['wrist_pos'].shape[0]-1})"]
    frame_indices = [0, -1]
    col_labels = ["Original", "Augmented (RH obj-center)"]
    datasets = [data_rh, data_rh_aug]

    # compute shared axis limits across all 4 panels
    all_pts = []
    for data in datasets:
        for fi in frame_indices:
            all_pts.append(data["wrist_pos"][fi].numpy())
            all_pts.append(data["obj_trajectory"][fi, :3, 3].numpy())
            for k in FINGERTIP_KEYS:
                all_pts.append(data["mano_joints"][k][fi].numpy())
    all_pts = np.array(all_pts)
    pad = 0.03
    xlim = (all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
    ylim = (all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)
    zlim = (all_pts[:, 2].min() - pad, all_pts[:, 2].max() + pad)

    fig, axes = plt.subplots(2, 2, subplot_kw={"projection": "3d"}, figsize=(14, 12))
    fig.suptitle(f"Fingertip positions — {data_idx}  (RH obj-center augmentation, 30°)", fontsize=13)

    for row, (fi, frame_label) in enumerate(zip(frame_indices, frame_labels)):
        for col, (data, col_label) in enumerate(zip(datasets, col_labels)):
            ax = axes[row, col]
            ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
            ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
            ax.set_title(f"{frame_label} — {col_label}", fontsize=10)

            wrist = data["wrist_pos"][fi].numpy()
            obj_pos = data["obj_trajectory"][fi, :3, 3].numpy()

            ax.scatter(*wrist, color="black", s=120, marker="s", zorder=10, label="wrist")
            ax.scatter(*obj_pos, color="dimgray", s=150, marker="x", linewidths=2.5,
                       zorder=10, label="object")

            for tip_key, color in zip(FINGERTIP_KEYS, FINGERTIP_COLORS):
                tip = data["mano_joints"][tip_key][fi].numpy()
                ax.scatter(*tip, color=color, s=100, zorder=5,
                           label=tip_key.replace("_tip", ""))
                ax.plot([wrist[0], tip[0]], [wrist[1], tip[1]], [wrist[2], tip[2]],
                        color=color, alpha=0.45, lw=1.2)

            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    import sys as _sys
    if "--first-last" in _sys.argv:
        _sys.argv.remove("--first-last")
        parser = argparse.ArgumentParser()
        parser.add_argument("--data_idx", default="3b1e6@12")
        parser.add_argument("--out", default="fingertips_first_last.png")
        args = parser.parse_args()

        print(f"Loading {args.data_idx}...")
        data_rh = load_demo(args.data_idx, "right")
        print(f"RH frames: {len(data_rh['wrist_pos'])}")

        angle = np.radians(30)
        cs, sn = float(np.cos(angle)), float(np.sin(angle))
        R30 = torch.tensor([[cs, -sn, 0.], [sn, cs, 0.], [0., 0., 1.]], dtype=torch.float32)
        data_rh_aug = aug_demo_rh_obj_center(data_rh, R30)

        visualize_first_last_frames(data_rh, data_rh_aug, args.data_idx, args.out)

    elif "--lh-obj-center" in _sys.argv:
        _sys.argv.remove("--lh-obj-center")
        parser = argparse.ArgumentParser()
        parser.add_argument("--data_idx", default="3b1e6@12")
        parser.add_argument("--out", default="lh_obj_center_aug.gif")
        parser.add_argument("--step", type=int, default=5, help="Render every Nth frame")
        args = parser.parse_args()

        print(f"Loading {args.data_idx}...")
        data_rh = load_demo(args.data_idx, "right")
        data_lh = load_demo(args.data_idx, "left")
        print(f"Frames: {len(data_rh['wrist_pos'])}")

        angle = np.radians(30)
        cs, sn = float(np.cos(angle)), float(np.sin(angle))
        R30 = torch.tensor([[cs, -sn, 0.], [sn, cs, 0.], [0., 0., 1.]], dtype=torch.float32)
        data_rh_aug, _ = aug_demo_lh_obj_center(data_rh, data_lh, R30)

        visualize_lh_obj_center_animation(data_rh, data_lh, data_rh_aug, args.data_idx, args.out, step=args.step)
    
    elif "--rh-obj-center" in _sys.argv:
        _sys.argv.remove("--rh-obj-center")
        parser = argparse.ArgumentParser()
        parser.add_argument("--data_idx", default="3b1e6@12")
        parser.add_argument("--out", default="rh_obj_center_aug.gif")
        parser.add_argument("--step", type=int, default=5, help="Render every Nth frame")
        args = parser.parse_args()

        print(f"Loading {args.data_idx}...")
        data_rh = load_demo(args.data_idx, "right")
        print(f"Frames: {len(data_rh['wrist_pos'])}")

        angle = np.radians(30)
        cs, sn = float(np.cos(angle)), float(np.sin(angle))
        R30 = torch.tensor([[cs, -sn, 0.], [sn, cs, 0.], [0., 0., 1.]], dtype=torch.float32)
        data_rh_aug = aug_demo_rh_obj_center(data_rh, R30)

        visualize_rh_obj_center_animation(data_rh, data_rh_aug, args.data_idx, args.out, step=args.step)
    
    else:
        main()
