"""Compare causal vs non-causal demo velocities for one sequence (right hand).

Pure numpy/scipy — NO torch, isaacgym, CUDA or pytorch3d. It loads the raw capture pkl
(wrist + fingertip positions, wrist quaternion) and the retargeted pkl (opt_dof_pos)
directly, then computes velocities the two ways the dataset supports (see
main/dataset/base.py compute_velocity / compute_angular_velocity / compute_dof_velocity):

  * non-causal (default): np.gradient (central diff) + Gaussian smoothing (looks ahead).
  * causal: backward finite diff + causal EMA (real-time realizable; matches the live
      stream / LiveTargetSource).

The loader's recenter / table-rotation / dexhand-offset are all rigid transforms, and
velocity MAGNITUDE is invariant to them, so computing straight off the raw pkl gives the
same |v| / |omega| the policy sees. The three velocity functions are reimplemented here
to match base.py exactly.

Figure 1  |v|            : wrist + 5 fingertips                       (6 subplots)
Figure 2  angular/joint  : wrist |omega| + 12 retargeted finger DOFs  (13 subplots)

    python data_stats/plot_causal_velocity.py --data_idx m_164621
"""

import os
import glob
import pickle
import argparse

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as Rot
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_REPO)

# --- constants ----------------------------------------------------------------------
# MyDataset capture is 60Hz, so the true timestep is 1/60 s. NOTE the loader's formula
# time_delta = 1/(120/skip) assumes a 120Hz base and therefore yields 1/120 for mydataset
# at skip=1 (it under-scales dt by 2x); we use the physical value instead. SKIP subsamples
# every Nth frame, so the timestep between plotted frames is SKIP/60. Keep SKIP=1 to use
# all frames at the true 60Hz.
SKIP = 1
DT = SKIP / 60.0               # 1/60 s at skip=1 (60Hz capture)
EMA_ALPHA = 0.4               # causal_ema_alpha default

TIPS = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
# fallback AVP joint names (raw meta has no avp_to_mano_joints); see my_dataset_RH.py
TIP_TO_AVP = {
    "thumb_tip": "Thumb_TIP", "index_tip": "Index_TIP", "middle_tip": "Middle_TIP",
    "ring_tip": "Ring_TIP", "pinky_tip": "Pinky_TIP",
}
# inspire right-hand DOF order (maniptrans_envs/lib/envs/dexhands/inspire.py)
INSPIRE_DOF = [
    "index_proximal", "index_intermediate", "middle_proximal", "middle_intermediate",
    "pinky_proximal", "pinky_intermediate", "ring_proximal", "ring_intermediate",
    "thumb_proximal_yaw", "thumb_proximal_pitch", "thumb_intermediate", "thumb_distal",
]

STYLE = {
    "nc": dict(color="#1f77b4", ls="-", lw=1.6, label="non-causal (gradient + Gaussian)"),
    "cs": dict(color="#ff7f0e", ls="--", lw=1.6, label="causal (backward diff + EMA)"),
}


# --- velocity kernels, reimplemented to match main/dataset/base.py --------------------
def causal_ema(x, alpha):
    out = np.empty_like(x)
    acc = np.zeros_like(x[0])
    for t in range(x.shape[0]):
        acc = alpha * x[t] + (1.0 - alpha) * acc
        out[t] = acc
    return out


def linear_velocity(p, causal):
    """p: [T, ...] positions -> velocity, same shape. Matches compute_velocity/compute_dof_velocity."""
    p = np.asarray(p, dtype=np.float64)
    if causal:
        diff = np.zeros_like(p)
        diff[1:] = (p[1:] - p[:-1]) / DT
        return causal_ema(diff, EMA_ALPHA)
    v = np.gradient(p, axis=0) / DT
    return gaussian_filter1d(v, 2, axis=0, mode="nearest")


def angular_velocity(rotmat, causal):
    """rotmat: [T, 3, 3] -> angular velocity [T, 3]. Matches compute_angular_velocity."""
    rotmat = np.asarray(rotmat, dtype=np.float64)
    diff_r = rotmat[1:] @ np.swapaxes(rotmat[:-1], -1, -2)  # rotation t-1 -> t
    diff_aa = Rot.from_matrix(diff_r).as_rotvec()           # [T-1, 3] (axis*angle)
    if causal:
        av = np.zeros((rotmat.shape[0], 3), dtype=np.float64)
        av[1:] = diff_aa / DT
        return causal_ema(av, EMA_ALPHA)
    av = diff_aa / DT
    av = np.concatenate([av, av[-1:]], axis=0)              # pad last -> [T, 3]
    return gaussian_filter1d(av, 2, axis=0, mode="nearest")


def mag(v):
    return np.linalg.norm(v, axis=-1)


# --- loading -------------------------------------------------------------------------
def load(data_idx):
    key = data_idx[2:] if data_idx.startswith("m_") else data_idx
    raw_matches = [
        p for p in glob.glob("data/my_dataset/*.pkl")
        if os.path.splitext(os.path.basename(p))[0].endswith(key)
    ]
    assert len(raw_matches) == 1, f"'{key}' matched {raw_matches}"
    raw_path = raw_matches[0]
    stem = os.path.splitext(os.path.basename(raw_path))[0]

    raw = pickle.load(open(raw_path, "rb"))
    h = raw["hands"]["right"]
    sl = slice(None, None, SKIP)

    wrist_pos = np.asarray(h["wrist_pos"])[sl]                       # [T, 3]
    wrist_rotmat = Rot.from_quat(np.asarray(h["wrist_quat"])[sl]).as_matrix()  # [T, 3, 3]
    tips = {t: np.asarray(h["joints_pos"][TIP_TO_AVP[t]])[sl] for t in TIPS}   # [T, 3] each

    ret_path = f"data/retargeting/my_dataset/mano2inspire_rh/{stem}_rh.pkl"
    opt_dof = np.asarray(pickle.load(open(ret_path, "rb"))["opt_dof_pos"])[sl]  # [T, 12], match SKIP

    return dict(wrist_pos=wrist_pos, wrist_rotmat=wrist_rotmat, tips=tips,
                opt_dof=opt_dof, stem=stem, T=len(wrist_pos))


# --- plotting ------------------------------------------------------------------------
def plot_linear(d, data_idx, out_path):
    panels = [("wrist", d["wrist_pos"])] + [(t, d["tips"][t]) for t in TIPS]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    axes = axes.ravel()
    for ax, (name, pos) in zip(axes, panels):
        v_nc = mag(linear_velocity(pos, causal=False))
        v_cs = mag(linear_velocity(pos, causal=True))
        frames = np.arange(len(v_nc))
        ax.plot(frames, v_nc, **STYLE["nc"])
        ax.plot(frames, v_cs, **STYLE["cs"])
        ax.set_title(name, fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_ylabel("|v|  (m/s)")
    for ax in axes[3:]:
        ax.set_xlabel("frame")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95),
               ncol=2, frameon=False, fontsize=11)
    fig.suptitle(f"Right-hand linear speed |v| — causal vs non-causal  ({data_idx})", y=0.995, fontsize=13)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"[saved] {out_path}")


def plot_angular(d, data_idx, out_path):
    n_dof = d["opt_dof"].shape[1]
    dof_names = INSPIRE_DOF if n_dof == len(INSPIRE_DOF) else [f"dof_{i}" for i in range(n_dof)]

    # wrist angular speed |omega|
    wa_nc = mag(angular_velocity(d["wrist_rotmat"], causal=False))
    wa_cs = mag(angular_velocity(d["wrist_rotmat"], causal=True))
    # finger-joint (DOF) velocities, signed
    dof_nc = linear_velocity(d["opt_dof"], causal=False)   # [T, n_dof]
    dof_cs = linear_velocity(d["opt_dof"], causal=True)

    panels = [("wrist |omega|", wa_nc, wa_cs, "rad/s")]
    for j in range(n_dof):
        panels.append((dof_names[j] + "  (DOF)", dof_nc[:, j], dof_cs[:, j], "rad/s"))

    ncols = 4
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 2.6 * nrows), sharex=True)
    axes = axes.ravel()
    for ax, (name, y_nc, y_cs, unit) in zip(axes, panels):
        frames = np.arange(len(y_nc))
        ax.plot(frames, y_nc, **STYLE["nc"])
        ax.plot(frames, y_cs, **STYLE["cs"])
        ax.set_title(name, fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_ylabel(unit)
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    # x-label on the bottom-most visible axis of each column
    for c in range(ncols):
        col = list(range(c, len(panels), ncols))
        if col:
            axes[col[-1]].set_xlabel("frame")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.975),
               ncol=2, frameon=False, fontsize=11)
    fig.suptitle(
        f"Right-hand angular: wrist |omega| + finger-joint DOF velocities — causal vs non-causal  ({data_idx})",
        y=0.997, fontsize=13,
    )
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"[saved] {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_idx", default="m_164621")
    ap.add_argument("--out_dir", default="vis_traj_outputs/causal_vel")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    d = load(args.data_idx)
    print(f"[info] {args.data_idx}: T={d['T']} frames, dof={d['opt_dof'].shape[1]}, "
          f"skip={SKIP}, dt={DT:.5f}s", flush=True)

    plot_linear(d, args.data_idx, os.path.join(args.out_dir, f"{args.data_idx}_linear_speed.png"))
    plot_angular(d, args.data_idx, os.path.join(args.out_dir, f"{args.data_idx}_angular_joint_vel.png"))


if __name__ == "__main__":
    main()
