"""Tune the causal POSITION-smoothing approach (low-pass positions, then backward-diff).

The user preferred "pos-smooth" over EMA-on-velocity: smoothing positions first, then
differencing, tracks the offline (non-causal) reference with far fewer spikes because it
does not amplify noise through the derivative. This sweeps the smoothing strength alpha so
you can pick a value before wiring it into base.py / live_target_source.py.

Linear signals: causal EMA on positions (seeded at frame 0), then backward diff.
Angular signals: causal nlerp EMA on the wrist QUATERNION (hemisphere-aligned, renormalised,
seeded at frame 0), then backward diff of the smoothed rotations. You cannot EMA rotation
matrices componentwise, so quaternion smoothing is the correct analogue for the wrist.

    python data_stats/pos_smooth_tune.py --data_idx m_164621
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_causal_velocity as pcv

DT = pcv.DT
ALPHAS = [0.5, 0.3, 0.2]
ACOL = {0.5: "#2ca02c", 0.3: "#17becf", 0.2: "#d62728"}


def ema_pos(p, a):
    """Causal EMA on positions, seeded at the first sample (no spurious frame-0 jump)."""
    p = np.asarray(p, dtype=np.float64)
    out = np.empty_like(p)
    acc = p[0].copy()
    for t in range(len(p)):
        acc = a * p[t] + (1.0 - a) * acc
        out[t] = acc
    return out


def possmooth_lin_speed(p, a):
    ps = ema_pos(p, a)
    v = np.zeros_like(ps)
    v[1:] = (ps[1:] - ps[:-1]) / DT
    return np.linalg.norm(v, axis=-1)


def nlerp_quat(q, a):
    """Causal nlerp EMA on a quaternion stream [T,4] xyzw (hemisphere-aligned, renormalised)."""
    q = np.asarray(q, dtype=np.float64)
    out = np.empty_like(q)
    acc = q[0].copy()
    out[0] = acc
    for t in range(1, len(q)):
        qc = q[t]
        if np.dot(acc, qc) < 0.0:      # keep on the same hemisphere before averaging
            qc = -qc
        acc = a * qc + (1.0 - a) * acc
        acc = acc / np.linalg.norm(acc)
        out[t] = acc
    return out


def possmooth_ang_speed(rotmat, a):
    q = Rot.from_matrix(rotmat).as_quat()          # [T,4] xyzw
    Rs = Rot.from_quat(nlerp_quat(q, a)).as_matrix()
    dr = Rs[1:] @ np.swapaxes(Rs[:-1], -1, -2)
    aa = Rot.from_matrix(dr).as_rotvec()
    w = np.zeros((len(Rs), 3))
    w[1:] = aa / DT
    return np.linalg.norm(w, axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_idx", default="m_164621")
    ap.add_argument("--out_dir", default="vis_traj_outputs/causal_vel")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    d = pcv.load(args.data_idx)
    tip = d["tips"]["index_tip"]
    R = d["wrist_rotmat"]

    # references
    tip_ref = pcv.mag(pcv.linear_velocity(tip, causal=False))
    tip_cur = pcv.mag(pcv.linear_velocity(tip, causal=True))   # current: EMA-on-vel a=0.4
    wr_ref = pcv.mag(pcv.angular_velocity(R, causal=False))
    wr_cur = pcv.mag(pcv.angular_velocity(R, causal=True))

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(tip_ref, color="k", lw=2.2, alpha=0.55, label="non-causal (ref, not live)")
    axes[0].plot(tip_cur, color="#ff7f0e", lw=1.2, label="causal EMA-on-vel a=0.4 (current)")
    for a in ALPHAS:
        axes[0].plot(possmooth_lin_speed(tip, a), color=ACOL[a], lw=1.8, label=f"pos-smooth a={a}")
    axes[0].set_title("index_tip  |v|", fontsize=12)
    axes[0].set_ylabel("|v|  (m/s)")

    axes[1].plot(wr_ref, color="k", lw=2.2, alpha=0.55, label="non-causal (ref, not live)")
    axes[1].plot(wr_cur, color="#ff7f0e", lw=1.2, label="causal EMA-on-vel a=0.4 (current)")
    for a in ALPHAS:
        axes[1].plot(possmooth_ang_speed(R, a), color=ACOL[a], lw=1.8, label=f"pos-smooth a={a}")
    axes[1].set_title("wrist  |omega|  (quaternion-smoothed)", fontsize=12)
    axes[1].set_ylabel("|omega|  (rad/s)")
    axes[1].set_xlabel("frame")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, ncol=2)
    fig.suptitle(f"Causal position-smoothing (smooth positions -> diff) alpha sweep  ({args.data_idx})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(args.out_dir, f"{args.data_idx}_pos_smooth_tune.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
