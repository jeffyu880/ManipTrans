"""Show ways to smooth the CAUSAL velocity (must stay past-only, to match the live stream).

Overlays, on the two noisiest signals (a fingertip |v| and wrist |omega|):
  * non-causal (gradient + Gaussian)      -- reference only; NOT realizable live
  * causal EMA alpha=0.4 (current default)
  * causal EMA alpha=0.2 / 0.1            -- lower alpha = smoother, more lag
  * causal double-EMA alpha=0.4           -- cascade two poles: sharper noise cut, moderate lag
  * causal position-smooth then diff      -- low-pass positions first (diff amplifies noise)

All variants use ONLY past/current samples, so they are live-realizable. The knob that is
already wired through the codebase (base.py + live_target_source.py) is `causalEmaAlpha`.

    python data_stats/smooth_causal_demo.py --data_idx m_164621
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

import plot_causal_velocity as pcv  # reuse loader + non-causal reference

DT = pcv.DT


def ema(x, a, seed_first=False):
    # seed_first: start the accumulator at x[0] (correct for smoothing *positions*, which
    # have no natural zero); default False starts at 0 (correct for velocities, whose first
    # sample is 0 — matches base._causal_ema / LiveTargetSource).
    out = np.empty_like(x)
    acc = np.array(x[0], dtype=np.float64) if seed_first else np.zeros_like(x[0])
    for t in range(len(x)):
        acc = a * x[t] + (1.0 - a) * acc
        out[t] = acc
    return out


def backdiff(p):
    p = np.asarray(p, dtype=np.float64)
    d = np.zeros_like(p)
    d[1:] = (p[1:] - p[:-1]) / DT
    return d


def causal_lin(p, a, order=1):
    d = backdiff(p)
    for _ in range(order):
        d = ema(d, a)
    return d


def causal_lin_possmooth(p, a):
    ps = ema(np.asarray(p, dtype=np.float64), a, seed_first=True)   # low-pass positions FIRST
    return backdiff(ps)


def ang_raw(R):
    dr = R[1:] @ np.swapaxes(R[:-1], -1, -2)
    aa = Rot.from_matrix(dr).as_rotvec()
    av = np.zeros((len(R), 3))
    av[1:] = aa / DT
    return av


def causal_ang(R, a, order=1):
    av = ang_raw(R)
    for _ in range(order):
        av = ema(av, a)
    return av


def mag(v):
    return np.linalg.norm(v, axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_idx", default="m_164621")
    ap.add_argument("--out_dir", default="vis_traj_outputs/causal_vel")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    d = pcv.load(args.data_idx)
    tip = d["tips"]["index_tip"]
    R = d["wrist_rotmat"]

    variants_lin = [
        ("non-causal (ref, not live)", mag(pcv.linear_velocity(tip, causal=False)), dict(color="k", lw=2.2, alpha=0.6)),
        ("causal EMA a=0.4 (current)", mag(causal_lin(tip, 0.4)), dict(color="#ff7f0e", lw=1.3)),
        ("causal EMA a=0.2", mag(causal_lin(tip, 0.2)), dict(color="#2ca02c", lw=1.6)),
        ("causal EMA a=0.1", mag(causal_lin(tip, 0.1)), dict(color="#d62728", lw=1.8)),
        ("causal double-EMA a=0.4", mag(causal_lin(tip, 0.4, order=2)), dict(color="#9467bd", lw=1.8, ls="--")),
        ("causal pos-smooth a=0.3 + diff", mag(causal_lin_possmooth(tip, 0.3)), dict(color="#17becf", lw=1.8, ls=":")),
    ]
    variants_ang = [
        ("non-causal (ref, not live)", mag(pcv.angular_velocity(R, causal=False)), dict(color="k", lw=2.2, alpha=0.6)),
        ("causal EMA a=0.4 (current)", mag(causal_ang(R, 0.4)), dict(color="#ff7f0e", lw=1.3)),
        ("causal EMA a=0.2", mag(causal_ang(R, 0.2)), dict(color="#2ca02c", lw=1.6)),
        ("causal EMA a=0.1", mag(causal_ang(R, 0.1)), dict(color="#d62728", lw=1.8)),
        ("causal double-EMA a=0.4", mag(causal_ang(R, 0.4, order=2)), dict(color="#9467bd", lw=1.8, ls="--")),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    for name, y, kw in variants_lin:
        axes[0].plot(np.arange(len(y)), y, label=name, **kw)
    axes[0].set_title("index_tip  |v|", fontsize=12)
    axes[0].set_ylabel("|v|  (m/s)")
    for name, y, kw in variants_ang:
        axes[1].plot(np.arange(len(y)), y, label=name, **kw)
    axes[1].set_title("wrist  |omega|", fontsize=12)
    axes[1].set_ylabel("|omega|  (rad/s)")
    axes[1].set_xlabel("frame")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, ncol=2)
    fig.suptitle(f"Smoothing the causal velocity (all variants are live-realizable)  ({args.data_idx})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(args.out_dir, f"{args.data_idx}_causal_smoothing.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
