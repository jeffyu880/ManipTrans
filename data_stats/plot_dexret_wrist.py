"""Plot the dex-retargeting baseline's wrist tracking error over a playthrough.

The pinch-gap CSV covers fingertips and contact force but carries nothing about the wrist, so
this is the only view of what the PD+feedforward wrist controller is actually achieving — which
is what its gains are tuned against.

    MANIPTRANS_DEXRET_LOG=wrist.csv python main/rl/train.py ... dexRetBaseline=true
    python data_stats/plot_dexret_wrist.py wrist.csv

Input columns (written by DexRetargetController.wrist_pd_ff, env 0 only):
    step, side, err_x, err_y, err_z, err_norm, rot_err_deg, speed, target_speed, force_n, torque_nm

`step` is the env's progress_buf, so it restarts each episode. Episodes are split on that reset
and overlaid, which makes run-to-run spread visible instead of averaging it away.

Reading the result: the position error does NOT converge to zero by design. The controller aims at
a wrist pulled back along the palm axis (DEXRET_WRIST_PULLBACK, 35% of the wrist-to-middle-MCP
span, ~31 mm) to stop the hand crowding the object, so a steady offset of that size is the
controller working, not failing. What matters is whether the error is FLAT after the initial
transient — drift means the wrist is losing the target, and oscillation means the gains are past
what the 60 Hz loop can hold.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Same palette as plot_pinch_gap.py so the baseline's figures sit alongside the existing ones.
SOURCES = {"rh": ("RH", "#2a78d6"), "lh": ("LH", "#eb6834")}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"

def style_axes(ax):
    """Apply the repo's plot styling to one axis.

    Args:
        ax: A matplotlib axis.

    Returns:
        None.
    """
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRIDLINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(True, color=GRIDLINE, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def split_episodes(steps):
    """Index ranges for each episode, split where the step counter restarts.

    Args:
        steps: (N,) array of the env's progress_buf, one row per control step.

    Returns:
        list of (start, stop) index pairs, one per episode.
    """
    breaks = np.flatnonzero(np.diff(steps) < 0) + 1
    edges = [0, *breaks.tolist(), len(steps)]
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def main():
    """Plot wrist position and rotation error per hand from a dexret wrist log.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("csv", help="wrist log from MANIPTRANS_DEXRET_LOG")
    parser.add_argument("--out", default=None, help="output PNG (default: alongside the CSV)")
    args = parser.parse_args()

    data = np.genfromtxt(args.csv, delimiter=",", names=True, dtype=None, encoding="utf-8")
    # Trajectory panels only if the log carries absolute positions (older logs have error only).
    has_traj = "ref_x" in (data.dtype.names or ())
    n_rows = 5 if has_traj else 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(12.4, 3.4 * n_rows), sharex=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)

    for col, (side, (label, color)) in enumerate(SOURCES.items()):
        rows = data[data["side"] == side]
        if not len(rows):
            for row in range(2):
                axes[row][col].set_visible(False)
            continue
        episodes = split_episodes(rows["step"])

        for ax, key, scale in ((axes[0][col], "err_norm", 1e3), (axes[1][col], "rot_err_deg", 1.0)):
            for i, (start, stop) in enumerate(episodes):
                # One line per episode, overlaid. Faint so a dozen episodes stay readable and the
                # spread between them is the visible quantity.
                ax.plot(
                    rows["step"][start:stop], rows[key][start:stop] * scale,
                    color=color, linewidth=1.3, alpha=0.85 if len(episodes) == 1 else 0.55,
                    label=f"episode {i + 1}" if col == 0 else None, zorder=3,
                )
            style_axes(ax)

        # Reference vs robot per axis, overlaid: a flat error curve cannot distinguish "both
        # tracking" from "both stationary", and these panels can.
        if has_traj:
            for axis_i, axis in enumerate("xyz"):
                ax = axes[2 + axis_i][col]
                start, stop = episodes[0]
                ax.plot(rows["step"][start:stop], rows[f"ref_{axis}"][start:stop],
                        color=INK_MUTED, linewidth=2.2, alpha=0.9, zorder=2,
                        label="Reference" if (col == 0 and axis_i == 0) else None)
                ax.plot(rows["step"][start:stop], rows[f"act_{axis}"][start:stop],
                        color=color, linewidth=1.3, zorder=3,
                        label="Robot" if (col == 0 and axis_i == 0) else None)
                style_axes(ax)
                if col == 0:
                    ax.set_ylabel(f"Wrist {axis.upper()} [m]", color=INK_SECONDARY, fontsize=10)

        axes[0][col].set_title(label, color=INK_SECONDARY, fontsize=10, fontweight="medium")

        stats = (f"{label}  pos {rows['err_norm'].mean() * 1e3:.1f} mm mean, "
                 f"{rows['err_norm'].max() * 1e3:.1f} max   |   "
                 f"rot {rows['rot_err_deg'].mean():.1f} deg mean, {rows['rot_err_deg'].max():.1f} max")
        print(stats)

    axes[0][0].set_ylabel("Wrist Position Error [mm]", color=INK_SECONDARY, fontsize=10)
    axes[1][0].set_ylabel("Wrist Rotation Error [deg]", color=INK_SECONDARY, fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel("Control Step", color=INK_SECONDARY, fontsize=10)

    # Header offsets are in inches, converted to figure fraction: with a variable row count the
    # figure height changes a lot, and a fixed fraction would collide the subtitle into the title.
    height = fig.get_figheight()
    fig.suptitle(
        "Wrist Tracking Error - dex-retargeting Baseline",
        x=0.011, y=1 - 0.28 / height,
        ha="left", color=INK, fontsize=13, fontweight="semibold",
    )
    fig.text(
        0.011, 1 - 0.62 / height,
        f"{os.path.basename(args.csv)} - target minus actual; the palm-axis pullback is folded "
        f"into both the target and the reset, so this should start and stay near zero",
        ha="left", color=INK_MUTED, fontsize=9.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 1 - 0.95 / height], h_pad=2.5)
    # Gather across every row: the episode lines are labelled on row 0 but Reference/Robot only
    # appear on the trajectory rows, so a single-axis grab would silently drop them.
    handles, labels = {}, []
    for ax in axes.ravel():
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in handles:
                handles[label] = handle
                labels.append(label)
    handles = [handles[l] for l in labels]
    if handles:
        axes[0][-1].legend(
            handles, labels, loc="lower right", bbox_to_anchor=(1.0, 1.10),
            ncol=len(handles), frameon=False, fontsize=9, labelcolor=INK_SECONDARY, handlelength=1.6,
        )

    out = args.out or os.path.splitext(args.csv)[0] + ".png"
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
