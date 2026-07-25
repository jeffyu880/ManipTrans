"""Plot per-fingertip contact forces on the manipulated object from a grip-log CSV.

The CSV is produced by DexHandManipBiHEnv._log_grip_state (MANIPTRANS_GRIP_CSV=<path>).

    python data_stats/plot_grip_forces.py grip_full_model.csv --side rh --frames 141

`*_f` columns are |net_cf| on each contact body. `*_in` columns are the same force
projected onto the direction from the fingertip toward the object COM; contact pushes the
FINGER away from the object, so a squeeze is negative there — this script negates it back
into a positive "squeeze" component.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
# categorical slots 1-5, validated for the light surface (adjacent CVD dE 9.1, normal 19.6)
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def style_axes(ax):
    """Recessive grid and axes: hairline grid, no top/right spines, muted ticks."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="both", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for edge in ("top", "right"):
        ax.spines[edge].set_visible(False)
    for edge in ("left", "bottom"):
        ax.spines[edge].set_color(BASELINE)
        ax.spines[edge].set_linewidth(1.0)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="grip log CSV")
    parser.add_argument("--side", default="rh", choices=["rh", "lh"])
    parser.add_argument("--frames", type=int, default=141, help="number of leading frames to plot")
    parser.add_argument("--obj-weight", type=float, default=None, help="object weight (N) reference line")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    data = np.genfromtxt(args.csv, delimiter=",", names=True)
    n = min(args.frames, len(data))
    frames = np.arange(n)
    side, hand = args.side, args.side.upper()

    per_finger = {f: data[f"{side}_{f}_f"][:n] for f in FINGERS}
    total = data[f"{side}_tip_f_sum"][:n]
    squeeze = -data[f"{side}_tip_in_sum"][:n]  # see module docstring: negate into "toward object"

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10, 7.2), sharex=True, gridspec_kw={"height_ratios": [1.6, 1]}
    )
    fig.patch.set_facecolor(SURFACE)

    for (finger, series), color in zip(per_finger.items(), SERIES_COLORS):
        ax_top.plot(frames, series, color=color, linewidth=2, label=finger, zorder=3)
        # direct label at the series peak — required relief for the sub-3:1 slots, and it
        # keeps identity off color alone
        if series.max() > 0:
            peak = int(np.argmax(series))
            # a peak in the last tenth would push its label off-axis: flip it to the left
            flip = peak > 0.9 * n
            ax_top.annotate(
                finger,
                xy=(peak, series[peak]),
                xytext=(-6 if flip else 4, 4),
                textcoords="offset points",
                ha="right" if flip else "left",
                color=INK_SECONDARY,
                fontsize=9,
                fontweight="medium",
                zorder=4,
            )

    silent = [f for f, s in per_finger.items() if s.max() == 0]
    if silent:
        ax_top.annotate(
            f"{' & '.join(silent)}: no contact — 0 N throughout",
            xy=(0.5, 0.72),
            xycoords="axes fraction",
            ha="center",
            color=INK_MUTED,
            fontsize=9,
        )

    ax_top.set_ylabel("fingertip contact force  |net_cf|  (N)", color=INK_SECONDARY, fontsize=10)
    ax_top.legend(
        frameon=False, ncol=5, loc="upper left", fontsize=9, labelcolor=INK_SECONDARY, handlelength=1.6
    )
    style_axes(ax_top)

    ax_bot.plot(frames, total, color=SERIES_COLORS[0], linewidth=2, label="total |force|", zorder=3)
    ax_bot.plot(frames, squeeze, color=SERIES_COLORS[1], linewidth=2, label="squeeze (toward cap)", zorder=3)
    if args.obj_weight is not None:
        ax_bot.axhline(args.obj_weight, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
        ax_bot.annotate(
            # the weight line sits on the baseline at this scale, so the label goes to open space
            f"dashed line: object weight {args.obj_weight * 1e3:.0f} mN",
            xy=(0.6, 0.78),
            xycoords="axes fraction",
            ha="center",
            color=INK_MUTED,
            fontsize=9,
        )
    ax_bot.set_ylabel("aggregate force (N)", color=INK_SECONDARY, fontsize=10)
    ax_bot.set_xlabel("frame", color=INK_SECONDARY, fontsize=10)
    ax_bot.set_xlim(0, n - 1)
    ax_bot.legend(frameon=False, ncol=2, loc="upper left", fontsize=9, labelcolor=INK_SECONDARY, handlelength=1.6)
    style_axes(ax_bot)

    fig.suptitle(
        f"{hand} fingertip contact forces on the cap — full model (residual on)",
        x=0.011,
        ha="left",
        color=INK,
        fontsize=13,
        fontweight="semibold",
    )
    fig.text(
        0.011,
        0.925,
        f"{os.path.basename(args.csv)} · frames 0-{n - 1} · Inspire hand, contact bodies only",
        ha="left",
        color=INK_MUTED,
        fontsize=9.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.915])

    out = args.out or os.path.splitext(args.csv)[0] + f"_{side}_forces.png"
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"wrote {out}")

    print(f"\n{hand} frames 0-{n - 1}:")
    for finger, series in per_finger.items():
        print(f"  {finger:7s} peak {series.max():6.3f} N   mean {series.mean():6.3f} N")
    print(f"  {'total':7s} peak {total.max():6.3f} N   mean {total.mean():6.3f} N")
    print(f"  {'squeeze':7s} peak {squeeze.max():6.3f} N   mean {squeeze.mean():6.3f} N")


if __name__ == "__main__":
    main()
