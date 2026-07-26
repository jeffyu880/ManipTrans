"""Plot per-fingertip contact forces on the manipulated object from a grip-log CSV.

The CSV is produced by DexHandManipBiHEnv._log_grip_state (MANIPTRANS_GRIP_CSV=<path>).

One CSV — per-finger detail for that rollout:

    python data_stats/plot_grip_forces.py grip_full_model.csv --side rh --frames 141

Several CSVs — one comparison figure across demos, squeeze over normalized time plus a
per-demo peak/mean strip grouped by reach distance:

    python data_stats/plot_grip_forces.py grip_logs/F0_individual/*.csv --side rh

`*_f` columns are |net_cf| on each contact body. `*_in` columns are the same force
projected onto the direction from the fingertip toward the object COM; contact pushes the
FINGER away from the object, so a squeeze is negative there — this script negates it back
into a positive "squeeze" component.
"""

import argparse
import csv as csvmod
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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


def demo_id(path):
    """m_140843 out of grip_m_140843.csv or grip_full_model__demo_m_140843.csv."""
    stem = os.path.splitext(os.path.basename(path))[0]
    match = re.search(r"m_\d+", stem)
    return match.group(0) if match else stem


def load_distances(folds_csv):
    """demo -> reach distance, so the comparison can group and colour by it."""
    path = folds_csv if os.path.isabs(folds_csv) else os.path.join(REPO, folds_csv)
    if not os.path.exists(path):
        print(f"WARNING: {path} not found — demos will be drawn ungrouped")
        return {}
    with open(path) as fh:
        return {
            r["demo_name"]: int(r["demo_distance"])
            for r in csvmod.DictReader(fh)
            if r.get("demo_name") and r.get("demo_distance")
        }


def plot_comparison(args, csv_paths):
    """Squeeze force across several rollouts: one line per demo, plus a peak/mean strip."""
    side, hand = args.side, args.side.upper()
    dist_of = load_distances(args.folds_csv)

    demos = []
    for path in sorted(csv_paths):
        data = np.genfromtxt(path, delimiter=",", names=True)
        squeeze = -data[f"{side}_tip_in_sum"]  # see module docstring
        demo = demo_id(path)
        demos.append(
            {
                "demo": demo,
                "dist": dist_of.get(demo),
                # normalized time, so demos of different length are comparable
                "t": np.linspace(0.0, 1.0, len(squeeze)),
                "squeeze": squeeze,
                "peak": float(squeeze.max()),
                "mean": float(squeeze.mean()),
            }
        )

    def color_of(d):
        return SERIES_COLORS[(d["dist"] - 1) % len(SERIES_COLORS)] if d["dist"] else INK_MUTED

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10, 8.6), gridspec_kw={"height_ratios": [1.45, 1]}
    )
    fig.patch.set_facecolor(SURFACE)

    for d in demos:
        ax_top.plot(d["t"], d["squeeze"], color=color_of(d), linewidth=1.6, alpha=0.9, zorder=3)
    ax_top.set_ylabel("Squeeze Force Toward Object (N)", color=INK_SECONDARY, fontsize=10)
    ax_top.set_xlabel("Normalized Time", color=INK_SECONDARY, fontsize=10)
    ax_top.set_xlim(0, 1)
    handles = [
        plt.Line2D([], [], color=SERIES_COLORS[(k - 1) % len(SERIES_COLORS)], lw=2, label=f"distance {k}")
        for k in sorted({d["dist"] for d in demos if d["dist"]})
    ]
    if handles:
        ax_top.legend(
            handles=handles, frameon=False, ncol=len(handles), loc="upper left",
            fontsize=9, labelcolor=INK_SECONDARY, handlelength=1.6,
        )
    style_axes(ax_top)

    # Strip plot: demos of the same distance are fanned out around it by a fixed offset, so the
    # spread is deterministic rather than jittered and the same demo lands in the same spot.
    by_dist = {}
    for d in demos:
        by_dist.setdefault(d["dist"], []).append(d)
    for dist, group in by_dist.items():
        for i, d in enumerate(sorted(group, key=lambda g: g["demo"])):
            x = (dist if dist else 0) + (i - (len(group) - 1) / 2) * 0.16
            c = color_of(d)
            ax_bot.plot([x, x], [d["mean"], d["peak"]], color=c, lw=1.2, alpha=0.55, zorder=2)
            ax_bot.plot(x, d["peak"], "o", color=c, ms=7, zorder=3)
            ax_bot.plot(x, d["mean"], "o", mfc=SURFACE, mec=c, mew=1.6, ms=6, zorder=3)
            ax_bot.annotate(
                d["demo"].replace("m_", ""),
                xy=(x, d["peak"]), xytext=(0, 7), textcoords="offset points",
                ha="center", color=INK_SECONDARY, fontsize=8,
            )
    ax_bot.set_ylabel("Squeeze Force (N)", color=INK_SECONDARY, fontsize=10)
    ax_bot.set_xlabel("Reach Distance", color=INK_SECONDARY, fontsize=10)
    ax_bot.set_xticks(sorted(k for k in by_dist if k))
    ax_bot.legend(
        handles=[
            plt.Line2D([], [], color=INK_MUTED, marker="o", ls="none", ms=7, label="peak"),
            plt.Line2D([], [], color=INK_MUTED, marker="o", ls="none", ms=6, mfc=SURFACE, mew=1.6,
                       label="mean"),
        ],
        frameon=False, ncol=2, loc="upper left", fontsize=9, labelcolor=INK_SECONDARY, handlelength=1.6,
    )
    style_axes(ax_bot)

    fig.suptitle(
        f"{hand} Squeeze Force Across {len(demos)} Demos — Individually Trained Policies",
        x=0.011, ha="left", color=INK, fontsize=13, fontweight="semibold",
    )
    missing = [d["demo"] for d in demos if not d["dist"]]
    note = f" · {len(missing)} demo(s) not in the fold CSV" if missing else ""
    fig.text(
        0.011, 0.938,
        f"one line per demo, coloured by reach distance · normalized time{note}",
        ha="left", color=INK_MUTED, fontsize=9.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.928])

    out = args.out or os.path.join(os.path.dirname(sorted(csv_paths)[0]), f"grip_comparison_{side}.png")
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"wrote {out}")

    print(f"\n{hand} squeeze, {len(demos)} demos:")
    print(f"  {'demo':>10} {'dist':>5} {'frames':>7} {'peak':>8} {'mean':>8}")
    for d in sorted(demos, key=lambda g: (g["dist"] or 0, g["demo"])):
        print(
            f"  {d['demo']:>10} {str(d['dist'] or '-'):>5} {len(d['squeeze']):>7} "
            f"{d['peak']:8.3f} {d['mean']:8.3f}"
        )


def plot_single(args):
    data = np.genfromtxt(args.csv[0], delimiter=",", names=True)
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
        f"{os.path.basename(args.csv[0])} · frames 0-{n - 1} · Inspire hand, contact bodies only",
        ha="left",
        color=INK_MUTED,
        fontsize=9.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.915])

    out = args.out or os.path.splitext(args.csv[0])[0] + f"_{side}_forces.png"
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"wrote {out}")

    print(f"\n{hand} frames 0-{n - 1}:")
    for finger, series in per_finger.items():
        print(f"  {finger:7s} peak {series.max():6.3f} N   mean {series.mean():6.3f} N")
    print(f"  {'total':7s} peak {total.max():6.3f} N   mean {total.mean():6.3f} N")
    print(f"  {'squeeze':7s} peak {squeeze.max():6.3f} N   mean {squeeze.mean():6.3f} N")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", nargs="+", help="grip log CSV(s); several switches to comparison mode")
    parser.add_argument("--side", default="rh", choices=["rh", "lh"])
    parser.add_argument("--frames", type=int, default=141, help="leading frames to plot (single CSV only)")
    parser.add_argument("--obj-weight", type=float, default=None, help="object weight (N) reference line")
    parser.add_argument(
        "--folds-csv", default="data_stats/reach_folds.csv",
        help="supplies demo -> reach distance for grouping and colour in comparison mode",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if len(args.csv) == 1:
        plot_single(args)
    else:
        plot_comparison(args, args.csv)


if __name__ == "__main__":
    main()
