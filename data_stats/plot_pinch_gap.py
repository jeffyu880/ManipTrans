"""Plot the thumb-index fingertip separation, human (Apple Vision Pro) vs robot (sim), from a
live run — split into the in-plane (XY) and vertical (Z) components.

The CSV is produced by DexHandManipBiHEnv._log_pinch_gap and written when live mode exits;
that same exit also runs this script, so it normally needs no manual invocation.

    python data_stats/plot_pinch_gap.py runs/<exp>/pinch_logs/pinch_gap__<stamp>.csv

Columns are `{rh,lh}_{avp,sim}_{thumb,index}_{x,y,z}` — fingertip positions in metres, all four
in the same gym frame — one row per control step. Separations are plotted in mm. The CSV also
carries `{rh,lh}_force_{thumb,index}_{x,y,z}` (net contact force on those fingers, N); this
script does not plot them.

The two curves are not expected to coincide: retargeting matches joint configuration, not the
human's absolute tip separation, and the Inspire fingers are a different length. What matters is
whether the robot gap tracks the human's, and where it stops closing (object in the way).
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

# categorical slots 1-2 of the repo palette (see plot_grip_forces.py); validated for the light
# surface: adjacent CVD dE 24.7, normal 33.6, both >= 3:1 contrast — all checks PASS
SOURCES = (("avp", "Human (AVP)", "#2a78d6"), ("sim", "Robot (Sim)", "#eb6834"))
# the force figure is a separate figure with its own legend, so it reuses the same two slots —
# they are the only ones in the repo palette that also clear the 3:1 contrast check
FINGERS = (("thumb", "Thumb", "#2a78d6"), ("index", "Index", "#eb6834"))
# all five fingers, in the env's _TIP_LABELS order. Palette slots 1-5, validated on this surface:
# CVD dE 9.1 worst adjacent, normal 19.6 — PASS; contrast WARNs on slots 3-5 (2.74/2.11/2.62),
# for which the printed per-finger stats table is the required relief.
TIPS = (
    ("thumb", "Thumb", "#2a78d6"),
    ("index", "Index", "#eb6834"),
    ("middle", "Middle", "#1baf7a"),
    ("ring", "Ring", "#eda100"),
    ("pinky", "Pinky", "#e87ba4"),
)
TIP_LABELS = tuple(f for f, _, _ in TIPS)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# (row label, how to reduce the thumb->index offset to one number)
MEASURES = (
    ("In-Plane XY", lambda d: np.hypot(d[0], d[1])),
    ("Vertical Z", lambda d: np.abs(d[2])),
)

# Cap diameter from the asset mesh (data/OakInk-v2/object_preview/align_ds/O02@0206@00001/scan.ply):
# bbox 32.35 x 47.65 x 33.53 mm — a cylinder along its long axis, so the diameter is the mean
# cross-section, (32.35 + 33.53) / 2. This is the floor a real pinch should respect: two fingertips
# closing on opposite sides of the cap end up a diameter apart, not a radius.
# Multiply by --cap-scale for a run with objScaleRH != 1.
CAP_DIAMETER_MM = 32.94


def column(data, name, n):
    return data[name][:n]


def offset(data, side, src, n):
    """thumb -> index offset per axis, [3, n], in mm."""
    return np.stack([data[f"{side}_{src}_thumb_{a}"][:n] - data[f"{side}_{src}_index_{a}"][:n] for a in "xyz"]) * 1e3


def force_mag(data, side, finger, n):
    """|net contact force| on one finger's contact body, [n], in N."""
    return np.linalg.norm(np.stack([data[f"{side}_force_{finger}_{a}"][:n] for a in "xyz"]), axis=0)


def squeeze(data, side, finger, n, origin="sim"):
    """Contact force projected onto the unit vector from `origin` toward the object COM: only the
    toward-cap part, dropping shear/tangential. Positive = pressing in, [n], in N.

    The force itself can only come from the contact body (thumb_distal / *_intermediate) — the
    _tip links carry no collision geometry, so net_cf on them is measured zero. The ORIGIN of the
    direction is a choice:
      "sim" — the fingertip frame, where contact physically happens. Only thumb and index have
              one logged, since those are the pinch pair the separation figure tracks.
      "cf"  — the contact body's own origin, back at the knuckle (23 mm from the thumb tip,
              44 mm from the index tip). Available for all five fingers, and exactly what
              _grip_side_metrics does, so it is what reproduces its {side}_tip_in_sum.
    """
    force = np.stack([data[f"{side}_force_{finger}_{a}"][:n] for a in "xyz"])
    to_com = np.stack([data[f"{side}_obj_com_{a}"][:n] - data[f"{side}_{origin}_{finger}_{a}"][:n] for a in "xyz"])
    # a fingertip exactly at the COM has no defined direction; floor the norm so it reads 0, not nan
    return (force * (to_com / np.maximum(np.linalg.norm(to_com, axis=0), 1e-12))).sum(axis=0)


def tracking_offset(data, side, finger, n):
    """Signed sim - AVP fingertip offset per axis, [3, n], in mm. Both are in the gym frame, so
    this is the retargeted hand's position error against the human it is following."""
    return np.stack([data[f"{side}_sim_{finger}_{a}"][:n] - data[f"{side}_avp_{finger}_{a}"][:n] for a in "xyz"]) * 1e3


def tracking_error(data, side, finger, n):
    """|sim fingertip - AVP fingertip| for one finger, [n], in mm."""
    return np.linalg.norm(tracking_offset(data, side, finger, n), axis=0)


def ramp(hue, name):
    """Sequential ramp for one categorical hue: light (early) -> saturated -> dark (late). Encodes
    time as lightness while the hue keeps carrying series identity."""
    light = tuple(1 - 0.22 * (1 - c) for c in plt.matplotlib.colors.to_rgb(hue))
    dark = tuple(0.38 * c for c in plt.matplotlib.colors.to_rgb(hue))
    return LinearSegmentedColormap.from_list(name, [light, hue, dark])


def draw_path(ax, x, y, steps, hue, name):
    """Fingertip path in the XY plane, coloured light->dark by control step, with an open marker
    at the start and a filled one at the end so direction is unambiguous."""
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    lc = LineCollection(
        np.concatenate([pts[:-1], pts[1:]], axis=1),
        cmap=ramp(hue, name),
        norm=plt.Normalize(steps[0], steps[-1]),
        linewidth=1.5,
        zorder=3,
    )
    lc.set_array(steps[:-1])
    ax.add_collection(lc)
    ax.plot(x[0], y[0], marker="o", ms=6, mfc="none", mec=hue, mew=1.4, zorder=4)
    ax.plot(x[-1], y[-1], marker="o", ms=6, color=hue, zorder=4)


def tip_in_sum(data, side, n):
    """_grip_side_metrics' {side}_tip_in_sum: the inward component summed over all five tips."""
    return sum(squeeze(data, side, f, n, origin="cf") for f in TIP_LABELS)


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


def build_figure(rows, sides, steps, title, subtitle, series_of, share_rows=None):
    """One figure: `rows` of (ylabel, key, hline) x one column per hand. series_of(key, side)
    yields (label, color, values) for a panel; hline is an optional (label, value) reference
    drawn across that row.

    share_rows selects which row indices put both hands on ONE y scale (None = all of them).
    A row is worth sharing only when the hands are comparable in magnitude — for contact force
    they are not, so a shared scale squashes the quieter hand flat against the axis."""
    n = len(steps)
    fig, axes = plt.subplots(
        len(rows), len(sides), figsize=(5.4 * len(sides) + 0.8, 3.4 * len(rows) + 1.0),
        sharex=True, squeeze=False,
    )
    fig.patch.set_facecolor(SURFACE)

    for row, (ylabel, key, hline) in enumerate(rows):
        for col, side in enumerate(sides):
            ax = axes[row][col]
            # identity comes from the figure legend: the two hues clear every palette check
            # (CVD dE 24.7, normal 33.6, both >= 3:1), so no in-plot labels are needed
            for label, color, values in series_of(key, side):
                ax.plot(steps, values, color=color, linewidth=1.5, label=label, zorder=3)
            if hline is not None:
                ax.axhline(
                    hline[1], color=INK_MUTED, linewidth=1.2, linestyle=(0, (5, 3)), label=hline[0], zorder=2
                )
            style_axes(ax)
            if col == 0:
                ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)
            if row == 0 and len(sides) > 1:
                ax.set_title(side.upper(), color=INK_SECONDARY, fontsize=10, fontweight="medium")
    # Put the shared rows onto one scale explicitly rather than via sharey="row", so the choice is
    # per row; the right-hand panel then drops its duplicate tick labels as sharey would have.
    for row in range(len(rows)):
        if share_rows is not None and row not in share_rows:
            continue
        lo = min(ax.get_ylim()[0] for ax in axes[row])
        hi = max(ax.get_ylim()[1] for ax in axes[row])
        for col, ax in enumerate(axes[row]):
            ax.set_ylim(lo, hi)
            if col:
                ax.tick_params(labelleft=False)
    for ax in axes[-1]:
        ax.set_xlabel("Control Step", color=INK_SECONDARY, fontsize=10)
    axes[0][0].set_xlim(0, max(1, n - 1))

    fig.suptitle(title, x=0.011, ha="left", color=INK, fontsize=13, fontweight="semibold")
    fig.text(0.011, 1 - 0.55 / fig.get_figheight(), subtitle, ha="left", color=INK_MUTED, fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 1 - 0.85 / fig.get_figheight()], h_pad=3.0)
    # One legend for the whole figure, gathered across every row (rows can carry different series)
    # and deduped. It hangs just above the top-right panel rather than in the header beside the
    # title, so it reads as belonging to the graphs; inside the axes it would sit on the curves.
    handles = {}
    for ax in axes.ravel():
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, handle)
    axes[0][-1].legend(
        handles.values(),
        handles.keys(),
        loc="lower right",
        bbox_to_anchor=(1.0, 1.10),
        ncol=len(handles),
        frameon=False,
        fontsize=9,
        labelcolor=INK_SECONDARY,
        handlelength=1.6,
    )
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="pinch-gap log CSV")
    parser.add_argument("--side", default="both", choices=["rh", "lh", "both"])
    parser.add_argument("--frames", type=int, default=None, help="plot only the leading N steps")
    parser.add_argument("--out", default=None, help="separation PNG; the force PNG sits beside it")
    parser.add_argument(
        "--cap-scale",
        type=float,
        default=1.0,
        help="objScaleRH the run used; scales the cap-diameter reference to match the sim's cap",
    )
    args = parser.parse_args()

    data = np.atleast_1d(np.genfromtxt(args.csv, delimiter=",", names=True))
    n = min(args.frames, len(data)) if args.frames else len(data)
    steps = np.arange(n)
    sides = ["rh", "lh"] if args.side == "both" else [args.side]
    offsets = {(side, src): offset(data, side, src, n) for side in sides for src, _, _ in SOURCES}
    forces = {(side, f): force_mag(data, side, f, n) for side in sides for f, _, _ in FINGERS}
    subtitle = f"{os.path.basename(args.csv)} · {n} control steps at 60 Hz"
    # The log records the objScaleRH the run actually used, so the reference matches the sim's cap
    # without being told. --cap-scale only fills in for logs written before that column existed.
    if "rh_obj_scale" in (data.dtype.names or ()):
        cap_scale = float(data["rh_obj_scale"][0])
        source = "objScaleRH from the log"
    else:
        cap_scale = args.cap_scale
        source = "--cap-scale (log predates rh_obj_scale)"
    diameter = CAP_DIAMETER_MM * cap_scale
    print(f"cap diameter {diameter:.1f} mm = {CAP_DIAMETER_MM} x {cap_scale:.3f} [{source}]")
    cap_ref = (f"Cap Diameter ({diameter:.1f} mm)", diameter)
    # squeeze is reported in the printed stats only, not plotted; logs written before the
    # contact-body/COM columns existed cannot produce it at all
    has_squeeze = f"{sides[0]}_obj_com_x" in (data.dtype.names or ())
    if not has_squeeze:
        print("note: CSV predates the cf/obj_com columns — no squeeze / tip_in_sum stats")

    fig = build_figure(
        [(f"Thumb-Index Gap, {m} [mm]", m, cap_ref if m == "In-Plane XY" else None) for m, _ in MEASURES],
        sides,
        steps,
        "Thumb-Index Fingertip Separation — Human vs Robot (Live)",
        subtitle,
        lambda key, side: [
            (label, color, dict(MEASURES)[key](offsets[(side, src)])) for src, label, color in SOURCES
        ],
    )
    out = args.out or os.path.splitext(args.csv)[0] + "_pinch.png"
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"wrote {out}")

    # Second figure: what the thumb and index are actually pushing on the object with, and how far
    # apart they are while doing it. Separate from the separation figure because the series mean a
    # different thing (finger, not source). Distance gets its own row rather than a second y-axis.
    # The inward/squeeze projection stays out of the plot — see the printed stats for it.
    rows = [
        ("Fingertip Contact Force |F| [N]", "force", None),
        ("Robot Tip Distance [mm]", "dist", cap_ref),
    ]

    def force_series(key, side):
        if key == "force":
            return [(label, color, forces[(side, f)]) for f, label, color in FINGERS]
        return [("Tip Distance", INK_SECONDARY, np.linalg.norm(offsets[(side, "sim")], axis=0))]

    # row 0 (contact force) gets a per-hand scale: RH and LH differ by more than an order of
    # magnitude, so one shared axis flattens whichever hand is quieter. Row 1 (tip distance) stays
    # shared — both hands are tens of mm, and the cap reference only compares across a common scale.
    fig_f = build_figure(
        rows, sides, steps, "Fingertip Contact Force on the Object — Live", subtitle, force_series,
        share_rows={1},
    )
    out_f = os.path.splitext(out)[0].replace("_pinch", "") + "_forces.png"
    fig_f.savefig(out_f, dpi=170, facecolor=SURFACE)
    print(f"wrote {out_f}")

    # Third figure: how far each retargeted fingertip sits from the human one it is following.
    # Only possible once all five fingers are logged on both sides.
    tracked = [f for f, _, _ in TIPS if f"{sides[0]}_avp_{f}_x" in (data.dtype.names or ())]
    if tracked:
        path_side = "rh" if "rh" in sides else sides[0]
        fig_t, axes_t = plt.subplots(2, 2, figsize=(11.6, 9.4))
        fig_t.patch.set_facecolor(SURFACE)

        # Row 1: error magnitude over time, one panel per hand.
        for col in range(2):
            ax = axes_t[0][col]
            if col < len(sides):
                for f, label, color in TIPS:
                    if f in tracked:
                        ax.plot(steps, tracking_error(data, sides[col], f, n), color=color, linewidth=1.5, zorder=3)
                ax.set_title(sides[col].upper(), color=INK_SECONDARY, fontsize=10, fontweight="medium")
                ax.set_xlim(0, max(1, n - 1))
                ax.set_xlabel("Control Step", color=INK_SECONDARY, fontsize=10)
                style_axes(ax)
            else:
                ax.set_visible(False)
        axes_t[0][0].set_ylabel("Fingertip Position Error [mm]", color=INK_SECONDARY, fontsize=10)

        # Row 2: the actual XY path of each fingertip, human vs robot, time as a light->dark ramp.
        # Separate axes from row 1 because x is a POSITION here, not a step; equal aspect so the
        # path keeps its true shape.
        for col, finger in enumerate(("thumb", "index")):
            ax = axes_t[1][col]
            for src, _, hue in SOURCES:
                draw_path(
                    ax,
                    data[f"{path_side}_{src}_{finger}_x"][:n] * 1e3,
                    data[f"{path_side}_{src}_{finger}_y"][:n] * 1e3,
                    steps,
                    hue,
                    f"{src}_{finger}",
                )
            ax.set_title(
                f"{path_side.upper()} {finger.capitalize()} — XY Path",
                color=INK_SECONDARY, fontsize=10, fontweight="medium",
            )
            ax.set_xlabel("X Position [mm]", color=INK_SECONDARY, fontsize=10)
            ax.autoscale_view()
            ax.set_aspect("equal", adjustable="datalim")
            style_axes(ax)
        axes_t[1][0].set_ylabel("Y Position [mm]", color=INK_SECONDARY, fontsize=10)

        fig_t.suptitle(
            "Fingertip Tracking Error — Sim vs AVP (Live)",
            x=0.011, ha="left", color=INK, fontsize=13, fontweight="semibold",
        )
        fig_t.text(
            0.011, 1 - 0.55 / fig_t.get_figheight(),
            f"{subtitle} · paths run light (early) to dark (late); open marker = start, filled = end",
            ha="left", color=INK_MUTED, fontsize=9.5,
        )
        # h_pad opens up the gap between the two rows: row 1 ends in an x-axis label and row 2
        # starts with a title plus its own legend, which collide at the default spacing.
        fig_t.tight_layout(rect=[0, 0, 1, 1 - 0.85 / fig_t.get_figheight()], h_pad=4.0)
        # Two legends, one per row: the same two hues mean FINGER in row 1 and SOURCE in row 2, so
        # a single combined legend would map blue to both "Thumb" and "Human". Each is anchored
        # just above its own row's right-hand axes, so it sits with the graphs it describes rather
        # than floating up in the header.
        axes_t[0][1].legend(
            handles=[Line2D([], [], color=c, lw=1.5, label=lab) for f, lab, c in TIPS if f in tracked],
            loc="lower right",
            bbox_to_anchor=(1.0, 1.10),
            ncol=len(tracked),
            frameon=False,
            fontsize=9,
            labelcolor=INK_SECONDARY,
            handlelength=1.6,
        )
        axes_t[1][1].legend(
            handles=[Line2D([], [], color=c, lw=1.5, label=lab) for _, lab, c in SOURCES],
            loc="lower right",
            bbox_to_anchor=(1.0, 1.10),
            ncol=len(SOURCES),
            frameon=False,
            fontsize=9,
            labelcolor=INK_SECONDARY,
            handlelength=1.6,
        )
        out_t = os.path.splitext(out)[0].replace("_pinch", "") + "_tracking.png"
        fig_t.savefig(out_t, dpi=170, facecolor=SURFACE)
        print(f"wrote {out_t}")
    else:
        print("note: CSV has no avp fingertip columns — skipping the tracking-error figure")

    for side in sides:
        print(f"\n{side.upper()} steps 0-{n - 1}:")
        for measure, reduce_fn in MEASURES:
            for src, label, _ in SOURCES:
                v = reduce_fn(offsets[(side, src)])
                print(f"  {measure:12s} {label:12s} min {v.min():7.1f}  mean {v.mean():7.1f}  max {v.max():7.1f}  mm")
        for f, label, _ in FINGERS:
            v = forces[(side, f)]
            print(f"  {'force':12s} {label:12s} min {v.min():7.3f}  mean {v.mean():7.3f}  max {v.max():7.3f}  N")
            if has_squeeze:
                s = squeeze(data, side, f, n)
                print(f"  {'squeeze':12s} {label:12s} min {s.min():7.3f}  mean {s.mean():7.3f}  max {s.max():7.3f}  N")
        if has_squeeze and f"{side}_force_pinky_x" in data.dtype.names:
            v = tip_in_sum(data, side, n)
            print(f"  {'tip_in_sum':12s} {'5 tips':12s} min {v.min():7.3f}  mean {v.mean():7.3f}  max {v.max():7.3f}  N")
        # per-axis tracking error: signed bias (mean of sim - avp) alongside the mean |offset|,
        # because a consistent offset in one axis and symmetric jitter read very differently
        tracked_here = [t for t in TIPS if f"{side}_avp_{t[0]}_x" in (data.dtype.names or ())]
        if tracked_here:
            print(f"  {'':12s} {'':12s} {'|dx|':>7} {'|dy|':>7} {'|dz|':>7} {'|d|':>7}   "
                  f"{'bias x':>7} {'bias y':>7} {'bias z':>7}  mm")
        for f, label, _ in tracked_here:
            d = tracking_offset(data, side, f, n)
            a = np.abs(d).mean(axis=1)
            b = d.mean(axis=1)
            e = tracking_error(data, side, f, n)
            print(f"  {'track err':12s} {label:12s} {a[0]:7.1f} {a[1]:7.1f} {a[2]:7.1f} {e.mean():7.1f}   "
                  f"{b[0]:+7.1f} {b[1]:+7.1f} {b[2]:+7.1f}")


if __name__ == "__main__":
    main()
