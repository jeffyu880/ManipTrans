"""Plot the thumb-index fingertip separation, human (Apple Vision Pro) vs robot (sim), from a
live run — split into the in-plane (XY) and vertical (Z) components.

The CSV is produced by DexHandManipBiHEnv._log_pinch_gap and written when live mode exits;
that same exit also runs this script, so it normally needs no manual invocation.

    python data_stats/plot_pinch_gap.py runs/<exp>/pinch_logs/pinch_gap__<stamp>.csv

Columns are `{rh,lh}_{avp,sim}_{thumb,index}_{x,y,z}` — fingertip positions in metres, all four
in the same gym frame — one row per control step. Separations are plotted in mm. The CSV also
carries `{rh,lh}_force_{thumb,index}_{x,y,z}` (net contact force on those fingers, N), plotted as
a third row beneath the two separation components.

Three figures are written beside the CSV: `_pinch.png` (separation + force), `_tracking.png`
(per-fingertip sim-vs-human error), and `_objects.png` (the cap's and bottle body's own
trajectories, from the `{rh,lh}_obj_com_*` columns — achieved pose only, no demo reference).

The two curves are not expected to coincide: retargeting matches joint configuration, not the
human's absolute tip separation, and the Inspire fingers are a different length. What matters is
whether the robot gap tracks the human's, and where it stops closing (object in the way).
"""

import argparse
import glob as globlib
import os
import re

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
# Object-trajectory figure: palette slots 1-3 for the three world axes. Validated on this
# surface — adjacent CVD dE 9.2 (deutan), normal-vision 27.6, both PASS; contrast WARNs on the
# green (2.74), for which the direct end-labels and the printed object stats are the relief.
OBJ_AXES = (("x", "X", "#2a78d6"), ("y", "Y", "#eb6834"), ("z", "Z", "#1baf7a"))
# RH carries the cap, LH the bottle body — the same convention as the dataset loaders.
OBJECTS = (("rh", "Cap (RH)"), ("lh", "Bottle Body (LH)"))
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
# The integrals additionally cover the full 3D separation, which the figure splits into the two
# components above but which is the natural single number for "how far apart were the tips".
INTEGRALS = MEASURES + (("3D Distance", lambda d: np.linalg.norm(d, axis=0)),)

FPS = 60.0  # control rate of the live log; sets dx for the separation integrals

# measured from 3d print
CAP_LOWER_DIAMETER_MM = 29.6
CAP_UPPER_DIAMETER_MM = 27.2
CAP_DIAMETER_MM = (CAP_LOWER_DIAMETER_MM + CAP_UPPER_DIAMETER_MM) / 2


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


def integrate(v, fps=FPS):
    """Area under a per-step series by the trapezoid rule, in <series unit> * s. The log is one row
    per control step, so dx is the control period — integrating a mm separation gives mm*s."""
    return np.trapz(v, dx=1.0 / fps)


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


def build_figure(rows, sides, steps, title, subtitle, series_of, share_rows=None, hline_sides=None):
    """One figure: `rows` of (ylabel, key, hline) x one column per hand. series_of(key, side)
    yields (label, color, values) for a panel; hline is an optional (label, value) reference
    drawn across that row.

    share_rows selects which row indices put both hands on ONE y scale (None = all of them).
    A row is worth sharing only when the hands are comparable in magnitude — for contact force
    they are not, so a shared scale squashes the quieter hand flat against the axis.

    hline_sides limits which hands get the reference line (None = all of them). The cap diameter
    only means something for the hand holding the cap."""
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
            if hline is not None and (hline_sides is None or side in hline_sides):
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
    # h_pad opens the gap between rows: a block that starts mid-figure hangs its legend above its
    # own top row, and tight_layout runs before the legends exist so it reserves no room for them.
    fig.tight_layout(rect=[0, 0, 1, 1 - 0.85 / fig.get_figheight()], h_pad=4.5)
    # One legend per contiguous block of rows whose series mean the same thing, hung just above that
    # block's top-right panel rather than in the header beside the title, so it reads as belonging to
    # the graphs it describes. Blocks matter because the separation rows and the force row reuse the
    # SAME two hues for different things (source vs finger): a single figure-wide legend would map
    # blue to both "Human (AVP)" and "Thumb". Rows are grouped by their series labels, so an hline
    # that only some rows carry joins its block's legend instead of splitting one off.
    row_labels = [tuple(label for label, _, _ in series_of(key, sides[0])) for _, key, _ in rows]
    start = 0
    for row in range(len(rows) + 1):
        if row < len(rows) and row_labels[row] == row_labels[start]:
            continue
        handles = {}
        for block_row in range(start, row):
            for ax in axes[block_row]:
                for handle, label in zip(*ax.get_legend_handles_labels()):
                    handles.setdefault(label, handle)
        axes[start][-1].legend(
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
        start = row
    return fig


# --compare: two hues for the two MODELS being compared. Same slots the single-run figures use
# for other meanings, which is fine because each figure carries its own legend — but it is why the
# legend is mandatory here. Validated on this surface: CVD dE 24.7 (protan), normal 33.6,
# contrast >= 3:1 — all checks PASS.
MODEL_COLORS = ("#2a78d6", "#eb6834")
# A step counts as "in contact" when the five contact bodies together carry more than this. PhysX
# reports exactly 0 N when nothing touches, so the threshold only rejects solver noise — it is not
# a tuned parameter, and any value in 0.01-0.1 N gives the same split.
CONTACT_FORCE_N = 0.05
# (label, is-in-contact) — the phase split both phase figures use
PHASES = (("Free", False), ("Contact", True))
# grouped-bar geometry: a 2px surface gap between adjacent bars, per the mark spec
BAR_W = 0.38
BAR_GAP = 0.02


def demo_key(path):
    """The m_<digits> index a log belongs to, so two model runs of the same demo can be paired."""
    m = re.search(r"(m_\d+)", os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)


def grouped_bars(ax, groups, series, ylabel, clip_at_zero=False):
    """One grouped bar block: series = [(label, color, means, stds, points)], one entry per group.

    Bars are the MEAN error, whiskers +/- 1 SD. `points` (per group, one value per demo) is
    overlaid when given: with a distribution this skewed the mean alone is misleading — a couple
    of diverged demos can carry it — and the dots show whether a tall bar is the whole population
    or two outliers. clip_at_zero stops the lower whisker crossing zero for unsigned quantities
    like a distance, where a negative value is not a possible measurement.

    Every bar is value-labelled: with this few bars a direct label on each is the table view, and
    it keeps the reading off color alone.
    """
    x = np.arange(len(groups))
    for i, (label, color, means, stds, points) in enumerate(series):
        off = (i - (len(series) - 1) / 2) * (BAR_W + BAR_GAP)
        lower = np.minimum(stds, means) if clip_at_zero else stds
        ax.bar(
            x + off, means, BAR_W, yerr=[lower, stds], label=label, color=color, zorder=3,
            error_kw=dict(ecolor=INK_MUTED, elinewidth=1.2, capsize=3),
        )
        if points is not None:
            for xi, vals in zip(x + off, points):
                vals = np.asarray(vals, dtype=float)
                # deterministic spread across the bar's width — no RNG, so reruns are identical
                spread = np.linspace(-BAR_W * 0.3, BAR_W * 0.3, len(vals)) if len(vals) > 1 else [0.0]
                ax.plot(
                    xi + np.asarray(spread), vals, "o", markersize=3.5, color=INK_SECONDARY,
                    alpha=0.55, markeredgewidth=0, zorder=5,
                )
        # The label is the BAR's value (the mean), so it sits at the bar tip — nudged sideways to
        # clear the error bar, which is drawn down the bar's centre. Parking it at the whisker end
        # instead reads as if it labelled the whisker.
        for xi, m, sd in zip(x + off, means, stds):
            up = m >= 0
            ax.annotate(
                f"{m:.1f}",
                xy=(xi + BAR_W * 0.30, m),
                xytext=(0, 3 if up else -3),
                textcoords="offset points",
                ha="center",
                va="bottom" if up else "top",
                fontsize=8,
                color=INK_SECONDARY,
                zorder=6,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(groups, color=INK_MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)
    style_axes(ax)
    ax.grid(False, axis="x")  # bars already separate the categories; vertical lines just add ink


def compare_models(specs, out_base, exclude=()):
    """Imitator vs residual, aggregated over every demo the two sets share.

    Two figures. The first is the thumb-index GAP error: how far each model's tip separation sits
    from the human's, split into the same in-plane/vertical components the single-run figure uses.
    The second is the per-fingertip POSITION error, which the gap cannot show — a model can match
    the separation while both fingers sit in the wrong place.

    Each demo is reduced to its own mean first and the demos are then averaged with equal weight,
    so length does not buy influence and the SD is demo-to-demo spread (see `agg`). Signed for the
    gap (so the mean reads as bias: positive = the model holds its tips further apart than the
    human) and unsigned for the fingertip distance, which has no sign.
    """
    sets = []
    for label, pattern in specs:
        paths = sorted(globlib.glob(pattern))
        if not paths:
            raise SystemExit(f"--compare {label}: no CSVs matched {pattern}")
        sets.append((label, {demo_key(p): p for p in paths}))

    shared = sorted(set.intersection(*[set(d) for _, d in sets]))
    if not shared:
        raise SystemExit("--compare: the sets share no demo index; nothing to compare")
    for label, d in sets:
        extra = sorted(set(d) - set(shared))
        if extra:
            print(f"note: {label} has {len(extra)} demo(s) the other set lacks, excluded: {extra}")

    # Excluding demos changes what every number below means, so it is named on the figures too —
    # a mean over a hand-filtered population that does not say so is the easiest chart to mislead
    # with.
    dropped = [k for k in shared if k in set(exclude)]
    missing = sorted(set(exclude) - set(shared))
    if missing:
        print(f"note: --exclude listed {missing}, which is not in the shared set anyway")
    shared = [k for k in shared if k not in set(exclude)]
    if not shared:
        raise SystemExit("--exclude removed every shared demo; nothing left to compare")
    note = f" · {len(dropped)} excluded: {', '.join(dropped)}" if dropped else ""
    if dropped:
        print(f"excluding {len(dropped)} demo(s): {', '.join(dropped)}")
    print(f"comparing {len(sets)} models over {len(shared)} shared demos: {', '.join(shared)}")

    gap, tip, phase, frac, gap_phase, tip_phase = {}, {}, {}, {}, {}, {}
    for label, paths in sets:
        for key in shared:
            data = np.atleast_1d(np.genfromtxt(paths[key], delimiter=",", names=True))
            names = data.dtype.names or ()
            n = len(data)
            for side in ("rh", "lh"):
                # Contact split: both metrics mean something different once the object is in the
                # way — the fingers physically cannot reach the commanded pose — so the phases are
                # separated rather than averaged together. One mask serves both.
                touching = (
                    sum(force_mag(data, side, f, n) for f, _, _ in TIPS) > CONTACT_FORCE_N
                    if f"{side}_force_pinky_x" in names
                    else None
                )

                def by_phase(store, key, series):
                    """Per-demo mean of `series` within each phase; nan when a phase never occurs."""
                    if touching is None:
                        return
                    for pname, want in PHASES:
                        sel = touching if want else ~touching
                        store.setdefault(key + (pname,), []).append(
                            float(series[sel].mean()) if sel.any() else np.nan
                        )

                for measure, reduce_fn in MEASURES:
                    human = reduce_fn(offset(data, side, "avp", n))
                    robot = reduce_fn(offset(data, side, "sim", n))
                    err = robot - human
                    gap.setdefault((label, side, measure), []).append(err)
                    by_phase(gap_phase, (label, side, measure), err)

                tracked_here = [f for f, _, _ in TIPS if f"{side}_avp_{f}_x" in names]
                for f in tracked_here:
                    err_f = tracking_error(data, side, f, n)
                    tip.setdefault((label, side, f), []).append(err_f)
                    by_phase(tip_phase, (label, side, f), err_f)
                if tracked_here and touching is not None:
                    by_phase(
                        phase, (label, side),
                        np.stack([tracking_error(data, side, f, n) for f in tracked_here]).mean(axis=0),
                    )
                    frac.setdefault((label, side), []).append(float(touching.mean()) * 100.0)

    def per_demo(store, key):
        """One number per demo: that demo's mean error over its whole trajectory."""
        return [float(np.mean(a)) for a in store[key]]

    def agg(store, key):
        """Mean and SD taken over PER-DEMO means, one sample per demo.

        Not over pooled control steps. Within an episode the error swings through the reach /
        close / lift phases (lag-1 autocorrelation ~0.99), so a step-pooled SD is dominated by
        that temporal sweep and runs 3-4x wider than the demo-to-demo spread — which made the
        whiskers disagree with the per-demo dots drawn on the same axis. Aggregating per demo
        first makes the whisker answer the question this comparison is for: does the difference
        between models hold from one demo to the next. Sample SD (ddof=1), since the demos are a
        sample of possible trajectories, and each demo counts once regardless of its length.
        """
        per = np.asarray(per_demo(store, key), dtype=float)
        per = per[~np.isnan(per)]  # a demo that never entered this phase simply does not vote
        if per.size == 0:
            return float("nan"), 0.0
        return float(per.mean()), float(per.std(ddof=1) if per.size > 1 else 0.0)

    # ---- figure 1: thumb-index gap error — one panel per component, both hands in each ----
    # The two components answer different questions (does the pinch close far enough vs are the
    # tips at the same height), so they get a panel each; the hands sit together inside a panel
    # because that is the comparison being made. Shared y so panel heights mean the same thing.
    fig, axes1 = plt.subplots(1, 2, figsize=(12.4, 5.4), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for col, (measure, _) in enumerate(MEASURES):
        series = []
        for i, (label, _) in enumerate(sets):
            stats = [agg(gap, (label, s, measure)) for s in ("rh", "lh")]
            series.append((
                label, MODEL_COLORS[i % len(MODEL_COLORS)],
                [a for a, _ in stats], [b for _, b in stats],
                [per_demo(gap, (label, s, measure)) for s in ("rh", "lh")],
            ))
        grouped_bars(
            axes1[col], ["RH", "LH"], series,
            "Gap Error, Model - Human [mm]" if col == 0 else "",
        )
        axes1[col].axhline(0, color=BASELINE, linewidth=1.0, zorder=2)  # zero = matches the human
        axes1[col].set_title(measure, color=INK_SECONDARY, fontsize=10, fontweight="medium")
    axes1[1].legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, ncol=len(sets), loc="best")
    fig.suptitle(
        "Thumb-Index Gap Error vs Human", x=0.011, ha="left", color=INK, fontsize=13, fontweight="semibold"
    )
    fig.tight_layout(rect=[0, 0, 1, 1 - 0.55 / fig.get_figheight()])
    out_gap = f"{out_base}_gap_error.png"
    fig.savefig(out_gap, dpi=170, facecolor=SURFACE)
    print(f"wrote {out_gap}")

    # ---- figure 2: per-fingertip position error, one panel per hand ----
    fingers = [(f, lab) for f, lab, _ in TIPS if any((label, "rh", f) in tip for label, _ in sets)]
    if fingers:
        fig2, axes2 = plt.subplots(1, 2, figsize=(12.4, 5.4), sharey=True)
        fig2.patch.set_facecolor(SURFACE)
        for col, side in enumerate(("rh", "lh")):
            series2 = []
            for i, (label, _) in enumerate(sets):
                stats = [agg(tip, (label, side, f)) for f, _ in fingers]
                series2.append((
                    label, MODEL_COLORS[i % len(MODEL_COLORS)],
                    [a for a, _ in stats], [b for _, b in stats],
                    [per_demo(tip, (label, side, f)) for f, _ in fingers],
                ))
            grouped_bars(
                axes2[col], [lab for _, lab in fingers], series2,
                "Fingertip Position Error [mm]" if col == 0 else "",
                clip_at_zero=True,  # |sim - human| is a distance; a negative whisker is not a value
            )
            axes2[col].set_title(side.upper(), color=INK_SECONDARY, fontsize=10, fontweight="medium")
        axes2[1].legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, ncol=len(sets), loc="best")
        fig2.suptitle(
            "Fingertip Position Error vs Human",
            x=0.011, ha="left", color=INK, fontsize=13, fontweight="semibold",
        )
        fig2.tight_layout(rect=[0, 0, 1, 1 - 0.55 / fig2.get_figheight()])
        out_tip = f"{out_base}_tip_error.png"
        fig2.savefig(out_tip, dpi=170, facecolor=SURFACE)
        print(f"wrote {out_tip}")

    # ---- figure 4: thumb-index gap error, split by contact phase ----
    # The pinch gap is the measure the contact split matters most for: in free space it is pure
    # imitation of the human's hand shape, while in contact the cap sets a floor on how far the
    # tips can close. Averaging the two together hides both.
    if gap_phase:
        fig4, axes4 = plt.subplots(2, 2, figsize=(11.0, 8.4))
        fig4.patch.set_facecolor(SURFACE)
        for row, (measure, _) in enumerate(MEASURES):
            for col, side in enumerate(("rh", "lh")):
                series4 = []
                for i, (label, _) in enumerate(sets):
                    stats = [agg(gap_phase, (label, side, measure, p)) for p, _ in PHASES]
                    series4.append((
                        label, MODEL_COLORS[i % len(MODEL_COLORS)],
                        [a for a, _ in stats], [b for _, b in stats],
                        [[v for v in per_demo(gap_phase, (label, side, measure, p)) if not np.isnan(v)]
                         for p, _ in PHASES],
                    ))
                ax4 = axes4[row][col]
                grouped_bars(
                    ax4, [p for p, _ in PHASES], series4,
                    "Gap Error, Model - Human [mm]" if col == 0 else "",
                )
                ax4.axhline(0, color=BASELINE, linewidth=1.0, zorder=2)
                ax4.set_title(
                    f"{side.upper()} — {measure}", color=INK_SECONDARY, fontsize=10, fontweight="medium"
                )
            # one y scale per measure so the hands are comparable without flattening the other row
            lo = min(a.get_ylim()[0] for a in axes4[row])
            hi = max(a.get_ylim()[1] for a in axes4[row])
            for a in axes4[row]:
                a.set_ylim(lo, hi)

        fig4.suptitle(
            "Thumb-Index Gap Error by Contact Phase",
            x=0.011, ha="left", color=INK, fontsize=13, fontweight="semibold",
        )
        fig4.tight_layout(rect=[0, 0, 1, 1 - 0.55 / fig4.get_figheight()], h_pad=3.0)
        fig4.legend(
            handles=[
                Line2D([], [], color=MODEL_COLORS[i % len(MODEL_COLORS)], lw=6, label=lab)
                for i, (lab, _) in enumerate(sets)
            ],
            loc="upper right", bbox_to_anchor=(0.995, 0.995), ncol=len(sets),
            frameon=False, fontsize=9, labelcolor=INK_SECONDARY, handlelength=1.6,
        )
        out_g = f"{out_base}_gap_phase_error.png"
        fig4.savefig(out_g, dpi=170, facecolor=SURFACE)
        print(f"wrote {out_g}")

    # ---- figure 3: fingertip error split by contact phase, plus how long each model touches ----
    # Averaged over the whole trajectory the fingertip error conflates two regimes: free space,
    # where tracking the reference is unobstructed, and contact, where the object physically
    # blocks the commanded pose. A model that never grips scores well on the pooled number by
    # avoiding the hard half, so the contact fraction is plotted beside the errors.
    if phase:
        fig3, axes3 = plt.subplots(1, 3, figsize=(15.0, 5.0))
        fig3.patch.set_facecolor(SURFACE)
        for col, side in enumerate(("rh", "lh")):
            series3 = []
            for i, (label, _) in enumerate(sets):
                stats = [agg(phase, (label, side, p)) for p in ("Free", "Contact")]
                series3.append((
                    label, MODEL_COLORS[i % len(MODEL_COLORS)],
                    [a for a, _ in stats], [b for _, b in stats],
                    [[v for v in per_demo(phase, (label, side, p)) if not np.isnan(v)]
                     for p in ("Free", "Contact")],
                ))
            grouped_bars(
                axes3[col], ["Free", "Contact"], series3,
                "Fingertip Position Error [mm]" if col == 0 else "",
                clip_at_zero=True,
            )
            axes3[col].set_title(side.upper(), color=INK_SECONDARY, fontsize=10, fontweight="medium")

        series_f = []
        for i, (label, _) in enumerate(sets):
            stats = [agg(frac, (label, s)) for s in ("rh", "lh")]
            series_f.append((
                label, MODEL_COLORS[i % len(MODEL_COLORS)],
                [a for a, _ in stats], [b for _, b in stats],
                [per_demo(frac, (label, s)) for s in ("rh", "lh")],
            ))
        grouped_bars(axes3[2], ["RH", "LH"], series_f, "Steps in Contact [%]", clip_at_zero=True)
        axes3[2].set_title("Time in Contact", color=INK_SECONDARY, fontsize=10, fontweight="medium")

        fig3.suptitle(
            "Fingertip Position Error by Contact Phase",
            x=0.011, ha="left", color=INK, fontsize=13, fontweight="semibold",
        )
        fig3.tight_layout(rect=[0, 0, 1, 1 - 0.55 / fig3.get_figheight()])
        # figure-level legend in the header: three panels with bars filling each one leave no
        # in-axes corner free, and loc="best" lands it on top of a bar
        fig3.legend(
            handles=[
                Line2D([], [], color=MODEL_COLORS[i % len(MODEL_COLORS)], lw=6, label=lab)
                for i, (lab, _) in enumerate(sets)
            ],
            loc="upper right", bbox_to_anchor=(0.995, 0.995), ncol=len(sets),
            frameon=False, fontsize=9, labelcolor=INK_SECONDARY, handlelength=1.6,
        )
        out_p = f"{out_base}_phase_error.png"
        fig3.savefig(out_p, dpi=170, facecolor=SURFACE)
        print(f"wrote {out_p}")

    # ---- figure 5: per-fingertip position error, split by contact phase ----
    # Figure 3 averages the five fingers into one number per phase, which hides that they do not
    # fail the same way — the thumb and index carry the pinch while the outer fingers mostly follow
    # along. Same contact split, one bar per finger.
    if tip_phase and fingers:
        fig5, axes5 = plt.subplots(len(PHASES), 2, figsize=(12.4, 4.2 * len(PHASES)), squeeze=False)
        fig5.patch.set_facecolor(SURFACE)
        for row, (pname, _) in enumerate(PHASES):
            for col, side in enumerate(("rh", "lh")):
                series5 = []
                for i, (label, _) in enumerate(sets):
                    stats = [agg(tip_phase, (label, side, f, pname)) for f, _ in fingers]
                    series5.append((
                        label, MODEL_COLORS[i % len(MODEL_COLORS)],
                        [a for a, _ in stats], [b for _, b in stats],
                        [[v for v in per_demo(tip_phase, (label, side, f, pname)) if not np.isnan(v)]
                         for f, _ in fingers],
                    ))
                ax5 = axes5[row][col]
                grouped_bars(
                    ax5, [lab for _, lab in fingers], series5,
                    "Fingertip Position Error [mm]" if col == 0 else "",
                    clip_at_zero=True,
                )
                ax5.set_title(
                    f"{side.upper()} — {pname}", color=INK_SECONDARY, fontsize=10, fontweight="medium"
                )
        # ONE y scale across the whole grid, not per row: the comparison the figure exists to make
        # is Free vs Contact, and giving each phase its own axis is exactly what would erase it.
        lo = min(a.get_ylim()[0] for a in axes5.ravel())
        hi = max(a.get_ylim()[1] for a in axes5.ravel())
        for a in axes5.ravel():
            a.set_ylim(lo, hi)

        fig5.suptitle(
            "Fingertip Position Error by Finger and Contact Phase",
            x=0.011, ha="left", color=INK, fontsize=13, fontweight="semibold",
        )
        fig5.tight_layout(rect=[0, 0, 1, 1 - 0.55 / fig5.get_figheight()], h_pad=3.0)
        fig5.legend(
            handles=[
                Line2D([], [], color=MODEL_COLORS[i % len(MODEL_COLORS)], lw=6, label=lab)
                for i, (lab, _) in enumerate(sets)
            ],
            loc="upper right", bbox_to_anchor=(0.995, 0.995), ncol=len(sets),
            frameon=False, fontsize=9, labelcolor=INK_SECONDARY, handlelength=1.6,
        )
        out_tp = f"{out_base}_tip_phase_error.png"
        fig5.savefig(out_tp, dpi=170, facecolor=SURFACE)
        print(f"wrote {out_tp}")

    # the numbers behind both figures
    print(f"\n{'measure':<22} " + " ".join(f"{lab:>22}" for lab, _ in sets))
    for side in ("rh", "lh"):
        for measure, _ in MEASURES:
            cells = " ".join(f"{agg(gap, (l, side, measure))[0]:>10.2f} +/-{agg(gap, (l, side, measure))[1]:>8.2f}"
                             for l, _ in sets)
            print(f"{side.upper() + ' gap ' + measure:<22} {cells}")
    for side in ("rh", "lh"):
        for f, lab in fingers:
            cells = " ".join(f"{agg(tip, (l, side, f))[0]:>10.2f} +/-{agg(tip, (l, side, f))[1]:>8.2f}"
                             for l, _ in sets)
            print(f"{side.upper() + ' tip ' + lab:<22} {cells}")
    for side in ("rh", "lh"):
        for measure, _ in MEASURES:
            for p, _ in PHASES:
                if (sets[0][0], side, measure, p) not in gap_phase:
                    continue
                cells = " ".join(
                    f"{agg(gap_phase, (l, side, measure, p))[0]:>10.2f} "
                    f"+/-{agg(gap_phase, (l, side, measure, p))[1]:>8.2f}" for l, _ in sets)
                print(f"{side.upper() + ' ' + measure[:7] + ' ' + p:<22} {cells}")
    for side in ("rh", "lh"):
        for p, _ in PHASES:
            if (sets[0][0], side, p) not in phase:
                continue
            cells = " ".join(f"{agg(phase, (l, side, p))[0]:>10.2f} +/-{agg(phase, (l, side, p))[1]:>8.2f}"
                             for l, _ in sets)
            print(f"{side.upper() + ' tip ' + p:<22} {cells}")
    for side in ("rh", "lh"):
        if (sets[0][0], side) in frac:
            cells = " ".join(f"{agg(frac, (l, side))[0]:>10.2f} +/-{agg(frac, (l, side))[1]:>8.2f}"
                             for l, _ in sets)
            print(f"{side.upper() + ' contact %':<22} {cells}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?", help="pinch-gap log CSV")
    parser.add_argument("--side", default="both", choices=["rh", "lh", "both"])
    parser.add_argument("--frames", type=int, default=None, help="plot only the leading N steps")
    parser.add_argument("--out", default=None, help="separation PNG; the force PNG sits beside it")
    parser.add_argument(
        "--cap-scale",
        type=float,
        default=1.0,
        help="objScaleRH the run used; scales the cap-diameter reference to match the sim's cap",
    )
    parser.add_argument(
        "--compare",
        action="append",
        metavar="LABEL=GLOB",
        help="compare models instead of plotting one run; repeat once per model, e.g. "
        "--compare 'Imitator=logs/pinch_gap__demo_*.csv' --compare 'Residual=logs/residual/*.csv'. "
        "Runs are paired by the m_<digits> index in the filename.",
    )
    parser.add_argument("--compare-out", default=None, help="output prefix for the --compare figures")
    parser.add_argument(
        "--exclude",
        default="",
        help="comma-separated demo indices to drop from --compare (e.g. m_140843,m_141658). "
        "The exclusion is named in the figure subtitles.",
    )
    args = parser.parse_args()

    if args.compare:
        specs = []
        for spec in args.compare:
            if "=" not in spec:
                raise SystemExit(f"--compare expects LABEL=GLOB, got {spec!r}")
            label, pattern = spec.split("=", 1)
            specs.append((label, pattern))
        # One model is allowed: the same figures then read as a breakdown of that model (per finger,
        # per phase) rather than a comparison, which is the only way to get the phase splits for a
        # single set of runs.
        if not specs:
            raise SystemExit("--compare needs at least one LABEL=GLOB")
        base = args.compare_out or os.path.join(os.path.dirname(specs[0][1]) or ".", "model_comparison")
        compare_models(specs, base, exclude=[e for e in args.exclude.split(",") if e])
        return

    if not args.csv:
        raise SystemExit("a CSV is required unless --compare is used")

    data = np.atleast_1d(np.genfromtxt(args.csv, delimiter=",", names=True))
    n = min(args.frames, len(data)) if args.frames else len(data)
    steps = np.arange(n)
    sides = ["rh", "lh"] if args.side == "both" else [args.side]
    offsets = {(side, src): offset(data, side, src, n) for side in sides for src, _, _ in SOURCES}
    forces = {(side, f): force_mag(data, side, f, n) for side in sides for f, _, _ in FINGERS}
    subtitle = f"{os.path.basename(args.csv)} · {n} control steps at {FPS:.0f} Hz"
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

    # Separation on top, contact force underneath: the force row answers what the gap row cannot,
    # namely whether the fingers stopped closing because the object is between them. The two series
    # sets mean different things (source vs finger) and reuse the same hues, so build_figure gives
    # each block its own legend. The inward/squeeze projection stays out — see the printed stats.
    rows = [(f"Thumb-Index Gap, {m} [mm]", m, cap_ref if m == "In-Plane XY" else None) for m, _ in MEASURES]
    rows.append(("Fingertip Contact Force |F| [N]", "force", None))

    def series(key, side):
        if key == "force":
            return [(label, color, forces[(side, f)]) for f, label, color in FINGERS]
        return [(label, color, dict(MEASURES)[key](offsets[(side, src)])) for src, label, color in SOURCES]

    # The separation rows share one y scale across hands; the force row does NOT — RH and LH differ
    # by more than an order of magnitude there, so a shared axis flattens whichever hand is quieter.
    fig = build_figure(
        rows,
        sides,
        steps,
        "Thumb-Index Fingertip Separation and Contact Force — Live",
        subtitle,
        series,
        share_rows=set(range(len(MEASURES))),
        hline_sides={"rh"},  # the RH holds the cap; on the LH the diameter is not a reference
    )
    out = args.out or os.path.splitext(args.csv)[0] + "_pinch.png"
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"wrote {out}")

    # Second figure: how far each retargeted fingertip sits from the human one it is following.
    # Only possible once all five fingers are logged on both sides.
    tracked = [f for f, _, _ in TIPS if f"{sides[0]}_avp_{f}_x" in (data.dtype.names or ())]
    if tracked:
        path_side = "rh" if "rh" in sides else sides[0]
        fig_t, axes_t = plt.subplots(3, 2, figsize=(11.6, 13.4))
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

        # Row 3: the vertical (Z) coordinate the XY path above cannot show, back on a time axis —
        # Z is the up-axis here (sim up_axis="z"), so this is fingertip height over the table.
        for col, finger in enumerate(("thumb", "index")):
            ax = axes_t[2][col]
            for src, label, color in SOURCES:
                ax.plot(
                    steps,
                    data[f"{path_side}_{src}_{finger}_z"][:n] * 1e3,
                    color=color, linewidth=1.5, label=label, zorder=3,
                )
            ax.set_title(
                f"{path_side.upper()} {finger.capitalize()} — Z Height",
                color=INK_SECONDARY, fontsize=10, fontweight="medium",
            )
            ax.set_xlim(0, max(1, n - 1))
            ax.set_xlabel("Control Step", color=INK_SECONDARY, fontsize=10)
            style_axes(ax)
        axes_t[2][0].set_ylabel("Z Position [mm]", color=INK_SECONDARY, fontsize=10)
        # both fingers on one height scale so thumb and index are directly comparable
        lo = min(ax.get_ylim()[0] for ax in axes_t[2])
        hi = max(ax.get_ylim()[1] for ax in axes_t[2])
        for col, ax in enumerate(axes_t[2]):
            ax.set_ylim(lo, hi)
            if col:
                ax.tick_params(labelleft=False)

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

    # Third figure: what the manipulated objects themselves did — the cap (RH) and the bottle body
    # (LH). The fingertip figures above say where the hands went; neither says whether the objects
    # followed. Only the ACHIEVED pose is logged (the CSV carries no ground-truth object pose), so
    # this shows what the objects did, not how well they tracked the demo.
    if has_squeeze:  # same columns the squeeze stats need: {side}_obj_com_*
        obj = {
            s: np.stack([column(data, f"{s}_obj_com_{a}", n) for a, _, _ in OBJ_AXES], axis=1)
            for s, _ in OBJECTS
        }
        fig_o, axes_o = plt.subplots(2, 2, figsize=(11.6, 8.6))
        fig_o.patch.set_facecolor(SURFACE)

        for col, (side, label) in enumerate(OBJECTS):
            track = obj[side]

            # Row 1: position per axis against time, in mm to match every other figure here.
            ax = axes_o[0][col]
            for k, (_, axis_label, color) in enumerate(OBJ_AXES):
                ax.plot(steps, track[:, k] * 1e3, color=color, linewidth=1.5, label=axis_label, zorder=3)
            # Direct end-labels, the relief the green's contrast WARN obligates. Axes ending at
            # nearly the same value would overprint (a static object sits at ~0 on two of them),
            # so walk them in value order and push each clear of the last.
            span = np.ptp(ax.get_ylim())
            placed = None
            for k in sorted(range(len(OBJ_AXES)), key=lambda i: track[-1, i]):
                y = track[-1, k] * 1e3
                y = y if placed is None else max(y, placed + 0.05 * (span or 1.0))
                placed = y
                ax.annotate(
                    OBJ_AXES[k][1], xy=(steps[-1], y), xytext=(4, 0), textcoords="offset points",
                    color=OBJ_AXES[k][2], fontsize=9, fontweight="medium", va="center",
                    annotation_clip=False,
                )
            ax.set_title(label, color=INK_SECONDARY, fontsize=10, fontweight="medium")
            ax.set_xlim(0, max(1, n - 1))
            ax.set_xlabel("Control Step", color=INK_SECONDARY, fontsize=10)
            style_axes(ax)

            # Row 2: the XY path, drawn with the same light->dark time ramp as the fingertip paths
            # so the two figures read the same way. Equal aspect keeps the path's true shape.
            ax = axes_o[1][col]
            draw_path(ax, track[:, 0] * 1e3, track[:, 1] * 1e3, steps, OBJ_AXES[0][2], f"{side}_obj")
            ax.set_title(f"{label} — XY Path", color=INK_SECONDARY, fontsize=10, fontweight="medium")
            ax.set_xlabel("X Position [mm]", color=INK_SECONDARY, fontsize=10)
            ax.autoscale_view()
            ax.set_aspect("equal", adjustable="datalim")
            style_axes(ax)

        axes_o[0][0].set_ylabel("Object Position [mm]", color=INK_SECONDARY, fontsize=10)
        axes_o[1][0].set_ylabel("Y Position [mm]", color=INK_SECONDARY, fontsize=10)
        fig_o.suptitle(
            "Object Trajectories — Cap and Bottle Body",
            x=0.011, ha="left", color=INK, fontsize=13, fontweight="semibold",
        )
        fig_o.text(
            0.011, 1 - 0.55 / fig_o.get_figheight(),
            f"{subtitle} · achieved pose only, no demo reference; "
            "paths run light (early) to dark (late), open marker = start",
            ha="left", color=INK_MUTED, fontsize=9.5,
        )
        fig_o.tight_layout(rect=[0, 0, 1, 1 - 0.85 / fig_o.get_figheight()], h_pad=4.0)
        axes_o[0][1].legend(
            loc="lower right", bbox_to_anchor=(1.0, 1.10), ncol=len(OBJ_AXES),
            frameon=False, fontsize=9, labelcolor=INK_SECONDARY, handlelength=1.6,
        )
        out_o = os.path.splitext(out)[0].replace("_pinch", "") + "_objects.png"
        fig_o.savefig(out_o, dpi=170, facecolor=SURFACE)
        print(f"wrote {out_o}")
    else:
        print("note: CSV has no obj_com columns — skipping the object-trajectory figure")

    for side in sides:
        print(f"\n{side.upper()} steps 0-{n - 1}:")
        for measure, reduce_fn in MEASURES:
            for src, label, _ in SOURCES:
                v = reduce_fn(offsets[(side, src)])
                print(f"  {measure:12s} {label:12s} min {v.min():7.1f}  mean {v.mean():7.1f}  max {v.max():7.1f}  mm")
        # Area under the separation curve for each source, and the signed gap between them: one
        # number for how much total tip-separation each accumulated over the run. Sensitive to
        # duration, so only compare integrals from runs of the same length (or use the means above).
        print(f"  {'':12s} {'':12s} {'human':>9} {'robot':>9} {'robot-human':>12}  mm*s")
        for measure, reduce_fn in INTEGRALS:
            human = integrate(reduce_fn(offsets[(side, "avp")]))
            robot = integrate(reduce_fn(offsets[(side, "sim")]))
            print(f"  {'integral':12s} {measure:12s} {human:9.1f} {robot:9.1f} {robot - human:+12.1f}")
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

    # Object motion: net displacement AND path length, because a time series alone confuses the
    # two ways an object ends where it started. Both small = it never moved; net small but path
    # large = it moved and came back.
    if has_squeeze:
        print(f"\n{'object':<18} {'net mm':>9} {'path mm':>9} {'z range mm':>11}")
        for side, label in OBJECTS:
            track = np.stack([column(data, f"{side}_obj_com_{a}", n) for a, _, _ in OBJ_AXES], axis=1)
            net = np.linalg.norm(track[-1] - track[0]) * 1e3
            path = np.linalg.norm(np.diff(track, axis=0), axis=1).sum() * 1e3
            z_range = (track[:, 2].max() - track[:, 2].min()) * 1e3
            print(f"{label:<18} {net:>9.1f} {path:>9.1f} {z_range:>11.1f}")


if __name__ == "__main__":
    main()
