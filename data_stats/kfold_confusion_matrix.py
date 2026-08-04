"""Fold x reach-distance success-rate matrix for the 5-fold LOO cross-validation.

Each of the 50 held-out demos belongs to exactly one (fold, distance) cell -- the split deals 2
demos per distance to each of the 5 folds (make_kfold.py, seed=0), so every cell holds 2 demos.
This renders the 5x5 matrix of mean HELD-OUT success rate, with a marginal 'Mean' row (per
distance) and column (per fold) and the overall CV number in the corner.

Success rate per demo is read from the dump's stats.txt (success/total over the first 128
completed episodes -- player.py counts them in arrival order, so it's unbiased). Reading stats.txt
(not results.txt) is deliberate: a demo with 0 successful rollouts has NO results.txt but DOES have
stats.txt with rate 0.0000, so genuine 0% demos are counted as 0 rather than dropped. A demo whose
dump never wrote stats.txt (true not-run) is reported separately and left out of the means.

    python data_stats/kfold_confusion_matrix.py            # writes data_stats/kfold_confusion_matrix.png
    python data_stats/kfold_confusion_matrix.py --no-plot  # print the matrix only
"""
import argparse
import csv
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_kfold import make_folds  # noqa: E402

RATE_RE = re.compile(r"rate:\s*([\d.]+)")


def demo_success_rate(dumps_root, run_prefix, fold, demo):
    """Success rate (0..1) from the newest stats.txt for (fold, demo); None if no stats.txt.
    run_prefix selects the CV (e.g. 'reach_5foldcv_holdout' original, 'reach_5foldcv_v2_holdout' v2)
    so v2 dumps aren't confused with the original-CV dumps that share the demo id."""
    pat = os.path.join(dumps_root, f"dump__*{run_prefix}{fold}_seed0*__demo_{demo}__*", "stats.txt")
    hits = sorted(glob.glob(pat), key=os.path.getmtime)
    if not hits:
        return None
    m = RATE_RE.search(open(hits[-1]).read())
    return float(m.group(1)) if m else None


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="data_stats/reach_demo.csv")
    ap.add_argument("--run-prefix", default="reach_5foldcv_holdout",
                    help="checkpoint/run name prefix to match dump folders (before the fold index); "
                         "use 'reach_5foldcv_v2_holdout' for the v2 pool")
    ap.add_argument("--dumps", default="dumps")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--per", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data_stats/kfold_confusion_matrix.png")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--per-demo", action="store_true",
                    help="annotate each cell with its individual per-demo rates (id + rate, one line "
                         "each) instead of the cell mean; cells are still colored by the mean")
    args = ap.parse_args()

    distance = {r["demo_name"]: r["demo_distance"]
                for r in csv.DictReader(open(args.csv)) if r.get("demo_name")}
    folds, _ = make_folds(args.csv, args.k, args.per, args.seed)
    dists = sorted({distance[d] for f in folds for d in f}, key=int)

    # cell[fold][dist] = list of per-demo rates; cell_demos = parallel list of (demo, rate) for the
    # --per-demo annotation; not-run demos tracked separately.
    cell = {kf: {d: [] for d in dists} for kf in range(args.k)}
    cell_demos = {kf: {d: [] for d in dists} for kf in range(args.k)}
    not_run = []
    for kf, ids in enumerate(folds):
        for demo in ids:
            r = demo_success_rate(args.dumps, args.run_prefix, kf, demo)
            if r is None:
                not_run.append((kf, distance[demo], demo))
                cell_demos[kf][distance[demo]].append((demo, None))
            else:
                cell[kf][distance[demo]].append(r)
                cell_demos[kf][distance[demo]].append((demo, r))

    # matrix of cell means
    M = [[mean(cell[kf][d]) for d in dists] for kf in range(args.k)]
    fold_mean = [mean([r for d in dists for r in cell[kf][d]]) for kf in range(args.k)]
    dist_mean = [mean([r for kf in range(args.k) for r in cell[kf][d]]) for d in dists]
    overall = mean([r for kf in range(args.k) for d in dists for r in cell[kf][d]])

    # ---- text report ----
    hdr = "        " + "".join(f"  dist {d} " for d in dists) + "  | Fold Mean"
    print(hdr)
    print("-" * len(hdr))
    for kf in range(args.k):
        row = "".join(f"  {M[kf][i]:6.3f} " for i in range(len(dists)))
        print(f"fold {kf} {row}  |  {fold_mean[kf]:6.3f}")
    print("-" * len(hdr))
    dm = "".join(f"  {v:6.3f} " for v in dist_mean)
    print(f"D Mean  {dm}  |  {overall:6.3f}  (overall CV)")
    if not_run:
        print(f"\nNOT RUN ({len(not_run)}, no stats.txt, excluded from means): "
              + ", ".join(f"f{f}/d{d}/{demo}" for f, d, demo in not_run))

    if args.no_plot:
        return

    # ---- heatmap: single-hue sequential (light->dark = low->high success) with margins ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    # single-hue teal ramp, light -> dark (sequential; validated monotone lightness)
    cmap = LinearSegmentedColormap.from_list("teal_seq", ["#f2fafb", "#a8dde0", "#4ba3ad", "#1f6f78", "#0b3a40"])

    nrows, ncols = args.k + 1, len(dists) + 1          # +1 for marginal Mean row/col
    grid = np.full((nrows, ncols), np.nan)
    for kf in range(args.k):
        for i in range(len(dists)):
            grid[kf, i] = M[kf][i]
        grid[kf, -1] = fold_mean[kf]
    for i in range(len(dists)):
        grid[-1, i] = dist_mean[i]
    grid[-1, -1] = overall

    cell_w = 1.7 if args.per_demo else 1.15   # per-demo labels need wider cells to fit "id rate" lines
    fig, ax = plt.subplots(figsize=(cell_w * ncols + 1.6, 1.05 * nrows + 1.2))
    im = ax.imshow(grid, cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal")

    xlabels = [f"Dist {d}" for d in dists] + ["Fold\nMean"]
    ylabels = [f"Fold {kf}" for kf in range(args.k)] + ["Dist Mean"]
    ax.set_xticks(range(ncols)); ax.set_xticklabels(xlabels)
    ax.set_yticks(range(nrows)); ax.set_yticklabels(ylabels)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # annotate every cell; text ink flips to white on dark fills for contrast. In --per-demo mode the
    # inner (fold x dist) cells list each demo's own rate; the marginal Mean row/col still show means.
    for r in range(nrows):
        for c in range(ncols):
            v = grid[r, c]
            if np.isnan(v):
                continue
            ink = "white" if v >= 0.55 else "#0b3a40"
            is_margin = (r == nrows - 1 or c == ncols - 1)
            if args.per_demo and not is_margin:
                lines = [f"{d.replace('m_', '')} {'  n/a' if rr is None else f'{rr:.2f}'}"
                         for d, rr in cell_demos[r][dists[c]]]
                ax.text(c, r, "\n".join(lines), ha="center", va="center", color=ink, fontsize=9)
            else:
                weight = "bold" if is_margin else "normal"
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", color=ink, fontsize=11, fontweight=weight)

    # 2px surface gaps between cells; a heavier gap before the marginal row/col
    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.axvline(ncols - 1.5, color="white", linewidth=5)
    ax.axhline(nrows - 1.5, color="white", linewidth=5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Held-Out Success Rate", rotation=270, labelpad=15)
    cbar.outline.set_visible(False)

    ax.set_title("5-Fold CV Held-Out Success Rate\nby Fold and Reach Distance", pad=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
