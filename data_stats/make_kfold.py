"""Deterministic K-fold split of the reach demos for cross-validation.

Single source of truth for the fold assignment used by BOTH sides of the CV:
  - training   (slurm/scitas/train_kfold_array.run): the 40 demos NOT in the held-out fold
  - validation (eval_kfold.sh):                       the 10 demos IN the held-out fold
Both call this script so the split can never drift between them.

Recipe (matches the numbers printed by --summary): group data_stats/reach_demo.csv by
`demo_distance`; for each distance in ascending order take the first K*per ids in CSV order,
shuffle them with a single seeded RNG (advanced once per distance, fixed order), then deal
`per` demos to each of the K folds. Every fold gets exactly `per` demos per distance; folds
are disjoint; ids beyond the first K*per per distance are dropped (reported by --summary).
The RNG is created once and consumed in a fixed order, so the whole split is reproducible
from (csv, k, per, seed) alone -- every array task recomputes the identical assignment.

Pure stdlib (csv + random): no torch / isaacgym / main.dataset import, so it runs instantly
on a login node with no GPU.

    python data_stats/make_kfold.py --fold 0 --which train      # 40 training ids (folds != 0)
    python data_stats/make_kfold.py --fold 0 --which heldout    # 10 held-out ids (fold 0)
    python data_stats/make_kfold.py --summary                   # all folds + dropped extras
"""
import argparse
import csv
import random


# Demos excluded from the pool entirely (never selected into any fold). m_141341 was dropped from
# distance 3 at the user's request; the next distance-3 demo in CSV order (m_133711) takes its
# place. Since a fixed-length Fisher-Yates shuffle advances the RNG by the same number of draws
# regardless of the list's contents, and each distance's pool stays at K*per after the swap, only
# distance 3's membership changes -- distances 1/2/4/5 are byte-for-byte unaffected.
EXCLUDE = {"m_141341"}


def make_folds(csv_path, k, per, seed):
    """Return (folds, dropped): folds is a list of k id-lists; dropped maps distance -> extras."""
    rows = [r for r in csv.DictReader(open(csv_path)) if r.get("demo_name")]
    by_dist = {}
    for r in rows:
        if r["demo_name"] in EXCLUDE:
            continue
        by_dist.setdefault(r["demo_distance"], []).append(r["demo_name"])

    folds = [[] for _ in range(k)]
    dropped = {}
    rng = random.Random(seed)  # created ONCE; advanced once per distance in ascending order
    for dist in sorted(by_dist, key=int):
        ids = by_dist[dist][: k * per]        # first k*per in CSV order
        dropped[dist] = by_dist[dist][k * per:]
        shuffled = ids[:]
        rng.shuffle(shuffled)
        for i in range(k):
            folds[i] += shuffled[i * per: (i + 1) * per]
    return folds, dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="data_stats/reach_demo.csv")
    ap.add_argument("--k", type=int, default=5, help="number of folds")
    ap.add_argument("--per", type=int, default=2, help="demos per distance per fold")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fold", type=int, help="fold index to emit (0..k-1)")
    ap.add_argument("--which", choices=["heldout", "train"], default="heldout",
                    help="heldout = the fold's own demos; train = all demos in the OTHER folds")
    ap.add_argument("--summary", action="store_true", help="print all folds + dropped extras, then exit")
    args = ap.parse_args()

    folds, dropped = make_folds(args.csv, args.k, args.per, args.seed)

    if args.summary:
        for i, f in enumerate(folds):
            per_dist = {}
            # recover distance labels by re-reading is overkill; just print the flat list
            print(f"FOLD {i} ({len(f)} demos): {','.join(f)}")
        n_dropped = sum(len(v) for v in dropped.values())
        print(f"DROPPED ({n_dropped}): " +
              "; ".join(f"dist {d}: {','.join(v)}" for d, v in sorted(dropped.items(), key=lambda kv: int(kv[0])) if v))
        return

    if args.fold is None or not (0 <= args.fold < args.k):
        ap.error(f"--fold must be in [0,{args.k - 1}] (got {args.fold})")

    if args.which == "heldout":
        out = folds[args.fold]
    else:
        out = [x for i, f in enumerate(folds) if i != args.fold for x in f]
    print(",".join(out))


if __name__ == "__main__":
    main()
