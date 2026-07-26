"""Honest 5-fold CV aggregation of the held-out validation results.

Reads the results.txt that eval_kfold.sh -> main/rl/eval_score.py wrote per held-out demo and
reports per-fold and overall cross-validation metrics.

Two deliberate differences from aggregate_results.py:
  1. A held-out demo with NO results.txt (zero successful rollouts) counts as succ_rate=0 in the
     success-rate mean -- so the CV success rate reflects ALL held-out demos, not just the ones
     that succeeded. Error metrics (er/et/ej/eft) are still averaged only over demos that had
     successes, since they are undefined for a demo with zero successful rollouts.
  2. Dumps are keyed by the EXACT held-out id. aggregate_results.py's folder regex
     `__demo_([^_]+...)` truncates `m_141857` to `m`, which mis-keys every mydataset demo.

The held-out sets come from data_stats/make_kfold.py (seed=0) -- the same generator training and
eval_kfold.sh use -- so this scores exactly the demos each fold held out.

    python data_stats/aggregate_kfold.py                 # all folds, defaults (k=5, per=2, seed=0)
    python data_stats/aggregate_kfold.py --dumps dumps    # explicit dump root
"""
import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_kfold import make_folds  # noqa: E402  (pure stdlib, safe to import)

# eval_score.py writes lines like "bih succ rate: 0.83", "bih er: 12.3 deg", etc.
PATTERNS = {
    "succ_rate": re.compile(r"bih succ rate:\s*([\d.]+)"),
    "er":        re.compile(r"bih er:\s*([\d.]+)"),
    "et":        re.compile(r"bih et:\s*([\d.]+)"),
    "ej":        re.compile(r"bih ej:\s*([\d.]+)"),
    "eft":       re.compile(r"bih eft:\s*([\d.]+)"),
}
ERROR_KEYS = ("er", "et", "ej", "eft")


def parse_results(path):
    text = open(path).read()
    out = {}
    for key, pat in PATTERNS.items():
        m = pat.search(text)
        if m is None:
            return None
        out[key] = float(m.group(1))
    return out


def newest_results(dumps_root, fold, demo):
    """Most recent results.txt for (fold, demo). Keys on the fold's checkpoint-name prefix and the
    exact demo id, so it never confuses folds or truncates mydataset ids."""
    pat = os.path.join(dumps_root, f"dump__reach_5foldcv_holdout{fold}_seed0*__demo_{demo}__*", "results.txt")
    hits = sorted(glob.glob(pat), key=os.path.getmtime)
    return hits[-1] if hits else None


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="data_stats/reach_demo.csv")
    ap.add_argument("--dumps", default="dumps", help="root dir holding the dump__* folders")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--per", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    folds, _ = make_folds(args.csv, args.k, args.per, args.seed)

    overall_succ = []                          # one entry per held-out demo (0 if no successes)
    overall_err = {m: [] for m in ERROR_KEYS}  # only demos that had successes
    fold_lines = []

    for k in range(args.k):
        succ, err = [], {m: [] for m in ERROR_KEYS}
        for demo in folds[k]:
            rf = newest_results(args.dumps, k, demo)
            r = parse_results(rf) if rf else None
            if r is None:
                succ.append(0.0)
                print(f"  fold {k}  {demo}: no results.txt -> succ_rate=0")
            else:
                succ.append(r["succ_rate"])
                for m in ERROR_KEYS:
                    err[m].append(r[m])
                print(f"  fold {k}  {demo}: succ={r['succ_rate']:.4f} "
                      f"er={r['er']:.3f} et={r['et']:.3f} ej={r['ej']:.3f} eft={r['eft']:.3f}")
        overall_succ += succ
        for m in ERROR_KEYS:
            overall_err[m] += err[m]
        n_zero = sum(1 for s in succ if s == 0.0)
        fold_lines.append(
            f"FOLD {k}: succ_rate={mean(succ):.4f} over {len(succ)} held-out "
            f"({n_zero} zero-success)  |  er={mean(err['er']):.3f} et={mean(err['et']):.3f} "
            f"ej={mean(err['ej']):.3f} eft={mean(err['eft']):.3f}  (over {len(err['er'])} scored)"
        )

    print("\n" + "=" * 72)
    for line in fold_lines:
        print(line)
    print("-" * 72)
    n_zero_all = sum(1 for s in overall_succ if s == 0.0)
    print(f"OVERALL CV succ_rate = {mean(overall_succ):.4f} over {len(overall_succ)} held-out demos "
          f"({n_zero_all} zero-success)")
    for m in ERROR_KEYS:
        unit = "deg" if m == "er" else "cm"
        print(f"  {m}: {mean(overall_err[m]):.4f} {unit}  (mean over {len(overall_err[m])} demos with successes)")


if __name__ == "__main__":
    main()
