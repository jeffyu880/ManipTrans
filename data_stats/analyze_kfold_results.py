"""Analyze the 5-fold CV held-out results by FOLD and by REACH DISTANCE.

Reads the per-demo results.txt that eval_kfold.sh -> main/rl/eval_score.py wrote into
dumps/dump__<ckpt_stem>__demo_<demo>__<ts>/ and reports:
  * per-fold success rate + tracking errors (er/et/ej/eft),
  * per-reach-distance success rate + tracking errors,
  * overall CV numbers,
  * demos with NO results.txt, listed separately (e.g. the disk-quota casualties at the
    tail of the last fold) -- these are NOT counted as genuine 0% success, since a missing
    file means "not scored", not "policy failed".

Success rate is eval_score.py's honest estimate over the first `num_rollouts_to_save` (128)
completed episodes (player.py counts every finished episode, saved in arrival order -- not
top-k -- so the rate is unbiased). er is in degrees; et/ej/eft in cm.

The demo->fold split comes from make_kfold.py (seed=0), the SAME generator training and eval
used. demo->distance comes from data_stats/reach_demo.csv. The dump glob uses a leading wildcard
(dump__*holdout{k}_seed0*) so it matches the real 'last_<ckpt>' prefixed dirs -- the bug that
made aggregate_kfold.py report 0/50.

    python data_stats/analyze_kfold_results.py                 # defaults (k=5, per=2, seed=0)
    python data_stats/analyze_kfold_results.py --dumps dumps
"""
import argparse
import csv
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_kfold import make_folds  # noqa: E402

PATTERNS = {
    "succ_rate": re.compile(r"bih succ rate:\s*([\d.]+)"),
    "er":        re.compile(r"bih er:\s*([\d.]+)"),
    "et":        re.compile(r"bih et:\s*([\d.]+)"),
    "ej":        re.compile(r"bih ej:\s*([\d.]+)"),
    "eft":       re.compile(r"bih eft:\s*([\d.]+)"),
}
ERR = ("er", "et", "ej", "eft")


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
    """Most recent results.txt for (fold, demo). Leading '*' matches the 'last_' ckpt-stem prefix."""
    pat = os.path.join(dumps_root, f"dump__*holdout{fold}_seed0*__demo_{demo}__*", "results.txt")
    hits = sorted(glob.glob(pat), key=os.path.getmtime)
    return hits[-1] if hits else None


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def fmt_group(label, recs, width=10):
    """One summary line for a set of parsed records (demos that scored)."""
    n = len(recs)
    sr = mean([r["succ_rate"] for r in recs])
    ers = {m: mean([r[m] for r in recs]) for m in ERR}
    return (f"{label:<{width}} n={n:>2}  succ={sr:6.3f}  "
            f"er={ers['er']:6.2f}°  et={ers['et']:5.2f}cm  ej={ers['ej']:5.2f}cm  eft={ers['eft']:5.2f}cm")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="data_stats/reach_demo.csv")
    ap.add_argument("--dumps", default="dumps")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--per", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    distance = {}
    for r in csv.DictReader(open(args.csv)):
        if r.get("demo_name"):
            distance[r["demo_name"]] = r["demo_distance"]

    folds, _ = make_folds(args.csv, args.k, args.per, args.seed)

    # Collect one record per held-out demo: fold, distance, metrics (or None if not scored).
    records = []   # dicts with fold, demo, dist, and metric keys (metrics absent if missing)
    missing = []   # (fold, demo, dist)
    for k, ids in enumerate(folds):
        for demo in ids:
            rf = newest_results(args.dumps, k, demo)
            m = parse_results(rf) if rf else None
            dist = distance.get(demo, "?")
            if m is None:
                missing.append((k, demo, dist))
            else:
                records.append({"fold": k, "demo": demo, "dist": dist, **m})

    print(f"Scored {len(records)}/{len(records) + len(missing)} held-out demos "
          f"({len(missing)} not scored)\n")

    print("=" * 78)
    print("BY FOLD (held-out set of each fold)")
    print("-" * 78)
    for k in range(args.k):
        recs = [r for r in records if r["fold"] == k]
        n_missing = sum(1 for f, _, _ in missing if f == k)
        tag = f"fold {k}"
        suffix = f"   [{n_missing} not scored]" if n_missing else ""
        print(fmt_group(tag, recs) + suffix if recs else f"{tag:<10} (no scored demos){suffix}")

    print("\n" + "=" * 78)
    print("BY REACH DISTANCE")
    print("-" * 78)
    for d in sorted({r["dist"] for r in records}, key=lambda x: int(x) if x.isdigit() else 999):
        recs = [r for r in records if r["dist"] == d]
        n_missing = sum(1 for _, _, dd in missing if dd == d)
        suffix = f"   [{n_missing} not scored]" if n_missing else ""
        print(fmt_group(f"dist {d}", recs) + suffix)

    print("\n" + "=" * 78)
    print(fmt_group("OVERALL", records))

    if missing:
        print("\n" + "-" * 78)
        print(f"NOT SCORED ({len(missing)}) -- no results.txt (disk-quota write failure, NOT a policy 0%):")
        for f, demo, d in missing:
            print(f"  fold {f}  dist {d}  {demo}")

    # Per-demo detail, grouped by distance then fold, for the full breakdown.
    print("\n" + "=" * 78)
    print("PER-DEMO (dist / fold / demo / succ / er / et / ej / eft)")
    print("-" * 78)
    for r in sorted(records, key=lambda r: (int(r["dist"]) if r["dist"].isdigit() else 999, r["fold"], r["demo"])):
        print(f"  d{r['dist']}  f{r['fold']}  {r['demo']:<11} succ={r['succ_rate']:6.3f}  "
              f"er={r['er']:6.2f}  et={r['et']:5.2f}  ej={r['ej']:5.2f}  eft={r['eft']:5.2f}")


if __name__ == "__main__":
    main()
