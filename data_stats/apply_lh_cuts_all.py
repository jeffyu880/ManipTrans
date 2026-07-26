"""Auto-detect and apply the terminal left-hand 'retract' cut to ALL MyDataset cap_* demos.

For every data/my_dataset/cap_*.pkl it:
  1. detects the terminal LH-wrist speed burst (the cut frame) from the FULL data
     (reads <name>_original.pkl when a backup already exists, so already-trimmed demos
     are detected from their original length),
  2. writes a review plot to vis_traj_outputs/lh_cut_analysis/<stem>.png,
  3. with --apply, truncates the raw + LH + RH retargeting pkls to [0, cut-1], preserving
     the full originals as <name>_original.pkl (created once; trims always read from it).

Detection + truncation reuse plot_lh_cut_analysis.py / apply_lh_cuts.py so behaviour matches
the 5 demos already cut.

Run:
    python data_stats/apply_lh_cuts_all.py            # detect + plot only (no writes)
    python data_stats/apply_lh_cuts_all.py --apply    # also truncate the pkls
"""
import os, sys, glob, re, pickle, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_lh_cut_analysis import interp_nan, speed, detect_terminal_cut  # detection helpers
from apply_lh_cuts import process_file, find_one, RAW_DIR, LH_DIR, RH_DIR  # truncation helpers

FPS = 60.0
OUT_DIR = "vis_traj_outputs/lh_cut_analysis"


def full_source(path):
    """Return the *_original backup if it exists (full data), else the path itself."""
    root, ext = os.path.splitext(path)
    orig = root + "_original" + ext
    return orig if os.path.exists(orig) else path


def demo_id(stem):
    m = re.search(r"(m_\d+)", stem)
    return m.group(1) if m else stem


def plot_demo(stem, lw_raw, lobj, v_left, cut, thr, T):
    t = np.arange(T)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    fig.suptitle(f"{stem}  (T={T}, {T/FPS:.2f}s)  LH wrist", fontsize=12)
    for c, lab in zip(range(3), "xyz"):
        ax[0].plot(t, lw_raw[:, c], label=f"wrist {lab}")
    ax[0].plot(t, lobj[:, 2], "k--", alpha=0.4, label="obj z")
    ax[0].set_title("LH wrist position"); ax[0].set_xlabel("frame"); ax[0].set_ylabel("m"); ax[0].legend(fontsize=7)
    ax[1].plot(t, v_left, "purple", label="v_left (smoothed, vel along retract dir)")
    ax[1].axhline(thr, color="orange", ls=":", label=f"thr={thr:.3f}")
    ax[1].axhline(0.0, color="gray", lw=0.6)
    ax[1].set_title("LH wrist velocity toward 'left'/retract dir  (cut = thr crossing)")
    ax[1].set_xlabel("frame"); ax[1].set_ylabel("m/s"); ax[1].legend(fontsize=7)
    if cut is not None:
        for a in ax:
            a.axvspan(cut, T - 1, color="red", alpha=0.12); a.axvline(cut, color="red", ls="--", lw=1.5)
        ax[0].text(cut, ax[0].get_ylim()[1], f" cut@{cut}", color="red", va="top", fontsize=9)
    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{stem}.png")
    plt.savefig(out, dpi=100); plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="truncate the pkls (default: detect+plot only)")
    args = ap.parse_args()

    raws = sorted(p for p in glob.glob(os.path.join(RAW_DIR, "cap_*.pkl")) if not p.endswith("_original.pkl"))
    rows = []
    for raw in raws:
        stem = os.path.splitext(os.path.basename(raw))[0]
        demo = demo_id(stem)
        src = full_source(raw)
        d = pickle.load(open(src, "rb"))
        lw_raw, n_nan_w = interp_nan(d["hands"]["left"]["wrist_pos"])
        lobj, _ = interp_nan(d["obj_transf"]["bottle_body"][:, :3, 3])
        T = len(lw_raw)
        cut, v_left, thr, left_dir = detect_terminal_cut(lw_raw)
        png = plot_demo(stem, lw_raw, lobj, v_left, cut, thr, T)
        rows.append((demo, stem, raw, src, T, cut))
        cut_s = "NONE" if cut is None else str(cut)
        print(f"[{demo}] T={T:>4}  cut={cut_s:>5}  src={'_original' if src!=raw else 'base':<9}  -> {png}")

    print("\n==== proposed cuts ====")
    print(f"{'demo':<10}{'T':>5}{'cut_start':>10}{'kept':>7}{'cut':>6}")
    for demo, stem, raw, src, T, cut in rows:
        kept = T if cut is None else cut
        cf = 0 if cut is None else (T - cut)
        print(f"{demo:<10}{T:>5}{('NONE' if cut is None else cut):>10}{kept:>7}{cf:>6}")

    if not args.apply:
        print("\n(detect+plot only — re-run with --apply to truncate the pkls)")
        return

    print("\n==== applying ====")
    for demo, stem, raw, src, T, cut in rows:
        if cut is None:
            print(f"[{demo}] no terminal burst detected -> SKIP (left full)")
            continue
        try:
            lh = find_one(LH_DIR, demo, "_lh.pkl")
            rh = find_one(RH_DIR, demo, "_rh.pkl")
        except AssertionError as e:
            print(f"[{demo}] retargeting pkl missing -> SKIP ({e})")
            continue
        print(f"[{demo}] keep [0,{cut-1}] ({cut} frames)")
        for p in (raw, lh, rh):
            process_file(p, cut, dry_run=False)
    print("\nDONE")


if __name__ == "__main__":
    main()
