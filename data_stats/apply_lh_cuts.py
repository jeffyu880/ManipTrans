"""Trim the terminal left-hand 'retract' motion off a set of MyDataset capping demos.

For each demo it truncates the demo to the first CUTS[demo] frames (keep [0, cut-1]) across
ALL three files that must stay frame-synced:
    data/my_dataset/<stem>.pkl                                  (raw capture, both hands+objs)
    data/retargeting/my_dataset/mano2inspire_lh/<stem>_lh.pkl   (LH retargeted)
    data/retargeting/my_dataset/mano2inspire_rh/<stem>_rh.pkl   (RH retargeted)

The cut frames were derived from the raw LH-wrist speed burst (see plot_lh_cut_analysis.py)
and visually confirmed. For each file:
  * the full original is preserved as <name>_original.pkl  (created once, never overwritten)
  * the truncated demo is written back to the ORIGINAL <name>.pkl
Re-running always trims FROM the *_original backup, so it is idempotent and re-adjustable.

Any field whose first dimension == the file's frame count T is sliced to [:cut]; everything
else (meta, finger_names, scalars, calibration transforms) is left untouched. meta.n_frames
is updated on the raw file.

Run:
    python data_stats/apply_lh_cuts.py            # apply
    python data_stats/apply_lh_cuts.py --dry-run  # show what would change, write nothing
"""
import os, glob, pickle, shutil, argparse
import numpy as np

# demo -> cut_start (== kept length; keep frames [0, cut_start-1])
CUTS = {
    "m_161551": 341,
    "m_170401": 241,
    "m_170527": 206,
    "m_170654": 228,
    "m_170753": 175,
}

RAW_DIR = "data/my_dataset"
LH_DIR = "data/retargeting/my_dataset/mano2inspire_lh"
RH_DIR = "data/retargeting/my_dataset/mano2inspire_rh"


def find_one(directory, demo, suffix):
    """Resolve the single base pkl for a demo, excluding *_original backups."""
    hits = [p for p in glob.glob(os.path.join(directory, f"*{demo}{suffix}"))
            if not p.endswith(f"_original{suffix}")]
    assert len(hits) == 1, f"expected 1 match for *{demo}{suffix} in {directory}, got {hits}"
    return hits[0]


def truncate(obj, T, n):
    """Recursively slice any ndarray/list whose first dim == T down to n; leave the rest."""
    if isinstance(obj, dict):
        return {k: truncate(v, T, n) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj[:n] if obj.ndim >= 1 and obj.shape[0] == T else obj
    if isinstance(obj, (list, tuple)):
        return type(obj)(obj[:n]) if len(obj) == T else obj
    return obj


def frame_count(data):
    """Frame count T for a file: meta.n_frames if present, else first opt_* array len."""
    if isinstance(data, dict) and "meta" in data and "n_frames" in data["meta"]:
        return int(data["meta"]["n_frames"])
    return int(np.asarray(data["opt_wrist_pos"]).shape[0])


def process_file(path, cut, dry_run):
    root, ext = os.path.splitext(path)
    backup = root + "_original" + ext
    # 1) preserve full original exactly once
    if not os.path.exists(backup):
        if not dry_run:
            shutil.copy2(path, backup)
        made = "  (backup created)"
    else:
        made = "  (backup already existed)"
    # 2) always trim FROM the full backup (or current file if dry-run before backup)
    src = backup if os.path.exists(backup) else path
    data = pickle.load(open(src, "rb"))
    T = frame_count(data)
    assert cut <= T, f"cut {cut} > T {T} for {path}"
    trimmed = truncate(data, T, cut)
    if isinstance(trimmed, dict) and "meta" in trimmed and "n_frames" in trimmed["meta"]:
        trimmed["meta"]["n_frames"] = cut
    if not dry_run:
        pickle.dump(trimmed, open(path, "wb"))
    print(f"    {os.path.basename(path):<40} T={T} -> {cut}{made}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for demo, cut in CUTS.items():
        raw = find_one(RAW_DIR, demo, ".pkl")
        lh = find_one(LH_DIR, demo, "_lh.pkl")
        rh = find_one(RH_DIR, demo, "_rh.pkl")
        print(f"[{demo}]  keep [0,{cut-1}]  ({cut} frames)")
        for p in (raw, lh, rh):
            process_file(p, cut, args.dry_run)
    print("\nDONE" + (" (dry-run, nothing written)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
