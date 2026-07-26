"""Annotate data_stats/reach_demo.csv with a per-demo `grasp_start` frame index.

`grasp_start` is the frame where the right hand first grasps the cap, so training/eval can begin
at the grasp instead of the reach phase. It is a 0-based index into the RAW capture, which for
mydataset (skip=1, maxDemoLength defaults to 1200 >> demo length) equals the loaded-buffer index
the env resets on -- feed it as `evalStartFrame` (bimanual, with randomStateInit=false) or
`rolloutBegin`.

Grasp rule -- FIRST SUSTAINED FINGERTIP CONTACT: the first frame where the Thumb_TIP is within
THUMB_CM of the cap surface AND (Index_TIP OR Middle_TIP within SECOND_CM), held for SUSTAIN
frames. This marks the grasp onset (fingers secure the cap, about to lift).

Distances are computed in the RAW capture frame with a plain scipy cdist (the standalone pattern
from maniptrans_envs/lib/envs/live/live_target_source.py:319 `_tips_distance`). This is exact:
the loader only ever applies rigid transforms (recenter / table_rot / mujoco2gym) shared by hand
and object, and the fingertips get no per-hand offset, so fingertip->cap distances -- and thus
the grasp frame -- are identical in the raw and gym frames. Pure pickle/numpy/trimesh/scipy: no
isaacgym, no torch, no GPU.

The CSV edit is additive: `grasp_start` is appended as the LAST column; `demo_distance` and
`demo_name` values and row ORDER are preserved (the K-fold split in make_kfold.py depends on
row order). Re-running always recomputes from the base columns (idempotent).

    python data_stats/find_grasp_start.py --dry-run   # print the table, write nothing
    python data_stats/find_grasp_start.py             # write grasp_start into reach_demo.csv
"""
import argparse
import csv
import glob
import os
import pickle

import numpy as np
import trimesh
from scipy.spatial.distance import cdist

CSV_PATH = "data_stats/reach_demo.csv"
DATA_DIR = "data/my_dataset"
CAP_MESH = "data/OakInk-v2/object_preview/align_ds/O02@0206@00001/scan.ply"  # RH bottle_cap

# Grasp-onset thresholds (cm) and how many consecutive frames must satisfy them.
THUMB_CM = 2.0
SECOND_CM = 3.0
SUSTAIN = 5


def resolve_pkl(demo_id):
    """Unique data/my_dataset/*_m_<id>.pkl for a demo id (excludes *_original backups)."""
    key = demo_id[2:] if demo_id.startswith("m_") else demo_id
    hits = [p for p in glob.glob(os.path.join(DATA_DIR, f"*_m_{key}.pkl")) if not p.endswith("_original.pkl")]
    if len(hits) != 1:
        raise FileNotFoundError(f"{demo_id}: expected 1 pkl, got {hits}")
    return hits[0]


def tip_cap_distances_cm(pkl_path, cap_verts):
    """Per-frame nearest distance (cm) from thumb/index/middle tips to the cap surface. [T] each."""
    raw = pickle.load(open(pkl_path, "rb"))
    jp = raw["hands"]["right"]["joints_pos"]
    cap = np.asarray(raw["obj_transf"]["bottle_cap"], dtype=np.float64)  # [T,4,4]
    out = {}
    for name, key in (("thumb", "Thumb_TIP"), ("index", "Index_TIP"), ("middle", "Middle_TIP")):
        tip = np.asarray(jp[key], dtype=np.float64)                      # [T,3]
        d = np.empty(len(cap))
        for t in range(len(cap)):
            verts = cap_verts @ cap[t, :3, :3].T + cap[t, :3, 3]         # [N,3] cap surface, frame t
            d[t] = cdist(tip[t][None], verts).min()
        out[name] = d * 100.0
    out["T"] = len(cap)
    return out


def grasp_start(d):
    """First frame of a SUSTAIN-long window where thumb<THUMB_CM and (index<SECOND_CM or middle<SECOND_CM)."""
    contact = (d["thumb"] < THUMB_CM) & ((d["index"] < SECOND_CM) | (d["middle"] < SECOND_CM))
    for t in range(len(contact) - SUSTAIN + 1):
        if contact[t: t + SUSTAIN].all():
            return t
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print the table, do not write the CSV")
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo)
    cap_verts = np.asarray(trimesh.load(CAP_MESH, process=False).vertices, dtype=np.float64)

    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        base_fields = [c for c in reader.fieldnames if c != "grasp_start"]  # drop any prior column
        rows = list(reader)

    print(f"{'dist':>4} {'demo':<11}{'T':>5}{'grasp_start':>12}{'phase':>7}")
    failed = []
    for r in rows:
        d = tip_cap_distances_cm(resolve_pkl(r["demo_name"]), cap_verts)
        gs = grasp_start(d)
        if gs is None:
            failed.append(r["demo_name"])
            r["grasp_start"] = ""
            phase = "-"
        else:
            assert 0 <= gs < d["T"], f"{r['demo_name']}: grasp_start {gs} out of [0,{d['T']})"
            r["grasp_start"] = gs
            phase = f"{gs / d['T']:.0%}"
        print(f"{r['demo_distance']:>4} {r['demo_name']:<11}{d['T']:>5}"
              f"{('NONE' if gs is None else gs):>12}{phase:>7}")

    if failed:
        print(f"\nWARNING: no grasp detected for {len(failed)}: {failed}")

    if args.dry_run:
        print("\n[dry-run] nothing written")
        return

    fields = base_fields + ["grasp_start"]
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows with grasp_start to {CSV_PATH}")


if __name__ == "__main__":
    main()
