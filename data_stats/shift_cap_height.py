"""Raise the cap trajectory of the 0721 cap_1/cap_2 demos until it stops clipping the table.

The 0721 session's cap_1..cap_5 groups are five cap start positions marching along the
table's Y axis toward the burner. The tracked floor is tilted ~3.7 deg relative to the sim
tabletop, so the further from the burner the cap starts, the deeper it sinks into the table
(cap_1 rests ~2.3 cm under the surface, cap_5 ~1.1 cm).

This lifts each cap_1/cap_2 demo by a per-demo constant chosen so the LOWEST cap-mesh vertex
over the whole trajectory sits exactly CLEARANCE above the table surface -- i.e. the mesh
stops intersecting the tabletop. ONLY the cap (obj_transf['bottle_cap']) is moved; the burner
body, both hands and every other field are left exactly as captured.

    NOTE: cap_3/4/5 are untouched and still clip by 1.1-2.2 cm, so after this cap_1/cap_2 rest
    ~1.2 cm HIGHER than the other groups rather than level with them.

    NOTE: because the hands are untouched, the right hand's grasp of the cap shifts by the
    offset. The offset is applied to the WHOLE trajectory, so the final capped pose rises by
    the same amount; the cap ends that far above where cap_3/4/5 seat it on the burner.

The offset is applied in the RAW capture frame, where +Y maps to gym +Z with unit scale (see
the RECENTER_FINE comment in my_dataset_RH.py), so a raw +Y shift is exactly a gym +Z shift.

Clearance is measured by replicating the loader's raw->gym chain, against the FULL cap mesh --
not the subsampled `obj_verts` cloud the sim prints, which reads a few tenths of a mm shallower.

For each demo:
  * the untouched capture is preserved as <stem>_original.pkl (created once, never overwritten)
  * the lifted demo is written back to <stem>.pkl
Re-running always re-derives from the *_original backup, so it is idempotent and re-tunable:
the offset is recomputed from the pristine capture rather than stacked on the previous one.

    python data_stats/shift_cap_height.py --dry-run      # show offsets, write nothing
    python data_stats/shift_cap_height.py                # lift until the mesh just touches
    python data_stats/shift_cap_height.py --clearance 0.002   # leave a 2 mm gap instead

The retargeted pkls under data/retargeting/my_dataset/mano2inspire_rh/ are derived from these
captures (mano2dexhand's fit reads object contact forces) and go stale on write -- re-run
retargeting for the shifted demos afterwards. The _lh ones are unaffected: the burner body,
which is the left hand's object, does not move.
"""
import argparse
import glob
import os
import pickle
import shutil

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

RAW_DIR = "data/my_dataset"
CAP_OBJ = "bottle_cap"      # the right hand's object; the one being moved
ANCHOR_OBJ = "bottle_body"  # burner body: the loader's recenter anchor, never touched
CAP_MESH = "data/OakInk-v2/object_preview/align_ds/O02@0206@00001/scan.ply"

# Mirrors my_dataset_RH.py. The offset is a difference of two clearances measured the same
# way, so these cancel out of it and only affect the absolute clearances reported below.
RECENTER_FINE = np.array([0.0, 0.05, 0.0])
TABLE_Z_ROT_DEG = 90.0
TABLE_Z = 0.4 + 0.015

SHIFT_GROUPS = ["cap_1", "cap_2"]
SESSION = "0721"


def mujoco2gym():
    m = np.eye(4)
    m[:3, :3] = R.from_rotvec([0, 0, -np.pi / 2]).as_matrix() @ R.from_rotvec([np.pi / 2, 0, 0]).as_matrix()
    m[:3, 3] = [0, 0, TABLE_Z]
    return m


def cap_clearance(raw, cap_verts):
    """[T] clearance of the lowest cap vertex above the table surface, in the gym frame."""
    traj = np.asarray(raw["obj_transf"][CAP_OBJ], dtype=np.float64).copy()
    anchor0 = np.asarray(raw["obj_transf"][ANCHOR_OBJ][0][:3, 3], dtype=np.float64)
    traj[:, :3, 3] += RECENTER_FINE - anchor0
    table_rot = R.from_rotvec([0.0, np.deg2rad(TABLE_Z_ROT_DEG), 0.0]).as_matrix()
    traj[:, :3, 3] = (table_rot @ traj[:, :3, 3].T).T
    traj[:, :3, :3] = table_rot @ traj[:, :3, :3]
    traj = mujoco2gym()[None] @ traj
    world = cap_verts @ traj[:, :3, :3].transpose(0, 2, 1) + traj[:, :3, 3][:, None]
    return world.min(axis=1)[:, 2] - TABLE_Z


def source_path(stem):
    """Always measure and derive from the pristine capture, never from a previous run."""
    backup = os.path.join(RAW_DIR, f"{stem}_original.pkl")
    return backup if os.path.exists(backup) else os.path.join(RAW_DIR, f"{stem}.pkl")


def stems(group):
    paths = sorted(glob.glob(os.path.join(RAW_DIR, f"{group}_{SESSION}_m_*.pkl")))
    return [os.path.basename(p)[:-4] for p in paths if not p.endswith("_original.pkl")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print offsets, write nothing")
    ap.add_argument("--clearance", type=float, default=0.0,
                    help="metres to leave between the lowest cap vertex and the tabletop (default 0)")
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo)
    cap_verts = np.asarray(trimesh.load(CAP_MESH, process=False).vertices, dtype=np.float64)

    for group in SHIFT_GROUPS:
        print(f"{group}:")
        for stem in stems(group):
            raw = pickle.load(open(source_path(stem), "rb"))
            before = float(cap_clearance(raw, cap_verts).min())
            offset = args.clearance - before
            if offset <= 0:
                print(f"  {stem}: min clearance {before:+.4f} m, already clear -- SKIPPED")
                continue

            traj = np.asarray(raw["obj_transf"][CAP_OBJ]).copy()
            traj[:, 1, 3] += offset  # raw +Y == gym +Z
            raw["obj_transf"][CAP_OBJ] = traj.astype(np.asarray(raw["obj_transf"][CAP_OBJ]).dtype)
            raw["cap_height_offset"] = offset  # provenance; the loader ignores unknown keys
            after = float(cap_clearance(raw, cap_verts).min())

            print(f"  {stem}: min clearance {before:+.4f} -> {after:+.4f} m  "
                  f"(offset {offset * 1000:+.2f} mm)")
            if args.dry_run:
                continue
            dst = os.path.join(RAW_DIR, f"{stem}.pkl")
            backup = os.path.join(RAW_DIR, f"{stem}_original.pkl")
            if not os.path.exists(backup):
                shutil.copy2(dst, backup)
            with open(dst, "wb") as f:
                pickle.dump(raw, f)

    if args.dry_run:
        print("\n[dry-run] nothing written")
    else:
        print("\nWritten. RH retargeted pkls for these demos are now stale -- re-run retargeting.")


if __name__ == "__main__":
    main()
