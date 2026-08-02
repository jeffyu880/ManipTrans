"""Translate retargeted hand poses vertically, in place.

Retargeting fits the dexhand to the demo's MANO targets and stores the result in
`data/retargeting/my_dataset/mano2<dexhand>_<side>/<stem>_<side>.pkl`, already in the **gym frame**
(`load_retargeted_data` reads `opt_wrist_pos` verbatim). So when the loader's recentering changes --
e.g. an object set's `recenter_fine` is corrected -- the demo targets and the objects move but the
stored `opt_*` do not, and the retargeted hands are left floating at the old height.

Refitting is the thorough fix; for a pure translation it is also unnecessary, because the fit is
translation-equivariant: shifting every target down by dz shifts the whole solution down by dz, and
the joint angles are unchanged. This applies that shift directly.

Only positions move: `opt_wrist_pos` and `opt_joints_pos` (the +Z column). `opt_wrist_rot` and
`opt_dof_pos` are rotations/joint angles and are invariant under translation.

Dry run by default -- pass --apply to write. Originals are copied to <pkl>.bak unless --no-backup.

    # what the 5 cm recenter_fine correction on cup_brush needs
    python data_stats/shift_retarget_height.py --data_idx m_191123 m_191211 --dz -0.05 --apply
"""

import argparse
import glob
import os
import pickle
import shutil
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)

RETARGET_ROOT = "data/retargeting/my_dataset"
POS_KEYS = ("opt_wrist_pos", "opt_joints_pos")  # translate; everything else is rotation/angles


def resolve(data_idx, dexhand, sides):
    """The retarget pkls for one capture index, per side."""
    key = data_idx[2:] if data_idx.startswith("m_") else data_idx
    out = []
    for side in sides:
        pattern = f"{RETARGET_ROOT}/mano2{dexhand}_{side}/*{key}_{side}.pkl"
        matches = sorted(glob.glob(pattern))
        if not matches:
            print(f"  [skip] no retarget pkl for {data_idx} {side} ({pattern})")
            continue
        assert len(matches) == 1, f"{data_idx} {side} matched {len(matches)} pkls: {matches}"
        out.append((side, matches[0]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_idx", nargs="+", required=True, help="capture indices, e.g. m_191123 m_191211")
    ap.add_argument("--dz", type=float, required=True, help="gym-frame Z shift in metres (negative = down)")
    ap.add_argument("--dexhand", default="inspire")
    ap.add_argument("--side", default="both", choices=["both", "rh", "lh"])
    ap.add_argument("--apply", action="store_true", help="write the change (default: dry run)")
    ap.add_argument("--no-backup", action="store_true", help="do not write <pkl>.bak")
    args = ap.parse_args()

    sides = ["rh", "lh"] if args.side == "both" else [args.side]
    print(f"{'APPLY' if args.apply else 'DRY RUN'}: dz = {args.dz:+.4f} m (gym Z)\n")

    for data_idx in args.data_idx:
        print(data_idx)
        for side, path in resolve(data_idx, args.dexhand, sides):
            d = pickle.load(open(path, "rb"))
            before = float(np.asarray(d["opt_wrist_pos"])[:, 2].mean())
            for k in POS_KEYS:
                if k in d:
                    arr = np.asarray(d[k], dtype=np.float64).copy()
                    arr[..., 2] += args.dz
                    d[k] = arr.astype(np.asarray(d[k]).dtype)
            after = float(np.asarray(d["opt_wrist_pos"])[:, 2].mean())
            moved = [k for k in POS_KEYS if k in d]
            print(f"  {side}: mean wrist z {before:.4f} -> {after:.4f} m   ({', '.join(moved)})")
            print(f"      {path}")
            if args.apply:
                if not args.no_backup:
                    shutil.copy2(path, path + ".bak")
                with open(path, "wb") as f:
                    pickle.dump(d, f)

    if not args.apply:
        print("\nNothing written. Re-run with --apply.")


if __name__ == "__main__":
    main()
