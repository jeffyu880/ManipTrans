"""Measure the constant gap between a demo's wrist TARGET and the retargeted wrist pose.

The reward pulls the dexhand's base link toward `wrist_pos` / `wrist_rot` from the demo loader, while
the retargeting (mano2dexhand.py) independently solves for the base-link pose that reproduces the
demo's MANO joints. If those two disagree, the wrist target is unreachable: driving the base link
onto it would move every fingertip off the pose the fingers are scored against. This measures the
disagreement, and reports it in the WRIST's own frame -- a body-fixed offset is constant there while
a tracking failure is not, so the two are easy to tell apart.

Why it exists: for the inspire hand `relative_translation` is the base-class default of zeros
(maniptrans_envs/lib/envs/dexhands/base.py:16; only allegro.py overrides it), so my_dataset_RH.py's
`wrist_pos += relative_translation` is a no-op and the target stays at the raw human wrist point.

    python data_stats/check_wrist_frame_offset.py cup_brush_lower_0808_m_153956
    python data_stats/check_wrist_frame_offset.py --dexhand inspire cup_brush_lower_0808_m_154027

Cross-check: the numbers here must match `{lh,rh}_diff_eef_{pos_dist,rot_angle}` at step 0 of an
evalThresholdDryRun recording, which measures the same gap inside the sim by a different route. They
did when this was written (~41-43 mm offline vs 40-46 mm in sim). If they ever diverge, the loader
replication below has drifted from the real loader -- trust the sim.

Runs without Isaac Gym or a GPU: object_sets.py is stdlib-only and is loaded by path so that
importing main.dataset (which registers mano2dexhand and pulls in isaacgym) is never triggered.
"""
import argparse
import importlib.util
import pickle
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

# Mirrors dexhandmanip_bih.py:388-392 — table centre z=0.4, half-height 0.015.
TABLE_SURFACE_Z = 0.415


def load_object_sets():
    """Import main/dataset/object_sets.py by path, bypassing the isaacgym-triggering package init.

    Returns:
        The imported object_sets module.
    """
    spec = importlib.util.spec_from_file_location("object_sets", "main/dataset/object_sets.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["object_sets"] = module  # dataclasses need the module resolvable by name
    spec.loader.exec_module(module)
    return module


def mujoco2gym():
    """Rebuild the mujoco->gym frame transform the env hands to every loader.

    Returns:
        (4, 4) float array, matching dexhandmanip_bih.py:399-404.
    """
    transf = np.eye(4)
    transf[:3, :3] = (
        R.from_rotvec([0, 0, -np.pi / 2]).as_matrix() @ R.from_rotvec([np.pi / 2, 0, 0]).as_matrix()
    )
    transf[:3, 3] = np.array([0, 0, TABLE_SURFACE_Z])
    return transf


def wrist_target(raw, side_key, object_sets, relative_rotation, lh_avp_fix):
    """Rebuild the wrist pose target the reward scores against, straight from the capture.

    Replicates my_dataset_{RH,LH}.py (relative_translation/-rotation, recentring onto the table,
    the table-axis rotation) followed by base.py's process_data frame transform.

    Args:
        raw: the unpickled capture dict.
        side_key: "right" or "left", the key into raw["hands"].
        object_sets: the module returned by load_object_sets.
        relative_rotation: (3, 3) the dexhand's MANO->dex wrist rotation.
        lh_avp_fix: (3, 3) extra correction the LH loader applies, identity for RH.

    Returns:
        (T, 3) target positions and (T, 3, 3) target rotations, in the gym frame.
    """
    names = object_sets.infer_object_set(raw["obj_id"]).resolve_names(raw["obj_id"])
    objset = object_sets.infer_object_set(raw["obj_id"])
    anchor0 = np.asarray(raw["obj_transf"][names["anchor"]][0][:3, 3])
    recenter = np.asarray(objset.recenter_fine) - anchor0
    table_rot = R.from_rotvec([0.0, np.deg2rad(object_sets.TABLE_Z_ROT_DEG), 0.0]).as_matrix()
    transf = mujoco2gym()

    # relative_translation is deliberately absent: it is zeros for inspire, which is the whole point
    # of this script. Add it here once it is set, and note the loader adds it in WORLD frame.
    pos = np.asarray(raw["hands"][side_key]["wrist_pos"]) + recenter
    pos = (table_rot @ pos.T).T
    pos = (transf[:3, :3] @ pos.T).T + transf[:3, 3]

    rot = R.from_quat(raw["hands"][side_key]["wrist_quat"]).as_matrix() @ relative_rotation
    rot = transf[:3, :3] @ (table_rot @ (rot @ lh_avp_fix))
    return pos, rot


def report(stem, side, object_sets, relative_rotation, lh_avp_fix, dexhand):
    """Print the position and rotation gap for one demo and hand, in world and wrist frames.

    Args:
        stem: capture filename stem, e.g. "cup_brush_lower_0808_m_153956".
        side: "rh" or "lh".
        object_sets: the module returned by load_object_sets.
        relative_rotation: (3, 3) the dexhand's MANO->dex wrist rotation for this side.
        lh_avp_fix: (3, 3) extra LH correction, identity for RH.
        dexhand: dexhand name, picking the retargeting subdirectory.

    Returns:
        (wrist_frame_offset_mm, geodesic_deg) as (3,) and scalar means, for the caller to aggregate.
    """
    raw = pickle.load(open(f"data/my_dataset/{stem}.pkl", "rb"))
    side_key = "right" if side == "rh" else "left"
    tgt_pos, tgt_rot = wrist_target(raw, side_key, object_sets, relative_rotation, lh_avp_fix)

    opt = pickle.load(open(f"data/retargeting/my_dataset/mano2{dexhand}_{side}/{stem}_{side}.pkl", "rb"))
    solved_pos = np.asarray(opt["opt_wrist_pos"])
    solved_rot = R.from_rotvec(np.asarray(opt["opt_wrist_rot"])).as_matrix()

    gap_world = solved_pos - tgt_pos
    # into the retargeted wrist's own frame: constant here == rigid body-fixed offset
    gap_local = np.einsum("nij,nj->ni", solved_rot.transpose(0, 2, 1), gap_world)
    dist = np.linalg.norm(gap_world, axis=-1)

    rel = np.einsum("nij,njk->nik", tgt_rot.transpose(0, 2, 1), solved_rot)
    rotvec_deg = np.degrees(R.from_matrix(rel).as_rotvec())
    geodesic = np.linalg.norm(rotvec_deg, axis=-1)

    print(f"  {side.upper()}  position gap  {dist.mean() * 1000:6.1f} mm (std {dist.std() * 1000:4.1f})")
    print(f"        world frame  mean {np.round(gap_world.mean(0) * 1000, 1)} mm  "
          f"std {np.round(gap_world.std(0) * 1000, 1)}")
    print(f"        WRIST frame  mean {np.round(gap_local.mean(0) * 1000, 1)} mm  "
          f"std {np.round(gap_local.std(0) * 1000, 1)}   <- constant => body-fixed offset")
    print(f"  {side.upper()}  rotation gap  {geodesic.mean():6.1f} deg (std {geodesic.std():4.1f}), "
          f"as a fixed rotvec {np.round(rotvec_deg.mean(0), 1)} deg")
    return gap_local.mean(0), geodesic.mean()


def main():
    """Entry point: report the wrist target vs retargeted gap for each capture given."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("stems", nargs="+", help="capture stems under data/my_dataset/, without .pkl")
    parser.add_argument("--dexhand", default="inspire", help="retargeting subdir to read (default: inspire)")
    args = parser.parse_args()

    object_sets = load_object_sets()

    def aa(vec):
        """Axis-angle vector to rotation matrix, matching the dexhand configs' aa_to_rotmat."""
        return R.from_rotvec(vec).as_matrix()

    # Copied from maniptrans_envs/lib/envs/dexhands/inspire.py:141-146 (RH) and :166-170 (LH), and
    # my_dataset_LH.py:118 for the AVP left-hand correction. Kept as literals so this script stays
    # importable without isaacgym; if those change, change them here.
    relative_rotation = {
        "rh": aa([-np.pi / 36, 0, 0]) @ aa([0, 0, np.pi / 36]) @ aa([0, 0, -np.pi / 2]) @ aa([0, np.pi, 0]),
        "lh": aa([-np.pi / 36, 0, 0]) @ aa([0, 0, -np.pi / 36]) @ aa([0, 0, np.pi / 2]),
    }
    lh_avp_fix = {"rh": np.eye(3), "lh": aa([0.0, np.pi, 0.0])}

    for stem in args.stems:
        print(f"\n=== {stem} ({args.dexhand}) ===")
        for side in ("rh", "lh"):
            report(stem, side, object_sets, relative_rotation[side], lh_avp_fix[side], args.dexhand)


if __name__ == "__main__":
    main()
