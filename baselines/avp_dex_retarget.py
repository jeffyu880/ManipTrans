"""Retarget a recorded AVP hand capture onto a dexterous robot hand with dex-retargeting.

This is the PURE-RETARGETING baseline for comparison against ManipTrans: a per-frame
optimisation solve from the human fingertip positions alone — no RL, no physics, no object.
Where ManipTrans learns a policy that must actually hold the object, this only asks "what
robot joint angles put the robot fingertips where the human's were", frame by frame. The gap
between the two is the point of the comparison.

    # what the pkl provides, without needing dex-retargeting installed
    python baselines/avp_dex_retarget.py data/my_dataset/<capture>.pkl --check

    # the actual retarget
    python baselines/avp_dex_retarget.py data/my_dataset/<capture>.pkl \
        --hand right --robot inspire --type position \
        --urdf-dir <dex-urdf>/robots/hands --out out/<capture>_rh.npz

Deliberately standalone — no isaacgym or `main.dataset` imports, so it runs in an env that
has dex-retargeting without dragging in the sim, and `--check` runs anywhere numpy does.

--- Input ---
A MyDataset capture pkl (data_collection_AVP_Optitrack_Synced.py). Hand data sits at
`hands.{left,right}`, 25 AVP joints per frame, already in the OptiTrack frame, in metres:

    joints_mat  (T, 25, 4, 4)   full pose per joint; [:, :, :3, 3] is the position
    joints_pos  {name: (T, 3)}  the same translations, keyed by name
    wrist_mat   (T, 4, 4)       == joints_mat[:, 0], the "Base" joint

`hands.finger_names` names the 25 joints and is the authority for their order; `hands.
avp_sync_ok` (T,) flags frames where the AVP stream was fresh. Stale frames are NaN.

--- The 25 -> 21 mapping ---
dex-retargeting indexes the human hand by the MANO/MediaPipe 21-keypoint layout: wrist, then
four joints per finger (thumb CMC/MCP/IP/TIP, the rest MCP/PIP/DIP/TIP). Its inspire config
asks for `target_link_human_indices: [4, 8, 12, 16, 20]` — the five fingertips in that layout.

AVP's skeleton is the same 21 plus a metacarpal `*_Base` for index/middle/ring/pinky. Dropping
those four lands exactly on MANO-21, with no interpolation and no renaming; the thumb already
matches. MANO21_JOINT_NAMES below lists the 21 wanted joints BY NAME and the indices are
resolved against each pkl's own `finger_names`, so a reordering upstream is handled rather
than silently mis-gathered.

--- Frames ---
The optimiser retargets hand SHAPE, not world pose, so keypoints go in wrist-relative and
canonically oriented: centre on the wrist, rotate by AVP's own measured `wrist_mat`, then relabel
the axes with OPERATOR2AVP. This follows Bunny-VisionPro. Upstream dex-retargeting instead fits a
frame to the wrist/index-MCP/middle-MCP triangle (`single_hand_detector.py`), because monocular
tracking has no wrist orientation available; we do have one, it is measured rather than inferred,
and it carries none of that fit's per-frame SVD jitter. The two differ by ~12.8 deg on the right
and ~179 deg (mirrored) on the left, so this is a real choice of frame, not a rounding detail.

Stripping global pose is correct for the finger solve but leaves the retargeted hand with no
trajectory. AVP's `wrist_mat` is carried into the output untouched so a downstream comparison can
reattach it — that wrist pose is the same quantity ManipTrans tracks, so the two are directly
comparable.
"""

import argparse
import os
import pickle
import sys

import numpy as np

# The 21 AVP joints that correspond to MANO/MediaPipe keypoints 0..20, in that order. Listed
# by name rather than index because the pkl carries its own `finger_names`, which is the real
# authority — see mano21_indices(). The four AVP metacarpals (Index_Base, Middle_Base,
# Ring_Base, Pinky_Base) are absent by design: MANO has no counterpart for them.
MANO21_JOINT_NAMES = [
    "Base",
    "Thumb_CMC", "Thumb_MCP", "Thumb_IP", "Thumb_TIP",
    "Index_MCP", "Index_PIP", "Index_DIP", "Index_TIP",
    "Middle_MCP", "Middle_PIP", "Middle_DIP", "Middle_TIP",
    "Ring_MCP", "Ring_PIP", "Ring_DIP", "Ring_TIP",
    "Pinky_MCP", "Pinky_PIP", "Pinky_DIP", "Pinky_TIP",
]

# Copied from dex_retargeting.constants rather than imported, so --check stays runnable in an
# env without the package. Values are fixed by the MANO convention, not by our data.
OPERATOR2MANO_RIGHT = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float64)
OPERATOR2MANO_LEFT = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]], dtype=np.float64)

# Bunny-VisionPro's axis relabeling, feeding the headset's wrist straight to the optimiser instead
# of fitting a frame to the keypoints. Owned here, not in dexret_controller, because this module is
# the dependency-free one — the reverse import would drag isaacgym into a script that runs without.
# Right == OPERATOR2MANO_RIGHT; left differs by exactly rotY(180), the same mirror
# AVP_LH_WRIST_CORRECTION applies in my_dataset_LH.py. Baking it in stops the mirror being a
# separate step that can be applied on one path and forgotten on the other.
OPERATOR2AVP = {
    "right": OPERATOR2MANO_RIGHT,
    "left": np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64),
}

# DexPilot is Bunny-VisionPro's choice: it adds a thumb-to-finger projection that snaps small gaps
# closed to stabilise grasps. "vector" drops that and matches wrist-to-fingertip only, so the
# fingers track the human instead of being pulled into a grasp prior — use it if the thumb rides
# its yaw limit (which the projection will do whenever the human's grip is tight).
DEFAULT_RETARGETING = "dexpilot"


def mano21_indices(avp_names):
    """Resolve where each MANO-21 keypoint sits in this capture's AVP joint list.

    Args:
        avp_names: list[str] of the capture's 25 AVP joint names, in storage order
            (the pkl's `hands.finger_names`).

    Returns:
        (21,) int array indexing the AVP joint axis in MANO/MediaPipe keypoint order.
    """
    missing = [name for name in MANO21_JOINT_NAMES if name not in avp_names]
    assert not missing, (
        f"AVP joints {missing} are absent from the capture's finger_names ({avp_names}). "
        f"Either the capture used a different hand skeleton, or MANO21_JOINT_NAMES needs "
        f"updating to match it — dex-retargeting indexes fingertips at 4/8/12/16/20, so the "
        f"order of MANO21_JOINT_NAMES is load-bearing, not just its contents."
    )
    return np.array([avp_names.index(name) for name in MANO21_JOINT_NAMES])


def load_avp_hand(pkl_path, hand):
    """Pull one hand out of a MyDataset capture, with a per-frame validity mask.

    Args:
        pkl_path: Path to the capture pkl.
        hand: "right" or "left".

    Returns:
        dict with `kp21` (T, 21, 3) world-frame keypoints in metres, MANO order; `wrist_mat`
        (T, 4, 4) global wrist pose; `ok` (T,) bool where the frame is usable; plus `fps`,
        `obj_transf`, `timestamps_s` and `meta` passed through from the capture.
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    hands = data["hands"]
    idx = mano21_indices(list(hands["finger_names"]))
    side = hands[hand]

    kp25 = np.asarray(side["joints_mat"], dtype=np.float64)[:, :, :3, 3]  # (T, 25, 3), metres

    # Two independent ways a frame can be unusable: the capture flagged the AVP stream stale,
    # or a joint arrived NaN. Neither is recoverable, so they collapse into one mask.
    ok = np.asarray(hands["avp_sync_ok"], dtype=bool) & ~np.isnan(kp25).any(axis=(1, 2))

    return {
        "kp21": kp25[:, idx, :],
        "wrist_mat": np.asarray(side["wrist_mat"], dtype=np.float64),
        "ok": ok,
        "fps": float(data["meta"].get("fps", 60.0)),
        "obj_transf": data.get("obj_transf", {}),
        "timestamps_s": np.asarray(data.get("timestamps_s", [])),
        "meta": data["meta"],
    }


def to_mano_frame(kp21, hand, wrist_mats):
    """Move world-frame keypoints into the wrist-centred frame the optimiser expects.

    Retargeting solves hand SHAPE and discards global pose, so the keypoints go in pose-independent
    — otherwise the same finger configuration would yield different joint angles depending on where
    the hand was pointing. The frame comes from AVP's own measured wrist rotation, relabeled by the
    fixed OPERATOR2AVP permutation; nothing is fitted to the keypoints.

    Args:
        kp21: (T, 21, 3) world-frame keypoints in MANO order.
        hand: "right" or "left", selecting the OPERATOR2AVP relabeling.
        wrist_mats: (T, 4, 4) AVP wrist poses; only the rotation block is used.

    Returns:
        (T, 21, 3) keypoints, wrist at the origin and axes in the optimiser's convention. Frames
        that were NaN on input stay NaN rather than propagating a bad frame.
    """
    operator2avp = OPERATOR2AVP[hand]
    out = np.full_like(kp21, np.nan)
    for t, frame_kp in enumerate(kp21):
        if np.isnan(frame_kp).any():
            continue
        centred = frame_kp - frame_kp[0:1, :]
        out[t] = centred @ wrist_mats[t][:3, :3] @ operator2avp
    return out


RETARGETING_TYPES = ("dexpilot", "vector")


def dex_urdf_dir():
    """Directory the configs' relative `urdf_path` resolves against — dex-urdf's hand models.

    dex-urdf is a separate checkout, not vendored, so this searches the usual places instead of
    hardcoding one machine's layout. Override with DEX_URDF_DIR.

    Returns:
        str absolute path to <dex-urdf>/robots/hands.
    """
    override = os.environ.get("DEX_URDF_DIR")
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [override] if override else [
        os.path.join(os.path.dirname(repo), "third_party", "dex-urdf", "robots", "hands"),
        os.path.join(repo, "third_party", "dex-urdf", "robots", "hands"),
        os.path.expanduser("~/dex-urdf/robots/hands"),
    ]
    for path in candidates:
        if path and os.path.isdir(path):
            return os.path.abspath(path)
    raise SystemExit(
        f"dex-urdf not found (looked in {candidates}). Clone it and point DEX_URDF_DIR at the "
        f"hands directory:\n"
        f"  git clone https://github.com/dexsuite/dex-urdf\n"
        f"  export DEX_URDF_DIR=<dex-urdf>/robots/hands"
    )


def default_config_path(robot, hand, retargeting=DEFAULT_RETARGETING):
    """Path to our dex-urdf retargeting config for one hand and optimiser.

    Args:
        robot: dex-retargeting robot name, e.g. "inspire".
        hand: "right" or "left".
        retargeting: "dexpilot" (inter-finger vectors plus DexPilot's grasp projection) or
            "vector" (wrist-to-fingertip vectors only, no projection).

    Returns:
        str path, which may not exist — only inspire has configs today.
    """
    assert retargeting in RETARGETING_TYPES, (
        f"retargeting must be one of {RETARGETING_TYPES}, got {retargeting!r}"
    )
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs",
                        f"{robot}_{hand}_{retargeting}.yml")


def build_retargeting(robot, retargeting_type, hand, urdf_dir, config_path=None):
    """Construct the dex-retargeting solver for one robot hand.

    Imported lazily so `--check` needs no install and this module stays importable without it.

    Args:
        robot: dex-retargeting RobotName member name, e.g. "inspire".
        retargeting_type: "position", "vector" or "dexpilot".
        hand: "right" or "left".
        urdf_dir: Directory the config's relative `urdf_path` resolves against — the URDFs
            ship in dex-urdf, not in the dex-retargeting wheel.

    Returns:
        A built SeqRetargeting solver.
    """
    try:
        from dex_retargeting.constants import (
            HandType,
            RetargetingType,
            RobotName,
            get_default_config_path,
        )
        from dex_retargeting.retargeting_config import RetargetingConfig
    except ImportError as exc:
        raise SystemExit(
            f"dex-retargeting is not importable ({exc}). Install it and the URDFs:\n"
            "  pip install 'dex-retargeting<0.5'   # 0.5+ needs numpy>=2, which breaks isaacgym\n"
            "  git clone https://github.com/dexsuite/dex-urdf\n"
            "then pass --urdf-dir <dex-urdf>/robots/hands"
        )

    RetargetingConfig.set_default_urdf_dir(urdf_dir)
    if config_path is None:
        config_path = get_default_config_path(
            RobotName[robot], RetargetingType[retargeting_type], HandType[hand]
        )
    assert config_path is not None and os.path.exists(str(config_path)), (
        f"no retargeting config at {config_path} for {robot}/{retargeting_type}/{hand}. Pass "
        f"--config explicitly, or --urdf-dir <dex-urdf>/robots/hands to use dex-retargeting's own."
    )
    print(f"[config] {config_path}")
    print(f"[urdf  ] resolved against {urdf_dir}")
    return RetargetingConfig.load_from_file(config_path).build()


def retarget_ref_value(retargeting, kp21):
    """The reference value a built solver expects, for whichever optimiser it wraps.

    DEXPILOT/VECTOR consume tip-minus-origin vectors and index `target_link_human_indices` as
    (2, n) pairs; POSITION consumes absolute keypoints and indexes it as (n,). Feeding one an array
    shaped for the other mis-gathers silently rather than raising, hence one shared gather.

    Args:
        retargeting: A built SeqRetargeting solver.
        kp21: (21, 3) keypoints already in the optimiser's frame.

    Returns:
        (n, 3) reference value to hand to `retargeting.retarget()`.
    """
    indices = retargeting.optimizer.target_link_human_indices
    if retargeting.optimizer.retargeting_type == "POSITION":
        return kp21[indices, :]
    return kp21[indices[1, :], :] - kp21[indices[0, :], :]


def retarget_sequence(retargeting, mano21_local):
    """Solve robot joint angles for every valid frame of a sequence.

    Args:
        retargeting: A built SeqRetargeting solver.
        mano21_local: (T, 21, 3) wrist-centred MANO-frame keypoints; NaN frames are skipped.

    Returns:
        (T, ndof) solved joint angles, NaN on frames that were skipped.
    """
    qpos = None
    for t, kp in enumerate(mano21_local):
        if np.isnan(kp).any():
            continue
        solved = retargeting.retarget(retarget_ref_value(retargeting, kp))
        if qpos is None:
            qpos = np.full((len(mano21_local), len(solved)), np.nan)
        qpos[t] = solved

    assert qpos is not None, (
        "every frame was NaN or stale, so nothing was retargeted. Check hands.avp_sync_ok in "
        "the capture — a take with no fresh AVP frames cannot be retargeted at all."
    )
    return qpos


def report_input(pkl_path, hand, rec):
    """Print what the capture provides and sanity-check the skeleton before solving.

    Bone lengths are the cheapest end-to-end check on the 25->21 gather and the units: a
    mis-indexed skeleton yields wildly varying "bones", and a unit error moves them by 10^3.

    Args:
        pkl_path: Path the capture was read from, for the header line.
        hand: "right" or "left".
        rec: The dict returned by load_avp_hand.

    Returns:
        None.
    """
    kp21, ok = rec["kp21"], rec["ok"]
    n_frames, n_ok = len(kp21), int(rec["ok"].sum())
    print(f"[input]  {os.path.basename(pkl_path)}  hand={hand}")
    print(f"         {n_frames} frames @ {rec['fps']:.0f} Hz, "
          f"{n_ok} usable ({100 * n_ok / max(n_frames, 1):.1f}%)")
    print(f"         objects: {list(rec['obj_transf'])}")

    valid = kp21[ok]
    if not len(valid):
        return
    for a, b, label in [(5, 6, "index MCP->PIP"), (6, 7, "index PIP->DIP"),
                        (7, 8, "index DIP->TIP"), (0, 9, "wrist->middle MCP")]:
        lengths = np.linalg.norm(valid[:, a] - valid[:, b], axis=-1) * 1e3
        print(f"[bone]   {label:20s} {lengths.mean():6.1f} mm  (sd {lengths.std():.2f})")


def main():
    """Run the offline AVP -> robot-hand retarget for one capture and hand.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("pkl", help="MyDataset capture pkl")
    parser.add_argument("--hand", default="right", choices=["right", "left"])
    parser.add_argument(
        "--robot",
        default="inspire",
        choices=["inspire", "allegro", "shadow", "svh", "leap", "ability", "panda"],
        help="inspire matches the hand ManipTrans retargets to",
    )
    parser.add_argument(
        "--type",
        dest="retargeting_type",
        default=DEFAULT_RETARGETING,
        choices=list(RETARGETING_TYPES),
        help="which shipped config to use: dexpilot = inter-finger vectors plus DexPilot's "
        "thumb-to-finger grasp projection (Bunny-VisionPro's choice); vector = wrist-to-fingertip "
        "vectors only, no projection, milder on the joint limits. Ignored if --config is given.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="retargeting config yml. Defaults to the ManipTrans-URDF config in baselines/configs, "
        "which target dex-urdf's inspire — the model dex-retargeting and Bunny-VisionPro were "
        "built against. Pass one of dex-retargeting's own to use their filtering settings.",
    )
    parser.add_argument(
        "--urdf-dir",
        default=None,
        help="dir the config's relative urdf_path resolves against. Defaults to dex_urdf_dir(), "
        "i.e. <dex-urdf>/robots/hands; override the search with DEX_URDF_DIR.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output path; .pkl or .npz by extension (default: .npz alongside the capture)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report what the capture provides and exit, without running dex-retargeting",
    )
    args = parser.parse_args()

    rec = load_avp_hand(args.pkl, args.hand)
    report_input(args.pkl, args.hand, rec)
    mano21_local = to_mano_frame(rec["kp21"], args.hand, rec["wrist_mat"])

    if args.check:
        print("\n[check] AVP -> MANO-21 conversion OK; rerun without --check to retarget")
        return

    # Default to dex-urdf's inspire — the model dex-retargeting and Bunny-VisionPro were built
    # against, and whose base frame already matches the convention the keypoints arrive in.
    config_path = args.config or default_config_path(
        args.robot, args.hand, args.retargeting_type
    )
    urdf_dir = args.urdf_dir or dex_urdf_dir()
    retargeting = build_retargeting(
        args.robot, args.retargeting_type, args.hand, urdf_dir, config_path
    )
    qpos = retarget_sequence(retargeting, mano21_local)
    joint_names = list(retargeting.optimizer.robot.dof_joint_names)

    out = args.out or os.path.splitext(args.pkl)[0] + f"_{args.robot}_{args.hand}_dexret.npz"
    payload = {
        "qpos": qpos,                    # (T, ndof) solved joint angles, NaN where skipped
        "joint_names": joint_names,
        "mano21_world": rec["kp21"],     # (T, 21, 3) human keypoints, OptiTrack frame
        "mano21_local": mano21_local,    # (T, 21, 3) what the optimiser actually saw
        "wrist_mat": rec["wrist_mat"],   # (T, 4, 4) global pose the finger solve discards
        "valid": rec["ok"],
        "fps": rec["fps"],
        "timestamps_s": rec["timestamps_s"],
        "robot": args.robot,
        "hand": args.hand,
        "retargeting_type": args.retargeting_type,
        "config": str(config_path),
        "source_pkl": os.path.abspath(args.pkl),
    }
    if out.endswith(".pkl"):
        with open(out, "wb") as f:
            pickle.dump(payload, f)
    else:
        np.savez(out, **{k: (np.array(v) if isinstance(v, list) else v) for k, v in payload.items()})

    solved = int(np.isfinite(qpos).all(axis=1).sum())
    print(f"[out]    {qpos.shape[1]} dof, {solved}/{len(qpos)} frames solved -> {out}")
    print(f"[joints] {joint_names}")


if __name__ == "__main__":
    sys.exit(main())
