"""Object sets — which props a capture/stream carries, which hand scores which, and their assets.

A ManipTrans bimanual scene has at most two *scored* bodies (one per hand: `manip_obj_rh`,
`manip_obj_lh`) and, optionally, **props** — bodies that exist to physics but are never a reward or
failure target. Which is which depends on the take:

  * alcohol burner (`bottle`): LH holds the body, RH brings the cap down — two scored objects.
  * cup + brush (`cup_brush`): BOTH hands manipulate the brush, and the cup is a passive receptacle
    the brush is placed into. One scored object (shared between the hands) plus one prop.

This module is the single place that decides that, for both paths:

  * **offline** — `infer_object_set(raw["obj_id"])` picks the set from what the capture recorded;
  * **live** — no pkl to infer from, so `objectSet` names it and `SetObj.match()` maps the Motive
    rigid-body names on the wire onto the same entries.

It also owns the asset lookup and the recentering constants. Those used to be duplicated verbatim in
`my_dataset_LH.py` and `my_dataset_RH.py` behind `!!! KEEP IDENTICAL !!!` comments; the set has to
own the anchor choice anyway, so they live here and both loaders import them.

Deliberately dependency-free (stdlib only) so it can be imported from the loaders, the env and
`LiveTargetSource` without dragging in torch/pytorch3d or tripping the isaacgym import order.

Adding a set: drop the prop's visual mesh in `data/my_dataset/obj_files/obj_vis/<asset_id>.{obj,ply}`
and its COACD decomposition + urdf in `coacd/`, then add one entry to `OBJECT_SETS`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

# --- AVP joints ------------------------------------------------------------------
# AVP joint frame -> ManipTrans mano_joints name. Kept as a fallback; the authoritative mapping is
# also stored in each pkl under meta['avp_to_mano_joints'].
AVP_TO_MANO_JOINTS = {
    "index_proximal": "Index_MCP", "index_intermediate": "Index_PIP",
    "index_distal": "Index_DIP", "index_tip": "Index_TIP",
    "middle_proximal": "Middle_MCP", "middle_intermediate": "Middle_PIP",
    "middle_distal": "Middle_DIP", "middle_tip": "Middle_TIP",
    "ring_proximal": "Ring_MCP", "ring_intermediate": "Ring_PIP",
    "ring_distal": "Ring_DIP", "ring_tip": "Ring_TIP",
    "pinky_proximal": "Pinky_MCP", "pinky_intermediate": "Pinky_PIP",
    "pinky_distal": "Pinky_DIP", "pinky_tip": "Pinky_TIP",
    "thumb_proximal": "Thumb_CMC", "thumb_intermediate": "Thumb_MCP",
    "thumb_distal": "Thumb_IP", "thumb_tip": "Thumb_TIP",
}

# --- Assets ----------------------------------------------------------------------
# The capture pkl stores obj_mesh_path/obj_urdf_path as None, so object assets are resolved by name.
# These two reuse the OakInk-v2 alcohol burner. obj_id -> (mesh for verts/BPS, coacd .urdf for sim).
OBJ_ASSETS = {
    "bottle_body": (  # alcohol burner body (O02@0206@00002)
        "data/OakInk-v2/object_preview/align_ds/O02@0206@00002/scan.ply",
        "data/OakInk-v2/coacd_object_preview/align_ds/O02@0206@00002/scan.urdf",
    ),
    "bottle_cap": (  # alcohol burner cap (O02@0206@00001)
        "data/OakInk-v2/object_preview/align_ds/O02@0206@00001/scan.ply",
        "data/OakInk-v2/coacd_object_preview/align_ds/O02@0206@00001/scan.urdf",
    ),
}

# Props printed for this rig live under the capture tree instead of OakInk's: the visual mesh
# (sampled for obj_verts/BPS) in obj_vis/, and the COACD convex decomposition plus its urdf in
# coacd/. The env loads that urdf with convex_decomposition_from_submeshes, so each COACD piece
# becomes its own convex collision shape.
MY_DATASET_OBJ_DIR = "data/my_dataset/obj_files"


def resolve_obj_assets(obj_id):
    """(verts/BPS mesh, sim urdf) for an obj_id.

    Anything not in OBJ_ASSETS is looked up by name under MY_DATASET_OBJ_DIR, so a new prop only
    needs its two files dropped in — no code change here. Raises with the exact missing paths
    rather than letting a typo'd Motive rigid-body name surface as a KeyError.
    """
    if obj_id in OBJ_ASSETS:
        return OBJ_ASSETS[obj_id]
    # the visual mesh is whichever of .ply/.obj was exported for the prop — trimesh reads both,
    # so a prop only ever needs to exist in one of the two formats
    mesh_candidates = [f"{MY_DATASET_OBJ_DIR}/obj_vis/{obj_id}{ext}" for ext in (".ply", ".obj")]
    mesh = next((p for p in mesh_candidates if os.path.exists(p)), None)
    urdf = f"{MY_DATASET_OBJ_DIR}/coacd/{obj_id}.urdf"
    missing = ([" or ".join(mesh_candidates)] if mesh is None else []) + (
        [] if os.path.exists(urdf) else [urdf]
    )
    assert not missing, (
        f"obj_id '{obj_id}' is not in OBJ_ASSETS and these files are missing: {missing}. "
        f"Either add it to OBJ_ASSETS or name the files after the capture's rigid body."
    )
    return mesh, urdf


# --- Trajectory recentering -------------------------------------------------------
# OptiTrack's world origin is not the sim table, so the raw capture lands off-center and too high.
# We subtract the anchor object's first-frame position so the scene starts at the raw origin, which
# mujoco2gym_transf maps onto the table center. Done in RAW frame so it is identical whether the
# loader applies the transform (train/test) or mano2dexhand applies it (retargeting). RECENTER_FINE
# nudges afterward, in RAW (AVP, Y-up) frame: +Y -> gym +Z (up/height), +X -> gym -Y, +Z -> gym -X.
# Raise +Y if the objects sink into the table.
# The nudge is really "lift the anchor's ORIGIN to where its base rests on the table", so the right
# value depends on where the anchor mesh's origin sits inside its own geometry — which differs per
# set. Each ObjectSet carries its own `recenter_fine`; this is the default (and the burner's).
#   bottle_body: origin is 4.88 cm above its base -> +0.05 leaves the base 0.12 cm above the table.
#   cup:         origin IS its base              -> +0.05 would float the whole scene 5 cm.
RECENTER_ANCHOR_OBJ = "bottle_body"
RECENTER_FINE = (0.0, 0.05, 0.0)  # (x, y, z) metres, raw frame

# Rotate the whole scene (both hands + objects) about the table's vertical axis. Applied in RAW
# frame after recentering: raw +Y maps to gym +Z (the table's up axis), so a rotation about raw Y is
# exactly a rotation about the table's Z. Done in raw frame (before mujoco2gym_transf) so it stays
# consistent for both train/test and retargeting. Flip the sign to reverse direction.
TABLE_Z_ROT_DEG = 90.0

# How far to pull the wrist back from the fingers. 0.25 = 25% of wrist-to-MCP distance toward the
# forearm. Increase if the hand reaches over the object.
#
# 0 for AVP captures: this drives the demo TRACKING TARGETS, so changing it moves every MyDataset
# reward and invalidates comparisons with anything trained before. The dex-retargeting baseline
# applies its own pullback to the retargeted wrist only (--wrist-pullback in
# baselines/dexret2dexhand.py), which touches reset init and playback but not the targets.
WRIST_PULLBACK = 0.0


# --- Sets --------------------------------------------------------------------------
@dataclass(frozen=True)
class SetObj:
    """One prop: what a capture/stream calls it, and what the sim loads for it."""

    asset_id: str  # name under data/my_dataset/obj_files (or a key of OBJ_ASSETS)
    names: Tuple[str, ...]  # rigid-body names that mean this prop, preferred first

    def match(self, published) -> str:
        """The entry of `published` (a capture's obj_id list / a frame's obj_ids) for this prop."""
        by_lower = {str(p).lower(): p for p in published}
        for candidate in self.names:
            if candidate.lower() in by_lower:
                return by_lower[candidate.lower()]
        raise KeyError(
            f"object '{self.asset_id}' expects one of {list(self.names)}, but the capture/stream "
            f"carries {list(published)}. Rename the Motive rigid body, or add its name to `names` "
            f"in main/dataset/object_sets.py."
        )

    def assets(self):
        """(verts/BPS mesh, sim urdf) for this prop."""
        return resolve_obj_assets(self.asset_id)


@dataclass(frozen=True)
class ObjectSet:
    name: str
    rh: SetObj  # the body the RIGHT hand's obs/reward track
    lh: SetObj  # the body the LEFT hand's obs/reward track (== rh when both hands share one body)
    anchor: SetObj  # whose first frame defines the scene origin (see RECENTER_* above)
    prop: Optional[SetObj] = None  # spawned, collides, never scored
    seating_cutoff: bool = True  # does liveResidualCutoff's cap-on-bottle test apply?
    # where the anchor's origin is placed after recentering, raw frame (see RECENTER_FINE). Must
    # match the anchor mesh's origin-to-base distance, or the whole scene floats / sinks.
    recenter_fine: Tuple[float, float, float] = RECENTER_FINE

    @property
    def shares_one_body(self) -> bool:
        """Both hands score the SAME body, so only one scored actor may be spawned."""
        return self.rh.asset_id == self.lh.asset_id

    def side(self, side: str) -> SetObj:
        return self.rh if side == "rh" else self.lh

    def resolve_names(self, published) -> dict:
        """Map this set onto one capture/frame's object names.

        Returns {"rh": name, "lh": name, "anchor": name} plus "prop" when the set declares one.
        Raises KeyError (via SetObj.match) if the set does not fit `published`.
        """
        keys = {
            "rh": self.rh.match(published),
            "lh": self.lh.match(published),
            "anchor": self.anchor.match(published),
        }
        if self.prop is not None:
            keys["prop"] = self.prop.match(published)
        return keys


DEFAULT_OBJECT_SET = "bottle"

OBJECT_SETS = {
    # OakInk-v2 alcohol burner: LH holds the body, RH brings the cap down onto it. Two scored
    # objects, no prop — what every recorded my_dataset capture uses, so it stays the default.
    "bottle": ObjectSet(
        name="bottle",
        rh=SetObj("bottle_cap", ("bottle_cap", "cap")),
        lh=SetObj("bottle_body", ("bottle_body", "bottle", "body")),
        anchor=SetObj("bottle_body", ("bottle_body", "bottle", "body")),
        prop=None,
        seating_cutoff=True,
    ),
    # 3D-printed cup + square brush. The BRUSH is moved and rotated in the air by both hands, so it
    # is the single scored body (both sides point at it and sharedObject splits its reward). The CUP
    # is a passive receptacle — spawned as a free rigid body the brush is placed into and held
    # upright by, never scored and never a failure target. It anchors the recentering because it is
    # the thing that stays put.
    "cup_brush": ObjectSet(
        name="cup_brush",
        rh=SetObj("square_brush", ("d2_brush", "square_brush", "brush")),
        lh=SetObj("square_brush", ("d2_brush", "square_brush", "brush")),
        anchor=SetObj("cup", ("d2_cup", "cup")),
        prop=SetObj("cup", ("d2_cup", "cup")),
        seating_cutoff=False,  # no cap-seats-on-bottle geometry here
        # the cup's mesh origin IS its base, so no lift — the burner's +0.05 would float the whole
        # scene (hands and brush included) 5 cm above the table
        recenter_fine=(0.0, 0.0, 0.0),
    ),
}


def get_object_set(name: str) -> ObjectSet:
    if name not in OBJECT_SETS:
        raise KeyError(
            f"unknown objectSet '{name}'. Known sets: {sorted(OBJECT_SETS)}. "
            f"Add one in main/dataset/object_sets.py."
        )
    return OBJECT_SETS[name]


def positional_object_set(obj_ids) -> ObjectSet:
    """The historical rule as an ObjectSet: LH = first id, RH = last id, no prop.

    The fallback for captures no registered set describes — notably single-tracked-object takes like
    SHARED_OBJ_m_170751.pkl (obj_id == ['bottle_cap']), where first and last are the same body and
    both hands therefore score it. Reproduces the pre-object-set loader behaviour exactly.
    """
    obj_ids = list(obj_ids)
    rh_id, lh_id = obj_ids[-1], obj_ids[0]
    anchor_id = RECENTER_ANCHOR_OBJ if RECENTER_ANCHOR_OBJ in obj_ids else obj_ids[0]
    return ObjectSet(
        name=f"positional[{','.join(obj_ids)}]",
        rh=SetObj(rh_id, (rh_id,)),
        lh=SetObj(lh_id, (lh_id,)),
        anchor=SetObj(anchor_id, (anchor_id,)),
        prop=None,
        seating_cutoff=True,
    )


def infer_object_set(obj_ids) -> ObjectSet:
    """Pick the set a recorded capture belongs to from the objects it tracked.

    Every registered set that fits is collected, so a name collision between two sets is an error
    rather than a silent first-match. Falls back to `positional_object_set` when none fits.
    """
    matches = []
    for objset in OBJECT_SETS.values():
        try:
            objset.resolve_names(obj_ids)
        except KeyError:
            continue
        matches.append(objset)
    if len(matches) > 1:
        raise KeyError(
            f"objects {list(obj_ids)} match more than one set ({[m.name for m in matches]}). "
            f"Make the `names` in main/dataset/object_sets.py unambiguous."
        )
    return matches[0] if matches else positional_object_set(obj_ids)


# Back-compat: the pre-object-set anchor helper, still the rule `positional_object_set` applies.
def recenter_anchor(obj_ids):
    """Which object's first frame defines the scene origin (positional fallback rule)."""
    return RECENTER_ANCHOR_OBJ if RECENTER_ANCHOR_OBJ in obj_ids else obj_ids[0]
