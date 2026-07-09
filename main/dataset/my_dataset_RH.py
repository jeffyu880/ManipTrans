import os
import pickle
from functools import lru_cache

import numpy as np
import torch
import trimesh
from pytorch3d.structures import Meshes
from scipy.spatial.transform import Rotation as R

from .base import ManipData
from .decorators import register_manipdata

# AVP joint frame -> ManipTrans mano_joints name. Kept here as a fallback; the
# authoritative mapping is also stored in each pkl under meta['avp_to_mano_joints'].
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

SIDE = "right"  # this loader reads the right hand

# The capture pkl stores obj_mesh_path/obj_urdf_path as None, so object assets are
# hard-coded here, reusing the OakInk-v2 alcohol burner body + cap. Keyed by the
# pkl's obj_id -> (mesh .ply for verts/BPS, coacd .urdf for sim).
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

# --- Trajectory recentering -------------------------------------------------------
# OptiTrack's world origin is not the sim table, so the raw capture lands off-center
# and too high. We subtract this object's first-frame position so the scene starts at
# the raw origin, which mujoco2gym_transf maps onto the table center. Done in RAW frame
# so it is identical whether the loader applies the transform (train/test) or
# mano2dexhand applies it (retargeting). RECENTER_FINE nudges afterward, in RAW (AVP,
# Y-up) frame: +Y -> gym +Z (up/height), +X -> gym -Y, +Z -> gym -X. Raise +Y if the
# objects sink into the table.
# !!! KEEP RECENTER_FINE IDENTICAL TO my_dataset_LH.py or the two hands will desync !!!
RECENTER_ANCHOR_OBJ = "bottle_body"
RECENTER_FINE = (0.0, 0.05, 0.0)  # (x, y, z) metres, raw frame

# Rotate the whole scene (both hands + objects) about the table's vertical axis.
# Applied in RAW frame after recentering: raw +Y maps to gym +Z (the table's up axis),
# so a rotation about raw Y is exactly a rotation about the table's Z. Done in raw frame
# (before mujoco2gym_transf) so it stays consistent for both train/test and retargeting.
# Flip the sign to reverse direction.
# !!! KEEP TABLE_Z_ROT_DEG IDENTICAL TO my_dataset_LH.py or the two hands will desync !!!
TABLE_Z_ROT_DEG = 90.0

# How far to pull the wrist back from the fingers. 0.25 = 25% of wrist-to-MCP
# distance toward the forearm. Increase if the hand reaches over the object.
WRIST_PULLBACK = 0.0

# # The OptiTrack rigid body origin for the cap is at the physical base (opening rim,
# # Y ≈ 0.016 m in OakInk mesh frame). The OakInk mesh origin sits ~1.6 cm outside
# # that rim (Y = 0). Post-multiply obj_traj by this to shift the tracked pose from
# # the OptiTrack body frame to the OakInk mesh frame. Tune CAP_Y_OFFSET if the cap
# # appears offset in sim — positive values move the mesh toward the dome end.
# CAP_Y_OFFSET = -0.016  # metres along cap local Y axis (opening → outside)


@register_manipdata("mydataset_rh")
class MyDatasetRH(ManipData):
    def __init__(
        self,
        *,
        data_dir: str = "data/my_dataset",
        split: str = "all",
        skip: int = 2,  # 60Hz capture -> 30Hz effective demo rate (fps/skip); matches dt=1/30 training
        fps: float = 60.0,  # OptiTrack+AVP native capture rate (Hz)
        device="cuda:0",
        mujoco2gym_transf=None,
        max_seq_len=int(1e10),
        dexhand=None,
        **kwargs,
    ):
        super().__init__(
            data_dir=data_dir,
            split=split,
            skip=skip,
            fps=fps,
            device=device,
            mujoco2gym_transf=mujoco2gym_transf,
            max_seq_len=max_seq_len,
            dexhand=dexhand,
            **kwargs,
        )
        # Map index (pkl stem) -> file path. Index passed at load time is the stem.
        pkls = [p for p in os.listdir(data_dir) if p.endswith(".pkl")]
        pkls.sort()
        self.data_pathes = [os.path.join(data_dir, p) for p in pkls]
        self.stem_to_path = {os.path.splitext(p)[0]: os.path.join(data_dir, p) for p in pkls}

    @lru_cache(maxsize=None)
    def __getitem__(self, index):
        assert self.mujoco2gym_transf is not None
        # Index is e.g. "m_160009": "m_" marker (see ManipDataFactory.dataset_type) plus the
        # last 6 digits of the filename stem. Strip the marker and resolve to the unique pkl
        # whose stem ends with those digits.
        key = str(index)
        if key.startswith("m_"):
            key = key[2:]  # drop the "m_" marker
        if key in self.stem_to_path:
            stem = key
        else:
            matches = [s for s in self.stem_to_path if s.endswith(key)]
            assert len(matches) == 1, (
                f"index '{key}' matched {len(matches)} pkls in {self.data_dir}: {sorted(matches)}"
            )
            stem = matches[0]
        pkl_path = self.stem_to_path[stem]

        raw = pickle.load(open(pkl_path, "rb"))
        hand = raw["hands"][SIDE]
        joint_map = raw["meta"].get("avp_to_mano_joints", AVP_TO_MANO_JOINTS)
        sl = slice(None, None, self.skip)

        # -- wrist --
        wrist_pos = torch.tensor(hand["wrist_pos"][sl], dtype=torch.float32, device=self.device)  # [T,3]
        wrist_quat = hand["wrist_quat"][sl]  # [T,4] xyzw
        length = wrist_pos.shape[0]

        # -- mano joints (world positions straight from AVP) --
        mano_joints = {
            name: torch.tensor(hand["joints_pos"][avp_name][sl], dtype=torch.float32, device=self.device)
            for name, avp_name in joint_map.items()
        }

        # ? hack for wrist position (mirrors oakink2/grab); tune WRIST_PULLBACK for AVP if needed
        middle_pos = mano_joints["middle_proximal"]
        wrist_pos = wrist_pos - (middle_pos - wrist_pos) * WRIST_PULLBACK
        wrist_pos += torch.tensor(self.dexhand.relative_translation, device=self.device)

        # -- wrist rotation: AVP quat -> matrix, then apply dexhand offset --
        wrist_rotmat = R.from_quat(wrist_quat).as_matrix()  # [T,3,3]
        rot_offset = np.repeat(self.dexhand.relative_rotation[None], length, axis=0)
        wrist_rot = torch.tensor(wrist_rotmat, dtype=torch.float32, device=self.device) @ torch.tensor(
            rot_offset, dtype=torch.float32, device=self.device
        )

        # -- object: RH holds the cap (bottle_cap, last in the list) --
        obj_id = raw["obj_id"][-1]
        obj_mesh_path, obj_urdf_path = OBJ_ASSETS[obj_id]  # hard-coded; pkl paths are None
        obj_traj = torch.tensor(raw["obj_transf"][obj_id][sl], dtype=torch.float32, device=self.device)  # [T,4,4]
        # # Shift from OptiTrack body frame (origin at cap opening rim) to OakInk mesh frame.
        # cap_offset = torch.eye(4, dtype=torch.float32, device=self.device)
        # cap_offset[1, 3] = CAP_Y_OFFSET
        # obj_traj = obj_traj @ cap_offset
        obj_mesh = trimesh.load(obj_mesh_path, process=False)
        mesh = Meshes(
            verts=torch.from_numpy(np.asarray(obj_mesh.vertices)[None].astype(np.float32)),
            faces=torch.from_numpy(np.asarray(obj_mesh.faces)[None].astype(np.float32)),
        )
        rs_verts_obj = self.random_sampling_pc(mesh)

        # -- recenter trajectory onto the table (raw frame; see RECENTER_* above) --
        anchor0 = torch.tensor(
            raw["obj_transf"][RECENTER_ANCHOR_OBJ][sl][0][:3, 3], dtype=torch.float32, device=self.device
        )
        recenter = torch.tensor(RECENTER_FINE, dtype=torch.float32, device=self.device) - anchor0
        wrist_pos = wrist_pos + recenter
        mano_joints = {k: v + recenter for k, v in mano_joints.items()}
        obj_traj[:, :3, 3] += recenter

        # -- rotate the whole scene about the table's vertical axis (see TABLE_Z_ROT_DEG) --
        # !!! KEEP IDENTICAL TO my_dataset_LH.py or the two hands will desync !!!
        table_rot = torch.tensor(
            R.from_rotvec([0.0, np.deg2rad(TABLE_Z_ROT_DEG), 0.0]).as_matrix(),
            dtype=torch.float32, device=self.device,
        )
        wrist_pos = (table_rot @ wrist_pos.T).T
        mano_joints = {k: (table_rot @ v.T).T for k, v in mano_joints.items()}
        wrist_rot = table_rot @ wrist_rot
        obj_traj[:, :3, 3] = (table_rot @ obj_traj[:, :3, 3].T).T
        obj_traj[:, :3, :3] = table_rot @ obj_traj[:, :3, :3]

        data = {
            "data_path": pkl_path,
            "obj_id": obj_id,
            "obj_verts": rs_verts_obj,
            "obj_urdf_path": obj_urdf_path,
            "obj_trajectory": obj_traj,
            "scene_objs": [],
            "wrist_pos": wrist_pos,
            "wrist_rot": wrist_rot,
            "mano_joints": mano_joints,
            "frame_ids": raw["frame_id_list"][sl],
        }

        self.process_data(data, stem, rs_verts_obj)

        opt_path = f"data/retargeting/my_dataset/mano2{str(self.dexhand)}/{stem}_rh.pkl"
        self.load_retargeted_data(data, opt_path)

        return data


if __name__ == "__main__":
    fdata = MyDatasetRH(mujoco2gym_transf=torch.eye(4, device="cuda:0"))
    print(fdata[list(fdata.stem_to_path.keys())[0]])
