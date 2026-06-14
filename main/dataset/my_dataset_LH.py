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

SIDE = "left"  # this loader reads the left hand


@register_manipdata("mydataset_lh")
class MyDatasetLH(ManipData):
    def __init__(
        self,
        *,
        data_dir: str = "data/my_dataset",
        split: str = "all",
        skip: int = 1,  # capture is already 60Hz
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
        # Accept "stem" or "stem_bih" / "stem@0"-style; normalise to the stem.
        stem = str(index).split("@")[0].replace("_bih", "")
        assert stem in self.stem_to_path, f"index '{stem}' not found in {self.data_dir}"
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

        # ? hack for wrist position (mirrors oakink2/grab); tune the 0.25 for AVP if needed
        middle_pos = mano_joints["middle_proximal"]
        wrist_pos = wrist_pos - (middle_pos - wrist_pos) * 0.25
        wrist_pos += torch.tensor(self.dexhand.relative_translation, device=self.device)

        # -- wrist rotation: AVP quat -> matrix, then apply dexhand offset --
        wrist_rotmat = R.from_quat(wrist_quat).as_matrix()  # [T,3,3]
        rot_offset = np.repeat(self.dexhand.relative_rotation[None], length, axis=0)
        wrist_rot = torch.tensor(wrist_rotmat, dtype=torch.float32, device=self.device) @ torch.tensor(
            rot_offset, dtype=torch.float32, device=self.device
        )

        # -- object: LH tracks the last object in the list (e.g. the cap) --
        obj_id = raw["obj_id"][-1]
        obj_traj = torch.tensor(raw["obj_transf"][obj_id][sl], dtype=torch.float32, device=self.device)  # [T,4,4]
        obj_mesh = trimesh.load(raw["obj_mesh_path"][obj_id], process=False)
        mesh = Meshes(
            verts=torch.from_numpy(np.asarray(obj_mesh.vertices)[None].astype(np.float32)),
            faces=torch.from_numpy(np.asarray(obj_mesh.faces)[None].astype(np.float32)),
        )
        rs_verts_obj = self.random_sampling_pc(mesh)

        data = {
            "data_path": pkl_path,
            "obj_id": obj_id,
            "obj_verts": rs_verts_obj,
            "obj_urdf_path": raw["obj_urdf_path"][obj_id],
            "obj_trajectory": obj_traj,
            "scene_objs": [],
            "wrist_pos": wrist_pos,
            "wrist_rot": wrist_rot,
            "mano_joints": mano_joints,
            "frame_ids": raw["frame_id_list"][sl],
        }

        self.process_data(data, stem, rs_verts_obj)

        opt_path = f"data/retargeting/my_dataset/mano2{str(self.dexhand)}/{stem}_lh.pkl"
        self.load_retargeted_data(data, opt_path)

        return data


if __name__ == "__main__":
    fdata = MyDatasetLH(mujoco2gym_transf=torch.eye(4, device="cuda:0"))
    print(fdata[list(fdata.stem_to_path.keys())[0]])
