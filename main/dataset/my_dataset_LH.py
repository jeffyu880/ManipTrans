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

# Object sets, asset lookup and the recentering constants are shared with the other loader, the env
# and LiveTargetSource — see main/dataset/object_sets.py, which is where they now live (they used to
# be duplicated verbatim in both loaders behind "KEEP IDENTICAL" comments). Re-exported here so
# existing imports of these names from this module keep working.
from .object_sets import (  # noqa: F401
    AVP_TO_MANO_JOINTS,
    MY_DATASET_OBJ_DIR,
    OBJ_ASSETS,
    RECENTER_ANCHOR_OBJ,
    RECENTER_FINE,
    TABLE_Z_ROT_DEG,
    WRIST_PULLBACK,
    infer_object_set,
    recenter_anchor,
    resolve_obj_assets,
)

SIDE = "left"  # this loader reads the left hand



@register_manipdata("mydataset_lh")
class MyDatasetLH(ManipData):
    def __init__(
        self,
        *,
        data_dir: str = "data/my_dataset",
        split: str = "all",
        skip: int = 1,  # capture is already 60Hz
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
        # wrist_pos = wrist_pos - (middle_pos - wrist_pos) * WRIST_PULLBACK
        wrist_pos += torch.tensor(self.dexhand.relative_translation, device=self.device)

        # -- wrist rotation: AVP quat -> matrix, then apply dexhand offset --
        wrist_rotmat = R.from_quat(wrist_quat).as_matrix()  # [T,3,3]
        rot_offset = np.repeat(self.dexhand.relative_rotation[None], length, axis=0)
        wrist_rot = torch.tensor(wrist_rotmat, dtype=torch.float32, device=self.device) @ torch.tensor(
            rot_offset, dtype=torch.float32, device=self.device
        )
        # AVP's left-hand wrist frame differs from the MANO frame that
        # dexhand.relative_rotation was tuned for, so the LH faces the wrong way.
        # Correct it with an extra rotation in the dex wrist's local frame. If the
        # hand is still off, change the axis ([0,0,pi] Z, [0,pi,0] Y, [pi,0,0] X) or
        # the angle until it points correctly.
        AVP_LH_WRIST_CORRECTION = R.from_rotvec([0.0, np.pi, 0.0]).as_matrix()  # 180 deg about Y
        wrist_rot = wrist_rot @ torch.tensor(
            AVP_LH_WRIST_CORRECTION, dtype=torch.float32, device=self.device
        )

        # -- object: which body this hand SCORES comes from the capture's object set (see
        # main/dataset/object_sets.py). For the burner that is the body; for a set where both hands
        # manipulate one object (cup_brush) both sides resolve to the same body, and the env then
        # spawns a single scored actor. `obj_id` carries the ASSET id, not the capture's rigid-body
        # name, because the env keys its asset cache and the shared-body check off it. --
        objset = infer_object_set(raw["obj_id"])
        names = objset.resolve_names(raw["obj_id"])
        obj_id = objset.lh.asset_id
        obj_mesh_path, obj_urdf_path = objset.lh.assets()  # pkl paths are None
        obj_traj = torch.tensor(
            raw["obj_transf"][names["lh"]][sl], dtype=torch.float32, device=self.device
        )  # [T,4,4]
        obj_mesh = trimesh.load(obj_mesh_path, force="mesh", process=False)
        mesh = Meshes(
            verts=torch.from_numpy(np.asarray(obj_mesh.vertices)[None].astype(np.float32)),
            faces=torch.from_numpy(np.asarray(obj_mesh.faces)[None].astype(np.float32)),
        )
        rs_verts_obj = self.random_sampling_pc(mesh)

        # move wrist position back as it is too close to the object

        # -- prop: spawned and collided with, but never scored (the cup the brush is placed into).
        # Only its pose is needed — no verts/BPS/tips, since it is not a reward or failure target. --
        prop_traj = None
        if objset.prop is not None:
            prop_traj = torch.tensor(
                raw["obj_transf"][names["prop"]][sl], dtype=torch.float32, device=self.device
            )  # [T,4,4]

        # -- recenter trajectory onto the table (raw frame; see object_sets.RECENTER_*) --
        anchor0 = torch.tensor(
            raw["obj_transf"][names["anchor"]][sl][0][:3, 3], dtype=torch.float32, device=self.device
        )
        recenter = torch.tensor(objset.recenter_fine, dtype=torch.float32, device=self.device) - anchor0
        wrist_pos = wrist_pos + recenter
        mano_joints = {k: v + recenter for k, v in mano_joints.items()}
        obj_traj[:, :3, 3] += recenter
        if prop_traj is not None:
            prop_traj[:, :3, 3] += recenter

        # -- rotate the whole scene about the table's vertical axis (see TABLE_Z_ROT_DEG) --
        table_rot = torch.tensor(
            R.from_rotvec([0.0, np.deg2rad(TABLE_Z_ROT_DEG), 0.0]).as_matrix(),
            dtype=torch.float32, device=self.device,
        )
        wrist_pos = (table_rot @ wrist_pos.T).T
        mano_joints = {k: (table_rot @ v.T).T for k, v in mano_joints.items()}
        wrist_rot = table_rot @ wrist_rot
        obj_traj[:, :3, 3] = (table_rot @ obj_traj[:, :3, 3].T).T
        obj_traj[:, :3, :3] = table_rot @ obj_traj[:, :3, :3]
        if prop_traj is not None:
            prop_traj[:, :3, 3] = (table_rot @ prop_traj[:, :3, 3].T).T
            prop_traj[:, :3, :3] = table_rot @ prop_traj[:, :3, :3]

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
        if prop_traj is not None:
            # keys are omitted entirely for prop-less sets, so pack_data never sees a ragged field
            data["prop_obj_id"] = objset.prop.asset_id
            data["prop_urdf_path"] = objset.prop.assets()[1]
            data["prop_trajectory"] = prop_traj

        self.process_data(data, stem, rs_verts_obj)

        opt_path = f"data/retargeting/my_dataset/mano2{str(self.dexhand)}/{stem}_lh.pkl"
        self.load_retargeted_data(data, opt_path)

        return data


if __name__ == "__main__":
    fdata = MyDatasetLH(mujoco2gym_transf=torch.eye(4, device="cuda:0"))
    print(fdata[list(fdata.stem_to_path.keys())[0]])
