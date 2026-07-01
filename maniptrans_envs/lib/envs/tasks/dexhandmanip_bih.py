from __future__ import annotations

import os
import random
from enum import Enum
from itertools import cycle
from time import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from ...utils import torch_jit_utils as torch_jit_utils
from bps_torch.bps import bps_torch
from gym import spaces
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import normalize_angle, quat_conjugate, quat_mul
from copy import deepcopy
import math
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
from main.dataset.factory import ManipDataFactory

# from main.dataset.favor_dataset_dexhand import FavorDatasetDexHand
from main.dataset.oakink2_dataset_dexhand_lh import OakInk2DatasetDexHandLH
from main.dataset.oakink2_dataset_dexhand_rh import OakInk2DatasetDexHandRH
from main.dataset.oakink2_dataset_utils import oakink2_obj_scale, oakink2_obj_mass
from main.dataset.transform import aa_to_quat, aa_to_rotmat, quat_to_rotmat, rotmat_to_aa, rotmat_to_quat, rot6d_to_aa
from torch import Tensor
from tqdm import tqdm
from ...asset_root import ASSET_ROOT


from ..core.config import ROBOT_HEIGHT, config
from ...envs.core.sim_config import sim_config
from ...envs.core.vec_task import VecTask
from ...utils.pose_utils import get_mat


def soft_clamp(x, lower, upper):
    return lower + torch.sigmoid(4 / (upper - lower) * (x - (lower + upper) / 2)) * (upper - lower)


class DexHandManipBiHEnv(VecTask):

    def __init__(
        self,
        cfg,
        *,
        rl_device: int = 0,
        sim_device: int = 0,
        graphics_device_id: int = 0,
        display: bool = False,
        record: bool = False,
        headless: bool = True,
    ):
        self._record = record
        self.cfg = cfg

        use_quat_rot = self.use_quat_rot = self.cfg["env"]["useQuatRot"]
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        _max_demo_len = self.cfg["env"].get("maxDemoLength", None)
        self.max_demo_length = _max_demo_len if _max_demo_len is not None else self.max_episode_length
        self.zero_residual = self.cfg["env"].get("zeroResidual", False)
        self.use_traj_aug = self.cfg["env"].get("useTrajAug", False)
        self.joint_noise_std = self.cfg["env"].get("jointNoiseCm", 0.0) / 100.0  # cm → meters
        self.failure_threshold_noise_compensation = self.cfg["env"].get("failureThresholdNoiseCompensation", 1.0)  # multiplier on finger failure thresholds; 1.0 = no change, >1 loosens to compensate for injected joint noise
        self.obs_hand_noise = self.cfg["env"].get("obsHandNoise", 0.0)
        self.obs_hand_vel_noise = self.cfg["env"].get("obsHandVelNoise", 0.0)
        # --live: stream targets from the laptop (AVP+Motive) instead of the demo buffer.
        # The demo is still loaded (assets/BPS/opt-init/buffer shapes); its target slots are
        # overwritten each step by the latest live frame, broadcast across all envs. See live/.
        self.live = self.cfg["env"].get("live", False)
        self.live_addr = self.cfg["env"].get("liveAddr", "128.178.169.131")
        self.live_port = int(self.cfg["env"].get("livePort", 5555))
        # liveBuffered: FIFO-consume every published frame (faithful trajectory replay) instead
        # of newest-only. Use true when replaying a recording (mock_publish); false for teleop.
        self.live_buffered = self.cfg["env"].get("liveBuffered", False)
        self.live_source = None
        if self.live:
            # Live overwrites every demo target slot each step; keep the buffer tiny so that
            # broadcast write is cheap (cur_idx clamps to seq_len-1, so a short demo is fine).
            self.max_demo_length = min(self.max_demo_length, 4)
        self.action_scale = self.cfg["env"]["actionScale"]
        # self.dexhand_rh_dof_noise = self.cfg["env"]["dexhand_rDofNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.training = self.cfg["env"]["training"]
        self.dexhand_rh = DexHandFactory.create_hand(self.cfg["env"]["dexhand"], "right")
        self.dexhand_lh = DexHandFactory.create_hand(self.cfg["env"]["dexhand"], "left")

        self.use_pid_control = self.cfg["env"]["usePIDControl"]
        if self.use_pid_control:
            self.Kp_rot = self.dexhand_rh.Kp_rot
            self.Ki_rot = self.dexhand_rh.Ki_rot
            self.Kd_rot = self.dexhand_rh.Kd_rot

            self.Kp_pos = self.dexhand_rh.Kp_pos
            self.Ki_pos = self.dexhand_rh.Ki_pos
            self.Kd_pos = self.dexhand_rh.Kd_pos

        self.cfg["env"]["numActions"] = (
            (1 + 6 + self.dexhand_lh.n_dofs) if use_quat_rot else (6 + self.dexhand_lh.n_dofs)
        ) * (2 if self.cfg["env"]["bimanual_mode"] == "united" else 1)
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]
        self.translation_scale = self.cfg["env"]["translationScale"]
        self.orientation_scale = self.cfg["env"]["orientationScale"]

        # a dict containing prop obs name to dump and their dimensions
        # used for distillation
        self._prop_dump_info = self.cfg["env"]["propDumpInfo"]

        # Values to be filled in at runtime
        self.rh_states = {}
        self.lh_states = {}
        self.dexhand_rh_handles = {}  # will be dict mapping names to relevant sim handles
        self.dexhand_lh_handles = {}  # will be dict mapping names to relevant sim handles
        self.objs_handles = {}  # for obj handlers
        self.objs_assets = {}
        self.num_dofs = None  # Total number of DOFs per env
        self.actions = None  # Current actions to be deployed

        self.dataIndices = self.cfg["env"]["dataIndices"]
        # self.dataIndices = [tuple([int(i) for i in idx.split("@")]) for idx in self.dataIndices]
        self._pending_demo_episode_rewards = {idx: [] for idx in self.dataIndices}
        self._pending_demo_episode_successes = {idx: [] for idx in self.dataIndices}
        self.obs_future_length = self.cfg["env"]["obsFutureLength"]
        self.rollout_state_init = self.cfg["env"]["rolloutStateInit"]
        self.random_state_init = self.cfg["env"]["randomStateInit"]

        self.tighten_method = self.cfg["env"]["tightenMethod"]
        self.tighten_factor = self.cfg["env"]["tightenFactor"]
        self.tighten_steps = self.cfg["env"]["tightenSteps"]

        self.rollout_len = self.cfg["env"].get("rolloutLen", None)
        self.rollout_begin = self.cfg["env"].get("rolloutBegin", None)
        self.use_pen_keypoint_reward = self.cfg["env"].get("usePenKeypointReward", False)
        self.use_coaxial_reward = self.cfg["env"].get("useCoaxialReward", False)
        self.eval_start_frame = self.cfg["env"].get("evalStartFrame", 0)

        assert len(self.dataIndices) == 1 or self.rollout_len is None, "rolloutLen only works with one data"
        assert len(self.dataIndices) == 1 or self.rollout_begin is None, "rolloutBegin only works with one data"

        # Tensor placeholders
        self._root_state = None  # State of root body        (n_envs, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        self._q = None  # Joint positions           (n_envs, n_dof)
        self._qd = None  # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        self.net_cf = None  # contact force
        self._eef_state = None  # end effector state (at grasping point)
        self._ftip_center_state = None  # center of fingertips
        self._eef_lf_state = None  # end effector state (at left fingertip)
        self._eef_rf_state = None  # end effector state (at left fingertip)
        self._j_eef = None  # Jacobian for end effector
        self._mm = None  # Mass matrix
        self._pos_control = None  # Position actions
        self._effort_control = None  # Torque actions
        self._dexhand_rh_effort_limits = None  # Actuator effort limits for dexhand_r
        self._dexhand_rh_dof_speed_limits = None  # Actuator speed limits for dexhand_r
        self._global_dexhand_rh_indices = None  # Unique indices corresponding to all envs in flattened array

        self.sim_device = torch.device(sim_device)
        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            display=display,
            record=record,
            headless=headless,
        )
        self._pen_tip_offset = torch.tensor([0.0, 0.0, 0.069], device=self.device, dtype=torch.float32)
        self._cap_open_offset = torch.tensor([0.0, 0.0, 0.039], device=self.device, dtype=torch.float32)
        self._z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=torch.float32)
        TARGET_OBS_DIM = (
            128
            + 5
            + (
                3
                + 3
                + 3
                + 4
                + 4
                + 3
                + 3
                + (self.dexhand_rh.n_bodies - 1) * 9
                + 3
                + 3
                + 3
                + 4
                + 4
                + 3
                + 3
                + self.dexhand_rh.n_bodies
            )
            * self.obs_future_length
        ) * 2
        self.obs_dict.update(
            {
                "target": torch.zeros((self.num_envs, TARGET_OBS_DIM), device=self.device),
            }
        )
        obs_space = self.obs_space.spaces
        obs_space["target"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(TARGET_OBS_DIM,),
        )
        self.obs_space = spaces.Dict(obs_space)

        # dexhand_r defaults
        # TODO hack here
        # default_pose = self.cfg["env"].get("dexhand_rDefaultDofPos", None)
        default_pose = torch.ones(self.dexhand_rh.n_dofs, device=self.device) * np.pi / 12
        if self.cfg["env"]["dexhand"] == "inspire":
            default_pose[8] = 0.3
            default_pose[9] = 0.01
        self.dexhand_rh_default_dof_pos = torch.tensor(default_pose, device=self.sim_device)
        self.dexhand_lh_default_dof_pos = torch.tensor(default_pose, device=self.sim_device)  # ? TODO check this
        # self.dexhand_rh_default_dof_pos = torch.tensor([-3.5322e-01,  -0.100e-01,  3.2278e-01, -2.51e+00,  1.6036e-01,
        #   2.564e+00, 0.5,  0.10,  0.10], device=self.sim_device)

        # load BPS model
        self.bps_feat_type = "dists"
        self.bps_layer = bps_torch(
            bps_type="grid_sphere", n_bps_points=128, radius=0.2, randomize=False, device=self.device
        )

        obj_verts_rh = self.demo_data_rh["obj_verts"]
        self.obj_bps_rh = self.bps_layer.encode(obj_verts_rh, feature_type=self.bps_feat_type)[self.bps_feat_type]
        obj_verts_lh = self.demo_data_lh["obj_verts"]
        self.obj_bps_lh = self.bps_layer.encode(obj_verts_lh, feature_type=self.bps_feat_type)[self.bps_feat_type]

        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # Refresh tensors
        self._refresh()

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.8
        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self._create_ground_plane()
        self._create_envs()

        if self.randomize:
            self.apply_randomizations(self.dr_randomizations)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _apply_joint_noise(self, hand):
        # print("APPLYING NOISE")
        from copy import copy
        d = copy(hand)
        d["wrist_pos"] = hand["wrist_pos"] + (torch.rand_like(hand["wrist_pos"]) * (2*self.joint_noise_std) - self.joint_noise_std)
        d["mano_joints"] = {
            k: v + (torch.rand_like(v) * (2*self.joint_noise_std) - self.joint_noise_std)
            for k, v in hand["mano_joints"].items()
        }
        return d

    def _create_envs(self):
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.camera_handlers_top = [] if self._record else None

        # * >>> import table asset
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.fix_base_link = True

        table_width_offset = 0.2
        table_asset = self.gym.create_box(self.sim, 0.8 + table_width_offset, 1.6, 0.03, table_asset_options)

        table_pos = gymapi.Vec3(-table_width_offset / 2, 0, 0.4)    # corresponds to the center of the table
        self.dexhand_rh_pose = gymapi.Transform()
        table_half_height = 0.015
        table_half_width = 0.4
        self._table_surface_z = table_surface_z = table_pos.z + table_half_height
        self._aug_center = torch.tensor([table_pos.x, table_pos.y, table_surface_z],
                                        dtype=torch.float32, device=self.sim_device)
        self.dexhand_rh_pose.p = gymapi.Vec3(-table_half_width, 0, table_surface_z + ROBOT_HEIGHT)
        self.dexhand_rh_pose.r = gymapi.Quat.from_euler_zyx(0, -np.pi / 2, 0)
        self.dexhand_lh_pose = deepcopy(self.dexhand_rh_pose)

        mujoco2gym_transf = np.eye(4)
        mujoco2gym_transf[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(
            np.array([np.pi / 2, 0, 0])
        )
        mujoco2gym_transf[:3, 3] = np.array([0, 0, self._table_surface_z])
        self.mujoco2gym_transf = torch.tensor(mujoco2gym_transf, device=self.sim_device, dtype=torch.float32)

        dataset_list = list(set([ManipDataFactory.dataset_type(data_idx) for data_idx in self.dataIndices]))

        self.demo_dataset_lh_dict = {}
        self.demo_dataset_rh_dict = {}

        for dataset_type in dataset_list:
            self.demo_dataset_lh_dict[dataset_type] = ManipDataFactory.create_data(
                manipdata_type=dataset_type,
                side="left",
                device=self.sim_device,
                mujoco2gym_transf=self.mujoco2gym_transf,
                max_seq_len=self.max_demo_length,
                dexhand=self.dexhand_lh,
                embodiment=self.cfg["env"]["dexhand"],
            )
            self.demo_dataset_rh_dict[dataset_type] = ManipDataFactory.create_data(
                manipdata_type=dataset_type,
                side="right",
                device=self.sim_device,
                mujoco2gym_transf=self.mujoco2gym_transf,
                max_seq_len=self.max_demo_length,
                dexhand=self.dexhand_rh,
                embodiment=self.cfg["env"]["dexhand"],
            )

        dexhand_rh_asset_file = self.dexhand_rh.urdf_path
        dexhand_lh_asset_file = self.dexhand_lh.urdf_path
        asset_options = gymapi.AssetOptions()
        asset_options.thickness = 0.001
        asset_options.angular_damping = 20
        asset_options.linear_damping = 20
        asset_options.max_linear_velocity = 50
        asset_options.max_angular_velocity = 100
        asset_options.fix_base_link = False
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        dexhand_rh_asset = self.gym.load_asset(self.sim, *os.path.split(dexhand_rh_asset_file), asset_options)
        dexhand_lh_asset = self.gym.load_asset(self.sim, *os.path.split(dexhand_lh_asset_file), asset_options)
        dexhand_rh_dof_stiffness = torch.tensor(
            [500] * self.dexhand_rh.n_dofs,
            dtype=torch.float,
            device=self.sim_device,
        )
        dexhand_rh_dof_damping = torch.tensor(
            [30] * self.dexhand_rh.n_dofs,
            dtype=torch.float,
            device=self.sim_device,
        )
        dexhand_lh_dof_stiffness = torch.tensor(
            [500] * self.dexhand_lh.n_dofs,
            dtype=torch.float,
            device=self.sim_device,
        )
        dexhand_lh_dof_damping = torch.tensor(
            [30] * self.dexhand_lh.n_dofs,
            dtype=torch.float,
            device=self.sim_device,
        )
        self.limit_info = {}
        asset_rh_dof_props = self.gym.get_asset_dof_properties(dexhand_rh_asset)
        asset_lh_dof_props = self.gym.get_asset_dof_properties(dexhand_lh_asset)
        self.limit_info["rh"] = {
            "lower": np.asarray(asset_rh_dof_props["lower"]).copy().astype(np.float32),
            "upper": np.asarray(asset_rh_dof_props["upper"]).copy().astype(np.float32),
        }
        self.limit_info["lh"] = {
            "lower": np.asarray(asset_lh_dof_props["lower"]).copy().astype(np.float32),
            "upper": np.asarray(asset_lh_dof_props["upper"]).copy().astype(np.float32),
        }

        rigid_shape_rh_props_asset = self.gym.get_asset_rigid_shape_properties(dexhand_rh_asset)
        for element in rigid_shape_rh_props_asset:
            element.friction = 4.0
            element.rolling_friction = 0.01
            element.torsion_friction = 0.01
        self.gym.set_asset_rigid_shape_properties(dexhand_rh_asset, rigid_shape_rh_props_asset)

        rigid_shape_lh_props_asset = self.gym.get_asset_rigid_shape_properties(dexhand_lh_asset)
        for element in rigid_shape_lh_props_asset:
            element.friction = 4.0
            element.rolling_friction = 0.01
            element.torsion_friction = 0.01
        self.gym.set_asset_rigid_shape_properties(dexhand_lh_asset, rigid_shape_lh_props_asset)

        self.num_dexhand_rh_bodies = self.gym.get_asset_rigid_body_count(dexhand_rh_asset)
        self.num_dexhand_rh_dofs = self.gym.get_asset_dof_count(dexhand_rh_asset)
        self.num_dexhand_lh_bodies = self.gym.get_asset_rigid_body_count(dexhand_lh_asset)
        self.num_dexhand_lh_dofs = self.gym.get_asset_dof_count(dexhand_lh_asset)

        print(f"Num dexhand_r Bodies: {self.num_dexhand_rh_bodies}")
        print(f"Num dexhand_r DOFs: {self.num_dexhand_rh_dofs}")
        print(f"Num dexhand_l Bodies: {self.num_dexhand_lh_bodies}")
        print(f"Num dexhand_l DOFs: {self.num_dexhand_lh_dofs}")

        dexhand_rh_dof_props = self.gym.get_asset_dof_properties(dexhand_rh_asset)
        self.dexhand_rh_dof_lower_limits = []
        self.dexhand_rh_dof_upper_limits = []
        self._dexhand_rh_effort_limits = []
        self._dexhand_rh_dof_speed_limits = []
        for i in range(self.num_dexhand_rh_dofs):
            dexhand_rh_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            dexhand_rh_dof_props["stiffness"][i] = dexhand_rh_dof_stiffness[i]
            dexhand_rh_dof_props["damping"][i] = dexhand_rh_dof_damping[i]

            self.dexhand_rh_dof_lower_limits.append(dexhand_rh_dof_props["lower"][i])
            self.dexhand_rh_dof_upper_limits.append(dexhand_rh_dof_props["upper"][i])
            self._dexhand_rh_effort_limits.append(dexhand_rh_dof_props["effort"][i])
            self._dexhand_rh_dof_speed_limits.append(dexhand_rh_dof_props["velocity"][i])

        self.dexhand_rh_dof_lower_limits = torch.tensor(self.dexhand_rh_dof_lower_limits, device=self.sim_device)
        self.dexhand_rh_dof_upper_limits = torch.tensor(self.dexhand_rh_dof_upper_limits, device=self.sim_device)
        self._dexhand_rh_effort_limits = torch.tensor(self._dexhand_rh_effort_limits, device=self.sim_device)
        self._dexhand_rh_dof_speed_limits = torch.tensor(self._dexhand_rh_dof_speed_limits, device=self.sim_device)

        # set dexhand_l dof properties
        dexhand_lh_dof_props = self.gym.get_asset_dof_properties(dexhand_lh_asset)
        self.dexhand_lh_dof_lower_limits = []
        self.dexhand_lh_dof_upper_limits = []
        self._dexhand_lh_effort_limits = []
        self._dexhand_lh_dof_speed_limits = []
        for i in range(self.num_dexhand_lh_dofs):
            dexhand_lh_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            dexhand_lh_dof_props["stiffness"][i] = dexhand_lh_dof_stiffness[i]
            dexhand_lh_dof_props["damping"][i] = dexhand_lh_dof_damping[i]

            self.dexhand_lh_dof_lower_limits.append(dexhand_lh_dof_props["lower"][i])
            self.dexhand_lh_dof_upper_limits.append(dexhand_lh_dof_props["upper"][i])
            self._dexhand_lh_effort_limits.append(dexhand_lh_dof_props["effort"][i])
            self._dexhand_lh_dof_speed_limits.append(dexhand_lh_dof_props["velocity"][i])

        self.dexhand_lh_dof_lower_limits = torch.tensor(self.dexhand_lh_dof_lower_limits, device=self.sim_device)
        self.dexhand_lh_dof_upper_limits = torch.tensor(self.dexhand_lh_dof_upper_limits, device=self.sim_device)
        self._dexhand_lh_effort_limits = torch.tensor(self._dexhand_lh_effort_limits, device=self.sim_device)
        self._dexhand_lh_dof_speed_limits = torch.tensor(self._dexhand_lh_dof_speed_limits, device=self.sim_device)

        # compute aggregate size
        num_dexhand_rh_bodies = self.gym.get_asset_rigid_body_count(dexhand_rh_asset)
        num_dexhand_rh_shapes = self.gym.get_asset_rigid_shape_count(dexhand_rh_asset)
        num_dexhand_lh_bodies = self.gym.get_asset_rigid_body_count(dexhand_lh_asset)
        num_dexhand_lh_shapes = self.gym.get_asset_rigid_shape_count(dexhand_lh_asset)

        self.dexhand_rs = []
        self.dexhand_ls = []
        self.envs = []

        assert len(self.dataIndices) == 1 or not self.rollout_state_init, "rollout_state_init only works with one data"

        # Pre-generate augmented demo versions at load time so aug is applied
        # consistently across all fields (positions, rotations, velocities, reset).
        num_aug = self.cfg["env"].get("numTrajAug", 400) if self.use_traj_aug else 1
        if not self.training:
            _rng_state = torch.get_rng_state()
            torch.manual_seed(self.cfg.get("seed", 42))
        aug_transforms = [self._sample_aug_transform(self.device, self._aug_center) for _ in range(num_aug - 1)]
        if not self.training:
            torch.set_rng_state(_rng_state)

        use_lh_obj_center_aug = self.cfg["env"].get("useLHObjCenterAug", False)
        use_rh_obj_center_aug = self.cfg["env"].get("useRHObjCenterAug", False)
        # Rotate the LH demo (left hand + the left object it holds) about the LH object center.
        use_lh_about_lh_obj_aug = self.cfg["env"].get("useLHAboutLHObjAug", False)
        # Default to table-center aug when no per-object aug is selected (backward compat)
        use_table_center_aug = self.cfg["env"].get("useTableCenterAug",
                                                    not (use_lh_obj_center_aug or use_rh_obj_center_aug
                                                         or use_lh_about_lh_obj_aug))
        if self.use_traj_aug:
            active = [n for n, f in [("LH-obj-center", use_lh_obj_center_aug),
                                     ("RH-obj-center", use_rh_obj_center_aug),
                                     ("LH-about-LH-obj", use_lh_about_lh_obj_aug),
                                     ("table-center",  use_table_center_aug)] if f]
            assert active, "useTrajAug=true but no aug type is enabled"
            print(f"Trajectory augmentation pipeline: {' -> '.join(active)}")

        aug_demos_lh = {}  # idx -> [raw, aug_1, ..., aug_{K-1}]
        aug_demos_rh = {}
        for idx in self.dataIndices:
            dt = ManipDataFactory.dataset_type(idx)
            raw_lh = self.demo_dataset_lh_dict[dt][idx]
            raw_rh = self.demo_dataset_rh_dict[dt][idx]
            aug_list_rh = [raw_rh]
            aug_list_lh = [raw_lh]
            for R, t, c in aug_transforms:
                rh, lh = raw_rh, raw_lh
                if use_lh_obj_center_aug:
                    rh, lh = self._aug_demo_lh_obj_center(rh, lh, R)
                if use_rh_obj_center_aug:
                    rh = self._aug_demo_rh_obj_center(rh, R)
                if use_lh_about_lh_obj_aug:
                    lh = self._aug_demo_lh_about_lh_obj(lh, R)
                if use_table_center_aug:
                    # table-center aug TRANSLATES the demo only (identity rotation -> rp(x)=x+t);
                    # no rotation about the table center.
                    eye = torch.eye(3, dtype=torch.float32, device=self.device)
                    rh = self._aug_demo(rh, eye, t, center=c)
                    lh = self._aug_demo(lh, eye, t, center=c)
                if self.joint_noise_std > 0:
                    rh = self._apply_joint_noise(rh)
                    lh = self._apply_joint_noise(lh)
                aug_list_rh.append(rh)
                aug_list_lh.append(lh)
            aug_demos_rh[idx] = aug_list_rh
            aug_demos_lh[idx] = aug_list_lh

        def segment_data(k, aug_demos):
            todo_list = self.dataIndices
            idx = todo_list[k % len(todo_list)]
            aug_k = (k // len(todo_list)) % num_aug
            # during test with aug, skip aug_k=0 (original) so all envs use augmented variants
            if not self.training and self.use_traj_aug and num_aug > 1:
                aug_k = (aug_k % (num_aug - 1)) + 1
            return aug_demos[idx][aug_k]

        # [num_envs, nT, ...] packed demo buffers. In --live mode these are still built here from
        # the reference demo (dataIndices) for assets/BPS/opt-init/buffer shapes, but nT is capped
        # tiny (max_demo_length=4) and their target slots are OVERWRITTEN in place each step by
        # _inject_live() with the latest live frame (LiveTargetSource.latest()) NOTE pack_data's .squeeze() requires num_envs >= 2.
        self.demo_data_lh = [segment_data(i, aug_demos_lh) for i in tqdm(range(self.num_envs))]
        self.demo_data_lh = self.pack_data(self.demo_data_lh, side="lh")
        self.demo_data_rh = [segment_data(i, aug_demos_rh) for i in tqdm(range(self.num_envs))]
        self.demo_data_rh = self.pack_data(self.demo_data_rh, side="rh")
        self.env_demo_idx = [i % len(self.dataIndices) for i in range(self.num_envs)]

        # Create environments
        self.manip_obj_rh_mass = []
        self.manip_obj_rh_com = []
        self.manip_obj_lh_mass = []
        self.manip_obj_lh_com = []
        num_per_row = int(np.sqrt(self.num_envs))
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            rh_current_asset, rh_sum_rigid_body_count, rh_sum_rigid_shape_count, rh_obj_scale, rh_obj_mass = (
                self._create_obj_assets(i, side="rh")
            )
            lh_current_asset, lh_sum_rigid_body_count, lh_sum_rigid_shape_count, lh_obj_scale, lh_obj_mass = (
                self._create_obj_assets(i, side="lh")
            )

            max_agg_bodies = (
                num_dexhand_rh_bodies
                + num_dexhand_lh_bodies
                + 1
                + rh_sum_rigid_body_count
                + lh_sum_rigid_body_count
                + (0 + (0 + self.dexhand_lh.n_bodies * 2 if not self.headless else 0))
            )  # 1 for table
            max_agg_shapes = (
                num_dexhand_rh_shapes
                + num_dexhand_lh_shapes
                + 1
                + rh_sum_rigid_shape_count
                + lh_sum_rigid_shape_count
                + (0 + (0 + self.dexhand_lh.n_bodies * 2 if not self.headless else 0))
            )
            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: dexhand_r should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # camera handler for view rendering
            if self.camera_handlers is not None:
                self.camera_handlers.append(
                    self.create_camera(
                        env=env_ptr,
                        isaac_gym=self.gym,
                    )
                )
            if self.camera_handlers_top is not None:
                self.camera_handlers_top.append(
                    self.create_camera_top(env=env_ptr, isaac_gym=self.gym)
                )

            # Create dexhand_r
            dexhand_rh_actor = self.gym.create_actor(
                env_ptr,
                dexhand_rh_asset,
                self.dexhand_rh_pose,
                "dexhand_r",
                i,
                (1 if self.dexhand_rh.self_collision else 0),
            )
            dexhand_lh_actor = self.gym.create_actor(
                env_ptr,
                dexhand_lh_asset,
                self.dexhand_lh_pose,
                "dexhand_l",
                i,
                (1 if self.dexhand_lh.self_collision else 0),
            )
            self.gym.enable_actor_dof_force_sensors(env_ptr, dexhand_rh_actor)
            self.gym.enable_actor_dof_force_sensors(env_ptr, dexhand_lh_actor)
            self.gym.set_actor_dof_properties(env_ptr, dexhand_rh_actor, dexhand_rh_dof_props)
            self.gym.set_actor_dof_properties(env_ptr, dexhand_lh_actor, dexhand_lh_dof_props)

            # Create table and obstacles
            table_pose = gymapi.Transform()
            table_pose.p = gymapi.Vec3(table_pos.x, table_pos.y, table_pos.z)
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, 0)
            table_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_actor)
            table_props[0].friction = 0.1  # ? only one table shape in each env
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_actor, table_props)
            # set table's color to be dark gray
            self.gym.set_rigid_body_color(env_ptr, table_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.1, 0.1, 0.1))

            self.obj_rh_handle, _ = self._create_obj_actor(
                env_ptr, i, rh_current_asset, side="rh"
            )  # the handle is all the same for all envs
            self.obj_lh_handle, _ = self._create_obj_actor(env_ptr, i, lh_current_asset, side="lh")
            self.gym.set_actor_scale(env_ptr, self.obj_rh_handle, rh_obj_scale)
            self.gym.set_actor_scale(env_ptr, self.obj_lh_handle, lh_obj_scale)
            obj_props_rh = self.gym.get_actor_rigid_body_properties(env_ptr, self.obj_rh_handle)
            obj_props_lh = self.gym.get_actor_rigid_body_properties(env_ptr, self.obj_lh_handle)
            obj_props_rh[0].mass = min(0.5, obj_props_rh[0].mass)  # * we only consider the mass less than 500g
            obj_props_lh[0].mass = min(0.5, obj_props_lh[0].mass)  # * we only consider the mass less than 500g

            if rh_obj_mass is not None:
                obj_props_rh[0].mass = rh_obj_mass
            if lh_obj_mass is not None:
                obj_props_lh[0].mass = lh_obj_mass

            # ! Updating the mass and scale might slightly alter the inertia tensor;
            # ! however, because the magnitude of our modifications is minimal, we temporarily neglect this effect.
            self.gym.set_actor_rigid_body_properties(env_ptr, self.obj_rh_handle, obj_props_rh)
            self.gym.set_actor_rigid_body_properties(env_ptr, self.obj_lh_handle, obj_props_lh)
            self.manip_obj_rh_mass.append(obj_props_rh[0].mass)
            self.manip_obj_rh_com.append(
                torch.tensor([obj_props_rh[0].com.x, obj_props_rh[0].com.y, obj_props_rh[0].com.z])
            )
            self.manip_obj_lh_mass.append(obj_props_lh[0].mass)
            self.manip_obj_lh_com.append(
                torch.tensor([obj_props_lh[0].com.x, obj_props_lh[0].com.y, obj_props_lh[0].com.z])
            )

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.dexhand_rs.append(dexhand_rh_actor)
            self.dexhand_ls.append(dexhand_lh_actor)

        self.manip_obj_rh_mass = torch.tensor(self.manip_obj_rh_mass, device=self.device)
        self.manip_obj_rh_com = torch.stack(self.manip_obj_rh_com, dim=0).to(self.device)
        self.manip_obj_lh_mass = torch.tensor(self.manip_obj_lh_mass, device=self.device)
        self.manip_obj_lh_com = torch.stack(self.manip_obj_lh_com, dim=0).to(self.device)

        # Setup data
        self.init_data()

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        dexhand_rh_handle = self.gym.find_actor_handle(env_ptr, "dexhand_r")
        dexhand_lh_handle = self.gym.find_actor_handle(env_ptr, "dexhand_l")
        self.dexhand_rh_handles = {
            k: self.gym.find_actor_rigid_body_handle(env_ptr, dexhand_rh_handle, k) for k in self.dexhand_rh.body_names
        }
        self.dexhand_lh_handles = {
            k: self.gym.find_actor_rigid_body_handle(env_ptr, dexhand_lh_handle, k) for k in self.dexhand_lh.body_names
        }
        self.dexhand_rh_cf_weights = {
            k: (1.0 if ("intermediate" in k or "distal" in k) else 0.0) for k in self.dexhand_rh.body_names
        }
        self.dexhand_lh_cf_weights = {
            k: (1.0 if ("intermediate" in k or "distal" in k) else 0.0) for k in self.dexhand_lh.body_names
        }
        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        _net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
        _dof_force = self.gym.acquire_dof_force_tensor(self.sim)

        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._rh_base_state = self._root_state[:, 0, :]
        self._lh_base_state = self._root_state[:, 1, :]

        # ? >>> for visualization
        if not self.headless:

            self.mano_joint_rh_points = [
                self._root_state[:, self.gym.find_actor_handle(env_ptr, f"rh_mano_joint_{i}"), :]
                for i in range(self.dexhand_rh.n_bodies)
            ]
            self.mano_joint_lh_points = [
                self._root_state[:, self.gym.find_actor_handle(env_ptr, f"lh_mano_joint_{i}"), :]
                for i in range(self.dexhand_lh.n_bodies)
            ]
        # ? <<<

        self._manip_obj_rh_handle = self.gym.find_actor_handle(env_ptr, "manip_obj_rh")
        self._manip_obj_rh_root_state = self._root_state[:, self._manip_obj_rh_handle, :]
        self._manip_obj_lh_handle = self.gym.find_actor_handle(env_ptr, "manip_obj_lh")
        self._manip_obj_lh_root_state = self._root_state[:, self._manip_obj_lh_handle, :]

        self.net_cf = gymtorch.wrap_tensor(_net_cf).view(self.num_envs, -1, 3)
        self.dof_force = gymtorch.wrap_tensor(_dof_force).view(self.num_envs, -1)
        self._manip_obj_rh_rigid_body_handle = self.gym.find_actor_rigid_body_handle(
            env_ptr, self._manip_obj_rh_handle, "base"
        )
        self._manip_obj_lh_rigid_body_handle = self.gym.find_actor_rigid_body_handle(
            env_ptr, self._manip_obj_lh_handle, "base"
        )
        self._manip_obj_rh_cf = self.net_cf[:, self._manip_obj_rh_rigid_body_handle, :]
        self._manip_obj_lh_cf = self.net_cf[:, self._manip_obj_lh_rigid_body_handle, :]

        self.dexhand_rh_root_state = self._root_state[:, dexhand_rh_handle, :]
        self.dexhand_lh_root_state = self._root_state[:, dexhand_lh_handle, :]

        self.apply_forces = torch.zeros(
            (self.num_envs, self._rigid_body_state.shape[1], 3), device=self.device, dtype=torch.float
        )
        self.apply_torque = torch.zeros(
            (self.num_envs, self._rigid_body_state.shape[1], 3), device=self.device, dtype=torch.float
        )
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.curr_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        if self.use_pid_control:
            self.rh_prev_pos_error = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.rh_prev_rot_error = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.rh_pos_error_integral = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.rh_rot_error_integral = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.lh_prev_pos_error = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.lh_prev_rot_error = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.lh_pos_error_integral = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.lh_rot_error_integral = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize indices
        self._global_dexhand_rh_indices = torch.tensor(
            [self.gym.find_actor_index(env, "dexhand_r", gymapi.DOMAIN_SIM) for env in self.envs],
            dtype=torch.int32,
            device=self.sim_device,
        ).view(self.num_envs, -1)
        self._global_dexhand_lh_indices = torch.tensor(
            [self.gym.find_actor_index(env, "dexhand_l", gymapi.DOMAIN_SIM) for env in self.envs],
            dtype=torch.int32,
            device=self.sim_device,
        ).view(self.num_envs, -1)

        self._global_manip_obj_rh_indices = torch.tensor(
            [self.gym.find_actor_index(env, "manip_obj_rh", gymapi.DOMAIN_SIM) for env in self.envs],
            dtype=torch.int32,
            device=self.sim_device,
        ).view(self.num_envs, -1)
        self._global_manip_obj_lh_indices = torch.tensor(
            [self.gym.find_actor_index(env, "manip_obj_lh", gymapi.DOMAIN_SIM) for env in self.envs],
            dtype=torch.int32,
            device=self.sim_device,
        ).view(self.num_envs, -1)

        CONTACT_HISTORY_LEN = 3
        self.rh_tips_contact_history = torch.ones(self.num_envs, CONTACT_HISTORY_LEN, 5, device=self.device).bool()
        self.lh_tips_contact_history = torch.ones(self.num_envs, CONTACT_HISTORY_LEN, 5, device=self.device).bool()

    @staticmethod
    def _sample_aug_transform(device, center):
        angle = (torch.rand(1).item() * 2 - 1) * (30.0 * np.pi / 180.0)
        cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
        R = torch.tensor([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]],
                         dtype=torch.float32, device=device)
        t = (torch.rand(2, device=device) * 2 - 1) * torch.tensor([0.05, 0.05], device=device)
        t = torch.cat([t, torch.zeros(1, device=device)])
        return R, t, center

    @staticmethod
    def _aug_demo(data, R, t, noise_std=0.0, center=None):
        """Return a shallow-copied demo dict with (R, t) applied to all world-space fields.
        Rotation is around `center` (defaults to world origin if None).
        In this case, the center is the center of the table
        """
        from copy import copy
        d = copy(data)

        def rp(x):   # rotate around center (table center point) + translate position [T, 3]
            if center is not None:
                return (R @ (x - center).T).T + center + t
            return (R @ x.T).T + t

        def rv(x):   # rotate velocity / angular velocity [T, 3]
            return (R @ x.T).T

        def raa(x):  # rotate axis-angle [T, 3]
            return rotmat_to_aa(R.unsqueeze(0) @ aa_to_rotmat(x))

        # Positions
        # print("NOISE STD: ", noise_std)
        d["wrist_pos"] = rp(data["wrist_pos"]) + torch.randn_like(d["wrist_pos"]) * noise_std
        d["opt_wrist_pos"] = rp(data["opt_wrist_pos"]) + torch.randn_like(d["opt_wrist_pos"]) * noise_std
        d["mano_joints"] = {
            k: rp(v) + (torch.randn_like(v) * noise_std)
            for k, v in data["mano_joints"].items()
        }

        # Object trajectory [T, 4, 4]
        obj = data["obj_trajectory"].clone()
        obj[:, :3, 3] = rp(obj[:, :3, 3])
        obj[:, :3, :3] = R.unsqueeze(0) @ obj[:, :3, :3]
        d["obj_trajectory"] = obj

        # Rotations (axis-angle)
        d["wrist_rot"] = raa(data["wrist_rot"])
        d["opt_wrist_rot"] = raa(data["opt_wrist_rot"])

        # Velocities (rotate only, translation doesn't affect velocity)
        d["wrist_velocity"] = rv(data["wrist_velocity"])
        d["wrist_angular_velocity"] = rv(data["wrist_angular_velocity"])
        d["obj_velocity"] = rv(data["obj_velocity"])
        d["obj_angular_velocity"] = rv(data["obj_angular_velocity"])
        d["opt_wrist_velocity"] = rv(data["opt_wrist_velocity"])
        d["opt_wrist_angular_velocity"] = rv(data["opt_wrist_angular_velocity"])
        d["mano_joints_velocity"] = {k: rv(v) for k, v in data["mano_joints_velocity"].items()}

        # tips_distance: scalar magnitude, rotation-invariant — no change
        # opt_dof_pos / opt_dof_velocity: joint angles — no change
        # obj_verts / bps: shape encoding, rotation-invariant — no change
        return d

    @staticmethod
    def _aug_demo_lh_obj_center(data_rh, data_lh, R, noise_std=0.0):
        """Rotate only the RH demo around the LH object center at each timestep.

        At each timestep t:
            p_rh_aug_t = R @ (p_rh_t - c_t) + c_t
        where c_t = LH object position at frame t.

        LH demo is unchanged.
        """
        from copy import copy
        d_rh = copy(data_rh)

        c_t   = data_lh["obj_trajectory"][:, :3, 3]  # [T, 3] — LH object center
        c_dot = data_lh["obj_velocity"]               # [T, 3] — LH object velocity (ċ)

        def rp(x):   # [T, 3]
            return (R @ (x - c_t).T).T + c_t

        def rv(x):
            # correct velocity for moving center: d/dt(R(p-c)+c) = R(ṗ-ċ)+ċ
            return (R @ (x - c_dot).T).T + c_dot        # can also just recalculate the velocity after performing the rotation on the points

        def raa(x):
            return rotmat_to_aa(R.unsqueeze(0) @ aa_to_rotmat(x))

        d_rh["wrist_pos"] = rp(d_rh["wrist_pos"]) + torch.randn_like(d_rh["wrist_pos"]) * noise_std
        d_rh["opt_wrist_pos"] = rp(d_rh["opt_wrist_pos"]) + torch.randn_like(d_rh["opt_wrist_pos"]) * noise_std
        d_rh["mano_joints"] = {
            k: rp(v) + (torch.randn_like(v) * noise_std)
            for k, v in d_rh["mano_joints"].items()
        }
        obj_rh = data_rh["obj_trajectory"].clone()
        obj_rh[:, :3, 3] = rp(obj_rh[:, :3, 3])
        obj_rh[:, :3, :3] = R.unsqueeze(0) @ obj_rh[:, :3, :3]
        d_rh["obj_trajectory"] = obj_rh
        d_rh["wrist_rot"] = raa(data_rh["wrist_rot"])
        d_rh["opt_wrist_rot"] = raa(data_rh["opt_wrist_rot"])
        d_rh["wrist_velocity"] = rv(data_rh["wrist_velocity"])
        d_rh["wrist_angular_velocity"] = rv(data_rh["wrist_angular_velocity"])
        d_rh["obj_velocity"] = rv(data_rh["obj_velocity"])
        d_rh["obj_angular_velocity"] = rv(data_rh["obj_angular_velocity"])
        d_rh["opt_wrist_velocity"] = rv(data_rh["opt_wrist_velocity"])
        d_rh["opt_wrist_angular_velocity"] = rv(data_rh["opt_wrist_angular_velocity"])
        d_rh["mano_joints_velocity"] = {k: rv(v) for k, v in data_rh["mano_joints_velocity"].items()}

        return d_rh, data_lh  # LH unchanged

    @staticmethod
    def _aug_demo_rh_obj_center(data_rh, R, noise_std=0.0):
        """Rotate only the RH demo around the RH object center at each timestep.

        At each timestep t:
            p_rh_aug_t = R @ (p_lh_t - c_t) + c_t
        where c_t = RH object position at frame t.

        """
        def rp(x):   # [T, 3]
            return (R @ (x - c_t).T).T + c_t

        def rv(x):
            # correct velocity for moving center: d/dt(R(p-c)+c) = R(ṗ-ċ)+ċ
            return (R @ (x - c_dot).T).T + c_dot        # can also just recalculate the velocity after performing the rotation on the points

        def raa(x):
            return rotmat_to_aa(R.unsqueeze(0) @ aa_to_rotmat(x))
        
        from copy import copy
        d_rh = copy(data_rh)

        c_t = data_rh["obj_trajectory"][:, :3, 3]  # [T, 3] - RH object center
        c_dot = data_rh["obj_velocity"]             # [T, 3] - RH object velocity

        d_rh["wrist_pos"] = rp(d_rh["wrist_pos"]) + torch.randn_like(d_rh["wrist_pos"]) * noise_std
        d_rh["opt_wrist_pos"] = rp(d_rh["opt_wrist_pos"]) + torch.randn_like(d_rh["opt_wrist_pos"]) * noise_std
        d_rh["mano_joints"] = {
            k: rp(v) + (torch.randn_like(v) * noise_std)
            for k, v in d_rh["mano_joints"].items()
        }
        obj_rh = data_rh["obj_trajectory"].clone()
        obj_rh[:, :3, 3] = rp(obj_rh[:, :3, 3])
        obj_rh[:, :3, :3] = R.unsqueeze(0) @ obj_rh[:, :3, :3]
        d_rh["obj_trajectory"] = obj_rh
        d_rh["wrist_rot"] = raa(data_rh["wrist_rot"])
        d_rh["opt_wrist_rot"] = raa(data_rh["opt_wrist_rot"])
        d_rh["wrist_velocity"] = rv(data_rh["wrist_velocity"])
        d_rh["wrist_angular_velocity"] = rv(data_rh["wrist_angular_velocity"])
        d_rh["obj_velocity"] = rv(data_rh["obj_velocity"])
        d_rh["obj_angular_velocity"] = rv(data_rh["obj_angular_velocity"])
        d_rh["opt_wrist_velocity"] = rv(data_rh["opt_wrist_velocity"])
        d_rh["opt_wrist_angular_velocity"] = rv(data_rh["opt_wrist_angular_velocity"])
        d_rh["mano_joints_velocity"] = {k: rv(v) for k, v in data_rh["mano_joints_velocity"].items()}

        return d_rh

    @staticmethod
    def _aug_demo_lh_about_lh_obj(data_lh, R, noise_std=0.0):
        """Rigidly rotate ONLY the LH demo (left hand + the left object it holds) about the
        LH object center at each timestep. RH demo is left unchanged.

        Pivot c_t = LH object position at frame t, so at each timestep:
            p_lh_aug_t = R @ (p_lh_t - c_t) + c_t     # wrist / joints orbit the object
            obj_pos_t  = c_t   (sits at the pivot -> position unchanged)
            obj_rot_t  = R @ obj_rot_t                # object spins in place with the hand
        The hand<->object relative grasp is preserved (both rotate by R about c_t). This is the
        LH counterpart of _aug_demo_lh_obj_center / _aug_demo_rh_obj_center; velocity handling
        mirrors them (linear velocity corrected for the moving center).
        """
        from copy import copy
        d_lh = copy(data_lh)

        c_t   = data_lh["obj_trajectory"][:, :3, 3]  # [T, 3] — LH object center (pivot)
        c_dot = data_lh["obj_velocity"]               # [T, 3] — LH object velocity (ċ)

        def rp(x):   # [T, 3]
            return (R @ (x - c_t).T).T + c_t

        def rv(x):
            # correct velocity for moving center: d/dt(R(p-c)+c) = R(ṗ-ċ)+ċ
            return (R @ (x - c_dot).T).T + c_dot

        def raa(x):
            return rotmat_to_aa(R.unsqueeze(0) @ aa_to_rotmat(x))

        d_lh["wrist_pos"] = rp(d_lh["wrist_pos"]) + torch.randn_like(d_lh["wrist_pos"]) * noise_std
        d_lh["opt_wrist_pos"] = rp(d_lh["opt_wrist_pos"]) + torch.randn_like(d_lh["opt_wrist_pos"]) * noise_std
        d_lh["mano_joints"] = {
            k: rp(v) + (torch.randn_like(v) * noise_std)
            for k, v in d_lh["mano_joints"].items()
        }
        obj_lh = data_lh["obj_trajectory"].clone()
        obj_lh[:, :3, 3] = rp(obj_lh[:, :3, 3])           # at the pivot -> position unchanged
        obj_lh[:, :3, :3] = R.unsqueeze(0) @ obj_lh[:, :3, :3]
        d_lh["obj_trajectory"] = obj_lh
        d_lh["wrist_rot"] = raa(data_lh["wrist_rot"])
        d_lh["opt_wrist_rot"] = raa(data_lh["opt_wrist_rot"])
        d_lh["wrist_velocity"] = rv(data_lh["wrist_velocity"])
        d_lh["wrist_angular_velocity"] = rv(data_lh["wrist_angular_velocity"])
        d_lh["obj_velocity"] = rv(data_lh["obj_velocity"])
        d_lh["obj_angular_velocity"] = rv(data_lh["obj_angular_velocity"])
        d_lh["opt_wrist_velocity"] = rv(data_lh["opt_wrist_velocity"])
        d_lh["opt_wrist_angular_velocity"] = rv(data_lh["opt_wrist_angular_velocity"])
        d_lh["mano_joints_velocity"] = {k: rv(v) for k, v in data_lh["mano_joints_velocity"].items()}

        return d_lh


    def pack_data(self, data, side="rh"):
        packed_data = {}
        packed_data["seq_len"] = torch.tensor([len(d["obj_trajectory"]) for d in data], device=self.device)
        max_len = packed_data["seq_len"].max()
        assert max_len <= self.max_episode_length, "max_len should be less than max_episode_length"

        def fill_data(stack_data):
            for i in range(len(stack_data)):
                if len(stack_data[i]) < max_len:
                    stack_data[i] = torch.cat(
                        [
                            stack_data[i],
                            stack_data[i][-1]
                            .unsqueeze(0)
                            .repeat(max_len - len(stack_data[i]), *[1 for _ in stack_data[i].shape[1:]]),
                        ],
                        dim=0,
                    )
            return torch.stack(stack_data).squeeze()

        for k in data[0].keys():
            if k == "mano_joints" or k == "mano_joints_velocity":
                mano_joints = []
                for d in data:
                    if side == "rh":
                        mano_joints.append(
                            torch.concat(
                                [
                                    d[k][self.dexhand_rh.to_hand(j_name)[0]]
                                    for j_name in self.dexhand_rh.body_names
                                    if self.dexhand_rh.to_hand(j_name)[0] != "wrist"
                                ],
                                dim=-1,
                            )
                        )
                    else:
                        mano_joints.append(
                            torch.concat(
                                [
                                    d[k][self.dexhand_lh.to_hand(j_name)[0]]
                                    for j_name in self.dexhand_lh.body_names
                                    if self.dexhand_lh.to_hand(j_name)[0] != "wrist"
                                ],
                                dim=-1,
                            )
                        )
                packed_data[k] = fill_data(mano_joints)
            elif type(data[0][k]) == torch.Tensor:
                stack_data = [d[k] for d in data]
                if k != "obj_verts":
                    packed_data[k] = fill_data(stack_data)
                else:
                    packed_data[k] = torch.stack(stack_data).squeeze()
            elif type(data[0][k]) == np.ndarray:
                raise RuntimeError("Using np is very slow.")
            else:
                packed_data[k] = [d[k] for d in data]
        return packed_data

    def allocate_buffers(self):
        # will also allocate extra buffers for data dumping, used for distillation
        super().allocate_buffers()

        # basic prop fields
        if not self.training:
            self.dump_fileds = {
                k: torch.zeros(
                    (self.num_envs, v),
                    device=self.device,
                    dtype=torch.float,
                )
                for k, v in self._prop_dump_info.items()
            }

    def _create_obj_assets(self, i, side="rh"):
        if side == "rh":
            obj_id = self.demo_data_rh["obj_id"][i]
        else:
            obj_id = self.demo_data_lh["obj_id"][i]

        if obj_id in self.objs_assets:
            current_asset = self.objs_assets[obj_id]
        else:
            asset_options = gymapi.AssetOptions()
            asset_options.override_com = True
            asset_options.override_inertia = True
            asset_options.convex_decomposition_from_submeshes = True
            asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
            asset_options.thickness = 0.001
            asset_options.max_linear_velocity = 50
            asset_options.max_angular_velocity = 100
            asset_options.fix_base_link = False
            asset_options.vhacd_enabled = True
            asset_options.vhacd_params = gymapi.VhacdParams()
            asset_options.vhacd_params.resolution = 200000
            asset_options.density = 200  # * the average density of low-fill-rate 3D-printed models
            if side == "rh":
                obj_urdf_path = self.demo_data_rh["obj_urdf_path"][i]
            else:
                obj_urdf_path = self.demo_data_lh["obj_urdf_path"][i]
            current_asset = self.gym.load_asset(self.sim, *os.path.split(obj_urdf_path), asset_options)

            rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(current_asset)
            for element in rigid_shape_props_asset:
                element.friction = 2.0  # * We increase the friction coefficient to compensate for missing skin deformation friction in simulation. See the Appx for details.
                element.rolling_friction = 0.05
                element.torsion_friction = 0.05
            self.gym.set_asset_rigid_shape_properties(current_asset, rigid_shape_props_asset)
            self.objs_assets[obj_id] = current_asset

        # * load assigned scale and mass for the object if available
        if obj_id in oakink2_obj_scale:
            scale = oakink2_obj_scale[obj_id]
        else:
            scale = 1.0

        if obj_id in oakink2_obj_mass:
            mass = oakink2_obj_mass[obj_id]
        else:
            mass = None

        sum_rigid_body_count = self.gym.get_asset_rigid_body_count(current_asset)
        sum_rigid_shape_count = self.gym.get_asset_rigid_shape_count(current_asset)
        return current_asset, sum_rigid_body_count, sum_rigid_shape_count, scale, mass

    def _create_obj_actor(self, env_ptr, i, current_asset, side="rh"):

        if side == "rh":
            obj_transf = self.demo_data_rh["obj_trajectory"][i][0]
        else:
            obj_transf = self.demo_data_lh["obj_trajectory"][i][0]

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(obj_transf[0, 3], obj_transf[1, 3], obj_transf[2, 3])
        obj_aa = rotmat_to_aa(obj_transf[:3, :3])
        obj_aa_angle = torch.norm(obj_aa)
        obj_aa_axis = obj_aa / obj_aa_angle
        pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(obj_aa_axis[0], obj_aa_axis[1], obj_aa_axis[2]), obj_aa_angle)

        # ? object actor filter bit is always 1
        if side == "rh":
            obj_actor = self.gym.create_actor(env_ptr, current_asset, pose, "manip_obj_rh", i, 0)
        else:
            obj_actor = self.gym.create_actor(env_ptr, current_asset, pose, "manip_obj_lh", i, 0)
        obj_index = self.gym.get_actor_index(env_ptr, obj_actor, gymapi.DOMAIN_SIM)

        if side == "rh":
            scene_objs = self.demo_data_rh["scene_objs"][i]
        else:
            scene_objs = self.demo_data_lh["scene_objs"][i]
        scene_asset_options = gymapi.AssetOptions()
        scene_asset_options.fix_base_link = True

        for so_id, scene_obj in enumerate(scene_objs):
            scene_obj_type = scene_obj["obj"].type
            scene_obj_size = scene_obj["obj"].size
            scene_obj_pose = scene_obj["pose"]
            if scene_obj_type == "cube":
                scene_asset = self.gym.create_box(
                    self.sim,
                    scene_obj_size[0],
                    scene_obj_size[1],
                    scene_obj_size[2],
                    scene_asset_options,
                )
                offset = np.eye(4)
                offset[:3, 3] = np.array(scene_obj_size) / 2
                scene_obj_pose = scene_obj_pose @ offset
            elif scene_obj_type == "cylinder":
                scene_asset = self.gym.create_box(
                    self.sim,
                    scene_obj_size[0] * 2,
                    scene_obj_size[0] * 2,
                    scene_obj_size[1],
                    scene_asset_options,
                )
            else:
                raise NotImplementedError
            scene_obj_pose = self.mujoco2gym_transf @ torch.tensor(
                scene_obj_pose, device=self.sim_device, dtype=torch.float32
            )
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(scene_obj_pose[0, 3], scene_obj_pose[1, 3], scene_obj_pose[2, 3])
            obj_aa = rotmat_to_aa(scene_obj_pose[:3, :3])
            obj_aa_angle = torch.norm(obj_aa)
            obj_aa_axis = obj_aa / obj_aa_angle
            pose.r = gymapi.Quat.from_axis_angle(
                gymapi.Vec3(obj_aa_axis[0], obj_aa_axis[1], obj_aa_axis[2]), obj_aa_angle
            )
            self.gym.create_actor(env_ptr, scene_asset, pose, f"scene_obj_{so_id}", i, 0)
        # add dummy scene object
        MAX_SCENE_OBJS = 0 + (0 if not self.headless else 0)
        for so_id in range(MAX_SCENE_OBJS - len(scene_objs)):
            scene_asset = self.gym.create_box(self.sim, 0.02, 0.04, 0.06, scene_asset_options)
            # ? collision filter bit is always 0b11111111, never collide with anything (except the ground)
            a = self.gym.create_actor(
                env_ptr,
                scene_asset,
                gymapi.Transform(),
                f"scene_obj_{so_id +  len(scene_objs)}",
                self.num_envs + 1,
                0b1,
            )
            c = [
                gymapi.Vec3(1, 1, 0.5),
                gymapi.Vec3(0.5, 1, 1),
                gymapi.Vec3(1, 0, 1),
                gymapi.Vec3(1, 1, 0),
                gymapi.Vec3(0, 1, 1),
                gymapi.Vec3(0, 0, 1),
                gymapi.Vec3(0, 1, 0),
                gymapi.Vec3(1, 0, 0),
            ][so_id + len(scene_objs)]
            self.gym.set_rigid_body_color(env_ptr, a, 0, gymapi.MESH_VISUAL, c)

        # * just for visualization purposes, add a small sphere at the finger positions
        if not self.headless:
            dexhand_template = self.dexhand_rh if side == "rh" else self.dexhand_lh
            for joint_vis_id, joint_name in enumerate(dexhand_template.body_names):
                joint_name = dexhand_template.to_hand(joint_name)[0]
                joint_point = self.gym.create_sphere(self.sim, 0.005, scene_asset_options)
                a = self.gym.create_actor(
                    env_ptr,
                    joint_point,
                    gymapi.Transform(),
                    f"{side}_mano_joint_{joint_vis_id}",
                    self.num_envs + 1,
                    0b1,
                )
                if "index" in joint_name:
                    inter_c = 70
                elif "middle" in joint_name:
                    inter_c = 130
                elif "ring" in joint_name:
                    inter_c = 190
                elif "pinky" in joint_name:
                    inter_c = 250
                elif "thumb" in joint_name:
                    inter_c = 10
                else:
                    inter_c = 0
                if "tip" in joint_name:
                    c = gymapi.Vec3(inter_c / 255, 200 / 255, 200 / 255)
                elif "proximal" in joint_name:
                    c = gymapi.Vec3(200 / 255, inter_c / 255, 200 / 255)
                elif "intermediate" in joint_name:
                    c = gymapi.Vec3(200 / 255, 200 / 255, inter_c / 255)
                else:
                    c = gymapi.Vec3(100 / 255, 150 / 255, 200 / 255)
                self.gym.set_rigid_body_color(env_ptr, a, 0, gymapi.MESH_VISUAL, c)

        return obj_actor, obj_index

    def _update_states(self):
        self.rh_states.update(
            {
                "q": self._q[:, : self.num_dexhand_rh_dofs],
                "cos_q": torch.cos(self._q[:, : self.num_dexhand_rh_dofs]),
                "sin_q": torch.sin(self._q[:, : self.num_dexhand_rh_dofs]),
                "dq": self._qd[:, : self.num_dexhand_rh_dofs],
                "base_state": self._rh_base_state[:, :],
            }
        )

        self.rh_states["joints_state"] = torch.stack(
            [self._rigid_body_state[:, self.dexhand_rh_handles[k], :][:, :10] for k in self.dexhand_rh.body_names],
            dim=1,
        )
        self.rh_states.update(
            {
                "manip_obj_pos": self._manip_obj_rh_root_state[:, :3],
                "manip_obj_quat": self._manip_obj_rh_root_state[:, 3:7],
                "manip_obj_vel": self._manip_obj_rh_root_state[:, 7:10],
                "manip_obj_ang_vel": self._manip_obj_rh_root_state[:, 10:],
            }
        )

        self.lh_states.update(
            {
                "q": self._q[:, self.num_dexhand_rh_dofs :],
                "cos_q": torch.cos(self._q[:, self.num_dexhand_rh_dofs :]),
                "sin_q": torch.sin(self._q[:, self.num_dexhand_rh_dofs :]),
                "dq": self._qd[:, self.num_dexhand_rh_dofs :],
                "base_state": self._lh_base_state[:, :],
            }
        )
        self.lh_states["joints_state"] = torch.stack(
            [self._rigid_body_state[:, self.dexhand_lh_handles[k], :][:, :10] for k in self.dexhand_lh.body_names],
            dim=1,
        )
        self.lh_states.update(
            {
                "manip_obj_pos": self._manip_obj_lh_root_state[:, :3],
                "manip_obj_quat": self._manip_obj_lh_root_state[:, 3:7],
                "manip_obj_vel": self._manip_obj_lh_root_state[:, 7:10],
                "manip_obj_ang_vel": self._manip_obj_lh_root_state[:, 10:],
            }
        )

    def _refresh(self):

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # Refresh states
        self._update_states()

    def compute_reward(self, actions):
        lh_rew_buf, lh_reset_buf, lh_success_buf, lh_failure_buf, lh_reward_dict, lh_error_buf = (
            self.compute_reward_side(actions, side="lh")
        )
        rh_rew_buf, rh_reset_buf, rh_success_buf, rh_failure_buf, rh_reward_dict, rh_error_buf = (
            self.compute_reward_side(actions, side="rh")
        )
        self.rew_buf = rh_rew_buf + lh_rew_buf
        self.reset_buf = rh_reset_buf | lh_reset_buf
        self.success_buf = rh_success_buf & lh_success_buf
        self.failure_buf = rh_failure_buf | lh_failure_buf
        self.error_buf = rh_error_buf | lh_error_buf

        rh_pos  = self._manip_obj_rh_root_state[:, :3]
        rh_quat = self._manip_obj_rh_root_state[:, 3:7]
        lh_pos  = self._manip_obj_lh_root_state[:, :3]
        lh_quat = self._manip_obj_lh_root_state[:, 3:7]

        if self.use_pen_keypoint_reward:
            pen_tip  = rh_pos + torch_jit_utils.quat_rotate(rh_quat, self._pen_tip_offset.expand(self.num_envs, -1))
            cap_open = lh_pos + torch_jit_utils.quat_rotate(lh_quat, self._cap_open_offset.expand(self.num_envs, -1))
            dist_pen_cap = torch.norm(pen_tip - cap_open, dim=-1)
            reward_pen_keypoint = torch.exp(-80 * dist_pen_cap)
            self.rew_buf = self.rew_buf + 2.0 * reward_pen_keypoint
            rh_reward_dict["reward_pen_keypoint"] = reward_pen_keypoint
            lh_reward_dict["reward_pen_keypoint"] = reward_pen_keypoint

        if self.use_coaxial_reward:
            # coaxial alignment: pen Z-axis (long axis) should be anti-parallel to cap Z-axis
            # only active when objects are within 0.05m of each other
            dist_centers = torch.norm(rh_pos - lh_pos, dim=-1)
            # distance from the pen tip to the mesh origin is 6.9cm, and the 
            # distance from the cap opening to the origin is 3.9cm
            # min distance between the two is 2cm
            near = (dist_centers < 0.14).float() 
            rh_z = torch_jit_utils.quat_rotate(rh_quat, self._z_axis.expand(self.num_envs, -1))
            lh_z = torch_jit_utils.quat_rotate(lh_quat, self._z_axis.expand(self.num_envs, -1))
            cos_align = torch.sum(rh_z * (-lh_z), dim=-1).clamp(-1.0, 1.0)
            reward_coax = (cos_align + 1.0) * 0.5 * near
            self.rew_buf = self.rew_buf + 3.0 * reward_coax
            rh_reward_dict["reward_coax"] = reward_coax
            lh_reward_dict["reward_coax"] = reward_coax

        self.reward_dict = {
            **{"rh_" + k: v for k, v in rh_reward_dict.items()},
            **{"lh_" + k: v for k, v in lh_reward_dict.items()},
        }

    def compute_reward_side(self, actions, side="rh"):
        side_demo_data = self.demo_data_rh if side == "rh" else self.demo_data_lh
        target_state = {}
        max_length = torch.clip(side_demo_data["seq_len"], 0, self.max_episode_length).float()
        cur_idx = self.progress_buf
        cur_wrist_pos = side_demo_data["wrist_pos"][torch.arange(self.num_envs), cur_idx]
        target_state["wrist_pos"] = cur_wrist_pos
        cur_wrist_rot = side_demo_data["wrist_rot"][torch.arange(self.num_envs), cur_idx]
        target_state["wrist_quat"] = aa_to_quat(cur_wrist_rot)[:, [1, 2, 3, 0]]

        target_state["wrist_vel"] = side_demo_data["wrist_velocity"][torch.arange(self.num_envs), cur_idx]
        target_state["wrist_ang_vel"] = side_demo_data["wrist_angular_velocity"][torch.arange(self.num_envs), cur_idx]

        target_state["tips_distance"] = side_demo_data["tips_distance"][torch.arange(self.num_envs), cur_idx]

        cur_joints_pos = side_demo_data["mano_joints"][torch.arange(self.num_envs), cur_idx]
        target_state["joints_pos"] = cur_joints_pos.reshape(self.num_envs, -1, 3)
        target_state["joints_vel"] = side_demo_data["mano_joints_velocity"][
            torch.arange(self.num_envs), cur_idx
        ].reshape(self.num_envs, -1, 3)

        cur_obj_transf = side_demo_data["obj_trajectory"][torch.arange(self.num_envs), cur_idx]
        target_state["manip_obj_pos"] = cur_obj_transf[:, :3, 3]
        target_state["manip_obj_quat"] = rotmat_to_quat(cur_obj_transf[:, :3, :3])[:, [1, 2, 3, 0]]

        target_state["manip_obj_vel"] = side_demo_data["obj_velocity"][torch.arange(self.num_envs), cur_idx]
        target_state["manip_obj_ang_vel"] = side_demo_data["obj_angular_velocity"][torch.arange(self.num_envs), cur_idx]

        target_state["tip_force"] = torch.stack(
            [
                self.net_cf[:, getattr(self, f"dexhand_{side}_handles")[k], :]
                for k in (self.dexhand_rh.contact_body_names if side == "rh" else self.dexhand_lh.contact_body_names)
            ],
            axis=1,
        )
        setattr(
            self,
            f"{side}_tips_contact_history",
            torch.concat(
                [
                    getattr(self, f"{side}_tips_contact_history")[:, 1:],
                    (torch.norm(target_state["tip_force"], dim=-1) > 0)[:, None],
                ],
                dim=1,
            ),
        )
        target_state["tip_contact_state"] = getattr(self, f"{side}_tips_contact_history")

        side_states = getattr(self, f"{side}_states")
        if side == "rh":
            power = torch.abs(torch.multiply(self.dof_force[:, : self.dexhand_rh.n_dofs], side_states["dq"])).sum(
                dim=-1
            )
        else:
            power = torch.abs(torch.multiply(self.dof_force[:, self.dexhand_rh.n_dofs :], side_states["dq"])).sum(
                dim=-1
            )
        target_state["power"] = power

        base_handle = getattr(self, f"dexhand_{side}_handles")[
            self.dexhand_rh.to_dex("wrist")[0] if side == "rh" else self.dexhand_lh.to_dex("wrist")[0]
        ]

        wrist_power = torch.abs(
            torch.sum(
                self.apply_forces[:, base_handle, :] * side_states["base_state"][:, 7:10],
                dim=-1,
            )
        )  # ? linear force * linear velocity
        wrist_power += torch.abs(
            torch.sum(
                self.apply_torque[:, base_handle, :] * side_states["base_state"][:, 10:],
                dim=-1,
            )
        )  # ? torque * angular velocity
        target_state["wrist_power"] = wrist_power

        if self.training:
            last_step = self.gym.get_frame_count(self.sim)
            if self.tighten_method == "None":
                scale_factor = 1.0
            elif self.tighten_method == "const":
                scale_factor = self.tighten_factor
            elif self.tighten_method == "linear_decay":
                scale_factor = 1 - (1 - self.tighten_factor) / self.tighten_steps * min(last_step, self.tighten_steps)
            elif self.tighten_method == "exp_decay":
                scale_factor = (np.e * 2) ** (-1 * last_step / self.tighten_steps) * (
                    1 - self.tighten_factor
                ) + self.tighten_factor
            elif self.tighten_method == "cos":
                scale_factor = (self.tighten_factor) + np.abs(
                    -1 * (1 - self.tighten_factor) * np.cos(last_step / self.tighten_steps * np.pi)
                ) * (2 ** (-1 * last_step / self.tighten_steps))
            else:
                raise NotImplementedError
        else:
            scale_factor = 1.0

        self.scale_factor = scale_factor

        assert not self.headless or isinstance(compute_imitation_reward, torch.jit.ScriptFunction)

        if self.rollout_len is not None:
            max_length = torch.clamp(max_length, 0, self.rollout_len + self.rollout_begin + 3 + 1)

        rew_buf, reset_buf, success_buf, failure_buf, reward_dict, error_buf = compute_imitation_reward(
            self.reset_buf,
            self.progress_buf,
            self.running_progress_buf,
            self.actions,
            side_states,
            target_state,
            max_length,
            scale_factor,
            self.failure_threshold_noise_compensation,
            (self.dexhand_rh if side == "rh" else self.dexhand_lh).weight_idx,
            self.training,
        )
        if not self.training and failure_buf[0].item():
            self._print_failure_reason(side, side_states, target_state, scale_factor, error_buf)
        self.total_rew_buf += rew_buf
        return rew_buf, reset_buf, success_buf, failure_buf, reward_dict, error_buf

    def _print_failure_reason(self, side, side_states, target_state, scale_factor, error_buf):
        idx = 0
        step = self.progress_buf[idx].item()

        cur_obj_pos  = side_states["manip_obj_pos"][idx]
        tgt_obj_pos  = target_state["manip_obj_pos"][idx]
        cur_obj_quat = side_states["manip_obj_quat"][idx]
        tgt_obj_quat = target_state["manip_obj_quat"][idx]
        cur_eef_pos  = side_states["base_state"][idx, :3]
        tgt_eef_pos  = target_state["wrist_pos"][idx]

        dexhand = self.dexhand_rh if side == "rh" else self.dexhand_lh
        widx = dexhand.weight_idx

        joints_pos = side_states["joints_state"][idx, 1:, :3]
        tgt_joints  = target_state["joints_pos"][idx]
        diff_j = torch.norm(tgt_joints - joints_pos, dim=-1)

        obj_pos_dist = torch.norm(tgt_obj_pos - cur_obj_pos).item()
        diff_obj_rot  = quat_mul(tgt_obj_quat.unsqueeze(0), quat_conjugate(cur_obj_quat.unsqueeze(0)))
        obj_rot_deg   = (quat_to_angle_axis(diff_obj_rot)[0].abs() / np.pi * 180).item()
        eef_pos_dist  = torch.norm(tgt_eef_pos - cur_eef_pos).item()

        def _mean_dist(keys):
            indices = [k - 1 for k in keys]
            return diff_j[indices].mean().item() if indices else 0.0

        thumb_dist  = _mean_dist(widx["thumb_tip"])
        index_dist  = _mean_dist(widx["index_tip"])
        middle_dist = _mean_dist(widx["middle_tip"])
        ring_dist   = _mean_dist(widx["ring_tip"])
        pinky_dist  = _mean_dist(widx["pinky_tip"])
        l1_dist     = _mean_dist(widx["level_1_joints"])
        l2_dist     = _mean_dist(widx["level_2_joints"])

        reasons = []
        if obj_pos_dist  > 0.03:    reasons.append(f"obj_pos={obj_pos_dist:.3f}m  (>0.030)")
        if obj_rot_deg   > 30:      reasons.append(f"obj_rot={obj_rot_deg:.1f}°   (>30°)")
        if thumb_dist    > 0.06:    reasons.append(f"thumb_tip={thumb_dist:.3f}m (>0.060)")
        if index_dist    > 0.06:    reasons.append(f"index_tip={index_dist:.3f}m (>0.060)")
        if middle_dist   > 0.06:    reasons.append(f"middle_tip={middle_dist:.3f}m (>0.060)")
        if ring_dist     > 0.06:    reasons.append(f"ring_tip={ring_dist:.3f}m (>0.060)")
        if pinky_dist    > 0.06:    reasons.append(f"pinky_tip={pinky_dist:.3f}m (>0.060)")
        if l1_dist       > 0.08:    reasons.append(f"level1={l1_dist:.3f}m (>0.080)")
        if l2_dist       > 0.08:    reasons.append(f"level2={l2_dist:.3f}m (>0.080)")
        if error_buf[idx].item():                   reasons.append("sanity_error(vel>threshold)")

        finger_names = ["thumb", "index", "middle", "ring", "pinky"]
        tip_dist = target_state["tips_distance"][idx]
        tip_contact = target_state["tip_contact_state"][idx]
        missed = (tip_dist < 0.005) & ~tip_contact.any(0)
        if missed.any():
            details = [f"{finger_names[i]}(dist={tip_dist[i]:.3f}m)" for i in range(5) if missed[i]]
            reasons.append("missed_contact: " + ", ".join(details))

        print(f"[FAIL {side} step={step}] " + " | ".join(reasons))

    def compute_observations(self):
        self._refresh()
        obs_rh = self.compute_observations_side("rh")
        obs_lh = self.compute_observations_side("lh")
        for k in obs_rh.keys():
            self.obs_dict[k] = torch.cat([obs_rh[k], obs_lh[k]], dim=-1)

    def compute_observations_side(self, side="rh"):
        # obs_keys: q, cos_q, sin_q, base_state
        side_states = getattr(self, f"{side}_states")
        side_demo_data = getattr(self, f"demo_data_{side}")

        obs_dict = {}

        obs_values = []
        for ob in self._obs_keys:
            if ob == "base_state":
                obs_values.append(
                    torch.cat([torch.zeros_like(side_states[ob][:, :3]), side_states[ob][:, 3:]], dim=-1)
                )  # ! ignore base position
            else:
                obs_values.append(side_states[ob])
        obs_dict["proprioception"] = torch.cat(obs_values, dim=-1)
        # privileged_obs_keys: dq, manip_obj_pos, manip_obj_quat, manip_obj_vel, manip_obj_ang_vel
        if len(self._privileged_obs_keys) > 0:
            pri_obs_values = []
            for ob in self._privileged_obs_keys:
                if ob == "manip_obj_pos":
                    pri_obs_values.append(side_states[ob] - side_states["base_state"][:, :3])
                elif ob == "manip_obj_com":
                    cur_com_pos = (
                        quat_to_rotmat(side_states["manip_obj_quat"][:, [1, 2, 3, 0]])
                        @ getattr(self, f"manip_obj_{side}_com").unsqueeze(-1)
                    ).squeeze(-1) + side_states["manip_obj_pos"]
                    pri_obs_values.append(cur_com_pos - side_states["base_state"][:, :3])
                elif ob == "manip_obj_weight":
                    prop = self.gym.get_sim_params(self.sim)
                    pri_obs_values.append((getattr(self, f"manip_obj_{side}_mass") * -1 * prop.gravity.z).unsqueeze(-1))
                elif ob == "tip_force":
                    tip_force = torch.stack(
                        [
                            self.net_cf[:, getattr(self, f"dexhand_{side}_handles")[k], :]
                            for k in (
                                self.dexhand_rh.contact_body_names
                                if side == "rh"
                                else self.dexhand_lh.contact_body_names
                            )
                        ],
                        axis=1,
                    )
                    tip_force = torch.cat(
                        [tip_force, torch.norm(tip_force, dim=-1, keepdim=True)], dim=-1
                    )  # add force magnitude
                    pri_obs_values.append(tip_force.reshape(self.num_envs, -1))
                else:
                    pri_obs_values.append(side_states[ob])
            obs_dict["privileged"] = torch.cat(pri_obs_values, dim=-1)

        next_target_state = {}

        cur_idx = self.progress_buf + 1
        cur_idx = torch.clamp(cur_idx, torch.zeros_like(side_demo_data["seq_len"]), side_demo_data["seq_len"] - 1)

        cur_idx = torch.stack(
            [cur_idx + t for t in range(self.obs_future_length)], dim=-1
        )  # [B, K], K = obs_future_length
        nE, nT = side_demo_data["wrist_pos"].shape[:2]
        nF = self.obs_future_length

        def indicing(data, idx):
            assert data.shape[0] == nE and data.shape[1] == nT
            remaining_shape = data.shape[2:]
            expanded_idx = idx
            for _ in remaining_shape:
                expanded_idx = expanded_idx.unsqueeze(-1)
            expanded_idx = expanded_idx.expand(-1, -1, *remaining_shape)
            return torch.gather(data, 1, expanded_idx)

        target_wrist_pos = indicing(side_demo_data["wrist_pos"], cur_idx)  # [B, K, 3]
        cur_wrist_pos = side_states["base_state"][:, :3]  # [B, 3]
        next_target_state["delta_wrist_pos"] = (target_wrist_pos - cur_wrist_pos[:, None]).reshape(nE, -1)

        target_wrist_vel = indicing(side_demo_data["wrist_velocity"], cur_idx)
        cur_wrist_vel = side_states["base_state"][:, 7:10]
        next_target_state["wrist_vel"] = target_wrist_vel.reshape(nE, -1)
        next_target_state["delta_wrist_vel"] = (target_wrist_vel - cur_wrist_vel[:, None]).reshape(nE, -1)

        target_wrist_rot = indicing(side_demo_data["wrist_rot"], cur_idx)
        cur_wrist_rot = side_states["base_state"][:, 3:7]

        next_target_state["wrist_quat"] = aa_to_quat(target_wrist_rot.reshape(nE * nF, -1))[:, [1, 2, 3, 0]]
        next_target_state["delta_wrist_quat"] = quat_mul(
            cur_wrist_rot[:, None].repeat(1, nF, 1).reshape(nE * nF, -1),
            quat_conjugate(next_target_state["wrist_quat"]),
        ).reshape(nE, -1)
        next_target_state["wrist_quat"] = next_target_state["wrist_quat"].reshape(nE, -1)

        target_wrist_ang_vel = indicing(side_demo_data["wrist_angular_velocity"], cur_idx)
        cur_wrist_ang_vel = side_states["base_state"][:, 10:13]
        next_target_state["wrist_ang_vel"] = target_wrist_ang_vel.reshape(nE, -1)
        next_target_state["delta_wrist_ang_vel"] = (target_wrist_ang_vel - cur_wrist_ang_vel[:, None]).reshape(nE, -1)

        target_joints_pos = indicing(side_demo_data["mano_joints"], cur_idx).reshape(nE, nF, -1, 3)
        cur_joint_pos = side_states["joints_state"][:, 1:, :3]  # skip the base joint
        next_target_state["delta_joints_pos"] = (target_joints_pos - cur_joint_pos[:, None]).reshape(self.num_envs, -1)

        target_joints_vel = indicing(side_demo_data["mano_joints_velocity"], cur_idx).reshape(nE, nF, -1, 3)
        cur_joint_vel = side_states["joints_state"][:, 1:, 7:10]  # skip the base joint
        next_target_state["joints_vel"] = target_joints_vel.reshape(self.num_envs, -1)
        next_target_state["delta_joints_vel"] = (target_joints_vel - cur_joint_vel[:, None]).reshape(self.num_envs, -1)

        target_obj_transf = indicing(side_demo_data["obj_trajectory"], cur_idx)
        target_obj_transf = target_obj_transf.reshape(nE * nF, 4, 4)
        next_target_state["delta_manip_obj_pos"] = (
            target_obj_transf[:, :3, 3].reshape(nE, nF, -1) - side_states["manip_obj_pos"][:, None]
        ).reshape(nE, -1)

        target_obj_vel = indicing(side_demo_data["obj_velocity"], cur_idx)
        cur_obj_vel = side_states["manip_obj_vel"]
        next_target_state["manip_obj_vel"] = target_obj_vel.reshape(nE, -1)
        next_target_state["delta_manip_obj_vel"] = (target_obj_vel - cur_obj_vel[:, None]).reshape(nE, -1)

        next_target_state["manip_obj_quat"] = rotmat_to_quat(target_obj_transf[:, :3, :3])[:, [1, 2, 3, 0]]
        next_target_state["delta_manip_obj_quat"] = quat_mul(
            side_states["manip_obj_quat"][:, None].repeat(1, nF, 1).reshape(nE * nF, -1),
            quat_conjugate(next_target_state["manip_obj_quat"]),
        ).reshape(nE, -1)
        next_target_state["manip_obj_quat"] = next_target_state["manip_obj_quat"].reshape(nE, -1)

        target_obj_ang_vel = indicing(side_demo_data["obj_angular_velocity"], cur_idx)
        cur_obj_ang_vel = side_states["manip_obj_ang_vel"]
        next_target_state["manip_obj_ang_vel"] = target_obj_ang_vel.reshape(nE, -1)
        next_target_state["delta_manip_obj_ang_vel"] = (target_obj_ang_vel - cur_obj_ang_vel[:, None]).reshape(nE, -1)

        next_target_state["obj_to_joints"] = torch.norm(
            side_states["manip_obj_pos"][:, None] - side_states["joints_state"][:, :, :3], dim=-1
        ).reshape(self.num_envs, -1)

        next_target_state["gt_tips_distance"] = indicing(side_demo_data["tips_distance"], cur_idx).reshape(nE, -1)

        next_target_state["bps"] = getattr(self, f"obj_bps_{side}")
        obs_dict["target"] = torch.cat(
            [
                next_target_state[ob]
                for ob in [  # ! must be in the same order as the following
                    "delta_wrist_pos",
                    "wrist_vel",
                    "delta_wrist_vel",
                    "wrist_quat",
                    "delta_wrist_quat",
                    "wrist_ang_vel",
                    "delta_wrist_ang_vel",
                    "delta_joints_pos",
                    "joints_vel",
                    "delta_joints_vel",
                    "delta_manip_obj_pos",
                    "manip_obj_vel",
                    "delta_manip_obj_vel",
                    "manip_obj_quat",
                    "delta_manip_obj_quat",
                    "manip_obj_ang_vel",
                    "delta_manip_obj_ang_vel",
                    "obj_to_joints",
                    "gt_tips_distance",
                    "bps",
                ]
            ],
            dim=-1,
        )

        if not self.training:
            manip_obj_root_state = getattr(self, f"_manip_obj_{side}_root_state")
            dexhand_handles = getattr(self, f"dexhand_{side}_handles")
            for prop_name in self._prop_dump_info.keys():
                if prop_name == "state_rh" and side == "rh":
                    self.dump_fileds[prop_name][:] = side_states["base_state"]
                elif prop_name == "state_lh" and side == "lh":
                    self.dump_fileds[prop_name][:] = side_states["base_state"]
                elif prop_name == "state_manip_obj_rh" and side == "rh":
                    self.dump_fileds[prop_name][:] = manip_obj_root_state
                elif prop_name == "state_manip_obj_lh" and side == "lh":
                    self.dump_fileds[prop_name][:] = manip_obj_root_state
                elif prop_name == "joint_state_rh" and side == "rh":
                    self.dump_fileds[prop_name][:] = torch.stack(
                        [self._rigid_body_state[:, dexhand_handles[k], :] for k in self.dexhand_rh.body_names],
                        dim=1,
                    ).reshape(self.num_envs, -1)
                elif prop_name == "joint_state_lh" and side == "lh":
                    self.dump_fileds[prop_name][:] = torch.stack(
                        [self._rigid_body_state[:, dexhand_handles[k], :] for k in self.dexhand_lh.body_names],
                        dim=1,
                    ).reshape(self.num_envs, -1)
                elif prop_name == "q_rh" and side == "rh":
                    self.dump_fileds[prop_name][:] = side_states["q"]
                elif prop_name == "q_lh" and side == "lh":
                    self.dump_fileds[prop_name][:] = side_states["q"]
                elif prop_name == "dq_rh" and side == "rh":
                    self.dump_fileds[prop_name][:] = side_states["dq"]
                elif prop_name == "dq_lh" and side == "lh":
                    self.dump_fileds[prop_name][:] = side_states["dq"]
                elif prop_name == "tip_force_rh" and side == "rh":
                    tip_force = torch.stack(
                        [self.net_cf[:, dexhand_handles[k], :] for k in self.dexhand_rh.contact_body_names],
                        axis=1,
                    )
                    self.dump_fileds[prop_name][:] = tip_force.reshape(self.num_envs, -1)
                elif prop_name == "tip_force_lh" and side == "lh":
                    tip_force = torch.stack(
                        [self.net_cf[:, dexhand_handles[k], :] for k in self.dexhand_lh.contact_body_names],
                        axis=1,
                    )
                    self.dump_fileds[prop_name][:] = tip_force.reshape(self.num_envs, -1)
                elif prop_name == "reward":
                    self.dump_fileds[prop_name][:] = self.rew_buf.reshape(self.num_envs, -1).detach()
                else:
                    pass
        return obs_dict

    def _reset_default(self, env_ids):
        if self.random_state_init:
            if self.rollout_begin is not None:
                seq_idx = (
                    torch.floor(
                        self.rollout_len * 0.98 * torch.rand_like(self.demo_data_rh["seq_len"][env_ids].float())
                    ).long()
                    + self.rollout_begin
                )
                seq_idx = torch.clamp(
                    seq_idx,
                    torch.zeros(1, device=self.device).long(),
                    torch.floor(self.demo_data_rh["seq_len"][env_ids] * 0.98).long(),
                )
            else:
                 seq_idx = torch.floor(
                    self.demo_data_rh["seq_len"][env_ids]
                    * 0.98
                    * torch.rand_like(self.demo_data_rh["seq_len"][env_ids].float())
                ).long()
        else:
            if self.rollout_begin is not None:
                seq_idx = self.rollout_begin * torch.ones_like(self.demo_data_rh["seq_len"][env_ids].long())
            else:
                # print("Starting from frame: ", self.eval_start_frame)
                seq_idx = self.eval_start_frame * torch.ones_like(self.demo_data_rh["seq_len"][env_ids].long())

        self._reset_default_side(env_ids, seq_idx, side="lh")
        self._reset_default_side(env_ids, seq_idx, side="rh")

        dexhand_multi_env_ids_int32 = torch.concat(
            [
                self._global_dexhand_rh_indices[env_ids].flatten(),
                self._global_dexhand_lh_indices[env_ids].flatten(),
            ]
        )
        manip_obj_multi_env_ids_int32 = torch.concat(
            [self._global_manip_obj_rh_indices[env_ids].flatten(), self._global_manip_obj_lh_indices[env_ids].flatten()]
        )

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(dexhand_multi_env_ids_int32),
            len(dexhand_multi_env_ids_int32),
        )
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(torch.concat([dexhand_multi_env_ids_int32, manip_obj_multi_env_ids_int32])),
            len(torch.concat([dexhand_multi_env_ids_int32, manip_obj_multi_env_ids_int32])),
        )
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._pos_control),
            gymtorch.unwrap_tensor(dexhand_multi_env_ids_int32),
            len(dexhand_multi_env_ids_int32),
        )

        self.progress_buf[env_ids] = seq_idx
        self.running_progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.success_buf[env_ids] = 0
        self.failure_buf[env_ids] = 0
        self.error_buf[env_ids] = 0
        self.total_rew_buf[env_ids] = 0
        self.apply_forces[env_ids] = 0
        self.apply_torque[env_ids] = 0
        self.curr_targets[env_ids] = 0
        self.prev_targets[env_ids] = 0

        if self.use_pid_control:
            self.rh_prev_pos_error[env_ids] = 0
            self.rh_prev_rot_error[env_ids] = 0
            self.rh_pos_error_integral[env_ids] = 0
            self.rh_rot_error_integral[env_ids] = 0
            self.lh_prev_pos_error[env_ids] = 0
            self.lh_prev_rot_error[env_ids] = 0
            self.lh_pos_error_integral[env_ids] = 0
            self.lh_rot_error_integral[env_ids] = 0

        self.lh_tips_contact_history[env_ids] = torch.ones_like(self.lh_tips_contact_history[env_ids]).bool()
        self.rh_tips_contact_history[env_ids] = torch.ones_like(self.rh_tips_contact_history[env_ids]).bool()

    def _reset_default_side(self, env_ids, seq_idx, side="rh"):

        side_demo_data = getattr(self, f"demo_data_{side}")

        if self.live:
            # Live has no retargeting (opt_dof_pos is the reference demo's, not the live target).
            # Start fingers at the dexhand's default pose; the frozen imitator commands finger
            # DOF targets every step and pulls the hand to its own output within a couple frames.
            # Auto-reset is disabled in live, so this init only happens once at startup.
            default = getattr(self, f"dexhand_{side}_default_dof_pos").to(self.device)
            dof_pos = default.unsqueeze(0).expand(env_ids.shape[0], -1).clone()
            dof_vel = torch.zeros_like(dof_pos)
        else:
            dof_pos = side_demo_data["opt_dof_pos"][env_ids, seq_idx]
            dof_pos = torch_jit_utils.tensor_clamp(
                dof_pos,
                getattr(self, f"dexhand_{side}_dof_lower_limits").unsqueeze(0),
                getattr(self, f"dexhand_{side}_dof_upper_limits").unsqueeze(0),
            )
            dof_vel = side_demo_data["opt_dof_velocity"][env_ids, seq_idx]
            dof_vel = torch_jit_utils.tensor_clamp(
                dof_vel,
                -1 * getattr(self, f"_dexhand_{side}_dof_speed_limits").unsqueeze(0),
                getattr(self, f"_dexhand_{side}_dof_speed_limits").unsqueeze(0),
            )

        opt_wrist_pos = side_demo_data["opt_wrist_pos"][env_ids, seq_idx]
        opt_wrist_rot = aa_to_quat(side_demo_data["opt_wrist_rot"][env_ids, seq_idx])
        opt_wrist_rot = opt_wrist_rot[:, [1, 2, 3, 0]]

        opt_wrist_vel = side_demo_data["opt_wrist_velocity"][env_ids, seq_idx]
        opt_wrist_ang_vel = side_demo_data["opt_wrist_angular_velocity"][env_ids, seq_idx]

        opt_hand_pose_vel = torch.concat([opt_wrist_pos, opt_wrist_rot, opt_wrist_vel, opt_wrist_ang_vel], dim=-1)

        getattr(self, f"_{side}_base_state")[env_ids, :] = opt_hand_pose_vel

        if side == "rh":
            self._q[env_ids, : self.num_dexhand_rh_dofs] = dof_pos
            self._qd[env_ids, : self.num_dexhand_rh_dofs] = dof_vel
            self._pos_control[env_ids, : self.num_dexhand_rh_dofs] = dof_pos
        else:
            self._q[env_ids, self.num_dexhand_rh_dofs :] = dof_pos
            self._qd[env_ids, self.num_dexhand_rh_dofs :] = dof_vel
            self._pos_control[env_ids, self.num_dexhand_rh_dofs :] = dof_pos

        # reset manip obj
        obj_pos_init = side_demo_data["obj_trajectory"][env_ids, seq_idx, :3, 3]
        obj_rot_init = side_demo_data["obj_trajectory"][env_ids, seq_idx, :3, :3]
        obj_rot_init = rotmat_to_quat(obj_rot_init)
        # [w, x, y, z] to [x, y, z, w]
        obj_rot_init = obj_rot_init[:, [1, 2, 3, 0]]

        obj_vel = side_demo_data["obj_velocity"][env_ids, seq_idx]
        obj_ang_vel = side_demo_data["obj_angular_velocity"][env_ids, seq_idx]

        manip_obj_root_state = getattr(self, f"_manip_obj_{side}_root_state")

        manip_obj_root_state[env_ids, :3] = obj_pos_init
        manip_obj_root_state[env_ids, 3:7] = obj_rot_init
        manip_obj_root_state[env_ids, 7:10] = obj_vel
        manip_obj_root_state[env_ids, 10:13] = obj_ang_vel

    def reset_idx(self, env_ids):
        self._refresh()
        if self.randomize:
            self.apply_randomizations(self.dr_randomizations)

        last_step = self.gym.get_frame_count(self.sim)
        if self.training and len(self.dataIndices) == 1 and last_step >= self.tighten_steps:
            running_steps = self.running_progress_buf[env_ids] - 1
            max_running_steps, max_running_idx = running_steps.max(dim=0)
            max_running_env_id = env_ids[max_running_idx]
            if max_running_steps > self.best_rollout_len:
                self.best_rollout_len = max_running_steps
                self.best_rollout_begin = self.progress_buf[max_running_env_id] - 1 - max_running_steps

        if len(self.dataIndices) > 1:
            for env_id in env_ids.tolist():
                demo_name = self.dataIndices[self.env_demo_idx[env_id]]
                self._pending_demo_episode_rewards[demo_name].append(self.total_rew_buf[env_id].item())
                self._pending_demo_episode_successes[demo_name].append(float(self.success_buf[env_id].item()))
        self._reset_default(env_ids)

    def reset_done(self):
        done_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(done_env_ids) > 0:
            self.reset_idx(done_env_ids)
            self.compute_observations()

        if not self.dict_obs_cls:
            self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

            # asymmetric actor-critic
            if self.num_states > 0:
                self.obs_dict["states"] = self.get_state()

        return self.obs_dict, done_env_ids

    def step(self, actions):
        obs, rew, done, info = super().step(actions)
        info["reward_dict"] = self.reward_dict
        info["total_rewards"] = self.total_rew_buf
        info["total_steps"] = self.progress_buf
        if len(self.dataIndices) > 1:
            info["per_demo_episode_rewards"] = {k: list(v) for k, v in self._pending_demo_episode_rewards.items()}
            info["per_demo_episode_successes"] = {k: list(v) for k, v in self._pending_demo_episode_successes.items()}
            for k in self._pending_demo_episode_rewards:
                self._pending_demo_episode_rewards[k].clear()
                self._pending_demo_episode_successes[k].clear()
        return obs, rew, done, info

    def pre_physics_step(self, actions):

        # ? >>> for visualization
        if not self.headless:

            cur_idx = self.progress_buf

            self.gym.clear_lines(self.viewer)

            def set_side_joint(cur_idx, side="rh"):
                cur_wrist_pos = getattr(self, f"demo_data_{side}")["wrist_pos"][torch.arange(self.num_envs), cur_idx]
                cur_mano_joint_pos = getattr(self, f"demo_data_{side}")["mano_joints"][
                    torch.arange(self.num_envs), cur_idx
                ].reshape(self.num_envs, -1, 3)
                cur_mano_joint_pos = torch.concat([cur_wrist_pos[:, None], cur_mano_joint_pos], dim=1)
                for k in range(len(getattr(self, f"mano_joint_{side}_points"))):
                    getattr(self, f"mano_joint_{side}_points")[k][:, :3] = cur_mano_joint_pos[:, k]
                for env_id, env_ptr in enumerate(self.envs):
                    for rh_k, k in zip(
                        self.dexhand_rh.body_names,
                        (self.dexhand_rh.body_names if side == "rh" else self.dexhand_lh.body_names),
                    ):
                        self.set_force_vis(
                            env_ptr,
                            rh_k,
                            torch.norm(self.net_cf[env_id, getattr(self, f"dexhand_{side}_handles")[k]], dim=-1) != 0,
                            side,
                        )

                    def add_lines(viewer, env_ptr, hand_joints, color):
                        assert hand_joints.shape[0] == self.dexhand_rh.n_bodies and hand_joints.shape[1] == 3
                        hand_joints = hand_joints.cpu().numpy()
                        lines = np.array([[hand_joints[b[0]], hand_joints[b[1]]] for b in self.dexhand_rh.bone_links])
                        for line in lines:
                            self.gym.add_lines(viewer, env_ptr, 1, line, color)

                    color = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
                    add_lines(self.viewer, env_ptr, cur_mano_joint_pos[env_id].cpu(), color)

            set_side_joint(cur_idx, "lh")
            set_side_joint(cur_idx, "rh")

            def draw_frame(env_ptr, pos, quat, axis_len=0.05):
                """Draw XYZ axes (R/G/B) at pos with orientation from quat [x,y,z,w]."""
                x, y, z, w = quat
                R = np.array([
                    [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)  ],
                    [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)  ],
                    [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
                ])
                for col_idx, color in enumerate([[1,0,0],[0,1,0],[0,0,1]]):
                    axis = R[:, col_idx]
                    line = np.array([pos, pos + axis_len * axis], dtype=np.float32)
                    self.gym.add_lines(self.viewer, env_ptr, 1, line,
                                       np.array([color], dtype=np.float32))

            env_ptr0 = self.envs[0]
            rh_state = self._manip_obj_rh_root_state[0].cpu().numpy()
            lh_state = self._manip_obj_lh_root_state[0].cpu().numpy()
            draw_frame(env_ptr0, rh_state[:3], rh_state[3:7])
            draw_frame(env_ptr0, lh_state[:3], lh_state[3:7])

        # ? <<< for visualization

        root_control_dim = 9 if self.use_pid_control else 6
        res_split_idx = (
            actions.shape[1] // 2
            if not self.use_pid_control
            else ((actions.shape[1] - 2 * (root_control_dim - 6)) // 2) + 2 * (root_control_dim - 6)
        )

        base_action = actions[:, :res_split_idx]  # ? in the range of [-1, 1]
        residual_action = actions[:, res_split_idx:] * 2  # ? the delta action is theoritically in the range of [-2, 2]
        if self.zero_residual:
            # print("USING NO RESIDUAL ACTIONS")
            residual_action = torch.zeros_like(residual_action)

        rh_dof_pos = (
            1.0 * base_action[:, root_control_dim : root_control_dim + self.num_dexhand_rh_dofs]
            + residual_action[:, 6 : 6 + self.num_dexhand_rh_dofs]
        )
        rh_dof_pos = torch.clamp(rh_dof_pos, -1, 1)

        lh_dof_pos = (
            1.0 * base_action[:, root_control_dim + root_control_dim + self.num_dexhand_rh_dofs :]
            + residual_action[:, 6 + 6 + self.num_dexhand_rh_dofs :]
        )
        lh_dof_pos = torch.clamp(lh_dof_pos, -1, 1)

        curr_act_moving_average = self.act_moving_average

        self.rh_curr_targets = torch_jit_utils.scale(
            rh_dof_pos,  # ! actions must in [-1, 1]
            self.dexhand_rh_dof_lower_limits,
            self.dexhand_rh_dof_upper_limits,
        )
        self.rh_curr_targets = (
            curr_act_moving_average * self.rh_curr_targets
            + (1.0 - curr_act_moving_average) * self.prev_targets[:, : self.num_dexhand_rh_dofs]
        )
        self.rh_curr_targets = torch_jit_utils.tensor_clamp(
            self.rh_curr_targets,
            self.dexhand_rh_dof_lower_limits,
            self.dexhand_rh_dof_upper_limits,
        )
        self.prev_targets[:, : self.num_dexhand_rh_dofs] = self.rh_curr_targets[:]

        self.lh_curr_targets = torch_jit_utils.scale(
            lh_dof_pos,
            self.dexhand_lh_dof_lower_limits,
            self.dexhand_lh_dof_upper_limits,
        )
        self.lh_curr_targets = (
            curr_act_moving_average * self.lh_curr_targets
            + (1.0 - curr_act_moving_average) * self.prev_targets[:, self.num_dexhand_rh_dofs :]
        )
        self.lh_curr_targets = torch_jit_utils.tensor_clamp(
            self.lh_curr_targets,
            self.dexhand_lh_dof_lower_limits,
            self.dexhand_lh_dof_upper_limits,
        )
        self.prev_targets[:, self.num_dexhand_rh_dofs :] = self.lh_curr_targets[:]

        if self.use_pid_control:
            rh_position_error = base_action[:, 0:3]
            self.rh_pos_error_integral += rh_position_error * self.dt
            self.rh_pos_error_integral = torch.clamp(self.rh_pos_error_integral, -1, 1)
            rh_pos_derivative = (rh_position_error - self.rh_prev_pos_error) / self.dt
            rh_force = (
                self.Kp_pos * rh_position_error
                + self.Ki_pos * self.rh_pos_error_integral
                + self.Kd_pos * rh_pos_derivative
            )
            self.rh_prev_pos_error = rh_position_error

            rh_force = rh_force + residual_action[:, 0:3] * self.dt * self.translation_scale * 500
            self.apply_forces[:, self.dexhand_rh_handles[self.dexhand_rh.to_dex("wrist")[0]], :] = (
                curr_act_moving_average * rh_force
                + (1.0 - curr_act_moving_average)
                * self.apply_forces[:, self.dexhand_rh_handles[self.dexhand_rh.to_dex("wrist")[0]], :]
            )

            lh_position_error = base_action[
                :, root_control_dim + self.num_dexhand_rh_dofs : root_control_dim + self.num_dexhand_rh_dofs + 3
            ]
            self.lh_pos_error_integral += lh_position_error * self.dt
            self.lh_pos_error_integral = torch.clamp(self.lh_pos_error_integral, -1, 1)
            lh_pos_derivative = (lh_position_error - self.lh_prev_pos_error) / self.dt
            lh_force = (
                self.Kp_pos * lh_position_error
                + self.Ki_pos * self.lh_pos_error_integral
                + self.Kd_pos * lh_pos_derivative
            )
            self.lh_prev_pos_error = lh_position_error

            lh_force = (
                lh_force
                + residual_action[:, 6 + self.num_dexhand_rh_dofs : 6 + self.num_dexhand_rh_dofs + 3]
                * self.dt
                * self.translation_scale
                * 500
            )
            self.apply_forces[:, self.dexhand_lh_handles[self.dexhand_lh.to_dex("wrist")[0]], :] = (
                curr_act_moving_average * lh_force
                + (1.0 - curr_act_moving_average)
                * self.apply_forces[:, self.dexhand_lh_handles[self.dexhand_lh.to_dex("wrist")[0]], :]
            )

            rh_rotation_error = base_action[:, 3:root_control_dim]
            rh_rotation_error = rot6d_to_aa(rh_rotation_error)
            self.rh_rot_error_integral += rh_rotation_error * self.dt
            self.rh_rot_error_integral = torch.clamp(self.rh_rot_error_integral, -1, 1)
            rh_rot_derivative = (rh_rotation_error - self.rh_prev_rot_error) / self.dt
            rh_torque = (
                self.Kp_rot * rh_rotation_error
                + self.Ki_rot * self.rh_rot_error_integral
                + self.Kd_rot * rh_rot_derivative
            )
            self.rh_prev_rot_error = rh_rotation_error

            rh_torque = rh_torque + residual_action[:, 3:6] * self.dt * self.orientation_scale * 200
            self.apply_torque[:, self.dexhand_rh_handles[self.dexhand_rh.to_dex("wrist")[0]], :] = (
                curr_act_moving_average * rh_torque
                + (1.0 - curr_act_moving_average)
                * self.apply_torque[:, self.dexhand_rh_handles[self.dexhand_rh.to_dex("wrist")[0]], :]
            )

            lh_rotation_error = base_action[
                :,
                root_control_dim
                + self.num_dexhand_rh_dofs
                + 3 : root_control_dim
                + self.num_dexhand_rh_dofs
                + root_control_dim,
            ]
            lh_rotation_error = rot6d_to_aa(lh_rotation_error)
            self.lh_rot_error_integral += lh_rotation_error * self.dt
            self.lh_rot_error_integral = torch.clamp(self.lh_rot_error_integral, -1, 1)
            lh_rot_derivative = (lh_rotation_error - self.lh_prev_rot_error) / self.dt
            lh_torque = (
                self.Kp_rot * lh_rotation_error
                + self.Ki_rot * self.lh_rot_error_integral
                + self.Kd_rot * lh_rot_derivative
            )
            self.lh_prev_rot_error = lh_rotation_error

            lh_torque = (
                lh_torque
                + residual_action[:, 6 + self.num_dexhand_rh_dofs + 3 : 6 + self.num_dexhand_rh_dofs + 6]
                * self.dt
                * self.orientation_scale
                * 200
            )
            self.apply_torque[:, self.dexhand_lh_handles[self.dexhand_lh.to_dex("wrist")[0]], :] = (
                curr_act_moving_average * lh_torque
                + (1.0 - curr_act_moving_average)
                * self.apply_torque[:, self.dexhand_lh_handles[self.dexhand_lh.to_dex("wrist")[0]], :]
            )
        else:
            rh_force = 1.0 * (base_action[:, 0:3] * self.dt * self.translation_scale * 500) + (
                residual_action[:, 0:3] * self.dt * self.translation_scale * 500
            )
            rh_torque = 1.0 * (base_action[:, 3:6] * self.dt * self.orientation_scale * 200) + (
                residual_action[:, 3:6] * self.dt * self.orientation_scale * 200
            )
            lh_force = 1.0 * (
                base_action[
                    :, root_control_dim + self.num_dexhand_rh_dofs : root_control_dim + self.num_dexhand_rh_dofs + 3
                ]
                * self.dt
                * self.translation_scale
                * 500
            ) + (
                residual_action[:, 6 + self.num_dexhand_rh_dofs : 6 + self.num_dexhand_rh_dofs + 3]
                * self.dt
                * self.translation_scale
                * 500
            )
            lh_torque = 1.0 * (
                base_action[
                    :, root_control_dim + self.num_dexhand_rh_dofs + 3 : root_control_dim + self.num_dexhand_rh_dofs + 6
                ]
                * self.dt
                * self.orientation_scale
                * 200
            ) + (
                residual_action[:, 6 + self.num_dexhand_rh_dofs + 3 : 6 + self.num_dexhand_rh_dofs + 6]
                * self.dt
                * self.orientation_scale
                * 200
            )

            self.apply_forces[:, self.dexhand_rh_handles[self.dexhand_rh.to_dex("wrist")[0]], :] = (
                curr_act_moving_average * rh_force
                + (1.0 - curr_act_moving_average)
                * self.apply_forces[:, self.dexhand_rh_handles[self.dexhand_rh.to_dex("wrist")[0]], :]
            )
            self.apply_torque[:, self.dexhand_rh_handles[self.dexhand_rh.to_dex("wrist")[0]], :] = (
                curr_act_moving_average * rh_torque
                + (1.0 - curr_act_moving_average)
                * self.apply_torque[:, self.dexhand_rh_handles[self.dexhand_rh.to_dex("wrist")[0]], :]
            )

            self.apply_forces[:, self.dexhand_lh_handles[self.dexhand_lh.to_dex("wrist")[0]], :] = (
                curr_act_moving_average * lh_force
                + (1.0 - curr_act_moving_average)
                * self.apply_forces[:, self.dexhand_lh_handles[self.dexhand_lh.to_dex("wrist")[0]], :]
            )
            self.apply_torque[:, self.dexhand_lh_handles[self.dexhand_lh.to_dex("wrist")[0]], :] = (
                curr_act_moving_average * lh_torque
                + (1.0 - curr_act_moving_average)
                * self.apply_torque[:, self.dexhand_lh_handles[self.dexhand_lh.to_dex("wrist")[0]], :]
            )

        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.apply_forces),
            gymtorch.unwrap_tensor(self.apply_torque),
            gymapi.ENV_SPACE,
        )

        self._pos_control[:] = self.prev_targets[:]

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))

    def _draw_obj_axes(self, axis_len=0.05):
        """Draw XYZ coordinate frames on both objects.

        Viewer mode: gym.add_lines debug draw.
        Headless camera mode: project 3-D axes into each camera image.
        """
        use_viewer = self.viewer is not None
        use_camera = self.camera_obs is not None

        if not use_viewer and not use_camera:
            return

        rh_states = self._manip_obj_rh_root_state.cpu()
        lh_states = self._manip_obj_lh_root_state.cpu()
        axes = [
            torch.tensor([axis_len, 0.0, 0.0]),
            torch.tensor([0.0, axis_len, 0.0]),
            torch.tensor([0.0, 0.0, axis_len]),
        ]
        viewer_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        if use_viewer:
            self.gym.clear_lines(self.viewer)

        for i, env_ptr in enumerate(self.envs):
            all_states = [rh_states, lh_states]

            if use_viewer:
                for states in all_states:
                    pos = states[i, :3]
                    q = states[i, 3:7]
                    ends = [torch_jit_utils.quat_rotate(q.unsqueeze(0), ax.unsqueeze(0)).squeeze(0) + pos
                            for ax in axes]
                    verts = []
                    for end in ends:
                        verts += [pos[0].item(), pos[1].item(), pos[2].item(),
                                   end[0].item(), end[1].item(), end[2].item()]
                    self.gym.add_lines(self.viewer, env_ptr, 3, verts, viewer_colors)

            if use_camera:
                import cv2
                import numpy as np
                cam_handle = self.camera_handlers[i]
                view_mat = np.array(self.gym.get_camera_view_matrix(self.sim, env_ptr, cam_handle))
                proj_mat = np.array(self.gym.get_camera_proj_matrix(self.sim, env_ptr, cam_handle))
                frame = self.camera_obs[i]  # RGBA GPU tensor [H, W, 4]
                H, W = frame.shape[:2]
                img = frame.cpu().numpy()[..., :3][..., ::-1].copy()  # RGBA→BGR uint8

                def project(pt):
                    p = np.array([pt[0].item(), pt[1].item(), pt[2].item(), 1.0])
                    p_clip = proj_mat @ (view_mat @ p)
                    if abs(p_clip[3]) < 1e-6:
                        return None
                    ndc = p_clip[:3] / p_clip[3]
                    u = int((ndc[0] + 1) * 0.5 * W)
                    v = int((1 - ndc[1]) * 0.5 * H)
                    if 0 <= u < W and 0 <= v < H:
                        return (u, v)
                    return None

                cv_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR: X=red, Y=green, Z=blue
                for states in all_states:
                    pos = states[i, :3]
                    q = states[i, 3:7]
                    origin = project(pos)
                    if origin is None:
                        continue
                    for ax, color in zip(axes, cv_colors):
                        tip = torch_jit_utils.quat_rotate(q.unsqueeze(0), ax.unsqueeze(0)).squeeze(0) + pos
                        tip_px = project(tip)
                        if tip_px is not None:
                            cv2.line(img, origin, tip_px, color, 2)

                frame[:, :, :3] = torch.from_numpy(img[..., ::-1].copy()).to(frame.device)

    def _ensure_live_source(self):
        if self.live_source is not None:
            return
        from ..live.live_target_source import LiveTargetSource

        ov_rh = self.demo_data_rh["obj_verts"]
        ov_lh = self.demo_data_lh["obj_verts"]
        if ov_rh.ndim == 3:
            ov_rh = ov_rh[0]
        if ov_lh.ndim == 3:
            ov_lh = ov_lh[0]
        self.live_source = LiveTargetSource(
            addr=self.live_addr,
            port=self.live_port,
            dexhand_rh=self.dexhand_rh,
            dexhand_lh=self.dexhand_lh,
            mujoco2gym_transf=self.mujoco2gym_transf,
            obj_verts_rh=ov_rh,
            obj_verts_lh=ov_lh,
            device=self.device,
            buffered=self.live_buffered,
        )
        self.live_source.start()
        # packed mano-joint order per side (dexhand body order minus wrist) — matches pack_data
        self._live_mano_order = {
            s: [dex.to_hand(j)[0] for j in dex.body_names if dex.to_hand(j)[0] != "wrist"]
            for s, dex in (("rh", self.dexhand_rh), ("lh", self.dexhand_lh))
        }

    def _inject_live(self):
        """Overwrite every demo target slot with the latest live frame, broadcast across envs."""
        self._ensure_live_source()
        f = self.live_source.latest()
        for side, demo in (("rh", self.demo_data_rh), ("lh", self.demo_data_lh)):
            t = f[side]
            demo["wrist_pos"][:, :] = t["wrist_pos"]
            demo["wrist_rot"][:, :] = t["wrist_rot"]
            demo["wrist_velocity"][:, :] = t["wrist_velocity"]
            demo["wrist_angular_velocity"][:, :] = t["wrist_angular_velocity"]
            demo["obj_trajectory"][:, :] = t["obj_trajectory"]
            demo["obj_velocity"][:, :] = t["obj_velocity"]
            demo["obj_angular_velocity"][:, :] = t["obj_angular_velocity"]
            demo["tips_distance"][:, :] = t["tips_distance"]
            order = self._live_mano_order[side]
            demo["mano_joints"][:, :] = torch.cat([t["mano_joints"][n] for n in order], dim=-1)
            demo["mano_joints_velocity"][:, :] = torch.cat(
                [t["mano_joints_velocity"][n] for n in order], dim=-1
            )

    def post_physics_step(self):
        if self.live:
            self._inject_live()

        self.compute_observations()
        self.compute_reward(self.actions)
        # self._draw_obj_axes()

        if self.live:
            self.reset_buf[:] = 0  # live teleop runs continuously; never auto-reset

        # Manual reset on viewer key 'N' (set in vec_task.render). Re-inits all envs — useful in
        # live mode (no auto-reset) to re-attempt a replay when the first playthroughs are broken.
        if getattr(self, "_reset_env_request", False):
            self._reset_env_request = False
            self.reset_idx(torch.arange(self.num_envs, device=self.device))
            if self.live and self.live_source is not None:
                self.live_source.request_publisher_reset()  # restart the replay trajectory too
            print("[env] manual reset (key N)")

        self.progress_buf += 1
        self.running_progress_buf += 1
        self.randomize_buf += 1
        if self.live:
            # No auto-reset means progress_buf would run past the tiny demo buffer; the reward
            # and set_side_joint read it UNclamped (only compute_observations clamps). Hold it
            # inside bounds — every demo slot already holds the latest live frame.
            self.progress_buf = torch.minimum(self.progress_buf, self.demo_data_rh["seq_len"] - 1)

    def create_camera(
        self,
        *,
        env,
        isaac_gym,
    ):
        """
        Only create front camera for view purpose
        """
        if self._record:
            camera_cfg = gymapi.CameraProperties()
            camera_cfg.enable_tensors = True
            camera_cfg.width = 1280
            camera_cfg.height = 720
            camera_cfg.horizontal_fov = 69.4

            camera = isaac_gym.create_camera_sensor(env, camera_cfg)
            cam_pos = gymapi.Vec3(0.80, -0.00, 0.7)
            cam_target = gymapi.Vec3(-1, -0.00, 0.3)
            isaac_gym.set_camera_location(camera, env, cam_pos, cam_target)
        else:
            camera_cfg = gymapi.CameraProperties()
            camera_cfg.enable_tensors = True
            camera_cfg.width = 320
            camera_cfg.height = 180
            camera_cfg.horizontal_fov = 69.4

            camera = isaac_gym.create_camera_sensor(env, camera_cfg)
            cam_pos = gymapi.Vec3(0.97, 0, 0.74)
            cam_target = gymapi.Vec3(-1, 0, 0.5)
            isaac_gym.set_camera_location(camera, env, cam_pos, cam_target)
        return camera

    def create_camera_top(self, *, env, isaac_gym):
        """Behind view camera for secondary recording."""
        camera_cfg = gymapi.CameraProperties()
        camera_cfg.enable_tensors = True
        camera_cfg.width = 1280
        camera_cfg.height = 720
        camera_cfg.horizontal_fov = 69.4
        camera = isaac_gym.create_camera_sensor(env, camera_cfg)
        cam_pos = gymapi.Vec3(-0.97, 0.0, 0.74)
        cam_target = gymapi.Vec3(1, 0.0, 0.3)
        isaac_gym.set_camera_location(camera, env, cam_pos, cam_target)
        return camera

    def set_camera(self):
        super().set_camera()
        self.camera_obs_top = None
        if self.camera_handlers_top is not None:
            self.camera_obs_top = []
            for env, handle in zip(self.envs, self.camera_handlers_top):
                self.camera_obs_top.append(
                    gymtorch.wrap_tensor(
                        self.gym.get_camera_image_gpu_tensor(self.sim, env, handle, gymapi.IMAGE_COLOR)
                    )
                )

    def set_force_vis(self, env_ptr, part_k, has_force, side):
        self.gym.set_rigid_body_color(
            env_ptr,
            self.gym.find_actor_handle(env_ptr, "dexhand_l" if side == "lh" else "dexhand_r"),
            getattr(self, f"dexhand_rh_handles")[part_k],  # tricks here, because the handle is the same
            gymapi.MESH_VISUAL,
            (
                gymapi.Vec3(
                    1.0,
                    0.6,
                    0.6,
                )
                if has_force
                else gymapi.Vec3(1.0, 1.0, 1.0)
            ),
        )


@torch.jit.script
def quat_to_angle_axis(q):
    # type: (Tensor) -> Tuple[Tensor, Tensor]
    # computes axis-angle representation from quaternion q
    # q must be normalized
    min_theta = 1e-5
    qx, qy, qz, qw = 0, 1, 2, 3

    sin_theta = torch.sqrt(1 - q[..., qw] * q[..., qw])
    angle = 2 * torch.acos(q[..., qw])
    angle = normalize_angle(angle)
    sin_theta_expand = sin_theta.unsqueeze(-1)
    axis = q[..., qx:qw] / sin_theta_expand

    mask = torch.abs(sin_theta) > min_theta
    default_axis = torch.zeros_like(axis)
    default_axis[..., -1] = 1

    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)
    return angle, axis


@torch.jit.script
def compute_imitation_reward(
    reset_buf: Tensor,
    progress_buf: Tensor,
    running_progress_buf: Tensor,
    actions: Tensor,
    states: Dict[str, Tensor],
    target_states: Dict[str, Tensor],
    max_length: List[int],
    scale_factor: float,
    noise_compensation: float,
    dexhand_weight_idx: Dict[str, List[int]],
    training: bool = True,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

    # type: (Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, Tensor], Tensor, float, float, Dict[str, List[int]], bool) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor], Tensor]

    # end effector pose reward
    current_eef_pos = states["base_state"][:, :3]
    current_eef_quat = states["base_state"][:, 3:7]

    target_eef_pos = target_states["wrist_pos"]
    target_eef_quat = target_states["wrist_quat"]
    diff_eef_pos = target_eef_pos - current_eef_pos
    diff_eef_pos_dist = torch.norm(diff_eef_pos, dim=-1)

    current_eef_vel = states["base_state"][:, 7:10]
    current_eef_ang_vel = states["base_state"][:, 10:13]
    target_eef_vel = target_states["wrist_vel"]
    target_eef_ang_vel = target_states["wrist_ang_vel"]

    diff_eef_vel = target_eef_vel - current_eef_vel
    diff_eef_ang_vel = target_eef_ang_vel - current_eef_ang_vel

    joints_pos = states["joints_state"][:, 1:, :3]
    target_joints_pos = target_states["joints_pos"]
    diff_joints_pos = target_joints_pos - joints_pos
    diff_joints_pos_dist = torch.norm(diff_joints_pos, dim=-1)

    # ? assign different weights to different joints
    # assert diff_joints_pos_dist.shape[1] == 17  # ignore the base joint
    diff_thumb_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["thumb_tip"]]].mean(dim=-1)
    diff_index_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["index_tip"]]].mean(dim=-1)
    diff_middle_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["middle_tip"]]].mean(dim=-1)
    diff_ring_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["ring_tip"]]].mean(dim=-1)
    diff_pinky_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["pinky_tip"]]].mean(dim=-1)
    diff_level_1_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["level_1_joints"]]].mean(dim=-1)
    diff_level_2_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["level_2_joints"]]].mean(dim=-1)

    joints_vel = states["joints_state"][:, 1:, 7:10]
    target_joints_vel = target_states["joints_vel"]
    diff_joints_vel = target_joints_vel - joints_vel

    reward_eef_pos = torch.exp(-40 * diff_eef_pos_dist)
    reward_thumb_tip_pos = torch.exp(-100 * diff_thumb_tip_pos_dist)
    reward_index_tip_pos = torch.exp(-90 * diff_index_tip_pos_dist)
    reward_middle_tip_pos = torch.exp(-80 * diff_middle_tip_pos_dist)
    reward_pinky_tip_pos = torch.exp(-60 * diff_pinky_tip_pos_dist)
    reward_ring_tip_pos = torch.exp(-60 * diff_ring_tip_pos_dist)
    reward_level_1_pos = torch.exp(-50 * diff_level_1_pos_dist)
    reward_level_2_pos = torch.exp(-40 * diff_level_2_pos_dist)

    reward_eef_vel = torch.exp(-1 * diff_eef_vel.abs().mean(dim=-1))
    reward_eef_ang_vel = torch.exp(-1 * diff_eef_ang_vel.abs().mean(dim=-1))
    reward_joints_vel = torch.exp(-1 * diff_joints_vel.abs().mean(dim=-1).mean(-1))

    current_dof_vel = states["dq"]

    diff_eef_rot = quat_mul(target_eef_quat, quat_conjugate(current_eef_quat))
    diff_eef_rot_angle = quat_to_angle_axis(diff_eef_rot)[0]
    reward_eef_rot = torch.exp(-1 * (diff_eef_rot_angle).abs())

    # object pose reward
    current_obj_pos = states["manip_obj_pos"]
    current_obj_quat = states["manip_obj_quat"]

    target_obj_pos = target_states["manip_obj_pos"]
    target_obj_quat = target_states["manip_obj_quat"]
    diff_obj_pos = target_obj_pos - current_obj_pos
    diff_obj_pos_dist = torch.norm(diff_obj_pos, dim=-1)

    reward_obj_pos = torch.exp(-80 * diff_obj_pos_dist)

    diff_obj_rot = quat_mul(target_obj_quat, quat_conjugate(current_obj_quat))
    diff_obj_rot_angle = quat_to_angle_axis(diff_obj_rot)[0]
    # measure tilt only: angle between object Z axes, ignoring roll around the long axis
    # z_axis = torch.zeros_like(current_obj_pos)
    # z_axis[:, 2] = 1.0
    # current_obj_z = quat_rotate(current_obj_quat, z_axis)
    # target_obj_z  = quat_rotate(target_obj_quat,  z_axis)
    # cos_tilt = torch.clamp((current_obj_z * target_obj_z).sum(dim=-1), -1.0, 1.0)
    # tilt_angle = torch.acos(cos_tilt)
    # reward_obj_rot = torch.exp(-3 * tilt_angle)
    reward_obj_rot = torch.exp(-10 * (diff_obj_rot_angle).abs())

    current_obj_vel = states["manip_obj_vel"]
    target_obj_vel = target_states["manip_obj_vel"]
    diff_obj_vel = target_obj_vel - current_obj_vel
    reward_obj_vel = torch.exp(-1 * diff_obj_vel.abs().mean(dim=-1))

    current_obj_ang_vel = states["manip_obj_ang_vel"]
    target_obj_ang_vel = target_states["manip_obj_ang_vel"]
    diff_obj_ang_vel = target_obj_ang_vel - current_obj_ang_vel
    reward_obj_ang_vel = torch.exp(-1 * diff_obj_ang_vel.abs().mean(dim=-1))

    reward_power = torch.exp(-10 * target_states["power"])
    reward_wrist_power = torch.exp(-2 * target_states["wrist_power"])

    finger_tip_force = target_states["tip_force"]
    finger_tip_distance = target_states["tips_distance"]
    contact_range = [0.02, 0.03]
    finger_tip_weight = torch.clamp(
        (contact_range[1] - finger_tip_distance) / (contact_range[1] - contact_range[0]), 0, 1
    )
    finger_tip_force_masked = finger_tip_force * finger_tip_weight[:, :, None]

    reward_finger_tip_force = torch.exp(-1 * (1 / (torch.norm(finger_tip_force_masked, dim=-1).sum(-1) + 1e-5)))

    error_buf = (
        (torch.norm(current_eef_vel, dim=-1) > 100)
        | (torch.norm(current_eef_ang_vel, dim=-1) > 200)
        | (torch.norm(joints_vel, dim=-1).mean(-1) > 100)
        | (torch.abs(current_dof_vel).mean(-1) > 200)
        | (torch.norm(current_obj_vel, dim=-1) > 100)
        | (torch.norm(current_obj_ang_vel, dim=-1) > 200)
    )  # sanity check

    if training:
        failed_execute = (
            (
                (diff_thumb_tip_pos_dist > (0.04 * noise_compensation) / 0.7 * scale_factor)
                | (diff_index_tip_pos_dist > (0.045 * noise_compensation) / 0.7 * scale_factor)
                | (diff_middle_tip_pos_dist > (0.05 * noise_compensation) / 0.7 * scale_factor)
                | (diff_pinky_tip_pos_dist > (0.06 * noise_compensation) / 0.7 * scale_factor)
                | (diff_ring_tip_pos_dist > (0.06 * noise_compensation) / 0.7 * scale_factor)
                | (diff_level_1_pos_dist > (0.07 * noise_compensation) / 0.7 * scale_factor)
                | (diff_level_2_pos_dist > (0.08 * noise_compensation) / 0.7 * scale_factor)
                | (diff_obj_pos_dist > (0.02) / 0.343 * scale_factor**3)
                | (diff_obj_rot_angle.abs() / np.pi * 180 > 30 / 0.343 * scale_factor**3)
                | torch.any((finger_tip_distance < 0.005) & ~(target_states["tip_contact_state"].any(1)), dim=-1)
            )
            & (running_progress_buf >= 8)
        ) | error_buf
    else:
        # print("AFDDFDSDFDSFSD COMPUTATION OF IMITATION REWARD")
        failed_execute = (
            (
                (diff_thumb_tip_pos_dist > 0.06)
                | (diff_index_tip_pos_dist > 0.06)
                # | (diff_middle_tip_pos_dist > 0.06)
                # | (diff_pinky_tip_pos_dist > 0.8)
                # | (diff_ring_tip_pos_dist > 0.06)
                # | (diff_level_1_pos_dist > 0.08)
                # | (diff_level_2_pos_dist > 0.08)
                | (diff_obj_pos_dist > 0.03)
                | (diff_obj_rot_angle.abs() / np.pi * 180 > 45)
            )
            & (running_progress_buf >= 8)
        ) | error_buf
        failed_execute = failed_execute | error_buf
    reward_execute = (
        0.1 * reward_eef_pos
        + 0.6 * reward_eef_rot
        + 0.9 * reward_thumb_tip_pos
        + 0.8 * reward_index_tip_pos
        + 0.75 * reward_middle_tip_pos
        + 0.6 * reward_pinky_tip_pos
        + 0.6 * reward_ring_tip_pos
        + 0.5 * reward_level_1_pos
        + 0.3 * reward_level_2_pos
        + 10.0 * reward_obj_pos
        + 10.0 * reward_obj_rot
        + 0.1 * reward_eef_vel
        + 0.05 * reward_eef_ang_vel
        + 0.1 * reward_joints_vel
        + 0.1 * reward_obj_vel
        + 0.1 * reward_obj_ang_vel
        + 1.0 * reward_finger_tip_force
        + 0.5 * reward_power
        + 0.5 * reward_wrist_power
    )

    succeeded = (
        progress_buf + 1 + 3 >= max_length
    ) & ~failed_execute  # reached the end of the trajectory, +3 for max future 3 steps
    reset_buf = torch.where(
        succeeded | failed_execute,
        torch.ones_like(reset_buf),
        reset_buf,
    )
    reward_dict = {
        "reward_eef_pos": reward_eef_pos,
        "reward_eef_rot": reward_eef_rot,
        "reward_eef_vel": reward_eef_vel,
        "reward_eef_ang_vel": reward_eef_ang_vel,
        "reward_joints_vel": reward_joints_vel,
        "reward_obj_pos": reward_obj_pos,
        "reward_obj_rot": reward_obj_rot,
        "reward_obj_vel": reward_obj_vel,
        "reward_obj_ang_vel": reward_obj_ang_vel,
        "reward_joints_pos": (
            reward_thumb_tip_pos
            + reward_index_tip_pos
            + reward_middle_tip_pos
            + reward_pinky_tip_pos
            + reward_ring_tip_pos
            + reward_level_1_pos
            + reward_level_2_pos
        ),
        "reward_power": reward_power,
        "reward_wrist_power": reward_wrist_power,
        "reward_finger_tip_force": reward_finger_tip_force,
    }

    return reward_execute, reset_buf, succeeded, failed_execute, reward_dict, error_buf
