from __future__ import annotations

import atexit
import glob
import os
import random
import shutil
import subprocess
import sys
from collections import deque
from enum import Enum
from itertools import cycle
from collections import deque
from time import perf_counter, time
from typing import Dict, List, Tuple

import numpy as np
import torch
from ...utils import torch_jit_utils as torch_jit_utils
from bps_torch.bps import bps_torch
from gym import spaces
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import normalize_angle, quat_apply, quat_conjugate, quat_mul
from copy import deepcopy
import math
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
from main.dataset.factory import ManipDataFactory

# from main.dataset.favor_dataset_dexhand import FavorDatasetDexHand
from main.dataset.oakink2_dataset_dexhand_lh import OakInk2DatasetDexHandLH
from main.dataset.oakink2_dataset_dexhand_rh import OakInk2DatasetDexHandRH
from main.dataset.oakink2_dataset_utils import oakink2_obj_scale, oakink2_obj_mass
from main.dataset.my_dataset_utils import my_dataset_obj_mass

# The thresholds the eval branch of compute_imitation_reward (bottom of this file) compares against.
# They live in their own stdlib-only module because the video wrapper and the offline summariser
# need the same numbers — see lib/utils/eval_thresholds.py for why the jit branch still carries a
# duplicate set of literals.
from lib.utils.eval_thresholds import EVAL_FAILURE_THRESHOLDS, EVAL_FAILURE_WARMUP_STEPS
from main.dataset.transform import aa_to_quat, aa_to_rotmat, quat_to_rotmat, rotmat_to_aa, rotmat_to_quat, rot6d_to_aa
from torch import Tensor
from tqdm import tqdm
from ...asset_root import ASSET_ROOT


from ..core.config import ROBOT_HEIGHT, config
from ...envs.core.sim_config import sim_config
from ...envs.core.vec_task import VecTask
from ...utils.pose_utils import get_mat
from ...utils.big_text import render_big_number
from ...envs.core import viewer_overlay

# The dexRetBaseline controller (see pre_physics_step). Safe at module scope even though
# dex-retargeting is an optional dependency: this module imports only numpy/torch and
# main.dataset.transform, which is already loaded above — the dex_retargeting import itself stays
# lazy inside DexRetargetController.__init__, so an env without it still loads fine.
from baselines.dexret_controller import DexRetargetController, packed_row_by_hand_name
from baselines.utils import (
    DEXRET_FIT_MODE,
    DEXRET_WRIST_FIT,
    DEXRET_WRIST_PULLBACK,
    pull_wrist_back,
)
from maniptrans_envs.lib.envs.core.record_cameras import (
    BEHIND_EYE, BEHIND_TARGET, FRONT_EYE, FRONT_TARGET,
    OVERHEAD_EYE, OVERHEAD_TARGET,
    RECORD_FOV, RECORD_HEIGHT, RECORD_WIDTH,
)
from maniptrans_envs.lib.envs.core import viewer_overlay


# Short labels for the contact bodies, in dexhand.contact_body_names order. Module level so the
# class-body comprehensions that build the grip/pinch column names can see it (a class attribute
# is not visible inside a comprehension's scope).
_TIP_LABELS = ("thumb", "index", "middle", "ring", "pinky")


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
        # dexRetBaseline: drive the hands from a per-frame dex-retargeting solve, not the policy.
        # Built lazily (see dexret_baseline_actions) — the controller reads buffers that do not
        # exist until _create_envs/init_data have run.
        self.dexret_baseline = self.cfg["env"].get("dexRetBaseline", False)
        self.dexret_type = self.cfg["env"].get("dexRetType", "dexpilot")
        # dexRetWristMode: how the baseline drives the wrist. "pid" keeps the env's PID branch
        # (gains on the dexhand class, shared with every PID run); "pd_ff" emits force/torque
        # from gains owned by the controller, with velocity feedforward. See its header.
        self.dexret_wrist_mode = self.cfg["env"].get("dexRetWristMode", "pd_ff")
        # dexRetWristFit: solve the wrist placement from the fingertips rather than sliding it back
        # along the palm axis by a hand-tuned fraction. See baselines/utils/wrist_fit.py.
        self.dexret_wrist_fit = self.cfg["env"].get("dexRetWristFit", DEXRET_WRIST_FIT)
        # dexRetCalibrate: live only — capture the wrist-fit constant this run instead of loading
        # the stored one. train.py stops the loop once the controller reports it is done.
        self.dexret_calibrate = self.cfg["env"].get("dexRetCalibrate", False)
        # dexRetFitMode: "constant" holds one fitted correction; "per_frame" re-solves every step.
        # per_frame needs no calibration, so it is the quickest way to try the fit live.
        self.dexret_fit_mode = self.cfg["env"].get("dexRetFitMode", DEXRET_FIT_MODE)
        # dexRetCalibFrames: length of a LIVE calibration capture, in control steps (0 = the
        # built-in 120). frames/60 is roughly its duration in seconds at the nominal rate.
        self.dexret_calib_frames = int(self.cfg["env"].get("dexRetCalibFrames", 0))
        self.dexret_controller = None
        # Latched once every env has both hands fully handed over. Past that point the retargeting
        # solve is multiplied by zero, so pre_physics_step stops paying its ~1.4 ms of pinocchio NLS
        # per control step -- roughly 8% of a 16.7 ms budget, which live teleop cannot spare. One
        # flag for all envs while the windows behind it are per-env, so ANY reset has to clear it.
        self.dexret_solve_complete = False
        # Per-hand multiplier on that hand's object asset scale (RH = the cap, LH = the bottle
        # body). >1 makes the fingers contact the object before reaching the commanded pose, so the
        # PD position error becomes grip force — the over-closure the imitator cannot produce on
        # its own. The object's MASS is held at its unscaled value (see _create_envs), so this
        # changes only the geometry the fingers close on, not the weight they carry. NOTE:
        # obj_verts/BPS and gt_tips_distance are sampled from the UNSCALED mesh, so a value != 1.0
        # desyncs the residual's shape observation from the physics; the frozen imitators don't see
        # shape at all (their target slice carries no bps/tips), so this stays consistent for
        # zeroResidual runs.
        self.obj_scale_rh = float(self.cfg["env"].get("objScaleRH", 1.0))
        self.obj_scale_lh = float(self.cfg["env"].get("objScaleLH", 1.0))
        # Same idea for the set's non-scored prop, which the two above cannot reach: for cup_brush
        # the scored body is the brush, so resizing the cup needs its own knob. Applied in
        # _create_prop_actor. 1.0 leaves the asset exactly as it loads today.
        self.prop_scale = float(self.cfg["env"].get("propScale", 1.0))
        # sharedObject: spawn ONE object both hands act on, instead of one per hand. See
        # _create_envs — the LH side aliases the RH actor rather than getting its own.
        self.shared_object = bool(self.cfg["env"].get("sharedObject", False))
        self.use_traj_aug = self.cfg["env"].get("useTrajAug", False)
        self.joint_noise_std = self.cfg["env"].get("jointNoiseCm", 0.0) / 100.0  # cm → meters
        self.failure_threshold_noise_compensation = self.cfg["env"].get("failureThresholdNoiseCompensation", 1.0)  # multiplier on finger failure thresholds; 1.0 = no change, >1 loosens to compensate for injected joint noise
        self.obs_hand_noise = self.cfg["env"].get("obsHandNoise", 0.0)
        self.obs_hand_vel_noise = self.cfg["env"].get("obsHandVelNoise", 0.0)
        # random action masking (TeleDexter Sec. C.8) — see config.yaml. Freezes a few DoFs per
        # hand at their previous command so the policy cannot rely on every joint responding on
        # time. Training only, matching the DR block; read from cfg rather than self.training,
        # which is not assigned until further down __init__.
        self.action_mask_prob = (
            self.cfg["env"].get("actionMaskProb", 0.0) if self.cfg["env"]["training"] else 0.0
        )
        self.action_mask_n_dofs = int(self.cfg["env"].get("actionMaskNumDofs", 3))
        self.action_mask_max_duration = int(self.cfg["env"].get("actionMaskMaxDuration", 10))
        self.action_mask_ramp_steps = int(self.cfg["env"].get("actionMaskRampSteps", 64000))
        # --live: stream targets from the laptop (AVP+Motive) instead of the demo buffer.
        # The demo is still loaded (assets/BPS/opt-init/buffer shapes); its target slots are
        # overwritten each step by the latest live frame, broadcast across all envs. See live/.
        self.live = self.cfg["env"].get("live", False)
        self.live_addr = self.cfg["env"].get("liveAddr", "10.50.227.40")
        self.live_port = int(self.cfg["env"].get("livePort", 5555))
        # liveBuffered: FIFO-consume every published frame (faithful trajectory replay) instead
        # of newest-only. Use true when replaying a recording (mock_publish); false for teleop.
        self.live_buffered = self.cfg["env"].get("liveBuffered", False)
        # objectSet: which props are on the table (bottle | cup_brush). Offline the loaders infer it
        # from what the capture recorded; live there is no pkl to infer from, so this names it. The
        # reference demo only supplies buffer shapes and the reset init in live mode, so the set —
        # not the demo — decides which asset each hand spawns, which Motive rigid body feeds each
        # side, and which body is a non-scored prop. See main/dataset/object_sets.py.
        self.live_object_set = None
        if self.live:
            from main.dataset.object_sets import DEFAULT_OBJECT_SET, get_object_set

            self.live_object_set = get_object_set(self.cfg["env"].get("objectSet", DEFAULT_OBJECT_SET))
        # recordDemoData: write the per-step frame provenance CSV during live runs. CSV only — one
        # tuple append per step, so it does not touch the control rate. recordDemoVideo adds viewer
        # capture on top and is what costs 60 Hz -> 30 Hz; see config.yaml. Both restart on a manual
        # reset so the CSV (and any video) covers exactly the last attempt.
        self.record_demo_data = self.cfg["env"].get("recordDemoData", False)
        self.record_demo_video = self.cfg["env"].get("recordDemoVideo", False)
        # liveRateOverlay: draw the achieved control rate in the viewer's top-left during live runs.
        # Live is the mode where the rate is the thing you are watching -- the policy is chasing a
        # 60 Hz stream and falling behind is the failure you need to see immediately, without
        # tabbing to a terminal. Viewer-only and live-only, so nothing else can be affected.
        self.live_rate_overlay = self.cfg["env"].get("liveRateOverlay", True)
        # Window over which the displayed rate is averaged. ~1 s at 60 Hz: long enough that the
        # number is readable rather than flickering, short enough to show a stall as it happens.
        self._rate_samples = deque(maxlen=60)
        self._rate_prev_time = None
        # logPinch gates the fingertip/contact CSV in EVERY mode, live included: it is a
        # diagnostic, and a run should not produce one unasked. MANIPTRANS_PINCH_CSV only chooses
        # where it goes. Offline the "human" fingertips come from the demo's MANO joints rather
        # than the AVP stream — see _fill_demo_pinch_pts.
        self.log_pinch = bool(self.cfg["env"].get("logPinch", False))
        self._pinch_demo_logging = not self.live and self.log_pinch
        # Live only: drop the residual once the cap has met the bottle, so the frozen imitators
        # alone hold the contact instead of the residual fighting it.
        self.live_residual_cutoff = self.cfg["env"].get("liveResidualCutoff", True)
        # residualGateDistance (metres; -1 = off): the distance at which a hand's residual WINDOW
        # opens, so the hand runs imitator -> imitator+residual -> imitator across one manipulation.
        # Measured HAND-TO-ITS-OWN-OBJECT: min(thumb, index) of tips_distance, the human fingertip to
        # the nearest point on that hand's object surface (base.py offline, LiveTargetSource live).
        # It marks the reach->grasp boundary, NOT the capping event -- cap-meets-bottle is an
        # object-to-object test and belongs to liveResidualCutoff. The frozen imitator drives the
        # reach and the retreat; the residual is spent on the span in contact. The reach->grasp
        # transition separates cleanly around ~0.03 m for the capping demos.
        self.residual_gate_distance = self.cfg["env"].get("residualGateDistance", -1.0)
        self.residual_gate_fade_steps = self.cfg["env"].get("residualGateFadeSteps", 12)
        # Hysteresis: the window closes at a WIDER distance than it opens at. Defaulted rather than
        # required because the only wrong answer is one at or below the engage distance, which turns
        # the window into a single threshold the residual chatters across.
        self.residual_gate_release_distance = self.cfg["env"].get("residualGateReleaseDistance", -1.0)
        if self.residual_gate_release_distance < 0:
            self.residual_gate_release_distance = 1.5 * self.residual_gate_distance
        assert self.residual_gate_distance < 0 or (
            self.residual_gate_release_distance > self.residual_gate_distance
        ), (
            f"residualGateReleaseDistance ({self.residual_gate_release_distance}) must exceed "
            f"residualGateDistance ({self.residual_gate_distance}); the window has to close farther "
            f"out than it opens or the residual chatters on the boundary. Leave it at -1 for 1.5x."
        )
        # reachController: who drives a hand while its residual window is shut. See config.yaml.
        # Kept separate from dexRetBaseline on purpose: that flag makes train.py step the env
        # directly and never build the policy, so staging on top of it would crossfade to a hand
        # nothing is driving.
        self.reach_controller = self.cfg["env"].get("reachController", "imitator")
        assert self.reach_controller in ("imitator", "dexret"), (
            f"reachController must be 'imitator' or 'dexret', got '{self.reach_controller}'."
        )
        if self.reach_controller == "dexret":
            assert not self.dexret_baseline, (
                "reachController=dexret and dexRetBaseline=true are mutually exclusive: the "
                "baseline replaces the whole action and train.py never loads the policy, leaving "
                "nothing to crossfade INTO. Drop dexRetBaseline and keep reachController=dexret."
            )
            assert self.residual_gate_distance >= 0, (
                "reachController=dexret needs a window to switch on: set residualGateDistance "
                "(metres, e.g. 0.03). Without one there is no reach phase to hand to the "
                "retargeter."
            )
        # switchModel: arbitrate per hand per frame between the residual and a dex-retargeting solve
        # while that hand's window is OPEN. Complement of reachController, which owns the SHUT span.
        # See config.yaml for why the object and finger terms enter the score differently.
        self.switch_model = bool(self.cfg["env"].get("switchModel", False))
        self.switch_model_obj_weight = float(self.cfg["env"].get("switchModelObjWeight", 0.7))
        self.switch_model_obj_scale = float(self.cfg["env"].get("switchModelObjScale", 0.02))
        self.switch_model_finger_scale = float(self.cfg["env"].get("switchModelFingerScale", 0.05))
        self.switch_model_dwell_steps = int(self.cfg["env"].get("switchModelDwellSteps", 3))
        self.switch_model_log = bool(self.cfg["env"].get("switchModelLog", True))
        # FK chains and the per-hand choice, built lazily like dexret_controller: the chains need the
        # actor dof names, which do not exist until _create_envs has run.
        self.switch_model_chains = None
        self.switch_model_choice = None  # (num_envs, 2) bool, True = that hand runs dex-retargeting
        self.switch_model_held = None  # (num_envs, 2) long, control steps the current choice has held
        self.switch_model_buf = None  # (rows, 13) device buffer, dumped at exit when switchModelLog
        self.switch_model_n = 0
        self.switch_model_solve_required = False  # latches once any window has opened
        if self.switch_model:
            assert self.residual_gate_distance >= 0, (
                "switchModel needs a window to arbitrate over: set residualGateDistance (metres, "
                "e.g. 0.03). Without one there is no manipulation phase, only a reach."
            )
            assert self.reach_controller != "dexret", (
                "switchModel and reachController=dexret both hand spans to the retargeter and would "
                "fight over the same gate weight. reachController owns the SHUT window, switchModel "
                "the OPEN one; pick one."
            )
            assert not self.dexret_baseline, (
                "switchModel and dexRetBaseline=true are mutually exclusive: the baseline replaces "
                "the whole action and train.py never loads the policy, leaving nothing to arbitrate "
                "against. Drop dexRetBaseline and keep switchModel."
            )
            assert 0.0 <= self.switch_model_obj_weight <= 1.0, (
                f"switchModelObjWeight must be in [0, 1], got {self.switch_model_obj_weight}; the "
                f"finger share is 1 - this."
            )
        # causal demo velocities (backward diff + EMA, emulating LiveTargetSource) vs default Gaussian.
        self.causal = self.cfg["env"].get("causal", False)
        self.causal_ema_alpha = self.cfg["env"].get("causalEmaAlpha", 0.3)
        self.causal_mode = self.cfg["env"].get("causalVelMode", "pos_ema")
        self.live_source = None
        if self.live:
            # Live overwrites every demo target slot each step; keep the buffer tiny so that
            # broadcast write is cheap (cur_idx clamps to seq_len-1, so a short demo is fine).
            self.max_demo_length = min(self.max_demo_length, 4)
        self.action_scale = self.cfg["env"]["actionScale"]
        # self.dexhand_rh_dof_noise = self.cfg["env"]["dexhand_rDofNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.training = self.cfg["env"]["training"]
        # Score the eval failure thresholds each step WITHOUT terminating on them, so a doomed
        # episode plays the demo to its last frame instead of being cut at the trip; see
        # EVAL_FAILURE_THRESHOLDS and score_eval_metrics. This is what makes the clause in
        # compute_imitation_reward's eval branch advisory (enforce_eval_thresholds). Eval-only —
        # during training those thresholds are live, so there is nothing to dry-run.
        self.eval_threshold_dry_run = (
            self.cfg["env"].get("evalThresholdDryRun", False) and not self.training
        )
        self.eval_dry_run = None  # per-step scores, refilled by compute_reward while the flag is on
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
        # Per-axis caps on the applied wrist force/torque (see ResDexHand.yaml). <=0 disables.
        self.max_wrist_force = self.cfg["env"].get("maxWristForce", -1.0)
        self.max_wrist_torque = self.cfg["env"].get("maxWristTorque", -1.0)
        # The frozen imitators were calibrated at their Stage-1 training rate (imitatorFps, 60Hz).
        # In the non-PID branch the applied wrist force/torque is base_action * dt * scale, so at a
        # lower control rate (larger self.dt) the SAME base_action would apply a proportionally
        # larger force than the imitator intends. Compute the BASE (imitator) wrist force/torque with
        # the imitator's training dt so the applied force stays rate-invariant; the residual keeps
        # self.dt (it is trained fresh at the current rate). This is a dt_imitator/dt_now scale on the
        # base wrist action and a no-op when the run rate equals imitatorFps.
        self.imitator_dt = 1.0 / float(self.cfg["env"].get("imitatorFps", 60.0))
        # baseWristImpulseInvariant: the wrist force is held for one control step (sim dt), so at a
        # lower control rate the SAME force produces a proportionally larger per-step velocity change
        # than at imitatorFps. Force-invariant (default, factor 1) preserves the imitator's
        # instantaneous force; impulse-invariant scales it by a further imitator_dt/sim_dt so the
        # per-step impulse matches Stage-1 training instead (smoother, but slower tracking).
        # Both are a no-op when the control rate equals imitatorFps.
        sim_dt = float(self.cfg["sim"]["dt"])
        if self.cfg["env"].get("baseWristImpulseInvariant", False):
            self.base_wrist_dt = self.imitator_dt * (self.imitator_dt / sim_dt)
        else:
            self.base_wrist_dt = self.imitator_dt

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
        # debugVis: draw the viewer-mode debug overlays (green demo skeletons, per-body contact
        # coloring, object axes). Pure Python + per-body gym calls, so it costs several ms per
        # step at 1 env — set false for real-time live runs, keep true when eyeballing tracking.
        self.debug_vis = bool(self.cfg["env"].get("debugVis", True))

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
        self.camera_handlers_overhead = [] if self._record else None

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
                target_fps=self.cfg["env"].get("demoTargetFps", None),
                causal=self.causal,
                causal_ema_alpha=self.causal_ema_alpha,
                causal_mode=self.causal_mode,
            )
            self.demo_dataset_rh_dict[dataset_type] = ManipDataFactory.create_data(
                manipdata_type=dataset_type,
                side="right",
                device=self.sim_device,
                mujoco2gym_transf=self.mujoco2gym_transf,
                max_seq_len=self.max_demo_length,
                dexhand=self.dexhand_rh,
                embodiment=self.cfg["env"]["dexhand"],
                target_fps=self.cfg["env"].get("demoTargetFps", None),
                causal=self.causal,
                causal_ema_alpha=self.causal_ema_alpha,
                causal_mode=self.causal_mode,
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
        # kept on self so _log_grip_state can reconstruct the PD's commanded torque
        self.dexhand_rh_dof_stiffness = dexhand_rh_dof_stiffness
        self.dexhand_lh_dof_stiffness = dexhand_lh_dof_stiffness
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

        # Flags are read before the seeded block below, which needs to know if rh_lobj_center is on.
        use_lh_obj_center_aug = self.cfg["env"].get("RH_LObj_Center_Aug", False)
        use_rh_obj_center_aug = self.cfg["env"].get("RH_RObj_Center_Aug", False)
        # Rotate the LH demo (left hand + the left object it holds) about the LH object center.
        use_lh_about_lh_obj_aug = self.cfg["env"].get("LH_LObj_Center_Aug", False)
        # Spin each object in place about world Z, hands unchanged.
        use_obj_rotation_aug = self.cfg["env"].get("useObjRotationAug", False)
        # Default to table-center aug when no per-object aug is selected (backward compat)
        use_table_center_aug = self.cfg["env"].get("RH_LH_Table_Center_Aug",
                                                    not (use_lh_obj_center_aug or use_rh_obj_center_aug
                                                         or use_lh_about_lh_obj_aug or use_obj_rotation_aug))

        if not self.training:
            _rng_state = torch.get_rng_state()
            torch.manual_seed(self.cfg.get("seed", 42))
        aug_transforms = [self._sample_aug_transform(self.device, self._aug_center) for _ in range(num_aug - 1)]
        # object-spin yaws are drawn independently per hand and per variant
        obj_rot_max_deg = self.cfg["env"].get("objRotationAugMaxAngleDeg", 90.0)
        obj_rot_transforms = [(self._sample_yaw(self.device, obj_rot_max_deg),
                               self._sample_yaw(self.device, obj_rot_max_deg))
                              for _ in range(num_aug - 1)]
        # rh_lobj_center and rh_robj_center both rotate the RH demo (about the LH vs RH object
        # center); applying both would double-rotate RH, so they are mutually exclusive. When both
        # flags are set, pick one per variant at random -- sampled here (inside the seeded block)
        # so test mode stays reproducible. True -> rh_lobj_center, False -> rh_robj_center.
        pick_rh_lobj_center = [torch.rand(1).item() < 0.5 for _ in range(num_aug - 1)]
        # rh_lobj_center's yaw is drawn PER DEMO instead of sharing R: the start-pose X shift grows
        # with how far the demo starts from the LH object pivot, so no single scene-wide angle fits
        # every demo. Each is rejection-sampled against rhLObjCenterAugMaxXShift.
        max_x_shift = self.cfg["env"].get("rhLObjCenterAugMaxXShift", 0.05)
        rh_lobj_center_yaws = {}
        if use_lh_obj_center_aug and num_aug > 1:
            for idx in self.dataIndices:
                dt = ManipDataFactory.dataset_type(idx)
                # offsets depend only on the demo, so they are measured once and reused per variant
                ux, uy = self.rh_lobj_center_start_offsets(
                    self.demo_dataset_rh_dict[dt][idx], self.demo_dataset_lh_dict[dt][idx]
                )
                rh_lobj_center_yaws[idx] = [
                    self.sample_rh_lobj_center_yaw(ux, uy, max_x_shift, self.device, str(idx))
                    for _ in range(num_aug - 1)
                ]
        if not self.training:
            torch.set_rng_state(_rng_state)

        if self.use_traj_aug:
            active = [n for n, f in [("RH-L-obj-center", use_lh_obj_center_aug),
                                     ("RH-obj-center", use_rh_obj_center_aug),
                                     ("LH-about-LH-obj", use_lh_about_lh_obj_aug),
                                     (f"obj-rotation(±{obj_rot_max_deg:g}°)", use_obj_rotation_aug),
                                     ("table-center",  use_table_center_aug)] if f]
            assert active, "useTrajAug=true but no aug type is enabled"
            print(f"Trajectory augmentation pipeline: {' -> '.join(active)}")
            if use_lh_obj_center_aug and use_rh_obj_center_aug:
                print("  (RH-L-obj-center and RH-obj-center are mutually exclusive to avoid hand collisions-> one is chosen "
                      "at random per augmented variant)")

        aug_demos_lh = {}  # idx -> [raw, aug_1, ..., aug_{K-1}]
        aug_demos_rh = {}
        for idx in self.dataIndices:
            dt = ManipDataFactory.dataset_type(idx)
            raw_lh = self.demo_dataset_lh_dict[dt][idx]
            raw_rh = self.demo_dataset_rh_dict[dt][idx]
            aug_list_rh = [raw_rh]
            aug_list_lh = [raw_lh]
            for i, ((R, t, c), (R_obj_rh, R_obj_lh)) in enumerate(zip(aug_transforms, obj_rot_transforms)):
                rh, lh = raw_rh, raw_lh
                if use_obj_rotation_aug:
                    rh = self._aug_demo_obj_rotation(rh, R_obj_rh)
                    lh = self._aug_demo_obj_rotation(lh, R_obj_lh)
                # Both rotate the RH demo, so at most one applies per variant; pick_rh_lobj_center[i]
                # picks when both are on. rh_lobj_center uses its own per-demo yaw, not the shared R.
                if use_lh_obj_center_aug and use_rh_obj_center_aug:
                    if pick_rh_lobj_center[i]:
                        rh, lh = self._aug_demo_rh_lobj_center_aug(rh, lh, rh_lobj_center_yaws[idx][i])
                    else:
                        rh = self._aug_demo_rh_robj_center_aug(rh, R)
                elif use_lh_obj_center_aug:
                    rh, lh = self._aug_demo_rh_lobj_center_aug(rh, lh, rh_lobj_center_yaws[idx][i])
                elif use_rh_obj_center_aug:
                    rh = self._aug_demo_rh_robj_center_aug(rh, R)
                if use_lh_about_lh_obj_aug:
                    lh = self._aug_demo_lh_lobj_center_aug(lh, R)
                if use_table_center_aug:
                    # table-center aug ROTATES the whole demo about the table center by R. No
                    # translation: the dataloader re-centers each object onto the table center, so
                    # the sampled XY shift would be undone downstream anyway.
                    rh = self._aug_demo_table_center(rh, R, center=c)
                    lh = self._aug_demo_table_center(lh, R, center=c)
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
        # _inject_live() with the latest live frame (LiveTargetSource.latest()). pack_data now keeps
        # the batch axis (stack_keep_batch), so num_envs==1 works.
        # _inject_live() with the latest live frame (LiveTargetSource.latest()). pack_data now keeps
        # the batch axis (stack_keep_batch), so num_envs==1 works.
        self.demo_data_lh = [segment_data(i, aug_demos_lh) for i in tqdm(range(self.num_envs))]
        self.demo_data_lh = self.pack_data(self.demo_data_lh, side="lh")
        self.demo_data_rh = [segment_data(i, aug_demos_rh) for i in tqdm(range(self.num_envs))]
        self.demo_data_rh = self.pack_data(self.demo_data_rh, side="rh")
        self.env_demo_idx = [i % len(self.dataIndices) for i in range(self.num_envs)]
        if self.live:
            # The props on the table come from objectSet, not from the reference demo — swap their
            # assets in now, before _create_obj_assets loads urdfs and __init__ BPS-encodes
            # obj_verts below.
            self._apply_live_object_set()
        # Both hands scoring the SAME body (a set like cup_brush where the brush is manipulated
        # bimanually, or a capture that tracked one object) means ONE scored actor: spawning two
        # overlapping copies of a body is never a valid scene. Inferred rather than configured —
        # `sharedObject` is purely a reward toggle (see _object_reward_shares).
        self._collapse_obj_actors = all(
            self.demo_data_rh["obj_id"][i] == self.demo_data_lh["obj_id"][i] for i in range(self.num_envs)
        )
        # A prop is a body that is spawned and collided with but never scored (object_sets.prop).
        self._has_prop = "prop_urdf_path" in self.demo_data_rh
        if self._collapse_obj_actors or self._has_prop:
            print(
                f"\033[94m[objects] scored actors: {1 if self._collapse_obj_actors else 2}"
                f"{' (both hands share one body)' if self._collapse_obj_actors else ''}"
                f" | prop: {self.demo_data_rh['prop_obj_id'][0] if self._has_prop else 'none'}\033[0m"
            )

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
            # the prop is one more actor in the aggregate, so its bodies/shapes must be budgeted too
            prop_body_count, prop_shape_count = (
                self._prop_asset_counts(i) if self._has_prop else (0, 0)
            )

            max_agg_bodies = (
                num_dexhand_rh_bodies
                + num_dexhand_lh_bodies
                + 1
                + rh_sum_rigid_body_count
                + lh_sum_rigid_body_count
                + prop_body_count
                + (0 + (0 + self.dexhand_lh.n_bodies * 2 if not self.headless else 0))
            )  # 1 for table
            max_agg_shapes = (
                num_dexhand_rh_shapes
                + num_dexhand_lh_shapes
                + 1
                + rh_sum_rigid_shape_count
                + lh_sum_rigid_shape_count
                + prop_shape_count
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
            if self.camera_handlers_overhead is not None:
                self.camera_handlers_overhead.append(
                    self.create_camera_overhead(env=env_ptr, isaac_gym=self.gym)
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
            if self._collapse_obj_actors:
                # One body for both hands: skip the LH actor and alias the RH one. Both sides then
                # read and write the same root state, and both hands' contacts act on it.
                self.obj_lh_handle = self.obj_rh_handle
            else:
                self.obj_lh_handle, _ = self._create_obj_actor(env_ptr, i, lh_current_asset, side="lh")
            if self._has_prop:
                # Non-scored prop (e.g. the cup the brush is placed into): a free rigid body with the
                # same physics as a scored object, present so the hands and the manipulated body have
                # something to interact with. Never read by obs, reward or the failure check.
                self._create_prop_actor(env_ptr, i)
            self.gym.set_actor_scale(env_ptr, self.obj_rh_handle, rh_obj_scale * self.obj_scale_rh)
            if not self._collapse_obj_actors:
                self.gym.set_actor_scale(env_ptr, self.obj_lh_handle, lh_obj_scale * self.obj_scale_lh)
            obj_props_rh = self.gym.get_actor_rigid_body_properties(env_ptr, self.obj_rh_handle)
            obj_props_lh = self.gym.get_actor_rigid_body_properties(env_ptr, self.obj_lh_handle)
            # set_actor_scale re-derives mass from density, so a scale of s hands back an
            # s^3-heavier object. Undo it: the scale exists to change the GEOMETRY the fingers
            # close on, not the weight they hold — a 1.15x cap would otherwise weigh 1.52x. Done
            # before the clamp/override below so an explicit oakink2_obj_mass still wins verbatim.
            obj_props_rh[0].mass /= self.obj_scale_rh**3
            obj_props_lh[0].mass /= self.obj_scale_lh**3
            obj_props_rh[0].mass = min(0.5, obj_props_rh[0].mass)  # * we only consider the mass less than 500g
            obj_props_lh[0].mass = min(0.5, obj_props_lh[0].mass)  # * we only consider the mass less than 500g

            if rh_obj_mass is not None:
                obj_props_rh[0].mass = rh_obj_mass
            if lh_obj_mass is not None:
                obj_props_lh[0].mass = lh_obj_mass

            # ! Updating the mass and scale might slightly alter the inertia tensor;
            # ! however, because the magnitude of our modifications is minimal, we temporarily neglect this effect.
            # An objScale != 1 is NOT minimal (the density-derived tensor is off by s^5 once the
            # mass is put back), so those runs recompute it; scale 1.0 keeps the original path.
            self.gym.set_actor_rigid_body_properties(
                env_ptr, self.obj_rh_handle, obj_props_rh, recomputeInertia=self.obj_scale_rh != 1.0
            )
            if not self._collapse_obj_actors:  # same actor when shared — writing twice double-applies
                self.gym.set_actor_rigid_body_properties(
                    env_ptr, self.obj_lh_handle, obj_props_lh, recomputeInertia=self.obj_scale_lh != 1.0
                )
            if i == 0:  # once, not per env: the objects as the sim actually built them
                banner = [("cap (RH)", rh_obj_scale, self.obj_scale_rh, "objScaleRH", obj_props_rh)]
                if self._collapse_obj_actors:
                    banner[0] = ("shared", rh_obj_scale, self.obj_scale_rh, "objScaleRH", obj_props_rh)
                else:
                    banner.append(("body (LH)", lh_obj_scale, self.obj_scale_lh, "objScaleLH", obj_props_lh))
                for tag, asset_s, mult, key, props in banner:
                    print(
                        f"\033[94m[scale] {tag} mesh x{asset_s * mult:.3f}"
                        f"  = asset {asset_s:.3f} x {key} {mult:.3f}"
                        f"  |  mass {props[0].mass * 1e3:.1f} g (held at the unscaled value)\033[0m"
                    )
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
        # Shared body: there is no manip_obj_lh actor — point the LH side at the RH one so every
        # per-side read (obs, reward, gating) and write (reset) lands on the one shared body.
        self._manip_obj_lh_handle = (
            self._manip_obj_rh_handle
            if self._collapse_obj_actors
            else self.gym.find_actor_handle(env_ptr, "manip_obj_lh")
        )
        # Non-scored prop: only its root state is touched (placed on reset); nothing reads it.
        # find_actor_handle returns -1 for a missing actor, which would silently index the LAST
        # actor's root state instead of failing, so both the presence and the absence are asserted.
        self._prop_obj_handle = self.gym.find_actor_handle(env_ptr, "prop_obj") if self._has_prop else None
        if self._has_prop:
            assert self._prop_obj_handle >= 0, "objectSet declares a prop but no 'prop_obj' actor was created"
        assert (self.gym.find_actor_handle(env_ptr, "manip_obj_lh") < 0) == self._collapse_obj_actors, (
            "manip_obj_lh must exist iff the two hands score different bodies"
        )
        self._prop_obj_root_state = (
            self._root_state[:, self._prop_obj_handle, :] if self._has_prop else None
        )
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

        # Random action masking state. _action_mask marks the DoFs currently frozen at their
        # previous command; _action_mask_steps counts down the remaining freeze duration. Both are
        # resampled per env independently, so across num_envs the batch sees a wide spectrum of
        # partially-stale joint commands at any instant.
        self._action_mask = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.bool, device=self.device)
        self._action_mask_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Per-hand residual window (see residual_gate_distance). Closed during the reach and again
        # after the retreat; open across the grasp. Held state, not a latch -- residual_gate_weights
        # opens and closes it on two thresholds.
        self.rh_residual_gate_open = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.lh_residual_gate_open = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Ramp position behind each hand's window, in control steps: counts up while the window is
        # open and back down when it closes, so the residual eases in and out instead of switching.
        self.rh_residual_gate_fade = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.lh_residual_gate_fade = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # latched in pre_physics_step once the live cap meets the bottle; cleared in reset_idx
        self._live_residual_latch = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Gate on the latch above: the cap must be seen OFF the bottle before the cutoff may arm.
        # Without it a reset cannot undo the cutoff, because reset_idx re-places both objects at the
        # live OptiTrack pose (see _reset_manip_obj) -- so if the cap is still physically on the
        # bottle, the seating test is true again on the very next step and re-latches immediately.
        # Starts disarmed for the same reason: a run begun with the cap already seated should not
        # cut the residual until the cap has actually been lifted.
        self._live_seating_armed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

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
        # shared body: no manip_obj_lh actor exists, so the LH indices are the RH ones
        self._global_manip_obj_lh_indices = (
            self._global_manip_obj_rh_indices
            if self._collapse_obj_actors
            else torch.tensor(
                [self.gym.find_actor_index(env, "manip_obj_lh", gymapi.DOMAIN_SIM) for env in self.envs],
                dtype=torch.int32,
                device=self.sim_device,
            ).view(self.num_envs, -1)
        )
        self._global_prop_obj_indices = (
            torch.tensor(
                [self.gym.find_actor_index(env, "prop_obj", gymapi.DOMAIN_SIM) for env in self.envs],
                dtype=torch.int32,
                device=self.sim_device,
            ).view(self.num_envs, -1)
            if self._has_prop
            else None
        )

        CONTACT_HISTORY_LEN = 3
        self.rh_tips_contact_history = torch.ones(self.num_envs, CONTACT_HISTORY_LEN, 5, device=self.device).bool()
        self.lh_tips_contact_history = torch.ones(self.num_envs, CONTACT_HISTORY_LEN, 5, device=self.device).bool()

    @staticmethod
    def _sample_aug_transform(device, center):
        # uniform in [-15, +30] degrees
        angle = (torch.rand(1).item() * 45.0 - 15.0) * (np.pi / 180.0)
        cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
        R = torch.tensor([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]],
                         dtype=torch.float32, device=device)
        t = (torch.rand(2, device=device) * 2 - 1) * torch.tensor([0.05, 0.05], device=device)
        t = torch.cat([t, torch.zeros(1, device=device)])
        return R, t, center

    @staticmethod
    def _sample_yaw(device, max_angle_deg):
        """Random yaw rotation about world Z, angle ~ U(-max_angle_deg, +max_angle_deg)."""
        angle = (torch.rand(1).item() * 2 - 1) * (max_angle_deg * np.pi / 180.0)
        cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
        return torch.tensor([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]],
                            dtype=torch.float32, device=device)

    @staticmethod
    def rh_lobj_center_start_offsets(data_rh, data_lh):
        """How far the RH start pose sits from the LH object, in X and Y.

        _aug_demo_rh_lobj_center_aug maps p -> R @ (p - c_t) + c_t about the LH object centre, so a
        point's displacement depends only on its offset u = p - c_t. Measuring u at frame 0 is
        enough to predict any candidate yaw's start shift without running the augmentation.

        Args:
            data_rh: RH demo dict. Covers every world-space position the aug rewrites: wrist_pos,
                opt_wrist_pos, obj_trajectory[:, :3, 3] and each mano_joints entry, all (T, 3).
            data_lh: LH demo dict; only obj_trajectory is read, as the pivot.

        Returns:
            (ux, uy), each a (N,) float tensor of frame-0 X and Y offsets. Z is dropped: a yaw about
            Z cannot displace a point along Z.
        """
        c0 = data_lh["obj_trajectory"][0, :3, 3]  # (3,) pivot at the first frame
        points = [data_rh["wrist_pos"][0], data_rh["opt_wrist_pos"][0],
                  data_rh["obj_trajectory"][0, :3, 3]]
        points += [v[0] for v in data_rh["mano_joints"].values()]
        offsets = torch.stack([(p - c0)[:2] for p in points], dim=0)
        return offsets[:, 0], offsets[:, 1]

    @staticmethod
    def start_abs_x_shift(ux, uy, angle_rad):
        """How far this yaw would move the demo's STARTING X position.

        For a yaw θ about Z, (R - I) u has X component (cosθ - 1) * ux - sinθ * uy.

        Args:
            ux: (N,) X offsets from the pivot at frame 0, from rh_lobj_center_start_offsets.
            uy: (N,) Y offsets from the pivot at frame 0.
            angle_rad: Candidate yaw, radians.

        Returns:
            float, the largest absolute X displacement over the start-pose points (wrist, object,
            joints), in metres -- never a max over time.
        """
        return ((np.cos(angle_rad) - 1.0) * ux - np.sin(angle_rad) * uy).abs().max().item()

    def sample_rh_lobj_center_yaw(self, ux, uy, max_x_shift, device, demo_name, max_attempts=1000):
        """Draw a rh_lobj_center yaw that moves the demo's STARTING X by at most max_x_shift.

        Rejection sampling over U(-15, +15) deg -- symmetric, unlike the U(-15, +30) of
        _sample_aug_transform used by the other aug types, since rotating RH about the LH object
        favours no direction. The feasible band narrows the further a demo starts from that pivot,
        so wide demos get weaker aug; θ -> 0 always passes, so the loop terminates. Only the START
        is bounded -- the augmented trajectory may diverge further later on, by design.

        Args:
            ux: (N,) X offsets from the pivot at frame 0, from rh_lobj_center_start_offsets.
            uy: (N,) Y offsets from the pivot at frame 0.
            max_x_shift: Budget in metres on the X displacement of the starting position.
            device: Torch device for the returned matrix.
            demo_name: Data index, quoted in the assertion so a failure names the offending demo.
            max_attempts: Draws to try before giving up.

        Returns:
            (3, 3) float32 yaw rotation matrix about world Z.
        """
        best = None
        for _ in range(max_attempts):
            angle = (torch.rand(1).item() * 30.0 - 15.0) * (np.pi / 180.0)
            shift = self.start_abs_x_shift(ux, uy, angle)
            best = shift if best is None else min(best, shift)
            if shift <= max_x_shift:
                cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
                return torch.tensor([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]],
                                    dtype=torch.float32, device=device)
        raise AssertionError(
            f"demo '{demo_name}': no yaw in [-15, +15] deg kept the rh_lobj_center START X shift "
            f"within rhLObjCenterAugMaxXShift={max_x_shift} m in {max_attempts} draws (closest was "
            f"{best:.4f} m). It starts {max(ux.abs().max().item(), uy.abs().max().item()):.3f} m "
            f"from the LH object, so raise rhLObjCenterAugMaxXShift or turn RH_LObj_Center_Aug off "
            f"for this demo."
        )

    @staticmethod
    def _aug_demo_obj_rotation(data, R):
        """Spin the object in place about world Z; the hand demo is left untouched.

        obj_pos_t stays put (the pivot is the object's own center), obj_rot_t -> R @ obj_rot_t.
        This deliberately breaks the demonstrated hand<->object grasp alignment: the residual
        policy must handle an object presented at a different yaw than in the demo.
        """
        from copy import copy
        d = copy(data)

        obj = data["obj_trajectory"].clone()
        obj[:, :3, :3] = R.unsqueeze(0) @ obj[:, :3, :3]
        d["obj_trajectory"] = obj
        # linear velocity is unchanged (position is unchanged); spin axis rotates with the object
        d["obj_angular_velocity"] = (R @ data["obj_angular_velocity"].T).T
        return d

    @staticmethod
    def _aug_demo_table_center(data, R, noise_std=0.0, center=None):
        """Return a shallow-copied demo dict with rotation R applied to all world-space fields.
        R is a rotation about `center` (the table center) with NO translation: the sampled
        table-center translation is intentionally dropped because the dataloader always re-centers
        each object onto the table center, so any XY shift applied here is undone downstream.
        """
        from copy import copy
        d = copy(data)

        def rp(x):   # rotate position about the table center, no translation [T, 3]
            if center is not None:
                return (R @ (x - center).T).T + center
            return (R @ x.T).T

        def rv(x):   # rotate velocity / angular velocity [T, 3]
            return (R @ x.T).T

        def raa(x):  # rotate axis-angle [T, 3]
            return rotmat_to_aa(R.unsqueeze(0) @ aa_to_rotmat(x))

        # Positions
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
    def _aug_demo_rh_lobj_center_aug(data_rh, data_lh, R, noise_std=0.0):
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
    def _aug_demo_rh_robj_center_aug(data_rh, R, noise_std=0.0):
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
    def _aug_demo_lh_lobj_center_aug(data_lh, R, noise_std=0.0):
        """Rigidly rotate ONLY the LH demo (left hand + the left object it holds) about the
        LH object center at each timestep. RH demo is left unchanged.

        Pivot c_t = LH object position at frame t, so at each timestep:
            p_lh_aug_t = R @ (p_lh_t - c_t) + c_t     # wrist / joints orbit the object
            obj_pos_t  = c_t   (sits at the pivot -> position unchanged)
            obj_rot_t  = R @ obj_rot_t                # object spins in place with the hand
        The hand<->object relative grasp is preserved (both rotate by R about c_t). This is the
        LH counterpart of _aug_demo_rh_lobj_center_aug / _aug_demo_rh_robj_center_aug; velocity handling
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

        def stack_keep_batch(stack_data):
            # torch.stack, then squeeze singleton dims EXCEPT dim 0 (the per-env batch axis), so
            # num_envs==1 keeps its env axis. A plain .squeeze() collapses [1, ...] -> [...] and
            # breaks the per-env [arange(num_envs), idx] indexing (this is why num_envs >= 2 was required).
            # Identical to .squeeze() for num_envs >= 2, where dim 0 is never a singleton.
            stacked = torch.stack(stack_data)
            for dim in range(stacked.ndim - 1, 0, -1):
                if stacked.shape[dim] == 1:
                    stacked = stacked.squeeze(dim)
            return stacked

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
            return stack_keep_batch(stack_data)

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
                    packed_data[k] = stack_keep_batch(stack_data)
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

    def _sample_obj_verts(self, mesh_path):
        """The 1000-point cloud behind the BPS shape observation and tips_distance, for one mesh.

        Goes through the dataset's own `random_sampling_pc`, which seeds itself, so the result is
        identical to what the loader would have produced for the same mesh offline.
        """
        import trimesh
        from pytorch3d.structures import Meshes

        sampler = next(iter(self.demo_dataset_rh_dict.values()))  # only reads self.device
        obj_mesh = trimesh.load(mesh_path, force="mesh", process=False)
        return sampler.random_sampling_pc(
            Meshes(
                verts=torch.from_numpy(np.asarray(obj_mesh.vertices)[None].astype(np.float32)),
                faces=torch.from_numpy(np.asarray(obj_mesh.faces)[None].astype(np.float32)),
            )
        )

    def _apply_live_object_set(self):
        """Live: point the demo buffers' object identity at the props `objectSet` names.

        demo_data_{rh,lh} are built from the reference demo (dataIndices), which in live mode exists
        only for buffer shapes and the retargeted reset init — its objects are whatever was captured,
        not necessarily what is on the table now. For each side whose body differs, swap in the set's
        obj_id (which asset _create_obj_assets loads), obj_urdf_path (the COACD urdf) and obj_verts.
        A side already holding the right body is left untouched, so the default bottle set stays a
        no-op. Sets with a prop also get the `prop_*` slots the offline loaders emit, so live and
        offline read the prop from exactly the same place.
        """
        for side, demo in (("rh", self.demo_data_rh), ("lh", self.demo_data_lh)):
            obj = self.live_object_set.side(side)
            if all(demo_obj_id == obj.asset_id for demo_obj_id in demo["obj_id"]):
                continue
            mesh_path, urdf_path = obj.assets()
            verts = self._sample_obj_verts(mesh_path)
            print(
                f"\033[94m[live] {side.upper()} object '{demo['obj_id'][0]}' (reference demo) -> "
                f"'{obj.asset_id}' ({self.live_object_set.name} set): {urdf_path}\033[0m"
            )
            demo["obj_id"] = [obj.asset_id] * self.num_envs
            demo["obj_urdf_path"] = [urdf_path] * self.num_envs
            demo["obj_verts"] = verts[None].expand(self.num_envs, -1, -1).contiguous()

        prop = self.live_object_set.prop
        if prop is None:
            # the reference demo may itself have carried a prop the live set does not want
            for demo in (self.demo_data_rh, self.demo_data_lh):
                for k in ("prop_obj_id", "prop_urdf_path", "prop_trajectory", "prop_static"):
                    demo.pop(k, None)
            return
        prop_urdf = prop.assets()[1]
        print(f"\033[94m[live] prop '{prop.asset_id}' ({self.live_object_set.name} set): {prop_urdf}\033[0m")
        for demo in (self.demo_data_rh, self.demo_data_lh):
            demo["prop_obj_id"] = [prop.asset_id] * self.num_envs
            demo["prop_urdf_path"] = [prop_urdf] * self.num_envs
            # from the LIVE set, not the reference demo: the reference may carry a different set's
            # prop (or none at all, in which case the offline loaders never emitted this key and
            # _create_prop_actor would KeyError on it)
            demo["prop_static"] = [self.live_object_set.prop_static] * self.num_envs
            if "prop_trajectory" not in demo:
                # the reference demo has no prop trajectory; allocate the slot _inject_live writes
                # into each step, shaped like obj_trajectory ([num_envs, nT, 4, 4])
                demo["prop_trajectory"] = (
                    torch.eye(4, device=self.device)
                    .expand_as(demo["obj_trajectory"])
                    .clone()
                )

    def _load_obj_asset(self, obj_id, obj_urdf_path, fix_base_link=False):
        """Load (and cache) a manipulable body's asset.

        Shared by the scored objects and the non-scored prop so the two get identical physics — the
        prop has to behave like a real object the hands and the manipulated body collide with.
        fix_base_link=True instead pins the body in place: full collision geometry and friction, but
        infinite effective mass, so nothing can push it (this is how the table is spawned). It is
        part of the cache key because the same obj_id can legitimately be wanted both ways — scored
        and free in one set, a pinned prop in another — and one asset cannot be both.
        """
        cache_key = (obj_id, fix_base_link)
        if cache_key in self.objs_assets:
            return self.objs_assets[cache_key]
        asset_options = gymapi.AssetOptions()
        asset_options.override_com = True
        asset_options.override_inertia = True
        asset_options.convex_decomposition_from_submeshes = True
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.thickness = 0.001
        asset_options.max_linear_velocity = 50
        asset_options.max_angular_velocity = 100
        asset_options.fix_base_link = fix_base_link
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params = gymapi.VhacdParams()
        asset_options.vhacd_params.resolution = 200000
        asset_options.density = 200  # * the average density of low-fill-rate 3D-printed models
        current_asset = self.gym.load_asset(self.sim, *os.path.split(obj_urdf_path), asset_options)

        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(current_asset)
        for element in rigid_shape_props_asset:
            element.friction = 2.0  # * We increase the friction coefficient to compensate for missing skin deformation friction in simulation. See the Appx for details.
            element.rolling_friction = 0.05
            element.torsion_friction = 0.05
        self.gym.set_asset_rigid_shape_properties(current_asset, rigid_shape_props_asset)
        self.objs_assets[cache_key] = current_asset
        return current_asset

    def _prop_asset_counts(self, i):
        """(rigid body count, rigid shape count) of the prop asset, for the aggregate budget."""
        # Same fix_base_link as _create_prop_actor will use, so this hits the same cache entry
        # instead of loading a second copy of the mesh just to count it.
        asset = self._load_obj_asset(
            self.demo_data_rh["prop_obj_id"][i],
            self.demo_data_rh["prop_urdf_path"][i],
            fix_base_link=self.demo_data_rh["prop_static"][i],
        )
        return (
            self.gym.get_asset_rigid_body_count(asset),
            self.gym.get_asset_rigid_shape_count(asset),
        )

    def _create_prop_actor(self, env_ptr, i):
        """Spawn the set's non-scored prop, free or pinned as the object set asks.

        Placed at its first demo/live frame; `reset_idx` re-places it from `prop_trajectory` after
        that. Nothing else touches it — it exists purely so the hands and the manipulated body have
        something physical to interact with (the cup the brush is set into, say).

        With `prop_static` (see main/dataset/object_sets.py) the body is pinned instead: it still
        collides and carries friction, but nothing can move it. The pose it is created at is then
        the pose it keeps, which is only right for a prop the capture shows standing still — the cup
        spans 1.7 mm across a whole take. `_reset_prop` still writes the pose, harmlessly: PhysX
        ignores root-state writes to a fixed base, and the target pose is the one it already has.
        """
        prop_static = self.demo_data_rh["prop_static"][i]
        asset = self._load_obj_asset(
            self.demo_data_rh["prop_obj_id"][i],
            self.demo_data_rh["prop_urdf_path"][i],
            fix_base_link=prop_static,
        )
        transf = self.demo_data_rh["prop_trajectory"][i][0]
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(transf[0, 3], transf[1, 3], transf[2, 3])
        aa = rotmat_to_aa(transf[:3, :3])
        angle = torch.norm(aa)
        axis = aa / angle if angle > 1e-8 else torch.tensor([0.0, 0.0, 1.0], device=aa.device)
        pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(axis[0], axis[1], axis[2]), angle)
        # collision filter 0, same as the scored objects: collides with the hands and everything else
        actor = self.gym.create_actor(env_ptr, asset, pose, "prop_obj", i, 0)
        # Resize the prop before the mass block below, not after: set_actor_scale re-derives mass
        # from density, so doing it later would silently undo an explicit my_dataset_obj_mass. The
        # prop's origin is its base, so this grows it away from the table rather than through it.
        if self.prop_scale != 1.0:
            self.gym.set_actor_scale(env_ptr, actor, self.prop_scale)
        # Mass comes from my_dataset_obj_mass, the way the scored objects take theirs from
        # oakink2_obj_mass; without an entry the prop keeps whatever asset_options.density implies
        # from its geometry, which for a receptacle is far too light (see my_dataset_utils).
        # A fixed base has infinite effective mass, so the table entry is inert while prop_static is
        # on -- skipped rather than written, so nothing reads back a mass the solver never uses.
        prop_mass = (
            None if prop_static else my_dataset_obj_mass.get(self.demo_data_rh["prop_obj_id"][i])
        )
        if prop_mass is not None:
            props = self.gym.get_actor_rigid_body_properties(env_ptr, actor)
            for body in props:
                body.mass = prop_mass / len(props)
            self.gym.set_actor_rigid_body_properties(env_ptr, actor, props, recomputeInertia=True)
        return actor

    def _create_obj_assets(self, i, side="rh"):
        if side == "rh":
            obj_id = self.demo_data_rh["obj_id"][i]
        else:
            obj_id = self.demo_data_lh["obj_id"][i]

        if side == "rh":
            obj_urdf_path = self.demo_data_rh["obj_urdf_path"][i]
        else:
            obj_urdf_path = self.demo_data_lh["obj_urdf_path"][i]
        current_asset = self._load_obj_asset(obj_id, obj_urdf_path)

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
                # debugVis off: the markers are never updated, so park them below the ground
                # instead of leaving a stale clump of dots at the env origin
                marker_transform = gymapi.Transform()
                if not self.debug_vis:
                    marker_transform.p = gymapi.Vec3(0.0, 0.0, -10.0)
                a = self.gym.create_actor(
                    env_ptr,
                    joint_point,
                    marker_transform,
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

    def _object_reward_shares(self):
        """Each hand's share of the object reward terms, as a per-env scalar.

        Without sharedObject every hand has its own body, so each takes the full term (1.0) and
        the reward is bit-identical to before. With ONE shared body the object would otherwise be
        rewarded twice — once per hand — which both doubles its weight against the hand-tracking
        terms and keeps charging a hand that has already let go.

        The share follows the DEMO's fingertip-to-object distances, i.e. the same signal and the
        same 2-3 cm contact band that finger_tip_weight already uses, so a handover moves through
        it smoothly: the left hand carries the object term while it holds, both share it 50/50
        while they are on the object together, and the right hand carries it once the left
        releases. Frames where neither hand is near the object split evenly, so the object term
        never vanishes and the two shares always sum to 1.
        """
        ones = torch.ones(self.num_envs, device=self.device)
        if not self.shared_object:
            return {"rh": ones, "lh": ones}
        idx = torch.arange(self.num_envs, device=self.device)
        near = {}
        for side, demo in (("rh", self.demo_data_rh), ("lh", self.demo_data_lh)):
            tips = demo["tips_distance"][idx, self.progress_buf]  # [N, 5] demo tip -> object surface
            near[side] = torch.clamp((0.03 - tips) / (0.03 - 0.02), 0.0, 1.0).amax(dim=-1)
        total = near["rh"] + near["lh"]
        return {s: torch.where(total > 1e-6, near[s] / total.clamp(min=1e-6), 0.5 * ones) for s in ("rh", "lh")}

    def compute_reward(self, actions):
        # both sides need each other's proximity to split the shared object's reward, so resolve it
        # once here rather than inside either side
        self._obj_reward_share = self._object_reward_shares()
        if self.eval_threshold_dry_run:
            # Rebuilt from scratch every step, holding no history: the wrapper that consumes it
            # (WandbVideoCaptureWrapper) reads it the moment step() returns and keeps its own
            # per-episode series, so there is nothing here to reset when an env resets.
            # running_progress_buf is captured pre-increment (post_physics_step bumps it after the
            # reward), so it is the step index the thresholds were scored at.
            self.eval_dry_run = {
                "thresholds": EVAL_FAILURE_THRESHOLDS,
                "warmup_steps": EVAL_FAILURE_WARMUP_STEPS,
                "running_progress": self.running_progress_buf.clone(),
                "metrics": {},
            }
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
            self._obj_reward_share[side],
            self.training,
            not self.eval_threshold_dry_run,
        )
        if not self.training and failure_buf[0].item():
            self._print_failure_reason(side, side_states, target_state, scale_factor, error_buf)
        if self.eval_threshold_dry_run:
            # side-prefixed into the flat dict compute_reward opened above: the thresholds are
            # per-hand, and either hand tripping is what would have ended the episode
            for name, value in self.score_eval_metrics(side, side_states, target_state).items():
                self.eval_dry_run["metrics"][f"{side}_{name}"] = value
        self.total_rew_buf += rew_buf
        return rew_buf, reset_buf, success_buf, failure_buf, reward_dict, error_buf

    def _print_failure_reason(self, side, side_states, target_state, scale_factor, error_buf):
        if self.live:
            return
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

    def score_eval_metrics(self, side, side_states, target_state):
        """Measure one hand's tracking error every step without terminating anything.

        Recomputes the four terms of EVAL_FAILURE_THRESHOLDS for every env, so a rollout that the
        eval branch would have cut short can report where it lost the trajectory and still play to
        the end of the demo. Unlike _print_failure_reason, which fires only on a real failure and
        only for env 0, this runs every step for every env and returns the raw numbers instead of
        printing a verdict.

        Also returns the two wrist terms, which no failure branch scores. They are diagnostic: the
        fingertip errors are world-frame distances, so a wrist that drifts off the demo drags every
        fingertip with it and looks exactly like fingers that curled wrong. Reading the two together
        is what separates the cases.

        Args:
            side: "rh" or "lh"; selects the hand whose weight indices are used.
            side_states: this hand's state dict, exactly as handed to compute_imitation_reward —
                needs "joints_state" (N, J+1, 13), "base_state" (N, 13), "manip_obj_pos" (N, 3) and
                "manip_obj_quat" (N, 4) xyzw.
            target_state: this hand's demo targets at the same step — needs "joints_pos" (N, J, 3),
                "wrist_pos" (N, 3), "wrist_quat" (N, 4) xyzw, "manip_obj_pos" (N, 3) and
                "manip_obj_quat" (N, 4) xyzw.

        Returns:
            dict of metric name -> (N,) float tensor, covering every key in EVAL_DRY_RUN_PANELS.
        """
        weight_idx = (self.dexhand_rh if side == "rh" else self.dexhand_lh).weight_idx
        # joints_state carries the wrist at index 0, which weight_idx counts but the demo targets
        # do not — hence the [:, 1:] and the k - 1, matching compute_imitation_reward.
        joints_pos = side_states["joints_state"][:, 1:, :3]
        diff_joints_pos_dist = torch.norm(target_state["joints_pos"] - joints_pos, dim=-1)

        def mean_over(finger):
            """Mean tracking error over one finger's weight indices, as the reward does.

            Args:
                finger: key into the hand's weight_idx table, e.g. "thumb_tip".

            Returns:
                (N,) float tensor of mean distances in metres.
            """
            return diff_joints_pos_dist[:, [k - 1 for k in weight_idx[finger]]].mean(dim=-1)

        diff_obj_rot = quat_mul(
            target_state["manip_obj_quat"], quat_conjugate(side_states["manip_obj_quat"])
        )
        # base_state is the wrist root, the same pair reward_eef_pos/reward_eef_rot score
        diff_eef_rot = quat_mul(
            target_state["wrist_quat"], quat_conjugate(side_states["base_state"][:, 3:7])
        )
        return {
            "diff_thumb_tip_pos_dist": mean_over("thumb_tip"),
            "diff_index_tip_pos_dist": mean_over("index_tip"),
            "diff_obj_pos_dist": torch.norm(
                target_state["manip_obj_pos"] - side_states["manip_obj_pos"], dim=-1
            ),
            "diff_obj_rot_angle": quat_to_angle_axis(diff_obj_rot)[0].abs() / np.pi * 180,
            "diff_eef_pos_dist": torch.norm(
                target_state["wrist_pos"] - side_states["base_state"][:, :3], dim=-1
            ),
            "diff_eef_rot_angle": quat_to_angle_axis(diff_eef_rot)[0].abs() / np.pi * 180,
        }

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
        if self.live:
            # Pull the latest live frame so the wrist + object (bottle/cap) slots hold the current
            # OptiTrack pose before the per-side reset reads them — objects reset to where OptiTrack
            # sees them, including the very first reset (before the first post_physics_step inject).
            self._inject_live()
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
        self._reset_prop(env_ids, seq_idx)

        dexhand_multi_env_ids_int32 = torch.concat(
            [
                self._global_dexhand_rh_indices[env_ids].flatten(),
                self._global_dexhand_lh_indices[env_ids].flatten(),
            ]
        )
        # shared body: both sides index the same actor, so submit it once — set_actor_root_state_
        # tensor_indexed takes a list of DISTINCT actors, and a repeated index is wasted work.
        manip_obj_multi_env_ids_int32 = (
            self._global_manip_obj_rh_indices[env_ids].flatten()
            if self._collapse_obj_actors
            else torch.concat(
                [
                    self._global_manip_obj_rh_indices[env_ids].flatten(),
                    self._global_manip_obj_lh_indices[env_ids].flatten(),
                ]
            )
        )
        if self._has_prop:
            manip_obj_multi_env_ids_int32 = torch.concat(
                [manip_obj_multi_env_ids_int32, self._global_prop_obj_indices[env_ids].flatten()]
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
        # a fresh episode must not inherit a freeze: prev_targets was just zeroed, so a surviving
        # mask would hold those DoFs at 0 rather than at a real previous command
        self._action_mask[env_ids] = False
        self._action_mask_steps[env_ids] = 0

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
            # Start fingers at the dexhand's default pose, then flag these envs to snap onto the
            # frozen imitator's base_action on the next pre_physics_step — a one-shot policy-output
            # init that shortcuts opt_dof_pos (the imitator supplies the retargeted finger pose).
            # Auto-reset is disabled in live, so this init only happens once at startup.
            default = getattr(self, f"dexhand_{side}_default_dof_pos").to(self.device)
            dof_pos = default.unsqueeze(0).expand(env_ids.shape[0], -1).clone()
            dof_vel = torch.zeros_like(dof_pos)
            if not hasattr(self, "_snap_fingers_to_base_action"):
                self._snap_fingers_to_base_action = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            # MANIPTRANS_DISABLE_SNAP=1 skips the one-shot finger snap, leaving fingers at the open
            # default pose so the frozen imitator closes them gradually via the PD — avoids the
            # snap's instantaneous grip closure penetrating the object and blowing up the contact
            # solver (ported from 39a9100 on fixed_vel_calc).
            if os.environ.get("MANIPTRANS_DISABLE_SNAP", "0") != "1":
                self._snap_fingers_to_base_action[env_ids] = True
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

        wrist_seq_idx = seq_idx
        if self.live:
            # Live has no retargeting → opt_wrist_* is the stale reference demo's, not the live
            # target. _inject_live only refreshes wrist_* (the human wrist), which is also what the
            # imitator tracks every step and where the base converges. Reset to that so the hand
            # snaps to the current live wrist instead of teleporting to the reference retarget pose.
            wrist_key = "wrist"
        elif self.dexret_baseline:
            # The retargeting controller commands the wrist toward the HUMAN wrist, so starting at
            # the retargeted pose hands it a standing error on step 0 — and with the PD gains high
            # enough to matter, that error saturates the force command and the base never recovers.
            # Start on the human wrist, at frame 0, so the controller opens with zero error.
            wrist_key = "wrist"
            wrist_seq_idx = torch.zeros_like(seq_idx)
        else:
            wrist_key = "opt_wrist"
        opt_wrist_pos = side_demo_data[f"{wrist_key}_pos"][env_ids, wrist_seq_idx]
        opt_wrist_rot = aa_to_quat(side_demo_data[f"{wrist_key}_rot"][env_ids, wrist_seq_idx])
        opt_wrist_rot = opt_wrist_rot[:, [1, 2, 3, 0]]

        # Not for dexRetType=position: there the controller's free joint solves the standoff
        # itself, measured from the human wrist, so a pullback here would displace the hand twice
        # and the reset would disagree with the target on every subsequent step. Same reasoning for
        # DEXRET_WRIST_FIT, which likewise solves the placement per frame — and unlike the pullback
        # its answer is not known here, since it needs the solved finger pose. Spawning on the raw
        # wrist and letting the controller pull the hand in is the consistent choice for both.
        if (self.dexret_baseline and DEXRET_WRIST_PULLBACK and not self.dexret_wrist_fit
                and self.dexret_type != "position"):
            # The controller aims at the wrist pulled back along the palm axis on EVERY step, so
            # reset there too. Landing on the raw wrist instead leaves the hand exactly one
            # pullback off target at step 0, which shows up as a spike the controller then
            # has to drive out — an artefact of the reset, not of the retargeting.
            packed = packed_row_by_hand_name(getattr(self, f"dexhand_{side}"))
            middle_pos = side_demo_data["mano_joints"][env_ids, wrist_seq_idx].reshape(
                len(env_ids), -1, 3
            )[:, packed["middle_proximal"]]
            opt_wrist_pos = pull_wrist_back(opt_wrist_pos, middle_pos, DEXRET_WRIST_PULLBACK)

        if self.live or self.dexret_baseline:
            # Zero the base velocity on reset (like the finger DOFs above) so a mid-motion reset
            # doesn't kick the free-floating base with the hand's velocity. For the baseline this
            # also matches its feedforward, which is re-seeded to zero on the same step.
            opt_wrist_vel = torch.zeros_like(opt_wrist_pos)
            opt_wrist_ang_vel = torch.zeros_like(opt_wrist_pos)
        else:
            opt_wrist_vel = side_demo_data[f"{wrist_key}_velocity"][env_ids, seq_idx]
            opt_wrist_ang_vel = side_demo_data[f"{wrist_key}_angular_velocity"][env_ids, seq_idx]

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

        # reset manip obj — in live mode obj_trajectory holds the live OptiTrack pose (injected in
        # _reset_default above), so the object is placed exactly where OptiTrack sees it.
        obj_pos_init = side_demo_data["obj_trajectory"][env_ids, seq_idx, :3, 3]
        obj_rot_init = side_demo_data["obj_trajectory"][env_ids, seq_idx, :3, :3]
        obj_rot_init = rotmat_to_quat(obj_rot_init)
        # [w, x, y, z] to [x, y, z, w]
        obj_rot_init = obj_rot_init[:, [1, 2, 3, 0]]

        if self.live:
            # Hold the object at rest at the OptiTrack pose — don't inherit the live EMA velocity,
            # which would fling the free object right after the snap (mirrors the wrist above).
            obj_vel = torch.zeros_like(obj_pos_init)
            obj_ang_vel = torch.zeros_like(obj_pos_init)
        else:
            obj_vel = side_demo_data["obj_velocity"][env_ids, seq_idx]
            obj_ang_vel = side_demo_data["obj_angular_velocity"][env_ids, seq_idx]

        manip_obj_root_state = getattr(self, f"_manip_obj_{side}_root_state")

        manip_obj_root_state[env_ids, :3] = obj_pos_init
        manip_obj_root_state[env_ids, 3:7] = obj_rot_init
        manip_obj_root_state[env_ids, 7:10] = obj_vel
        manip_obj_root_state[env_ids, 10:13] = obj_ang_vel

    def _reset_prop(self, env_ids, seq_idx):
        """Place the non-scored prop and leave it at rest.

        This is the only thing that ever writes the prop's pose: live, `_inject_live` keeps
        `prop_trajectory` current but the body is only moved here, on an explicit reset. Always at
        rest — the prop is a receptacle, and inheriting a demo/EMA velocity would send it sliding.
        """
        if not self._has_prop:
            return
        transf = self.demo_data_rh["prop_trajectory"][env_ids, seq_idx]
        self._prop_obj_root_state[env_ids, :3] = transf[:, :3, 3]
        self._prop_obj_root_state[env_ids, 3:7] = rotmat_to_quat(transf[:, :3, :3])[:, [1, 2, 3, 0]]
        self._prop_obj_root_state[env_ids, 7:13] = 0.0

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
        # re-close the live approach residual gate so the reach phase (imitator-only) runs again
        self.rh_residual_gate_open[env_ids] = False
        self.lh_residual_gate_open[env_ids] = False
        # residual back off for the fresh reach; the gate flags above were just cleared with it
        self.rh_residual_gate_fade[env_ids] = 0.0
        self.lh_residual_gate_fade[env_ids] = 0.0
        # The skip flag is one bool for every env while the windows above are per-env, so any reset
        # at all can un-finish the handover -- clear it and let the next steps re-check. Re-checking
        # an env that was never finished is free. The solve also has to forget its step history:
        # the steps it was skipped for are a gap, and the envs that just reset are exactly the ones
        # the resumed solve drives at full authority. Confined to reachController=dexret so a plain
        # dexRetBaseline run keeps its existing warm start.
        self.dexret_solve_complete = False
        # Same for switchModel's mirror latch, and for the arbiter's own state: a reset puts every
        # hand back before its window, so the held choice and its dwell describe an episode that no
        # longer exists and would otherwise carry a stale selection into the new reach.
        self.switch_model_solve_required = False
        if self.switch_model_choice is not None:
            self.switch_model_choice[:] = False
            self.switch_model_held[:] = 0
        # forget the announced window state so a fresh episode re-announces from scratch
        self._residual_window_state_rh = None
        self._residual_window_state_lh = None
        if self.reach_controller == "dexret" and self.dexret_controller is not None:
            self.dexret_controller.reset_step_history()
        # Post-reset diagnostics + stabilizers (ported from 39a9100 on fixed_vel_calc):
        # print per-hand velocities / contact forces / applied wrist force for the first steps
        # after a reset, and optionally hold the residual at zero (MANIPTRANS_RESIDUAL_WARMUP=N)
        # so the frozen imitator eases the hand into an in-distribution pose before it engages.
        self._post_reset_debug_window = 30
        self._post_reset_debug_steps = self._post_reset_debug_window
        self._residual_warmup_steps = int(os.environ.get("MANIPTRANS_RESIDUAL_WARMUP", "0"))
        # re-arm the live residual cutoff for the envs that just restarted. Clearing the latch alone
        # is not enough: the objects were just re-placed at the live OptiTrack pose, so a cap still
        # physically on the bottle re-latches on the next step. Disarm too, and require the cap to
        # be lifted before the cutoff can fire again.
        self._live_residual_latch[env_ids] = False
        self._live_seating_armed[env_ids] = False
        self._live_residual_cut = False

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

    def _thick_line_segments(self, segment, radius=0.002, ring_count=6):
        """Offset copies of one segment that make it render ~4x thicker; returns (n, 2, 3) float32.

        gym.add_lines has no line-width control, so we ring `ring_count` copies at `radius`
        around the segment (in the plane perpendicular to it) plus the original centre line.
        `segment` is a (2, 3) array of the two endpoints.
        """
        segment = np.asarray(segment, dtype=np.float32)
        direction = segment[1] - segment[0]
        length = np.linalg.norm(direction)
        if length < 1e-8:
            return segment[None]
        direction = direction / length
        # two unit vectors spanning the plane perpendicular to the segment
        reference = np.array([1.0, 0.0, 0.0]) if abs(direction[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        perp_u = np.cross(direction, reference)
        perp_u = perp_u / (np.linalg.norm(perp_u) + 1e-8)
        perp_v = np.cross(direction, perp_u)
        offsets = [np.zeros(3, dtype=np.float32)]
        for i in range(ring_count):
            angle = 2.0 * np.pi * i / ring_count
            offsets.append((radius * (np.cos(angle) * perp_u + np.sin(angle) * perp_v)).astype(np.float32))
        return np.stack([segment + offset for offset in offsets]).astype(np.float32)

    def _add_thick_lines(self, env_ptr, segment, color, radius=0.002, ring_count=6):
        """Draw one segment ~4x thicker — all offset copies in a single add_lines call."""
        segments = self._thick_line_segments(segment, radius=radius, ring_count=ring_count)
        segment_colors = np.repeat(np.asarray(color, dtype=np.float32).reshape(1, 3), len(segments), axis=0)
        self.gym.add_lines(self.viewer, env_ptr, len(segments), segments.reshape(-1, 3), segment_colors)

    def draw_rate_overlay(self):
        """Show the achieved control rate in the viewer's top-left, in yellow.

        Measured return-to-return between successive pre_physics_step calls, so it includes policy
        inference and the runner's overhead -- the rate the loop actually achieves, which is the
        number that matters when the policy is chasing a 60 Hz live stream. It carries no cuda
        syncs of its own, so unlike MANIPTRANS_STEP_TIMING it does not slow down what it measures.

        Isaac Gym has no 2D text API, so this is a billboard of world-space line segments
        re-anchored to the live camera every frame (see core/viewer_overlay.py). Drawn AFTER the
        debug_vis block so its clear_lines cannot wipe it; when debug_vis is off this owns the
        clear instead, otherwise last frame's glyphs would accumulate.

        Returns:
            None. No-op until two steps have been timed.
        """
        now = perf_counter()
        if self._rate_prev_time is not None:
            self._rate_samples.append(now - self._rate_prev_time)
        self._rate_prev_time = now
        if not self._rate_samples:
            return
        mean_dt = sum(self._rate_samples) / len(self._rate_samples)
        hz = 1.0 / max(mean_dt, 1e-6)

        if not self.debug_vis:
            self.gym.clear_lines(self.viewer)

        # Ask for the camera in env 0's frame, which is the frame add_lines wants its vertices in,
        # so no origin arithmetic is needed and the two cannot disagree.
        env_ptr = self.envs[0]
        camera = self.gym.get_viewer_camera_transform(self.viewer, env_ptr)
        # Isaac Gym's camera looks along its local +z -- see isaacgym/python/examples/projectiles.py,
        # which fires along r.rotate(Vec3(0, 0, 1)) from the camera position.
        forward = camera.r.rotate(gymapi.Vec3(0.0, 0.0, 1.0))
        size = self.gym.get_viewer_size(self.viewer)
        aspect = size.x / max(size.y, 1)

        # 4.5 cm at 1 m under a 90 deg fov is ~8% of the view height -> ~36 px tall in a 900 px
        # window: legible after a screen capture is downscaled, without covering the hands.
        text_height = 0.045
        origin, right, up = viewer_overlay.corner_anchor(
            np.array([camera.p.x, camera.p.y, camera.p.z]),
            np.array([forward.x, forward.y, forward.z]),
            aspect=aspect,
            text_height=text_height,
        )
        segments = viewer_overlay.text_segments(f"{hz:.1f} HZ", origin, right, up, text_height)
        if not len(segments):
            return
        # Thicken with the same helper the skeleton overlay uses, so both read alike on video.
        thick = np.concatenate(
            [self._thick_line_segments(seg, radius=text_height * 0.03, ring_count=4) for seg in segments]
        )
        colors = np.tile(np.array([[1.0, 1.0, 0.0]], dtype=np.float32), (len(thick), 1))
        self.gym.add_lines(self.viewer, env_ptr, len(thick), thick.reshape(-1, 3), colors)

    def dexret_baseline_actions(self):
        """Solve this step's action with dex-retargeting instead of taking it from the policy.

        Constructs the controller on first call rather than in __init__: it reads the packed demo
        buffers, the per-hand dof limits and the rigid-body state, none of which exist until
        _create_envs/init_data have run.

        Returns:
            (num_envs, 2 * (9 + n_dofs) + num_actions) float32 tensor in [-1, 1], laid out exactly
            as the rl_games player's output: base half from the solve, residual half zero.
        """
        if self.dexret_controller is None:
            assert "pinocchio" in sys.modules, (
                "pinocchio was never imported, so dex-retargeting is about to load it AFTER "
                "isaacgym, and every call into it will die with 'No Python class registered for "
                "C++ class std::vector<std::string>'. main/rl/train.py imports it first for "
                "exactly this reason — run the baseline through that entry point, or import "
                "pinocchio before isaacgym in whichever entry point you are using."
            )
            self.dexret_controller = DexRetargetController(
                self, robot=self.cfg["env"]["dexhand"], retargeting=self.dexret_type,
                wrist_mode=self.dexret_wrist_mode, wrist_fit=self.dexret_wrist_fit,
                calibrate=self.dexret_calibrate, fit_mode=self.dexret_fit_mode,
                calib_frames=self.dexret_calib_frames,
            )
        return self.dexret_controller.compute_action()

    def announce_residual_window(self, gate_weights):
        """Print each hand's residual window as it opens and closes, so live teleop is observable.

        Without this the window is silent and the operator cannot tell whether the residual is
        engaged. Live and env 0 only: it costs one device read per control step, negligible against
        a single-env teleop loop but a real per-step sync across a training batch. State is
        coarsened to reaching/fading/holding so a hand ramping through the fade announces once
        instead of on every step of the ramp.

        Args:
            gate_weights: (num_envs, 2) window weights, column 0 = RH, column 1 = LH.

        Returns:
            None.
        """
        if not self.live:
            return
        for side, weight in zip(("RH", "LH"), gate_weights[0].tolist()):  # one read, both hands
            if weight <= 0.0:
                state, colour = "OFF -- reaching", "1;91"      # red, imitator/retargeter alone
            elif weight >= 1.0:
                state, colour = "ON -- holding", "1;92"        # green, residual at full authority
            else:
                state, colour = "fading", "1;93"               # yellow, mid-crossfade
            attr = f"_residual_window_state_{side.lower()}"
            if state != getattr(self, attr, None):
                setattr(self, attr, state)
                print(f"\033[{colour}m[live] {side} residual {state} (w={weight:.2f})\033[0m")

    def dexret_solve_skippable(self, gate_weights):
        """Whether this control step can drop the dex-retargeting solve entirely.

        True only once every env has both hands fully across, where the crossfade weight is 1 and
        the solve is multiplied by zero, so its NLS buys nothing. A live wrist-fit calibration is
        the exception: it accumulates its samples inside the solve, so skipping would leave it
        short of its target forever and the run would never finish calibrating.

        Reading a device tensor into a Python bool costs a host sync, so the answer is cached and
        only recomputed while it is still False -- the sync is paid during the reach, which is
        exactly when the far more expensive solve is running anyway.

        Args:
            gate_weights: (num_envs, 2) window weights, or None when the window is disabled.

        Returns:
            bool True if the solve can be skipped this step.
        """
        if self.dexret_solve_complete:
            return True
        if gate_weights is None:
            return False  # no window means no handover, so the retargeter drives throughout
        if getattr(self.dexret_controller, "calibrating", False):
            return False
        # exact, not tolerant: the fade counter accumulates whole steps and clamps at fade_steps
        if bool(gate_weights.min() >= 1.0):
            self.dexret_solve_complete = True
        return self.dexret_solve_complete

    def switch_model_solve_skippable(self, gate_weights):
        """Whether switchModel can drop the dex-retargeting solve this control step.

        The mirror of dexret_solve_skippable. That one skips once every window is fully OPEN, where
        reachController=dexret multiplies the solve by zero; switchModel needs the solve exactly
        then, because an open window is when it is a candidate. So the skip holds only while every
        window is still fully shut, and once any has opened the solve is required for the rest of
        the episode -- latched, so the host sync is paid during the reach and never again.

        Args:
            gate_weights: (num_envs, 2) window weights, or None when the window is disabled.

        Returns:
            bool True if the solve can be skipped this step.
        """
        if self.switch_model_solve_required:
            return False
        if gate_weights is None:
            return True  # no window means nothing is ever arbitrated
        if bool(gate_weights.max() > 0.0):
            self.switch_model_solve_required = True
        return not self.switch_model_solve_required

    def residual_gate_weights(self):
        """Per-hand ramp position of the residual window, in [0, 1].

        Each hand runs imitator -> imitator+residual -> imitator across one manipulation: the window
        opens when that hand's human thumb or index fingertip reaches its object, and closes again
        once the fingertips retreat past the wider release distance. Two thresholds rather than one
        because a fingertip resting on a single boundary would flip the residual every step, and
        each flip is a step discontinuity in the commanded action. The ramp then eases the residual
        in and out instead of switching it, so neither edge of the window is a jump.

        tips_distance is the HUMAN fingertip -> object-surface distance (built in base.py for demo
        playback, overwritten every step by _inject_live when live). Its columns are
        (thumb, index, middle, ring, pinky) at both ends, so [:, :2] is the pinch pair.

        Returns:
            (num_envs, 2) float32 in [0, 1]; column 0 = RH, column 1 = LH. 0 = imitator alone drives
            that hand, 1 = its residual is applied in full.
        """
        env_ids = torch.arange(self.num_envs, device=self.device)
        fade_steps = max(float(self.residual_gate_fade_steps), 1.0)
        weights = []
        for side in ("rh", "lh"):
            demo = getattr(self, f"demo_data_{side}")
            # live caps the demo buffer at nT=4 and freezes progress_buf inside it, so clamp
            tip_idx = torch.clamp(self.progress_buf, 0, demo["tips_distance"].shape[1] - 1)
            pinch_dist = demo["tips_distance"][env_ids, tip_idx][:, :2].min(dim=-1).values
            gate = getattr(self, f"{side}_residual_gate_open")
            # release first, then engage: inside the band neither fires and the window holds its
            # state, which is what makes it hysteresis rather than two independent thresholds
            gate &= pinch_dist < self.residual_gate_release_distance
            gate |= pinch_dist <= self.residual_gate_distance
            fade = getattr(self, f"{side}_residual_gate_fade")
            fade += gate.float() * 2.0 - 1.0  # +1 while open, -1 while closed
            fade.clamp_(0.0, fade_steps)
            weights.append(fade / fade_steps)
        return torch.stack(weights, dim=-1)

    def expand_hand_weights(self, weights, rh_width, total_width):
        """Broadcast one per-hand weight per action column.

        Both halves of the action vector are laid out [RH root | RH dofs | LH root | LH dofs], so a
        single split at rh_width separates the hands within either half.

        Args:
            weights: (num_envs, 2) per-hand weights, column 0 = RH, column 1 = LH.
            rh_width: number of leading columns belonging to the right hand.
            total_width: width of the action half being expanded.

        Returns:
            (num_envs, total_width) float32 weights.
        """
        return torch.cat(
            [weights[:, 0:1].expand(-1, rh_width), weights[:, 1:2].expand(-1, total_width - rh_width)],
            dim=-1,
        )

    def build_switch_model_chains(self):
        """Build the per-hand FK chains switchModel scores candidates with.

        Lazy for the same reason dexret_baseline_actions builds its controller lazily: the Isaac dof
        ORDER is only knowable once the actors exist, and pytorch_kinematics orders its joints by the
        URDF rather than by Isaac, so a permutation has to be measured against a live actor. Built
        once and cached; pk is imported here rather than at module scope so a run with switchModel
        off never pays for it.

        Returns:
            None. Fills self.switch_model_chains with {side: (chain, isaac2chain, tip_names)}.
        """
        import pytorch_kinematics as pk

        env_ptr = self.envs[0]
        self.switch_model_chains = {}
        for side, dex in (("rh", self.dexhand_rh), ("lh", self.dexhand_lh)):
            chain = pk.build_chain_from_urdf(open(dex.urdf_path).read())
            # CPU, not self.device: the chain walk is Python-side, so on cuda the cost is
            # kernel-launch overhead rather than compute. Measured END TO END on this 12-DoF chain,
            # including the ~100 floats each way, it is 1.92 ms/hand against 2.07 on cuda -- a 7%
            # win, not the 4x the isolated FK timing suggests, because the round trip eats most of
            # it. Worth taking only because the dexret solve already pulls to the host every step to
            # feed pinocchio, so the pipeline is synced here either way and this adds no new stall.
            chain = chain.to(dtype=torch.float32, device="cpu")
            handle = self.gym.find_actor_handle(env_ptr, "dexhand_l" if side == "lh" else "dexhand_r")
            dof_names = self.gym.get_actor_dof_names(env_ptr, handle)
            isaac2chain = torch.tensor(
                [dof_names.index(j) for j in chain.get_joint_parameter_names()],
                device=self.device,
                dtype=torch.long,
            )
            tip_names = [dex.to_dex(f"{f}_tip")[0] for f in _TIP_LABELS]
            # Rows of the packed mano_joints holding the operator's five fingertips. Same packing
            # _setup_pinch_logging resolves (dexhand body order minus the wrist), recomputed here so
            # the arbiter does not depend on pinch logging having been armed -- that only happens
            # live or under logPinch, and switchModel is also meaningful in demo playback.
            order = [dex.to_hand(j)[0] for j in dex.body_names if dex.to_hand(j)[0] != "wrist"]
            missing = [f"{f}_tip" for f in _TIP_LABELS if f"{f}_tip" not in order]
            assert not missing, f"{side}: fingertips {missing} absent from packed mano order {order}"
            mano_rows = torch.tensor(
                [order.index(f"{f}_tip") for f in _TIP_LABELS], device=self.device, dtype=torch.long
            )
            self.switch_model_chains[side] = (chain, isaac2chain, tip_names, mano_rows)

    def predict_tip_positions(self, dof_targets, side):
        """Where a candidate's joint targets would put that hand's five fingertips.

        Forward kinematics of the COMMANDED targets, which is a genuine one-step prediction: the
        hand is PD-driven toward them, so FK of the target is where the fingers are heading. Left in
        the chain's root (hand-base) frame deliberately -- both candidates share the same wrist, so
        the wrist cancels out of any comparison between them and never has to be reconstructed from
        their root channels, whose meaning differs between the PID and non-PID branches.

        Args:
            dof_targets: (num_envs, n_dofs) joint targets in Isaac dof order, radians.
            side: "rh" or "lh".

        Returns:
            (num_envs, 5, 3) fingertip positions in the hand-base frame, metres, ordered by
            _TIP_LABELS.
        """
        chain, isaac2chain, tip_names, _ = self.switch_model_chains[side]
        ret = chain.forward_kinematics(dof_targets[:, isaac2chain].cpu())
        tips = torch.stack([ret[k].get_matrix()[:, :3, 3] for k in tip_names], dim=1)
        return tips.to(dof_targets.device)

    def switch_model_selection(self, base_action, residual_action, dexret_actions, root_control_dim, res_split_idx):
        """Pick, per hand, whether dex-retargeting or the residual drives this frame.

        Scores both candidates on how well they would follow the operator's fingers, then shifts the
        decision by how badly the OBJECT is currently being tracked. Only the finger term separates
        the candidates -- the object term cannot, because the object's response needs simulation and
        is only observable for whichever controller actually ran. It is charged to dex-retargeting:
        a drifting object means contact is the difficulty, which is the residual's regime. Without
        that shift the arbiter is degenerate, since dex-retargeting is the argmin of exactly the
        finger term. A dwell holds each choice for switchModelDwellSteps so a near-tie cannot
        chatter the commanded action.

        Args:
            base_action: (num_envs, res_split_idx) imitator base half of the policy action.
            residual_action: (num_envs, W) residual half, already scaled by 2.
            dexret_actions: (num_envs, action_dim) the retargeting solve, laid out like the policy's.
            root_control_dim: width of one hand's root block in the base half (9 under PID, else 6).
            res_split_idx: column where the base half ends and the residual half begins.

        Returns:
            (num_envs, 2) bool, True where that hand should run dex-retargeting. Column 0 = RH.
        """
        if self.switch_model_chains is None:
            self.build_switch_model_chains()
        n_rh = self.num_dexhand_rh_dofs
        # dexret_actions is the player's FULL [base | residual] vector; only its base half shares
        # base_action's layout, so truncate before slicing hands out of it. Slicing the full vector
        # open-ended runs the LH block off into the residual half.
        dex_base = dexret_actions[:, :res_split_idx]
        scores = []
        for side in ("rh", "lh"):
            # the two candidates' finger targets, in the same [-1, 1] units the base half carries
            if side == "rh":
                lo, hi = self.dexhand_rh_dof_lower_limits, self.dexhand_rh_dof_upper_limits
                base_dofs = base_action[:, root_control_dim : root_control_dim + n_rh]
                res_dofs = residual_action[:, 6 : 6 + n_rh]
                dex_dofs = dex_base[:, root_control_dim : root_control_dim + n_rh]
            else:
                lo, hi = self.dexhand_lh_dof_lower_limits, self.dexhand_lh_dof_upper_limits
                base_dofs = base_action[:, 2 * root_control_dim + n_rh :]
                res_dofs = residual_action[:, 6 + 6 + n_rh :]
                dex_dofs = dex_base[:, 2 * root_control_dim + n_rh :]
            # Cheap, and it catches the whole class of layout slip that a silent broadcast would
            # turn into a wrong score instead of an error.
            n_side = n_rh if side == "rh" else self.num_dexhand_lh_dofs
            assert base_dofs.shape[1] == res_dofs.shape[1] == dex_dofs.shape[1] == n_side, (
                f"switchModel {side} slice width mismatch: base={base_dofs.shape[1]} "
                f"res={res_dofs.shape[1]} dex={dex_dofs.shape[1]}, expected {n_side}. The action "
                f"layout is [RH root | RH dofs | LH root | LH dofs] per half, and dexret_actions "
                f"carries BOTH halves -- slice its base half before splitting hands out of it."
            )
            # Ranked on scaled pre-EMA targets: the moving average is w*curr + (1-w)*prev with the
            # same w and prev for both candidates, so it is monotone and cannot reorder them.
            q_res = torch_jit_utils.scale(torch.clamp(base_dofs + res_dofs, -1, 1), lo, hi)
            q_dex = torch_jit_utils.scale(torch.clamp(dex_dofs, -1, 1), lo, hi)
            # Operator fingertips into the same hand-base frame the FK returns. Read from the demo
            # buffer rather than _live_pinch_pts so this works offline too -- live, _inject_live has
            # already overwritten these slots with the AVP frame, so it is the same numbers either way.
            demo = getattr(self, f"demo_data_{side}")
            mano_rows = self.switch_model_chains[side][3]
            env_ids = torch.arange(self.num_envs, device=self.device)
            m_idx = torch.clamp(self.progress_buf, 0, demo["mano_joints"].shape[1] - 1)
            op_world = demo["mano_joints"][env_ids, m_idx].reshape(self.num_envs, -1, 3)[:, mano_rows]
            wrist = getattr(self, f"{side}_states")["base_state"]
            wrist_pos, wrist_quat = wrist[:, :3], wrist[:, 3:7]
            op_local = quat_apply(
                quat_conjugate(wrist_quat)[:, None].expand(-1, op_world.shape[1], -1).reshape(-1, 4),
                (op_world - wrist_pos[:, None]).reshape(-1, 3),
            ).reshape(self.num_envs, -1, 3)
            # Both candidates in ONE forward_kinematics call: the chain walk is Python-side per
            # call, so batching the two halves the per-step cost of the arbiter's only real work.
            tips = self.predict_tip_positions(torch.cat([q_res, q_dex], dim=0), side)
            err = torch.norm(tips - op_local.repeat(2, 1, 1), dim=-1).mean(dim=-1)
            f_res, f_dex = err[: self.num_envs], err[self.num_envs :]
            # observed object drift: the operator's Motive pose against the simulated body
            o_idx = torch.clamp(self.progress_buf, 0, demo["obj_trajectory"].shape[1] - 1)
            op_obj = demo["obj_trajectory"][env_ids, o_idx][:, :3, 3]
            sim_obj = getattr(self, f"_manip_obj_{side}_root_state")[:, :3]
            obj_err = torch.norm(op_obj - sim_obj, dim=-1)
            w_obj = self.switch_model_obj_weight
            s_res = (1.0 - w_obj) * (f_res / self.switch_model_finger_scale)
            s_dex = (1.0 - w_obj) * (f_dex / self.switch_model_finger_scale) + w_obj * (
                obj_err / self.switch_model_obj_scale
            )
            scores.append((s_res, s_dex, f_res, f_dex, obj_err))
        want = torch.stack([s_dex < s_res for s_res, s_dex, _, _, _ in scores], dim=-1)

        if self.switch_model_choice is None:
            self.switch_model_choice = torch.zeros((self.num_envs, 2), dtype=torch.bool, device=self.device)
            self.switch_model_held = torch.zeros((self.num_envs, 2), dtype=torch.long, device=self.device)
        # dwell: a flip is only allowed once the current choice has been held long enough
        self.switch_model_held += 1
        may_flip = self.switch_model_held >= max(self.switch_model_dwell_steps, 1)
        flipping = may_flip & (want != self.switch_model_choice)
        self.switch_model_choice = torch.where(flipping, want, self.switch_model_choice)
        self.switch_model_held = torch.where(
            flipping, torch.zeros_like(self.switch_model_held), self.switch_model_held
        )
        if self.switch_model_log:
            # Device buffer, transferred once at exit -- the same discipline _log_pinch_gap keeps and
            # for the same reason: reading these thirteen scalars to the host every step is thirteen
            # cuda syncs, which is what stalls the control loop. Env 0, which is the whole run live.
            row = torch.stack(
                [
                    torch.full_like(scores[0][0][0], float(self.control_steps)),
                    scores[0][2][0], scores[0][3][0], scores[0][4][0], scores[0][0][0], scores[0][1][0],
                    self.switch_model_choice[0, 0].to(scores[0][0].dtype),
                    scores[1][2][0], scores[1][3][0], scores[1][4][0], scores[1][0][0], scores[1][1][0],
                    self.switch_model_choice[0, 1].to(scores[0][0].dtype),
                ]
            )
            if self.switch_model_buf is None:
                self.switch_model_buf = torch.empty(8192, row.numel(), device=self.device)
                self.switch_model_n = 0
            if self.switch_model_n == len(self.switch_model_buf):
                self.switch_model_buf = torch.cat(
                    [self.switch_model_buf, torch.empty_like(self.switch_model_buf)]
                )
            self.switch_model_buf[self.switch_model_n] = row
            self.switch_model_n += 1
        return self.switch_model_choice

    def _refresh_action_mask(self):
        """Tick down and resample the random action mask (TeleDexter Sec. C.8).

        Envs whose freeze has expired become eligible for a fresh mask, drawn with probability
        action_mask_prob. A fresh mask freezes action_mask_n_dofs DoFs PER HAND (uniformly
        without replacement within that hand's DoF block) for d ~ U{1, d_max}. d_max ramps from 1
        to action_mask_max_duration over action_mask_ramp_steps control steps, following the
        paper's cubic (1 - sigma^3) expansion with sigma decaying linearly 1 -> 0.7, so the policy
        meets brief stalls before long ones. Masks refresh independently per env, so the batch
        spans a wide spectrum of partially-stale commands at any instant.
        """
        if self.action_mask_prob <= 0.0:
            return

        self._action_mask_steps = torch.clamp(self._action_mask_steps - 1, min=0)
        expired = self._action_mask_steps == 0
        self._action_mask[expired] = False  # no-op for envs that never held one

        eligible = expired & (torch.rand(self.num_envs, device=self.device) < self.action_mask_prob)
        n_new = int(eligible.sum())
        if n_new == 0:
            return
        new_ids = eligible.nonzero(as_tuple=False).flatten()

        sigma_min = 0.7
        progress = min(self.control_steps / max(self.action_mask_ramp_steps, 1), 1.0)
        sigma = 1.0 - (1.0 - sigma_min) * progress
        d_max = 1 + (self.action_mask_max_duration - 1) * (1 - sigma**3) / (1 - sigma_min**3)
        d_max = max(1, int(d_max))
        self._action_mask_steps[new_ids] = torch.randint(
            1, d_max + 1, (n_new,), device=self.device, dtype=torch.long
        )

        # per hand so the masked fraction matches the paper's single-hand density rather than
        # being halved across the concatenated BiH DoF vector
        n_rh = self.num_dexhand_rh_dofs
        for offset, n_hand in ((0, n_rh), (n_rh, self.num_dofs - n_rh)):
            k = min(self.action_mask_n_dofs, n_hand)
            # argsort of uniform noise gives an independent random permutation per env; take k
            picks = torch.rand(n_new, n_hand, device=self.device).argsort(dim=1)[:, :k]
            self._action_mask[new_ids.unsqueeze(1), picks + offset] = True

    def pre_physics_step(self, actions):

        # Swap in the retargeting baseline's action before anything reads `actions`. Everything
        # downstream is untouched, so the baseline is scored by the same physics, termination and
        # logging as a policy run — which is the whole point of driving it from in here rather than
        # precomputing a trajectory. The controller reads demo_data_{rh,lh}, so this follows the
        # live stream in live mode and the demo buffer otherwise, with no branch of its own.
        # The window weight decides whether the solve is still worth running, so it has to be known
        # BEFORE the solve rather than at the crossfade below. Computed once and reused: this
        # mutates the window state and the fade counter, so a second call would advance the fade at
        # twice the rate. Reading it here is exact -- it depends only on progress_buf and
        # tips_distance, neither of which pre_physics_step touches.
        gate_weights = self.residual_gate_weights() if self.residual_gate_distance >= 0 else None

        # dexRetBaseline replaces the action outright. reachController=dexret instead KEEPS the
        # policy's action and holds the solve beside it, so the two can be crossfaded per hand after
        # the split below: retargeting through the reach, imitator once that hand's window opens.
        dexret_actions = None
        if self.dexret_baseline:
            actions = self.dexret_baseline_actions()
        elif self.reach_controller == "dexret" and not self.dexret_solve_skippable(gate_weights):
            dexret_actions = self.dexret_baseline_actions()
        elif self.switch_model and not self.switch_model_solve_skippable(gate_weights):
            # switchModel needs the solve for the OPPOSITE span to reachController=dexret: it is a
            # candidate while a window is open, so the skip predicate is mirrored below.
            dexret_actions = self.dexret_baseline_actions()
        if dexret_actions is not None:
            assert dexret_actions.shape == actions.shape, (
                f"dex-retargeting returned {tuple(dexret_actions.shape)} but the policy emitted "
                f"{tuple(actions.shape)}. Crossfading needs the solve laid out like the player's "
                f"full [base | residual] vector. See DexRetargetController.compute_action()."
            )

        # ? >>> for visualization
        if not self.headless and self.debug_vis:

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
                    # all bones' thick-line copies in ONE add_lines call (per-bone calls add up)
                    assert hand_joints.shape[0] == self.dexhand_rh.n_bodies and hand_joints.shape[1] == 3
                    segments = np.concatenate(
                        [
                            self._thick_line_segments(np.stack([hand_joints[b[0]], hand_joints[b[1]]]))
                            for b in self.dexhand_rh.bone_links
                        ]
                    )  # green finger-pose skeleton, ~4x thick
                    segment_colors = np.repeat(color, len(segments), axis=0)
                    self.gym.add_lines(viewer, env_ptr, len(segments), segments.reshape(-1, 3), segment_colors)

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
            # remove the drawing of object frames optimizations
            # rh_state = self._manip_obj_rh_root_state[0].cpu().numpy()
            # lh_state = self._manip_obj_lh_root_state[0].cpu().numpy()
            # draw_frame(env_ptr0, rh_state[:3], rh_state[3:7])
            # draw_frame(env_ptr0, lh_state[:3], lh_state[3:7])

        # ? <<< for visualization

        # After the block above, whose clear_lines would otherwise wipe the glyphs.
        if self.live_rate_overlay and self.live and self.viewer is not None:
            self.draw_rate_overlay()

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

        # Post-reset residual warm-up (MANIPTRANS_RESIDUAL_WARMUP=N): zero the residual for the
        # first N steps after a reset so only the frozen imitator drives the hand into the grip —
        # the residual is trained on coherent in-distribution states, not freshly teleported ones.
        residual_warmup_remaining = getattr(self, "_residual_warmup_steps", 0)
        if residual_warmup_remaining > 0:
            residual_action = torch.zeros_like(residual_action)
            self._residual_warmup_steps = residual_warmup_remaining - 1

        # Live: once the cap is seated on the bottle, hand control back to the frozen imitators —
        # the residual is trained on the approach and fights the contact. Latches until reset.
        # Compares the SIMULATED root states (the two mesh origins). Seated measures
        # 3-5 mm on both axes; 10 mm is ~2x margin, and a missed latch fails silently.
        # The test is cap-on-bottle geometry, so object sets without that relation (cup_brush, where
        # the two props meeting means nothing) opt out via seating_cutoff regardless of the flag.
        if self.live and self.live_residual_cutoff and self.live_object_set.seating_cutoff:
            rh_obj_pos = self._manip_obj_rh_root_state[:, :3]
            lh_obj_pos = self._manip_obj_lh_root_state[:, :3]
            seated = ((rh_obj_pos[:, 2] - lh_obj_pos[:, 2]).abs() < 0.03) & (
                torch.norm(rh_obj_pos[:, :2] - lh_obj_pos[:, :2], dim=-1) < 0.010
            )
            # # Sim-distance readout for retuning the gates (env 0; live is single-env), ~6 Hz.
            # self._live_seated_print_ctr = getattr(self, "_live_seated_print_ctr", 0) + 1
            # if self._live_seated_print_ctr % 10 == 0:
            #     d_z = (rh_obj_pos[0, 2] - lh_obj_pos[0, 2]).abs().item() * 100
            #     d_xy = torch.norm(rh_obj_pos[0, :2] - lh_obj_pos[0, :2]).item() * 100
            #     d_3d = torch.norm(rh_obj_pos[0] - lh_obj_pos[0]).item() * 100
            #     print(
            #         f"\033[1;96m[live] cap-bottle sim dist  dz={d_z:6.2f} cm  dxy={d_xy:6.2f} cm  "
            #         f"d3d={d_3d:6.2f} cm  seated={bool(seated[0])}\033[0m"
            #     )
            # Arm on the first frame the cap is seen clear of the bottle, and only then allow the
            # cutoff to latch. This is what makes the reset keys work: reset_idx disarms, so the
            # residual stays ON after a reset until the cap is genuinely lifted and re-seated,
            # instead of re-latching off the pose it was reset into.
            self._live_seating_armed |= ~seated
            self._live_residual_latch |= seated & self._live_seating_armed
            residual_action = torch.where(
                self._live_residual_latch[:, None], torch.zeros_like(residual_action), residual_action
            )
            if bool(self._live_residual_latch[0]) != getattr(self, "_live_residual_cut", False):
                self._live_residual_cut = bool(self._live_residual_latch[0])
                state = "OFF -- cap seated on bottle (release)" if self._live_residual_cut else "ON"
                print(f"\033[1;91m[live] residual {state}\033[0m")  # bright red

        # Residual window (residualGateDistance >= 0): the imitator drives the hand for the whole
        # episode and only the residual is gated, so each hand runs imitator -> imitator+residual ->
        # imitator across one manipulation. The residual is spent on the grasp, and the reach and
        # the retreat are left to the imitator that was trained for them. The residual's RH block is
        # 6 wrist entries + its dofs even under PID control, where the BASE root widens to 9.
        # No host sync here: the old gate's per-hand transition prints cost two device syncs a step,
        # which is what got it marked deprecated -- the gate arithmetic itself was never the cost.
        if gate_weights is not None:
            self.announce_residual_window(gate_weights)
            if self.switch_model and dexret_actions is not None:
                # switchModel: arbitrate BEFORE the gate multiply below, so the residual candidate is
                # scored at the strength the policy actually emitted rather than part-way through a
                # fade. Restricted to hands whose window is open -- the reach belongs to the imitator
                # in this mode, and a hand that has not engaged yet has nothing to arbitrate.
                choice = self.switch_model_selection(
                    base_action, residual_action, dexret_actions, root_control_dim, res_split_idx
                )
                active = choice & (gate_weights > 0.0)
                base_w = self.expand_hand_weights(
                    active.float(), root_control_dim + self.num_dexhand_rh_dofs, res_split_idx
                )
                base_action = (1.0 - base_w) * base_action + base_w * dexret_actions[:, :res_split_idx]
                # A hand on the retargeter drops its residual outright: the residual was trained
                # against the IMITATOR's base and is incoherent added on top of a solve it never saw.
                residual_action = residual_action * self.expand_hand_weights(
                    (~active).float(), 6 + self.num_dexhand_rh_dofs, residual_action.shape[1]
                )
            residual_action = residual_action * self.expand_hand_weights(
                gate_weights, 6 + self.num_dexhand_rh_dofs, residual_action.shape[1]
            )
            if dexret_actions is not None and self.reach_controller == "dexret":
                # reachController=dexret: the same weight crossfades the BASE from the retargeting
                # solve to the imitator, so a hand rides the retargeter over exactly the span its
                # residual is off for, and the two hand over together instead of at separate
                # moments. The base half's RH root is root_control_dim wide (9 under PID) against
                # the residual half's 6, so it expands against its own width.
                # Guarded on reach_controller because switchModel also fills dexret_actions, and
                # this crossfade would then fire on top of the arbiter's own selection.
                base_w = self.expand_hand_weights(
                    gate_weights, root_control_dim + self.num_dexhand_rh_dofs, res_split_idx
                )
                base_action = base_w * base_action + (1.0 - base_w) * dexret_actions[:, :res_split_idx]

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

        # Decide this step's frozen DoFs once, before either hand's target is built, so both
        # hands are held against the same mask.
        self._refresh_action_mask()

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
        # Masked DoFs execute the previous command instead of this step's. prev_targets then
        # takes the held value, so a multi-step freeze holds one command for its whole duration.
        self.rh_curr_targets = torch.where(
            self._action_mask[:, : self.num_dexhand_rh_dofs],
            self.prev_targets[:, : self.num_dexhand_rh_dofs],
            self.rh_curr_targets,
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
        self.lh_curr_targets = torch.where(
            self._action_mask[:, self.num_dexhand_rh_dofs :],
            self.prev_targets[:, self.num_dexhand_rh_dofs :],
            self.lh_curr_targets,
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
            rh_force = 1.0 * (base_action[:, 0:3] * self.base_wrist_dt * self.translation_scale * 500) + (
                residual_action[:, 0:3] * self.dt * self.translation_scale * 500
            )
            rh_torque = 1.0 * (base_action[:, 3:6] * self.base_wrist_dt * self.orientation_scale * 200) + (
                residual_action[:, 3:6] * self.dt * self.orientation_scale * 200
            )
            lh_force = 1.0 * (
                base_action[
                    :, root_control_dim + self.num_dexhand_rh_dofs : root_control_dim + self.num_dexhand_rh_dofs + 3
                ]
                * self.base_wrist_dt
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
                * self.base_wrist_dt
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

            # Post-reset diagnostic (ported from 39a9100): for ~30 steps after a reset, print each
            # hand's ACTUAL wrist velocity, finger dof velocity, and object velocity, plus contact
            # forces and the applied wrist force/torque — separates "policy pushes the hand"
            # (|F|/|T| large first) from "contact solver ejects it" (handCF/objCF spike first,
            # velocity jumps with small |F|). env 0 only.
            # debug_steps_remaining = getattr(self, "_post_reset_debug_steps", 0)
            # if debug_steps_remaining > 0:
            #     def _norm(tensor):
            #         return float(torch.linalg.norm(tensor[0]).item())

            #     n_rh = self.num_dexhand_rh_dofs
            #     rh_obj = self._manip_obj_rh_root_state
            #     lh_obj = self._manip_obj_lh_root_state
            #     # net_cf holds the previous physics step's contact reactions; handCF = largest
            #     # contact force over that hand's bodies. Nonzero => touching; a spike => penetration.
            #     self.gym.refresh_net_contact_force_tensor(self.sim)
            #     if not hasattr(self, "_dbg_rh_body_idx"):
            #         self._dbg_rh_body_idx = list(self.dexhand_rh_handles.values())
            #         self._dbg_lh_body_idx = list(self.dexhand_lh_handles.values())
            #         self._dbg_rh_body_names = list(self.dexhand_rh_handles.keys())
            #     rh_cf_per_body = self.net_cf[0, self._dbg_rh_body_idx].norm(dim=-1)
            #     rh_hand_cf = float(rh_cf_per_body.max().item())
            #     rh_cf_body = self._dbg_rh_body_names[int(rh_cf_per_body.argmax().item())]
            #     lh_hand_cf = float(self.net_cf[0, self._dbg_lh_body_idx].norm(dim=-1).max().item())
            #     elapsed = self._post_reset_debug_window - debug_steps_remaining
            #     print(
            #         f"[reset-dbg t+{elapsed:02d}] "
            #         f"RH: wrist|v|={_norm(self._rh_base_state[:, 7:10]):.3f} "
            #         f"wrist|w|={_norm(self._rh_base_state[:, 10:13]):.3f} "
            #         f"fingers|dq|={_norm(self._qd[:, :n_rh]):.3f} "
            #         f"obj|v|={_norm(rh_obj[:, 7:10]):.3f} "
            #         f"handCF={rh_hand_cf:.2f}@{rh_cf_body} objCF={_norm(self._manip_obj_rh_cf):.2f} "
            #         f"|F|={_norm(rh_force):.3f} |T|={_norm(rh_torque):.3f}  ||  "
            #         f"LH: wrist|v|={_norm(self._lh_base_state[:, 7:10]):.3f} "
            #         f"wrist|w|={_norm(self._lh_base_state[:, 10:13]):.3f} "
            #         f"fingers|dq|={_norm(self._qd[:, n_rh:]):.3f} "
            #         f"obj|v|={_norm(lh_obj[:, 7:10]):.3f} "
            #         f"handCF={lh_hand_cf:.2f} objCF={_norm(self._manip_obj_lh_cf):.2f} "
            #         f"|F|={_norm(lh_force):.3f} |T|={_norm(lh_torque):.3f}"
            #     )
            #     self._post_reset_debug_steps = debug_steps_remaining - 1

        # Safety cap on the applied wrist wrench: an OOD live target saturates the residual's wrist
        # channels in one direction and, with no angular damping on the free-floating base, spins the
        # hand up. Clamp each wrist body's force/torque per axis before it reaches the sim (<=0 = off).
        # Live teleop only — training keeps the unclamped wrench so learned dynamics are unchanged.
        if self.live and (self.max_wrist_force > 0 or self.max_wrist_torque > 0):
            rh_wrist_handle = self.dexhand_rh_handles[self.dexhand_rh.to_dex("wrist")[0]]
            lh_wrist_handle = self.dexhand_lh_handles[self.dexhand_lh.to_dex("wrist")[0]]
            if self.max_wrist_force > 0:
                for wrist_handle in (rh_wrist_handle, lh_wrist_handle):
                    self.apply_forces[:, wrist_handle, :].clamp_(-self.max_wrist_force, self.max_wrist_force)
            if self.max_wrist_torque > 0:
                for wrist_handle in (rh_wrist_handle, lh_wrist_handle):
                    self.apply_torque[:, wrist_handle, :].clamp_(-self.max_wrist_torque, self.max_wrist_torque)

        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.apply_forces),
            gymtorch.unwrap_tensor(self.apply_torque),
            gymapi.ENV_SPACE,
        )

        # Live one-shot init: on the first pre_physics_step after a reset, teleport the fingers onto
        # the frozen imitator's output (base_action, scaled to joint limits) instead of waiting for
        # the PD to converge — the learned analog of opt_dof_pos. The flag is set in
        # _reset_default_side (live branch) and cleared here after a single application.
        snap_mask = getattr(self, "_snap_fingers_to_base_action", None)
        if snap_mask is not None and bool(snap_mask.any()):
            print("SNAPPING")
            snap_env_ids = snap_mask.nonzero(as_tuple=False).flatten()
            rh_base_fingers = torch.clamp(
                base_action[snap_env_ids, root_control_dim : root_control_dim + self.num_dexhand_rh_dofs], -1, 1
            )
            lh_base_fingers = torch.clamp(
                base_action[snap_env_ids, root_control_dim + root_control_dim + self.num_dexhand_rh_dofs :], -1, 1
            )
            rh_snap = torch_jit_utils.scale(
                rh_base_fingers, self.dexhand_rh_dof_lower_limits, self.dexhand_rh_dof_upper_limits
            )
            lh_snap = torch_jit_utils.scale(
                lh_base_fingers, self.dexhand_lh_dof_lower_limits, self.dexhand_lh_dof_upper_limits
            )
            self._q[snap_env_ids, : self.num_dexhand_rh_dofs] = rh_snap
            self._q[snap_env_ids, self.num_dexhand_rh_dofs :] = lh_snap
            self._qd[snap_env_ids] = 0.0
            self.prev_targets[snap_env_ids, : self.num_dexhand_rh_dofs] = rh_snap
            self.prev_targets[snap_env_ids, self.num_dexhand_rh_dofs :] = lh_snap
            snap_dexhand_ids = torch.concat(
                [
                    self._global_dexhand_rh_indices[snap_env_ids].flatten(),
                    self._global_dexhand_lh_indices[snap_env_ids].flatten(),
                ]
            )
            self.gym.set_dof_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self._dof_state),
                gymtorch.unwrap_tensor(snap_dexhand_ids),
                len(snap_dexhand_ids),
            )
            snap_mask[snap_env_ids] = False

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
            object_set=self.live_object_set,  # same entry _apply_live_object_set spawned from
            buffered=self.live_buffered,
            ema_alpha=self.causal_ema_alpha,
            causal_vel_mode=self.causal_mode,
        )
        self.live_source.start()
        # packed mano-joint order per side (dexhand body order minus wrist) — matches pack_data
        self._live_mano_order = {
            s: [dex.to_hand(j)[0] for j in dex.body_names if dex.to_hand(j)[0] != "wrist"]
            for s, dex in (("rh", self.dexhand_rh), ("lh", self.dexhand_lh))
        }
        # latest() returns mano joints as one [N,3] tensor in live_source.mano_names order; precompute
        # the gather index that reorders those rows into the packed body order above, so _inject_live
        # is a single indexed reshape instead of an N-way torch.cat.
        mano_row = {name: i for i, name in enumerate(self.live_source.mano_names)}
        self._live_mano_perm = {
            s: torch.tensor([mano_row[n] for n in order], device=self.device, dtype=torch.long)
            for s, order in self._live_mano_order.items()
        }
        # _log_pinch_gap state, all resolved here so its per-step path stays branch-free: the rows
        # of the live [N,3] mano tensor holding the five fingertips. The matching dexhand _tip
        # bodies and the gap buffer are shared with the demo-playback path.
        self._pinch_mano_rows = torch.tensor(
            [mano_row[f"{f}_tip"] for f in _TIP_LABELS], device=self.device, dtype=torch.long
        )
        self._setup_pinch_logging()
        print(f"[pinch] armed on viewer key N; will log fingertips (human vs sim) -> {self._pinch_csv}")

    def _setup_pinch_logging(self):
        """Buffers and body indices shared by both pinch-logging paths.

        Idempotent, because the two callers arm at different times: the live path from
        _ensure_live_source, the demo-playback path lazily on its first logged step. Everything
        here depends only on the dexhands and the config, never on the live stream — the one
        live-specific piece (_pinch_mano_rows) stays in _ensure_live_source.
        """
        if getattr(self, "_pinch_buf", None) is not None:
            return
        self._pinch_body_idx = {
            s: [getattr(self, f"dexhand_{s}_handles")[dex.to_dex(f"{f}_tip")[0]] for f in _TIP_LABELS]
            for s, dex in (("rh", self.dexhand_rh), ("lh", self.dexhand_lh))
        }
        # Rows of the demo's packed mano_joints holding the five fingertips. That packing is the
        # dexhand body order minus the wrist (see the mano_joints branch of the data packer), so
        # the tips are located by name within that same order.
        self._demo_pinch_rows = {}
        for s, dex in (("rh", self.dexhand_rh), ("lh", self.dexhand_lh)):
            order = [dex.to_hand(j)[0] for j in dex.body_names if dex.to_hand(j)[0] != "wrist"]
            missing = [f"{f}_tip" for f in _TIP_LABELS if f"{f}_tip" not in order]
            assert not missing, f"{s}: fingertips {missing} absent from packed mano order {order}"
            self._demo_pinch_rows[s] = torch.tensor(
                [order.index(f"{f}_tip") for f in _TIP_LABELS], device=self.device, dtype=torch.long
            )
        self._pinch_scales = torch.tensor(
            [self.obj_scale_rh, self.obj_scale_lh], device=self.device, dtype=torch.float32
        )
        # gap buffer (doubles on demand; 8192 rows ~= 2.3 min at 60 Hz)
        self._pinch_buf = torch.empty(8192, len(self._PINCH_COLS), device=self.device)
        self._pinch_n = 0
        self._live_pinch_pts = {}  # per side, this frame's human thumb/index tips
        # train.py resolves the default path into runs/<exp>/pinch_logs/ next to the checkpoint
        # being played, the same convention as the grip logs.
        self._pinch_csv = os.environ.get("MANIPTRANS_PINCH_CSV", "") or "pinch_gap.csv"
        # Live recording is armed by a manual reset (viewer key N), so the log starts at a known
        # trajectory start instead of capturing the approach/settling before the run proper.
        # Each further reset discards what was buffered and starts over — see _do_manual_reset.
        # Demo playback has a definite start of its own and arms itself immediately.
        self._pinch_armed = False
        # viewer frames for the demo video land beside the CSV and are encoded at exit
        self._pinch_frames_dir = os.path.splitext(self._pinch_csv)[0] + "_frames"
        # Which live frame each control step consumed, for tying the recorded video to externally
        # filmed footage (see dump_live_provenance). Host tuples, not a device buffer like
        # _pinch_buf: every column is already a scalar, so appending forces no device sync.
        # Only filled when recordDemoData is on, and cleared by each manual reset with the PNGs.
        self._live_provenance = []
        atexit.register(self._dump_pinch_gap)

    def _fill_demo_pinch_pts(self):
        """Demo-playback counterpart to the live tip injection.

        Takes env 0's five fingertips from the demo's MANO joints at the current step. The demo
        buffer is already in the gym frame, the same one the sim tip bodies are read in, so the
        two are directly comparable — but note the reference is the demo's MANO hand, NOT a live
        AVP capture, which is what the `avp` columns mean for a run logged this way.
        """
        for side in ("rh", "lh"):
            joints = getattr(self, f"demo_data_{side}")["mano_joints"]
            idx = int(self.progress_buf[0].clamp(max=joints.shape[1] - 1))
            self._live_pinch_pts[side] = joints[0, idx].reshape(-1, 3)[self._demo_pinch_rows[side]]

    def _inject_live(self):
        """Overwrite every demo target slot with the latest live frame, broadcast across envs."""
        self._ensure_live_source()
        f = self.live_source.latest()
        # report only skipped frames (an every-step print costs ms of console I/O at 60 Hz)
        prev_seq = getattr(self, "_live_prev_seq", None)
        # if prev_seq is not None and f["seq"] - prev_seq > 1:
        #     print(f"[live] skipped {f['seq'] - prev_seq - 1} frame(s): seq {prev_seq} -> {f['seq']}")
        self._live_prev_seq = f["seq"]
        # Which live frame this control step actually consumed. The pairing has to be recorded
        # rather than inferred from a nominal rate: CONFLATE drops frames when the sim consumes
        # slower than the publisher (the seq_gap branch in LiveTargetSource.latest), and when the
        # sim outruns the publisher the same seq is consumed twice (the _out_cache branch). Joined
        # against the PNG names -- both keyed by control_steps -- this dates every recorded frame.
        # t_step_s is the DESKTOP's clock, so t_step_s - t_capture_s is the end-to-end teleop lag --
        # but only if the two machines are NTP-synced, since it spans them (the same caveat
        # live_streaming/debug/sub_print.py raises about its age_ms). The default alignment uses
        # t_capture_s alone and is immune to that skew; only --show-latency reads this column.
        if self.record_demo_data:
            self._live_provenance.append(
                (self.control_steps, f["seq"], f["t_capture_s"], time(), bool(f["stale"]), bool(f["sync_ok"]))
            )
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
            perm = self._live_mano_perm[side]  # rows of the [N,3] tensor in packed body order
            demo["mano_joints"][:, :] = t["mano_joints"][perm].reshape(-1)
            demo["mano_joints_velocity"][:, :] = t["mano_joints_velocity"][perm].reshape(-1)
            # keep this frame's human thumb/index tips for _log_pinch_gap (device tensor, no sync)
            self._live_pinch_pts[side] = t["mano_joints"][self._pinch_mano_rows]
        if self._has_prop and "prop" in f:
            # Refresh the prop's tracked pose, but do NOT write it to the sim here: only reset_idx
            # reads this. Teleporting a body every step would push force through its contact with
            # the manipulated object instead of letting the two interact physically.
            self.demo_data_rh["prop_trajectory"][:, :] = f["prop"]["obj_trajectory"]

    def _do_manual_reset(self, label):
        """Re-init all envs (and restart the live replay), triggered by a viewer key."""
        # Live: restart the replay FIRST, then wait for the freshly-published frame 0 before
        # resetting, so the object teleports to the restarted start pose instead of the stale held
        # frame. mock_publish pauses ~1 s on reset, so allow a bit more than that for the frame.
        if self.live and self.live_source is not None:
            self.live_source.request_publisher_reset()
            self.live_source.flush_and_wait_fresh(timeout_s=1.5)  # > mock_publish's 1 s reset pause
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        # This fires inside post_physics_step AFTER this step's compute_observations, and live mode
        # never runs reset_done — without a recompute, the next action (and the one-shot finger
        # snap consuming its base_action) would be computed from the PRE-reset world and kick the
        # freshly teleported hands. Mirrors reset_done's reset_idx -> compute_observations order.
        self.compute_observations()
        # M/N must hand control back to the residual, unconditionally. reset_idx already clears the
        # seating latch, but re-assert it here so the guarantee lives at the key press rather than
        # depending on reset_idx's internals -- and clear the post-reset warm-up too, which
        # reset_idx re-reads from MANIPTRANS_RESIDUAL_WARMUP and which would otherwise hold the
        # residual at zero for its first N steps, making the key look like it had not worked.
        # Announced in green to pair with the red "residual OFF" the seating cutoff prints.
        if self.live:
            self._live_residual_latch[:] = False
            self._live_seating_armed[:] = False
            self._live_residual_cut = False
            self._residual_warmup_steps = 0
            print(f"\033[1;92m[live] residual ON -- reset by {label}\033[0m")
        # Every manual reset restarts the pinch log: whatever was buffered is discarded and
        # recording begins again from this trajectory start, so the CSV written at exit always
        # holds exactly the last attempt rather than every attempt concatenated.
        if hasattr(self, "_pinch_armed"):
            dropped = self._pinch_n
            self._pinch_n = 0
            self._pinch_armed = True
            frames = self._reset_pinch_frames()
            print(
                f"[pinch] recording restarted ({label})"
                f"{f', discarded {dropped} row(s)' if dropped else ''}"
                f"{f' and {frames} frame(s)' if frames else ''} -> {self._pinch_csv}"
            )

    # ── grip diagnostics ──────────────────────────────────────────────────────────────
    # Short labels for the printed/CSV columns, in dexhand.contact_body_names order.
    _GRIP_TIP_LABELS = _TIP_LABELS

    def _grip_contact_tensors(self, side):
        """Per-fingertip contact state for env 0, as tensors (no host sync): net contact force
        [5,3], the contact bodies' positions [5,3], and the object's world COM [3].

        Note the bodies are dexhand.contact_body_names (thumb_distal, *_intermediate), NOT the _tip links:
        the tips carry no collision geometry, so PhysX reports exactly 0 N on them.
        """
        handles = getattr(self, f"dexhand_{side}_handles")
        obj_state = getattr(self, f"_manip_obj_{side}_root_state")
        idx = [handles[k] for k in getattr(self, f"dexhand_{side}").contact_body_names]
        obj_com = obj_state[0, :3] + torch_jit_utils.quat_rotate(
            obj_state[0:1, 3:7], getattr(self, f"manip_obj_{side}_com")[0:1]
        )[0]
        return self.net_cf[0, idx], self._rigid_body_state[0, idx, :3], obj_com

    def _grip_side_metrics(self, side):
        """Steady-state grip measures for env 0 of one hand. Returns a flat {name: float}.

        Three independent measures, because none alone is sufficient:
          * per-contact-body contact force (net_cf). net_cf is a NET sum per body, so opposing
            contacts cancel — read it per fingertip and never off the object, whose reading is
            table reaction plus every finger and is ~0 for a balanced grasp.
          * actual joint torque (dof_force) plus how close it sits to the URDF effort limit;
            at saturation more over-closure buys no extra force.
          * the PD's commanded torque, stiffness * (target - q) — the P term the policy actually
            controls. The damping term is dropped: it vanishes at the steady state this measures.
        `inward` projects each fingertip force onto the direction from that fingertip to the
        object COM, so only the cap-directed (squeezing) component is counted.
        """
        dexhand = getattr(self, f"dexhand_{side}")
        obj_state = getattr(self, f"_manip_obj_{side}_root_state")
        forces, tip_pos, obj_com = self._grip_contact_tensors(side)
        inward_dir = torch.nn.functional.normalize(obj_com.unsqueeze(0) - tip_pos, dim=-1)
        magnitude = forces.norm(dim=-1)
        inward = (forces * inward_dir).sum(dim=-1)

        dof_slice = (
            slice(0, self.num_dexhand_rh_dofs) if side == "rh" else slice(self.num_dexhand_rh_dofs, None)
        )
        effort = getattr(self, f"_dexhand_{side}_effort_limits")
        tau_actual = self.dof_force[0, dof_slice].abs()
        margin = self._pos_control[0, dof_slice] - self._q[0, dof_slice]
        tau_commanded = (getattr(self, f"dexhand_{side}_dof_stiffness") * margin).abs()

        metrics = {}
        for label, mag, inw in zip(self._GRIP_TIP_LABELS, magnitude, inward):
            metrics[f"{side}_{label}_f"] = float(mag)
            metrics[f"{side}_{label}_in"] = float(inw)
        metrics[f"{side}_tip_f_sum"] = float(magnitude.sum())
        metrics[f"{side}_tip_in_sum"] = float(inward.sum())
        metrics[f"{side}_tau_max"] = float(tau_actual.max())
        metrics[f"{side}_tau_sum"] = float(tau_actual.sum())
        metrics[f"{side}_tau_cmd_max"] = float(tau_commanded.max())
        metrics[f"{side}_sat_dofs"] = float((tau_actual >= 0.99 * effort).sum())
        metrics[f"{side}_margin_deg_max"] = float(torch.rad2deg(margin.abs().max()))
        metrics[f"{side}_obj_z"] = float(obj_state[0, 2])
        return metrics

    def _log_grip_state(self):
        """MANIPTRANS_GRIP_LOG=N prints env 0's grip state every N steps; MANIPTRANS_GRIP_CSV=<path>
        writes one row per step. Call after compute_observations so net_cf/dof_force are refreshed.

        train.py resolves a bare filename (or "auto") in MANIPTRANS_GRIP_CSV to
        runs/<exp>/grip_logs/ next to the checkpoint being played, so the rows land with the model
        that produced them; a value containing a directory is used verbatim."""
        if not hasattr(self, "_grip_log_every"):
            self._grip_log_every = int(os.environ.get("MANIPTRANS_GRIP_LOG", "0"))
            self._grip_log_csv = os.environ.get("MANIPTRANS_GRIP_CSV", "")
            self._grip_log_file = None
            self._grip_log_step = 0
            if self._grip_log_every > 0 or self._grip_log_csv:
                for side in ("rh", "lh"):
                    mass = float(getattr(self, f"manip_obj_{side}_mass")[0])
                    effort = getattr(self, f"_dexhand_{side}_effort_limits")
                    print(
                        f"[grip] {side.upper()} object mass={mass * 1e3:.1f} g "
                        f"(weight {mass * 9.81:.3f} N)  objScale x{getattr(self, f'obj_scale_{side}'):.3f}"
                        f"  dof effort limit="
                        f"{float(effort.min()):.3f}..{float(effort.max()):.3f} Nm"
                    )
        if self._grip_log_every <= 0 and not self._grip_log_csv:
            return

        metrics = {"step": float(self._grip_log_step)}
        metrics.update(self._grip_side_metrics("rh"))
        metrics.update(self._grip_side_metrics("lh"))

        if self._grip_log_csv:
            if self._grip_log_file is None:
                self._grip_log_file = open(self._grip_log_csv, "w")
                self._grip_log_file.write(",".join(metrics.keys()) + "\n")
            self._grip_log_file.write(",".join(f"{v:.6g}" for v in metrics.values()) + "\n")
            self._grip_log_file.flush()

        if self._grip_log_every > 0 and self._grip_log_step % self._grip_log_every == 0:
            for side in ("rh", "lh"):
                tips = "  ".join(
                    f"{label}={metrics[f'{side}_{label}_f']:5.2f}" for label in self._GRIP_TIP_LABELS
                )
                print(
                    f"[grip t={self._grip_log_step:5d}] {side.upper()} tipF(N) {tips} "
                    f"| sum={metrics[f'{side}_tip_f_sum']:6.2f} inward={metrics[f'{side}_tip_in_sum']:6.2f} "
                    f"| tau(Nm) act_max={metrics[f'{side}_tau_max']:.3f} cmd_max={metrics[f'{side}_tau_cmd_max']:.3f} "
                    f"sat={int(metrics[f'{side}_sat_dofs'])}/{len(getattr(self, f'_dexhand_{side}_effort_limits'))} "
                    f"| margin={metrics[f'{side}_margin_deg_max']:5.1f}deg objZ={metrics[f'{side}_obj_z']:.3f}"
                )
        self._grip_log_step += 1

    # ── live pinch-gap diagnostics ────────────────────────────────────────────────────
    # One [2, 3] (thumb, index) x (x, y, z) block per entry, in the order _log_pinch_gap stacks
    # them. `avp` = the human hand in the live frame, `sim` = the dexhand fingertip marker (both
    # positions, m); `force` = net contact force on that finger's contact body (N); `cf` = that
    # contact body's own position (m) — the force acts there, not at the _tip marker, so the
    # squeeze direction has to be measured from it. Trailed by each object's COM (m), which is
    # what "toward the target" points at.
    _PINCH_BLOCKS = ("rh_avp", "rh_sim", "lh_avp", "lh_sim", "rh_force", "lh_force", "rh_cf", "lh_cf")
    # ...the per-object scale multipliers trail the vectors: constant for the run, but recorded so
    # the plot script can size the cap reference correctly without being told the run's config.
    _PINCH_COLS = (
        [f"{b}_{f}_{a}" for b in _PINCH_BLOCKS for f in _TIP_LABELS for a in "xyz"]
        + [f"{s}_obj_com_{a}" for s in ("rh", "lh") for a in "xyz"]
        + ["rh_obj_scale", "lh_obj_scale"]
    )

    def _log_pinch_gap(self):
        """Env 0's thumb/index fingertip POSITIONS and contact FORCES, per control step.

        Two callers fill the human side: live mode (the AVP stream, via _inject_live) and demo
        playback (the demo's MANO joints, via _fill_demo_pinch_pts). The column names say `avp`
        either way, so which one produced a CSV is a property of the run, not of the file.

        `avp` columns are the Apple Vision Pro fingertips in the current live frame; `sim` columns
        are the dexhand's own _tip bodies — all five fingers on both, so the per-finger tracking
        error (sim - avp) is directly available. LiveTargetSource has already mapped
        the AVP joints into the gym frame, so all four points share one coordinate system and any
        separation derived from them is directly comparable (metres). `force` and `cf` come from
        _grip_contact_tensors — the same source _grip_side_metrics reads — so they cover ALL FIVE
        contact bodies, not just the pinching pair: net contact force (N) and the contact bodies'
        own positions (m). `obj_com` is the object's world centre of mass. Together they let the
        plot script rebuild the inward/squeeze projection (force . unit(obj_com - origin)) with
        either origin convention, and the 5-tip sum that _grip_side_metrics reports.

        Raw vectors rather than precomputed distances/magnitudes: it keeps the decision of what to
        measure (in-plane vs vertical separation, per-axis drift, force direction) in the plotting
        script, where it can be changed without re-running the session. See
        data_stats/plot_pinch_gap.py, which derives the XY and Z thumb-index gaps.

        Rows accumulate in a preallocated device buffer and cross to the host once, at exit:
        nothing here touches the host, so the whole logger is a handful of small kernels per step
        and never stalls the pipeline. State is set up in _ensure_live_source, and the caller only
        reaches this in live mode, so there is no per-step branch to pay for either.
        """
        rh_force, rh_cf, rh_com = self._grip_contact_tensors("rh")
        lh_force, lh_cf, lh_com = self._grip_contact_tensors("lh")
        blocks = torch.stack(
            [
                self._live_pinch_pts["rh"],
                self._rigid_body_state[0, self._pinch_body_idx["rh"], :3],
                self._live_pinch_pts["lh"],
                self._rigid_body_state[0, self._pinch_body_idx["lh"], :3],
                rh_force,
                lh_force,
                rh_cf,
                lh_cf,
            ]
        )  # [8, 5, 3] = _PINCH_BLOCKS x _TIP_LABELS x xyz
        if self._pinch_n == len(self._pinch_buf):
            self._pinch_buf = torch.cat([self._pinch_buf, torch.empty_like(self._pinch_buf)])
        self._pinch_buf[self._pinch_n] = torch.cat(
            [blocks.view(-1), rh_com, lh_com, self._pinch_scales]
        )  # _PINCH_COLS order
        self._pinch_n += 1

    def _reset_pinch_frames(self):
        """Restart this attempt's provenance rows, and the viewer capture when recordDemoVideo is on.

        Returns how many captured frames were discarded — always 0 with video off, since nothing
        was being captured. No-op headless, where there is no viewer and no manual reset key."""
        if self.viewer is None or not self.record_demo_data:
            return 0
        # Drop the rows from the previous attempt so the CSV written at exit covers exactly this
        # one, mirroring _do_manual_reset's _pinch_n = 0.
        self._live_provenance.clear()
        # The control step that becomes CSV row 0, so the video stamp can be the same index the
        # plots use on their x-axis. This runs at the END of post_physics_step — after this step's
        # row was logged and then discarded — and control_steps increments after post_physics_step,
        # so the next step to be logged is control_steps + 1.
        self._pinch_step0 = self.control_steps + 1
        # Frame capture is opt-in and separate: arming it costs half the control rate (each PNG
        # readback overruns vec_task.render's pacing budget), which would corrupt the frame
        # statistics the CSV exists to measure. See recordDemoVideo in config.yaml.
        if not self.record_demo_video:
            return 0
        dropped = len(glob.glob(os.path.join(self._pinch_frames_dir, "frame_*.png")))
        shutil.rmtree(self._pinch_frames_dir, ignore_errors=True)
        os.makedirs(self._pinch_frames_dir, exist_ok=True)
        self.record_frames_dir = self._pinch_frames_dir  # consumed by vec_task.render
        self.record_frames = True
        return dropped

    def _encode_pinch_video(self):
        """atexit: fold the captured viewer frames into an mp4 beside the CSV and drop the PNGs.

        Frames are written once per drawn viewer frame, so the real-time rate is the control rate
        divided by renderDecimation — encode at that fps and the video plays back at 1x."""
        frames = sorted(glob.glob(os.path.join(getattr(self, "_pinch_frames_dir", ""), "frame_*.png")))
        if not frames:
            return
        out = os.path.splitext(self._pinch_csv)[0] + ".mp4"
        fps = 1.0 / (self.dt * self.control_freq_inv * self.render_decimation)
        try:
            import imageio

            from lib.utils.wandb_utils import WandbVideoCaptureWrapper

            # Burn the step number top-left with the same helper the wandb video wrapper uses, so
            # both recordings are labelled identically. The label is the control step relative to
            # the reset — i.e. exactly the CSV row and the plots' x-axis, and sim time / dt. It is
            # read back from the filename (vec_task names each frame by control_steps) rather than
            # from the frame's ordinal: the render decimation counter is not reset by the reset
            # key, so the first captured frame can land up to renderDecimation-1 steps late and a
            # fixed ordinal * renderDecimation would be offset for the whole recording.
            step0 = getattr(self, "_pinch_step0", 0)
            with imageio.get_writer(out, fps=fps, macro_block_size=None) as writer:
                for path in frames:
                    step = int(os.path.basename(path)[len("frame_") : -len(".png")]) - step0
                    frame = torch.from_numpy(imageio.imread(path))
                    writer.append_data(WandbVideoCaptureWrapper._burn_frame_number(frame, step))
            shutil.rmtree(self._pinch_frames_dir, ignore_errors=True)
            print(f"[pinch] wrote {len(frames)} frames at {fps:.1f} fps to {out}")
        except Exception as exc:  # keep the PNGs so nothing is lost if encoding fails
            print(f"[pinch] video encode failed ({exc}); {len(frames)} frames left in {self._pinch_frames_dir}")

    def dump_live_provenance(self):
        """Write the control-step -> live-frame table that dates every frame of the recorded video.

        Each row pairs one control step with the live frame it consumed, so an externally filmed
        clip can be aligned against the sim: `t_capture_s` is the laptop's epoch clock, the only
        absolute time shared with the hand that was filmed. `video_frame` is the frame's ordinal in
        the encoded mp4, or -1 for a step the viewer never drew (renderDecimation draws one step in
        N), which is what lets the compositor map an mp4 frame straight to a wall-clock instant.
        `t_step_s` is the desktop's own clock at that step; it is only meaningful against
        `t_capture_s` if the two machines are NTP-synced, and only --show-latency uses it.

        Must run BEFORE _encode_pinch_video, which deletes the PNGs this reads the captured step
        numbers from. Steps are parsed from the filenames rather than counted, for the reason given
        in _encode_pinch_video: the decimation counter survives the reset key, so the first capture
        can land up to renderDecimation-1 steps late.

        Returns:
            int number of rows written; 0 if the run recorded nothing.
        """
        rows = getattr(self, "_live_provenance", [])
        if not rows:
            return 0
        captured = sorted(
            int(os.path.basename(p)[len("frame_") : -len(".png")])
            for p in glob.glob(os.path.join(getattr(self, "_pinch_frames_dir", ""), "frame_*.png"))
        )
        video_frame = {step: i for i, step in enumerate(captured)}
        out = os.path.splitext(self._pinch_csv)[0] + "_live_frames.csv"
        with open(out, "w") as fh:
            fh.write("video_frame,control_step,seq,t_capture_s,t_step_s,stale,sync_ok\n")
            for control_step, seq, t_capture_s, t_step_s, stale, sync_ok in rows:
                # %.6f = microseconds; these are ~1.7e9 epochs, so this sits at the edge of
                # float64's decimal precision and well under one 60 Hz frame either way.
                fh.write(
                    f"{video_frame.get(control_step, -1)},{control_step},{seq},"
                    f"{t_capture_s:.6f},{t_step_s:.6f},{int(stale)},{int(sync_ok)}\n"
                )
        self._live_provenance = []  # atexit can fire twice if the env is also closed explicitly
        print(f"[live] wrote {len(rows)} provenance rows ({len(captured)} on video) to {out}")
        return len(rows)

    def dump_switch_model_log(self):
        """Write switchModel's per-step scores and selection beside the pinch CSV.

        Both RAW terms are written alongside the weighted scores, because the raw ranges are what
        switchModelObjScale / switchModelFingerScale have to be calibrated from -- the 0.7/0.3 split
        means nothing until each term reads ~1.0 at "clearly bad". `chose_dexret` is the decision
        after the dwell, so comparing it against the scores also shows how often the dwell suppressed
        a flip. Env 0 only, which is the whole run live.

        Returns:
            int number of rows written; 0 if the arbiter never ran.
        """
        if not getattr(self, "switch_model_n", 0):
            return 0
        rows = self.switch_model_buf[: self.switch_model_n].cpu().numpy()  # one transfer
        self.switch_model_n = 0  # atexit can fire twice if the env is also closed explicitly
        out = os.path.splitext(self._pinch_csv)[0] + "_switch_model.csv"
        np.savetxt(
            out,
            rows,
            delimiter=",",
            header=(
                "control_step,"
                "rh_finger_res,rh_finger_dex,rh_obj_err,rh_score_res,rh_score_dex,rh_chose_dexret,"
                "lh_finger_res,lh_finger_dex,lh_obj_err,lh_score_res,lh_score_dex,lh_chose_dexret"
            ),
            comments="",
            fmt="%.6f",
        )
        rh_frac = rows[:, 6].mean()
        lh_frac = rows[:, 12].mean()
        print(
            f"[switchModel] wrote {len(rows)} rows to {out} "
            f"(dex-retargeting chosen RH {100 * rh_frac:.1f}%, LH {100 * lh_frac:.1f}%)"
        )
        return len(rows)

    def _dump_pinch_gap(self):
        """atexit: write the buffer as a CSV, encode the demo video, and render the companion
        plots. Every viewer exit path in vec_task.render goes through sys.exit(), which runs
        atexit handlers."""
        self.dump_live_provenance()  # before _encode_pinch_video, which deletes the PNGs it reads
        self.dump_switch_model_log()
        self._encode_pinch_video()
        if not getattr(self, "_pinch_n", 0):
            return
        rows = self._pinch_buf[: self._pinch_n].cpu().numpy()  # one transfer
        self._pinch_n = 0  # atexit can fire twice if the env is also closed explicitly
        np.savetxt(
            self._pinch_csv,
            rows,
            delimiter=",",
            header=",".join(self._PINCH_COLS),
            comments="",
            fmt="%.6f",
        )
        print(f"[pinch] wrote {len(rows)} rows to {self._pinch_csv}")

        script = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")),
            "data_stats",
            "plot_pinch_gap.py",
        )
        if os.environ.get("MANIPTRANS_PINCH_PLOT", "1") == "0":
            print(f"[pinch] to plot: python {script} {self._pinch_csv}")
            return
        # plot in a child process: this runs during interpreter shutdown with the sim being torn
        # down around us, and pulling matplotlib into that is the fragile half of the job.
        try:
            subprocess.run([sys.executable, script, self._pinch_csv], timeout=120, check=True)
        except Exception as exc:
            print(f"[pinch] auto-plot failed ({exc}); run: python {script} {self._pinch_csv}")

    def post_physics_step(self):
        # per-phase timing published to vec_task's [timing] print (MANIPTRANS_STEP_TIMING=N)
        timing = getattr(self, "_step_timing_every", 0) > 0
        if timing:
            inject_start_time = self._timing_checkpoint()
        if self.live:
            self._inject_live()
        if timing:
            inject_end_time = self._timing_checkpoint()

        self.compute_observations()
        if timing:
            observations_end_time = self._timing_checkpoint()
        if self.live:
            # Imitation reward/termination is meaningless against a live stream and costs ~5 ms
            # per step; resets are forced off below anyway. Keep step()'s info contract alive.
            if not hasattr(self, "reward_dict"):
                self.reward_dict = {}
        else:
            self.compute_reward(self.actions)
        # self._draw_obj_axes()
        self._log_grip_state()
        # _inject_live above has already filled this step's human tips; _pinch_armed gates recording
        # until the first manual reset (viewer key N)
        if self.live and self.log_pinch and self._pinch_armed:
            self._log_pinch_gap()
        elif self._pinch_demo_logging:
            # Demo playback: no live stream to inject the human tips, so read them off the demo.
            # Armed from the first step — playback already starts at the trajectory start.
            self._setup_pinch_logging()
            self._fill_demo_pinch_pts()
            self._log_pinch_gap()
        if timing:
            reward_end_time = self._timing_checkpoint()
            self._post_phase_ms = {
                "post.inject_live": (inject_end_time - inject_start_time) * 1e3,
                "post.observations": (observations_end_time - inject_end_time) * 1e3,
                "post.reward": (reward_end_time - observations_end_time) * 1e3,
            }

        if self.live:
            self.reset_buf[:] = 0  # live teleop runs continuously; never auto-reset

        # Manual reset on viewer key 'N' (set in vec_task.render). Re-inits all envs — useful in
        # live mode (no auto-reset) to re-attempt a replay when the first playthroughs are broken.
        if getattr(self, "_reset_env_request", False):
            self._reset_env_request = False
            self._do_manual_reset("key N")

        # Delayed reset on viewer key 'M': scheduled 2 s ahead in vec_task.render. The sim keeps
        # running while a big ASCII countdown (2..1) prints once per second, then all envs re-init
        # like the N reset once the wall-clock deadline passes.
        reset_env_request_at = getattr(self, "_reset_env_request_at", None)
        if reset_env_request_at is not None:
            seconds_left = reset_env_request_at - time()
            if seconds_left <= 0:
                self._reset_env_request_at = None
                self._countdown_last_shown = None
                self._do_manual_reset("key M")
            else:
                current_count = math.ceil(seconds_left)
                if current_count != getattr(self, "_countdown_last_shown", None):
                    self._countdown_last_shown = current_count
                    print(f"\n{render_big_number(current_count)}\n")

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
            camera_cfg.width = RECORD_WIDTH
            camera_cfg.height = RECORD_HEIGHT
            camera_cfg.horizontal_fov = RECORD_FOV

            camera = isaac_gym.create_camera_sensor(env, camera_cfg)
            cam_pos = gymapi.Vec3(*FRONT_EYE)
            cam_target = gymapi.Vec3(*FRONT_TARGET)
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

    def _record_camera(self, *, env, isaac_gym, eye, target):
        """One off-screen recording camera at a fixed pose. Shared by all three views so they
        cannot drift apart in resolution or FOV, which is what makes them comparable."""
        camera_cfg = gymapi.CameraProperties()
        camera_cfg.enable_tensors = True
        camera_cfg.width = RECORD_WIDTH
        camera_cfg.height = RECORD_HEIGHT
        camera_cfg.horizontal_fov = RECORD_FOV
        camera = isaac_gym.create_camera_sensor(env, camera_cfg)
        isaac_gym.set_camera_location(camera, env, gymapi.Vec3(*eye), gymapi.Vec3(*target))
        return camera

    def create_camera_top(self, *, env, isaac_gym):
        """The BEHIND view, saved as the `_behind` video.

        The method name is historical: this camera has always looked from behind the hands, and
        the file it produced was misleadingly called `_top` until the overhead camera below made
        the real thing available.
        """
        return self._record_camera(
            env=env, isaac_gym=isaac_gym, eye=BEHIND_EYE, target=BEHIND_TARGET
        )

    def create_camera_overhead(self, *, env, isaac_gym):
        """Genuinely top-down view, saved as the `_top` video."""
        return self._record_camera(
            env=env, isaac_gym=isaac_gym, eye=OVERHEAD_EYE, target=OVERHEAD_TARGET
        )

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
        self.camera_obs_overhead = None
        if self.camera_handlers_overhead is not None:
            self.camera_obs_overhead = []
            for env, handle in zip(self.envs, self.camera_handlers_overhead):
                self.camera_obs_overhead.append(
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
    obj_reward_share: Tensor,
    training: bool = True,
    enforce_eval_thresholds: bool = True,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

    # type: (Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, Tensor], Tensor, float, float, Dict[str, List[int]], Tensor, bool, bool) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor], Tensor]

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
        threshold_trip = (
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
        )
        # Scoring the thresholds and ENDING the episode on them are two different things. With
        # evalThresholdDryRun the caller wants to watch the whole trajectory play out past the
        # trip, so the clause is still computed (score_eval_metrics reads the same quantities and
        # writes the verdict beside each video) but only error_buf -- a genuine velocity blow-up,
        # after which there is nothing left to watch -- can cut the episode short.
        # NOTE the knock-on: an episode that trips and then runs to max_length is counted
        # `succeeded` below, so the success rate of a dry-run eval is inflated by construction.
        # Read the per-episode verdicts, not the rate.
        if enforce_eval_thresholds:
            failed_execute = threshold_trip | error_buf
        else:
            failed_execute = error_buf
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
        # obj_reward_share is this hand's share of the object terms. It is 1 for the normal
        # one-object-per-hand setup; with a shared object the two hands' shares sum to 1, so the
        # object is rewarded once in total and credited to whichever hand the demo has holding it.
        + obj_reward_share * 10.0 * reward_obj_pos
        + obj_reward_share * 10.0 * reward_obj_rot
        + 0.1 * reward_eef_vel
        + 0.05 * reward_eef_ang_vel
        + 0.1 * reward_joints_vel
        + obj_reward_share * 0.1 * reward_obj_vel
        + obj_reward_share * 0.1 * reward_obj_ang_vel
        + 1.0 * reward_finger_tip_force
        + 0.5 * reward_power
        + 0.5 * reward_wrist_power
    )

    succeeded = (
        progress_buf + 1 >= max_length
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
