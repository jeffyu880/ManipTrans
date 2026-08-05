"""Per-frame dex-retargeting controller that stands in for the ManipTrans policy.

This is the pure-retargeting baseline: no RL, no learned dynamics. Every control step it reads
the human hand target the env already holds, solves robot joint angles to put the robot
fingertips where the human's were, and emits those as the env's action. The RL stack is bypassed
entirely — only the env, its physics, its termination logic and its logging remain.

It produces the SAME action vector the rl_games player normally produces, so
`dexhandmanip_bih.pre_physics_step` is untouched:

    [ base_action (2 hands x (3 pos err + 6 rot6d + 12 dofs)) | residual_action (zeros) ]

which is 78 dims for inspire/BiH with `usePIDControl=true`. Run with `zeroResidual=true` as well,
so the residual half is neutralised in the env even if something feeds it.

--- Why it reads from the env rather than from a pkl ---
`demo_data_{rh,lh}["mano_joints"]` is overwritten every step by `_inject_live()` in live mode and
holds the recorded demo otherwise. Reading it means one code path serves offline and online with
no branching — the whole reason this controller is not a precomputed trajectory.

--- Frames ---
dex-retargeting solves hand SHAPE in a wrist-relative, canonically-oriented frame, and discards
global pose. So the fingers come from the solve and the WRIST comes from the human target, driven
through the env's `usePIDControl` branch. Note that branch consumes an *error*, not a target,
which is why this has to run in the loop against live sim state rather than being precomputed.

That frame comes from the AVP wrist, following Bunny-VisionPro: undo what the loader folded into
`wrist_rot` (loader_to_avp_rotation), then relabel with OPERATOR2AVP. Upstream dex-retargeting
instead fits a frame to the wrist/index-MCP/middle-MCP triangle, having no wrist orientation from
monocular tracking; ours is measured, not inferred, so it carries none of that fit's SVD jitter.
The two differ by 12.8 deg (right) and 179.2 deg (left, mirrored) — a real choice, not rounding.

--- Which URDF ---
`baselines/configs/` points at DEX-URDF's inspire, what dex-retargeting and Bunny-VisionPro were
built against, whose base already matches the frame the keypoints arrive in. See that header for
the trade-offs against ManipTrans's own URDF.
"""

import os

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from main.dataset.transform import aa_to_rotmat, quat_to_rotmat, rotmat_to_rot6d

from baselines.utils import (
    DEFAULT_RETARGETING,
    DEXRET_WRIST_PULLBACK,
    MANO21_JOINT_NAMES,
    OPERATOR2AVP,
    default_config_path,
    dex_urdf_dir,
    pull_wrist_back,
    retarget_ref_value,
)

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")

# MANO-21 slots dex-retargeting reads: the wrist (which the keypoints are centred on) plus the five
# fingertips. The rest are never touched — four of them (*_distal) do not exist in the env's packed
# buffer at all. The frame comes from the wrist alone, so the MCPs the 3-point fit used are gone.
REQUIRED_MANO_SLOTS = (0, 4, 8, 12, 16, 20)

# --- PD + feedforward wrist control (wrist_mode="pd_ff", requires usePIDControl=False) ---
#
# The env's PID branch is also a force controller, but its gains live on the dexhand class
# (inspire.py), are shared with every other PID run, and force BOTH hands onto the RH's values
# (dexhandmanip_bih.py:147-152). Worse, it is handed a position ERROR only, so it cannot know the
# target is moving. These gains live here instead, where they are ours to sweep.
#
# Derived, not guessed. The hand is ~0.192 kg and the asset applies linear_damping=20, i.e. a drag
# of -m*d*v = -3.84 N per m/s. Critical damping is 2*sqrt(Kp*m):
#   Kp=20  (the env default) -> 2*sqrt(20*0.192)  = 3.92, and drag alone gives 3.84 -> already
#          critically damped, so its Kd=0.1 is doing 2.6% of the work and is effectively inert.
#   Kp=200 -> 2*sqrt(200*0.192) = 12.4, minus 3.84 of drag -> Kd = 8.5, and now Kd matters.
# Kp=200 also puts steady-state droop at 5 mm per newton of contact (was 50 mm at Kp=20) and the
# closed loop at ~5.1 Hz — near the practical ceiling for a 60 Hz loop, so do not go much stiffer.
WRIST_KP = 200.0        # N/m
WRIST_KD = 8.5          # N.s/m, on the MEASURED velocity (strictly causal)
# Rotation needs gains ~2000x smaller than the linear ones, because the hand's rotational inertia
# is 9.44e-5 kg.m^2 against its 0.192 kg of mass. A sampled controller is stable only while
# Kp*dt^2/J < 4 and Kd*dt/J < 2; at 60 Hz with this inertia that caps Kp_rot at 1.36 and Kd_rot at
# 0.011. The first pass used 2.0 and 0.1 -- 1.5x and 9x past those bounds -- which oscillated at
# the control rate and grew every step.
# These sit at the same fraction of the bound as the linear gains (7% and 37%), and agree with the
# physics: critical damping is 2*sqrt(Kp_rot*I) = 0.0061, of which angular_damping=20 already
# supplies I*d = 0.0019, leaving 0.0042.
WRIST_KP_ROT = 0.10     # N.m/rad, on the axis-angle of the orientation error
WRIST_KD_ROT = 0.0042   # N.m.s/rad, on the measured angular velocity

# Must match asset_options.linear_damping in dexhandmanip_bih.py:422. Compensating it is exact
# arithmetic, not a tuned term: the drag is -m*d*v on the MEASURED velocity, so adding +m*d*v back
# cancels it outright and removes the standing lag it would otherwise cause when tracking a moving
# target (3.84*v/Kp = ~10 mm at 0.5 m/s).
ASSET_LINEAR_DAMPING = 20.0

# Velocity feedforward. Kd*(v_target - v) is what takes tracking lag at 0.5 m/s from ~31 mm to ~0,
# but the causal target velocity is noisy, and feeding that forward injects force jitter. It cannot
# DESTABILISE anything (feedforward is open loop, outside the feedback path), so the fix is simply
# to smooth it hard and scale it back:
#   WRIST_FF_GAIN  0 = no feedforward (accept the lag), 1 = full. Halving it halves both the lag
#                  reduction and the injected noise.
#   WRIST_FF_EMA   smoothing on the target velocity. Small = smoother but laggier, which eats into
#                  the very lag the feedforward exists to remove — that trade is the thing to sweep.
# The velocity is differenced from the target position INSIDE this controller and EMA'd here, so it
# is strictly causal, identical offline and live, and independent of the env's `causal` flag (whose
# default path is NOT causal: np.gradient plus a symmetric Gaussian, ~+-133 ms of look-ahead).
WRIST_FF_GAIN = 0.5
WRIST_FF_EMA = 0.2

# Ceiling on the free-joint wrist offset (`position` config only). The solve is a local NLS and
# a degenerate frame can hand back a nonsense pose; without a clamp that teleports the hand.
# 8 cm is well past any real hand-size mismatch and well short of leaving the workspace.
MAX_WRIST_OFFSET = 0.08


def solver_dof_perm(solver, dexhand):
    """Indices selecting the solver's solved joints into the env's dof order.

    Matches BY NAME, not position — a silent reordering would be invisible and corrupt every frame.
    Compares on the stripped name because dex-urdf omits the R_/L_ prefix that `dexhand.dof_names`
    carries, which also keeps this working against ManipTrans's own (prefixed) URDF.

    Args:
        solver: A built SeqRetargeting solver.
        dexhand: A DexHand instance (e.g. `env.dexhand_rh`).

    Returns:
        (n_dofs,) int array indexing the solver's output into `dexhand.dof_names` order.
    """
    solved = list(solver.optimizer.robot.dof_joint_names)
    strip = lambda name: name[2:] if name[:2] in ("R_", "L_") else name
    lookup = {strip(name): i for i, name in enumerate(solved)}
    missing = [n for n in dexhand.dof_names if strip(n) not in lookup]
    assert not missing, (
        f"env dofs {missing} are absent from the solver's joints ({solved}). The retargeting "
        f"config and the env must describe the SAME hand — check urdf_path in baselines/configs/ "
        f"against dexhand.urdf_path."
    )
    return np.array([lookup[strip(n)] for n in dexhand.dof_names])


def loader_to_avp_rotation(dexhand):
    """Rotation undoing what the loader added to the AVP wrist, recovering the headset's own frame.

    The loader post-multiplies `dexhand.relative_rotation` onto the AVP wrist, and the left hand
    gets a further 180-deg-about-Y mirror; undoing both recovers what the headset reported. Read
    off the dexhand rather than hardcoded, so it tracks relative_rotation instead of going stale.

    Args:
        dexhand: A DexHand instance (e.g. `env.dexhand_rh`).

    Returns:
        (3, 3) rotation M with R_avp_wrist == R_loader_wrist @ M.
    """
    rel = np.asarray(dexhand.relative_rotation, dtype=np.float64)
    if dexhand.side == "lh":
        # 180 deg about Y — the same correction as AVP_LH_WRIST_CORRECTION in
        # my_dataset_LH.__getitem__ and _LH_WRIST_CORRECTION in live/live_target_source.py.
        return (rel @ R.from_rotvec([0.0, np.pi, 0.0]).as_matrix()).T
    return rel.T


def hand_local_transform(hand, wrist_rot_aa, loader_to_avp):
    """Single 3x3 taking wrist-centred world keypoints into the frame the optimiser expects.

    Bunny-VisionPro's method: recover the headset's wrist rotation and relabel its axes with a
    fixed permutation. Nothing is fitted, so there is no calibration to go stale. The keypoints are
    not consulted at all — depending only on the wrist is what avoids the 3-point fit's SVD jitter.

    Args:
        hand: "right" or "left".
        wrist_rot_aa: (3,) axis-angle wrist rotation in the loader/env frame.
        loader_to_avp: (3, 3) from loader_to_avp_rotation(dexhand), undoing what the loader folded
            into wrist_rot.

    Returns:
        (3, 3) matrix M, applied as `local = centred @ M`.
    """
    avp_wrist = R.from_rotvec(wrist_rot_aa).as_matrix() @ loader_to_avp
    return avp_wrist @ OPERATOR2AVP[hand]


def mano_slot_to_hand_name():
    """Map each MANO-21 keypoint index to ManipTrans's `mano_joints` key.

    Bridges the two naming schemes via the repo's own table rather than a second copy of it:
    `MANO21_JOINT_NAMES` gives the AVP name per MANO slot, and `AVP_TO_MANO_JOINTS` gives
    {ManipTrans key: AVP name}.

    Returns:
        dict {int: str} for every MANO-21 slot that has a ManipTrans counterpart.
    """
    from main.dataset.object_sets import AVP_TO_MANO_JOINTS

    avp_to_hand = {avp: hand for hand, avp in AVP_TO_MANO_JOINTS.items()}
    return {
        slot: avp_to_hand[avp_name]
        for slot, avp_name in enumerate(MANO21_JOINT_NAMES)
        if avp_name in avp_to_hand
    }


def packed_row_by_hand_name(dexhand):
    """Row index of each `mano_joints` key inside the env's flattened per-step buffer.

    The env packs `mano_joints` by walking `dexhand.body_names` and taking `to_hand(body)[0]`,
    skipping the wrist (`dexhandmanip_bih.py:1259-1285`). Some hand names map to two bodies
    (inspire's `thumb_proximal`), so the first occurrence wins.

    Args:
        dexhand: A DexHand instance (e.g. `env.dexhand_rh`).

    Returns:
        dict {str: int} mapping a `mano_joints` key to its row in the packed buffer.
    """
    rows, row = {}, 0
    for body in dexhand.body_names:
        hand_name = dexhand.to_hand(body)[0]
        if hand_name == "wrist":
            continue
        rows.setdefault(hand_name, row)
        row += 1
    return rows


class DexRetargetController:
    """Solves robot joint angles from the env's human hand target, once per control step.

    Args:
        env: The live `DexHandManipBiHEnv`.
        robot: dex-retargeting robot name; only "inspire" has ManipTrans configs today.
        retargeting: which optimiser to solve with — "dexpilot" (inter-finger vectors plus
            DexPilot's thumb-to-finger grasp projection) or "vector" (wrist-to-fingertip vectors
            only). Set by the dexRetType config knob.
        wrist_pullback: fraction of the wrist-to-middle-MCP span to pull the commanded wrist back
            by, matching what dexret2dexhand bakes into its pkls so the two paths agree. NOTE this
            commands away from `demo_data["wrist_pos"]`, which is also what the reward tracks, so
            it trades wrist-tracking score for clearance. 0 disables.
    """

    def __init__(self, env, robot="inspire", retargeting=DEFAULT_RETARGETING,
                 wrist_pullback=DEXRET_WRIST_PULLBACK, wrist_mode="pid"):
        from dex_retargeting.retargeting_config import RetargetingConfig

        self.env = env
        self.wrist_pullback = wrist_pullback
        self.wrist_mode = wrist_mode
        assert wrist_mode in ("pid", "pd_ff"), f"wrist_mode must be pid or pd_ff, got {wrist_mode!r}"
        if wrist_mode == "pid":
            assert env.use_pid_control, (
                "wrist_mode='pid' drives the wrist through the env's PID branch, which only exists "
                "when usePIDControl=true. Re-run with usePIDControl=true, or use wrist_mode='pd_ff'."
            )
        else:
            assert not env.use_pid_control, (
                "wrist_mode='pd_ff' emits wrist FORCE/TORQUE, which the env only accepts on its "
                "non-PID branch (base_action[0:3] force, [3:6] torque). Re-run with "
                "usePIDControl=false."
            )
            # Total mass of the articulation, read from the URDF the sim actually loaded rather
            # than hardcoded — the gain derivation above is only valid for this number.
            from lxml import etree

            masses = [
                float(m.get("value"))
                for m in etree.parse(env.dexhand_rh.urdf_path).getroot().iter("mass")
            ]
            self.hand_mass = sum(masses)
            # Per-hand, per-env state for the causal target-velocity estimate. None until the
            # first step, and re-seeded on reset (progress_buf == 0) so the position jump at a
            # reset does not differentiate into an enormous spurious feedforward spike.
            self.prev_target_pos = {"rh": None, "lh": None}
            self.ff_velocity = {"rh": None, "lh": None}

            # Optional per-step wrist trace, written once at exit (see wrist_pd_ff).
            log_path = os.environ.get("MANIPTRANS_DEXRET_LOG", "")
            self.wrist_log = [] if log_path else None
            if log_path:
                import atexit
                import csv

                def dump_wrist_log():
                    """Write the accumulated wrist trace as a CSV. Registered atexit."""
                    if not self.wrist_log:
                        return
                    with open(log_path, "w", newline="") as handle:
                        writer = csv.writer(handle)
                        writer.writerow([
                            "step", "side", "err_x", "err_y", "err_z", "err_norm",
                            "rot_err_deg", "speed", "target_speed", "force_n", "torque_nm",
                            "ref_x", "ref_y", "ref_z", "act_x", "act_y", "act_z",
                        ])
                        writer.writerows(self.wrist_log)
                    print(f"[dexret] wrote {len(self.wrist_log)} wrist rows to {log_path}")

                    # Plot it straight away, in a child process: this runs during interpreter
                    # shutdown with the sim being torn down, and pulling matplotlib into that is
                    # the fragile half of the job. Mirrors _dump_pinch_gap.
                    if os.environ.get("MANIPTRANS_DEXRET_PLOT", "1") == "0":
                        return
                    import subprocess
                    import sys

                    script = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data_stats", "plot_dexret_wrist.py",
                    )
                    try:
                        subprocess.run([sys.executable, script, log_path], timeout=120, check=True)
                    except Exception as exc:
                        print(f"[dexret] auto-plot failed ({exc}); run: python {script} {log_path}")

                atexit.register(dump_wrist_log)

        # dex-retargeting resolves the config's relative `urdf_path` against this.
        RetargetingConfig.set_default_urdf_dir(dex_urdf_dir())

        self.solvers, self.dof_perm, self.rows, self.slot_names = {}, {}, {}, mano_slot_to_hand_name()
        self.dummy_idx = {}
        self.loader_to_avp = {}
        for side, hand in (("rh", "right"), ("lh", "left")):
            config_path = default_config_path(robot, hand, retargeting)
            assert os.path.exists(config_path), (
                f"no retargeting config at {config_path}. Only inspire ships one; add a config "
                f"for '{robot}' modelled on the inspire ones in baselines/configs/."
            )
            solver = RetargetingConfig.load_from_file(config_path).build()
            dexhand = getattr(env, f"dexhand_{side}")

            self.dof_perm[side] = solver_dof_perm(solver, dexhand)
            # The `position` config adds a 6-DOF free joint; the other configs do not. Only the
            # three translation DOFs are used -- the rotation ones would fight the wrist
            # orientation we already take from the human.
            solved_names = list(solver.optimizer.robot.dof_joint_names)
            dummy = [f"dummy_{axis}_translation_joint" for axis in "xyz"]
            self.dummy_idx[side] = (
                np.array([solved_names.index(n) for n in dummy])
                if all(n in solved_names for n in dummy) else None
            )
            self.solvers[side] = solver
            self.rows[side] = packed_row_by_hand_name(dexhand)
            self.loader_to_avp[side] = loader_to_avp_rotation(dexhand)
            # wrist_error reads this row every step; fail at construction instead of per-frame.
            assert not wrist_pullback or "middle_proximal" in self.rows[side], (
                f"{side}: wrist_pullback needs 'middle_proximal' in the env's packed mano_joints "
                f"buffer, which has {sorted(self.rows[side])}. Pass wrist_pullback=0 to skip it."
            )

    def mano21_from_env(self, side, env_idx, step_idx):
        """Assemble the MANO-21 keypoint array for one hand from the env's current target.

        Only the six slots dex-retargeting reads are filled; the rest stay zero and are never
        indexed. Values are in the Isaac Gym world frame, metres — the same frame offline demo
        data and the live stream both end up in.

        Args:
            side: "rh" or "lh".
            env_idx: Which env row to read.
            step_idx: Index into the demo/live buffer (the env's clamped `progress_buf`).

        Returns:
            (21, 3) float64 array of world-frame keypoints.
        """
        demo = getattr(self.env, f"demo_data_{side}")
        joints = demo["mano_joints"][env_idx, step_idx].reshape(-1, 3)
        kp = np.zeros((21, 3), dtype=np.float64)
        kp[0] = demo["wrist_pos"][env_idx, step_idx].cpu().numpy()
        for slot in REQUIRED_MANO_SLOTS:
            if slot == 0:
                continue
            kp[slot] = joints[self.rows[side][self.slot_names[slot]]].cpu().numpy()
        return kp

    def solve_dofs(self, side, kp21, wrist_rot_aa):
        """Retarget one frame of human keypoints to robot joint angles.

        Args:
            side: "rh" or "lh".
            kp21: (21, 3) world-frame MANO keypoints.
            wrist_rot_aa: (3,) axis-angle wrist rotation from the env's target buffer, which is
                what the hand-local frame is built from.

        Returns:
            ((n_dofs,) joint angles in radians ordered to match `dexhand.dof_names`,
             (3,) world-frame wrist offset the solve asks for, or None when the config has no
             free joint).
        """
        hand = "right" if side == "rh" else "left"
        centred = kp21 - kp21[0:1, :]
        frame = hand_local_transform(hand, wrist_rot_aa, self.loader_to_avp[side])
        local = centred @ frame

        solver = self.solvers[side]
        solved = solver.retarget(retarget_ref_value(solver, local))

        # With add_dummy_free_joint (the `position` config) the optimiser also solves WHERE the
        # hand goes, not just how it closes -- so those three translation DOFs are the correction
        # that puts the robot's fingertips on the human's. Supplying the human wrist and ignoring
        # them is what leaves the grasp off-centre, and no scalar pullback can fix it because the
        # pullback moves along one axis while this offset is 3-D.
        #
        # `local = centred @ frame`, so a local column vector maps back to world as `frame @ v`.
        offset = None
        if self.dummy_idx[side] is not None:
            offset = frame @ solved[self.dummy_idx[side]]
            norm = float(np.linalg.norm(offset))
            if norm > MAX_WRIST_OFFSET:
                # A degenerate solve would otherwise teleport the hand. Clamp rather than trust it.
                offset = offset * (MAX_WRIST_OFFSET / norm)
        return solved[self.dof_perm[side]], offset

    def wrist_error(self, side, env_idx, step_idx, target_pos):
        """Wrist pose error the env's PID branch consumes: 3 position + 6 rotation dims.

        The PID branch is given an ERROR, not a target (`pre_physics_step`), so this differences
        the human target against the current sim wrist every step.

        Args:
            side: "rh" or "lh".
            env_idx: Which env row to read.
            step_idx: Index into the demo/live buffer.
            target_pos: (3,) wrist position target, with the pullback and any solved free-joint
                offset already applied by compute_action.

        Returns:
            (9,) float32 array: world-frame position error then the 6D form of the rotation error.
        """
        demo = getattr(self.env, f"demo_data_{side}")
        # Actor root state, for the same reason as wrist_pd_ff: the rigid-body tensor lags a reset
        # until the next simulate(), so it is stale on the first control step of a run.
        current = getattr(self.env, f"_{side}_base_state")[env_idx]

        pos_error = target_pos - current[:3]

        # demo wrist_rot is axis-angle; the sim quaternion is Isaac Gym's xyzw while
        # quat_to_rotmat wants wxyz (verified empirically), hence the [3, 0, 1, 2] reindex.
        target_rotmat = aa_to_rotmat(demo["wrist_rot"][env_idx, step_idx][None])[0]
        current_rotmat = quat_to_rotmat(current[3:7][[3, 0, 1, 2]][None])[0]
        rot_error = rotmat_to_rot6d(target_rotmat @ current_rotmat.transpose(-1, -2))

        return torch.cat([pos_error, rot_error.reshape(-1)]).float().cpu().numpy()

    def update_target_velocity(self, side, target_pos):
        """Causal EMA estimate of how fast the wrist target is moving.

        Differenced here rather than read from `demo_data["wrist_velocity"]` on purpose. That
        buffer follows the env's `causal` flag, whose DEFAULT path is not causal at all
        (`base.py:119-121` uses np.gradient plus a symmetric Gaussian, sigma=2 — about +-133 ms of
        look-ahead at 60 Hz). Feeding that forward would hand the offline baseline anticipation the
        live system cannot have. Differencing the target we were actually given, one step at a
        time, is strictly causal and behaves identically offline and live.

        Args:
            side: "rh" or "lh".
            target_pos: (num_envs, 3) this control step's wrist target, world frame.

        Returns:
            (num_envs, 3) smoothed target velocity, zeroed on the first step and on any env that
            has just reset.
        """
        control_dt = self.env.dt * self.env.control_freq_inv
        previous = self.prev_target_pos[side]
        if previous is None:
            velocity = torch.zeros_like(target_pos)
            self.ff_velocity[side] = torch.zeros_like(target_pos)
        else:
            raw = (target_pos - previous) / control_dt
            velocity = WRIST_FF_EMA * raw + (1.0 - WRIST_FF_EMA) * self.ff_velocity[side]

        # A reset teleports the target, and differencing that jump yields a huge spike. Re-seed
        # those envs instead, matching what LiveTargetSource does on its first frame.
        just_reset = (self.env.progress_buf == 0).unsqueeze(-1)
        velocity = torch.where(just_reset, torch.zeros_like(velocity), velocity)

        self.ff_velocity[side] = velocity
        self.prev_target_pos[side] = target_pos.clone()
        return velocity

    def wrist_pd_ff(self, side, target_pos, target_velocity):
        """Wrist force and torque for one hand: PD on the pose error, plus feedforward.

        Emits what the env's NON-PID branch expects — a force and a torque, already divided by the
        scaling it applies (`base_action * base_wrist_dt * scale * 500`), so the values commanded
        here arrive as the newtons computed here.

        The force is
            Kp*(x_t - x) + Kd*(v_t*ff - v) + m*d*v
        where the last term cancels the asset's own drag (see ASSET_LINEAR_DAMPING). Both the
        damping and the drag term read the MEASURED velocity from sim state, so only the small
        feedforward term depends on any derivative of the demo.

        Args:
            side: "rh" or "lh".
            target_pos: (num_envs, 3) wrist position target, world frame.
            target_velocity: (num_envs, 3) causal target velocity from update_target_velocity.

        Returns:
            (num_envs, 6) float32 tensor: 3 force dims then 3 torque dims, in the env's action
            units (nominally within [-1, 1]; saturation is reported by compute_action).
        """
        env = self.env
        demo = getattr(env, f"demo_data_{side}")
        # The hand's ACTOR ROOT state, not _rigid_body_state[wrist]. reset_idx writes this buffer
        # directly (:2094), whereas the rigid-body tensor only reflects a reset after a simulate()
        # call — so on the first control step of a run the rigid-body view still holds the spawn
        # pose with a velocity that was never written, which reads as tens of m/s of garbage and
        # saturates the command. This is also the buffer the env's own reward uses as "base_state".
        state = getattr(env, f"_{side}_base_state")

        position, velocity = state[:, :3], state[:, 7:10]
        angular_velocity = state[:, 10:13]

        spring = WRIST_KP * (target_pos - position)
        damper = WRIST_KD * (WRIST_FF_GAIN * target_velocity - velocity)
        drag = self.hand_mass * ASSET_LINEAR_DAMPING * velocity
        force = spring + damper + drag

        # MANIPTRANS_DEXRET_DEBUG=N: dump the force breakdown for the first N control steps of env
        # 0. A saturated command says only that SOMETHING is huge; this says which term.
        debug_steps = int(os.environ.get("MANIPTRANS_DEXRET_DEBUG", "0") or 0)
        if debug_steps and getattr(self, "debug_count", 0) < debug_steps and side == "rh":
            self.debug_count = getattr(self, "debug_count", 0) + 1
            print(
                f"\033[96m[dexret {side} step {self.debug_count}] "
                f"target={target_pos[0].tolist()} pos={position[0].tolist()}\n"
                f"    |err|={float((target_pos - position)[0].norm()):.4f} m  "
                f"|v|={float(velocity[0].norm()):.3f} m/s  "
                f"|v_t|={float(target_velocity[0].norm()):.3f} m/s\n"
                f"    spring={float(spring[0].norm()):8.2f} N  damper={float(damper[0].norm()):8.2f} N  "
                f"drag={float(drag[0].norm()):8.2f} N  total={float(force[0].norm()):8.2f} N\033[0m"
            )

        # Orientation: axis-angle of the error rotation is the natural "position error" here.
        # No feedforward on rotation yet — it would need the target angular velocity, which is the
        # noisier of the two derivatives; the damping term alone is strictly causal.
        step_idx = torch.clamp(env.progress_buf, 0, demo["wrist_rot"].shape[1] - 1)
        rows = torch.arange(env.num_envs, device=env.device)
        target_rotmat = aa_to_rotmat(demo["wrist_rot"][rows, step_idx])
        current_rotmat = quat_to_rotmat(state[:, 3:7][:, [3, 0, 1, 2]])
        error_aa = torch.as_tensor(
            R.from_matrix(
                (target_rotmat @ current_rotmat.transpose(-1, -2)).cpu().numpy()
            ).as_rotvec(),
            device=env.device,
            dtype=torch.float32,
        )
        torque = WRIST_KP_ROT * error_aa - WRIST_KD_ROT * angular_velocity

        # MANIPTRANS_DEXRET_LOG=<path.csv>: record what the wrist controller is actually tracking,
        # for env 0. The pinch CSV covers fingertips and contact but carries nothing about the
        # wrist, so without this there is no way to see the tracking error being tuned against.
        if self.wrist_log is not None:
            position_error = (target_pos - position)[0]
            self.wrist_log.append((
                int(self.env.progress_buf[0]), side,
                float(position_error[0]), float(position_error[1]), float(position_error[2]),
                float(position_error.norm()),
                float(np.degrees(np.linalg.norm(error_aa[0].cpu().numpy()))),
                float(velocity[0].norm()), float(target_velocity[0].norm()),
                float(force[0].norm()), float(torque[0].norm()),
                # absolute positions too, so reference and robot can be overlaid rather than only
                # their difference — a flat error hides whether both are moving or both are stuck
                float(target_pos[0][0]), float(target_pos[0][1]), float(target_pos[0][2]),
                float(position[0][0]), float(position[0][1]), float(position[0][2]),
            ))

        # Undo the env's own scaling so the action carries our newtons verbatim.
        force_action = force / (env.base_wrist_dt * env.translation_scale * 500.0)
        torque_action = torque / (env.base_wrist_dt * env.orientation_scale * 200.0)
        return torch.cat([force_action, torque_action], dim=-1).float()

    def compute_action(self):
        """Build the full action tensor for this control step.

        Returns:
            (num_envs, action_dim) float32 tensor in [-1, 1] — the base half filled from the
            retargeting solve, the residual half zero.
        """
        from maniptrans_envs.lib.utils import torch_jit_utils

        env = self.env
        n_env = env.num_envs
        # pid: 3 position error + 6D rotation. pd_ff: 3 force + 3 torque, which is what the env's
        # non-PID branch consumes. Both asserted against use_pid_control in __init__.
        root_dim = 9 if self.wrist_mode == "pid" else 6
        n_dofs = env.num_dexhand_rh_dofs
        per_hand = root_dim + n_dofs

        base = torch.zeros((n_env, 2 * per_hand), device=env.device)
        step_idx = torch.clamp(env.progress_buf, 0, getattr(env, "demo_data_rh")["mano_joints"].shape[1] - 1)

        for hand_i, side in enumerate(("rh", "lh")):
            lower = getattr(env, f"dexhand_{side}_dof_lower_limits")
            upper = getattr(env, f"dexhand_{side}_dof_upper_limits")
            offset = hand_i * per_hand
            demo = getattr(env, f"demo_data_{side}")

            # Fingers first, for every env: with the `position` config the solve also returns the
            # wrist offset that puts the robot's fingertips on the human's, and the wrist target
            # below depends on it. With `vector`/`dexpilot` the offset is None and the wrist is the
            # human's, as before.
            solve_offsets = torch.zeros((n_env, 3), device=env.device)
            for env_idx in range(n_env):
                t = int(step_idx[env_idx])
                # Same buffer the wrist target reads, so the finger frame and the wrist agree —
                # and _inject_live keeps both current in live mode.
                wrist_rot_aa = demo["wrist_rot"][env_idx, t].cpu().numpy()
                qpos, wrist_offset = self.solve_dofs(
                    side, self.mano21_from_env(side, env_idx, t), wrist_rot_aa
                )
                qpos = torch.as_tensor(qpos, device=env.device, dtype=torch.float32)
                if wrist_offset is not None:
                    solve_offsets[env_idx] = torch.as_tensor(
                        wrist_offset, device=env.device, dtype=torch.float32
                    )
                # exact inverse of the env's scale() at pre_physics_step, so the commanded angle
                # survives the [-1,1] action interface unchanged
                base[env_idx, offset + root_dim : offset + per_hand] = torch_jit_utils.unscale(
                    qpos, lower, upper
                )

            rows = torch.arange(n_env, device=env.device)
            target_pos = demo["wrist_pos"][rows, step_idx]

            # The pullback and the solved offset are two answers to the SAME question -- where the
            # hand base goes relative to the human wrist -- so applying both displaces the hand
            # twice. The dummy translation is measured from the keypoint origin, i.e. the human
            # wrist itself, so it already includes whatever standoff the grasp needs; adding a
            # pullback on top pushes the hand further back along the palm axis, which for a
            # downward grasp reads as the hand floating above the object.
            if self.dummy_idx[side] is not None:
                target_pos = target_pos + solve_offsets
            elif self.wrist_pullback:
                # No free joint (vector/dexpilot): the wrist is the human's, and the pullback is
                # the only standoff available.
                middle = demo["mano_joints"][rows, step_idx].reshape(n_env, -1, 3)[
                    :, self.rows[side]["middle_proximal"]
                ]
                target_pos = pull_wrist_back(target_pos, middle, self.wrist_pullback)

            if self.wrist_mode == "pd_ff":
                # The feedforward keeps per-env state, so it must be stepped exactly once per side
                # per control step — hence batched here rather than inside the loop above.
                target_velocity = self.update_target_velocity(side, target_pos)
                base[:, offset : offset + root_dim] = self.wrist_pd_ff(
                    side, target_pos, target_velocity
                )
            else:
                for env_idx in range(n_env):
                    base[env_idx, offset : offset + root_dim] = torch.as_tensor(
                        self.wrist_error(side, env_idx, int(step_idx[env_idx]),
                                         target_pos[env_idx]),
                        device=env.device,
                    )

        # The [-1,1] clamp below is a hard force ceiling in pd_ff mode: at translationScale=1 the
        # env turns action=1 into base_wrist_dt*1*500 = 8.33 N. Saturating means the commanded
        # force was silently truncated, so say so rather than let it look like tracking error.
        if self.wrist_mode == "pd_ff":
            wrist_cols = torch.cat([
                base[:, i * per_hand : i * per_hand + root_dim] for i in (0, 1)
            ], dim=-1)
            if wrist_cols.abs().max() > 1.0 and not getattr(self, "saturation_warned", False):
                self.saturation_warned = True
                print(
                    f"\033[1;93m[dexret] wrist command saturated "
                    f"(|action|={wrist_cols.abs().max():.2f} > 1). The force ceiling is "
                    f"base_wrist_dt*translationScale*500 = "
                    f"{env.base_wrist_dt * env.translation_scale * 500:.1f} N; raise "
                    f"translationScale for headroom.\033[0m"
                )

        residual = torch.zeros((n_env, env.num_actions), device=env.device)
        return torch.clamp(torch.cat([base, residual], dim=1), -1.0, 1.0)
