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
import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from main.dataset.transform import aa_to_rotmat, quat_to_rotmat, rotmat_to_rot6d

from baselines.utils import (
    DEFAULT_RETARGETING,
    DEXRET_FIT_CALIB_FRAMES,
    DEXRET_FIT_CALIB_PINCH_FRAC,
    DEXRET_FIT_CALIB_SAMPLES,
    DEXRET_FIT_MAX_ANGLE_DEG,
    DEXRET_FIT_MAX_TRANSLATION,
    DEXRET_FIT_MODE,
    DEXRET_FIT_OVERRIDE_FREE_JOINT,
    DEXRET_FIT_POINTS,
    DEXRET_FIT_WORLD_FREEZE,
    DEXRET_FIT_WEIGHTS,
    DEXRET_ESCAPE_DIST,
    DEXRET_PROJECT_DIST,
    DEXRET_SCALING_FACTOR,
    DEXRET_SOLVE_URDF,
    DEXRET_WRIST_FIT,
    DEXRET_WRIST_PULLBACK,
    MANO21_JOINT_NAMES,
    OPERATOR2AVP,
    average_rigid,
    calibration_path,
    default_config_path,
    solve_urdf_dir,
    fit_wrist_to_fingertips,
    load_calibration,
    pull_wrist_back,
    save_calibration,
    retarget_ref_value,
    rotmat_to_rotvec,
    rotvec_to_rotmat,
    tip_rms,
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
#
# MEASURED CONSEQUENCE of sitting this far below the bound: wrist ROTATION error runs 12-17 deg
# mean (34 deg peak) through a lift, against 1.6 mm of position error. The fingertips hang ~80-100
# mm below the wrist, so degrees of missing pitch are millimetres of missing fingertip height --
# this is a direct cost to how high the hand lifts. Sweep with MANIPTRANS_DEXRET_KP_ROT /
# MANIPTRANS_DEXRET_KD_ROT.
#
# Feasible range at 60 Hz with J = 9.44e-5: Kp_rot < 1.36, Kd_rot < 2J/dt = 0.0113. Critical
# damping is Kd = 2*sqrt(Kp*J) - J*angular_damping, so Kp = 0.3 needs Kd = 0.0087 (77% of the Kd
# bound) and anything past Kp ~ 0.35 cannot be critically damped at this rate -- it will ring.
# 0.30 / 0.0087, raised from 0.10 / 0.0042. Swept on three demos: rotation error over the lift falls
# 15.9 -> 7.5 deg, the fingertips' rise RELATIVE to the wrist goes 7.6 -> 29.5 mm (the human's is
# ~32), the lift shortfall closes 36.9 -> 15.1 mm and the cap is lifted 94.7 -> 110.5 mm, all with
# RH contact unchanged (59.8 -> 59.6%). Kp_rot = 0.50 tracks rotation better still (4.8 deg) but
# cannot be critically damped at 60 Hz, and it rings: cap lift collapses to 76 mm and contact to
# 51%. 0.30 is the largest value that stays critically dampable.
WRIST_KP_ROT = 0.30     # N.m/rad, on the axis-angle of the orientation error
WRIST_KD_ROT = 0.0087   # N.m.s/rad, on the measured angular velocity

WRIST_KP_ROT = float(os.environ.get("MANIPTRANS_DEXRET_KP_ROT", WRIST_KP_ROT))
WRIST_KD_ROT = float(os.environ.get("MANIPTRANS_DEXRET_KD_ROT", WRIST_KD_ROT))

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

# The robot links the fingertip fit matches, and the MANO-21 slots they are matched against. Same
# order in both, and the same five tips every config already targets — see `finger_tip_link_names`
# in baselines/configs/*.yml. Unprefixed because the fit runs against dex-urdf, like the solve.
FIT_TIP_LINKS = ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip")
FIT_MANO_SLOTS = (4, 8, 12, 16, 20)

# DEXRET_FIT_POINTS="all": every joint that exists on BOTH sides, as (robot link, MANO slot). The
# link names are shared between dex-urdf and ManipTrans's inspire (modulo the R_/L_ prefix, which
# the fit does not use since it runs against dex-urdf), and the MANO slots come straight from
# MANO21_JOINT_NAMES via AVP_TO_MANO_JOINTS. Verified present in both: 16 links in the URDF, 16
# rows in the packed buffer.
#
# The four finger *_distal joints are absent from ManipTrans's packed buffer by design
# (inspire.hand2dex_mapping lists them as "missing"), so 16 is the ceiling, not 20.
FIT_ALL_PAIRS = (
    ("thumb_proximal", 1), ("thumb_intermediate", 2), ("thumb_distal", 3), ("thumb_tip", 4),
    ("index_proximal", 5), ("index_intermediate", 6), ("index_tip", 8),
    ("middle_proximal", 9), ("middle_intermediate", 10), ("middle_tip", 12),
    ("ring_proximal", 13), ("ring_intermediate", 14), ("ring_tip", 16),
    ("pinky_proximal", 17), ("pinky_intermediate", 18), ("pinky_tip", 20),
)

# Live warm-up for DEXRET_FIT_MODE="constant" when DEXRET_FIT_CALIB_FRAMES is left at 0. Two
# seconds at 60 Hz: long enough for the operator to have moved through more than one pose, short
# enough that the correction stops drifting before any real manipulation starts.
LIVE_CALIB_FRAMES = 120


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
            it trades wrist-tracking score for clearance. 0 disables. Ignored when `wrist_fit` is
            on, which answers the same question properly.
        wrist_fit: solve the 6-DOF wrist placement that puts the robot's fingertips on the human's,
            instead of the scalar pullback. Applies only to the configs with no free joint
            (vector/dexpilot); `position`/`position_free` already solve their own placement.
        calibrate: live only — capture a fresh wrist-fit calibration this run and write it, rather
            than loading the stored one. Ignored offline, where the demo is the calibration source.
        calib_path: override where the calibration is read from and written to.
        fit_mode: "constant" (fit once and hold) or "per_frame" (re-solve every control step).
            per_frame needs no calibration at all, live or offline.
    """

    def __init__(self, env, robot="inspire", retargeting=DEFAULT_RETARGETING,
                 wrist_pullback=DEXRET_WRIST_PULLBACK, wrist_mode="pid",
                 wrist_fit=DEXRET_WRIST_FIT, calibrate=False, calib_path=None,
                 fit_mode=DEXRET_FIT_MODE):
        from dex_retargeting.retargeting_config import RetargetingConfig

        self.env = env
        self.wrist_pullback = wrist_pullback
        self.wrist_fit = wrist_fit
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
        RetargetingConfig.set_default_urdf_dir(solve_urdf_dir())

        self.solvers, self.dof_perm, self.rows, self.slot_names = {}, {}, {}, mano_slot_to_hand_name()
        self.dummy_idx = {}
        self.loader_to_avp = {}
        self.tip_link_idx = {}
        self.dummy_all_idx = {}
        self.solver_to_sim = {}
        self.base_align = {}
        # (robot link, MANO slot) pairs the fit is solved against, and the slots mano21_from_env
        # therefore has to populate: the solver's own six plus whatever the fit adds.
        self.fit_pairs = (
            FIT_ALL_PAIRS if DEXRET_FIT_POINTS == "all"
            else tuple(zip(FIT_TIP_LINKS, FIT_MANO_SLOTS))
        )
        self.fit_slots = [slot for _, slot in self.fit_pairs]
        self.needed_slots = sorted(set(REQUIRED_MANO_SLOTS) | set(self.fit_slots))
        self.fit_weights = self.resolve_fit_weights()
        self.fit_stats = {
            "frames": 0, "clamped": 0, "rms_before": 0.0, "rms_after": 0.0, "seconds": 0.0,
            # The fit is recomputed from scratch every frame, so it carries no smoothness
            # guarantee of its own — and it feeds a FORCE-controlled wrist, where a jumpy target
            # is a jumpy force. These track how far the correction moves between consecutive
            # frames; compare the mm/frame against the wrist's own motion before trusting it.
            "steps": 0, "d_trans": 0.0, "d_trans_max": 0.0, "d_rot_deg": 0.0,
            "d_trans_max_at": None,
            "calib_frames": 0, "calib_clamped": 0,
            "calib_seconds": 0.0,
        }
        self.prev_fit = {"rh": None, "lh": None}
        # "constant" mode: the frozen hand-local correction per env row, and the samples being
        # accumulated toward it while live mode warms up. Keyed by env because each row may hold a
        # different demo, and one demo's correction is not another's.
        assert fit_mode in ("constant", "per_frame"), (
            f"fit_mode must be constant or per_frame, got {fit_mode!r}"
        )
        self.fit_mode = fit_mode
        self.fit_constant = {"rh": {}, "lh": {}}
        self.fit_world_frozen = {"rh": {}, "lh": {}}
        self.fit_samples = {"rh": {}, "lh": {}}
        self.robot = robot
        self.retargeting = retargeting
        self.calib_path = calib_path or calibration_path(robot, retargeting)
        # Live only: True means this run exists to capture a calibration and should stop once it
        # has one. False means load the stored calibration and teleoperate.
        self.calibrating = False
        for side, hand in (("rh", "right"), ("lh", "left")):
            config_path = default_config_path(robot, hand, retargeting)
            assert os.path.exists(config_path), (
                f"no retargeting config at {config_path}. Only inspire ships one; add a config "
                f"for '{robot}' modelled on the inspire ones in baselines/configs/."
            )
            config = RetargetingConfig.load_from_file(config_path)
            # scaling_factor is consumed by build(), so overriding it here is equivalent to editing
            # the yml -- and keeps the sweep out of a tracked config file. POSITION does not take
            # one, so leave it alone there rather than silently setting an ignored field.
            if DEXRET_SCALING_FACTOR is not None and config.type.lower() != "position":
                config.scaling_factor = DEXRET_SCALING_FACTOR
            # Same deal for the projection thresholds, which only DexPilot reads -- setting them on
            # a vector/position config would be silently ignored, so don't pretend it took.
            if config.type.lower() == "dexpilot":
                if DEXRET_PROJECT_DIST is not None:
                    config.project_dist = DEXRET_PROJECT_DIST
                if DEXRET_ESCAPE_DIST is not None:
                    config.escape_dist = DEXRET_ESCAPE_DIST
            solver = config.build()
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
            # ALL six free-joint DOFs, rotation included. dummy_idx holds only the translations,
            # which is what the offset path consumes; zeroing for FK needs the rotations too, or
            # the fit would see the fingers already rotated into the optimiser's chosen pose and
            # solve a correction on top of it rather than the placement itself.
            self.dummy_all_idx[side] = np.array(
                [i for i, n in enumerate(solved_names) if n.startswith("dummy_")]
            ) if any(n.startswith("dummy_") for n in solved_names) else None
            self.solvers[side] = solver
            self.rows[side] = packed_row_by_hand_name(dexhand)
            self.loader_to_avp[side] = loader_to_avp_rotation(dexhand)

            # The fit runs forward kinematics on the solved pose, so it needs its match frames by
            # index. Resolved once — get_link_index raises on a name the URDF does not have, and
            # discovering that on frame 1 of a sweep is worse than discovering it here. The same
            # goes for the human side: a MANO slot with no row in the packed buffer would read as
            # a zero vector and quietly drag the fit toward the origin.
            # ManipTrans's URDF prefixes every link with R_/L_; dex-urdf does not. Resolve against
            # whichever this model uses rather than assuming, so the same fit_pairs table serves both.
            links = set(solver.optimizer.robot.link_names)
            prefix = "" if FIT_TIP_LINKS[0] in links else ("R_" if hand == "right" else "L_")
            missing = [n for n, _ in self.fit_pairs if prefix + n not in links]
            assert not missing, (
                f"{side}: {missing} absent from {config_path}'s URDF (prefix {prefix!r}). "
                f"Links: {sorted(links)}"
            )
            self.tip_link_idx[side] = [
                solver.optimizer.robot.get_link_index(prefix + name) for name, _ in self.fit_pairs
            ]
            unmatched = [
                self.slot_names.get(slot, f"slot {slot}") for _, slot in self.fit_pairs
                if self.slot_names.get(slot) not in self.rows[side]
            ]
            assert not unmatched, (
                f"{side}: DEXRET_FIT_POINTS={DEXRET_FIT_POINTS!r} needs {unmatched} in the env's "
                f"packed mano_joints buffer, which has {sorted(self.rows[side])}. Use "
                f"DEXRET_FIT_POINTS=tips."
            )

            # The optimiser treats the robot's URDF base as coincident with the MANO frame the
            # keypoints arrive in, so `frame` IS the solver's base orientation in world. The sim,
            # though, is commanded the loader's wrist rotation. The two differ by a constant:
            #     R_loader_wrist == frame @ OPERATOR2AVP.T @ loader_to_avp.T
            # which is exact to 1e-15 for both hands (a 180 deg rotation either way). So the fitted
            # correction reaches the sim as `frame @ R_fit @ solver_to_sim`, and R_fit = I gives
            # back today's command unchanged — the invariant the whole wiring rests on.
            # Constant relating the SOLVED model's base frame to the SIM's hand root. When the two
            # models are the same URDF there is nothing to relate, so it collapses to identity --
            # and `base_align` below then carries the whole rotation into the frame the keypoints
            # are expressed in, which is where it belongs: the optimiser must see the hand in the
            # base frame of the model it is solving.
            avp_to_sim = OPERATOR2AVP[hand].T @ self.loader_to_avp[side].T
            if DEXRET_SOLVE_URDF == "maniptrans":
                self.solver_to_sim[side] = np.eye(3)
                self.base_align[side] = avp_to_sim
            else:
                self.solver_to_sim[side] = avp_to_sim
                self.base_align[side] = np.eye(3)

            # wrist_error reads this row every step; fail at construction instead of per-frame.
            assert not wrist_pullback or "middle_proximal" in self.rows[side], (
                f"{side}: wrist_pullback needs 'middle_proximal' in the env's packed mano_joints "
                f"buffer, which has {sorted(self.rows[side])}. Pass wrist_pullback=0 to skip it."
            )

        # Mutually exclusive with the free joint, which already solves placement: applying both
        # would displace the hand twice, the same fault the pullback had against `position`.
        if self.wrist_fit and any(idx is not None for idx in self.dummy_idx.values()):
            if DEXRET_FIT_OVERRIDE_FREE_JOINT:
                # Keep the fit and drop the free joint's answer. dummy_idx is what the offset path
                # reads, so clearing it routes solve_dofs through the fit branch.
                self.dummy_idx = {side: None for side in self.dummy_idx}
                print(
                    "\033[96m[dexret] free joint present but overridden: the wrist comes from the "
                    "fingertip fit, and the solved free-joint pose is discarded.\033[0m"
                )
            else:
                self.wrist_fit = False
                print(
                    "\033[1;93m[dexret] wrist_fit disabled: this config adds a free joint, which "
                    "already solves the wrist placement. Applying both would displace twice.\033[0m"
                )
        if self.wrist_fit and self.wrist_pullback:
            self.wrist_pullback = 0.0
            print(
                "\033[96m[dexret] wrist_fit supersedes the scalar pullback; pullback set to "
                "0.\033[0m"
            )

        # Live + constant is the only combination that needs a stored calibration: offline the demo
        # buffer IS the calibration source and is pre-passed at first use. Deciding here, once,
        # keeps the per-frame path free of any "have we calibrated yet" branching beyond a dict
        # lookup.
        if self.wrist_fit and self.fit_mode == "constant" and env.live:
            if calibrate:
                self.calibrating = True
                target = DEXRET_FIT_CALIB_FRAMES or LIVE_CALIB_FRAMES
                print(
                    f"\033[1;96m[dexret] CALIBRATION RUN — capturing {target} frames per hand.\n"
                    f"  Move both hands through the motion you intend to teleoperate: reach out,\n"
                    f"  close into the grasp you will use, and open again. The constant is a\n"
                    f"  median over what it sees, so a single held pose calibrates only that pose.\n"
                    f"  This run stops on its own and writes {self.calib_path}.\033[0m"
                )
            elif not self.load_live_calibration():
                self.calibrating = True
                print(
                    f"\033[1;93m[dexret] no calibration at {self.calib_path}, so this run will "
                    f"capture one before teleoperating. Run with dexRetCalibrate=true to do that "
                    f"deliberately.\033[0m"
                )

        # The fit's whole claim is that it reduces fingertip error, so report whether it did rather
        # than leaving that to be assumed. rms_before is the error with the wrist on the human's
        # (what vector/dexpilot did before); rms_after is what the fit achieved, and the gap
        # between them is the fit's entire contribution. A high clamp rate means the limits in
        # constants.py are binding and the numbers below are not the unconstrained fit.
        if self.wrist_fit:
            import atexit

            def dump_fit_stats():
                """Print the fingertip-fit summary. Registered atexit."""
                n = self.fit_stats["frames"]
                if not n:
                    return
                before = self.fit_stats["rms_before"] / n * 1e3
                after = self.fit_stats["rms_after"] / n * 1e3
                # Whether this can run live is a timing question, so answer it here rather than
                # from a benchmark that does not go through the same code. SeqRetargeting counts
                # its own optimiser time; the fit is timed alongside it. Both are PER HAND, so a
                # control step costs twice what is printed, against a 1/(dt*control_freq_inv)
                # budget -- 16.7 ms at the default 60 Hz.
                solve_ms = 1e3 * sum(s.accumulated_time for s in self.solvers.values()) / n
                fit_ms = 1e3 * self.fit_stats["seconds"] / n
                budget_ms = 1e3 * self.env.dt * self.env.control_freq_inv
                print(
                    f"[dexret] per hand-solve: {solve_ms:.2f} ms retarget + {fit_ms:.2f} ms fit; "
                    f"both hands = {2 * (solve_ms + fit_ms):.2f} ms of a {budget_ms:.1f} ms "
                    f"control step ({100 * 2 * (solve_ms + fit_ms) / budget_ms:.0f}% of budget)"
                )
                if self.fit_stats["calib_seconds"]:
                    # One-off, and never paid live -- there the constant is loaded from file.
                    print(
                        f"[dexret] plus a one-off {self.fit_stats['calib_seconds']:.2f} s "
                        f"calibration pre-pass (offline only)"
                    )
                print(
                    f"[dexret] wrist fit ({self.fit_mode}) over {n} solves: fingertip RMS "
                    f"{before:.1f} -> {after:.1f} mm "
                    f"({100 * (1 - after / before) if before else 0:.0f}% lower), "
                    f"clamped on {100 * self.fit_stats['clamped'] / n:.0f}% of frames"
                )
                cf = self.fit_stats["calib_frames"]
                if cf:
                    print(
                        f"[dexret] calibration clamp rate: "
                        f"{100 * self.fit_stats['calib_clamped'] / cf:.0f}% of {cf} solved "
                        f"samples hit a limit "
                        f"({1e3 * DEXRET_FIT_MAX_TRANSLATION:.0f} mm / "
                        f"{DEXRET_FIT_MAX_ANGLE_DEG:.0f} deg)"
                    )
                steps = self.fit_stats["steps"]
                if steps:
                    print(
                        f"[dexret] fit smoothness: correction moves "
                        f"{1e3 * self.fit_stats['d_trans'] / steps:.2f} mm and "
                        f"{self.fit_stats['d_rot_deg'] / steps:.2f} deg per frame on average "
                        f"(worst jump {1e3 * self.fit_stats['d_trans_max']:.1f} mm at "
                        f"{self.fit_stats['d_trans_max_at']})"
                    )

            atexit.register(dump_fit_stats)

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
        for slot in self.needed_slots:
            if slot == 0:
                continue
            kp[slot] = joints[self.rows[side][self.slot_names[slot]]].cpu().numpy()
        return kp

    def solve_dofs(self, side, kp21, wrist_rot_aa, env_idx=0):
        """Retarget one frame of human keypoints to robot joint angles.

        Args:
            side: "rh" or "lh".
            kp21: (21, 3) world-frame MANO keypoints.
            wrist_rot_aa: (3,) axis-angle wrist rotation from the env's target buffer, which is
                what the hand-local frame is built from.
            env_idx: Which env row this solve is for; the constant-mode correction is per env.

        Returns:
            ((n_dofs,) joint angles in radians ordered to match `dexhand.dof_names`,
             (3,) world-frame wrist offset the solve asks for, or None when neither the free joint
             nor the fingertip fit supplies one,
             (3, 3) world-frame wrist rotation the fit asks for, or None when the fit is off).
        """
        hand = "right" if side == "rh" else "left"
        centred = kp21 - kp21[0:1, :]
        frame = hand_local_transform(hand, wrist_rot_aa, self.loader_to_avp[side]) @ self.base_align[side]
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
        offset, rotation = None, None
        if self.dummy_idx[side] is not None:
            offset = frame @ solved[self.dummy_idx[side]]
            norm = float(np.linalg.norm(offset))
            if norm > MAX_WRIST_OFFSET:
                # A degenerate solve would otherwise teleport the hand. Clamp rather than trust it.
                offset = offset * (MAX_WRIST_OFFSET / norm)
        elif self.wrist_fit:
            offset, rotation = self.fit_wrist(side, solved, local, frame, env_idx)

        return solved[self.dof_perm[side]], offset, rotation

    def resolve_fit_weights(self):
        """Per-point weights for the fit, aligned to `self.fit_pairs`.

        Two forms, because the two point sets want different things. A plain list is positional
        over the five fingertips. A dict is matched by SUBSTRING against each robot link name, so
        one entry covers a whole joint class across all five fingers — `{"tip": 1, "proximal":
        0.2}` down-weights every knuckle at once, which is the point when there are 16 of them.

        Longest matching key wins so a specific name beats a general one ("thumb_tip" over "tip"),
        and anything unmatched sits at 1.0.

        Returns:
            (n_points,) float array, or None for uniform.
        """
        weights = DEXRET_FIT_WEIGHTS
        if weights is None:
            return None

        if isinstance(weights, dict):
            resolved = []
            for link, _ in self.fit_pairs:
                keys = [k for k in weights if k in link]
                resolved.append(weights[max(keys, key=len)] if keys else 1.0)
            resolved = np.array(resolved, dtype=np.float64)
            assert (resolved > 0).sum() >= 3 and resolved.sum() > 0, (
                f"fit weights leave only {(resolved > 0).sum()} points with non-zero weight; "
                f"Kabsch needs at least 3 to determine a rotation. Weights: {DEXRET_FIT_WEIGHTS}"
            )
            return resolved

        # Positional: only meaningful when it lines up with the active point set.
        if len(weights) != len(self.fit_pairs):
            print(
                f"\033[1;93m[dexret] DEXRET_FIT_WEIGHTS has {len(weights)} entries but "
                f"DEXRET_FIT_POINTS={DEXRET_FIT_POINTS!r} matches {len(self.fit_pairs)} points; "
                f"weighting ignored (uniform). Use the named form to weight by joint class."
                f"\033[0m"
            )
            return None
        return np.array(weights, dtype=np.float64)

    def fk_points(self, side, solved):
        """Robot match points in the hand's BASE frame, from forward kinematics on a solved pose.

        The single place FK is run for the fit, so the fit and the error it reports can never
        disagree about which frame the robot's points are in — they did once, and the symptom was
        a "fit" that appeared to make fingertip error eight times worse.

        With a free joint being overridden, `solved` carries the optimiser's chosen hand pose in
        its dummy DOFs; zeroing them is what puts the fingers back in the base frame, which is the
        frame the human keypoints are expressed in.

        Args:
            side: "rh" or "lh".
            solved: (robot.dof,) full solver output.

        Returns:
            (n_points, 3) positions of `self.fit_pairs`' robot links, base frame.
        """
        if self.dummy_all_idx[side] is not None:
            solved = np.asarray(solved).copy()
            solved[self.dummy_all_idx[side]] = 0.0
        robot = self.solvers[side].optimizer.robot
        robot.compute_forward_kinematics(solved)
        return np.stack([robot.get_link_pose(i)[:3, 3] for i in self.tip_link_idx[side]])

    def local_fit(self, side, solved, local):
        """The hand-local rigid correction seating the solved hand's fingertips on the human's.

        Runs forward kinematics on the pose just solved, then solves the rigid transform that best
        seats those five tips on the human's. Because the finger angles are already fixed by this
        point, the placement is a closed-form Procrustes problem — no optimiser, no gains.

        Both point sets live in the optimiser's hand-local frame, where the robot's base is at the
        origin unrotated and the human's wrist is at the origin as well, so an already-perfect
        placement returns (identity, 0) and the caller reduces to the previous behaviour exactly.

        Args:
            side: "rh" or "lh".
            solved: (robot.dof,) full solver output, mimic joints already expanded by
                SeqRetargeting.retarget — which is what makes it valid to feed straight to FK.
            local: (21, 3) human keypoints in the optimiser's frame, wrist-centred.

        Returns:
            ((3, 3) rotation, (3,) translation, dict diagnostics), all in the hand-local frame.
        """
        robot_tips = self.fk_points(side, solved)

        return fit_wrist_to_fingertips(
            robot_tips,
            local[self.fit_slots],
            max_translation=DEXRET_FIT_MAX_TRANSLATION,
            max_angle_rad=np.radians(DEXRET_FIT_MAX_ANGLE_DEG),
            weights=self.fit_weights,
        )

    def calibrate_constant(self, side, env_idx):
        """Solve the whole calibration window once and collapse it to a single hand-local offset.

        Re-solving every frame adapts as the grasp closes, but gives no smoothness guarantee — and
        the wrist is force-controlled, so a jump in the correction is a spike in the commanded
        force. Averaging the fit over the motion trades that adaptivity for a correction that
        cannot jump at all, because it never changes.

        Strides through the window rather than solving every frame: the result is an average, so
        more samples stop helping long before the cost stops growing.

        The solver is warm-started from its previous answer, so this pass would otherwise leave it
        primed on the last calibration frame instead of the demo's start. `reset()` puts it back.

        Args:
            side: "rh" or "lh".
            env_idx: Which env row to calibrate against — each row may hold a different demo, so
                the constant is per env, not shared.

        Returns:
            ((3, 3) rotation, (3,) translation) in the hand-local frame.
        """
        hand = "right" if side == "rh" else "left"
        demo = getattr(self.env, f"demo_data_{side}")
        solver = self.solvers[side]

        total = demo["mano_joints"].shape[1]
        window = min(total, DEXRET_FIT_CALIB_FRAMES) if DEXRET_FIT_CALIB_FRAMES else total
        frames = self.calibration_frames(side, env_idx, window)

        rotations, translations, world_xyz, clamped = [], [], [], 0
        for t in frames:
            kp21 = self.mano21_from_env(side, env_idx, t)
            frame = hand_local_transform(
                hand, demo["wrist_rot"][env_idx, t].cpu().numpy(), self.loader_to_avp[side]
            ) @ self.base_align[side]
            local = (kp21 - kp21[0:1, :]) @ frame
            solved = solver.retarget(retarget_ref_value(solver, local))
            rotation, translation, stats = self.local_fit(side, solved, local)
            rotations.append(rotation)
            translations.append(translation)
            clamped += int(stats["clamped"])
            world_xyz.append(frame @ translation)

        solver.reset()
        rotation, translation = average_rigid(rotations, translations)
        # The clamp rate on the CALIBRATION frames, which is the only place clamping can affect a
        # constant fit -- the constant itself is applied unclamped thereafter. Reported because a
        # constant that sits at its limit is a truncated answer, not a converged one, and nothing
        # else in the output distinguishes the two.
        angle_deg = np.degrees(np.linalg.norm(rotmat_to_rotvec(rotation)))
        at_limit = (
            angle_deg > 0.98 * DEXRET_FIT_MAX_ANGLE_DEG
            or 1e3 * np.linalg.norm(translation) > 0.98 * 1e3 * DEXRET_FIT_MAX_TRANSLATION
        )
        self.fit_stats["calib_frames"] += len(rotations)
        self.fit_stats["calib_clamped"] += clamped
        print(
            f"[dexret] {side} constant fit from {len(rotations)} samples over frames "
            f"{frames[0]}-{frames[-1]} of {window}: "
            f"|t| = {1e3 * np.linalg.norm(translation):.1f} mm, "
            f"angle = {angle_deg:.1f} deg, "
            f"{100 * clamped / max(1, len(rotations)):.0f}% of samples clamped"
            + (f", world-z {1e3 * np.median(np.array(world_xyz), axis=0)[2]:+.1f} mm "
               f"(spread {1e3 * (np.percentile(np.array(world_xyz)[:,2], 90) - np.percentile(np.array(world_xyz)[:,2], 10)):.1f} mm"
               f"{f' — frozen:{DEXRET_FIT_WORLD_FREEZE}'})" if world_xyz else "")
            + ("  \033[1;93m<-- AT THE LIMIT: raise DEXRET_FIT_MAX_* and re-measure\033[0m"
               if at_limit else "")
        )
        # Median for the same reason the translation uses one: a degenerate frame should not move it.
        self.fit_world_frozen[side][env_idx] = np.median(np.array(world_xyz), axis=0)
        return rotation, translation

    def calibration_frames(self, side, env_idx, window):
        """Which frames of the demo to calibrate the constant on.

        Most of a demo is reach and retreat, where the hand sits ~150 mm from the object and its
        exact placement is irrelevant. The grasp is the only phase where the placement matters, and
        it is a minority of the frames — so averaging over everything lets the part nobody cares
        about dominate the constant. This ranks frames by how close THIS hand's pinch pair (thumb
        and index) is to THIS hand's object surface and keeps the nearest fraction.

        Per hand, deliberately: `tips_distance` is measured against each hand's own object, so the
        RH window lands on the cap pinch and the LH window on the bottle-body grasp, which are not
        the same frames.

        Falls back to striding the whole window if `tips_distance` is missing — it is built by the
        dataset loader, so a source that does not provide it should degrade rather than crash.

        Args:
            side: "rh" or "lh".
            env_idx: Which env row to read the demo from.
            window: Number of leading frames eligible for selection.

        Returns:
            list[int] frame indices in ASCENDING order, at most DEXRET_FIT_CALIB_SAMPLES of them.
            Ascending because the solver is warm-started from its previous answer; walking the
            timeline backwards or at random would seed every solve from an unrelated pose.
        """
        demo = getattr(self.env, f"demo_data_{side}")
        tips = demo.get("tips_distance") if hasattr(demo, "get") else None
        fraction = DEXRET_FIT_CALIB_PINCH_FRAC[side]

        if tips is None or fraction >= 1.0:
            if tips is None and fraction < 1.0:
                print(
                    f"\033[1;93m[dexret] {side}: no tips_distance in the demo buffer, so the "
                    f"calibration cannot be narrowed to the pinch; using the whole demo.\033[0m"
                )
            stride = max(1, window // DEXRET_FIT_CALIB_SAMPLES)
            return list(range(0, window, stride))

        # tips_distance is [T, 5] in (thumb, index, middle, ring, pinky) order — the same order as
        # FIT_TIP_LINKS — and holds the nearest distance from each fingertip to the object SURFACE.
        pinch = tips[env_idx, :window, :2].mean(dim=-1)
        keep = max(3, int(round(window * fraction)))
        selected = torch.sort(torch.argsort(pinch)[:keep]).values

        stride = max(1, len(selected) // DEXRET_FIT_CALIB_SAMPLES)
        return [int(t) for t in selected[::stride]]

    def load_live_calibration(self):
        """Adopt a stored calibration, or announce that one has to be captured.

        Called once at construction, for live runs in constant mode only. Offline the demo itself
        is the calibration source and no file is involved.

        Returns:
            bool True if a usable calibration was loaded.
        """
        stored = load_calibration(
            self.calib_path, self.robot, self.retargeting, DEXRET_SCALING_FACTOR
        )
        if stored is None:
            return False

        # One calibration describes the operator's hand, so it applies to every env row.
        for side, (rotation, translation) in stored.items():
            for env_idx in range(self.env.num_envs):
                self.fit_constant[side][env_idx] = (rotation, translation)
            print(
                f"\033[96m[dexret] {side} calibration loaded from {self.calib_path}: "
                f"|t| = {1e3 * np.linalg.norm(translation):.1f} mm, angle = "
                f"{np.degrees(np.linalg.norm(rotmat_to_rotvec(rotation))):.1f} deg\033[0m"
            )
        return True

    def record_calibration_sample(self, side, env_idx, rotation, translation):
        """Accumulate one frame toward the live calibration, and finish it when there are enough.

        Only env 0 is sampled: the calibration measures the operator, and every env row is fed the
        same live stream, so the other rows would contribute copies rather than information.

        Args:
            side: "rh" or "lh".
            env_idx: Which env row produced this sample.
            rotation: (3, 3) this frame's hand-local fitted rotation.
            translation: (3,) this frame's hand-local fitted translation.
        """
        if env_idx != 0:
            return

        samples = self.fit_samples[side].setdefault(0, ([], []))
        samples[0].append(rotation)
        samples[1].append(translation)

        target = DEXRET_FIT_CALIB_FRAMES or LIVE_CALIB_FRAMES
        collected = len(samples[0])
        if collected % max(1, target // 4) == 0 and collected < target:
            print(f"[dexret calibrating] {side} {collected}/{target} frames")
        if collected < target:
            return

        rotation, translation = average_rigid(*samples)
        for idx in range(self.env.num_envs):
            self.fit_constant[side][idx] = (rotation, translation)
        self.fit_samples[side].pop(0)
        print(
            f"\033[92m[dexret calibrating] {side} done over {collected} frames: "
            f"|t| = {1e3 * np.linalg.norm(translation):.1f} mm, angle = "
            f"{np.degrees(np.linalg.norm(rotmat_to_rotvec(rotation))):.1f} deg\033[0m"
        )

        if self.calibration_complete():
            self.save_live_calibration(collected)

    def calibration_complete(self):
        """Whether every hand now has a frozen constant. Polled by the driver loop to know when to
        stop a calibration run.

        Returns:
            bool True once both hands are calibrated.
        """
        return all(0 in self.fit_constant[side] for side in ("rh", "lh"))

    def save_live_calibration(self, samples):
        """Write the captured constants so later live runs load them instead of re-measuring.

        Args:
            samples: how many frames each side was averaged over, recorded for provenance.
        """
        path = save_calibration(
            self.calib_path, self.robot, self.retargeting, DEXRET_SCALING_FACTOR,
            {
                side: (*self.fit_constant[side][0], samples)
                for side in ("rh", "lh")
            },
        )
        print(
            f"\033[1;92m[dexret] calibration written to {path}. Re-run without "
            f"dexRetCalibrate=true and live teleop will use it from the first frame.\033[0m"
        )

    def fit_wrist(self, side, solved, local, frame, env_idx):
        """World-frame wrist offset and rotation from the fingertip fit.

        Args:
            side: "rh" or "lh".
            solved: (robot.dof,) full solver output for this frame.
            local: (21, 3) human keypoints in the optimiser's frame, wrist-centred.
            frame: (3, 3) from hand_local_transform; maps a local column vector to world.
            env_idx: Which env row this solve is for.

        Returns:
            ((3,) world-frame wrist offset, (3, 3) world-frame wrist rotation for the sim).
        """
        # Where the constant comes from differs by mode, and only one of the three paths pays for a
        # full Kabsch solve per frame:
        #   offline + constant — pre-pass the whole demo once, here, on first use.
        #   live + constant    — loaded from the calibration file at construction, or being
        #                        captured right now by an explicit calibration run.
        #   per_frame          — no constant; solved fresh below.
        constant = self.fit_constant[side].get(env_idx)
        if constant is None and self.fit_mode == "constant" and not self.env.live:
            # Timed apart from the per-frame cost on purpose: folding a ~0.8 s pre-pass into the
            # per-frame average reports a figure several times the real one, and live never pays it
            # at all. The steady-state number is what the control budget actually cares about.
            calib_started = time.perf_counter()
            constant = self.calibrate_constant(side, env_idx)
            self.fit_stats["calib_seconds"] += time.perf_counter() - calib_started
            self.fit_constant[side][env_idx] = constant

        started = time.perf_counter()
        if constant is not None:
            rotation, translation = constant
            # A constant correction still has a per-frame error, and the only honest way to say
            # what holding it fixed costs is to measure it on the same tips the per-frame fit is
            # measured on. FK plus two weighted norms, no SVD — ~0.04 ms, so scoring it is free.
            tips = self.fk_points(side, solved)
            human = local[self.fit_slots]
            stats = {
                "clamped": False,
                "rms_before": tip_rms(tips, human, weights=self.fit_weights),
                "rms_after": tip_rms(tips, human, rotation, translation, self.fit_weights),
            }
        else:
            rotation, translation, stats = self.local_fit(side, solved, local)
            if self.calibrating:
                # Live capture: `_inject_live` overwrites only the CURRENT frame, so there is no
                # demo to pre-pass — the constant has to be built from the operator's own frames as
                # they arrive. This is the only window where the full fit runs live.
                self.record_calibration_sample(side, env_idx, rotation, translation)

        self.fit_stats["seconds"] += time.perf_counter() - started
        self.fit_stats["frames"] += 1
        self.fit_stats["clamped"] += int(stats["clamped"])
        self.fit_stats["rms_before"] += stats["rms_before"]
        self.fit_stats["rms_after"] += stats["rms_after"]

        world_translation = frame @ translation
        frozen = self.fit_world_frozen[side].get(env_idx)
        if frozen is not None and DEXRET_FIT_WORLD_FREEZE != "none":
            # Hold the correction still in world rather than letting it rotate with the hand, so the
            # wrist's own rise is not eaten by the correction swinging. "z" keeps x/y rotating,
            # which is the grasp centring the fit exists to provide.
            if DEXRET_FIT_WORLD_FREEZE == "xyz":
                world_translation = frozen.copy()
            else:
                world_translation = np.array(
                    [world_translation[0], world_translation[1], frozen[2]]
                )
        world_rotation = frame @ rotation @ self.solver_to_sim[side]

        # Frame-to-frame movement of the correction ITSELF, in world. Measured against the previous
        # frame's correction for the same hand, so the hand's own motion is not counted: what is
        # left is how much the fit alone jumps. Env 0 only -- this runs inside the per-env loop and
        # the point is a time series, not a population.

        previous = self.prev_fit[side]
        if previous is not None:
            delta = float(np.linalg.norm(world_translation - previous[0]))
            self.fit_stats["steps"] += 1
            self.fit_stats["d_trans"] += delta
            if delta > self.fit_stats["d_trans_max"]:
                self.fit_stats["d_trans_max"] = delta
                # WHERE the worst jump happens decides whether it matters. progress_buf == 0 is an
                # episode reset: prev_fit still holds the last frame of the PREVIOUS episode, so
                # the difference is a demo discontinuity, not the fit moving.
                self.fit_stats["d_trans_max_at"] = (side, int(self.env.progress_buf[env_idx]))
            self.fit_stats["d_rot_deg"] += np.degrees(
                np.linalg.norm(rotmat_to_rotvec(world_rotation @ previous[1].T))
            )
        self.prev_fit[side] = (world_translation, world_rotation)

        return world_translation, world_rotation

    def wrist_error(self, side, env_idx, step_idx, target_pos, fit_rotation=None):
        """Wrist pose error the env's PID branch consumes: 3 position + 6 rotation dims.

        The PID branch is given an ERROR, not a target (`pre_physics_step`), so this differences
        the human target against the current sim wrist every step.

        Args:
            side: "rh" or "lh".
            env_idx: Which env row to read.
            step_idx: Index into the demo/live buffer.
            target_pos: (3,) wrist position target, with the pullback and any solved free-joint
                offset already applied by compute_action.
            fit_rotation: (3, 3) wrist orientation from the fingertip fit, or None to track the
                human's wrist orientation as before.

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
        target_rotmat = (
            fit_rotation if fit_rotation is not None
            else aa_to_rotmat(demo["wrist_rot"][env_idx, step_idx][None])[0]
        )
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

    def wrist_pd_ff(self, side, target_pos, target_velocity, fit_rotations=None):
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
            fit_rotations: (num_envs, 3, 3) wrist orientations from the fingertip fit, or None to
                track the human's wrist orientation as before.

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
        target_rotmat = (
            fit_rotations if fit_rotations is not None
            else aa_to_rotmat(demo["wrist_rot"][rows, step_idx])
        )
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
            fit_rotations = torch.zeros((n_env, 3, 3), device=env.device) if self.wrist_fit else None
            for env_idx in range(n_env):
                t = int(step_idx[env_idx])
                # Same buffer the wrist target reads, so the finger frame and the wrist agree —
                # and _inject_live keeps both current in live mode.
                wrist_rot_aa = demo["wrist_rot"][env_idx, t].cpu().numpy()
                qpos, wrist_offset, wrist_rotation = self.solve_dofs(
                    side, self.mano21_from_env(side, env_idx, t), wrist_rot_aa, env_idx
                )
                qpos = torch.as_tensor(qpos, device=env.device, dtype=torch.float32)
                if wrist_offset is not None:
                    solve_offsets[env_idx] = torch.as_tensor(
                        wrist_offset, device=env.device, dtype=torch.float32
                    )
                if wrist_rotation is not None:
                    fit_rotations[env_idx] = torch.as_tensor(
                        wrist_rotation, device=env.device, dtype=torch.float32
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
            if self.dummy_idx[side] is not None or self.wrist_fit:
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
                    side, target_pos, target_velocity, fit_rotations
                )
            else:
                for env_idx in range(n_env):
                    base[env_idx, offset : offset + root_dim] = torch.as_tensor(
                        self.wrist_error(
                            side, env_idx, int(step_idx[env_idx]), target_pos[env_idx],
                            None if fit_rotations is None else fit_rotations[env_idx],
                        ),
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
