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

from baselines.avp_dex_retarget import (
    DEFAULT_RETARGETING,
    MANO21_JOINT_NAMES,
    OPERATOR2AVP,
    default_config_path,
    dex_urdf_dir,
    retarget_ref_value,
)

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")

# MANO-21 slots dex-retargeting reads: the wrist (which the keypoints are centred on) plus the five
# fingertips. The rest are never touched — four of them (*_distal) do not exist in the env's packed
# buffer at all. The frame comes from the wrist alone, so the MCPs the 3-point fit used are gone.
REQUIRED_MANO_SLOTS = (0, 4, 8, 12, 16, 20)


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
    """

    def __init__(self, env, robot="inspire", retargeting=DEFAULT_RETARGETING):
        from dex_retargeting.retargeting_config import RetargetingConfig

        self.env = env
        assert env.use_pid_control, (
            "DexRetargetController drives the wrist through the env's PID branch, which only "
            "exists when usePIDControl=true. Re-run with usePIDControl=true."
        )

        # dex-retargeting resolves the config's relative `urdf_path` against this.
        RetargetingConfig.set_default_urdf_dir(dex_urdf_dir())

        self.solvers, self.dof_perm, self.rows, self.slot_names = {}, {}, {}, mano_slot_to_hand_name()
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
            self.solvers[side] = solver
            self.rows[side] = packed_row_by_hand_name(dexhand)
            self.loader_to_avp[side] = loader_to_avp_rotation(dexhand)

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
            (n_dofs,) joint angles in radians, ordered to match `dexhand.dof_names`.
        """
        hand = "right" if side == "rh" else "left"
        centred = kp21 - kp21[0:1, :]
        local = centred @ hand_local_transform(hand, wrist_rot_aa, self.loader_to_avp[side])

        solver = self.solvers[side]
        return solver.retarget(retarget_ref_value(solver, local))[self.dof_perm[side]]

    def wrist_error(self, side, env_idx, step_idx):
        """Wrist pose error the env's PID branch consumes: 3 position + 6 rotation dims.

        The PID branch is given an ERROR, not a target (`pre_physics_step`), so this differences
        the human target against the current sim wrist every step.

        Args:
            side: "rh" or "lh".
            env_idx: Which env row to read.
            step_idx: Index into the demo/live buffer.

        Returns:
            (9,) float32 array: world-frame position error then the 6D form of the rotation error.
        """
        demo = getattr(self.env, f"demo_data_{side}")
        dexhand = getattr(self.env, f"dexhand_{side}")
        wrist_body = getattr(self.env, f"dexhand_{side}_handles")[dexhand.to_dex("wrist")[0]]
        current = self.env._rigid_body_state[env_idx, wrist_body]

        pos_error = demo["wrist_pos"][env_idx, step_idx] - current[:3]

        # demo wrist_rot is axis-angle; the sim quaternion is Isaac Gym's xyzw while
        # quat_to_rotmat wants wxyz (verified empirically), hence the [3, 0, 1, 2] reindex.
        target_rotmat = aa_to_rotmat(demo["wrist_rot"][env_idx, step_idx][None])[0]
        current_rotmat = quat_to_rotmat(current[3:7][[3, 0, 1, 2]][None])[0]
        rot_error = rotmat_to_rot6d(target_rotmat @ current_rotmat.transpose(-1, -2))

        return torch.cat([pos_error, rot_error.reshape(-1)]).float().cpu().numpy()

    def compute_action(self):
        """Build the full action tensor for this control step.

        Returns:
            (num_envs, action_dim) float32 tensor in [-1, 1] — the base half filled from the
            retargeting solve, the residual half zero.
        """
        from maniptrans_envs.lib.utils import torch_jit_utils

        env = self.env
        n_env = env.num_envs
        root_dim = 9  # 3 position error + 6D rotation; asserted PID in __init__
        n_dofs = env.num_dexhand_rh_dofs
        per_hand = root_dim + n_dofs

        base = torch.zeros((n_env, 2 * per_hand), device=env.device)
        step_idx = torch.clamp(env.progress_buf, 0, getattr(env, "demo_data_rh")["mano_joints"].shape[1] - 1)

        for hand_i, side in enumerate(("rh", "lh")):
            lower = getattr(env, f"dexhand_{side}_dof_lower_limits")
            upper = getattr(env, f"dexhand_{side}_dof_upper_limits")
            offset = hand_i * per_hand
            demo = getattr(env, f"demo_data_{side}")
            for env_idx in range(n_env):
                t = int(step_idx[env_idx])
                # Same buffer wrist_error reads, so the finger frame and the wrist target agree —
                # and _inject_live keeps both current in live mode.
                wrist_rot_aa = demo["wrist_rot"][env_idx, t].cpu().numpy()
                qpos = self.solve_dofs(side, self.mano21_from_env(side, env_idx, t), wrist_rot_aa)
                qpos = torch.as_tensor(qpos, device=env.device, dtype=torch.float32)
                # exact inverse of the env's scale() at pre_physics_step, so the commanded angle
                # survives the [-1,1] action interface unchanged
                base[env_idx, offset : offset + root_dim] = torch.as_tensor(
                    self.wrist_error(side, env_idx, t), device=env.device
                )
                base[env_idx, offset + root_dim : offset + per_hand] = torch_jit_utils.unscale(
                    qpos, lower, upper
                )

        residual = torch.zeros((n_env, env.num_actions), device=env.device)
        return torch.clamp(torch.cat([base, residual], dim=1), -1.0, 1.0)
