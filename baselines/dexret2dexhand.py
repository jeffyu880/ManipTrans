"""Retarget a MyDataset capture with dex-retargeting and write a ManipTrans retargeted pkl.

Sibling of `main/dataset/mano2dexhand.py`: same output format, same output path, different
solver. Where mano2dexhand optimises the hand pose inside an Isaac Gym sim against the MANO
joints, this runs dex-retargeting's per-frame solve. Swap which one you run and everything
downstream — `playback_trajectory.py`, the loaders, reset init — is unchanged.

    python baselines/dexret2dexhand.py --data_idx m_101123 --side right
    python data_stats/playback_trajectory.py --data_idx m_101123 --side right --record

Unlike `avp_dex_retarget.py` (which reads a capture pkl directly and is frame-agnostic), this
goes through the dataset loader, so the frame chain — recentre, table rotation,
`mujoco2gym_transf` — is applied by exactly the code the env and playback use. That matters
because `opt_wrist_pos`/`opt_wrist_rot` must land in the Isaac Gym frame to line up with the
loader's own `wrist_pos`; getting that wrong puts the hand in the wrong place while the fingers
still look correct.

--- What comes from where ---
    opt_dof_pos    dex-retargeting's solve on the loader's `mano_joints`, in the hand-local frame
                   built from the AVP wrist (see dexret_controller.hand_local_transform). Because
                   that frame comes from `wrist_rot`, both it and the joints go through the same
                   transform chain and the chain cancels — but only as a pair, so feeding joints
                   from one frame and a wrist from another would silently rotate the hand.
    opt_wrist_pos  straight from the loader's `wrist_pos` — dex-retargeting discards global
                   pose, so the wrist can only come from the human.
    opt_wrist_rot  likewise, the loader's `wrist_rot` (axis-angle).
    opt_joints_pos NOT written. `base.py` only reads it for an ablation and has it commented
                   out; computing it would need FK we do not otherwise need.
"""

import argparse
import os
import pickle

# pinocchio (which dex-retargeting solves through) binds its C++ types at import, and isaacgym's
# own libs shadow the symbols it binds against. Loaded after isaacgym it still imports, but the
# solver build dies with a bare "No Python class registered for C++ class std::vector<std::string>".
# It does not pull in torch, so importing it here breaks neither ordering rule.
import pinocchio  # noqa: F401  MUST precede isaacgym
import isaacgym  # noqa: F401  must precede torch (see CLAUDE.md)
import numpy as np
import torch

from data_stats.playback_trajectory import env_mujoco2gym_transf
from main.dataset.factory import ManipDataFactory
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory

from baselines.avp_dex_retarget import (
    DEFAULT_RETARGETING,
    RETARGETING_TYPES,
    default_config_path,
    dex_urdf_dir,
    retarget_ref_value,
)

# The frame constants and logic live in dexret_controller: the in-env controller and this script
# consume the same loader-frame wrist, so they have one owner rather than two copies.
from baselines.dexret_controller import (
    REQUIRED_MANO_SLOTS,
    hand_local_transform,
    loader_to_avp_rotation,
    mano_slot_to_hand_name,
    solver_dof_perm,
)


def mano21_from_loader(data, step, slot_names):
    """Build the MANO-21 keypoint array for one frame out of the loader's data dict.

    Only the eight slots dex-retargeting reads are filled; the rest stay zero and are never
    indexed. The loader keeps `mano_joints` as a dict keyed by ManipTrans joint name, so this is
    a straight lookup — no packed-row arithmetic, unlike the in-env controller.

    Args:
        data: One sequence from the loader, already in the Isaac Gym frame.
        step: Frame index.
        slot_names: {MANO slot: ManipTrans joint name}, from mano_slot_to_hand_name().

    Returns:
        (21, 3) float64 world-frame keypoints.
    """
    kp = np.zeros((21, 3), dtype=np.float64)
    kp[0] = data["wrist_pos"][step].cpu().numpy()
    for slot in REQUIRED_MANO_SLOTS:
        if slot == 0:
            continue
        kp[slot] = data["mano_joints"][slot_names[slot]][step].cpu().numpy()
    return kp


def main():
    """Solve a whole sequence with dex-retargeting and dump it as a ManipTrans retargeted pkl.

    Returns:
        None.
    """
    from dex_retargeting.retargeting_config import RetargetingConfig

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data_idx", required=True, help="e.g. m_101123")
    parser.add_argument("--side", default="right", choices=["right", "left"])
    parser.add_argument("--dexhand", default="inspire")
    parser.add_argument(
        "--retargeting",
        default=DEFAULT_RETARGETING,
        choices=list(RETARGETING_TYPES),
        help="dexpilot = inter-finger vectors plus DexPilot's thumb-to-finger grasp projection "
        "(Bunny-VisionPro's choice); vector = wrist-to-fingertip vectors only, no projection",
    )
    parser.add_argument(
        "--wrist-pullback",
        type=float,
        default=0.35,
        help="pull the retargeted wrist back toward the forearm by this fraction of the "
        "wrist-to-middle-MCP span — the same hack oakink2/grab apply in their loaders, which use "
        "0.25; 0.35 here, tuned up on the AVP captures where the hand crowded the object (~33 mm). "
        "0 disables. NOTE this shifts opt_wrist_pos only, so it moves reset init and playback, "
        "NOT the demo tracking targets",
    )
    parser.add_argument("--out", default=None, help="override the output pkl path")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    side = args.side
    slot_names = mano_slot_to_hand_name()
    mujoco2gym = env_mujoco2gym_transf(args.device)

    dexhand = DexHandFactory.create_hand(args.dexhand, side)
    dataset = ManipDataFactory.create_data(
        manipdata_type=ManipDataFactory.dataset_type(args.data_idx),
        side=side, device=args.device,
        mujoco2gym_transf=mujoco2gym, dexhand=dexhand, verbose=False,
    )
    data = dataset[args.data_idx]

    RetargetingConfig.set_default_urdf_dir(dex_urdf_dir())
    config_path = default_config_path(args.dexhand, side, args.retargeting)
    assert os.path.exists(config_path), (
        f"no retargeting config at {config_path}. Only inspire ships one; add a config for "
        f"'{args.dexhand}' modelled on the inspire ones."
    )
    solver = RetargetingConfig.load_from_file(config_path).build()

    perm = solver_dof_perm(solver, dexhand)
    loader_to_avp = loader_to_avp_rotation(dexhand)

    n_frames = len(data["wrist_pos"])
    opt_dof_pos = np.zeros((n_frames, len(dexhand.dof_names)), dtype=np.float32)
    for step in range(n_frames):
        kp = mano21_from_loader(data, step, slot_names)
        centred = kp - kp[0:1, :]
        local = centred @ hand_local_transform(
            side, data["wrist_rot"][step].cpu().numpy(), loader_to_avp
        )
        opt_dof_pos[step] = solver.retarget(retarget_ref_value(solver, local))[perm]

    # Pull the wrist back along its palm axis (wrist -> middle MCP). The loaders leave
    # WRIST_PULLBACK at 0 because it would move the demo tracking targets and so every reward;
    # here it touches only what this pkl feeds — reset init and playback. Equivariant under the
    # loader's rigid transforms (and relative_translation is zero), so applying it post-transform
    # matches what the loaders would have produced.
    opt_wrist_pos = data["wrist_pos"].cpu().numpy()
    if args.wrist_pullback:
        middle_pos = data["mano_joints"]["middle_proximal"].cpu().numpy()
        opt_wrist_pos = opt_wrist_pos - (middle_pos - opt_wrist_pos) * args.wrist_pullback

    stem = os.path.splitext(os.path.basename(data["data_path"]))[0]
    suffix = "rh" if side == "right" else "lh"
    out = args.out or os.path.join(
        "data", "retargeting", "my_dataset", f"mano2{dexhand}", f"{stem}_{suffix}.pkl"
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(
            {
                "opt_wrist_pos": opt_wrist_pos,
                "opt_wrist_rot": data["wrist_rot"].cpu().numpy(),
                "opt_dof_pos": opt_dof_pos,
                "wrist_pullback": args.wrist_pullback,
                # the loader subsamples by `skip`, so stamp the rate these frames are
                # actually at — base.py re-subsamples against it on load
                "retarget_fps": dataset.fps / dataset.skip,
                "retarget_source": "dex-retargeting",
            },
            f,
        )
    print(f"[{side}] {n_frames} frames, {opt_dof_pos.shape[1]} dof -> {out}")


if __name__ == "__main__":
    main()
