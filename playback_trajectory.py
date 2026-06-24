"""Animated playback of a capture's hand + object trajectory.

Unlike view_retarget_frames.py (static, one frame per env), this plays the whole
sequence over time in a single env. For a bimanual capture it shows BOTH hands and
BOTH objects together, in the table frame the env uses -- handy for sanity-checking
the captured trajectory and the relative placement of the two hands/objects.

It teleports the dexhand(s) onto the retargeted poses (opt_wrist_pos/rot/dof) and the
object(s) onto obj_trajectory each frame -- it does NOT simulate dynamics or optimize,
and writes nothing. If a side has no retargeting pkl yet, load_retargeted_data falls
back to the raw wrist pose + zero finger dofs (so you still see the wrist+object path).

Examples:
    python playback_trajectory.py --data_idx m_164621 --side both          # both hands
    python playback_trajectory.py --data_idx m_164621 --side left --speed 0.5
    python playback_trajectory.py --data_idx m_164621 --start 0 --end 9     # just 0..9

Controls: --speed (playback rate vs realtime), --start/--end (frame window),
--no_loop (play once and hold on the last frame).
"""

import os
import time
from isaacgym import gymapi, gymtorch, gymutil  # before torch
import numpy as np
import torch
from termcolor import cprint

from main.dataset.factory import ManipDataFactory
from main.dataset.transform import aa_to_quat, aa_to_rotmat, rotmat_to_quat
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory


def env_mujoco2gym_transf(device):
    """Same table transform the BiH env builds (dexhandmanip_bih.py:281-286); the saved
    opt_* are in this frame, so obj_trajectory must be transformed by it to match."""
    table_surface_z = 0.4 + 0.015
    m = np.eye(4)
    m[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(np.array([np.pi / 2, 0, 0]))
    m[:3, 3] = np.array([0, 0, table_surface_z])
    return torch.tensor(m, device=device, dtype=torch.float32)


def hand_root13(data, device):
    """[T,13] root state (pos, quat xyzw, zero vel) from retargeted wrist."""
    pos = data["opt_wrist_pos"].to(device).float()                       # [T,3]
    quat = aa_to_quat(data["opt_wrist_rot"].to(device).float())[:, [1, 2, 3, 0]]  # [T,4] xyzw
    return torch.cat([pos, quat, torch.zeros((pos.shape[0], 6), device=device)], dim=1)


def obj_root13(data, device):
    traj = data["obj_trajectory"].to(device).float()                     # [T,4,4]
    pos = traj[:, :3, 3]
    quat = rotmat_to_quat(traj[:, :3, :3])[:, [1, 2, 3, 0]]               # xyzw
    return torch.cat([pos, quat, torch.zeros((pos.shape[0], 6), device=device)], dim=1)


def main():
    args = gymutil.parse_arguments(
        description="Playback of hand + object trajectory",
        headless=True,
        custom_parameters=[
            {"name": "--data_idx", "type": str, "default": "m_164621"},
            {"name": "--dexhand", "type": str, "default": "inspire"},
            {"name": "--side", "type": str, "default": "both", "help": "left | right | both"},
            {"name": "--speed", "type": float, "default": 1.0, "help": "playback rate vs realtime"},
            {"name": "--start", "type": int, "default": 0},
            {"name": "--end", "type": int, "default": -1, "help": "last frame (inclusive); -1 = end"},
            {"name": "--no_loop", "action": "store_true"},
        ],
    )

    device = "cuda:0"
    sides = ["right", "left"] if args.side == "both" else [args.side]
    m2g = env_mujoco2gym_transf(device)

    # ---- load data per side ----
    hands = []  # list of dicts: side, dexhand, root13[T,13], obj13[T,13], dof[T,n], obj_urdf
    T = None
    for side in sides:
        dexhand = DexHandFactory.create_hand(args.dexhand, side)
        dtype = ManipDataFactory.dataset_type(args.data_idx)
        demo = ManipDataFactory.create_data(
            manipdata_type=dtype, side=side, device=device,
            mujoco2gym_transf=m2g, dexhand=dexhand, verbose=True,
        )
        data = demo[args.data_idx]
        h = {
            "side": side,
            "dexhand": dexhand,
            "root13": hand_root13(data, device),
            "obj13": obj_root13(data, device),
            "dof": data["opt_dof_pos"].to(device).float(),
            "obj_urdf": data["obj_urdf_path"],
        }
        hands.append(h)
        T = h["dof"].shape[0] if T is None else min(T, h["dof"].shape[0])
    cprint(f"Loaded {args.data_idx} sides={sides}: {T} frames", "cyan")

    start = max(0, args.start)
    end = T - 1 if args.end < 0 else min(args.end, T - 1)
    assert start <= end, f"bad frame window [{start},{end}]"

    # ---- sim ----
    gym = gymapi.acquire_gym()
    sp = gymapi.SimParams()
    sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0, 0, 0)  # pure kinematic playback
    sp.substeps = 1
    sp.physx.solver_type = 1
    sp.physx.num_position_iterations = 4
    sp.physx.num_velocity_iterations = 1
    sp.physx.num_threads = args.num_threads
    sp.physx.use_gpu = args.use_gpu
    sp.use_gpu_pipeline = args.use_gpu_pipeline
    sim_device = args.sim_device if args.use_gpu_pipeline else "cpu"
    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sp)

    plane = gymapi.PlaneParams()
    plane.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane)

    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        cprint("Failed to create viewer (don't run --headless).", "red")
        return

    # hand asset options (floating base, no gravity) -- mirror mano2dexhand
    hand_opts = gymapi.AssetOptions()
    hand_opts.fix_base_link = False
    hand_opts.disable_gravity = True
    hand_opts.flip_visual_attachments = False
    hand_opts.collapse_fixed_joints = False
    hand_opts.default_dof_drive_mode = gymapi.DOF_MODE_POS

    obj_opts = gymapi.AssetOptions()
    obj_opts.fix_base_link = True
    obj_opts.disable_gravity = True
    obj_opts.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
    obj_opts.thickness = 0.001
    obj_opts.vhacd_enabled = False
    obj_opts.density = 200

    env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)
    identity = gymapi.Transform()

    # table: same box the BiH env uses (1.0 x 1.6 x 0.03 at (-0.1,0,0.4), surface z=0.415)
    table_opts = gymapi.AssetOptions()
    table_opts.fix_base_link = True
    table_width_offset = 0.2
    table_asset = gym.create_box(sim, 0.8 + table_width_offset, 1.6, 0.03, table_opts)
    table_pose = gymapi.Transform()
    table_pose.p = gymapi.Vec3(-table_width_offset / 2, 0.0, 0.4)
    table_actor = gym.create_actor(env, table_asset, table_pose, "table", 0, 0)
    gym.set_rigid_body_color(env, table_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.1, 0.1, 0.1))

    # create actors per side: hand then object (dof tensor order follows creation order)
    for h in hands:
        dx = h["dexhand"]
        h_root, h_file = os.path.split(dx.urdf_path)
        hand_asset = gym.load_asset(sim, h_root, h_file, hand_opts)
        n_dofs = gym.get_asset_dof_count(hand_asset)

        actor = gym.create_actor(env, hand_asset, identity, f"hand_{h['side']}", 0,
                                 (1 if dx.self_collision else 0))
        dof_props = gym.get_asset_dof_properties(hand_asset)
        for i in range(n_dofs):
            dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            dof_props["stiffness"][i] = 1000.0
            dof_props["damping"][i] = 50.0
        gym.set_actor_dof_properties(env, actor, dof_props)
        h["hand_actor"] = actor
        h["n_dofs"] = n_dofs

        obj_asset = gym.load_asset(sim, *os.path.split(h["obj_urdf"]), obj_opts)
        h["obj_actor"] = gym.create_actor(env, obj_asset, identity, f"obj_{h['side']}", 0, 0)

    # camera on the table center
    gym.viewer_camera_look_at(viewer, env, gymapi.Vec3(0.6, 0.6, 0.9), gymapi.Vec3(-0.1, 0.0, 0.42))
    gym.prepare_sim(sim)

    gym.refresh_actor_root_state_tensor(sim)
    _root = gym.acquire_actor_root_state_tensor(sim)
    _dof = gym.acquire_dof_state_tensor(sim)
    root_state = gymtorch.wrap_tensor(_root)           # [n_actors,13]
    dof_state = gymtorch.wrap_tensor(_dof)             # [total_dofs,2]

    # pin the (static) table row so the per-frame set_actor_root_state_tensor never
    # drags it to the origin -- only hand/object rows are written in the loop.
    root_state[table_actor, :] = torch.tensor(
        [-table_width_offset / 2, 0.0, 0.4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        device=device, dtype=root_state.dtype,
    )

    # per-side dof slice offsets in creation order
    off = 0
    for h in hands:
        h["dof_slice"] = slice(off, off + h["n_dofs"])
        off += h["n_dofs"]

    dt = 1.0 / (60.0 * max(args.speed, 1e-3))
    frame = start
    last_print = -1
    while not gym.query_viewer_has_closed(viewer):
        for h in hands:
            root_state[h["hand_actor"]] = h["root13"][frame]
            root_state[h["obj_actor"]] = h["obj13"][frame]
            dof_state[h["dof_slice"], 0] = h["dof"][frame]
            dof_state[h["dof_slice"], 1] = 0
        gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_state))
        gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_state))
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(dof_state[:, 0].contiguous()))

        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, False)
        gym.sync_frame_time(sim)
        time.sleep(dt)

        if frame != last_print and frame % 10 == 0:
            cprint(f"  frame {frame}/{end}", "yellow")
            last_print = frame

        if frame >= end:
            if args.no_loop:
                # hold on last frame until the window is closed
                while not gym.query_viewer_has_closed(viewer):
                    gym.step_graphics(sim)
                    gym.draw_viewer(viewer, sim, False)
                    gym.sync_frame_time(sim)
                break
            frame = start
        else:
            frame += 1

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
