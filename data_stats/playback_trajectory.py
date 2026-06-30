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
    python data_stats/playback_trajectory.py --data_idx m_164621 --side both       # both hands
    python data_stats/playback_trajectory.py --data_idx m_164621 --side left --speed 0.5
    python data_stats/playback_trajectory.py --data_idx m_164621 --start 0 --end 9  # just 0..9

Controls: --speed (playback rate vs realtime), --start/--end (frame window),
--no_loop (play once and hold on the last frame).
"""

import os
import sys

# This script lives in <repo>/data_stats/, so the repo root is its parent's parent.
# Put it on sys.path (for `main`/`maniptrans_envs` imports) and make it the CWD (the
# dataset loaders use paths relative to the repo root, e.g. data/...).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)

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
            {"name": "--record", "type": str, "default": "",
             "help": "output video path (e.g. out.mp4); records one pass start->end then exits"},
            {"name": "--record_fps", "type": int, "default": -1,
             "help": "video fps; default = round(60*speed) so it matches on-screen speed"},
            {"name": "--width", "type": int, "default": 1280, "help": "record camera width"},
            {"name": "--height", "type": int, "default": 720, "help": "record camera height"},
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

        # frame-0 sanity: min distance from the cap mesh bottom to the table surface.
        # cap is the right hand's object; obj_verts is the sampled surface cloud in the
        # mesh-local frame, obj_trajectory[0] places it in the gym/table frame.
        if side == "right":
            verts_local = data["obj_verts"].to(device).float()          # [N,3] mesh frame
            T0 = data["obj_trajectory"][0].to(device).float()           # [4,4] gym frame, frame 0
            verts_world = verts_local @ T0[:3, :3].T + T0[:3, 3]        # [N,3]
            min_z = verts_world[:, 2].min().item()
            table_z = m2g[2, 3].item()
            cprint(
                f"[frame 0] cap bottom -> table: {min_z - table_z:+.4f} m "
                f"(cap min z={min_z:.4f}, table z={table_z:.4f}; negative = penetrating)",
                "magenta",
            )

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
    # Kinematic playback needs no GPU physics; force CPU PhysX so it does not depend on the
    # node's GPU arch being supported by IsaacGym's prebuilt PhysX-GPU kernels (unsupported
    # GPUs crash with "PhysX Internal CUDA error / illegal instruction" during simulate()).
    # Graphics still use the GPU (graphics_device_id) for the off-screen camera; the gym
    # state tensors are therefore on CPU and the per-frame poses are moved to CPU below.
    sp.physx.use_gpu = False
    sp.use_gpu_pipeline = False
    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sp)

    plane = gymapi.PlaneParams()
    plane.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane)

    # Recording is headless (off-screen camera sensor, no display); interactive needs a viewer.
    recording = bool(args.record)
    viewer = None
    if not recording:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            cprint("Failed to create viewer (use --record for headless capture).", "red")
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

    # camera on the table center (off-screen sensor for recording, else the viewer camera)
    CAM_EYE = gymapi.Vec3(0.6, 0.6, 0.9)
    CAM_TARGET = gymapi.Vec3(-0.1, 0.0, 0.42)
    cam = None
    if recording:
        cam_props = gymapi.CameraProperties()
        cam_props.width = args.width
        cam_props.height = args.height
        cam = gym.create_camera_sensor(env, cam_props)
        gym.set_camera_location(cam, env, CAM_EYE, CAM_TARGET)
    else:
        gym.viewer_camera_look_at(viewer, env, CAM_EYE, CAM_TARGET)
    gym.prepare_sim(sim)

    gym.refresh_actor_root_state_tensor(sim)
    _root = gym.acquire_actor_root_state_tensor(sim)
    _dof = gym.acquire_dof_state_tensor(sim)
    root_state = gymtorch.wrap_tensor(_root)           # [n_actors,13]
    dof_state = gymtorch.wrap_tensor(_dof)             # [total_dofs,2]

    # The gym state tensors live on the sim device (CPU here); the per-frame poses were built
    # on the data device (cuda, required by the chamfer loader), so move them to the gym device.
    gym_dev = root_state.device
    for h in hands:
        h["root13"] = h["root13"].to(gym_dev)
        h["obj13"] = h["obj13"].to(gym_dev)
        h["dof"] = h["dof"].to(gym_dev)

    # pin the (static) table row so the per-frame set_actor_root_state_tensor never
    # drags it to the origin -- only hand/object rows are written in the loop.
    root_state[table_actor, :] = torch.tensor(
        [-table_width_offset / 2, 0.0, 0.4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        device=gym_dev, dtype=root_state.dtype,
    )

    # per-side dof slice offsets in creation order
    off = 0
    for h in hands:
        h["dof_slice"] = slice(off, off + h["n_dofs"])
        off += h["n_dofs"]

    # recording: capture each off-screen camera frame to a temp PNG, encode to video at the end
    if recording:
        import tempfile
        frames_dir = tempfile.mkdtemp(prefix="playback_rec_")
        cap_paths = []
        cprint(f"[headless] Recording one pass [{start}..{end}] -> {args.record}", "cyan")

    dt = 1.0 / (60.0 * max(args.speed, 1e-3))
    frame = start
    last_print = -1
    while viewer is None or not gym.query_viewer_has_closed(viewer):
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
        if recording:
            gym.render_all_camera_sensors(sim)
            p = os.path.join(frames_dir, f"{len(cap_paths):05d}.png")
            gym.write_camera_image_to_file(sim, env, cam, gymapi.IMAGE_COLOR, p)
            cap_paths.append(p)
        else:
            gym.draw_viewer(viewer, sim, False)
            gym.sync_frame_time(sim)
            time.sleep(dt)

        if frame != last_print and frame % 10 == 0:
            cprint(f"  frame {frame}/{end}", "yellow")
            last_print = frame

        if frame >= end:
            if recording:
                break  # captured the full pass
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

    if viewer is not None:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

    if recording and cap_paths:
        fps = args.record_fps if args.record_fps > 0 else max(1, round(60 * args.speed))
        try:
            import imageio.v2 as imageio
            with imageio.get_writer(args.record, fps=fps, macro_block_size=None) as w:
                for p in cap_paths:
                    w.append_data(imageio.imread(p))
            cprint(f"Saved {len(cap_paths)} frames @ {fps}fps -> {args.record}", "green")
        except Exception as e:
            cprint(f"Could not encode video ({e}). PNG frames are in {frames_dir}", "red")
            cprint(f"Encode manually:  ffmpeg -framerate {fps} -i {frames_dir}/%05d.png {args.record}", "yellow")


if __name__ == "__main__":
    main()
