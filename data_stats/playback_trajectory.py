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
    python data_stats/playback_trajectory.py --data_idx m_164621 --record          # -> mp4, headless

Controls: --speed (playback rate vs realtime), --start/--end (frame window),
--no_loop (play once and hold on the last frame).

Recording (--record): renders one pass start->end through an off-screen camera sensor to a
video file, then exits. It creates NO viewer, so it runs on a headless server (needs only a
valid --graphics_device_id for the GPU camera). Output defaults to
vis_traj_outputs/playback/<data_idx>_<side>.mp4 (override with --record_path).
mp4 needs an ffmpeg backend (`pip install imageio-ffmpeg`, bundles ffmpeg, no system dep);
frames stream straight into the encoder (no intermediate PNGs) -- without a backend it errors
rather than writing anything.

Interactive (non-recording): each loop plays through at realtime (scaled by --speed);
at the end it pauses and waits for SPACE to start the next loop. Every frame prints the
TARGET thumb-tip and middle-tip -> object surface distances (demo MANO keypoints, not
the sim bodies) for every loaded hand.
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
            {"name": "--record", "action": "store_true",
             "help": "record one pass start->end to a video (headless, no viewer), then exit"},
            {"name": "--record_path", "type": str, "default": "",
             "help": "output video path; default vis_traj_outputs/playback/<data_idx>_<side>.mp4"},
            {"name": "--record_fps", "type": int, "default": -1,
             "help": "video fps; default = round(60*speed) so it matches on-screen speed"},
            {"name": "--width", "type": int, "default": 1280, "help": "record camera width"},
            {"name": "--height", "type": int, "default": 720, "help": "record camera height"},
        ],
    )

    recording = args.record
    out_path = ""
    if recording:
        if args.record_path:
            out_path = args.record_path
        else:
            os.makedirs("vis_traj_outputs/playback", exist_ok=True)
            safe_idx = args.data_idx.replace("/", "_").replace("@", "_")
            out_path = f"vis_traj_outputs/playback/{safe_idx}_{args.side}.mp4"
        if args.graphics_device_id < 0:
            cprint("--record needs a GPU for the off-screen camera; pass --graphics_device_id >= 0.", "red")
            return

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
            "obj_verts": data["obj_verts"].to(device).float(),       # [N,3] mesh frame
            "obj_T": data["obj_trajectory"].to(device).float(),      # [T,4,4] gym frame
            # target fingertips (demo MANO keypoints, == target_states["joints_pos"] entries)
            "thumb_tip_target": data["mano_joints"]["thumb_tip"].to(device).float(),    # [T,3]
            "middle_tip_target": data["mano_joints"]["middle_tip"].to(device).float(),  # [T,3]
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
    # Use the GPU for PhysX + the tensor pipeline (from --sim_device / --pipeline, default
    # cuda / gpu). Off-screen camera recording needs the GPU graphics pipeline anyway, so
    # keeping physics on the same GPU avoids a CPU-physics / GPU-render split that can crash
    # render_all_camera_sensors. If this node's GPU arch is unsupported by IsaacGym's prebuilt
    # kernels, fall back with `--sim_device cpu --pipeline cpu` (kinematic playback is happy on
    # CPU physics); recording still needs a real --graphics_device_id for the camera.
    sp.physx.use_gpu = args.use_gpu
    sp.use_gpu_pipeline = args.use_gpu_pipeline
    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sp)

    plane = gymapi.PlaneParams()
    plane.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane)

    # Recording renders through an off-screen camera sensor and needs no viewer, so it works
    # on a headless server. Interactive playback needs the viewer for its window + SPACE key.
    viewer = None
    if not recording:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            cprint("Failed to create viewer (don't run --headless; use --record instead).", "red")
            return
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "advance")

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

    # Move the per-frame poses onto whatever device the gym state tensors live on: cuda with the
    # GPU pipeline (default), cpu with --pipeline cpu. The poses were built on the data device
    # (cuda, required by the chamfer loader), so this is a no-op under the GPU pipeline.
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

    # recording: stream each off-screen camera frame straight into the video writer -- NO PNG
    # frames are written to disk. If there's no ffmpeg backend, error out now (before the render
    # pass) instead of producing anything.
    video_writer = None
    if recording:
        import imageio.v2 as imageio
        fps = args.record_fps if args.record_fps > 0 else max(1, round(60 * args.speed))
        try:
            # needs an ffmpeg backend: `pip install imageio-ffmpeg` (bundles ffmpeg, no system dep)
            video_writer = imageio.get_writer(out_path, fps=fps, macro_block_size=None)
        except Exception as e:
            cprint(f"Cannot open video writer for {out_path} ({e}).", "red")
            cprint("Install an ffmpeg backend:  pip install imageio-ffmpeg", "cyan")
            gym.destroy_sim(sim)
            return
        cprint(f"[headless] Recording one pass [{start}..{end}] @ {fps}fps -> {out_path}", "cyan")

    def apply_frame(frame):
        """Teleport hands+objects onto `frame` and step the sim so body poses update."""
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

    def print_thumb_dist(frame):
        """Min distance from each hand's TARGET fingertips (demo MANO keypoints) to that
        hand's object surface cloud -- a pure target_states distance, no sim state."""
        parts = []
        for h in hands:
            T = h["obj_T"][frame]                                    # [4,4] gym frame
            verts_world = h["obj_verts"] @ T[:3, :3].T + T[:3, 3]    # [N,3]
            d_thumb = (verts_world - h["thumb_tip_target"][frame]).norm(dim=-1).min().item()
            d_middle = (verts_world - h["middle_tip_target"][frame]).norm(dim=-1).min().item()
            parts.append(f"{h['side']}: thumb {d_thumb * 100:.2f} cm, middle {d_middle * 100:.2f} cm")
        cprint(f"[frame {frame}/{end}] target tip -> obj surface  " + " | ".join(parts), "green")

    dt = 1.0 / (60.0 * max(args.speed, 1e-3))

    if recording:
        # auto-advance one pass start->end, streaming each rendered camera frame into the video.
        # No viewer here -- step_graphics + render_all_camera_sensors update the sensor image
        # headlessly, then get_camera_image reads it back into memory (nothing hits disk).
        n_written = 0
        for i, frame in enumerate(range(start, end + 1)):
            # Frame-0 gets flushed markers before each native graphics call: if IsaacGym's
            # off-screen renderer segfaults/SIGFPEs on this GPU, the last line in the log names
            # the offending call (simulate vs step_graphics vs render vs get_camera_image).
            dbg = i == 0
            if dbg: print("[rec] f0: apply_frame (simulate + fetch)...", flush=True)
            apply_frame(frame)
            if dbg: print("[rec] f0: step_graphics...", flush=True)
            gym.step_graphics(sim)
            if dbg: print("[rec] f0: render_all_camera_sensors...", flush=True)
            gym.render_all_camera_sensors(sim)
            if dbg: print("[rec] f0: get_camera_image (RGBA->RGB)...", flush=True)
            img = gym.get_camera_image(sim, env, cam, gymapi.IMAGE_COLOR)
            img = np.ascontiguousarray(img.reshape(args.height, args.width, 4)[:, :, :3])  # drop alpha
            if dbg: print("[rec] f0: append_data to video writer...", flush=True)
            video_writer.append_data(img)
            if dbg: print("[rec] f0: OK -- first frame streamed to video.", flush=True)
            n_written += 1
            if i % 30 == 0 or frame == end:
                print_thumb_dist(frame)
        video_writer.close()
        cprint(f"Saved {n_written} frames @ {fps}fps -> {out_path}", "green")
    else:
        # interactive: each loop plays through at realtime; at the end it pauses and
        # waits for SPACE to start the next loop (--no_loop just stays paused).
        frame = start
        playing = True
        while not gym.query_viewer_has_closed(viewer):
            if playing:
                apply_frame(frame)
                print_thumb_dist(frame)

            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, False)
            gym.sync_frame_time(sim)
            time.sleep(dt)

            space = any(
                evt.action == "advance" and evt.value > 0
                for evt in gym.query_viewer_action_events(viewer)
            )

            if playing:
                if frame >= end:
                    playing = False
                    if not args.no_loop:
                        cprint("End of loop -- press SPACE to play the next loop.", "cyan")
                else:
                    frame += 1
            elif space and not args.no_loop:
                frame = start
                playing = True

    if viewer is not None:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
