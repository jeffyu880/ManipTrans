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
data_stats/vis_traj_outputs/retarget_playback/<data_idx>_<side>.mp4 (override with --record_path).
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

import pickle
import time
from isaacgym import gymapi, gymtorch, gymutil  # before torch
import numpy as np
import torch
from termcolor import cprint

from main.dataset.factory import ManipDataFactory
from maniptrans_envs.lib.envs.core.record_cameras import RECORD_FOV, VIEWS
from main.dataset.transform import aa_to_quat, aa_to_rotmat, rotmat_to_quat
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
from maniptrans_envs.lib.envs.tasks.dexhandmanip_bih import DexHandManipBiHEnv


def env_mujoco2gym_transf(device):
    """Same table transform the BiH env builds (dexhandmanip_bih.py:281-286); the saved
    opt_* are in this frame, so obj_trajectory must be transformed by it to match."""
    table_surface_z = 0.4 + 0.015
    m = np.eye(4)
    m[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(np.array([np.pi / 2, 0, 0]))
    m[:3, 3] = np.array([0, 0, table_surface_z])
    return torch.tensor(m, device=device, dtype=torch.float32)


def apply_dexret_override(data, dexhand, side, device):
    """Swap the loaded `opt_*` for the dex-retargeting baseline's, for viewing only.

    dexret2dexhand writes to its own root (DEXRET_PLAYBACK_ROOT) that no loader resolves, so the
    baseline's trajectories are invisible to training by construction. This is the one place that
    reads them: it overrides the three keys after the dataset loader has run, so nothing in
    main/dataset sees the swap and no trained run can pick it up.

    Args:
        data: One sequence from the loader, mutated in place.
        dexhand: The DexHand instance the sequence was loaded for.
        side: "right" or "left".
        device: Torch device for the loaded tensors.

    Returns:
        str path that was read.
    """
    from baselines.utils import DEXRET_PLAYBACK_ROOT

    stem = os.path.splitext(os.path.basename(data["data_path"]))[0]
    suffix = "rh" if side == "right" else "lh"
    path = os.path.join(
        DEXRET_PLAYBACK_ROOT, f"mano2{dexhand}", f"dex_retarget_{stem}_{suffix}.pkl"
    )
    assert os.path.exists(path), (
        f"no dex-retargeting pkl at {path}. Generate it first:\n"
        f"  python baselines/dexret2dexhand.py --data_idx <idx> --side {side}"
    )
    with open(path, "rb") as f:
        opt = pickle.load(f)

    # The loader subsamples by `skip`; a mismatch means the two were built at different rates and
    # the poses would be silently time-shifted against the objects.
    n_loaded = len(data["opt_wrist_pos"])
    assert len(opt["opt_wrist_pos"]) == n_loaded, (
        f"{path} has {len(opt['opt_wrist_pos'])} frames but the loader produced {n_loaded}. "
        f"Re-run dexret2dexhand for this capture — they were built at different subsample rates."
    )
    for key in ("opt_wrist_pos", "opt_wrist_rot", "opt_dof_pos"):
        data[key] = torch.tensor(opt[key], device=device, dtype=torch.float32)
    return path


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
            {"name": "--dex_retarget", "action": "store_true",
             "help": "play the dex-retargeting baseline's poses from data/dex_retarget_playback/ "
                     "instead of mano2dexhand's. Viewing only -- no loader resolves that root, so "
                     "training can never see these trajectories"},
            {"name": "--record", "action": "store_true",
             "help": "record one pass start->end to a video (headless, no viewer), then exit"},
            {"name": "--record_path", "type": str, "default": "",
             "help": "output video path; default data_stats/vis_traj_outputs/retarget_playback/<data_idx>_<side>.mp4"},
            {"name": "--record_fps", "type": int, "default": -1,
             "help": "video fps; default = round(60*speed) so it matches on-screen speed"},
            {"name": "--view", "type": str, "default": "front",
             "help": "camera pose, matching the env's recordings: front | behind"},
            {"name": "--width", "type": int, "default": 1280, "help": "record camera width"},
            {"name": "--height", "type": int, "default": 720, "help": "record camera height"},
            # trajectory augmentation preview (mirror the training useXxxAug flags); bimanual, so
            # they need --side both. One transform is sampled and applied via the SAME BiH-env
            # static methods in the SAME chain order as training. The short code in [brackets] is
            # appended to the default recording filename (gymutil can't alias short+long flags).
            {"name": "--use_rh_robj_center_aug", "action": "store_true",
             "help": "[rhoc] preview RH_RObj_Center_Aug: rotate RH demo about the RH object center"},
            {"name": "--use_lh_lobj_center_aug", "action": "store_true",
             "help": "[lhoc] preview LH_LObj_Center_Aug: rotate LH hand+object about the LH object center"},
            {"name": "--use_rh_lobj_center_aug", "action": "store_true",
             "help": "[rhloc] preview RH_LObj_Center_Aug: rotate RH demo about the LH object center"},
            {"name": "--use_table_center_aug", "action": "store_true",
             "help": "[tc] preview RH_LH_Table_Center_Aug: rotate both demos about the table center"},
            {"name": "--aug_seed", "type": int, "default": -1,
             "help": "seed for the single sampled aug transform (-1 = random each run)"},
            {"name": "--axis_len", "type": float, "default": 0.08,
             "help": "length (m) of the drawn object coordinate frames; 0 disables"},
        ],
    )

    # Aug flags mirror the training useXxxAug flags. Each carries a short code appended to the
    # default recording filename so the output name reflects which augs were applied.
    aug_specs = [
        ("rh_lobj_center", args.use_rh_lobj_center_aug, "rhloc"),  # rotate RH about LH obj center
        ("rh_robj_center", args.use_rh_robj_center_aug, "rhroc"),  # rotate RH about RH obj center
        ("lh_lobj_center", args.use_lh_lobj_center_aug, "lhloc"),  # rotate LH hand+obj about LH obj center
        ("table_center",    args.use_table_center_aug,    "tc"),     # rotate both about table center
    ]
    aug_flags = {key: enabled for key, enabled, _ in aug_specs}
    use_aug = any(aug_flags.values())
    aug_suffix = "".join(f"_{code}" for _, enabled, code in aug_specs if enabled)

    recording = args.record
    out_path = ""
    if recording:
        if args.record_path:
            out_path = args.record_path
        else:
            os.makedirs("data_stats/vis_traj_outputs/retarget_playback", exist_ok=True)
            safe_idx = args.data_idx.replace("/", "_").replace("@", "_")
            out_path = f"data_stats/vis_traj_outputs/retarget_playback/{safe_idx}_{args.side}{aug_suffix}.mp4"
        if args.graphics_device_id < 0:
            cprint("--record needs a GPU for the off-screen camera; pass --graphics_device_id >= 0.", "red")
            return

    device = "cuda:0"
    sides = ["right", "left"] if args.side == "both" else [args.side]
    m2g = env_mujoco2gym_transf(device)

    if use_aug and set(sides) != {"right", "left"}:
        cprint("Augmentation preview is bimanual (it needs both objects); pass --side both.", "red")
        return

    # ---- load raw demo data per side ----
    dexhands, raw = {}, {}
    for side in sides:
        dexhand = DexHandFactory.create_hand(args.dexhand, side)
        dtype = ManipDataFactory.dataset_type(args.data_idx)
        demo = ManipDataFactory.create_data(
            manipdata_type=dtype, side=side, device=device,
            mujoco2gym_transf=m2g, dexhand=dexhand, verbose=True,
        )
        dexhands[side] = dexhand
        raw[side] = demo[args.data_idx]
        if args.dex_retarget:
            src = apply_dexret_override(raw[side], dexhand, side, device)
            cprint(f"[dex-retarget] {side}: opt_* from {src}", "cyan")

    # ---- optional trajectory augmentation ----
    # One transform is sampled (rot up to +-30 deg about z, XY shift up to +-5 cm) and pushed through
    # the SAME BiH-env static methods in the SAME chain order as training, so this shows exactly one
    # augmented variant. --aug_seed fixes the transform for a repeatable preview.
    if use_aug:
        aug_pivot_center = torch.tensor([-0.1, 0.0, m2g[2, 3].item()], device=device, dtype=torch.float32)
        if args.aug_seed >= 0:
            rng_state = torch.get_rng_state()
            torch.manual_seed(args.aug_seed)
        aug_rotation, aug_translation, aug_pivot = DexHandManipBiHEnv._sample_aug_transform(device, aug_pivot_center)
        if args.aug_seed >= 0:
            torch.set_rng_state(rng_state)

        rh, lh = raw["right"], raw["left"]
        active = []
        if aug_flags["rh_robj_center"]:
            # angle = -15.0 * (np.pi / 180.0)
            # cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
            # R = torch.tensor([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]],
            #                  dtype=torch.float32, device=device)
            rh = DexHandManipBiHEnv._aug_demo_rh_robj_center_aug(rh, aug_rotation)
            active.append("RH-obj-center")
        if aug_flags["lh_lobj_center"]:
            # angle = 30 * (np.pi / 180.0)
            # cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
            # R = torch.tensor([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]],
            #                  dtype=torch.float32, device=device)
            lh = DexHandManipBiHEnv._aug_demo_lh_lobj_center_aug(lh, aug_rotation)
            active.append("LH-about-LH-obj")
        if aug_flags["rh_lobj_center"]:
            # angle = -15 * (np.pi / 180.0)
            # cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
            # R = torch.tensor([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]],
            #                  dtype=torch.float32, device=device)
            rh, lh = DexHandManipBiHEnv._aug_demo_rh_lobj_center_aug(rh, lh, aug_rotation)
            active.append("LH-obj-center")
        if aug_flags["table_center"]:
            # rotate BOTH demos about the table center by the sampled R (no translation, matching
            # the env: the dataloader re-centers objects onto the table center).
            rh = DexHandManipBiHEnv._aug_demo_table_center(rh, aug_rotation, center=aug_pivot)
            lh = DexHandManipBiHEnv._aug_demo_table_center(lh, aug_rotation, center=aug_pivot)
            active.append("table-center")
        raw["right"], raw["left"] = rh, lh
        angle_deg = np.degrees(np.arctan2(aug_rotation[1, 0].item(), aug_rotation[0, 0].item()))
        seed_str = str(args.aug_seed) if args.aug_seed >= 0 else "random"
        cprint(f"Aug pipeline: {' -> '.join(active)}  (z-rot {angle_deg:+.1f} deg, "
               f"seed={seed_str})", "magenta")

    # ---- build per-hand render buffers ----
    hands = []  # list of dicts: side, dexhand, root13[T,13], obj13[T,13], dof[T,n], obj_urdf
    T = None
    for side in sides:
        data = raw[side]
        dexhand = dexhands[side]

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
    draw_axes = args.axis_len > 0  # draw each object's coordinate frame (RGB = XYZ)

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

    # Same poses the BiH env records through (record_cameras.py), so a playback and a
    # capture_video run of the same demo can be put side by side. This used to be a private
    # oblique 3/4 view, which made the two incomparable.
    eye, target = VIEWS[args.view]
    CAM_EYE = gymapi.Vec3(*eye)
    CAM_TARGET = gymapi.Vec3(*target)
    cam = None
    if recording:
        cam_props = gymapi.CameraProperties()
        cam_props.width = args.width
        cam_props.height = args.height
        cam_props.horizontal_fov = RECORD_FOV
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
        import cv2  # for the frame-number overlay (and object-axis lines) drawn onto each frame
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

    def object_axis_segments(frame):
        """World-space [origin, x_end, y_end, z_end] for each object's pose frame at `frame`.
        Returns a list of (4,3) float32 arrays (one per hand). rot columns are the world-space
        directions of the object's local X/Y/Z axes, so end_i = origin + rot[:,i]*axis_len."""
        segs = []
        for h in hands:
            obj_pose = h["obj_T"][frame]                          # [4,4] gym frame
            origin = obj_pose[:3, 3]
            ends = origin[None, :] + obj_pose[:3, :3].T * args.axis_len   # [3,3]: row i = axis-i tip
            segs.append(torch.cat([origin[None, :], ends], dim=0).cpu().numpy())  # [4,3]
        return segs

    def draw_axes_viewer(frame):
        """Redraw the object frames as viewer debug lines (X red, Y green, Z blue)."""
        gym.clear_lines(viewer)
        axis_colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        for seg in object_axis_segments(frame):
            verts = np.stack([seg[0], seg[1], seg[0], seg[2], seg[0], seg[3]]).astype(np.float32)
            gym.add_lines(viewer, env, 3, verts, axis_colors)

    # Recording has no viewer, so axes are drawn as a cv2 overlay: project each object frame's
    # world points through the (static) camera view*proj into pixels. The image is RGB, so axis
    # colors are RGB tuples (X red, Y green, Z blue).
    draw_axes_cv2 = None
    if recording and draw_axes:
        view_proj = (np.array(gym.get_camera_view_matrix(sim, env, cam))
                     @ np.array(gym.get_camera_proj_matrix(sim, env, cam)))

        def project_to_pixels(pts):
            """[N,3] world -> ([N,2] pixel coords, [N] bool mask of points in front of the camera)."""
            homog = np.concatenate([pts, np.ones((pts.shape[0], 1), np.float32)], axis=1)
            clip = homog @ view_proj
            w = clip[:, 3]
            ndc = clip[:, :3] / w[:, None]
            u = (ndc[:, 0] * 0.5 + 0.5) * args.width
            v = (0.5 - ndc[:, 1] * 0.5) * args.height
            return np.stack([u, v], axis=1), w > 1e-6

        def draw_axes_cv2(img, frame):
            axis_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # RGB: X red, Y green, Z blue
            for seg in object_axis_segments(frame):
                pixels, in_front = project_to_pixels(seg)
                if not in_front[0]:
                    continue
                origin_px = tuple(np.round(pixels[0]).astype(int))
                for i in range(3):
                    if in_front[i + 1]:
                        end_px = tuple(np.round(pixels[i + 1]).astype(int))
                        cv2.line(img, origin_px, end_px, axis_colors[i], 2, cv2.LINE_AA)

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
            if draw_axes:
                draw_axes_cv2(img, frame)
            # frame counter in the top-left, yellow (img is RGB, so yellow = (255, 255, 0));
            # scale the font/thickness with the frame height so it reads at any --width/--height.
            font_scale = args.height / 720.0
            cv2.putText(img, f"frame {frame}", (int(0.02 * args.width), int(0.07 * args.height)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0),
                        max(1, round(2 * font_scale)), cv2.LINE_AA)
            if dbg: print("[rec] f0: append_data to video writer...", flush=True)
            video_writer.append_data(img)
            if dbg: print("[rec] f0: OK -- first frame streamed to video.", flush=True)
            n_written += 1
            if i % 30 == 0 or frame == end:
                print_thumb_dist(frame)
        video_writer.close()
        cprint(f"Saved {n_written} frames @ {fps}fps -> {out_path}", "green")
        # IsaacGym's headless camera pipeline segfaults during destroy_sim / interpreter
        # teardown on some driver stacks, turning a successful recording into exit code 139.
        # The video is already closed, so skip teardown and end the process here with 0.
        os._exit(0)
    else:
        # interactive: each loop plays through at realtime; at the end it pauses and
        # waits for SPACE to start the next loop (--no_loop just stays paused).
        frame = start
        playing = True
        while not gym.query_viewer_has_closed(viewer):
            if playing:
                apply_frame(frame)
                print_thumb_dist(frame)
                if draw_axes:
                    draw_axes_viewer(frame)

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
