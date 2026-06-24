"""Read-only viewer for retargeted dexhand poses against the tracked object.

Loads the *already-saved* retargeting pkl (via the normal dataset path) and the
object trajectory, then statically displays one chosen frame per env so you can
inspect hand<->object interpenetration at reset. It does NOT optimize and does
NOT overwrite any retargeting pkl (unlike mano2dexhand.py).

Each env shows one frame: env i == --frames[i]. Camera starts on the middle env.

Example (inspect the LH burner-body grasp of m_164601 at frames 0 and 20):
    python view_retarget_frames.py --data_idx m_164601 --side left \
        --dexhand inspire --frames 0,20

The dataset is created with an IDENTITY mujoco2gym transform so the object
(transformed by the dataset) and the opt_* (loaded raw from the pkl) live in the
same frame -- this is the exact frame the retargeting was produced in, so the
relative hand/object geometry (and any penetration) is faithful.
"""

import os
from isaacgym import gymapi, gymtorch, gymutil  # must come before torch
import numpy as np
import torch
from termcolor import cprint

from main.dataset.factory import ManipDataFactory
from main.dataset.transform import aa_to_quat, aa_to_rotmat, rotmat_to_quat
from main.dataset.mano2dexhand import Mano2Dexhand
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory


def _parse_frames(s):
    """Parse a frames spec: comma-separated indices and/or 'a-b' inclusive ranges.
    e.g. '0,20' -> [0,20];  '0-9' -> [0..9];  '0-3,8,20' -> [0,1,2,3,8,20]."""
    out = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return out


def _table_grid_lines(step=0.05):
    """Line segments outlining the env's table top surface (z=0.415) so object/table
    penetration is visible. Matches dexhandmanip_bih.py:268-274 (box 1.0x1.6 centered
    at (-0.1,0,0.4), surface at z=0.415). Returns (verts[N*2,3], colors[N,3]) float32."""
    z = 0.415
    x0, x1 = -0.6, 0.4   # center -0.1 +/- 0.5
    y0, y1 = -0.8, 0.8   # center  0.0 +/- 0.8
    segs = []
    x = x0
    while x <= x1 + 1e-6:
        segs.append(((x, y0, z), (x, y1, z)))
        x += step
    y = y0
    while y <= y1 + 1e-6:
        segs.append(((x0, y, z), (x1, y, z)))
        y += step
    verts = np.array(segs, dtype=np.float32).reshape(-1, 3)
    colors = np.tile(np.array([0.3, 0.3, 0.3], dtype=np.float32), (len(segs), 1))
    return verts, colors


def _env_mujoco2gym_transf(device):
    """The exact table transform the BiH env builds at startup
    (dexhandmanip_bih.py:281-286). The saved opt_* are in this frame, so the
    object trajectory must be transformed by the same matrix to line up."""
    table_half_height = 0.015
    table_pos_z = 0.4
    table_surface_z = table_pos_z + table_half_height
    m = np.eye(4)
    m[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(np.array([np.pi / 2, 0, 0]))
    m[:3, 3] = np.array([0, 0, table_surface_z])
    return torch.tensor(m, device=device, dtype=torch.float32)


def main():
    args = gymutil.parse_arguments(
        description="View retargeted dexhand frames (read-only)",
        headless=True,  # registers the --headless flag (defaults to False -> viewer on)
        custom_parameters=[
            {"name": "--data_idx", "type": str, "default": "m_164601"},
            {"name": "--dexhand", "type": str, "default": "inspire"},
            {"name": "--side", "type": str, "default": "left"},
            {"name": "--frames", "type": str, "default": "0,20",
             "help": "frame indices, one env each: comma list and/or ranges, e.g. '0,20' or '0-9'"},
        ],
    )

    frames = _parse_frames(args.frames)
    assert len(frames) > 0, "need at least one frame"

    dexhand = DexHandFactory.create_hand(args.dexhand, args.side)

    # Use the SAME table transform the env uses: the saved opt_* (hand) are stored in
    # that frame, so obj_trajectory must be transformed by it too or the object lands
    # in the wrong place relative to the hand.
    dataset_type = ManipDataFactory.dataset_type(args.data_idx)
    demo = ManipDataFactory.create_data(
        manipdata_type=dataset_type,
        side=args.side,
        device="cuda:0",
        mujoco2gym_transf=_env_mujoco2gym_transf("cuda:0"),
        dexhand=dexhand,
        verbose=True,
    )
    data = demo[args.data_idx]

    T = data["opt_wrist_pos"].shape[0]
    for f in frames:
        assert 0 <= f < T, f"frame {f} out of range [0,{T})"
    cprint(f"Loaded {args.data_idx} side={args.side}: {T} frames, showing {frames}", "cyan")

    # one env per requested frame
    args.num_envs = len(frames)
    viz = Mano2Dexhand(args, dexhand, data["obj_urdf_path"])
    if viz.headless:
        cprint("This is a viewer -- do NOT pass --headless. Nothing to show.", "red")
        return

    sel = torch.tensor(frames, device=viz.sim_device, dtype=torch.long)

    # --- gather per-env (per-frame) poses in the sim frame ---
    wrist_pos = data["opt_wrist_pos"][sel].to(viz.sim_device).float()                 # [E,3]
    wrist_quat = aa_to_quat(data["opt_wrist_rot"][sel].to(viz.sim_device).float())    # [E,4] wxyz
    wrist_quat = wrist_quat[:, [1, 2, 3, 0]]                                          # -> xyzw

    dof_pos = data["opt_dof_pos"][sel].to(viz.sim_device).float()                     # [E,n_dofs]
    dof_pos = torch.clamp(dof_pos, viz.dexhand_dof_lower_limits, viz.dexhand_dof_upper_limits)

    obj_traj = data["obj_trajectory"][sel].to(viz.sim_device).float()                # [E,4,4]
    obj_pos = obj_traj[:, :3, 3]
    obj_quat = rotmat_to_quat(obj_traj[:, :3, :3])[:, [1, 2, 3, 0]]                   # -> xyzw

    obj_actor = viz.obj_actor
    # The object is fix_base_link=True (static) so its net contact force reads ~0
    # regardless of penetration; the depenetration reaction lands on the dynamic hand
    # bodies instead. Read those. dexhand_handles maps body name -> env-local rigid
    # body index into _net_cf.
    hand_body_names = list(viz.dexhand_handles.keys())
    hand_body_idx = torch.tensor(
        [viz.dexhand_handles[n] for n in hand_body_names], device=viz.sim_device, dtype=torch.long
    )

    # table surface grid (this viewer has no table actor, only a ground plane)
    tbl_verts, tbl_colors = _table_grid_lines()
    n_tbl_lines = len(tbl_colors)

    # fingertip bodies (the real *_tip links, same as eval_score.py: weight_idx "tip"
    # entries index into body_names) + object surface point cloud for tip->mesh dist.
    # NOTE: dexhand.contact_body_names are the intermediate/distal contact-sensing links,
    # NOT the fingertips -- don't use those here.
    tip_labels = [k for k in dexhand.weight_idx if "tip" in k]          # thumb_tip, index_tip, ...
    tip_names = [dexhand.body_names[dexhand.weight_idx[k][0]] for k in tip_labels]
    tip_idx = torch.tensor([viz.dexhand_handles[n] for n in tip_names], device=viz.sim_device, dtype=torch.long)
    obj_verts_local = data["obj_verts"]
    if not torch.is_tensor(obj_verts_local):
        obj_verts_local = torch.as_tensor(np.asarray(obj_verts_local))
    obj_verts_local = obj_verts_local.to(viz.sim_device).float()  # [N,3]

    it = 0
    printed = False
    while not viz.gym.query_viewer_has_closed(viz.viewer):
        # re-assert hand + object root states (hand floats with gravity disabled)
        viz._root_state[:, 0, :3] = wrist_pos
        viz._root_state[:, 0, 3:7] = wrist_quat
        viz._root_state[:, 0, 7:] = 0
        viz._root_state[:, obj_actor, :3] = obj_pos
        viz._root_state[:, obj_actor, 3:7] = obj_quat
        viz._root_state[:, obj_actor, 7:] = 0
        viz.gym.set_actor_root_state_tensor(viz.sim, gymtorch.unwrap_tensor(viz._root_state))

        # teleport the fingers exactly onto opt_dof_pos (state + target so PD holds)
        viz._dof_state[..., 0] = dof_pos
        viz._dof_state[..., 1] = 0
        viz.gym.set_dof_state_tensor(viz.sim, gymtorch.unwrap_tensor(viz._dof_state))
        viz.gym.set_dof_position_target_tensor(viz.sim, gymtorch.unwrap_tensor(dof_pos.contiguous()))

        viz.gym.simulate(viz.sim)
        viz.gym.fetch_results(viz.sim, True)
        viz.gym.refresh_net_contact_force_tensor(viz.sim)
        viz.gym.refresh_rigid_body_state_tensor(viz.sim)
        viz.gym.step_graphics(viz.sim)

        # fingertip world positions + nearest object surface point (for vis + readout)
        tips_world = viz._rigid_body_state[:, tip_idx, :3]                       # [E,n_tips,3]
        verts_world = (
            torch.einsum("eij,nj->eni", obj_traj[:, :3, :3], obj_verts_local)
            + obj_traj[:, None, :3, 3]
        )                                                                       # [E,N,3]
        nn_i = torch.cdist(tips_world, verts_world).min(dim=2).indices          # [E,n_tips]
        tips_near = torch.gather(verts_world, 1, nn_i[..., None].expand(-1, -1, 3))  # [E,n_tips,3]

        viz.gym.clear_lines(viz.viewer)
        for e, env in enumerate(viz.envs):
            # table top surface grid so object/table penetration is visible
            viz.gym.add_lines(viz.viewer, env, n_tbl_lines, tbl_verts, tbl_colors)
            # one line per fingertip: tip -> closest point on object (yellow)
            tip_segs = torch.stack([tips_world[e], tips_near[e]], dim=1).reshape(-1, 3)
            tip_segs = tip_segs.detach().cpu().numpy().astype(np.float32)
            tip_cols = np.tile(np.array([1.0, 1.0, 0.0], dtype=np.float32), (len(tip_names), 1))
            viz.gym.add_lines(viz.viewer, env, len(tip_names), tip_segs, tip_cols)

        viz.gym.draw_viewer(viz.viewer, viz.sim, False)
        viz.gym.sync_frame_time(viz.sim)

        it += 1
        if not printed and it >= 30:  # let contact settle before sampling
            hand_cf = torch.norm(viz._net_cf[:, hand_body_idx], dim=-1)  # [E, n_hand_bodies]
            total = hand_cf.sum(dim=1)                                   # [E]
            peak, peak_i = hand_cf.max(dim=1)                            # [E]

            # tip->mesh distance from the same tip/nearest-point pairs drawn above
            dmin = torch.norm(tips_world - tips_near, dim=2)             # [E,n_tips]
            for e, f in enumerate(frames):
                worst = hand_body_names[peak_i[e].item()]
                cprint(
                    f"  env {e}  frame {f}:  hand contact force  total={total[e].item():.2f} N  "
                    f"peak={peak[e].item():.2f} N @ {worst}",
                    "yellow",
                )
                tip_str = "  ".join(f"{tn}={dmin[e, i].item() * 100:.2f}cm" for i, tn in enumerate(tip_labels))
                cprint(f"            tip->mesh:  {tip_str}", "green")
            printed = True

    viz.gym.destroy_viewer(viz.viewer)
    viz.gym.destroy_sim(viz.sim)


if __name__ == "__main__":
    main()
