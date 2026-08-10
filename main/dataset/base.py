from abc import ABC, abstractmethod
import os
from scipy.ndimage import gaussian_filter1d
import numpy as np
import torch
from main.dataset.transform import aa_to_rotmat, caculate_align_mat, rotmat_to_aa
from torch.utils.data import Dataset
from pytorch3d.ops import sample_points_from_meshes
from termcolor import cprint
import pickle


class ManipData(Dataset, ABC):
    def __init__(
        self,
        *,
        data_dir: str,
        split: str = "all",
        skip: int = 2,
        fps: float = 120.0,  # native source rate (Hz); OakInk2/GRAB=120, mydataset passes 60
        target_fps=None,     # desired training rate (Hz); if set, derives skip = round(fps/target_fps)
        device="cuda:0",
        mujoco2gym_transf=None,
        max_seq_len=int(1e10),
        dexhand=None,
        verbose=True,
        causal=False,
        causal_ema_alpha=0.4,
        causal_mode="pos_ema",
        **kwargs,
    ):
        self.data_dir = data_dir
        self.split = split
        # native source rate (Hz): 120 for OakInk2/GRAB, 60 for mydataset. Velocity time_delta
        # below is 1/(fps/skip); hardcoding 120 treats a 60Hz capture as 120Hz (dt 2x too small).
        self.fps = fps
        # target_fps is the single knob for the training rate: choose it and skip is DERIVED, so the
        # raw AND the retarget (load_retargeted_data) subsample to the same rate. Falls back to the
        # passed skip when target_fps is None (unchanged behavior). self.target_fps is the ACTUAL
        # effective rate (fps/skip) — the velocity time base and what the retarget is subsampled to.
        if target_fps is not None:
            self.skip = max(1, round(self.fps / float(target_fps)))
        else:
            self.skip = skip
        self.target_fps = self.fps / self.skip
        self.data_pathes = None

        # causal=True: compute demo velocities causally, emulating LiveTargetSource so offline
        # targets match the live stream. Default False keeps the non-causal np.gradient +
        # Gaussian filter (which looks ahead in time). causal_mode selects the causal method:
        #   pos_ema — low-pass positions, then backward-diff (linear velocities only; smoother).
        #   vel_ema — backward-diff, then EMA the velocity (the original causal path).
        self.causal = causal
        self.causal_ema_alpha = causal_ema_alpha
        self.causal_mode = causal_mode

        self.dexhand = dexhand
        self.device = device

        self.verbose = verbose

        # ? modify this depending on the origin point
        self.transf_offset = torch.eye(4, dtype=torch.float32, device=mujoco2gym_transf.device)

        self.mujoco2gym_transf = mujoco2gym_transf
        self.max_seq_len = max_seq_len

        # caculate contact

        import chamfer_distance as chd

        self.ch_dist = chd.ChamferDistance()

    def __len__(self):
        return len(self.data_pathes)

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @staticmethod
    def _causal_ema(x, alpha, seed_first=False):
        """Causal (forward-only) exponential moving average over the time axis (axis 0).

        seed_first=False: acc starts at 0, so out[0]=alpha*x[0]. Used for VELOCITIES, whose
        first sample is preset to 0 (no previous frame to diff against) — matches
        LiveTargetSource's zero-velocity first frame.
        seed_first=True: acc starts at x[0], so out[0]=x[0]. Used when smoothing POSITIONS
        (pos_ema mode); positions have no natural zero, so seeding at 0 would inject a huge
        spurious frame-0 jump into the derivative.
        """
        out = np.empty_like(x)
        acc = np.array(x[0]) if seed_first else np.zeros_like(x[0])
        for t in range(x.shape[0]):
            acc = alpha * x[t] + (1.0 - alpha) * acc
            out[t] = acc
        return out

    @staticmethod
    def compute_velocity(p, time_delta, guassian_filter=True, causal=False, ema_alpha=0.4, causal_mode="pos_ema"):
        # [T, K, 3]
        if causal:
            # Causal (real-time-realizable) paths — use only past/current frames, never look
            # ahead, so offline targets match the live stream (LiveTargetSource).
            p_np = p.cpu().numpy()
            if causal_mode == "pos_ema":
                # pos_ema: low-pass the POSITIONS (EMA seeded at frame 0), then backward-diff.
                # Smoothing before differencing avoids amplifying position noise through the
                # derivative — cleaner than EMAing the velocity, and tracks the offline signal.
                p_s = ManipData._causal_ema(p_np, ema_alpha, seed_first=True)
                velocity = np.zeros_like(p_s)
                velocity[1:] = (p_s[1:] - p_s[:-1]) / time_delta
            else:
                # vel_ema: backward-diff raw positions, then EMA the velocity.
                diff = np.zeros_like(p_np)
                diff[1:] = (p_np[1:] - p_np[:-1]) / time_delta
                velocity = ManipData._causal_ema(diff, ema_alpha)
        else:
            velocity = np.gradient(p.cpu().numpy(), axis=0) / time_delta
            if guassian_filter:
                velocity = gaussian_filter1d(velocity, 2, axis=0, mode="nearest")
        return torch.from_numpy(velocity).to(p)

    @staticmethod
    def compute_angular_velocity(r, time_delta: float, guassian_filter=True, causal=False, ema_alpha=0.4):
        # [T, K, 3, 3]
        if causal:
            # Causal path — mirrors LiveTargetSource._ang: backward difference
            # rel[t] = R[t] @ R[t-1].T (rotation t-1 -> t), axis-angle / time_delta, causal EMA.
            diff_r = r[1:] @ r[:-1].transpose(-1, -2)  # [T-1, K, 3, 3] rotation t-1 -> t
            diff_aa = rotmat_to_aa(diff_r).cpu().numpy()  # [T-1, K, 3]
            angular_velocity = np.zeros((r.shape[0],) + diff_aa.shape[1:], dtype=diff_aa.dtype)  # [T, K, 3]
            angular_velocity[1:] = diff_aa / time_delta
            angular_velocity = ManipData._causal_ema(angular_velocity, ema_alpha)
            return torch.from_numpy(angular_velocity).to(r)
        diff_r = r[1:] @ r[:-1].transpose(-1, -2)  # [T-1, K, 3, 3]
        diff_aa = rotmat_to_aa(diff_r).cpu().numpy()  # [T-1, K, 3]
        diff_angle = np.linalg.norm(diff_aa, axis=-1)  # [T-1, K]
        diff_axis = diff_aa / (diff_angle[:, :, None] + 1e-8)  # [T-1, K, 3]
        angular_velocity = diff_axis * diff_angle[:, :, None] / time_delta  # [T-1, K, 3]
        angular_velocity = np.concatenate([angular_velocity, angular_velocity[-1:]], axis=0)  # [T, K, 3]
        if guassian_filter:
            angular_velocity = gaussian_filter1d(angular_velocity, 2, axis=0, mode="nearest")
        return torch.from_numpy(angular_velocity).to(r)

    @staticmethod
    def compute_dof_velocity(dof, time_delta, guassian_filter=True, causal=False, ema_alpha=0.4):
        # [T, K]
        if causal:
            dof_np = dof.cpu().numpy()
            diff = np.zeros_like(dof_np)
            diff[1:] = (dof_np[1:] - dof_np[:-1]) / time_delta
            velocity = ManipData._causal_ema(diff, ema_alpha)
        else:
            velocity = np.gradient(dof.cpu().numpy(), axis=0) / time_delta
            if guassian_filter:
                velocity = gaussian_filter1d(velocity, 2, axis=0, mode="nearest")
        return torch.from_numpy(velocity).to(dof)

    def random_sampling_pc(self, mesh):
        numpy_random_state = np.random.get_state()
        torch_random_state = torch.random.get_rng_state()
        torch_random_state_cuda = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        np.random.seed(0)
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        rs_verts_obj = sample_points_from_meshes(mesh, 1000, return_normals=False).to(self.device).squeeze(0)

        # reset seed
        np.random.set_state(numpy_random_state)
        torch.random.set_rng_state(torch_random_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(torch_random_state_cuda)

        return rs_verts_obj

    def process_data(self, data, idx, rs_verts_obj):
        data["obj_trajectory"] = self.mujoco2gym_transf @ data["obj_trajectory"]
        # A prop (object_sets.ObjectSet.prop) is spawned and collided with but never scored, so it
        # needs the frame transform and the length cut below — and nothing else (no velocity, no
        # tips_distance, no BPS).
        if "prop_trajectory" in data:
            data["prop_trajectory"] = self.mujoco2gym_transf @ data["prop_trajectory"]
        # side = "RH" if "RH" in type(self).__name__ else "LH"
        # path = self.data_pathes[idx] if self.data_pathes is not None else idx
        # print(f"[{side}] {path} | obj_trajectory[0] pos: {data['obj_trajectory'][0, :3, 3].tolist()}")
        data["wrist_pos"] = (self.mujoco2gym_transf[:3, :3] @ data["wrist_pos"].T).T + self.mujoco2gym_transf[:3, 3]
        data["wrist_rot"] = rotmat_to_aa(self.mujoco2gym_transf[:3, :3] @ data["wrist_rot"])
        for k in data["mano_joints"].keys():
            data["mano_joints"][k] = (
                self.mujoco2gym_transf[:3, :3] @ data["mano_joints"][k].T
            ).T + self.mujoco2gym_transf[:3, 3]

        # caculate distance
        obj_verts_transf = (data["obj_trajectory"][:, :3, :3] @ rs_verts_obj.T[None]).transpose(-1, -2) + data[
            "obj_trajectory"
        ][:, :3, 3][:, None]

        tip_list = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]

        tips = torch.cat(
            [data["mano_joints"][t_k][:, None] for t_k in (tip_list)],
            dim=1,
        )
        tips_near, _, _, _ = self.ch_dist(tips, obj_verts_transf)
        # tips_contact = tips_near <= 0.008**2  # ? 8mm, ch_dist return square distance closest to any point on the object
        data["tips_distance"] = torch.sqrt(tips_near)

        data["obj_velocity"] = self.compute_velocity(
            data["obj_trajectory"][:, None, :3, 3], 1 / (self.fps / self.skip), guassian_filter=True,
            causal=self.causal, ema_alpha=self.causal_ema_alpha, causal_mode=self.causal_mode,
        ).squeeze(1)
        data["obj_angular_velocity"] = self.compute_angular_velocity(
            data["obj_trajectory"][:, None, :3, :3], 1 / (self.fps / self.skip), guassian_filter=True,
            causal=self.causal, ema_alpha=self.causal_ema_alpha,
        ).squeeze(1)
        data["wrist_velocity"] = self.compute_velocity(
            data["wrist_pos"][:, None], 1 / (self.fps / self.skip), guassian_filter=True,
            causal=self.causal, ema_alpha=self.causal_ema_alpha, causal_mode=self.causal_mode,
        ).squeeze(1)
        data["wrist_angular_velocity"] = self.compute_angular_velocity(
            aa_to_rotmat(data["wrist_rot"][:, None]), 1 / (self.fps / self.skip), guassian_filter=True,
            causal=self.causal, ema_alpha=self.causal_ema_alpha,
        ).squeeze(1)
        data["mano_joints_velocity"] = {}
        for k in data["mano_joints"].keys():
            data["mano_joints_velocity"][k] = self.compute_velocity(
                data["mano_joints"][k], 1 / (self.fps / self.skip), guassian_filter=True,
                causal=self.causal, ema_alpha=self.causal_ema_alpha, causal_mode=self.causal_mode,
            )

        if len(data["obj_trajectory"]) > self.max_seq_len:
            cprint(
                f"WARN: {data['data_path']} is too long : {len(data['obj_trajectory'])}, cut to {self.max_seq_len}",
                "yellow",
            )
            data["obj_trajectory"] = data["obj_trajectory"][: self.max_seq_len]
            if "prop_trajectory" in data:
                data["prop_trajectory"] = data["prop_trajectory"][: self.max_seq_len]
            data["obj_velocity"] = data["obj_velocity"][: self.max_seq_len]
            data["obj_angular_velocity"] = data["obj_angular_velocity"][: self.max_seq_len]
            data["wrist_pos"] = data["wrist_pos"][: self.max_seq_len]
            data["wrist_rot"] = data["wrist_rot"][: self.max_seq_len]
            for k in data["mano_joints"].keys():
                data["mano_joints"][k] = data["mano_joints"][k][: self.max_seq_len]
            data["wrist_velocity"] = data["wrist_velocity"][: self.max_seq_len]
            data["wrist_angular_velocity"] = data["wrist_angular_velocity"][: self.max_seq_len]
            for k in data["mano_joints_velocity"].keys():
                data["mano_joints_velocity"][k] = data["mano_joints_velocity"][k][: self.max_seq_len]
            data["tips_distance"] = data["tips_distance"][: self.max_seq_len]

        cprint(
            f"[{type(self).__name__}] demo {data['data_path']} | {len(data['obj_trajectory'])} frames",
            "cyan",
        )

    def load_retargeted_data(self, data, retargeted_data_path):
        if not os.path.exists(retargeted_data_path):
            if self.verbose:
                cprint(f"\nWARNING: {retargeted_data_path} does not exist.", "red")
                cprint(f"WARNING: This may lead to a slower transfer process or even failure to converge.", "red")
                cprint(
                    f"WARNING: It is recommended to first execute the retargeting code to obtain initial values.\n",
                    "red",
                )
            data.update(
                {
                    "opt_wrist_pos": data["wrist_pos"],
                    "opt_wrist_rot": data["wrist_rot"],
                    "opt_dof_pos": torch.zeros([data["wrist_pos"].shape[0], self.dexhand.n_dofs], device=self.device),
                }
            )
        else:
            opt_params = pickle.load(open(retargeted_data_path, "rb"))
            # Subsample the retarget to the training rate. The pkl stamps the fps its frames were
            # retargeted at (retarget_fps); bring that down to self.target_fps so it stays aligned
            # with the raw (subsampled by self.skip in the loader). Legacy pkls with no stamp are
            # assumed already at the load rate (retarget_skip=1) -> unchanged behavior.
            retarget_fps = opt_params.get("retarget_fps", None)
            retarget_skip = max(1, round(retarget_fps / self.target_fps)) if retarget_fps else 1
            rsl = slice(None, None, retarget_skip)
            data.update(
                {
                    "opt_wrist_pos": torch.tensor(opt_params["opt_wrist_pos"][rsl], device=self.device),
                    "opt_wrist_rot": torch.tensor(opt_params["opt_wrist_rot"][rsl], device=self.device),
                    "opt_dof_pos": torch.tensor(opt_params["opt_dof_pos"][rsl], device=self.device),
                    # "opt_joints_pos": torch.tensor(opt_params["opt_joints_pos"], device=self.device), # ? only used for ablation study
                }
            )
        data["opt_wrist_velocity"] = self.compute_velocity(
            data["opt_wrist_pos"][:, None], 1 / (self.fps / self.skip), guassian_filter=True,
            causal=self.causal, ema_alpha=self.causal_ema_alpha,
        ).squeeze(1)
        data["opt_wrist_angular_velocity"] = self.compute_angular_velocity(
            aa_to_rotmat(data["opt_wrist_rot"][:, None]), 1 / (self.fps / self.skip), guassian_filter=True,
            causal=self.causal, ema_alpha=self.causal_ema_alpha,
        ).squeeze(1)
        data["opt_dof_velocity"] = self.compute_dof_velocity(
            data["opt_dof_pos"], 1 / (self.fps / self.skip), guassian_filter=True,
            causal=self.causal, ema_alpha=self.causal_ema_alpha,
        )
        # data["opt_joints_velocity"] = self.compute_velocity(
        #     data["opt_joints_pos"], 1 / (self.fps / self.skip), guassian_filter=True
        # ) # ? only used for ablation study
        if len(data["opt_wrist_pos"]) > self.max_seq_len:
            data["opt_wrist_pos"] = data["opt_wrist_pos"][: self.max_seq_len]
            data["opt_wrist_rot"] = data["opt_wrist_rot"][: self.max_seq_len]
            data["opt_wrist_velocity"] = data["opt_wrist_velocity"][: self.max_seq_len]
            data["opt_wrist_angular_velocity"] = data["opt_wrist_angular_velocity"][: self.max_seq_len]
            data["opt_dof_pos"] = data["opt_dof_pos"][: self.max_seq_len]
            data["opt_dof_velocity"] = data["opt_dof_velocity"][: self.max_seq_len]
            # ? only used for ablation study
            # data["opt_joints_pos"] = data["opt_joints_pos"][: self.max_seq_len]
            # data["opt_joints_velocity"] = data["opt_joints_velocity"][: self.max_seq_len]
        assert len(data["opt_wrist_pos"]) == len(data["obj_trajectory"])
