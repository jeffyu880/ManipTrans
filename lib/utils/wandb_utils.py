import os
import json
import gym
import torch
import wandb
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rl_games.common.algo_observer import AlgoObserver

from lib.utils.utils import retry
from lib.utils.reformat import omegaconf_to_dict


class WandbAlgoObserver(AlgoObserver):
    """Need this to propagate the correct experiment name after initialization."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def before_init(self, base_name, config, experiment_name):
        """
        Must call initialization of Wandb before RL-games summary writer is initialized, otherwise
        sync_tensorboard does not work.
        """

        import wandb

        wandb_unique_id = f"uid_{experiment_name}"
        print(f"Wandb using unique id {wandb_unique_id}")

        cfg = self.cfg

        # Wandb's file watcher crashes if it checks a tfevents file size before
        # rl-games has written it. Patch PolicyLive.current_size to return 0 for missing files.
        from wandb.filesync import dir_watcher as _dw
        _orig_size = _dw.PolicyLive.current_size.fget
        def _safe_current_size(self):
            try:
                return _orig_size(self)
            except FileNotFoundError:
                return 0
        _dw.PolicyLive.current_size = property(_safe_current_size)

        # Delete corrupted wandb resume file before init so retries start clean
        resume_file = os.path.join("wandb", "wandb-resume.json")
        if os.path.exists(resume_file):
            try:
                with open(resume_file) as _f:
                    json.load(_f)
            except Exception:
                os.remove(resume_file)

        # Patch tensorboard exactly once, OUTSIDE the retry below. wandb raises
        # "Tensorboard already patched" if patch() runs twice (and sync_tensorboard=True
        # patches again internally). Previously patch() lived inside the retried
        # init_wandb(), so any init failure made the retry re-patch and throw this error,
        # masking the real cause. root_logdir="runs" points the sync at rl-games' TB output
        # and already enables syncing, so sync_tensorboard=True is dropped from init().
        try:
            wandb.tensorboard.patch(root_logdir="runs")
        except Exception as exc:
            print(f"wandb tensorboard patch skipped (already patched?): {exc}")

        # this can fail occasionally, so we try a couple more times
        @retry(3, exceptions=(Exception,))
        def init_wandb():
            wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                group=cfg.wandb_group,
                tags=cfg.wandb_tags,
                id=wandb_unique_id,
                name=experiment_name,
                resume=True,
                settings=wandb.Settings(start_method="fork"),
            )

            # Hide the redundant per-reward "global_step" panels that wandb's tensorboard
            # sync auto-creates. add_scalars (used for reward_dict) makes one TB sub-namespace
            # per key, so each gets its own global_step series (e.g.
            # reward_dict/time/rh_reward_eef_vel/global_step) whose value is just the step
            # counter. hidden=True keeps the data but removes the auto-generated panels.
            # Scoped to reward_dict so the top-level global_step panel stays visible.
            # Best-effort: older wandb (e.g. 0.12.x) only accepts trailing-`*` globs and
            # raises on this mid-string pattern — don't let a cosmetic tweak abort wandb init.
            try:
                wandb.define_metric("*reward_dict*global_step*", hidden=True)
            except Exception as exc:
                print(f"Skipping define_metric (unsupported glob on this wandb version): {exc}")

            if cfg.wandb_logcode_dir:
                wandb.run.log_code(root=cfg.wandb_logcode_dir)
                print("wandb running directory........", wandb.run.dir)

        print("Initializing WandB...")
        try:
            init_wandb()
        except Exception as exc:
            print(f"Could not initialize WandB! {exc}")

        if isinstance(self.cfg, dict):
            wandb.config.update(self.cfg, allow_val_change=True)
        else:
            wandb.config.update(omegaconf_to_dict(self.cfg), allow_val_change=True)

    def after_init(self, algo):
        from collections import defaultdict
        self._per_demo_rewards = defaultdict(list)
        self._per_demo_successes = defaultdict(list)

    def process_infos(self, infos, done_indices):
        if "per_demo_episode_rewards" not in infos:
            return
        for demo_name, rewards in infos["per_demo_episode_rewards"].items():
            if rewards:
                self._per_demo_rewards[demo_name].extend(rewards)
                self._per_demo_successes[demo_name].extend(infos["per_demo_episode_successes"][demo_name])

    def after_print_stats(self, frame, epoch_num, total_time):
        if not self._per_demo_rewards:
            return
        log_dict = {}
        for demo_name, rewards in self._per_demo_rewards.items():
            if rewards:
                log_dict[f"per_demo/{demo_name}/episode_reward"] = np.mean(rewards)
                log_dict[f"per_demo/{demo_name}/success_rate"] = np.mean(self._per_demo_successes[demo_name])
        if log_dict:
            wandb.log(log_dict, step=frame)
        self._per_demo_rewards.clear()
        self._per_demo_successes.clear()


class WandbVideoCaptureWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        n_parallel_recorders: int = 1,
        n_successful_videos_to_record: int = 50,
        local_video_dir: str = "videos",
    ):
        super().__init__(env)
        n_parallel_recorders = min(n_parallel_recorders, env.num_envs)
        self._n_recorders = n_parallel_recorders
        self._videos = [[] for _ in range(n_parallel_recorders)]
        self._rcd_idxs = [i for i in range(env.num_envs) if i % (env.num_envs // n_parallel_recorders) == 0][
            :n_parallel_recorders
        ]
        self._videos_top = [[] for _ in range(n_parallel_recorders)]
        self._n_video_saved = 0
        self._n_successful_video_saved = 0
        self._n_successful_videos_to_record = n_successful_videos_to_record
        self._n_failed_episodes = 0
        self._max_failed_episodes = 10
        self._warmup = env.num_envs * 2
        self._episodes_seen = 0
        self._local_video_dir = local_video_dir
        os.makedirs(local_video_dir, exist_ok=True)

    def reset(self, **kwargs):
        self._videos = [[] for _ in range(self._n_recorders)]
        self._videos_top = [[] for _ in range(self._n_recorders)]
        return super().reset(**kwargs)

    @staticmethod
    def _burn_frame_number(frame_tensor, frame_num, demo_frame_id=None):
        """Burn frame number text onto a (H, W, C) uint8 tensor (C=3 or 4). Returns numpy array."""
        from PIL import Image, ImageDraw, ImageFont
        arr = frame_tensor.cpu().numpy()
        img = Image.fromarray(arr[..., :3])
        draw = ImageDraw.Draw(img)
        text = f"step {frame_num}" if demo_frame_id is None else f"step {frame_num} | demo {demo_frame_id}"
        font = None
        for font_path in [
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]:
            try:
                font = ImageFont.truetype(font_path, 33)
                break
            except (IOError, OSError):
                continue
        if font is None:
            try:
                font = ImageFont.load_default(size=33)
            except TypeError:
                font = ImageFont.load_default()
        draw.text((4, 4), text, fill=(255, 255, 0), font=font)
        result = np.array(img)
        if arr.shape[-1] == 4:
            result = np.concatenate([result, arr[..., 3:4]], axis=-1)
        return result

    def _get_demo_frame_id(self, env_idx):
        try:
            frame_ids = self.env.demo_data_rh["frame_ids"][env_idx]
            step = self.env.progress_buf[env_idx].item()
            step = min(step, len(frame_ids) - 1)
            return frame_ids[step]
        except Exception:
            return None

    def _save_video(self, frames, local_path):
        import imageio
        video = torch.stack(frames)[..., :-1]  # RGBA -> RGB
        video = video.to(dtype=torch.uint8).permute(0, 3, 1, 2).detach().cpu().numpy()
        imageio.mimsave(local_path, video.transpose(0, 2, 3, 1), fps=10)
        print(f"Saved video: {local_path}")
        if wandb.run is not None:
            key = os.path.splitext(os.path.basename(local_path))[0]
            wandb.log({f"test_video/{key}": wandb.Video(local_path, fps=10, format="mp4")})

    def step(self, action):
        obs, reward, done, info = super().step(action)
        has_top = getattr(self.env, "camera_obs_top", None) is not None
        for i, idx in enumerate(self._rcd_idxs):
            frame_num = len(self._videos[i])
            demo_frame_id = self._get_demo_frame_id(idx)
            frame = self.env.camera_obs[idx].to(dtype=torch.uint8)
            frame_np = self._burn_frame_number(frame, frame_num, demo_frame_id)
            self._videos[i].append(torch.from_numpy(frame_np).to(frame.device))
            if has_top:
                frame_top = self.env.camera_obs_top[idx].to(dtype=torch.uint8)
                frame_top_np = self._burn_frame_number(frame_top, frame_num, demo_frame_id)
                self._videos_top[i].append(torch.from_numpy(frame_top_np).to(frame_top.device))
        if torch.any(done):
            n_done = done.sum().item()
            self._episodes_seen += n_done
            past_warmup = self._episodes_seen > self._warmup
            for i, idx in enumerate(self._rcd_idxs):
                if done[idx]:
                    frames = self._videos[i]
                    frames_top = self._videos_top[i]
                    self._videos[i] = []
                    self._videos_top[i] = []
                    if not past_warmup:
                        continue
                    succeeded = self.env.success_buf
                    failed = self.env.failure_buf
                    status = "timeout"
                    if succeeded[idx]:
                        status = "success"
                        self._n_successful_video_saved += 1
                    elif failed[idx]:
                        status = "failure"
                        self._n_failed_episodes += 1
                    else:
                        self._n_failed_episodes += 1
                    base = f"video-{self._n_video_saved}_{status}"
                    self._save_video(
                        frames,
                        os.path.join(self._local_video_dir, f"{base}.mp4"),
                    )
                    if has_top and frames_top:
                        self._save_video(
                            frames_top,
                            os.path.join(self._local_video_dir, f"{base}_top.mp4"),
                        )
                    self._n_video_saved += 1
                    if self._n_successful_video_saved >= self._n_successful_videos_to_record:
                        os._exit(0)
                    if self._n_failed_episodes >= self._max_failed_episodes:
                        print(f"Reached {self._max_failed_episodes} failed/timeout episodes, exiting.")
                        os._exit(0)
        return obs, reward, done, info


def _rotmat_to_euler(rotmat):
    """Convert (N,3,3) rotation matrices to (N,3) Euler angles (roll/pitch/yaw) in degrees."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(rotmat).as_euler("xyz", degrees=True)


def _quat_to_euler(quat_xyzw):
    """Convert (N,4) quaternions [x,y,z,w] to (N,3) Euler angles in degrees."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat(quat_xyzw).as_euler("xyz", degrees=True)


class TrajectoryPlotWrapper(gym.Wrapper):
    """Records ground-truth vs actual object poses for env 0 and saves a plot per episode."""

    def __init__(self, env, plot_dir: str = "traj_plots", n_episodes: int = 10):
        super().__init__(env)
        self._plot_dir = plot_dir
        self._n_episodes = n_episodes
        self._episode_count = 0
        self._data = {"rh_act_pos": [], "rh_act_rot": [], "lh_act_pos": [], "lh_act_rot": [],
                      "rh_tgt_pos": [], "rh_tgt_rot": [], "lh_tgt_pos": [], "lh_tgt_rot": [],
                      "rh_act_wrist_pos": [], "rh_act_wrist_rot": [],
                      "lh_act_wrist_pos": [], "lh_act_wrist_rot": [],
                      "rh_tgt_wrist_pos": [], "rh_tgt_wrist_rot": [],
                      "lh_tgt_wrist_pos": [], "lh_tgt_wrist_rot": []}
        os.makedirs(plot_dir, exist_ok=True)

    def _collect(self):
        raw = self.env
        idx = 0
        prog = raw.progress_buf[idx].item()
        rh_len = raw.demo_data_rh["obj_trajectory"].shape[1] - 1
        lh_len = raw.demo_data_lh["obj_trajectory"].shape[1] - 1

        self._data["rh_act_pos"].append(raw._manip_obj_rh_root_state[idx, :3].cpu().numpy().copy())
        self._data["lh_act_pos"].append(raw._manip_obj_lh_root_state[idx, :3].cpu().numpy().copy())
        self._data["rh_act_rot"].append(raw._manip_obj_rh_root_state[idx, 3:7].cpu().numpy().copy())  # xyzw
        self._data["lh_act_rot"].append(raw._manip_obj_lh_root_state[idx, 3:7].cpu().numpy().copy())

        rh_traj = raw.demo_data_rh["obj_trajectory"][idx, min(prog, rh_len)]
        lh_traj = raw.demo_data_lh["obj_trajectory"][idx, min(prog, lh_len)]
        self._data["rh_tgt_pos"].append(rh_traj[:3, 3].cpu().numpy().copy())
        self._data["lh_tgt_pos"].append(lh_traj[:3, 3].cpu().numpy().copy())
        self._data["rh_tgt_rot"].append(rh_traj[:3, :3].cpu().numpy().copy())  # rotmat
        self._data["lh_tgt_rot"].append(lh_traj[:3, :3].cpu().numpy().copy())

        rh_wrist_len = raw.demo_data_rh["wrist_pos"].shape[1] - 1
        lh_wrist_len = raw.demo_data_lh["wrist_pos"].shape[1] - 1
        self._data["rh_act_wrist_pos"].append(raw.rh_states["base_state"][idx, :3].cpu().numpy().copy())
        self._data["rh_act_wrist_rot"].append(raw.rh_states["base_state"][idx, 3:7].cpu().numpy().copy())  # xyzw
        self._data["lh_act_wrist_pos"].append(raw.lh_states["base_state"][idx, :3].cpu().numpy().copy())
        self._data["lh_act_wrist_rot"].append(raw.lh_states["base_state"][idx, 3:7].cpu().numpy().copy())
        self._data["rh_tgt_wrist_pos"].append(raw.demo_data_rh["wrist_pos"][idx, min(prog, rh_wrist_len)].cpu().numpy().copy())
        self._data["rh_tgt_wrist_rot"].append(raw.demo_data_rh["wrist_rot"][idx, min(prog, rh_wrist_len)].cpu().numpy().copy())  # axis-angle
        self._data["lh_tgt_wrist_pos"].append(raw.demo_data_lh["wrist_pos"][idx, min(prog, lh_wrist_len)].cpu().numpy().copy())
        self._data["lh_tgt_wrist_rot"].append(raw.demo_data_lh["wrist_rot"][idx, min(prog, lh_wrist_len)].cpu().numpy().copy())

    def _save_plot(self, status):
        from scipy.spatial.transform import Rotation as _R
        d = {k: np.array(v) for k, v in self._data.items()}
        t = np.arange(len(d["rh_act_pos"]))

        def _fix_quat_signs(quats):
            # q and -q represent the same rotation; flip sign when consecutive
            # quats diverge to keep the sequence continuous
            out = quats.copy()
            for i in range(1, len(out)):
                if np.dot(out[i], out[i - 1]) < 0:
                    out[i] = -out[i]
            return out

        def _rotvec_deg(rot):
            return np.degrees(np.unwrap(rot.as_rotvec(), axis=0))

        rh_act_rotvec = _rotvec_deg(_R.from_quat(_fix_quat_signs(d["rh_act_rot"])))
        lh_act_rotvec = _rotvec_deg(_R.from_quat(_fix_quat_signs(d["lh_act_rot"])))
        rh_tgt_rotvec = _rotvec_deg(_R.from_matrix(d["rh_tgt_rot"]))
        lh_tgt_rotvec = _rotvec_deg(_R.from_matrix(d["lh_tgt_rot"]))

        pos_labels  = ["x (m)", "y (m)", "z (m)"]
        rot_labels  = ["rx (°)", "ry (°)", "rz (°)"]
        colors = ["r", "g", "b"]

        fig, axes = plt.subplots(4, 3, figsize=(15, 16))
        fig.suptitle(f"Episode {self._episode_count} ({status}) — Object Position & Orientation")

        row_data = [
            ("RH pos", d["rh_tgt_pos"], d["rh_act_pos"], pos_labels),
            ("RH rot", rh_tgt_rotvec,   rh_act_rotvec,   rot_labels),
            ("LH pos", d["lh_tgt_pos"], d["lh_act_pos"], pos_labels),
            ("LH rot", lh_tgt_rotvec,   lh_act_rotvec,   rot_labels),
        ]

        for row, (title_prefix, tgt, act, ylabels) in enumerate(row_data):
            for col, (ylabel, color) in enumerate(zip(ylabels, colors)):
                ax = axes[row, col]
                ax.plot(t, tgt[:, col], "--", color=color, alpha=0.6, label="GT")
                ax.plot(t, act[:, col], "-",  color=color, label="Policy")
                ax.set_title(f"{title_prefix} {ylabel}")
                ax.set_xlabel("frame")
                ax.set_ylabel(ylabel)
                ax.legend(fontsize=7)
                ax.grid(True)

        plt.tight_layout()
        path = os.path.join(self._plot_dir, f"traj_ep{self._episode_count:03d}_{status}.png")
        plt.savefig(path, dpi=100)
        plt.close(fig)
        print(f"Saved trajectory plot: {path}")

        self._save_rotdiff_plot(d, t, status)
        self._save_objdist_plot(d, t, status)
        self._save_wrist_plot(d, t, status)

    def _save_objdist_plot(self, d, t, status):
        act_dist = np.linalg.norm(d["rh_act_pos"] - d["lh_act_pos"], axis=-1)
        tgt_dist = np.linalg.norm(d["rh_tgt_pos"] - d["lh_tgt_pos"], axis=-1)

        fig, ax = plt.subplots(figsize=(10, 4))
        fig.suptitle(f"Episode {self._episode_count} ({status}) — Inter-Object Distance")
        ax.plot(t, tgt_dist, "--", color="steelblue", alpha=0.7, label="GT")
        ax.plot(t, act_dist, "-",  color="steelblue", label="Policy")
        ax.set_xlabel("step")
        ax.set_ylabel("distance (m)")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        path = os.path.join(self._plot_dir, f"object_dist_ep{self._episode_count:03d}_{status}.png")
        plt.savefig(path, dpi=100)
        plt.close(fig)
        print(f"Saved object distance plot: {path}")

    def _save_rotdiff_plot(self, d, t, status):
        from scipy.spatial.transform import Rotation
        rh_act_R = Rotation.from_quat(d["rh_act_rot"])
        rh_tgt_R = Rotation.from_matrix(d["rh_tgt_rot"])
        rh_diff_deg = np.degrees((rh_tgt_R * rh_act_R.inv()).magnitude())

        lh_act_R = Rotation.from_quat(d["lh_act_rot"])
        lh_tgt_R = Rotation.from_matrix(d["lh_tgt_rot"])
        lh_diff_deg = np.degrees((lh_tgt_R * lh_act_R.inv()).magnitude())

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"Episode {self._episode_count} ({status}) — Rotation Error (geodesic)")
        for ax, diff, label in zip(axes, [rh_diff_deg, lh_diff_deg], ["RH", "LH"]):
            ax.plot(t, diff, color="purple")
            ax.set_title(f"{label} rotation error")
            ax.set_xlabel("frame")
            ax.set_ylabel("angle (°)")
            ax.grid(True)
        plt.tight_layout()
        path = os.path.join(self._plot_dir, f"rotdiff_ep{self._episode_count:03d}_{status}.png")
        plt.savefig(path, dpi=100)
        plt.close(fig)
        print(f"Saved rotation diff plot: {path}")

    def _save_wrist_plot(self, d, t, status):
        from scipy.spatial.transform import Rotation

        rh_act_wrist_R = Rotation.from_quat(np.array(d["rh_act_wrist_rot"]))
        lh_act_wrist_R = Rotation.from_quat(np.array(d["lh_act_wrist_rot"]))
        rh_tgt_wrist_R = Rotation.from_rotvec(np.array(d["rh_tgt_wrist_rot"]))
        lh_tgt_wrist_R = Rotation.from_rotvec(np.array(d["lh_tgt_wrist_rot"]))

        rh_wrist_rot_err = np.degrees((rh_tgt_wrist_R * rh_act_wrist_R.inv()).magnitude())
        lh_wrist_rot_err = np.degrees((lh_tgt_wrist_R * lh_act_wrist_R.inv()).magnitude())

        # Use rotvec (axis-angle) components for both to avoid Euler angle ambiguity
        # near ±180° where decomposition is non-unique and can produce apparent sign flips.
        rh_act_wrist_rotvec = np.degrees(rh_act_wrist_R.as_rotvec())
        lh_act_wrist_rotvec = np.degrees(lh_act_wrist_R.as_rotvec())
        rh_tgt_wrist_rotvec = np.degrees(rh_tgt_wrist_R.as_rotvec())
        lh_tgt_wrist_rotvec = np.degrees(lh_tgt_wrist_R.as_rotvec())

        rh_act_wrist_pos = np.array(d["rh_act_wrist_pos"])
        lh_act_wrist_pos = np.array(d["lh_act_wrist_pos"])
        rh_tgt_wrist_pos = np.array(d["rh_tgt_wrist_pos"])
        lh_tgt_wrist_pos = np.array(d["lh_tgt_wrist_pos"])

        pos_labels = ["x (m)", "y (m)", "z (m)"]
        rot_labels = ["rx (°)", "ry (°)", "rz (°)"]
        colors = ["r", "g", "b"]

        fig, axes = plt.subplots(4, 3, figsize=(15, 16))
        fig.suptitle(f"Episode {self._episode_count} ({status}) — Wrist Position & Orientation")

        row_data = [
            ("RH wrist pos", rh_tgt_wrist_pos, rh_act_wrist_pos, pos_labels),
            ("RH wrist rot", rh_tgt_wrist_rotvec, rh_act_wrist_rotvec, rot_labels),
            ("LH wrist pos", lh_tgt_wrist_pos, lh_act_wrist_pos, pos_labels),
            ("LH wrist rot", lh_tgt_wrist_rotvec, lh_act_wrist_rotvec, rot_labels),
        ]
        for row, (title_prefix, tgt, act, ylabels) in enumerate(row_data):
            for col, (ylabel, color) in enumerate(zip(ylabels, colors)):
                ax = axes[row, col]
                ax.plot(t, tgt[:, col], "--", color=color, alpha=0.6, label="GT")
                ax.plot(t, act[:, col], "-", color=color, label="Policy")
                ax.set_title(f"{title_prefix} {ylabel}")
                ax.set_xlabel("frame")
                ax.set_ylabel(ylabel)
                ax.legend(fontsize=7)
                ax.grid(True)
        plt.tight_layout()
        path = os.path.join(self._plot_dir, f"wrist_traj_ep{self._episode_count:03d}_{status}.png")
        plt.savefig(path, dpi=100)
        plt.close(fig)
        print(f"Saved wrist trajectory plot: {path}")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"Episode {self._episode_count} ({status}) — Wrist Rotation Error (geodesic)")
        for ax, err, label in zip(axes, [rh_wrist_rot_err, lh_wrist_rot_err], ["RH", "LH"]):
            ax.plot(t, err, color="darkorange")
            ax.set_title(f"{label} wrist rotation error")
            ax.set_xlabel("frame")
            ax.set_ylabel("angle (°)")
            ax.grid(True)
        plt.tight_layout()
        path = os.path.join(self._plot_dir, f"wrist_rotdiff_ep{self._episode_count:03d}_{status}.png")
        plt.savefig(path, dpi=100)
        plt.close(fig)
        print(f"Saved wrist rotation diff plot: {path}")

    def _reset_buffers(self):
        for k in self._data:
            self._data[k] = []

    def reset(self, **kwargs):
        self._reset_buffers()
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self._collect()
        if done[0]:
            succeeded = self.env.success_buf[0].item()
            self._save_plot("success" if succeeded else "failure")
            self._episode_count += 1
            self._reset_buffers()
            if self._episode_count >= self._n_episodes:
                os._exit(0)
        return obs, reward, done, info
