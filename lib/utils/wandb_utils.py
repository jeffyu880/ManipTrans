import os
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

        # this can fail occasionally, so we try a couple more times
        @retry(3, exceptions=(Exception,))
        def init_wandb():
            wandb.tensorboard.patch(root_logdir="runs")
            wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                group=cfg.wandb_group,
                tags=cfg.wandb_tags,
                sync_tensorboard=True,
                id=wandb_unique_id,
                name=experiment_name,
                resume=True,
                settings=wandb.Settings(start_method="fork"),
            )

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
        self._n_video_saved = 0
        self._n_successful_video_saved = 0
        self._n_successful_videos_to_record = n_successful_videos_to_record
        self._n_failed_episodes = 0
        self._max_failed_episodes = 10
        self._local_video_dir = local_video_dir
        os.makedirs(local_video_dir, exist_ok=True)

    def reset(self, **kwargs):
        self._videos = [[] for _ in range(self._n_recorders)]
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        for i, idx in enumerate(self._rcd_idxs):
            self._videos[i].append(self.env.camera_obs[idx].clone())
        if torch.any(done):
            for i, idx in enumerate(self._rcd_idxs):
                if done[idx]:
                    # if len(self._videos[i]) <= 20:
                    #     self._videos[i] = []
                    #     continue
                    video = torch.stack(self._videos[i])[..., :-1]  # (T, H, W, C), RGBA -> RGB
                    video = video.to(dtype=torch.uint8)
                    video = video.permute(0, 3, 1, 2).detach().cpu().numpy()  # (T, C, H, W)
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
                        self._n_failed_episodes += 1  # timeout also counts
                    local_path = os.path.join(
                        self._local_video_dir,
                        f"video-{self._n_video_saved}_{status}.mp4",
                    )
                    import imageio
                    imageio.mimsave(local_path, video.transpose(0, 2, 3, 1), fps=10)
                    print(f"Saved video: {local_path}")
                    if wandb.run is not None:
                        wandb.log({f"test_video/video-{self._n_video_saved}_{status}": wandb.Video(local_path, fps=10, format="mp4")})
                    self._n_video_saved += 1
                    self._videos[i] = []
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
                      "rh_tgt_pos": [], "rh_tgt_rot": [], "lh_tgt_pos": [], "lh_tgt_rot": []}
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

    def _save_plot(self, status):
        d = {k: np.array(v) for k, v in self._data.items()}
        t = np.arange(len(d["rh_act_pos"]))

        rh_act_euler = _quat_to_euler(d["rh_act_rot"])
        lh_act_euler = _quat_to_euler(d["lh_act_rot"])
        rh_tgt_euler = _rotmat_to_euler(d["rh_tgt_rot"])
        lh_tgt_euler = _rotmat_to_euler(d["lh_tgt_rot"])

        pos_labels  = ["x (m)", "y (m)", "z (m)"]
        rot_labels  = ["yaw (°)", "pitch (°)", "roll (°)"]
        colors = ["r", "g", "b"]

        fig, axes = plt.subplots(4, 3, figsize=(15, 16))
        fig.suptitle(f"Episode {self._episode_count} ({status}) — Object Position & Orientation")

        row_data = [
            ("RH pos", d["rh_tgt_pos"], d["rh_act_pos"], pos_labels),
            ("RH rot", rh_tgt_euler,    rh_act_euler,    rot_labels),
            ("LH pos", d["lh_tgt_pos"], d["lh_act_pos"], pos_labels),
            ("LH rot", lh_tgt_euler,    lh_act_euler,    rot_labels),
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
