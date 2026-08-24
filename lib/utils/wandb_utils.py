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

# Panel layout for the eval-threshold dry-run plot, and the thresholds it draws, owned together so
# the live plot and data_stats/summarize_dry_run.py cannot drift apart.
from lib.utils.eval_thresholds import (
    EVAL_DRY_RUN_PANELS,
    find_threshold_trip,
    threshold_for,
    unit_for,
)

# Eval rollout videos record one frame per env step, and the env steps at 60 Hz (dt = 1/60 when
# demoTargetFps is null, see main/cfg/task/ResDexHand.yaml:142), so 60 fps is 1x playback.
# Encode at 30 fps -> 0.5x, slow enough to judge the capping contact. Frame content is untouched;
# only the encode rate changes, so a video runs 2x longer than the episode it shows.
EVAL_VIDEO_FPS = 30


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


# Playback speed for captured videos, as a divisor on the control rate: 1 = real time,
# 2 = half speed. One frame is recorded per control step, so encoding at control_rate /
# VIDEO_SLOWDOWN gives exactly that. Raise it to study contact frame by frame.
VIDEO_SLOWDOWN = 2


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
        self._videos_overhead = [[] for _ in range(n_parallel_recorders)]
        # One list of per-step metric rows per recorder, in lockstep with self._videos, filled only
        # while the env publishes eval_dry_run (evalThresholdDryRun=true). Cleared at the same
        # points as the frame buffers so row k always describes the frame burned "step k".
        self._eval_metrics = [[] for _ in range(n_parallel_recorders)]
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
        self._videos_overhead = [[] for _ in range(self._n_recorders)]
        self._eval_metrics = [[] for _ in range(self._n_recorders)]
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

    def video_fps(self):
        """Encode rate that plays the capture back at VIDEO_SLOWDOWN times slower than real time.

        One frame is appended per control step, so real time is the control rate itself
        (1 / (dt * control_freq_inv), 60 Hz by default). The previous hardcoded 10 fps therefore
        played every recording 6x slower than the run actually happened.

        Returns:
            int frames per second for the encoder; 30 if the env does not expose a control rate.
        """
        try:
            control_rate = 1.0 / (self.env.dt * self.env.control_freq_inv)
        except AttributeError:
            control_rate = 60.0
        return max(1, round(control_rate / VIDEO_SLOWDOWN))

    def _save_video(self, frames, local_path):
        """Encode one rollout's captured frames to mp4 and mirror it to wandb.

        Args:
            frames: list of (H, W, 4) uint8 RGBA frames, one per env step of the episode.
            local_path: Destination .mp4 path; its stem becomes the wandb key.

        Returns:
            None. Writes local_path and, when a wandb run is active, logs the video.
        """
        import imageio
        video = torch.stack(frames)[..., :-1]  # RGBA -> RGB
        video = video.to(dtype=torch.uint8).permute(0, 3, 1, 2).detach().cpu().numpy()
        fps = self.video_fps()
        imageio.mimsave(local_path, video.transpose(0, 2, 3, 1), fps=fps)
        print(f"Saved video: {local_path} ({fps} fps, {1/VIDEO_SLOWDOWN:g}x real time)")
        if wandb.run is not None:
            key = os.path.splitext(os.path.basename(local_path))[0]
            wandb.log({f"test_video/{key}": wandb.Video(local_path, fps=fps, format="mp4")})

    def collect_eval_metrics(self):
        """Append this step's dry-run error terms for every recorded env, one row per recorder.

        No-op unless the env publishes eval_dry_run (evalThresholdDryRun=true), so an ordinary
        recording session pays nothing. Every metric is pulled across in a single device->host
        transfer: a per-metric .item() would force one sync per term per recorder per step.

        Returns:
            None. Appends one dict of floats to each self._eval_metrics[i].
        """
        dry_run = getattr(self.env, "eval_dry_run", None)
        if not dry_run:
            return
        names = sorted(dry_run["metrics"])
        stacked = torch.stack([dry_run["metrics"][n] for n in names], dim=0)  # (n_metrics, n_envs)
        values = stacked[:, self._rcd_idxs].detach().cpu().numpy()
        progress = dry_run["running_progress"][self._rcd_idxs].detach().cpu().numpy()
        for i in range(self._n_recorders):
            row = dict(zip(names, values[:, i].tolist()))
            # steps since this env's last reset, the quantity the eval branch's warmup gate reads
            row["running_progress"] = int(progress[i])
            self._eval_metrics[i].append(row)

    def report_dry_run(self, rows, base, env_idx, status):
        """Say where the eval thresholds would have quit this episode, and by how much.

        The eval branch of compute_imitation_reward no longer terminates on these thresholds, so an
        episode plays to the end of the demo (or to a velocity blow-up) regardless. This is the
        verdict it would have received, keyed to the video of the same name. Written to
        <base>_metrics.txt as well as printed: SLURM .out files on this cluster go stale mid-job,
        and the verdict is the one thing that cannot be recovered from the mp4.

        Args:
            rows: list of per-step metric dicts for this episode.
            base: video stem this episode was saved under, e.g. "video-3_failure".
            env_idx: which env of the vectorised batch this episode ran in.
            status: how the episode actually ended — success / failure / timeout.

        Returns:
            None. Prints the verdict and writes it next to the video.
        """
        step, terms = find_threshold_trip(rows)
        head = f"[DRY-RUN {base} env={env_idx}]"
        if step is None:
            dry_run = getattr(self.env, "eval_dry_run", None) or {}
            warmup = dry_run.get("warmup_steps", 0)
            # Closest approach, so "never tripped" still says how much headroom was left. Measured
            # over the same rows find_dry_run_trip looks at: the settle window reads large on every
            # term, and reporting a margin from steps the threshold ignores would be nonsense.
            scored = [row for row in rows if row["running_progress"] >= warmup]
            worst_name, worst_ratio = "", 0.0
            for name in sorted(rows[0]):
                threshold = threshold_for(name)
                if threshold is None or not scored:
                    continue
                ratio = max(row[name] for row in scored) / threshold
                if ratio > worst_ratio:
                    worst_name, worst_ratio = name, ratio
            margin = ""
            if worst_name:
                margin = f", closest {worst_name} at {worst_ratio * 100:.0f}% of its limit"
            lines = [f"{head} thresholds never tripped over {len(rows)} steps{margin}"]
        else:
            demo_frame = ""
            try:
                frame_ids = self.env.demo_data_rh["frame_ids"][env_idx]
                demo_frame = f", demo frame {frame_ids[min(step, len(frame_ids) - 1)]}"
            except Exception:
                pass
            lines = [
                f"{head} would have quit at step {step} of {len(rows)}{demo_frame}; "
                f"episode actually ran on and was scored '{status}'"
            ]
            for name, value, threshold in terms:
                unit = unit_for(name)
                lines.append(f"    {name} = {value:.4f} {unit} (> {threshold:.4f} {unit})")
        for line in lines:
            print(line)
        with open(os.path.join(self._local_video_dir, f"{base}_metrics.txt"), "w") as handle:
            handle.write("\n".join(lines) + "\n")

    def save_eval_metrics(self, rows, base, env_idx, status):
        """Write this episode's dry-run error series next to its video, as a CSV and a plot.

        Args:
            rows: list of per-step metric dicts for this episode.
            base: video stem to match, e.g. "video-3_failure"; outputs are <base>_metrics.csv/.png.
            env_idx: which env of the vectorised batch this episode ran in.
            status: how the episode actually ended — success / failure / timeout.

        Returns:
            None. Writes two files into self._local_video_dir.
        """
        if not rows:
            return
        dry_run = getattr(self.env, "eval_dry_run", None) or {}
        thresholds = dry_run.get("thresholds", {})
        warmup = dry_run.get("warmup_steps", 0)
        names = [n for n in sorted(rows[0]) if n != "running_progress"]
        steps = np.arange(len(rows))

        csv_path = os.path.join(self._local_video_dir, f"{base}_metrics.csv")
        with open(csv_path, "w") as handle:
            handle.write(",".join(["step", "running_progress"] + names) + "\n")
            for step, row in enumerate(rows):
                cells = [str(step), str(row["running_progress"])] + [f"{row[n]:.6g}" for n in names]
                handle.write(",".join(cells) + "\n")

        trip_step, trip_terms = find_threshold_trip(rows)
        # 2x3 row-major, so the left two columns hold the four scored terms and the right column
        # holds the two wrist diagnostics -- see EVAL_DRY_RUN_PANELS for why the wrist is there.
        fig, axes = plt.subplots(2, 3, figsize=(18, 8))
        verdict = "thresholds never tripped"
        if trip_step is not None:
            verdict = f"threshold trip at step {trip_step}"
        fig.suptitle(f"{base} (env {env_idx}, ended '{status}') — Eval Threshold Dry Run, {verdict}")
        for ax, (key, title, ylabel, _) in zip(axes.flat, EVAL_DRY_RUN_PANELS):
            for side, color in (("rh", "tab:blue"), ("lh", "tab:red")):
                name = f"{side}_{key}"
                if name in names:
                    series = [row[name] for row in rows]
                    ax.plot(steps, series, "-", color=color, label=side.upper())
            if key in thresholds:
                ax.axhline(
                    thresholds[key], color="k", ls="--", lw=1, label=f"limit {thresholds[key]:g}"
                )
            else:
                # a diagnostic panel: say so, rather than leave the reader hunting for a limit line
                ax.set_facecolor("#fafafa")
            if warmup:
                # the eval branch ignores everything left of this, so shade it rather than crop it
                ax.axvspan(0, min(warmup, len(rows)), color="grey", alpha=0.15)
            if trip_step is not None:
                ax.axvline(trip_step, color="tab:orange", lw=1.5)
            suffix = "" if key in thresholds else ", not scored"
            ax.set_title(f"{title} ({key}{suffix})", fontsize=9)
            ax.set_xlabel("Step")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        png_path = os.path.join(self._local_video_dir, f"{base}_metrics.png")
        plt.savefig(png_path, dpi=110)
        plt.close(fig)
        print(f"Saved eval metrics: {png_path} and {os.path.basename(csv_path)}")
        if wandb.run is not None:
            wandb.log({f"test_metrics/{base}": wandb.Image(png_path)})

    def step(self, action):
        obs, reward, done, info = super().step(action)
        has_top = getattr(self.env, "camera_obs_top", None) is not None
        has_overhead = getattr(self.env, "camera_obs_overhead", None) is not None
        self.collect_eval_metrics()
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
            if has_overhead:
                frame_oh = self.env.camera_obs_overhead[idx].to(dtype=torch.uint8)
                frame_oh_np = self._burn_frame_number(frame_oh, frame_num, demo_frame_id)
                self._videos_overhead[i].append(torch.from_numpy(frame_oh_np).to(frame_oh.device))
        if torch.any(done):
            n_done = done.sum().item()
            self._episodes_seen += n_done
            past_warmup = self._episodes_seen > self._warmup
            for i, idx in enumerate(self._rcd_idxs):
                if done[idx]:
                    frames = self._videos[i]
                    frames_top = self._videos_top[i]
                    frames_overhead = self._videos_overhead[i]
                    self._videos[i] = []
                    self._videos_top[i] = []
                    self._videos_overhead[i] = []
                    metric_rows = self._eval_metrics[i]
                    self._eval_metrics[i] = []
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
                    # Every file names the view it actually contains. The unsuffixed file used to be
                    # the front camera and `_top` used to be the BEHIND camera, which meant the one
                    # thing no file held was a top-down view.
                    base = f"video-{self._n_video_saved}_{status}"
                    self._save_video(
                        frames,
                        os.path.join(self._local_video_dir, f"{base}_front.mp4"),
                    )
                    if has_top and frames_top:
                        self._save_video(
                            frames_top,
                            os.path.join(self._local_video_dir, f"{base}_behind.mp4"),
                        )
                    if has_overhead and frames_overhead:
                        self._save_video(
                            frames_overhead,
                            os.path.join(self._local_video_dir, f"{base}_top.mp4"),
                        )
                    if metric_rows:
                        self.report_dry_run(metric_rows, base, idx, status)
                        self.save_eval_metrics(metric_rows, base, idx, status)
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
