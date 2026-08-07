# train.py
# Script to train policies in Isaac Gym
#
# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# pinocchio (which dex-retargeting solves through) registers its boost-python bindings at import,
# and isaacgym's own C++ libs shadow the symbols they bind against. Imported AFTER isaacgym it
# still loads, but every call into it dies with a bare "No Python class registered for C++ class
# std::vector<std::string>". Importing it here — first, before anything pulls in isaacgym — is the
# whole fix; it does not import torch, so it does not violate isaacgym's own ordering rule.
# Optional dependency: absent just means dexRetBaseline is unavailable, which the env asserts on.
try:
    import pinocchio  # noqa: F401
except ImportError:
    pass

import hydra

from omegaconf import DictConfig, OmegaConf
import lib


def preprocess_train_config(cfg, config_dict):
    """
    Adding common configuration parameters to the rl_games train config.
    An alternative to this is inferring them in task-specific .yaml files, but that requires repeating the same
    variable interpolations in each config.
    """

    train_cfg = config_dict["params"]["config"]

    train_cfg["device"] = cfg.rl_device

    train_cfg["population_based_training"] = False
    train_cfg["pbt_idx"] = None

    train_cfg["full_experiment_name"] = cfg.get("full_experiment_name")

    print(f"Using rl_device: {cfg.rl_device}")
    print(f"Using sim_device: {cfg.sim_device}")
    print(train_cfg)

    try:
        model_size_multiplier = config_dict["params"]["network"]["mlp"]["model_size_multiplier"]
        if model_size_multiplier != 1:
            units = config_dict["params"]["network"]["mlp"]["units"]
            for i, u in enumerate(units):
                units[i] = u * model_size_multiplier
            print(
                f'Modified MLP units by x{model_size_multiplier} to {config_dict["params"]["network"]["mlp"]["units"]}'
            )
    except KeyError:
        pass

    return config_dict


@hydra.main(version_base="1.1", config_name="config", config_path="../cfg")
def launch_rlg_hydra(cfg: DictConfig):
    import os
    from datetime import datetime

    import isaacgym
    from hydra.utils import to_absolute_path
    OmegaConf.set_struct(cfg, False)

    if cfg.display:
        import cv2
        import numpy as np

        cv2.imshow("dummy", np.zeros((1, 1, 3), dtype=np.uint8))
        cv2.waitKey(1)

    import maniptrans_envs
    from lib.utils.reformat import omegaconf_to_dict, print_dict
    from lib.utils.utils import set_np_formatting, set_seed

    from lib.utils.rlgames_utils import (
        RLGPUAlgoObserver,
        MultiObserver,
        ComplexObsRLGPUEnv,
    )
    from lib.utils.wandb_utils import WandbAlgoObserver
    from rl_games.common import env_configurations, vecenv
    from lib.rl.runner import Runner
    from lib.rl.network_builder import DictObsBuilder
    from lib.rl.sep_network_builder import SepDictObsBuilder
    from lib.rl.models import ModelA2CContinuousLogStd, SepModelA2CContinuousLogStd
    from rl_games.algos_torch.model_builder import register_network, register_model
    from lib.utils.wandb_utils import WandbVideoCaptureWrapper, TrajectoryPlotWrapper
    from lib.rl.network_builder_residual_sh import ResRHDictObsBuilder, ResLHDictObsBuilder
    from lib.rl.network_builder_residual_bih import ResBiHDictObsBuilder
    from lib.rl.res_models import (
        ModelA2CContinuousLogStdResRH,
        ModelA2CContinuousLogStdResLH,
        ModelA2CContinuousLogStdResBiH,
    )

    register_model("my_continuous_a2c_logstd", ModelA2CContinuousLogStd)
    register_network("dict_obs_actor_critic", DictObsBuilder)
    register_network("sep_dict_obs_actor_critic", SepDictObsBuilder)
    register_model("sep_my_continuous_a2c_logstd", SepModelA2CContinuousLogStd)

    register_network("res_rh_dict_obs_actor_critic", ResRHDictObsBuilder)
    register_model("res_rh_my_continuous_a2c_logstd", ModelA2CContinuousLogStdResRH)
    register_network("res_lh_dict_obs_actor_critic", ResLHDictObsBuilder)
    register_model("res_lh_my_continuous_a2c_logstd", ModelA2CContinuousLogStdResLH)
    register_network("res_bih_dict_obs_actor_critic", ResBiHDictObsBuilder)
    register_model("res_bih_my_continuous_a2c_logstd", ModelA2CContinuousLogStdResBiH)

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        if type(cfg.checkpoint) == str:
            cfg.checkpoint = to_absolute_path(cfg.checkpoint)
        elif type(cfg.checkpoint) == list:
            cfg.checkpoint = [to_absolute_path(cp) for cp in cfg.checkpoint]

    # MANIPTRANS_GRIP_CSV (the grip logger in dexhandmanip_bih.py): a bare filename, or "auto",
    # is resolved to runs/<exp>/grip_logs/ next to the checkpoint that produced it — the same
    # convention as the videos below. A value with a directory in it is left untouched.
    grip_csv = os.environ.get("MANIPTRANS_GRIP_CSV", "")
    if grip_csv and os.path.dirname(grip_csv) == "":
        if grip_csv in ("auto", "1"):
            demo_str = "-".join(str(d) for d in cfg.task.env.get("dataIndices", [])) or "all"
            arm = "imitator_only" if cfg.task.env.get("zeroResidual", False) else "full_model"
            grip_csv = f"grip_{arm}__demo_{demo_str}.csv"
        ckpt = cfg.checkpoint[0] if type(cfg.checkpoint) == list else cfg.checkpoint
        if ckpt:  # no checkpoint (training) leaves the CSV in the cwd
            grip_dir = os.path.join(os.path.dirname(os.path.dirname(ckpt)), "grip_logs")
            os.makedirs(grip_dir, exist_ok=True)
            grip_csv = os.path.join(grip_dir, grip_csv)
        os.environ["MANIPTRANS_GRIP_CSV"] = grip_csv
        print(f"[grip] logging to {grip_csv}")

    # The live thumb-index gap logger (_log_pinch_gap) is always on in live mode; give it a
    # timestamped path under runs/<exp>/pinch_logs/ so back-to-back live sessions don't overwrite
    # each other. MANIPTRANS_PINCH_CSV with a directory in it is used verbatim.
    if cfg.task.env.get("live", False):
        pinch_csv = os.environ.get("MANIPTRANS_PINCH_CSV", "")
        if os.path.dirname(pinch_csv) == "":
            import time as _time

            pinch_csv = pinch_csv or f"pinch_gap__{_time.strftime('%m-%d-%H-%M-%S')}.csv"
            ckpt = cfg.checkpoint[0] if type(cfg.checkpoint) == list else cfg.checkpoint
            if ckpt:  # no checkpoint leaves the CSV in the cwd
                pinch_dir = os.path.join(os.path.dirname(os.path.dirname(ckpt)), "pinch_logs")
                os.makedirs(pinch_dir, exist_ok=True)
                pinch_csv = os.path.join(pinch_dir, pinch_csv)
            os.environ["MANIPTRANS_PINCH_CSV"] = pinch_csv

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # set numpy formatting for printing only
    set_np_formatting()

    # global rank of the GPU
    global_rank = int(os.getenv("RANK", "0"))

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank)

    def create_isaacgym_env():
        kwargs = dict(
            sim_device=cfg.sim_device,
            rl_device=cfg.rl_device,
            graphics_device_id=cfg.graphics_device_id,
            multi_gpu=cfg.multi_gpu,
            cfg=cfg.task,
            display=cfg.display,
            record=cfg.capture_video,
            has_headless_arg=False,
        )
        if not cfg.headless:
            assert "pcd" not in cfg.task_name.lower(), "TODO: add GUI support for PCD tasks"
        if "pcd" not in cfg.task_name.lower():
            kwargs["headless"] = cfg.headless
            kwargs["has_headless_arg"] = True
        envs = maniptrans_envs.lib.make(**kwargs)
        if cfg.plot_trajectories:
            data_indices = cfg.task.env.get("dataIndices", [])
            indices_tag = "+".join(str(d) for d in data_indices) if data_indices else "all"
            base_plot_dir = os.path.join(os.path.dirname(os.path.dirname(cfg.checkpoint)), "traj_plots") if cfg.checkpoint else os.path.join(experiment_dir, "traj_plots")
            plot_dir = os.path.join(base_plot_dir, indices_tag)
            envs = TrajectoryPlotWrapper(envs, plot_dir=plot_dir, n_episodes=cfg.n_traj_episodes)
        if cfg.capture_video:
            envs.is_vector_env = True
            data_indices = cfg.task.env.get("dataIndices", [])
            indices_tag = "+".join(str(d) for d in data_indices) if data_indices else "all"
            # Save videos next to the checkpoint: runs/<exp>/videos/epoch_<N>/<indices>
            if cfg.checkpoint:
                # Group videos by the training epoch so each evaluated checkpoint gets its
                # own folder. The epoch is usually in the filename (e.g. ..._ep_1100_...);
                # fall back to the epoch stored inside the .pth if it isn't.
                import re
                m = re.search(r"_ep_(\d+)", os.path.basename(cfg.checkpoint))
                if m:
                    epoch = m.group(1)
                else:
                    try:
                        epoch = torch.load(cfg.checkpoint, map_location="cpu").get("epoch", "unknown")
                    except Exception as exc:
                        print(f"Could not read epoch from checkpoint {cfg.checkpoint}: {exc}")
                        epoch = "unknown"
                video_dir = os.path.join(
                    os.path.dirname(os.path.dirname(cfg.checkpoint)), "videos", indices_tag, f"epoch_{epoch}"
                )
            else:
                video_dir = os.path.join(experiment_dir, "videos", indices_tag)
            envs = WandbVideoCaptureWrapper(
                envs,
                n_parallel_recorders=cfg.n_parallel_recorders,
                n_successful_videos_to_record=cfg.n_successful_videos_to_record,
                local_video_dir=video_dir,
            )
        return envs

    env_configurations.register(
        "rlgpu",
        {
            "vecenv_type": "RLGPU",
            "env_creator": create_isaacgym_env,
        },
    )

    obs_spec = {}
    if "central_value_config" in cfg.rl_train.params.config:
        critic_net_cfg = cfg.rl_train.params.config.central_value_config.network
        obs_spec["states"] = {
            "names": list(critic_net_cfg.inputs.keys()),
            "concat": not critic_net_cfg.name == "complex_net",
            "space_name": "state_space",
        }

    vecenv.register("RLGPU", lambda config_name, num_actors: ComplexObsRLGPUEnv(config_name))

    rlg_config_dict = omegaconf_to_dict(cfg.rl_train)
    rlg_config_dict = preprocess_train_config(cfg, rlg_config_dict)
    # rlg_config_dict["params"]["config"]["minibatch_size"] = int((cfg.num_envs/4096) * 1024)

    observers = [RLGPUAlgoObserver()]

    if cfg.wandb_activate:
        cfg.seed += global_rank
        if global_rank == 0:
            # initialize wandb only once per multi-gpu run
            wandb_observer = WandbAlgoObserver(cfg)
            observers.append(wandb_observer)

    def build_runner(algo_observer):
        runner = Runner(algo_observer)
        return runner

    # convert CLI arguments into dictionary
    # create runner and set the settings
    runner = build_runner(MultiObserver(observers))
    runner.load(rlg_config_dict)
    runner.reset()

    # dump config dict
    if cfg.test:
        prefix = "dump__" if cfg.save_rollouts else "test__"
        ckpt_stem = ""
        if cfg.checkpoint:
            ckpt_stem = os.path.splitext(os.path.basename(cfg.checkpoint.replace("\\", "/")))[0]
        demo_str = "-".join(str(d) for d in cfg.task.env.get("dataIndices", []))
        demo_part = f"__demo_{demo_str}" if demo_str else ""
        experiment_dir = os.path.join(
            "dumps",
            prefix + ckpt_stem + demo_part + "__{date:%m-%d-%H-%M-%S}".format(date=datetime.now()),
        )
    else:
        experiment_dir = os.path.join(
            "runs",
            cfg.rl_train.params.config.name  # + "__dataIdx_"
            # + "-".join(str(k) for k in cfg.dataIndices)
            + "__" + "{date:%m-%d-%H-%M-%S}".format(date=datetime.now()),
        )
        cfg.rl_train.params.config.full_experiment_name = experiment_dir.replace("runs/", "")
        runner.params["config"]["full_experiment_name"] = experiment_dir.replace("runs/", "")
    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    data_indices = cfg.task.env.get("dataIndices", [])
    with open(os.path.join(experiment_dir, "demos.txt"), "w") as f:
        f.write("\n".join(str(d) for d in data_indices))

    # dexRetBaseline replaces the action inside pre_physics_step before anything reads it, so the
    # policy contributes nothing. Routing it through rl_games anyway would build the residual
    # network and load both frozen imitator checkpoints purely to discard their output, so step the
    # env directly instead: the retargeter is the only thing moving the hand. Same env, same
    # wrappers, same rewards, termination and pinch logging — only the action source differs.
    if cfg.test and cfg.task.env.get("dexRetBaseline", False):
        import torch

        # A bare MANIPTRANS_DEXRET_LOG filename resolves into this run's output dir, so the wrist
        # trace and its plot land beside config.yaml and demos.txt instead of the cwd. Same
        # convention MANIPTRANS_PINCH_CSV follows. A value with a directory is used verbatim.
        wrist_log = os.environ.get("MANIPTRANS_DEXRET_LOG", "")
        if wrist_log and not os.path.dirname(wrist_log):
            os.environ["MANIPTRANS_DEXRET_LOG"] = os.path.join(experiment_dir, wrist_log)

        envs = create_isaacgym_env()
        num_envs = envs.num_envs
        # pre_physics_step overwrites this wholesale; it exists only so vec_task.step has something
        # of the right width to clamp and stash in self.actions.
        idle = torch.zeros((num_envs, envs.num_actions), device=cfg.rl_device)
        envs.reset()

        # Live teleop never auto-resets — dexhandmanip_bih.py:3258 forces reset_buf to 0 so the
        # stream runs continuously — so `done` never fires and there are no episodes to count.
        # Run until the viewer is closed (vec_task.render exits the process) or Ctrl-C, instead of
        # spinning toward an episode budget that can never be reached.
        live = bool(cfg.task.env.get("live", False))
        # The video/plot wrappers are gym.Wrapper, so the task itself is behind .unwrapped; the
        # controller is built lazily on the first pre_physics_step, hence the getattr in the loop
        # rather than a lookup here.
        unwrapped = getattr(envs, "unwrapped", envs)
        calibrating = live and bool(cfg.task.env.get("dexRetCalibrate", False))
        episodes, steps = 0, 0
        max_steps = cfg.task.env.get("episodeLength", 1200) * cfg.num_rollouts_to_run + 10
        # Wall-clock rate of the WHOLE loop, not just the controller. The controller reports its
        # own slice, but what decides whether live teleop holds 60 Hz is this: physics substeps,
        # observation assembly, any recording, and the retargeting solve together. Timed from the
        # second step so the first one's lazy controller construction does not skew it.
        import time as _time
        loop_started = None
        try:
            while live or (episodes < cfg.num_rollouts_to_run and steps < max_steps):
                # vec_task.reset() only hands back the first observations — it does NOT place
                # anything (see its docstring). reset_idx runs from reset_done(), which the
                # rl_games player calls at the top of every iteration. Skip it and the first
                # control step sees the hand at its spawn pose, half a metre off the demo and
                # already moving, which saturates the wrist command on step 1.
                envs.reset_done()
                _, _, done, _ = envs.step(idle)
                steps += 1
                if loop_started is None:
                    loop_started = _time.perf_counter()

                # A calibration run exists only to capture the wrist-fit constant, and the
                # controller writes it the moment both hands have enough samples. Stopping here
                # rather than making the operator judge when to Ctrl-C is the difference between an
                # explicit calibration step and a warm-up they have to guess the end of.
                controller = getattr(unwrapped, "dexret_controller", None)
                if calibrating and controller is not None and controller.calibration_complete():
                    print("[dexret] calibration captured; stopping. Re-run without "
                          "dexRetCalibrate=true to teleoperate.")
                    break

                # Live never terminates, so a rate printed at exit is a rate you only learn after
                # the session. Report it as it goes instead: whether this holds 60 Hz is the whole
                # question when trying per_frame fitting on a live stream.
                if live and steps % 120 == 0 and loop_started is not None and steps > 1:
                    per_step = 1e3 * (_time.perf_counter() - loop_started) / (steps - 1)
                    hz = 1e3 / per_step
                    colour = "\033[92m" if hz >= 58 else ("\033[93m" if hz >= 45 else "\033[91m")
                    print(f"{colour}[dexret] {hz:5.1f} Hz ({per_step:5.2f} ms/step) "
                          f"over {steps} steps\033[0m")

                finished = int(done.sum().item())
                if finished:
                    episodes += finished
                    print(f"[dexret] {episodes}/{cfg.num_rollouts_to_run} episodes, {steps} steps")
        except KeyboardInterrupt:
            print(f"\n[dexret] interrupted after {steps} steps")
        if loop_started is not None and steps > 1:
            elapsed = _time.perf_counter() - loop_started
            per_step = 1e3 * elapsed / (steps - 1)
            budget = 1e3 * cfg.task.env.get("dt", 1 / 120) * cfg.task.env.get("controlFrequencyInv", 2)
            print(
                f"[dexret] loop rate: {per_step:.2f} ms/step = {1e3 / per_step:.1f} Hz over "
                f"{steps - 1} steps"
                + (f" (real time needs {budget:.1f} ms/step)" if budget else "")
            )
        print(f"[dexret] done: {episodes} episodes over {steps} steps -> {experiment_dir}")
        return

    runner.run(
        {
            "train": not cfg.test,
            "play": cfg.test,
            "checkpoint": cfg.checkpoint,
            "from_ckpt_epoch": cfg.from_ckpt_epoch,
            "sigma": cfg.sigma if cfg.sigma != "" else None,
            "save_rollouts": {
                "save_rollouts": cfg.save_rollouts,
                "rollout_saving_fpath": os.path.join(experiment_dir, "rollouts.hdf5"),
                "save_successful_rollouts_only": cfg.save_successful_rollouts_only,
                "num_rollouts_to_save": cfg.num_rollouts_to_save,
                "num_rollouts_to_run": cfg.num_rollouts_to_run,
                "min_episode_length": cfg.min_episode_length,
                "stats_fpath": os.path.join(experiment_dir, "stats.txt"),
            },
        }
    )


if __name__ == "__main__":
    launch_rlg_hydra()
