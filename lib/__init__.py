from omegaconf import DictConfig, OmegaConf


def _is_cuda_solver(x, y):
    if isinstance(y, int):
        return y >= 0
    if isinstance(y, str):
        if "cuda" in y.lower():
            return True
        else:
            return x.lower() in y.lower()


def get_ndof(hand_name):
    from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory

    return DexHandFactory.create_hand(hand_name, "right").n_dofs


def get_nbody(hand_name):
    from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory

    return DexHandFactory.create_hand(hand_name, "right").n_bodies


OmegaConf.register_new_resolver("eq", lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver("contains", _is_cuda_solver)
OmegaConf.register_new_resolver("if", lambda pred, a, b: a if pred else b)
OmegaConf.register_new_resolver("resolve_default", lambda default, arg: default if arg == "" else arg)
OmegaConf.register_new_resolver("is_both_hands", lambda dim, side: dim if side != "BiH" else dim * 2)
OmegaConf.register_new_resolver("is_sep_model", lambda mode, model: ("" if mode != "sep" else "sep_") + model)
OmegaConf.register_new_resolver(
    "is_united_model", lambda mode, side, dim: dim * 2 if mode == "united" and side == "BiH" else dim
)
OmegaConf.register_new_resolver(
    "res_side",
    lambda side, model: ("res_lh_" if side == "LH" else ("res_rh_" if side == "RH" else "res_bih_")) + model,
)
OmegaConf.register_new_resolver("concat", lambda x, y: x + y)
OmegaConf.register_new_resolver("multiply", lambda x, y: x * y)
OmegaConf.register_new_resolver("floor_divide", lambda x, y: x // y)
OmegaConf.register_new_resolver(
    "find_rl_train_config",
    lambda x: x + "PPO" if x[-3:] != "PCD" else x[:-3] + "PPO",
)


def _resolved_fps(target_fps, default=60.0):
    # target_fps comes from ${...demoTargetFps}: None/null/"" -> use the default (native 60Hz) rate.
    if target_fps is None or (isinstance(target_fps, str) and target_fps.strip().lower() in ("", "null", "none")):
        return float(default)
    return float(target_fps)


# demoTargetFps couples the sim rate to the demo rate so it's a single "train at X Hz" knob:
#   fps_dt       -> dt = 1/target_fps
#   fps_substeps -> keeps the physics sub-step at 1/120 s (substeps = 120/target_fps)
#   fps_scale60  -> scales a 60Hz step-count (e.g. episodeLength) linearly with the rate
# null demoTargetFps -> the 60Hz defaults (dt 1/60, substeps 2, unchanged step counts).
OmegaConf.register_new_resolver("fps_dt", lambda target_fps: 1.0 / _resolved_fps(target_fps))
OmegaConf.register_new_resolver("fps_substeps", lambda target_fps: max(1, round(120.0 / _resolved_fps(target_fps))))
OmegaConf.register_new_resolver(
    "fps_scale60",
    lambda value_at_60hz, target_fps: max(1, round(float(value_at_60hz) * _resolved_fps(target_fps) / 60.0)),
)
OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver(
    "ndof",
    lambda x: get_ndof(x),
)  # assuming right and left hands have the same number of dofs
OmegaConf.register_new_resolver(
    "nbody",
    lambda x: get_nbody(x),
)  # assuming right and left hands have the same number of bodies
