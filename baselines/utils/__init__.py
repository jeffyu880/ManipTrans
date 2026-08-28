"""Shared pieces of the dex-retargeting baseline, used by both the offline script and the env.

Deliberately stdlib + numpy only. `dexret_controller` imports this and then pulls in isaacgym via
`main.dataset.transform`; keeping this package clean means nothing here can trip the
isaacgym-before-torch import order, and the constants stay usable without the sim.
"""

from baselines.utils.constants import (
    DEFAULT_RETARGETING,
    DEXRET_FIT_CALIB_FRAMES,
    DEXRET_FIT_CALIB_PINCH_FRAC,
    DEXRET_FIT_CALIB_SAMPLES,
    DEXRET_FIT_MAX_ANGLE_DEG,
    DEXRET_FIT_MAX_TRANSLATION,
    DEXRET_FIT_MODE,
    DEXRET_FIT_OVERRIDE_FREE_JOINT,
    DEXRET_FIT_POINTS,
    DEXRET_FIT_WORLD_FREEZE,
    DEXRET_FIT_WEIGHTS,
    DEXRET_ESCAPE_DIST,
    DEXRET_PROJECT_DIST,
    DEXRET_SCALING_FACTOR,
    DEXRET_SOLVE_URDF,
    DEXRET_WRIST_FIT,
    DEXRET_WRIST_PULLBACK,
    MANO21_JOINT_NAMES,
    OPERATOR2AVP,
    OPERATOR2MANO_LEFT,
    OPERATOR2MANO_RIGHT,
    RETARGETING_TYPES,
)
from baselines.utils.calibration import (
    CALIB_DIR,
    calibration_path,
    load_calibration,
    save_calibration,
)
from baselines.utils.paths import (
    DEXRET_PLAYBACK_ROOT,
    default_config_path,
    dex_urdf_dir,
    solve_urdf_dir,
)
from baselines.utils.contact_targets import contact_target_set, validate_contact_config
from baselines.utils.retarget import pull_wrist_back, retarget_ref_value
from baselines.utils.wrist_fit import (
    average_rigid,
    clamp_rigid,
    fit_wrist_to_fingertips,
    kabsch,
    rotmat_to_rotvec,
    rotvec_to_rotmat,
    tip_rms,
)

__all__ = [
    "CALIB_DIR",
    "DEFAULT_RETARGETING",
    "DEXRET_FIT_CALIB_FRAMES",
    "DEXRET_FIT_CALIB_PINCH_FRAC",
    "DEXRET_FIT_CALIB_SAMPLES",
    "DEXRET_FIT_MAX_ANGLE_DEG",
    "DEXRET_FIT_MAX_TRANSLATION",
    "DEXRET_FIT_MODE",
    "DEXRET_FIT_OVERRIDE_FREE_JOINT",
    "DEXRET_FIT_POINTS",
    "DEXRET_FIT_WORLD_FREEZE",
    "DEXRET_FIT_WEIGHTS",
    "DEXRET_ESCAPE_DIST",
    "DEXRET_PLAYBACK_ROOT",
    "DEXRET_PROJECT_DIST",
    "DEXRET_SCALING_FACTOR",
    "DEXRET_SOLVE_URDF",
    "DEXRET_WRIST_FIT",
    "DEXRET_WRIST_PULLBACK",
    "MANO21_JOINT_NAMES",
    "OPERATOR2AVP",
    "OPERATOR2MANO_LEFT",
    "OPERATOR2MANO_RIGHT",
    "RETARGETING_TYPES",
    "average_rigid",
    "calibration_path",
    "clamp_rigid",
    "default_config_path",
    "dex_urdf_dir",
    "solve_urdf_dir",
    "fit_wrist_to_fingertips",
    "kabsch",
    "load_calibration",
    "pull_wrist_back",
    "save_calibration",
    "retarget_ref_value",
    "rotmat_to_rotvec",
    "rotvec_to_rotmat",
    "tip_rms",
]
