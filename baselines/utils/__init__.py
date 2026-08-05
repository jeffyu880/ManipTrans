"""Shared pieces of the dex-retargeting baseline, used by both the offline script and the env.

Deliberately stdlib + numpy only. `dexret_controller` imports this and then pulls in isaacgym via
`main.dataset.transform`; keeping this package clean means nothing here can trip the
isaacgym-before-torch import order, and the constants stay usable without the sim.
"""

from baselines.utils.constants import (
    DEFAULT_RETARGETING,
    DEXRET_WRIST_PULLBACK,
    MANO21_JOINT_NAMES,
    OPERATOR2AVP,
    OPERATOR2MANO_LEFT,
    OPERATOR2MANO_RIGHT,
    RETARGETING_TYPES,
)
from baselines.utils.paths import DEXRET_PLAYBACK_ROOT, default_config_path, dex_urdf_dir
from baselines.utils.retarget import pull_wrist_back, retarget_ref_value

__all__ = [
    "DEFAULT_RETARGETING",
    "DEXRET_PLAYBACK_ROOT",
    "DEXRET_WRIST_PULLBACK",
    "MANO21_JOINT_NAMES",
    "OPERATOR2AVP",
    "OPERATOR2MANO_LEFT",
    "OPERATOR2MANO_RIGHT",
    "RETARGETING_TYPES",
    "default_config_path",
    "dex_urdf_dir",
    "pull_wrist_back",
    "retarget_ref_value",
]
