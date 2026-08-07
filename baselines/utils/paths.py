"""Locating the URDFs and retargeting configs the baseline solves against."""

import os

from baselines.utils.constants import DEFAULT_RETARGETING, DEXRET_SOLVE_URDF, RETARGETING_TYPES

_BASELINES = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Where dexret2dexhand writes. Its own root, so it can never collide with mano2dexhand's output in
# data/retargeting/my_dataset/, which is the path the loaders resolve. The mano2<dexhand> level
# below this mirrors that layout, so swapping a file in is a copy rather than a rename.
DEXRET_PLAYBACK_ROOT = os.path.join("data", "dex_retarget_playback")


def dex_urdf_dir():
    """Directory the configs' relative `urdf_path` resolves against — dex-urdf's hand models.

    dex-urdf is a separate checkout, not vendored, so this searches the usual places instead of
    hardcoding one machine's layout. Override with DEX_URDF_DIR.

    Returns:
        str absolute path to <dex-urdf>/robots/hands.
    """
    override = os.environ.get("DEX_URDF_DIR")
    repo = os.path.dirname(_BASELINES)
    candidates = [override] if override else [
        os.path.join(os.path.dirname(repo), "third_party", "dex-urdf", "robots", "hands"),
        os.path.join(repo, "third_party", "dex-urdf", "robots", "hands"),
        os.path.expanduser("~/dex-urdf/robots/hands"),
    ]
    for path in candidates:
        if path and os.path.isdir(path):
            return os.path.abspath(path)
    raise SystemExit(
        f"dex-urdf not found (looked in {candidates}). Clone it and point DEX_URDF_DIR at the "
        f"hands directory:\n"
        f"  git clone https://github.com/dexsuite/dex-urdf\n"
        f"  export DEX_URDF_DIR=<dex-urdf>/robots/hands"
    )


def solve_urdf_dir():
    """Directory the active config's relative `urdf_path` resolves against.

    Which model the fit solves against is a real choice, not plumbing: dex-urdf is what
    dex-retargeting and Bunny-VisionPro were built for, while ManipTrans's own URDF is the hand the
    SIMULATOR runs. Solving against a different model than the one being executed leaves a
    systematic fingertip error that no amount of wrist fitting can remove. See DEXRET_SOLVE_URDF.

    Returns:
        str absolute path to the directory holding <robot>_hand/<robot>_hand_<side>.urdf.
    """
    if DEXRET_SOLVE_URDF == "maniptrans":
        repo = os.path.dirname(_BASELINES)
        return os.path.abspath(os.path.join(repo, "maniptrans_envs", "assets"))
    return dex_urdf_dir()


def default_config_path(robot, hand, retargeting=DEFAULT_RETARGETING):
    """Path to our dex-urdf retargeting config for one hand and optimiser.

    Args:
        robot: dex-retargeting robot name, e.g. "inspire".
        hand: "right" or "left".
        retargeting: one of RETARGETING_TYPES.

    Returns:
        str path, which may not exist — only inspire has configs today.
    """
    assert retargeting in RETARGETING_TYPES, (
        f"retargeting must be one of {RETARGETING_TYPES}, got {retargeting!r}"
    )
    # The ManipTrans-URDF variants live alongside, suffixed, so both sets stay side by side and
    # switching is a single constant rather than an edit to every config.
    suffix = "_mt" if DEXRET_SOLVE_URDF == "maniptrans" else ""
    return os.path.join(_BASELINES, "configs", f"{robot}_{hand}_{retargeting}{suffix}.yml")
