"""Retargeting targets for the links that actually grip, not just the fingertips.

Every shipped dex-retargeting config targets the five `*_tip` links and nothing else. But the hand
itself declares a different set of contact surfaces — `contact_body_names` in
`maniptrans_envs/lib/envs/dexhands/inspire.py` is `thumb_distal` plus the four `*_intermediate`
links, and not one of them is a fingertip. A solve constrained only at the tips is therefore free
to return a finger pose whose gripping surfaces never reach the object, which is exactly what the
cup+brush baseline does: tips near the brush, fingers not around it.

This module owns the target set that fixes that, and the MANO indices it needs. Both are DERIVED
from mappings the repo already owns rather than restated:

  * robot link -> MANO joint name: `AVP_TO_MANO_JOINTS` in `main/dataset/object_sets.py`
  * MANO joint name -> slot: `MANO21_JOINT_NAMES` in `baselines/utils/constants.py`

Neither is restated here. `FIT_ALL_PAIRS` in `dexret_controller.py` already pairs robot links with
MANO slots the same way for the WRIST FIT, which since DEXRET_FIT_POINTS="all" already uses all 16
shared joints — contact bodies included. It is only the FINGER SOLVE that is still tip-only, and
that is what this module supplies targets for.

`object_sets` is loaded by file path, not imported. Importing anything under `main.dataset` runs
that package's `__init__`, which registers `mano2dexhand.py` and pulls in isaacgym — and this module
is imported from `baselines.utils`, which is deliberately testable without a simulator. Same trick
CLAUDE.md documents for using a `main/dataset` helper in isolation.

Stdlib plus that one file, matching the rest of `baselines.utils`.
"""

import importlib.util
import os
import sys

from baselines.utils.constants import MANO21_JOINT_NAMES

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The hand's own contact surfaces, in the order dexhands/inspire.py lists them. Restated here
# rather than imported for the isaacgym reason in the module docstring; validate_contact_config()
# is what keeps the two honest.
CONTACT_BODY_NAMES = (
    "thumb_distal",
    "index_intermediate",
    "middle_intermediate",
    "ring_intermediate",
    "pinky_intermediate",
)

TIP_BODY_NAMES = ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip")


def avp_to_mano_joints():
    """The repo's robot-link -> MANO-joint-name table, loaded without importing main.dataset.

    Returns:
        dict mapping e.g. "index_intermediate" -> "Index_PIP".
    """
    path = os.path.join(_REPO, "main", "dataset", "object_sets.py")
    spec = importlib.util.spec_from_file_location("object_sets_for_contact_targets", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.AVP_TO_MANO_JOINTS


def mano_joint_indices():
    """Slot of every MANO joint name, read off the list that already owns that order.

    A thin wrapper on MANO21_JOINT_NAMES rather than a second copy of the layout: the ordering is
    fixed by the MANO convention and the repo already commits to it in one place, so restating it
    here would be the duplicated-constant pattern the codebase has already paid for once.

    Returns:
        dict mapping e.g. "Index_PIP" -> 6.
    """
    return {name: slot for slot, name in enumerate(MANO21_JOINT_NAMES)}


def contact_target_set(side):
    """The ten (robot link, MANO index) targets a contact-aware solve should match.

    Five fingertips, so the solve keeps everything the tip-only configs already got right, then the
    five contact surfaces, which is the part that was missing.

    Args:
        side: "right" or "left" — selects the R_/L_ link-name prefix of ManipTrans's own URDF.

    Returns:
        (link_names, human_indices): both length-10 lists, index i of one corresponding to index i
        of the other. Tips occupy slots 0-4 and contact bodies 5-9.
    """
    assert side in ("right", "left"), f"side must be right or left, got {side!r}"
    prefix = "R_" if side == "right" else "L_"
    avp = avp_to_mano_joints()
    mano = mano_joint_indices()
    link_names, human_indices = [], []
    for body in TIP_BODY_NAMES + CONTACT_BODY_NAMES:
        assert body in avp, (
            f"'{body}' is not in AVP_TO_MANO_JOINTS (main/dataset/object_sets.py). Add it there "
            f"rather than special-casing it here."
        )
        joint = avp[body]
        assert joint in mano, (
            f"AVP_TO_MANO_JOINTS maps '{body}' to '{joint}', which is not a MANO joint name. "
            f"Expected one of {sorted(mano)}."
        )
        link_names.append(prefix + body)
        human_indices.append(mano[joint])
    return link_names, human_indices


def validate_contact_config(config_path, side):
    """Assert a contact config's target lists still match what contact_target_set derives.

    The YAML has to be static for dex-retargeting to read it, so this is what stops it drifting
    away from AVP_TO_MANO_JOINTS or from the hand's contact_body_names. Cheap enough to call once
    at controller build time.

    Args:
        config_path: path to an `inspire_<side>_contact_mt.yml`.
        side: "right" or "left", matching that file.

    Returns:
        None. Raises AssertionError naming the disagreement if the file is out of date.
    """
    import yaml  # noqa: PLC0415 — only needed here, and baselines.utils stays import-light

    with open(config_path) as handle:
        cfg = yaml.safe_load(handle)["retargeting"]
    want_links, want_indices = contact_target_set(side)
    got_links = cfg["target_task_link_names"]
    got_indices = cfg["target_link_human_indices"][1]
    assert got_links == want_links, (
        f"{config_path}: target_task_link_names is {got_links}, derivation says {want_links}. "
        f"Regenerate the file from contact_target_set() instead of editing it by hand."
    )
    assert got_indices == want_indices, (
        f"{config_path}: human indices are {got_indices}, derivation says {want_indices}. "
        f"Regenerate the file from contact_target_set() instead of editing it by hand."
    )
