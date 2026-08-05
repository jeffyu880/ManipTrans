"""Frame conventions and tuning constants for the dex-retargeting baseline."""

import numpy as np

# The 21 AVP joints corresponding to MANO/MediaPipe keypoints 0..20, in that order. By name rather
# than index because each capture carries its own `finger_names`, which is the real authority. The
# four AVP metacarpals are absent by design: MANO has no counterpart for them.
MANO21_JOINT_NAMES = [
    "Base",
    "Thumb_CMC", "Thumb_MCP", "Thumb_IP", "Thumb_TIP",
    "Index_MCP", "Index_PIP", "Index_DIP", "Index_TIP",
    "Middle_MCP", "Middle_PIP", "Middle_DIP", "Middle_TIP",
    "Ring_MCP", "Ring_PIP", "Ring_DIP", "Ring_TIP",
    "Pinky_MCP", "Pinky_PIP", "Pinky_DIP", "Pinky_TIP",
]

# Copied from dex_retargeting.constants rather than imported, so this module needs no install.
# Values are fixed by the MANO convention, not by our data.
OPERATOR2MANO_RIGHT = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float64)
OPERATOR2MANO_LEFT = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]], dtype=np.float64)

# Bunny-VisionPro's axis relabeling, feeding the headset's wrist straight to the optimiser instead
# of fitting a frame to the keypoints. Right == OPERATOR2MANO_RIGHT; left differs by exactly
# rotY(180), the same mirror AVP_LH_WRIST_CORRECTION applies in my_dataset_LH.py. Baking it in stops
# the mirror being a separate step that can be applied on one path and forgotten on the other.
OPERATOR2AVP = {
    "right": OPERATOR2MANO_RIGHT,
    "left": np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64),
}

RETARGETING_TYPES = ("dexpilot", "vector")

# DexPilot is Bunny-VisionPro's choice: it adds a thumb-to-finger projection that snaps small gaps
# closed to stabilise grasps. "vector" drops that and matches wrist-to-fingertip only, so the
# fingers track the human instead of being pulled into a grasp prior — use it if the thumb rides
# its yaw limit (which the projection will do whenever the human's grip is tight).
DEFAULT_RETARGETING = "dexpilot"

# Fraction of the wrist-to-middle-MCP span to pull the retargeted wrist back toward the forearm,
# the hack oakink2/grab apply in their loaders at 0.25. 0.35 here, tuned by eye on m_101919 where
# the AVP hands crowded the object (~33 mm). Owned here so the offline pkls and the in-env
# controller cannot drift apart. NOTE this is NOT object_sets.WRIST_PULLBACK, which stays 0 because
# that one moves the demo tracking targets and so every reward.
DEXRET_WRIST_PULLBACK = 0.35
