"""Frame conventions and tuning constants for the dex-retargeting baseline."""

import os

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

# Which hand model the fit solves against.
#   "maniptrans" — ManipTrans's own inspire, i.e. the hand the SIMULATOR actually runs. DEFAULT.
#   "dex"        — dex-urdf's inspire, what dex-retargeting and Bunny-VisionPro were built for.
#                  Kept for comparison against the published method; it is measurably worse here.
#
# These are NOT the same kinematic model. Commanding identical joint values and measuring where the
# fingertips actually end up: achieved minus commanded z is +14 mm on the LH and -9 mm on the RH, a
# ~23 mm inter-hand swing that accounts for essentially the whole LH-vs-RH placement discrepancy.
# The fit cannot correct it -- Kabsch matches the fingertips of the model it is given, and the sim
# then executes those joint angles on a different one.
#
# "maniptrans" was rejected in the original design because that URDF's base sits ~167 deg from the
# MANO frame the keypoints arrive in, so a solver fed raw MANO-frame targets sees a hand pointing
# the wrong way. dexret_controller now composes the hand-local transform with solver_to_sim before
# handing keypoints to the optimiser, so they arrive already in the target URDF's own base frame
# and that objection no longer applies.
DEXRET_SOLVE_URDF = "maniptrans"

_solve_urdf_env = os.environ.get("MANIPTRANS_DEXRET_SOLVE_URDF", "").strip()
if _solve_urdf_env:
    assert _solve_urdf_env in ("dex", "maniptrans"), (
        f"MANIPTRANS_DEXRET_SOLVE_URDF must be dex or maniptrans, got {_solve_urdf_env!r}"
    )
    DEXRET_SOLVE_URDF = _solve_urdf_env

RETARGETING_TYPES = ("dexpilot", "vector", "position", "position_free")

# DexPilot is Bunny-VisionPro's choice: it adds a thumb-to-finger projection that snaps small gaps
# closed to stabilise grasps. "vector" drops that and matches wrist-to-fingertip only, so the
# fingers track the human instead of being pulled into a grasp prior — use it if the thumb rides
# its yaw limit (which the projection will do whenever the human's grip is tight).
DEFAULT_RETARGETING = "dexpilot"

# --- Fingertip wrist fit (baselines/utils/wrist_fit.py) ---
#
# Supersedes the scalar pullback below for the configs with no free joint (vector/dexpilot): rather
# than sliding the wrist along one hand-tuned axis, solve the 6-DOF rigid placement that puts the
# robot's fingertips on the human's. See wrist_fit.py for why the pullback cannot do this.
DEXRET_WRIST_FIT = True

# Ceilings on how far the fit may depart from the human's own wrist pose, applied per frame. The
# fit is a least-squares answer built from five points, so a frame with the thumb on its limit or
# the tips near-collinear can be numerically fine and physically wrong; clamping degrades it
# smoothly instead of dropping the frame. 8 cm is well past any hand-size mismatch and well short
# of leaving the workspace. 30 deg comfortably covers the ~25 deg by which dex-urdf's base is
# misaligned with the MANO frame the solver assumes -- the systematic error the fit exists to
# absorb -- without letting a bad frame invert the hand.
# Raised from 80 mm / 30 deg, which was a guess made before any of this was measured and turned
# out to be doing tuning work rather than outlier guarding: it bound 24% of dexpilot's calibration
# samples on one demo and 81% of `position`'s RH samples on another, costing `position` 3 mm of fit
# accuracy (16.3 -> 13.3 mm RMS once released). At 150 mm / 60 deg the clamp binds on 0% of
# dexpilot samples and its metrics are identical to no clamp at all, so it is back to being what it
# was meant to be -- a backstop against a degenerate solve teleporting the hand, not a limit the
# normal answer runs into.
DEXRET_FIT_MAX_TRANSLATION = 0.15   # metres
DEXRET_FIT_MAX_ANGLE_DEG = 60.0

_max_trans_env = os.environ.get("MANIPTRANS_DEXRET_FIT_MAX_TRANSLATION", "").strip()
if _max_trans_env:
    DEXRET_FIT_MAX_TRANSLATION = float(_max_trans_env)
_max_angle_env = os.environ.get("MANIPTRANS_DEXRET_FIT_MAX_ANGLE_DEG", "").strip()
if _max_angle_env:
    DEXRET_FIT_MAX_ANGLE_DEG = float(_max_angle_env)

# How often the fit is solved.
#   "per_frame" — re-solve every control step from that step's finger pose. Adapts as the grasp
#                 closes, but carries no smoothness guarantee: measured at 1.3 mm/frame typical
#                 with a 73 mm worst-case jump, and the wrist is force-controlled, so a jump in
#                 the target is a spike in the commanded force.
#   "constant"  — solve once over a calibration pass, take a robust average, and hold it for the
#                 whole run. Smooth by construction and free at runtime (two matrix products), at
#                 the cost of not tracking how the correction should change through the motion.
# The constant is stored in the HAND-LOCAL frame, so it rotates with the wrist rather than being
# a fixed world displacement — a fixed world offset would be wrong the moment the hand turns.
#
# "constant" is the default because it measured BETTER, not merely smoother. Over three demos at
# scaling 1.00 it roughly doubles RH contact (35.6% vs 19.0%) for a 3% worse fingertip error, and
# cuts frame-to-frame movement of the correction from 1.30 mm to 0.16 mm with no jump above 1.8 mm.
# The reason is that a per-frame fit chases the fingertips, so as the fingers close it walks the
# hand off the object; a fixed standoff lets them press.
DEXRET_FIT_MODE = "constant"
# NOTE there is deliberately NO MANIPTRANS_DEXRET_FIT_MODE env override: main/cfg always
# populates task.env.dexRetFitMode, so an env var would be silently ignored through
# main/rl/train.py. Use the CLI flag `dexRetFitMode=per_frame`.

# Frames used to calibrate the constant. 0 = the whole demo (offline; the demo buffer is already
# in memory, so this is a one-off pre-pass costing ~1 ms per sampled frame). A positive value
# takes the first N frames instead, which is what live mode needs — there is no whole demo to
# average over, so it calibrates on the opening frames and then freezes.
DEXRET_FIT_CALIB_FRAMES = 0

# Cap on how many frames the calibration pass actually solves; it strides through the window
# rather than solving every frame. The constant is an average, so 200 samples pin it as well as
# 2000 and keep startup under a second.
DEXRET_FIT_CALIB_SAMPLES = 200

# Fraction of the demo to calibrate on, taken as the frames where THAT HAND's thumb and index tips
# are nearest its own object surface (`tips_distance`) — the pinch. 1.0 = the whole demo.
#
# Averaging over the whole demo answers "where should the hand sit on average", but most of a demo
# is reach and retreat, where the hand is ~150 mm from the object and its exact placement does not
# matter. The grasp is the only phase where it does, and it is a minority of the frames, so the
# constant ends up dominated by the part of the motion nobody cares about.
#
# 0.25 measured on the fold_4 demos as a single CONTIGUOUS window of ~90 frames sitting mid-demo,
# where the thumb/index tips average ~22 mm from the cap against ~150 mm over the full demo — a 7x
# concentration onto the grasp. Contiguous matters: the solver is warm-started, so a window that
# jumped around the timeline would be solving from a stale seed every sample.
#
DEXRET_FIT_CALIB_PINCH_FRAC = {"rh": 0.25, "lh": 0.25}

# Which robot/human point pairs the wrist fit is solved against.
#   "tips" — the five fingertips only. The correspondences dex-retargeting itself optimises, so
#            they are the ones known to line up; but five tips of a nearly-closed hand are close to
#            coplanar, which is the worst case for pinning a rotation by SVD.
#   "all"  — additionally the proximal and intermediate links of every finger, plus the thumb's
#            distal: 16 pairs instead of 5. Spread across the palm rather than clustered at the
#            fingers, so the rotation is far better conditioned, and the palm is anchored rather
#            than inferred from the tips alone. The cost is that only the tips are correspondences
#            the retargeting was solved for -- the others are name matches between two different
#            hands, so any systematic joint-position offset now biases the fit.
# The four *_distal finger joints are excluded: ManipTrans's packed buffer has no row for them
# (inspire.hand2dex_mapping marks them "missing"), so there is nothing to match against.
# How much of the wrist correction is held fixed in the WORLD frame instead of rotating with the
# hand.
#   "none" — the whole correction is hand-local, as originally written.
#   "z"    — only the vertical component is world-fixed; x/y still rotate.
#   "xyz"  — the entire correction is a fixed world offset.
#
# The correction is stored hand-local so it follows the wrist. That is right for the horizontal
# part, which is what centres the grasp — but its world-z projection then swings as the wrist
# pitches (the per-frame corrections span 32-80 mm in world-z over a calibration window), and that
# swing subtracts directly from how far the commanded wrist rises.
#
# "z" is the default on measured grounds, not on the axis being special a priori: see the sweep in
# the module docstring of baselines/utils/wrist_fit.py. Set MANIPTRANS_DEXRET_FIT_WORLD_FREEZE.
DEXRET_FIT_WORLD_FREEZE = "z"

_freeze_env = os.environ.get("MANIPTRANS_DEXRET_FIT_WORLD_FREEZE", "").strip()
if _freeze_env:
    assert _freeze_env in ("none", "z", "xyz"), (
        f"MANIPTRANS_DEXRET_FIT_WORLD_FREEZE must be none|z|xyz, got {_freeze_env!r}"
    )
    DEXRET_FIT_WORLD_FREEZE = _freeze_env


# Which robot points the Kabsch fit matches onto the human's.
#   "tips" — the five fingertips, as originally written.
#   "all"  — every joint present on BOTH the robot and MANO: 16 points, including the *_proximal
#            knuckles. (The four finger *_distal joints are absent from ManipTrans's packed buffer
#            by design, so 16 is the ceiling, not 20.)
#
# "all" because five fingertips are a badly conditioned set to fit a rigid transform from: they are
# near-coplanar (smallest/largest singular value of the centred cloud is 0.04-0.14) and they all sit
# ~15 cm from the base frame, so the fitted ROTATION has a long lever arm and is weakly constrained
# about the cloud normal. The fit then amplifies small changes in finger pose into large wrist
# displacements. Measured on m_154027, lowering DexPilot's project_dist from 0.030 to 0.015 -- which
# changes the raw finger solve by 2 mm, and for the better:
#
#                        fit |t|           fit angle        clamped      RH fingertip error
#     tips (5 points)    70.8 -> 151.6 mm  32.7 -> 60.0 deg  0 -> 100%   18.0 -> 68.0 mm
#     all (16 points)    39.3 ->  41.6 mm  15.4 -> 18.2 deg  0 ->   0%   25.7 -> 29.4 mm
#
# The tips-only fit saturates the clamp and swings the whole hand 50 mm; the 16-point fit barely
# moves. Confirmed as a rotation-lever effect by a one-point (index-tip) fit, where the rotation is
# identically zero by construction and the same threshold change moves the hand 0.1 mm.
#
# The cost is ~7 mm of fingertip accuracy at the default threshold (tips is the better fit WHEN it
# is well conditioned), bought against a fit that no longer falls over. Verified across the 10
# reach_5foldcv holdout demos: 0% of samples clamped on every one, RMS 53-63% lower.
# Set MANIPTRANS_DEXRET_FIT_POINTS=tips to restore the old behaviour.
DEXRET_FIT_POINTS = "all"

# Normally the fit and a config's own 6-DOF free joint (`position`, `position_free`) are mutually
# exclusive — both answer "where does the hand go", and applying both displaces it twice. Setting
# this keeps the FIT and discards the free joint's answer instead of disabling the fit.
#
# Worth trying because the two solve the same question by different means: the free joint folds
# placement into the same non-linear least-squares as the finger angles, where it is one more set
# of coupled DOFs for a local optimiser to get wrong; Kabsch solves it in closed form once the
# angles are fixed. `position` measured 0% RH contact with its own free joint, so its finger solve
# has never been evaluated with a placement that works.
#
# The free joint's DOFs are zeroed before forward kinematics when this is on, so the fit sees the
# fingers in the hand's base frame rather than at the pose the optimiser chose.
DEXRET_FIT_OVERRIDE_FREE_JOINT = False

_fit_override_env = os.environ.get("MANIPTRANS_DEXRET_FIT_OVERRIDE_FREE", "").strip()
if _fit_override_env:
    DEXRET_FIT_OVERRIDE_FREE_JOINT = _fit_override_env not in ("0", "false", "False")

_fit_points_env = os.environ.get("MANIPTRANS_DEXRET_FIT_POINTS", "").strip()
if _fit_points_env:
    assert _fit_points_env in ("tips", "all"), (
        f"MANIPTRANS_DEXRET_FIT_POINTS must be tips or all, got {_fit_points_env!r}"
    )
    DEXRET_FIT_POINTS = _fit_points_env

# Relative weight per matched point, as {substring of the robot link name: weight}. Longest
# matching key wins, unmatched links get 1.0, and only the ratios matter (they are normalised).
#
# This form exists because the points are not equally trustworthy. The five *_tip pairs are the
# correspondences dex-retargeting actually optimises, so they are known to line up. The proximal
# and intermediate pairs are name matches between two anatomically different hands, and carry
# whatever systematic joint-position offset that implies. Weighting them down keeps the extra
# points' benefit — they are spread across the palm, so they condition the rotation far better
# than five near-coplanar tips — without letting their correspondence error steer the fit.
#
# Measured with DEXRET_FIT_POINTS="all" and UNIFORM weights: RH contact fell 50.1% -> 31.8% and LH
# tip error rose 33.2 -> 38.0 mm against the tips-only fit. The knuckles dragged the hand off the
# object. That is the failure this knob is for.
#
# Also accepts a plain 5-list for the "tips" set (positional, thumb->pinky), and None for uniform.
# Set MANIPTRANS_DEXRET_FIT_WEIGHTS to "3,3,1,1,1" (positional) or "tip=1,intermediate=0.3" (named).
# Positional weights, if given as a plain list, are per fingertip in MANO order (thumb, index,
# middle, ring, pinky) and only valid for the "tips" set. Note that biasing toward the pinch pair
# did NOT help: 3,3,1,1,1 and 8,8,1,1,1 both measured indistinguishable from uniform, and zeroing
# three of the five is not a pinch-focused fit but an undetermined one -- Kabsch needs three
# non-collinear points to pin a rotation, and two leave it free about the line joining them
# (measured: 77 mm tip error, hand driven into the object at 3 N). Hence the >= 3 assert below.
DEXRET_FIT_WEIGHTS = None

_fit_weights_env = os.environ.get("MANIPTRANS_DEXRET_FIT_WEIGHTS", "").strip()
if _fit_weights_env:
    if "=" in _fit_weights_env:
        DEXRET_FIT_WEIGHTS = {
            k.strip(): float(v) for k, v in
            (part.split("=") for part in _fit_weights_env.split(","))
        }
        assert all(w >= 0 for w in DEXRET_FIT_WEIGHTS.values()), (
            f"MANIPTRANS_DEXRET_FIT_WEIGHTS must be non-negative, got {_fit_weights_env!r}"
        )
    else:
        DEXRET_FIT_WEIGHTS = [float(v) for v in _fit_weights_env.split(",")]
        assert len(DEXRET_FIT_WEIGHTS) == 5, (
            f"positional MANIPTRANS_DEXRET_FIT_WEIGHTS needs 5 values "
            f"(thumb,index,middle,ring,pinky), got {_fit_weights_env!r}"
        )
        # 1 or 3+, never 2. With a single point the centred cloud is the zero vector, so the
        # covariance is zero and Kabsch returns R = I exactly -- a pure translation that lands that
        # one point on its target. That is degenerate but WELL DEFINED, and useful: it is the
        # zero-rotation fit, with none of the lever-arm amplification that makes the 5-tip rotation
        # swing the whole hand. With two points the rotation about the axis through them is
        # unconstrained and Kabsch returns an arbitrary large one (144 deg on a random test pair) --
        # silently wrong in exactly the way this assert exists to catch.
        _nonzero = sum(w > 0 for w in DEXRET_FIT_WEIGHTS)
        assert _nonzero == 1 or _nonzero >= 3, (
            f"MANIPTRANS_DEXRET_FIT_WEIGHTS needs 1 non-zero weight (pure translation, R = I) or "
            f"at least 3 (full rigid fit); {_fit_weights_env!r} gives {_nonzero}. Two points leave "
            f"the rotation about the axis through them unconstrained."
        )

# Override for the config's own `scaling_factor` (dexpilot/vector only; `position` takes none).
# dex-retargeting ships 1.15 for inspire, i.e. the robot is asked to spread its fingers 15% wider
# than the human did. With the wrist jammed into the object by the pullback that never showed; with
# the fit placing the wrist properly it becomes a gap the fingers never close.
#
# 1.00 because contact is what a manipulation baseline is for: over three demos it takes RH frames
# in contact from 5% to 19% under the per-frame fit, at the cost of ~2 mm of fingertip error. The
# error rises by construction -- closing the fingers moves their tips off the human's -- so the two
# metrics necessarily disagree here and contact is the one that decides whether the hand grips.
# None = leave the config's own value alone. Set MANIPTRANS_DEXRET_SCALING to sweep it.
DEXRET_SCALING_FACTOR = 1.00

_scaling_env = os.environ.get("MANIPTRANS_DEXRET_SCALING", "").strip()
if _scaling_env:
    DEXRET_SCALING_FACTOR = float(_scaling_env)

# DexPilot's thumb-to-finger projection thresholds (metres), dexpilot only. The optimiser latches a
# pair "projected" once the HUMAN's gap falls below project_dist and only releases it above
# escape_dist -- a Schmitt trigger. While latched the target gap is overwritten with eta1 = 0.1 mm
# and weighted 200, against 15 for the absolute wrist-to-fingertip vectors, so the closure outranks
# fingertip placement by ~13x and the fingers are driven shut.
#
# The library ships 0.03 / 0.05, tuned for a hand pinching in free space. On the cup+brush set the
# human's thumb-index gap averages 27.0 mm (RH) / 28.9 mm (LH) -- just under the 30 mm trigger -- so
# it latches on 74% / 58% of frames and the robot's gap collapses to 7.5 / 11.6 mm. A light hold on
# a handle is read as a hard pinch. Lower project_dist to keep a grip aperture that is part of the
# motion from being quantised away.
#
# 0.015 / 0.025, halving the library's 0.030 / 0.050 and keeping its 5:3 ratio. Measured on
# m_154027: at the library default the robot's thumb-index gap is 7.6 mm against a human 23.6 mm
# (RH) and 3.2 vs 27.6 (LH) -- a light hold on a handle rendered as a hard pinch. At 0.015 it is
# 19.7 / 18.1 mm, and at 0.010 the error is -3.7 mm on both hands. 0.015 rather than 0.010 because
# the two are within noise of each other on every measure (RH tip 29.4 vs 29.2 mm, LH 28.0 vs 27.8)
# and 0.015 stays further from the point where a genuine pinch would stop registering.
#
# BOTH move together, deliberately. escape_dist is the release threshold, and dropping project_dist
# alone would WIDEN the hysteresis band: a pair latching at 15 mm would stay latched until the hand
# opened past 50 mm, which on this data happens on 0.9% of RH frames. That is the opposite of the
# intended effect.
#
# This is only safe alongside DEXRET_FIT_POINTS="all". With the five-fingertip fit the same change
# saturates the clamp and swings the hand 50 mm -- see the table above that constant.
# None = leave the config's own value alone. Set MANIPTRANS_DEXRET_PROJECT_DIST / _ESCAPE_DIST to
# sweep them, or to 0.03 / 0.05 to restore DexPilot's shipped behaviour.
DEXRET_PROJECT_DIST = 0.015
DEXRET_ESCAPE_DIST = 0.025

_project_env = os.environ.get("MANIPTRANS_DEXRET_PROJECT_DIST", "").strip()
if _project_env:
    DEXRET_PROJECT_DIST = float(_project_env)

_escape_env = os.environ.get("MANIPTRANS_DEXRET_ESCAPE_DIST", "").strip()
if _escape_env:
    DEXRET_ESCAPE_DIST = float(_escape_env)

# One value applies to both hands; "rh=0.25,lh=1.0" sets them separately.
_pinch_frac_env = os.environ.get("MANIPTRANS_DEXRET_CALIB_PINCH_FRAC", "").strip()
if _pinch_frac_env:
    if "=" in _pinch_frac_env:
        DEXRET_FIT_CALIB_PINCH_FRAC = {
            k.strip(): float(v) for k, v in
            (part.split("=") for part in _pinch_frac_env.split(","))
        }
    else:
        value = float(_pinch_frac_env)
        DEXRET_FIT_CALIB_PINCH_FRAC = {"rh": value, "lh": value}
    assert set(DEXRET_FIT_CALIB_PINCH_FRAC) == {"rh", "lh"} and all(
        0 < v <= 1.0 for v in DEXRET_FIT_CALIB_PINCH_FRAC.values()
    ), (
        f"MANIPTRANS_DEXRET_CALIB_PINCH_FRAC needs one fraction in (0, 1], or 'rh=..,lh=..'; "
        f"got {_pinch_frac_env!r}"
    )


# Fraction of the wrist-to-middle-MCP span to pull the retargeted wrist back toward the forearm,
# the hack oakink2/grab apply in their loaders at 0.25. 0.38 here, tuned by eye on m_101919 where
# the AVP hands crowded the object (~34 mm on an 88 mm span). Owned here so the offline pkls and the in-env
# controller cannot drift apart. NOTE this is NOT object_sets.WRIST_PULLBACK, which stays 0 because
# that one moves the demo tracking targets and so every reward.
DEXRET_WRIST_PULLBACK = 0.38

# Sweepable like the other constants here. Only has any effect with dexRetWristFit=false: the fit
# supersedes the scalar pullback and zeroes it (dexret_controller.py, "wrist_fit supersedes").
_pullback_env = os.environ.get("MANIPTRANS_DEXRET_WRIST_PULLBACK", "").strip()
if _pullback_env:
    DEXRET_WRIST_PULLBACK = float(_pullback_env)
