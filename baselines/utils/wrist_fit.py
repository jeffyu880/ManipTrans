"""Placing the retargeted wrist by fitting the robot's fingertips onto the human's.

`vector` and `dexpilot` solve the fingers ALONE, in the wrist frame, and the wrist itself is then
supplied from the human. That leaves the grasp off-centre for two reasons, neither of which a wrist
taken from the human can fix:

  * dex-retargeting scales the human's inter-finger vectors by `scaling_factor` (1.15 for inspire),
    so the robot's fingertips deliberately do not sit where the human's did relative to the wrist;
  * the solver assumes the robot's URDF base is aligned with the MANO frame the keypoints arrive
    in, and for dex-urdf's inspire that holds only to ~25 deg.

`pull_wrist_back` was the previous answer and it is strictly weaker: it slides the wrist along ONE
axis (wrist -> middle-MCP) by a hand-tuned fraction, while the displacement is a full 6-DOF rigid
error. What we actually want is the hand pose that puts the fingertips where the human's are, and
because the finger angles are already fixed by the time we ask, that is a plain orthogonal
Procrustes problem with a closed-form answer -- no optimiser, no gains, no tuning.

Deliberately numpy-only (no scipy), matching the rest of `baselines.utils`, so the sim is not
needed to test any of this.
"""

import numpy as np


def kabsch(source, target, weights=None):
    """Rigid transform best mapping `source` onto `target`, in the least-squares sense.

    Solves `argmin_{R,t} sum_i w_i ||R s_i + t - q_i||^2` over rotations (det = +1) and
    translations, via the orthogonal Procrustes / Kabsch construction. Reflections are excluded by
    flipping the sign of the last singular vector when the raw solution has det = -1; without that
    guard a near-planar point set (which five fingertips of a flat hand very nearly are) can
    silently return a mirrored "rotation" that turns the hand inside out.

    Args:
        source: (n, 3) points in the frame being placed -- here the robot's fingertips, read off
            forward kinematics in the hand's base frame.
        target: (n, 3) points to land on, same order -- here the human's fingertips.
        weights: (n,) non-negative per-point weights, or None for uniform. Rescaled internally, so
            only their ratios matter.

    Returns:
        ((3, 3) rotation R, (3,) translation t) such that `R @ s + t` approximates `q`.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    assert source.shape == target.shape and source.ndim == 2 and source.shape[1] == 3, (
        f"kabsch needs matching (n, 3) point sets, got {source.shape} and {target.shape}"
    )

    if weights is None:
        w = np.ones(len(source))
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        assert len(w) == len(source) and np.all(w >= 0) and w.sum() > 0, (
            f"kabsch weights must be {len(source)} non-negative values that are not all zero"
        )
    w = w / w.sum()

    source_mean = (w[:, None] * source).sum(axis=0)
    target_mean = (w[:, None] * target).sum(axis=0)

    covariance = (w[:, None] * (source - source_mean)).T @ (target - target_mean)
    u, _, vt = np.linalg.svd(covariance)
    # det(V U^T) is +1 for a rotation and -1 for a reflection; folding it into the middle factor is
    # the standard fix and costs nothing when the solution was already proper.
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return rotation, target_mean - rotation @ source_mean


def rotmat_to_rotvec(rotation):
    """Axis-angle vector of a rotation matrix, magnitude = angle in radians.

    Hand-rolled rather than scipy so this module stays numpy-only. Uses the antisymmetric part for
    ordinary angles and falls back to the symmetric part near pi, where sin(angle) -> 0 makes the
    usual formula lose all its precision.

    Args:
        rotation: (3, 3) rotation matrix.

    Returns:
        (3,) axis-angle vector.
    """
    rotation = np.asarray(rotation, dtype=np.float64)
    # Clipped because round-off can push the trace a hair outside [-1, 3] and arccos then returns
    # NaN, which would propagate silently into the clamp below.
    angle = np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-8:
        return np.zeros(3)
    if angle < np.pi - 1e-4:
        axis = np.array([
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]) / (2.0 * np.sin(angle))
        return axis * angle
    # Near pi: R + I = 2 * outer(axis, axis), so the largest diagonal entry gives the axis with the
    # best conditioning. The sign is arbitrary at exactly pi, which is harmless -- +pi and -pi about
    # the same axis are the same rotation.
    symmetric = (rotation + np.eye(3)) / 2.0
    axis = np.sqrt(np.maximum(np.diag(symmetric), 0.0))
    major = int(np.argmax(axis))
    axis = symmetric[:, major] / axis[major]
    return axis / np.linalg.norm(axis) * angle


def rotvec_to_rotmat(rotvec):
    """Rotation matrix of an axis-angle vector, by Rodrigues' formula.

    Args:
        rotvec: (3,) axis-angle vector, magnitude = angle in radians.

    Returns:
        (3, 3) rotation matrix.
    """
    rotvec = np.asarray(rotvec, dtype=np.float64)
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.eye(3)
    axis = rotvec / angle
    cross = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def clamp_rigid(rotation, translation, max_translation, max_angle_rad):
    """Limit how far a fitted transform may depart from the identity.

    The fit is only as trustworthy as the five points it is built from. A frame where the thumb is
    saturated against a joint limit, or where the tips are momentarily near-collinear, can produce a
    perfectly valid least-squares answer that is nonetheless a large, wrong displacement -- and
    because the wrist is force-controlled, a large step in the target is a large force. Clamping
    scales the correction back toward the human's own wrist pose (identity here) rather than
    rejecting it, so tracking degrades smoothly instead of dropping out for a frame.

    Args:
        rotation: (3, 3) fitted rotation, relative to the current hand orientation.
        translation: (3,) fitted translation, metres, relative to the current wrist position.
        max_translation: Largest allowed |translation|, metres. 0 or None disables the limit.
        max_angle_rad: Largest allowed rotation angle, radians. 0 or None disables the limit.

    Returns:
        ((3, 3) rotation, (3,) translation, bool clamped) -- `clamped` is True if either limit bit.
    """
    clamped = False

    if max_translation:
        norm = float(np.linalg.norm(translation))
        if norm > max_translation:
            translation = translation * (max_translation / norm)
            clamped = True

    if max_angle_rad:
        rotvec = rotmat_to_rotvec(rotation)
        angle = float(np.linalg.norm(rotvec))
        if angle > max_angle_rad:
            rotation = rotvec_to_rotmat(rotvec * (max_angle_rad / angle))
            clamped = True

    return rotation, translation, clamped


def tip_rms(robot_tips, human_tips, rotation=None, translation=None, weights=None):
    """Weighted RMS fingertip distance, optionally after applying a transform to the robot's tips.

    Exists so the constant-offset mode can be scored on the same footing as the per-frame fit:
    a constant that is never re-solved still has a per-frame error, and the only honest way to say
    how much it gives up is to measure it against the same tips on the same frames.

    Args:
        robot_tips: (n, 3) robot fingertip positions.
        human_tips: (n, 3) human fingertip positions, same frame and order.
        rotation: (3, 3) applied to robot_tips first, or None for no rotation.
        translation: (3,) added after the rotation, or None for none.
        weights: (n,) per-point weights, or None for uniform.

    Returns:
        float RMS distance, metres.
    """
    tips = np.asarray(robot_tips, dtype=np.float64)
    if rotation is not None:
        tips = tips @ np.asarray(rotation, dtype=np.float64).T
    if translation is not None:
        tips = tips + np.asarray(translation, dtype=np.float64)

    w = np.ones(len(tips)) if weights is None else np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    return float(np.sqrt((w * ((tips - np.asarray(human_tips)) ** 2).sum(axis=1)).sum()))


def average_rigid(rotations, translations):
    """Single rigid transform summarising a sequence of them.

    Used to collapse a whole demo's worth of per-frame fits into one constant offset. The two
    components are averaged differently on purpose:

      * translation by the PER-AXIS MEDIAN, not the mean. A handful of frames — a thumb on its
        joint limit, tips momentarily near-collinear — produce large bad offsets, and a mean lets
        any one of them drag the constant. The median ignores them outright.
      * rotation by the chordal L2 mean: average the matrices entrywise, then project back onto
        SO(3) with an SVD. This is the closed-form minimiser of the summed squared Frobenius
        distance, and unlike averaging Euler angles or axis-angle vectors it has no wrap-around
        failure. It is not robust the way a median is, but the rotations vary far less than the
        translations do, so it does not need to be.

    Args:
        rotations: sequence of (3, 3) rotation matrices.
        translations: matching sequence of (3,) translations.

    Returns:
        ((3, 3) rotation, (3,) translation).
    """
    rotations = np.asarray(rotations, dtype=np.float64)
    translations = np.asarray(translations, dtype=np.float64)
    assert len(rotations) == len(translations) and len(rotations) > 0, (
        f"average_rigid needs matching non-empty sequences, got {len(rotations)} rotations and "
        f"{len(translations)} translations"
    )

    u, _, vt = np.linalg.svd(rotations.mean(axis=0))
    d = np.sign(np.linalg.det(u @ vt))
    return u @ np.diag([1.0, 1.0, d]) @ vt, np.median(translations, axis=0)


def fit_wrist_to_fingertips(robot_tips, human_tips, max_translation=0.0, max_angle_rad=0.0,
                            weights=None):
    """Wrist correction placing the robot's fingertips onto the human's.

    Both point sets must already be in the SAME frame: the optimiser's hand-local frame, where the
    robot's base sits at the origin unrotated and the human's wrist sits at the origin too. The
    returned transform is therefore the correction to today's placement, and an exact fit returns
    (identity, zero) -- which is the invariant the wiring is tested against.

    Args:
        robot_tips: (5, 3) robot fingertip positions from forward kinematics on the solved joint
            angles, in the hand's base frame.
        human_tips: (5, 3) human fingertip positions, wrist-centred, same frame and finger order.
        max_translation: Clamp on |t|, metres. 0 disables.
        max_angle_rad: Clamp on the rotation angle, radians. 0 disables.
        weights: (5,) per-finger weights, or None for uniform.

    Returns:
        ((3, 3) rotation, (3,) translation, dict diagnostics) where the diagnostics carry the
        weighted RMS fingertip error before and after the fit, in metres, plus whether a clamp bit.
    """
    robot_tips = np.asarray(robot_tips, dtype=np.float64)
    human_tips = np.asarray(human_tips, dtype=np.float64)

    rotation, translation = kabsch(robot_tips, human_tips, weights)
    rotation, translation, clamped = clamp_rigid(
        rotation, translation, max_translation, max_angle_rad
    )

    w = np.ones(len(robot_tips)) if weights is None else np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    rms = lambda tips: float(np.sqrt((w * ((tips - human_tips) ** 2).sum(axis=1)).sum()))
    fitted = robot_tips @ rotation.T + translation

    return rotation, translation, {
        "rms_before": rms(robot_tips),
        "rms_after": rms(fitted),
        "clamped": clamped,
    }
