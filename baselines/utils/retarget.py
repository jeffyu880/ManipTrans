"""Shaping the solver's input and placing the retargeted wrist."""

from baselines.utils.constants import DEXRET_WRIST_PULLBACK


def retarget_ref_value(retargeting, kp21):
    """The reference value a built solver expects, for whichever optimiser it wraps.

    DEXPILOT/VECTOR consume tip-minus-origin vectors and index `target_link_human_indices` as
    (2, n) pairs; POSITION consumes absolute keypoints and indexes it as (n,). Feeding one an array
    shaped for the other mis-gathers silently rather than raising, hence one shared gather.

    Args:
        retargeting: A built SeqRetargeting solver.
        kp21: (21, 3) keypoints already in the optimiser's frame.

    Returns:
        (n, 3) reference value to hand to `retargeting.retarget()`.
    """
    indices = retargeting.optimizer.target_link_human_indices
    if retargeting.optimizer.retargeting_type == "POSITION":
        return kp21[indices, :]
    return kp21[indices[1, :], :] - kp21[indices[0, :], :]


def pull_wrist_back(wrist_pos, middle_pos, fraction=DEXRET_WRIST_PULLBACK):
    """Move the wrist back along its own palm axis, away from the fingers.

    Stops the retargeted hand crowding the object. A rigid translation, so finger joint angles are
    unaffected; and equivariant under the loader's rigid transform chain, so applying it to
    already-transformed positions matches what the loaders would have produced.

    Args:
        wrist_pos: (..., 3) wrist positions, tensor or array.
        middle_pos: (..., 3) middle-MCP positions, same frame, shape and type.
        fraction: How far back, as a fraction of the wrist-to-middle-MCP span. 0 = no shift.

    Returns:
        (..., 3) shifted wrist positions, same type as `wrist_pos`.
    """
    if not fraction:
        return wrist_pos
    return wrist_pos - (middle_pos - wrist_pos) * fraction
