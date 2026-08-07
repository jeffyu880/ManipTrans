"""Persisting the wrist-fit constant so a live session does not have to re-derive it.

The constant offset (see wrist_fit.py) is a property of the OPERATOR'S hand against the ROBOT'S
hand, not of any one session: it answers "where does this robot's base have to sit for its
fingertips to land where this person's do". That does not change between runs, so re-measuring it
at the start of every teleop session is wasted warm-up during which the hand is being driven by a
correction that is still moving.

So it is measured once by an explicit calibration run and written here; every later live run loads
it and is correct from the first frame.

Stored as JSON rather than a pickle because it is nine floats and three ints per hand and wants to
be readable, diffable and hand-editable. The provenance fields are not decoration — a calibration
taken against a different robot, optimiser or scaling factor is silently wrong rather than
obviously wrong, so `load_calibration` refuses one that does not match instead of trusting it.
"""

import json
import os

import numpy as np

# Alongside the other generated data, not in the source tree: a calibration is a measurement of a
# person, so it is no more checked-in than a capture is.
CALIB_DIR = os.path.join("data", "dexret_calibration")

# Bumped when the meaning of the stored numbers changes (frame convention, what is averaged).
# An old file is then refused rather than silently reinterpreted.
CALIB_FORMAT = 1


def calibration_path(robot, retargeting, directory=None):
    """Where the calibration for one robot/optimiser pair lives.

    Args:
        robot: dex-retargeting robot name, e.g. "inspire".
        retargeting: optimiser name, e.g. "dexpilot".
        directory: override the containing directory; None uses CALIB_DIR.

    Returns:
        str path to the JSON file, which may not exist yet.
    """
    return os.path.join(directory or CALIB_DIR, f"{robot}_{retargeting}.json")


def save_calibration(path, robot, retargeting, scaling, per_side):
    """Write a calibration, creating its directory.

    Args:
        path: destination JSON path.
        robot: dex-retargeting robot name, recorded for the match check on load.
        retargeting: optimiser name, likewise.
        scaling: the scaling_factor in force when this was measured, likewise. None if the
            config's own value was used unmodified.
        per_side: {"rh"|"lh": (rotation (3, 3), translation (3,), n_samples int)}.

    Returns:
        str the path written.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "format": CALIB_FORMAT,
        "robot": robot,
        "retargeting": retargeting,
        "scaling": scaling,
        "sides": {
            side: {
                "rotation": np.asarray(rotation, dtype=float).tolist(),
                "translation": np.asarray(translation, dtype=float).tolist(),
                "samples": int(samples),
            }
            for side, (rotation, translation, samples) in per_side.items()
        },
    }
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    return path


def load_calibration(path, robot, retargeting, scaling):
    """Read a calibration, or return None if there is not a usable one.

    Refuses rather than adapts on any mismatch. A calibration measured against a different hand or
    a different optimiser is not approximately right — the frames and the finger kinematics both
    differ — and silently using it would show up as a tracking problem nobody would trace back
    here.

    Args:
        path: JSON path; a missing file is not an error, it just means "not calibrated yet".
        robot: expected robot name.
        retargeting: expected optimiser name.
        scaling: expected scaling factor, or None.

    Returns:
        {"rh"|"lh": ((3, 3) rotation, (3,) translation)} or None.
    """
    if not os.path.exists(path):
        return None

    try:
        with open(path) as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        print(f"\033[1;93m[dexret] ignoring unreadable calibration {path}: {exc}\033[0m")
        return None

    mismatches = [
        f"{name} {payload.get(name)!r} != {expected!r}"
        for name, expected in (
            ("format", CALIB_FORMAT),
            ("robot", robot),
            ("retargeting", retargeting),
            ("scaling", scaling),
        )
        if payload.get(name) != expected
    ]
    if mismatches:
        print(
            f"\033[1;93m[dexret] ignoring calibration {path}: it was taken for a different setup "
            f"({'; '.join(mismatches)}). Re-run with dexRetCalibrate=true.\033[0m"
        )
        return None

    return {
        side: (np.array(entry["rotation"], dtype=float), np.array(entry["translation"], dtype=float))
        for side, entry in payload["sides"].items()
    }
