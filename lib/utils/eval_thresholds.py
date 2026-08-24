"""Owns the eval-mode failure thresholds and how the dry-run report presents them.

Three places need these numbers and must not drift apart: the env scores them every step
(maniptrans_envs/lib/envs/tasks/dexhandmanip_bih.py), the video wrapper plots and prints them
(lib/utils/wandb_utils.py), and data_stats/summarize_dry_run.py re-derives the verdicts offline from
the CSVs a finished run leaves behind — SLURM .out files on this cluster go stale mid-job, so the
printed verdict is not a durable record of what happened.

Stdlib only, deliberately: the env module must stay importable before torch, and the summariser has
to run without Isaac Gym or a GPU.
"""

# The four still-active terms of the eval (`else`) branch of compute_imitation_reward, in the units
# the dry-run reports: metres for the distances, degrees for the rotation. That branch ends in
# `failed_execute = error_buf`, which throws the whole comparison away — a test episode now stops
# only on the velocity sanity check. These numbers therefore no longer terminate anything; they are
# the yardstick evalThresholdDryRun measures against, so a rollout can report where it WOULD have
# been cut off while still playing to the end of the demo.
#
# The jit branch keeps its own literals because torch.jit.script cannot close over a
# Dict[str, float], and _print_failure_reason keeps a third copy (with 30 deg for the rotation, left
# from an older tuning). Retune the branch and you must retune here.
EVAL_FAILURE_THRESHOLDS = {
    "diff_thumb_tip_pos_dist": 0.06,  # m, mean over the thumb-tip weight indices
    "diff_index_tip_pos_dist": 0.06,  # m, mean over the index-tip weight indices
    "diff_obj_pos_dist": 0.03,  # m, scored object centre vs the demo
    "diff_obj_rot_angle": 45.0,  # deg, geodesic angle between scored object and demo orientation
}

# The eval branch gates every term on `running_progress_buf >= 8`: the first steps after a reset are
# a settle window where the hand has not caught up with the demo yet and every term reads large.
EVAL_FAILURE_WARMUP_STEPS = 8

# Panels of the dry-run plot: the metric key (minus its rh_/lh_ prefix), the panel title, the
# y-axis label, and the unit used when the same number is printed. Laid out row-major into a 2x3
# grid, so the left two columns are the scored terms in the eval branch's own order (finger tips,
# then object pose) and the right column is the wrist.
#
# The two wrist entries are DIAGNOSTIC: they are absent from EVAL_FAILURE_THRESHOLDS because no
# failure branch scores the wrist at all (it is rewarded — reward_eef_pos/reward_eef_rot — but never
# terminates anything). They are plotted without a limit line and ignored by find_threshold_trip.
# They earn their place because the fingertip terms are WORLD-frame distances, so a drifting wrist
# carries every fingertip with it: without these panels a wrist drift is indistinguishable from
# fingers that curled wrong.
EVAL_DRY_RUN_PANELS = (
    ("diff_thumb_tip_pos_dist", "Thumb Tip Error", "Distance [m]", "m"),
    ("diff_index_tip_pos_dist", "Index Tip Error", "Distance [m]", "m"),
    ("diff_eef_pos_dist", "Wrist Position Error", "Distance [m]", "m"),
    ("diff_obj_pos_dist", "Object Position Error", "Distance [m]", "m"),
    ("diff_obj_rot_angle", "Object Rotation Error", "Angle [deg]", "deg"),
    ("diff_eef_rot_angle", "Wrist Rotation Error", "Angle [deg]", "deg"),
)


def threshold_for(column):
    """Look up the limit a metric column is judged against, ignoring its rh_/lh_ prefix.

    Args:
        column: a CSV/metrics column name such as "lh_diff_obj_pos_dist", or a bookkeeping
            column such as "step" that is not scored at all.

    Returns:
        The threshold as a float, or None if the column is not one of the scored terms.
    """
    return EVAL_FAILURE_THRESHOLDS.get(column.split("_", 1)[-1])


def find_threshold_trip(rows, warmup_steps=EVAL_FAILURE_WARMUP_STEPS):
    """Find the first step at which the eval thresholds would have ended an episode.

    The rule, not just the numbers, lives here: the video wrapper applies it live and
    data_stats/summarize_dry_run.py re-applies it to CSVs long after the run, and the two must
    agree. Any step inside the warmup window is skipped, exactly as the eval branch's
    `running_progress_buf >= 8` gate does.

    Args:
        rows: per-step metric dicts, oldest first. Each needs a "running_progress" entry (steps
            since that env's last reset) plus the scored columns, e.g. "lh_diff_obj_pos_dist".
        warmup_steps: settle window to ignore; defaults to the eval branch's own gate.

    Returns:
        (step, terms) where step indexes into rows and terms is a list of
        (column, value, threshold) for every term over its limit at that step; (None, []) if the
        thresholds were never breached.
    """
    for step, row in enumerate(rows):
        if row.get("running_progress", step) < warmup_steps:
            continue
        over = []
        for column in sorted(row):
            threshold = threshold_for(column)
            if threshold is not None and row[column] > threshold:
                over.append((column, row[column], threshold))
        if over:
            return step, over
    return None, []


def unit_for(column):
    """Give the unit a metric column is reported in, so printed numbers carry m or deg.

    Args:
        column: a metrics column name such as "rh_diff_obj_rot_angle".

    Returns:
        "m" or "deg" for a scored term, "" for anything else.
    """
    key = column.split("_", 1)[-1]
    for panel_key, _, _, unit in EVAL_DRY_RUN_PANELS:
        if panel_key == key:
            return unit
    return ""
