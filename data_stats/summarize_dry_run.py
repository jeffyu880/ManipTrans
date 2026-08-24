"""Turn the per-episode dry-run CSVs a recording left behind into one verdict table per demo.

evalThresholdDryRun scores the eval failure thresholds every step without enforcing them, and the
video wrapper drops the four scored error series next to each mp4 as <video-stem>_metrics.csv. This
re-applies the same rule offline (lib/utils/eval_thresholds.find_threshold_trip owns it) and prints,
per episode, the step the thresholds would have quit at and which term tripped -- the thing the
live printout says, recovered from disk, because SLURM .out files on this cluster go stale mid-job.

Also back-fills <video-stem>_metrics.txt for recordings made before the wrapper started writing it.

    python data_stats/summarize_dry_run.py runs/cup_brush_19demo_dryrun/videos/*/epoch_1100
    python data_stats/summarize_dry_run.py --write-txt runs/cup_brush_19demo_dryrun/videos/m_154012/epoch_1100

Stdlib plus the repo's own threshold module only: no Isaac Gym, no GPU, no torch.
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.utils.eval_thresholds import (  # noqa: E402
    EVAL_FAILURE_THRESHOLDS,
    find_threshold_trip,
    threshold_for,
    unit_for,
)


def read_metric_rows(csv_path):
    """Load one episode's per-step metric rows, floats except the two step counters.

    Args:
        csv_path: a <video-stem>_metrics.csv written by WandbVideoCaptureWrapper.

    Returns:
        list of dicts, oldest step first, with "step" and "running_progress" as ints and every
        scored column as a float.
    """
    rows = []
    with open(csv_path) as handle:
        for raw in csv.DictReader(handle):
            row = {k: float(v) for k, v in raw.items()}
            row["step"] = int(row["step"])
            row["running_progress"] = int(row["running_progress"])
            rows.append(row)
    return rows


def peak_ratios(rows, warmup):
    """Rank the scored terms by how close each came to its own limit over an episode.

    Args:
        rows: per-step metric dicts for one episode.
        warmup: settle window to exclude, matching the gate find_threshold_trip applies.

    Returns:
        list of (column, peak_value, ratio_to_threshold) sorted worst first; empty if the episode
        never left the warmup window.
    """
    scored = [row for row in rows if row["running_progress"] >= warmup]
    if not scored:
        return []
    ranked = []
    for column in sorted(rows[0]):
        threshold = threshold_for(column)
        if threshold is None:
            continue
        peak = max(row[column] for row in scored)
        ranked.append((column, peak, peak / threshold))
    return sorted(ranked, key=lambda item: -item[2])


def wrist_line(rows):
    """Summarise how far each wrist strayed, since no threshold ever reports on it.

    The fingertip terms are world-frame distances, so a wrist drift shows up there as if the
    fingers had curled wrong. Peak wrist error is what tells the two apart.

    Args:
        rows: per-step metric dicts for one episode.

    Returns:
        A single indented string, or "" if the CSV predates the wrist columns.
    """
    parts = []
    for side in ("lh", "rh"):
        pos, rot = f"{side}_diff_eef_pos_dist", f"{side}_diff_eef_rot_angle"
        if pos not in rows[0] or rot not in rows[0]:
            continue
        peak_pos = max(row[pos] for row in rows)
        peak_rot = max(row[rot] for row in rows)
        parts.append(f"{side.upper()} {peak_pos:.4f} m / {peak_rot:.1f} deg")
    return "    wrist peak: " + " | ".join(parts) if parts else ""


def verdict_lines(rows, stem):
    """Build the human-readable verdict for one episode, the same wording the live run prints.

    Args:
        rows: per-step metric dicts for one episode.
        stem: video stem the episode was saved under, e.g. "video-3_failure".

    Returns:
        list of strings, one per output line.
    """
    step, terms = find_threshold_trip(rows)
    if step is None:
        ranked = peak_ratios(rows, warmup=rows[0]["running_progress"])
        margin = ""
        if ranked:
            column, peak, ratio = ranked[0]
            margin = f", closest {column} peaked at {peak:.4f} {unit_for(column)} ({ratio * 100:.0f}% of limit)"
        lines = [f"[{stem}] thresholds never tripped over {len(rows)} steps{margin}"]
    else:
        lines = [f"[{stem}] would have quit at step {step} of {len(rows)}"]
        for column, value, threshold in terms:
            unit = unit_for(column)
            lines.append(f"    {column} = {value:.4f} {unit} (> {threshold:.4f} {unit})")
    wrist = wrist_line(rows)
    if wrist:
        lines.append(wrist)
    return lines


def summarize_dir(video_dir, write_txt):
    """Report every episode recorded into one video directory, then the directory's totals.

    Args:
        video_dir: a .../videos/<idx>/epoch_<N>/ folder holding *_metrics.csv files.
        write_txt: also write each episode's verdict to <video-stem>_metrics.txt.

    Returns:
        (n_episodes, n_tripped, mean_trip_fraction) for the directory; the fraction is of episode
        length, so 0.5 means the thresholds gave out halfway through the demo.
    """
    csv_paths = sorted(
        (p for p in os.listdir(video_dir) if p.endswith("_metrics.csv")),
        key=lambda p: int(p.split("-")[1].split("_")[0]),
    )
    if not csv_paths:
        return 0, 0, 0.0
    print(f"\n=== {video_dir} ===")
    trip_fractions = []
    for name in csv_paths:
        stem = name[: -len("_metrics.csv")]
        rows = read_metric_rows(os.path.join(video_dir, name))
        if not rows:
            continue
        lines = verdict_lines(rows, stem)
        for line in lines:
            print(line)
        step, _ = find_threshold_trip(rows)
        if step is not None:
            trip_fractions.append(step / len(rows))
        if write_txt:
            with open(os.path.join(video_dir, f"{stem}_metrics.txt"), "w") as handle:
                handle.write("\n".join(lines) + "\n")
    n_tripped = len(trip_fractions)
    mean_fraction = sum(trip_fractions) / n_tripped if n_tripped else 0.0
    print(
        f"-- {n_tripped}/{len(csv_paths)} episodes would have been cut short"
        + (f", on average {mean_fraction * 100:.0f}% of the way through" if n_tripped else "")
    )
    return len(csv_paths), n_tripped, mean_fraction


def main():
    """Entry point: summarise each video directory given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("video_dirs", nargs="+", help="videos/<idx>/epoch_<N>/ folders to read")
    parser.add_argument(
        "--write-txt",
        action="store_true",
        help="also write <video-stem>_metrics.txt beside each CSV (back-fill for older runs)",
    )
    args = parser.parse_args()

    limits = ", ".join(f"{k} > {v:g} {unit_for(k)}" for k, v in EVAL_FAILURE_THRESHOLDS.items())
    print(f"Eval failure thresholds: {limits}")

    totals = [summarize_dir(d, args.write_txt) for d in args.video_dirs]
    episodes = sum(t[0] for t in totals)
    tripped = sum(t[1] for t in totals)
    print(f"\n=== overall: {tripped}/{episodes} episodes would have been cut short ===")


if __name__ == "__main__":
    main()
