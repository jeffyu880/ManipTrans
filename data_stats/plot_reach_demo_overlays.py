"""Overlay figures for the reach demos, grouped by whatever the input CSV's first column is.

reach_demo.csv groups by `demo_distance`; reach_folds.csv groups by `fold` (the k-fold
validation splits). Two PNGs per group, each overlaying every demo in that group.

reach_<group>.png — what the cap does:
  * right-hand wrist speed vs normalized time
  * cap speed vs normalized time
  * cap trajectory in the bottle-body frame, top-down (body-local X-Z)
  * cap trajectory in the bottle-body frame, side (horizontal radius vs body-local Y)

wrist_<group>.png — where the hand goes:
  * wrist trajectory in the bottle-body frame, top-down and side
  * wrist height vs normalized time

The body frame is the OptiTrack rigid-body frame of `bottle_body`, i.e.
T_rel = inv(T_body) @ T_cap. Expressing the cap there divides out where and at what
yaw the burner happened to sit on the table, which is what makes demos comparable.
Motive is Y-up and body-local Y sits within a few degrees of world up (verified per
demo, see TILT_WARN_DEG), so body-local Y reads directly as height above the bottle.

Speeds use the same estimator the training pipeline does — ManipData.compute_velocity in
main/dataset/base.py, causal branch with causal_mode="pos_ema", which is what the reach
runs use (causal=true in slurm/scitas/train_reach_dist_array.run). Keeping it identical
means these plots show the velocities the policy is actually asked to track, including
the causal filter's lag, rather than a cleaner offline estimate.

Lines are coloured by capture session (0713 / 0721 / 0724) so batch effects stand out.

Usage:
    python data_stats/plot_reach_demo_overlays.py [csv] [out_dir] [causal_ema_alpha] [group_col]
    python data_stats/plot_reach_demo_overlays.py data_stats/reach_folds.csv          # by fold
    python .../plot_reach_demo_overlays.py data_stats/reach_folds.csv "" 0.3 demo_distance
"""

import csv
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve(path):
    """Relative paths are repo-root-relative, so the script runs from any cwd."""
    return path if os.path.isabs(path) or os.path.exists(path) else os.path.join(REPO, path)


def arg(i, default=None):
    """Positional arg i, treating an empty string as "keep the default" so later args are reachable."""
    return sys.argv[i] if len(sys.argv) > i and sys.argv[i] else default


csv_path = resolve(arg(1, "data_stats/reach_demo.csv"))
out_dir = resolve(arg(2, "data_stats/reach_demo_outputs"))
# Lower alpha = smoother but laggier. 0.3 is main/cfg/config.yaml's causalEmaAlpha, which is
# what the reach runs train with — override only to preview a different setting.
EMA_ALPHA = float(arg(3, 0.3))

DATA_DIR = os.path.join(REPO, "data/my_dataset")
SIDE = "right"  # the reaching hand (holds the cap); "left" holds the bottle body
SKIP = 1  # MyDataset is captured at 60 Hz already, so the loaders do not subsample
UP = np.array([0.0, 1.0, 0.0])  # Motive +Y is the height axis (== gym +Z)
TILT_WARN_DEG = 8.0  # body-local Y this far off world up means the burner was tilted
SESSION_CMAPS = {"0713": "Blues", "0721": "Oranges", "0724": "Greens"}

# `_original` files are pre-shift_cap_height.py backups, never the demo itself.
STEMS = sorted(
    os.path.splitext(p)[0]
    for p in os.listdir(DATA_DIR)
    if p.endswith(".pkl") and not p.endswith("_original.pkl")
)


def causal_ema(x, alpha, seed_first=False):
    """Forward-only EMA over the time axis — copy of ManipData._causal_ema (main/dataset/base.py)."""
    out = np.empty_like(x)
    acc = np.array(x[0]) if seed_first else np.zeros_like(x[0])
    for t in range(x.shape[0]):
        acc = alpha * x[t] + (1.0 - alpha) * acc
        out[t] = acc
    return out


def speed(pos, time_delta):
    """Speed magnitude via ManipData.compute_velocity, causal `pos_ema` branch: low-pass the
    positions with a forward-only EMA, then backward-diff. Seeding the EMA at frame 0 matters —
    seeding at zero would inject a spurious jump into the first derivative."""
    p_s = causal_ema(pos, EMA_ALPHA, seed_first=True)
    velocity = np.zeros_like(p_s)
    velocity[1:] = (p_s[1:] - p_s[:-1]) / time_delta
    return np.linalg.norm(velocity, axis=1)


def load_demo(demo_name):
    """Resolve `m_XXXXXX` to its capture pkl (same matching rule as my_dataset_RH.py)."""
    key = demo_name[2:] if demo_name.startswith("m_") else demo_name
    matches = [s for s in STEMS if s.endswith(key)]
    assert len(matches) == 1, f"'{demo_name}' matched {len(matches)} pkls: {matches}"
    stem = matches[0]
    raw = pickle.load(open(os.path.join(DATA_DIR, stem + ".pkl"), "rb"))

    body = raw["obj_transf"]["bottle_body"].astype(np.float64)
    cap = raw["obj_transf"]["bottle_cap"].astype(np.float64)
    wrist = raw["hands"][SIDE]["wrist_pos"].astype(np.float64)
    t = raw["timestamps_s"] - raw["timestamps_s"][0]

    body_inv = np.linalg.inv(body)
    rel = (body_inv @ cap)[:, :3, 3]  # cap origin in the body frame
    # Wrist in the same frame, so hand paths are comparable across table placements.
    wrist_rel = (body_inv[:, :3, :3] @ wrist[:, :, None])[:, :, 0] + body_inv[:, :3, 3]
    tilt = np.degrees(np.arccos(np.clip(body[:, :3, 1] @ UP, -1.0, 1.0))).mean()
    # Fixed time base, exactly as base.py passes 1 / (self.fps / self.skip) — not the
    # measured timestamps, so these speeds match the ones the policy is trained against.
    time_delta = 1.0 / (raw["meta"]["fps"] / SKIP)

    return {
        "name": demo_name,
        "stem": stem,
        "session": stem.split("_m_")[0].split("_")[-1],
        "t_norm": t / t[-1],
        "duration": t[-1],
        "wrist_speed": speed(wrist, time_delta),
        "cap_speed": speed(cap[:, :3, 3], time_delta),
        "rel": rel,
        "wrist_rel": wrist_rel,
        "tilt": tilt,
    }


def session_colors(demos):
    """One hue per capture session, shaded within it, so batch effects are obvious."""
    by_session = {}
    for d in demos:
        by_session.setdefault(d["session"], []).append(d["name"])
    colors = {}
    for session, names in sorted(by_session.items()):
        cmap = plt.get_cmap(SESSION_CMAPS.get(session, "Greys"))
        for i, name in enumerate(names):
            colors[name] = cmap(0.45 + 0.5 * i / max(len(names) - 1, 1))
    return colors


def draw_path(ax, u, v, color):
    """One demo's trajectory in a 2D projection, with its start and end marked."""
    ax.plot(u, v, color=color, lw=1.2)
    ax.plot(u[0], v[0], "o", color=color, ms=5)
    ax.plot(u[-1], v[-1], "*", color=color, ms=11)


def finish(fig, axes, demos, colors, title, path):
    """Grid, shared legend, title and save — identical across both figure types."""
    for ax in axes.ravel():
        ax.grid(alpha=0.3)
    handles = [
        plt.Line2D([], [], color=colors[d["name"]], lw=2, label=d.get("label", d["name"]))
        for d in demos
    ]
    handles += [
        plt.Line2D([], [], color="k", marker="o", ls="none", ms=5, label="Start"),
        plt.Line2D([], [], color="k", marker="*", ls="none", ms=11, label="End"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 7), fontsize=9, frameon=False)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0.07, 1, 0.97])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_group(title, demos, path):
    colors = session_colors(demos)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    (ax_wrist, ax_cap), (ax_top, ax_side) = axes

    for d in demos:
        c = colors[d["name"]]
        ax_wrist.plot(d["t_norm"], d["wrist_speed"], color=c, lw=1.2)
        ax_cap.plot(d["t_norm"], d["cap_speed"], color=c, lw=1.2)
        # Top-down: both body-local horizontal axes. Side: distance from the bottle
        # axis vs height, which is invariant to the demo's approach azimuth.
        x, y, z = d["rel"][:, 0], d["rel"][:, 1], d["rel"][:, 2]
        draw_path(ax_top, x, z, c)
        draw_path(ax_side, np.hypot(x, z), y, c)

    ax_wrist.set(xlabel="Normalized Time", ylabel="Speed [m/s]", title="RH Wrist Speed")
    ax_cap.set(xlabel="Normalized Time", ylabel="Speed [m/s]", title="Cap Speed")
    ax_top.set(xlabel="Body-Local X [m]", ylabel="Body-Local Z [m]", title="Cap Path, Top-Down")
    ax_side.set(
        xlabel="Horizontal Distance From Bottle Axis [m]",
        ylabel="Body-Local Y (Height) [m]",
        title="Cap Path, Side View",
    )
    for ax in (ax_top, ax_side):
        ax.plot(0, 0, "k+", ms=12, mew=1.5)  # the bottle body origin
        ax.set_aspect("equal", adjustable="datalim")

    finish(fig, axes, demos, colors, title, path)


def plot_wrist_group(title, demos, path):
    colors = session_colors(demos)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax_top, ax_side, ax_height = axes

    for d in demos:
        c = colors[d["name"]]
        x, y, z = d["wrist_rel"][:, 0], d["wrist_rel"][:, 1], d["wrist_rel"][:, 2]
        draw_path(ax_top, x, z, c)
        draw_path(ax_side, np.hypot(x, z), y, c)
        ax_height.plot(d["t_norm"], y, color=c, lw=1.2)

    ax_top.set(xlabel="Body-Local X [m]", ylabel="Body-Local Z [m]", title="Wrist Path, Top-Down")
    ax_side.set(
        xlabel="Horizontal Distance From Bottle Axis [m]",
        ylabel="Body-Local Y (Height) [m]",
        title="Wrist Path, Side View",
    )
    ax_height.set(
        xlabel="Normalized Time", ylabel="Body-Local Y (Height) [m]", title="Wrist Height Over Time"
    )
    for ax in (ax_top, ax_side):
        ax.plot(0, 0, "k+", ms=12, mew=1.5)  # the bottle body origin
        ax.set_aspect("equal", adjustable="datalim")

    finish(fig, axes, demos, colors, f"Wrist Position — {title}", path)


# The CSV's FIRST column is the grouping key — `demo_distance` in reach_demo.csv, `fold` in
# reach_folds.csv — so the same script covers both without a mode flag.
GROUP_STYLES = {"demo_distance": ("Reach Distance", "dist"), "fold": ("Fold", "fold")}

rows = [r for r in csv.DictReader(open(csv_path)) if r.get("demo_name")]
# Defaults to the first column; pass a column name to regroup the same corpus a different way,
# e.g. reach_folds.csv grouped by demo_distance = the 10-demos-per-distance view.
group_col = arg(4, list(rows[0].keys())[0])
assert group_col in rows[0], f"'{group_col}' is not a column in {csv_path}: {list(rows[0])}"
group_label, slug = GROUP_STYLES.get(group_col, (group_col.replace("_", " ").title(), group_col))

groups = {}
distances = {}
for row in rows:
    groups.setdefault(row[group_col], []).append(row["demo_name"])
    # A fold spans every distance, so tag the legend with it; redundant when grouping BY distance.
    if group_col != "demo_distance" and row.get("demo_distance"):
        distances[row["demo_name"]] = row["demo_distance"]

for key, names in sorted(groups.items()):
    demos = [load_demo(n) for n in names]
    for d in demos:
        d["label"] = f"{d['name']}  d{distances[d['name']]}" if d["name"] in distances else d["name"]
    title = f"{group_label} {key} — {len(demos)} Demos"
    reach_png = os.path.join(out_dir, f"reach_{slug}{key}.png")
    wrist_png = os.path.join(out_dir, f"wrist_{slug}{key}.png")
    plot_group(title, demos, reach_png)
    plot_wrist_group(title, demos, wrist_png)
    print(f"\n{group_label} {key}  ->  {reach_png}, {wrist_png}")
    print(f"  {'demo':>10} {'sess':>5} {'dur[s]':>7} {'v_mean':>7} {'v_peak':>7} {'startR':>7} {'endY':>7} {'tilt°':>6}")
    for d in demos:
        r = np.hypot(d["rel"][:, 0], d["rel"][:, 2])
        flag = "  <-- body tilted" if d["tilt"] > TILT_WARN_DEG else ""
        print(
            f"  {d['name']:>10} {d['session']:>5} {d['duration']:7.2f} {d['wrist_speed'].mean():7.3f} "
            f"{d['wrist_speed'].max():7.3f} {r[0]:7.3f} {d['rel'][-1, 1]:7.4f} {d['tilt']:6.1f}{flag}"
        )
