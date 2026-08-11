"""Validate the RH_LObj_Center_Aug X-shift budget by sampling and plotting augmented demos.

Draws N yaws per demo with the SAME sampler the env uses at load time
(DexHandManipBiHEnv.sample_rh_lobj_center_yaw), applies the SAME augmentation
(_aug_demo_rh_lobj_center_aug), and checks that no augmented trajectory strays further than
`rhLObjCenterAugMaxXShift` from the original along world X.

One demo per reach distance, so the figure shows the intended trade directly: that aug rotates the
RH demo about the LH OBJECT, so the displacement it causes is linear in the hand-to-object lever
arm. A far-reaching demo therefore has to accept a narrower yaw band than a close one to stay
inside the same budget.

The env code is read straight out of maniptrans_envs/lib/envs/tasks/dexhandmanip_bih.py by AST and
exec'd, rather than imported: importing the env pulls in isaacgym/gymtorch, which needs ninja and a
GPU. The dataset loaders are imported through a sys.modules shim for the same reason (main.dataset's
__init__ auto-registers mano2dexhand.py, which imports isaacgym). Both mean this runs on a login
node while still exercising the real code.

Usage:
    python data_stats/plot_rh_lobj_center_aug.py [n_samples] [out_png] [max_x_shift]
"""

import ast
import importlib.util
import os
import sys
import textwrap
import types

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_SRC = os.path.join(REPO, "maniptrans_envs/lib/envs/tasks/dexhandmanip_bih.py")
POOL_CSV = os.path.join(REPO, "data_stats/reach_demo_v2.csv")

# Mirrors dexhandmanip_bih.py:375-390 -- table_pos.z + table_half_height, then the fixed raw->gym
# rotation. Reproduced (not imported) so the demos land in exactly the frame the aug sees.
TABLE_SURFACE_Z = 0.4 + 0.015
N_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 40
OUT_PNG = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "data_stats/reach_demo_outputs/rh_lobj_center_aug.png")
MAX_X_SHIFT = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05  # matches config.yaml's default


def shim_package(name, path):
    """Register an empty package so its submodules import without running the real __init__.

    Args:
        name: Dotted package name, e.g. "main.dataset".
        path: Directory the package's submodules live in.

    Returns:
        The stub module object, already in sys.modules.
    """
    pkg = types.ModuleType(name)
    pkg.__path__ = [path]
    pkg.__package__ = name
    sys.modules[name] = pkg
    return pkg


def load_loaders_and_dexhand():
    """Bring up MyDatasetRH/LH and an Inspire hand config without importing isaacgym.

    Once the stub packages are in sys.modules, plain import_module resolves every submodule
    normally. That matters: loading a module twice under the same name would rebuild its classes,
    and the registry decorators would then populate a DexHandFactory that is no longer the one
    create_hand is called on.

    Returns:
        (ManipDataFactory, dexhand_rh, dexhand_lh).
    """
    ds_dir = os.path.join(REPO, "main/dataset")
    shim_package("main", os.path.join(REPO, "main"))
    shim_package("main.dataset", ds_dir)
    importlib.import_module("main.dataset.transform")
    for name in ("my_dataset_RH", "my_dataset_LH"):  # their @register_manipdata does the rest
        importlib.import_module("main.dataset." + name)

    dx_dir = os.path.join(REPO, "maniptrans_envs/lib/envs/dexhands")
    for pkg, path in (("maniptrans_envs", "maniptrans_envs"),
                      ("maniptrans_envs.lib", "maniptrans_envs/lib"),
                      ("maniptrans_envs.lib.envs", "maniptrans_envs/lib/envs"),
                      ("maniptrans_envs.lib.envs.dexhands", "maniptrans_envs/lib/envs/dexhands")):
        shim_package(pkg, os.path.join(REPO, path))
    dex_factory = importlib.import_module("maniptrans_envs.lib.envs.dexhands.factory").DexHandFactory
    dex_factory.auto_register_hands(dx_dir, "maniptrans_envs.lib.envs.dexhands")
    return (sys.modules["main.dataset.factory"].ManipDataFactory,
            dex_factory.create_hand("inspire", "right"),
            dex_factory.create_hand("inspire", "left"))


def load_env_functions(names):
    """Exec the named methods out of the env source, so the real code is what gets tested.

    Args:
        names: Set of function names to pull out of dexhandmanip_bih.py.

    Returns:
        Dict {name: function}, sharing one namespace so they can call each other.
    """
    src = open(ENV_SRC).read()
    lines = src.splitlines()
    chunks = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            chunks.append(textwrap.dedent("\n".join(lines[node.lineno - 1: node.end_lineno])))
    assert len(chunks) == len(names), f"found {len(chunks)} of {len(names)} functions in {ENV_SRC}"
    transform = sys.modules["main.dataset.transform"]
    ns = {"torch": torch, "np": np,
          "aa_to_rotmat": transform.aa_to_rotmat, "rotmat_to_aa": transform.rotmat_to_aa}
    exec("\n\n".join(chunks), ns)
    return ns


def mujoco2gym():
    """The fixed raw-capture -> gym transform the env builds, as a (4, 4) tensor."""
    aa_to_rotmat = sys.modules["main.dataset.transform"].aa_to_rotmat
    t = np.eye(4)
    t[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(np.array([np.pi / 2, 0, 0]))
    t[:3, 3] = np.array([0, 0, TABLE_SURFACE_Z])
    return torch.tensor(t, dtype=torch.float32)


def one_demo_per_distance(csv_path):
    """First demo listed at each reach distance, so the figure spans the whole reach range.

    Args:
        csv_path: reach_demo_v2.csv (columns demo_distance, demo_name).

    Returns:
        List of (distance string, demo name), ascending by distance.
    """
    import csv as csvmod
    first = {}
    for row in csvmod.DictReader(open(csv_path)):
        first.setdefault(row["demo_distance"], row["demo_name"])
    return [(d, first[d]) for d in sorted(first, key=int)]


def yaw_degrees(rotation):
    """Signed yaw of a (3, 3) rotation about Z, in degrees."""
    return float(np.degrees(np.arctan2(rotation[1, 0].item(), rotation[0, 0].item())))


def main():
    """Sample, augment, check the budget, and write the figure."""
    manip_factory, dexhand_rh, dexhand_lh = load_loaders_and_dexhand()
    env_fns = load_env_functions({"rh_lobj_center_start_offsets", "start_abs_x_shift",
                                  "sample_rh_lobj_center_yaw", "_aug_demo_rh_lobj_center_aug"})
    # sample_rh_lobj_center_yaw is a bound method in the env; here it only needs start_abs_x_shift.
    shim_self = types.SimpleNamespace(start_abs_x_shift=env_fns["start_abs_x_shift"])

    transf = mujoco2gym()
    common = dict(device="cpu", mujoco2gym_transf=transf, max_seq_len=1200,
                  embodiment="inspire", target_fps=None, causal=True,
                  causal_ema_alpha=0.3, causal_mode="pos_ema")
    data_rh = manip_factory.create_data(manipdata_type="mydataset", side="right",
                                        dexhand=dexhand_rh, **common)
    data_lh = manip_factory.create_data(manipdata_type="mydataset", side="left",
                                        dexhand=dexhand_lh, **common)

    picks = one_demo_per_distance(POOL_CSV)
    torch.manual_seed(42)

    fig, axes = plt.subplots(2, len(picks), figsize=(4.1 * len(picks), 8.4))
    # 40 samples are one population, not 40 categories -> a single sequential hue keyed to |yaw|,
    # never a cycled categorical palette. The reference demo gets an ink colour, not a ramp step.
    ramp = plt.get_cmap("viridis")
    ink = "#111111"
    budget_c = "#b3261e"
    rows = []

    for col, (dist, demo) in enumerate(picks):
        rh, lh = data_rh[demo], data_lh[demo]
        ux, uy = env_fns["rh_lobj_center_start_offsets"](rh, lh)
        lever = float(torch.sqrt(ux ** 2 + uy ** 2).max())

        base = rh["wrist_pos"].cpu().numpy()
        t_norm = np.linspace(0.0, 1.0, base.shape[0])
        angles, shifts, paths = [], [], []
        for _ in range(N_SAMPLES):
            rot = env_fns["sample_rh_lobj_center_yaw"](shim_self, ux, uy, MAX_X_SHIFT, "cpu", demo)
            aug_rh, _ = env_fns["_aug_demo_rh_lobj_center_aug"](rh, lh, rot)
            path = aug_rh["wrist_pos"].cpu().numpy()
            angles.append(yaw_degrees(rot))
            paths.append(path)
            shifts.append(path[:, 0] - base[:, 0])
        angles = np.array(angles)
        shifts = np.array(shifts)
        norm = plt.Normalize(0.0, max(np.abs(angles).max(), 1e-6))

        ax_top, ax_dx = axes[0, col], axes[1, col]
        for path, dx, ang in zip(paths, shifts, angles):
            c = ramp(norm(abs(ang)))
            ax_top.plot(path[:, 0], path[:, 1], color=c, lw=0.8, alpha=0.55)
            ax_top.plot(path[0, 0], path[0, 1], "o", color=c, ms=4, alpha=0.9)
            ax_dx.plot(t_norm, dx, color=c, lw=0.8, alpha=0.5)
            ax_dx.plot(0.0, dx[0], "o", color=c, ms=4, alpha=0.9, zorder=6)
        ax_top.plot(base[:, 0], base[:, 1], color=ink, lw=2.0, zorder=5)
        ax_top.plot(base[0, 0], base[0, 1], "o", color=ink, ms=6, zorder=6)
        pivot = lh["obj_trajectory"][:, :3, 3].cpu().numpy()
        ax_top.plot(pivot[:, 0], pivot[:, 1], "+", color=budget_c, ms=9, mew=1.6, zorder=6)

        for sign in (-1, 1):
            ax_dx.axhline(sign * MAX_X_SHIFT, color=budget_c, lw=1.4, ls="--", zorder=4)
        ax_dx.axhline(0.0, color=ink, lw=1.2, zorder=3)

        start_worst = float(np.abs(shifts[:, 0]).max())  # the constrained quantity
        demo_worst = float(np.abs(shifts).max())         # unconstrained, shown for context
        ax_top.set_title(f"d{dist} — {demo}\nStart Lever {lever:.2f} m, |Yaw| ≤ {np.abs(angles).max():.1f}°",
                         fontsize=10)
        ax_top.set(xlabel="World X [m]", ylabel="World Y [m]" if col == 0 else "")
        ax_top.set_aspect("equal", adjustable="datalim")
        ax_dx.set(xlabel="Normalized Time",
                  ylabel="Wrist ΔX From Original [m]" if col == 0 else "")
        ax_dx.set_ylim(-1.1 * max(1.25 * MAX_X_SHIFT, demo_worst),
                       1.1 * max(1.25 * MAX_X_SHIFT, demo_worst))
        ax_dx.set_title(f"Start |ΔX| ≤ {start_worst:.4f} m  (later {demo_worst:.3f} m, unbounded)",
                        fontsize=9, color=ink if start_worst <= MAX_X_SHIFT else budget_c)
        for ax in (ax_top, ax_dx):
            ax.grid(alpha=0.25, lw=0.6)
        rows.append((dist, demo, lever, np.abs(angles).max(), angles.std(), start_worst,
                     demo_worst, bool(start_worst <= MAX_X_SHIFT + 1e-9)))

    handles = [plt.Line2D([], [], color=ink, lw=2, label="Original Demo"),
               plt.Line2D([], [], color=ramp(0.55), lw=1.2, label=f"{N_SAMPLES} Augmented Samples"),
               plt.Line2D([], [], color=ramp(0.55), marker="o", ls="none", ms=6, label="Start Pose"),
               plt.Line2D([], [], color=budget_c, lw=1.4, ls="--", label=f"±{MAX_X_SHIFT} m Start Budget"),
               plt.Line2D([], [], color=budget_c, marker="+", ls="none", ms=9, label="LH Object (Pivot)")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=10, frameon=False)
    fig.suptitle(f"RH_LObj_Center_Aug — {N_SAMPLES} Sampled Yaws Per Demo, "
                 f"rhLObjCenterAugMaxXShift = {MAX_X_SHIFT} m On The Starting Position", fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)

    print(f"\nwrote {OUT_PNG}\n")
    print(f"{'dist':>5} {'demo':>10} {'startLever[m]':>14} {'|yaw|max':>9} {'yaw std':>8} "
          f"{'start|dX|':>10} {'later|dX|':>10} {'within':>7}")
    for dist, demo, lever, amax, astd, start_worst, demo_worst, ok in rows:
        print(f"   d{dist} {demo:>10} {lever:14.3f} {amax:8.1f}° {astd:7.1f}° "
              f"{start_worst:10.4f} {demo_worst:10.4f} {str(ok):>7}")
    assert all(r[-1] for r in rows), "a sampled yaw broke the start-X budget -- constraint not holding"
    print(f"\nALL {len(rows) * N_SAMPLES} sampled starts within {MAX_X_SHIFT} m along X "
          f"(later frames intentionally unbounded).")


if __name__ == "__main__":
    main()
