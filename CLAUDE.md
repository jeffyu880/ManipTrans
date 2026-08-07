# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git
- Never add `Co-Authored-By: Claude` or any self-attribution to commit messages.
- Commit messages: one bullet point per feature. Only write more than a line per feature when the
  implementation is genuinely large.

## Communication
- When making code changes, always show which file and line(s) are being modified and briefly describe what is changing before or alongside the edit.

## Autonomy & permissions
- **OK to run without asking:** read-only commands (status checks, reading SLURM state via `squeue`/`sacct`/`scontrol`, reading files); writing completely new files; and verifying/validating scripts (dry-runs, `bash -n`/syntax checks, sanity checks that don't modify tracked files).
- **Ask first:** modifying any existing script or file — confirm before making the change.

## Code style
- Follow the `coding-rules` skill (`.claude/skills/coding-rules/SKILL.md`) when writing or editing code.
- **Plots**: axis labels and titles use Title Case — `Cap Speed`, `Normalized Time`, `Body-Local X [m]`. Units stay in their brackets.

## Python Environment
- Use the **`maniptrans`** conda env for all Python commands: `/home/jsyu/miniconda/envs/maniptrans/bin/python` (or `conda activate maniptrans`). numpy 1.23.5, Python 3.8.
- **Isaac Gym import order**: `isaacgym` must be imported *before* `torch`, or it raises `ImportError: PyTorch was imported before isaacgym modules`. Importing anything under `main.dataset` triggers the package `__init__`, which auto-registers `mano2dexhand.py` (imports isaacgym) — so a standalone script that does `import torch` then `from main.dataset...` will fail. To use a helper like `main/dataset/transform.py` in isolation, load it directly by file path with `importlib.util.spec_from_file_location` to bypass the package `__init__`.

# ManipTrans

**ManipTrans: Efficient Dexterous Bimanual Manipulation Transfer via Residual Learning** (CVPR 2025)

Transfers human hand-object manipulation demonstrations to dexterous robot hands using a two-stage residual learning approach in Isaac Gym.

## Documentation map

Deep reference lives in `docs/`. Consult these (and read the relevant one before diving into code):

| Doc | Contents |
|---|---|
| [docs/README.md](docs/README.md) | Docs index, one-paragraph overview, quick commands |
| [docs/pipeline.md](docs/pipeline.md) | End-to-end pipeline (data → retargeting → imitator → residual → test/eval → live), runtime step loop, retargeted-pkl format, **full CLI parameter table** |
| [docs/architecture.md](docs/architecture.md) | Two-stage network internals, observation/target/dim tables, step-loop diagram, reward, dexterous-hand configs, domain randomization |
| [docs/datasets.md](docs/datasets.md) | OakInk-V2 + MyDataset formats, index routing, annotation/pickle layouts, cap-mesh geometry, LH-cut trimming, adding a new dataset |
| [docs/experiments.md](docs/experiments.md) | Alcohol-burner capping (LOO), trajectory augmentation flags, experiment naming, SLURM workflow |
| [docs/baselines.md](docs/baselines.md) | dex-retargeting (DexPilot) baseline: best configuration, offline + live commands, wrist-fit placement, runtime cost, limitations |
| [maniptrans_envs/lib/envs/live/README.md](maniptrans_envs/lib/envs/live/README.md) | Live streaming (`--live`) architecture, transport, config, debugging |
| [README.md](README.md) | Installation, prerequisites, per-hand usage, extending, DexManipNet, citation |

## Project Structure

```
ManipTrans/
├── main/
│   ├── cfg/                        # Hydra configs (config.yaml, task/, rl_train/)
│   ├── dataset/                    # Dataset loaders (OakInk-V2, GRAB, FAVOR, MyDataset)
│   └── rl/                         # Training entry point, eval scoring
├── maniptrans_envs/lib/envs/
│   ├── tasks/                      # Isaac Gym environments
│   │   ├── dexhandimitator.py      # Stage 1: imitator env (single hand)
│   │   ├── dexhandmanip_bih.py     # Stage 2: bimanual residual env
│   │   └── dexhandmanip_sh.py      # Stage 2: single-hand residual env
│   ├── live/                       # Live-streaming target source (see live/README.md)
│   └── dexhands/                   # Hand configs (inspire, shadow, allegro, ...)
├── lib/rl/
│   ├── network_builder_residual_bih.py   # BiH residual network
│   ├── network_builder_residual_sh.py    # Single-hand residual network
│   └── res_models.py                     # Residual model wrappers
├── docs/                           # Detailed reference docs (see Documentation map)
├── DexManipNet/                    # DexManipNet dataset utilities
└── data/
    ├── OakInk-v2/                  # Raw OakInk-V2 annotations and meshes
    ├── my_dataset/                 # MyDataset (OptiTrack + AVP) captures
    ├── retargeting/                # Preprocessed retargeted data (generated)
    ├── body_utils/body_models/smplx/
    ├── mano_v1_2/
    └── smplx_extra/body_upper_idx.pt
```

## Two-Stage Architecture (summary)

- **Stage 1 — Imitator:** a per-hand policy (`DexHandImitatorRH/LH`) that tracks a reference demo trajectory. Checkpoints: `assets/imitator_{rh,lh}_{dexhand}.pth`.
- **Stage 2 — Residual policy (`ResDexHand`):** trained on top of **frozen** imitators. For bimanual tasks it is **three networks** — a frozen RH imitator, a frozen LH imitator, and a trained residual MLP. Each imitator sees only its own hand's obs half and emits a `base_action`; the residual MLP sees both halves (plus both base actions) and emits `delta_action`. The final sim action is `base_action + delta_action`, combined in `pre_physics_step`. Only the residual MLP is optimized, and it is what coordinates bimanual contact.

Networks: [lib/rl/network_builder_residual_bih.py](lib/rl/network_builder_residual_bih.py), [lib/rl/res_models.py](lib/rl/res_models.py). Full obs/target/dim tables, observation sources, and the annotated step loop: [docs/architecture.md](docs/architecture.md).

## Task indexing & datasets (summary)

Tasks are `HHHHH@S` (first 5 hash chars + stage index), e.g. `20aed@0`. The `_bih`/`_rh`/`_lh` suffixes are stripped before `--data_idx`/`dataIndices`. The index prefix picks the loader (`main/dataset/factory.py`):

| Index format | Dataset type |
|---|---|
| `HHHHH@S` | `oakink2` |
| `gN` | `grabdemo` |
| `vN` | `visionpro` |
| `m_...` (contains `m_`) | `mydataset` (OptiTrack + AVP capture) |
| `NM` (mirrored) | `oakink2_mirrored` |
| other | `favor` |

Each sequence merges **two** sources: raw anno → SMPLX/MANO → `mano_joints`/`wrist_*`/`obj_trajectory` (policy tracking targets); retargeted pkl → `opt_*` (sim reset init only). Full formats, bimanual detection, MyDataset (`m_` convention, object assets, cap geometry, LH cuts), and adding a dataset: [docs/datasets.md](docs/datasets.md).

## Training (summary)

```bash
# 1. Retarget (preprocessing) — bimanual = run both sides
python main/dataset/mano2dexhand.py --data_idx 20aed@0 --side right --dexhand inspire --headless --iter 7000
python main/dataset/mano2dexhand.py --data_idx 20aed@0 --side left  --dexhand inspire --headless --iter 7000

# 2. Stage-1 imitator (per hand)
python main/rl/train.py task=DexHandImitator dexhand=inspire side=RH headless=true \
    num_envs=4096 dataIndices=[g0] experiment=imitator_rh_inspire

# 3. Stage-2 residual (bimanual)
python main/rl/train.py task=ResDexHand dexhand=inspire side=BiH headless=true \
    num_envs=4096 learning_rate=2e-4 test=false randomStateInit=true dataIndices=[20aed@0] \
    rh_base_model_checkpoint=assets/imitator_rh_inspire.pth \
    lh_base_model_checkpoint=assets/imitator_lh_inspire.pth \
    early_stop_epochs=1000 actionsMovingAverage=0.6 experiment=cross_20aed@0_inspire

# 4. Test (headless=false, num_envs=4, test=true, randomStateInit=false, add checkpoint=...)
```

Output: `runs/<experiment>__<MM-DD-HH-MM-SS>/{config.yaml, demos.txt, nn/<experiment>.pth}`.

Note: if a checkpoint path or experiment name contains commas (multi-demo names), wrap the value in single quotes inside double quotes so Hydra doesn't parse it as a list: `"experiment='name_with,comma'"`. Single-hand training and per-hand variants: [README.md](README.md) / [docs/pipeline.md](docs/pipeline.md).

## Key CLI parameters (most used)

| Parameter | Default | Notes |
|---|---|---|
| `task` | — | `DexHandImitator` (Stage 1) or `ResDexHand` (Stage 2) |
| `side` | — | `RH`, `LH`, `BiH` |
| `dexhand` | `inspire` | `inspire`, `shadow`, `allegro`, `artimano`, `xhand`, `inspireftp` |
| `dataIndices` | — | e.g. `[20aed@0]`, `[g0]`, `[m_170805]` (strip `_bih`/`_rh`/`_lh`) |
| `num_envs` | `8192` | ~4096 train, 4–16 test |
| `actionsMovingAverage` | `1.0` | temporal action smoothing; **prefer 0.6 for BiH** |
| `randomStateInit` | `True` | RSI — true for train, false for test |
| `headless` | `True` | disable rendering (true for training) |
| `checkpoint` | `''` | path to `.pth` to resume/test |
| `learning_rate` | `5e-4` | `2e-4` typical for the residual policy |
| `early_stop_epochs` | huge | epochs without improvement before stopping (~1000 for complex tasks) |
| `zeroResidual` | `False` | run imitator-only baseline |
| `usePIDControl` | `False` | PID wrist control instead of direct position |

Full parameter table (augmentation, noise, rollouts, W&B, live flags, etc.): [docs/pipeline.md](docs/pipeline.md#full-parameter-reference).

## Live streaming (summary)

`live=true` feeds the policy a **live** hand+object target stream (Apple Vision Pro hands + OptiTrack/Motive objects), or replays a recorded `.pkl` to test the pipeline. A reference demo is still loaded (`dataIndices`) for assets/BPS/reset init, but its target slots are overwritten in place every step by the latest live frame. Transport is ZeroMQ PUB/SUB + `msgpack`; `liveBuffered=true` = FIFO faithful replay, `false` = newest-only teleop. Full architecture, transport, config knobs, and debugging: [maniptrans_envs/lib/envs/live/README.md](maniptrans_envs/lib/envs/live/README.md).

## Current experiment (summary)

Bimanual alcohol-burner capping (`b5fa3@10_bih`) under leave-one-out evaluation, with trajectory augmentation and a SLURM submission workflow. Full setup (LOO demo split, augmentation flags, naming convention, SLURM steps): [docs/experiments.md](docs/experiments.md).

## Reward & domain randomization (summary)

The imitation reward is computed per hand and summed for bimanual — it tracks wrist pose/vel, finger joint positions, and object pose/vel against the demo, with failure thresholds tightened over training. Domain randomization (gravity, object friction) is driven by `randomization_params` in the task YAML. Details: [docs/architecture.md](docs/architecture.md#reward-imitation).
