# ManipTrans Documentation

**ManipTrans: Efficient Dexterous Bimanual Manipulation Transfer via Residual Learning** (CVPR 2025).

Transfers human hand–object manipulation demonstrations to dexterous robot hands using a
two-stage residual-learning approach in Isaac Gym.

This `docs/` folder is a synthesized, navigable entry point. It does **not** replace the
authoritative sources below — it summarizes them and links back to them.

---

## Start here

| Doc | What it covers |
|---|---|
| [pipeline.md](pipeline.md) | End-to-end pipeline: data → retargeting → Stage-1 imitator → Stage-2 residual → test/eval → live streaming, plus the runtime step loop, observations, reward, and the **full CLI parameter table**. |
| [architecture.md](architecture.md) | Two-stage network internals, observation/target/dim tables, step-loop diagram, reward, dexterous-hand configs, domain randomization. |
| [datasets.md](datasets.md) | OakInk-V2 + MyDataset formats, index routing, annotation/pickle layouts, cap-mesh geometry, LH-cut trimming, adding a new dataset. |
| [experiments.md](experiments.md) | Alcohol-burner capping (LOO), trajectory augmentation flags, imitator-only baseline, experiment naming, SLURM workflow. |

## Authoritative sources (canonical, kept up to date)

| Source | Scope |
|---|---|
| [`../README.md`](../README.md) | Installation, prerequisites (datasets, MANO/SMPL-X, checkpoints), full per-hand usage commands, extending to new datasets / hands, DexManipNet, citation, license. |
| [`../CLAUDE.md`](../CLAUDE.md) | Deep reference: architecture internals, observation/target layouts, dataset formats (OakInk-V2, MyDataset), reward, domain randomization, the current capping experiment, SLURM workflow. |
| [`../maniptrans_envs/lib/envs/live/README.md`](../maniptrans_envs/lib/envs/live/README.md) | Live-streaming architecture (`--live`): publisher/consumer seams, `LiveTargetSource`, wire protocol, config knobs, debugging. |
| `.claude/skills/coding-rules/SKILL.md` | Project coding conventions (variable naming). Invoke with `/coding-rules`. |

---

## The two-stage idea in one paragraph

Each hand first gets a **frozen per-hand imitator** (Stage 1) that tracks a reference
trajectory. A **residual policy** (Stage 2) is then trained on top of the frozen imitators; it
observes both hands jointly and emits a correction (`delta_action`) that is added to the
imitator base actions before being sent to the simulator. For bimanual tasks the policy is
three networks — a frozen RH imitator, a frozen LH imitator, and the trained residual MLP — and
only the residual MLP is optimized. See [pipeline.md](pipeline.md#stage-2-residual-policy).

## Quick command reference

```bash
# 1. Retarget a demo (Stage 0 preprocessing), bimanual = run both sides
python main/dataset/mano2dexhand.py --data_idx 20aed@0 --side right --dexhand inspire --headless --iter 7000
python main/dataset/mano2dexhand.py --data_idx 20aed@0 --side left  --dexhand inspire --headless --iter 7000

# 2. Train the Stage-1 imitator (per hand)
python main/rl/train.py task=DexHandImitator dexhand=inspire side=RH headless=true \
    num_envs=4096 dataIndices=[g0] experiment=imitator_rh_inspire

# 3. Train the Stage-2 residual policy (bimanual)
python main/rl/train.py task=ResDexHand dexhand=inspire side=BiH headless=true \
    num_envs=4096 learning_rate=2e-4 test=false randomStateInit=true dataIndices=[20aed@0] \
    rh_base_model_checkpoint=assets/imitator_rh_inspire.pth \
    lh_base_model_checkpoint=assets/imitator_lh_inspire.pth \
    early_stop_epochs=1000 actionsMovingAverage=0.6 experiment=cross_20aed@0_inspire
```

Full parameter tables and testing/live commands are in [pipeline.md](pipeline.md).
