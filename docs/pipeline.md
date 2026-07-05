# ManipTrans Pipeline

End-to-end flow from a human hand–object demonstration to a trained dexterous-robot policy, and
finally to test/eval or live teleoperation. Synthesized from [`../README.md`](../README.md),
[`../CLAUDE.md`](../CLAUDE.md), and
[`../maniptrans_envs/lib/envs/live/README.md`](../maniptrans_envs/lib/envs/live/README.md) —
consult those for the authoritative details.

```
Raw demo data  (OakInk-V2 / MyDataset / GRAB / FAVOR)
      │
      ▼   Step 1 — Retargeting        main/dataset/mano2dexhand.py
Retargeted pkl (opt_wrist_pos/rot, opt_dof_pos, opt_joints_pos)   ── used for sim reset init ONLY
      │
      ▼   Step 2 — Stage-1 Imitator   task=DexHandImitator
Frozen per-hand imitator checkpoints   assets/imitator_{rh,lh}_{dexhand}.pth
      │
      ▼   Step 3 — Stage-2 Residual   task=ResDexHand   (imitators frozen; residual MLP trained)
Residual policy checkpoint   runs/<experiment>__<date>/nn/<experiment>.pth
      │
      ├──►  Test / eval          test=true          (held-out demos, or zeroResidual baseline)
      └──►  Live streaming       live=true          (AVP + OptiTrack, or a mock .pkl replay)
```

---

## Step 0 — Data sources & indexing

Raw demonstrations are loaded per sequence and merged from **two** sources into one `data` dict:

1. **Raw annotation** → SMPLX/MANO forward pass → human fingertip/joint world positions. Stored
   as `mano_joints`, `wrist_pos`, `wrist_rot`, `obj_trajectory`. These are the **policy tracking
   targets**.
2. **Retargeted pkl** (produced in Step 1) → adds `opt_wrist_pos`, `opt_wrist_rot`, `opt_dof_pos`,
   `opt_joints_pos`. These are used for **sim reset initialization only** — never as tracking
   targets.

| Key | Source | Used for |
|---|---|---|
| `mano_joints`, `wrist_pos`, `wrist_rot`, `obj_trajectory` | SMPLX/MANO on raw anno | policy tracking targets + reward |
| `opt_wrist_pos`, `opt_wrist_rot`, `opt_dof_pos`, `opt_joints_pos` | retargeted pkl | sim reset init only |

The policy learns to move the **dexterous-hand fingertips** (from live sim state) to where the
**human MANO fingertips** were in the demo; the retargeted data only provides a good starting pose.

### Dataset routing (`main/dataset/factory.py`)

The index prefix picks the dataset loader; the factory appends `_rh` / `_lh`.

| Index format | Dataset type | Example |
|---|---|---|
| `HHHHH@S` | `oakink2` | `20aed@0` (primitive 0 of sequence hash `20aed…`) |
| `gN` | `grabdemo` | `g0` |
| `vN` | `visionpro` | `v3` |
| contains `m_` | `mydataset` (OptiTrack + Apple Vision Pro capture, 60 Hz) | `m_170805` |
| `NM` (mirrored) | `oakink2_mirrored` | |
| other | `favor` | |

- OakInk-V2 is 120 Hz and is downsampled `skip=2` → 60 Hz; MyDataset is already 60 Hz (`skip=1`).
- Bimanual vs. single-hand is determined by the sequence's `program_info` intervals (see
  [datasets.md](datasets.md#bimanual-vs-single-hand-tasks)).
- Detailed pkl/annotation layouts: [datasets.md](datasets.md) (OakInk-V2 and MyDataset sections).

---

## Step 1 — Retargeting (preprocessing)

Optimizes a collision-free trajectory from MANO to the dexterous hand, giving near-object,
non-colliding states for reference state initialization (RSI). Output goes to
`data/retargeting/<dataset>/mano2{dexhand}/`.

```bash
# Single hand
python main/dataset/mano2dexhand.py --data_idx g0 --dexhand inspire --headless --iter 2000

# Bimanual — run BOTH sides
python main/dataset/mano2dexhand.py --data_idx 20aed@0 --side right --dexhand inspire --headless --iter 7000
python main/dataset/mano2dexhand.py --data_idx 20aed@0 --side left  --dexhand inspire --headless --iter 7000
```

Retargeted pkl contents (per hand, `T` frames):

```python
{
    'opt_wrist_pos':  ndarray[T, 3],        # wrist position (world, meters)
    'opt_wrist_rot':  ndarray[T, 3],        # wrist rotation (axis-angle)
    'opt_dof_pos':    ndarray[T, n_dofs],   # finger joint angles
    'opt_joints_pos': ndarray[T, n_bodies, 3],  # rigid-body positions (world)
}
```

---

## Step 2 — Stage-1 Imitator

A per-hand imitation policy (`DexHandImitatorRH/LH` env) that tracks the reference trajectory.

- **Inputs** (per hand): `proprioception` (live joint + wrist state), `privileged` (velocities,
  object pose/vel, tip contact force, object mass/com), `target` (look-ahead demo deltas + BPS).
- **Output** (per hand): `[wrist_pos(3), wrist_rot(3 axis-angle), finger_dofs(n_dofs)]` =
  `6 + n_dofs` dims (variants for `useQuatRot` / `usePIDControl`).
- **Checkpoints**: `assets/imitator_{rh,lh}_{dexhand}.pth`.

```bash
python main/rl/train.py task=DexHandImitator dexhand=inspire side=RH headless=true \
    num_envs=4096 dataIndices=[g0] experiment=imitator_rh_inspire
```

Pretrained imitator checkpoints can also be downloaded (see [`../README.md`](../README.md)
Prerequisites → Imitator Checkpoints).

---

## Step 2 (Stage 2) — Residual Policy

The residual policy (`ResDexHand`) trains on top of **frozen** imitators. For bimanual tasks both
RH and LH imitators are loaded and kept in eval mode.

**Forward pass (bimanual) — three networks:**

1. Full obs is `[rh_obs, lh_obs]` concatenated per key.
2. Frozen **RH imitator** reads the RH slice → `rh_base_action`.
3. Frozen **LH imitator** reads the LH slice → `lh_base_action`.
4. Trained **residual MLP** reads `encode(full obs)` + both base actions → `delta_action`.
5. Final sim action = `base_action + delta_action` (split/combined in `pre_physics_step`).

Only the residual MLP is optimized; each frozen imitator sees only its own hand, so the residual
MLP is what coordinates bimanual contact. Code:
[`lib/rl/network_builder_residual_bih.py`](../lib/rl/network_builder_residual_bih.py),
[`lib/rl/res_models.py`](../lib/rl/res_models.py).

```bash
# Single hand
python main/rl/train.py task=ResDexHand dexhand=inspire side=RH headless=true \
    num_envs=4096 learning_rate=2e-4 test=false randomStateInit=true \
    rh_base_model_checkpoint=assets/imitator_rh_inspire.pth \
    lh_base_model_checkpoint=assets/imitator_lh_inspire.pth \
    dataIndices=[g0] early_stop_epochs=100 actionsMovingAverage=0.4 experiment=cross_g0_inspire

# Bimanual
python main/rl/train.py task=ResDexHand dexhand=inspire side=BiH headless=true \
    num_envs=4096 learning_rate=2e-4 test=false randomStateInit=true dataIndices=[20aed@0] \
    rh_base_model_checkpoint=assets/imitator_rh_inspire.pth \
    lh_base_model_checkpoint=assets/imitator_lh_inspire.pth \
    early_stop_epochs=1000 actionsMovingAverage=0.6 experiment=cross_20aed@0_inspire
```

Output layout: `runs/<experiment>__<MM-DD-HH-MM-SS>/{config.yaml, demos.txt, nn/<experiment>.pth}`.

---

## Step 3 — Test / Eval

```bash
python main/rl/train.py task=ResDexHand dexhand=inspire side=BiH headless=false \
    num_envs=4 test=true randomStateInit=false dataIndices=[20aed@0] actionsMovingAverage=0.6 \
    rh_base_model_checkpoint=assets/imitator_rh_inspire.pth \
    lh_base_model_checkpoint=assets/imitator_lh_inspire.pth \
    "checkpoint='runs/cross_20aed@0_inspire__<date>/nn/cross_20aed@0_inspire.pth'"
```

- **Imitator-only baseline:** add `zeroResidual=true` to zero the residual delta and run only the
  frozen imitator (no retraining).
- **Comma quoting:** if a checkpoint path or experiment name contains commas (multi-demo names),
  wrap the value in single quotes inside double quotes so Hydra doesn't parse it as a list.
- Rollouts can be saved for offline eval/distillation via `save_rollouts=true` (see the parameter
  table below).

---

## Step 3 (alt) — Live streaming inference (`live=true`)

Instead of a preloaded demo, feed the policy a **live** hand+object target stream (Apple Vision
Pro hands + OptiTrack/Motive objects), or replay a recorded `.pkl` to test the pipeline. Full
architecture: [`../maniptrans_envs/lib/envs/live/README.md`](../maniptrans_envs/lib/envs/live/README.md).

**How it fits the pipeline:** a reference demo is still loaded (`dataIndices` → assets, BPS,
`opt_*` reset init, buffer shapes), but its target slots are **overwritten in place every step**
by the latest live frame:

- `live/live_target_source.py` — `LiveTargetSource`: ZMQ SUB, unpacks `wire` frames, maps
  OptiTrack→gym in `_transform_side()` (mirrors the `my_dataset_{RH,LH}` loaders), computes
  causal-EMA velocities and fingertip distances; `latest()` returns one gym-frame target/step.
- `tasks/dexhandmanip_bih.py` — `_inject_live()` overwrites the `demo_data_{rh,lh}` target slots
  (called at the top of `post_physics_step()`); auto-reset is disabled and `progress_buf` is
  clamped into the tiny 4-frame buffer.

Transport is ZeroMQ PUB/SUB + `msgpack`. `liveBuffered=false` → newest-only/CONFLATE (real teleop,
may skip frames); `liveBuffered=true` → FIFO, consume every frame in order (faithful replay).

```bash
# terminal 1 — mock publisher: dump one pass fast so the sim never starves
python <Motion_Capture>/src/live_streaming/debug/mock_publish.py \
    --pkl m_170805 --once --rate-hz 200 --addr 0.0.0.0 --port 5555

# terminal 2 — sim consumes every frame in order (liveBuffered=true), same demo as dataIndices
python main/rl/train.py task=ResDexHand dexhand=inspire side=BiH headless=false \
    num_envs=2 test=true randomStateInit=false \
    live=true liveBuffered=true liveAddr=127.0.0.1 livePort=5555 \
    dataIndices=[m_170805] \
    rh_base_model_checkpoint=assets/imitator_rh_inspire.pth \
    lh_base_model_checkpoint=assets/imitator_lh_inspire.pth \
    "checkpoint='runs/capping_alcohol_burner_AUG_noisem_161551,m_170401,m_170527,m_170654,m_170753__06-30-16-52-51/nn/last_capping_alcohol_burner_AUG_noisem_161551,m_170401,m_170527,m_170654,m_170753_ep_900_rew_1844.3008_sr_0.4311926066875458_fr_0.5688073039054871.pth'"
```
Running from the real live stream (AVP hands + OptiTrack/Motive objects, teleop):

```bash
# terminal 1 — real publisher on the laptop (AVP + Motive → wire), binds all interfaces
python src/live_streaming/live_publish.py --addr 0.0.0.0 --port 5555

# terminal 2 — sim, newest-only/CONFLATE (liveBuffered=false) for real-time teleop;
#              liveAddr = the laptop's LAN IP. Start the publisher first (start() waits ~10 s).
python main/rl/train.py task=ResDexHand dexhand=inspire side=BiH headless=false \
    num_envs=2 test=true randomStateInit=false \
    live=true liveBuffered=false liveAddr=10.50.227.40 livePort=5555 \
    dataIndices=[m_170805] \
    rh_base_model_checkpoint=assets/imitator_rh_inspire.pth \
    lh_base_model_checkpoint=assets/imitator_lh_inspire.pth \
    "checkpoint='runs/capping_alcohol_burner_AUG_noisem_161551,m_170401,m_170527,m_170654,m_170753__06-30-16-52-51/nn/last_capping_alcohol_burner_AUG_noisem_161551,m_170401,m_170527,m_170654,m_170753_ep_900_rew_1844.3008_sr_0.4311926066875458_fr_0.5688073039054871.pth'"
```

Bring-up gotchas (rate mismatch, `num_envs ≥ 2`, `progress_buf` clamp, hand key mapping) are in
the live README's debugging section.

---

## Runtime step loop

One `env.step` runs at 60 Hz (`dt = 1/60`, 2 substeps, `controlFrequencyInv = 1` → one
`gym.simulate()` per action). Each step:

1. **Observe** — build the obs dict from live PhysX tensors (`proprioception`, `privileged`) plus
   demo look-ahead (`target`); for BiH the obs is `[rh ‖ lh]`.
2. **Policy** — frozen imitators produce `base_action`; residual MLP produces `delta_action`; the
   env emits `[base ‖ residual]`.
3. **Decode** (`pre_physics_step`) — split base/residual, sum, clamp, scale to joint limits, apply
   a moving average (`actionsMovingAverage`); wrist via PID force/torque or direct target.
4. **Apply** — PD finger targets + wrist forces to the sim.
5. **Physics** — `gym.simulate()` (writes net contact force).
6. **Read back** (`post_physics_step`) — refresh tensors, compute the next obs and reward
   (tracking the demo), check failure/success/timeout, reset.

Full annotated diagram: [architecture.md](architecture.md#step-loop).

---

## Observations & reward (quick reference)

- **`proprioception`** and **`privileged`** are pure live sim state (where the robot is now).
- **`target`** carries demo intent (where it should go); its `delta_*` terms are `demo − live`,
  read at `progress_buf + 1`. It also includes a 128-dim BPS object-shape encoding.
- **Reward** is computed per hand and summed for bimanual — it tracks wrist pose/vel, finger joint
  positions, and object pose/vel against the demo. Failure triggers when tracked quantities exceed
  per-joint thresholds (tightened over training). Optional extras for capping tasks:
  `usePenKeypointReward`, `useCoaxialReward`.

Exact dims, ordering, and per-value sources: [architecture.md](architecture.md#target-observation).

---

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
| `live` / `liveBuffered` / `liveAddr` / `livePort` | see live docs | live-streaming path |

The full table (augmentation, noise, rollouts, W&B, etc.) is below.

---

## Full parameter reference

| Parameter | Default | Description |
|---|---|---|
| `task` | — | `DexHandImitator` or `ResDexHand` |
| `side` | — | `RH`, `LH`, `BiH` |
| `dexhand` | `inspire` | `inspire`, `shadow`, `allegro`, `artimano`, `xhand`, `inspireftp` |
| `dataIndices` | — | List of task indices, e.g. `[20aed@0]` or `[g0]`. Strip `_bih`/`_rh`/`_lh` suffixes. |
| `num_envs` | `8192` | Parallel envs (8192 typical for training, 4 for testing) |
| `maxDemoLength` | `None` | Cap all demos to this many frames (useful for balanced multi-demo training) |
| `early_stop_epochs` | `9999999` | Epochs without improvement before stopping (1000 for complex tasks) |
| `actionsMovingAverage` | `1.0` | Temporal smoothing on actions. **Prefer 0.6 for BiH** — empirically better than 0.4 or 1.0. |
| `randomStateInit` | `True` | RSI — start from random demo frame (true for train, false for test) |
| `usePIDControl` | `False` | Use PID wrist control instead of direct position |
| `headless` | `True` | Disable rendering (true for training) |
| `checkpoint` | `''` | Path to `.pth` to resume or test |
| `learning_rate` | `5e-4` | PPO learning rate (2e-4 typical for residual policy) |
| `max_iterations` | `9999999` | Hard cap on training iterations |
| `wandb_activate` | `False` | Enable Weights & Biases logging |
| `wandb_entity` | `None` | W&B entity (username or team) |
| `wandb_project` | `None` | W&B project name |
| `save_rollouts` | `False` | Save rollout episodes to `rollouts.hdf5` (for eval or distillation) |
| `num_rollouts_to_save` | `10000` | Max rollouts to write to HDF5 before stopping |
| `num_rollouts_to_run` | `1e10` | Max completed episodes before stopping; must be `> num_envs * 2` to pass warmup |
| `save_successful_rollouts_only` | `True` | If false, save both successful and failed rollouts |
| `useTrajAug` | `False` | Enable trajectory augmentation (random XY offset ±3cm, Z-rotation ±10°) at load time. **Must be `true` for any augmentation to occur** — it is the master switch; `useLHObjCenterAug` and `numTrajAug` have no effect without it. |
| `numTrajAug` | `20` | Number of pre-augmented demo versions per demo (envs cycle through these) |
| `useLHObjCenterAug` | `False` | Rotate augmentation around the left-hand object center instead of the table center. Requires `useTrajAug=true`. |
| `jointNoiseCm` | `0.0` | Gaussian noise std (cm) added to MANO joint keypoint positions — simulates hand pose estimator error |
| `useCoaxialReward` | `False` | Extra reward for pen/cap Z-axis alignment (pen capping tasks) |
| `usePenKeypointReward` | `False` | Extra reward for pen tip proximity to cap opening |
| `evalStartFrame` | `0` | Frame index to start evaluation rollouts from |
| `live` | `False` | Stream targets live (AVP+Motive, or a mock replay) instead of the demo. See [live streaming](../maniptrans_envs/lib/envs/live/README.md). |
| `liveAddr` | `10.50.227.40` | Address the desktop ZMQ SUB connects to (laptop IP for teleop; `127.0.0.1` for local replay) |
| `livePort` | `5555` | ZMQ port for the live stream |
| `liveBuffered` | `False` | `True` = FIFO, consume **every** published frame in order (faithful replay); `False` = newest-only/CONFLATE (real-time teleop, may skip frames) |

Augmentation flag interactions (`useTableCenterAug`, `useRHObjCenterAug`, `useLHAboutLHObjAug`,
chaining order) are documented in [experiments.md](experiments.md#trajectory-augmentation).
