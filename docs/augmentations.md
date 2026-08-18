# Augmentation & Randomization

Every knob that perturbs what the policy sees or does during training, in one place: spatial
trajectory augmentation, target noise, causal velocities, random action masking, domain
randomization, and reference state initialization.

The organizing question for all of them is **when** they are applied — at demo load time (baked
into a variant, fixed for the whole run) or per step (resampled continuously). Getting this wrong
is the usual source of confusion, so it leads.

> `experiments.md` also has a short augmentation section. It predates several renames — this file
> is the current reference. See [Corrections to experiments.md](#corrections-to-experimentsmd).

---

## When each one is applied

| Knob | Applied | Resampled |
|---|---|---|
| `useTrajAug` + the `*_Aug` flags | **load time**, in `_create_envs` | never — `numTrajAug` variants are baked once |
| `jointNoiseCm` | **load time**, per augmented variant | never — each variant carries one fixed noise draw |
| `causal` / `causalVelMode` / `causalEmaAlpha` | **load time**, in the dataset loader | never |
| `actionMaskProb` (random action masking) | **per step**, in `pre_physics_step` | every step, per env |
| `task.randomization_params` (gravity, friction) | **per episode** | every `frequency: 32` steps |
| `randomStateInit` (RSI) | **per reset** | every episode |

The first three are effectively *dataset* properties. Only action masking and domain randomization
vary within a single env's lifetime.

---

## Spatial trajectory augmentation

Pre-generates `numTrajAug` rigidly-transformed copies of each demo at `_create_envs` time
(`dexhandmanip_bih.py:608-727`). Env `i` is permanently bound to variant `i % numTrajAug`, and the
object actor is spawned from that variant — so a variant is fixed for the whole run, not
resampled per episode.

`useTrajAug=true` is the master switch. With it off, `numTrajAug` is forced to 1 and only the raw
demo is used.

### The flags

| Flag | Transforms | Pivot | Angle range |
|---|---|---|---|
| `RH_LH_Table_Center_Aug` | both hands + both objects | table center (XY) | U(−15°, +30°) |
| `RH_LObj_Center_Aug` | **RH demo only** | LH object center, per frame | per-demo, rejection-sampled from U(−15°, +15°) |
| `RH_RObj_Center_Aug` | **RH demo only** | RH object center, per frame | U(−15°, +30°) |
| `LH_LObj_Center_Aug` | **LH demo only** (left hand + left object, rigidly) | LH object position, per frame | U(−15°, +30°) |
| `useObjRotationAug` | each object spun in place, hands untouched | each object's own center | U(±`objRotationAugMaxAngleDeg`), default 90° |

The shared transform also carries a ±5 cm XY translation (`_sample_aug_transform`,
`dexhandmanip_bih.py:1093`), but table-center aug deliberately drops it — the dataloader re-centers
each object onto the table center downstream, so the shift would be undone anyway.

### Order and exclusivity

Enabled augs **chain** in this fixed order (`dexhandmanip_bih.py:677-701`), each operating on the
previous result:

1. `useObjRotationAug`
2. `RH_LObj_Center_Aug` **XOR** `RH_RObj_Center_Aug`
3. `LH_LObj_Center_Aug`
4. `RH_LH_Table_Center_Aug`
5. `jointNoiseCm`

Step 2 is the exception to chaining. Both flags rotate the RH demo, so applying both would
double-rotate it into hand collisions. They are **mutually exclusive**: when both are set, one is
chosen at random *per augmented variant*, so a `numTrajAug=400` run gets roughly 200 of each.

### Defaults and the backward-compatibility fallback

`RH_LH_Table_Center_Aug` defaults to **true** when none of the per-object augs are selected
(`dexhandmanip_bih.py:618-621`). So `useTrajAug=true` alone gives table-center aug, which is what
older experiment names assume.

### `rhLObjCenterAugMaxXShift`

Read only when `RH_LObj_Center_Aug=true`. A budget in metres (default `0.05`) on how far the aug
may move the demo's **starting** position along world X.

`RH_LObj_Center_Aug` rotates the RH demo about the LH object, so the start-pose shift is linear in
the starting lever arm — the same angle moves a demo that starts far from the pivot much further
than one that starts close. The yaw is therefore drawn **per demo** by rejection sampling, keeping
the first draw whose start-pose |ΔX| fits the budget. Far-starting demos automatically end up with
smaller angles.

Only the *start* pose is bounded; the augmented trajectory may diverge further later, which is what
leaves the aug any effect at all.

### Test mode

With `test=true` and `numTrajAug > 1`, variant 0 (the unaugmented original) is skipped so every env
uses an augmented variant, and the RNG is seeded from `cfg.seed` for reproducibility
(`dexhandmanip_bih.py:622-635, 711-714`).

---

## Target noise — `jointNoiseCm`

Perturbs the MANO wrist position and every joint keypoint, simulating hand-pose-estimator error.
`_apply_joint_noise`, `dexhandmanip_bih.py:402`.

Two properties that are easy to get wrong:

- **The distribution is uniform, not Gaussian.** The code is
  `x + (rand_like(x) * 2σ − σ)`, i.e. `U[−σ, +σ]` with `σ = jointNoiseCm / 100` metres.
- **It is applied at load time, per augmented variant** — inside the aug loop, after all spatial
  transforms. Each of the `numTrajAug` variants carries one fixed noise draw for the whole run. It
  is *not* resampled per step or per episode.

### `failureThresholdNoiseCompensation`

A multiplier on the per-finger and object failure thresholds (default `1.0` = unchanged). Injected
joint noise moves the target, so an unmodified threshold would terminate episodes for tracking
error the policy cannot avoid. Raise this alongside `jointNoiseCm` — the current capping run pairs
`jointNoiseCm=0.5` with `failureThresholdNoiseCompensation=1.5`.

---

## Causal velocities — `causal`

Not augmentation strictly, but it changes the target stream the same way. Demo velocities are by
default computed with a **non-causal** Gaussian filter that looks ahead in time. Live teleop cannot
look ahead, so `causal=true` recomputes them the way `LiveTargetSource` does, closing a
train/deploy mismatch (`main/dataset/base.py:100`).

| Knob | Options | Meaning |
|---|---|---|
| `causal` | `False` (default) / `True` | use causal velocity estimation |
| `causalVelMode` | `pos_ema` (default) / `vel_ema` | `pos_ema` low-passes positions then differentiates; `vel_ema` differentiates then smooths the velocity |
| `causalEmaAlpha` | `0.3` | smoothing strength for whichever EMA the mode selects |

Angular velocity always uses `vel_ema` regardless of the mode.

**Enable this for anything destined for live teleop.** The current capping run sets `causal=true`.

---

## Random action masking — `actionMaskProb`

Freezes a few DoFs at their previously commanded joint target, so the policy cannot rely on every
joint responding on time (TeleDexter, arXiv 2607.11481 Sec. C.8). This is the only augmentation
that acts on the **action** channel rather than the observation.

| Knob | Default | Meaning |
|---|---|---|
| `actionMaskProb` | `0.0` (off; paper uses `0.15`) | per-step probability of starting a freeze, when none is active |
| `actionMaskNumDofs` | `3` | DoFs frozen **per hand**, drawn uniformly without replacement |
| `actionMaskMaxDuration` | `10` | the ceiling `d_max` saturates at |
| `actionMaskRampSteps` | `64000` | control steps over which `d_max` ramps 1 → `actionMaskMaxDuration` |

Mechanism (`_refresh_action_mask`, `dexhandmanip_bih.py:2773`): each step, envs with no active mask
draw one with probability `actionMaskProb`. A fresh mask picks `actionMaskNumDofs` DoFs per hand and
a duration `d ~ U{1, d_max}`. For those `d` steps the masked DoFs execute the previous command
instead of the new one — applied *after* the moving average and clamp, so what freezes is the final
commanded joint target. `prev_targets` then receives the held value, so a multi-step freeze holds
one command throughout rather than drifting.

`d_max` follows the paper's cubic schedule: σ decays linearly 1 → 0.7 over `actionMaskRampSteps`,
and `d_max = 1 + (max−1)·(1−σ³)/(1−0.7³)`. The curve is **concave** — it rises faster than linear
early and flattens near the end:

| progress | 0 | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|
| `d_max` | 1 | 3 | 6 | 8 | 10 |

`d_max` is only the upper bound; each draw is uniform over `{1, d_max}`, so one-step freezes keep
occurring at full difficulty and the *mean* duration goes 1 → 5.5.

**Two things to know before enabling:**

- At `actionMaskProb=0.15` roughly **half of all env-steps** have some DoF frozen in steady state
  (p=0.15 with mean duration 5.5). That is the paper's own setting, but "15%" understates it.
- `control_steps` restarts at 0 whenever the env is constructed, so **resuming from a checkpoint
  replays the ramp** from `d_max = 1`.

Training only — forced to `0.0` when `test=true`. Cleared per env on reset, so a fresh episode
never inherits a freeze against a zeroed `prev_targets`.

---

## Domain randomization

Driven by `task.randomization_params` in `main/cfg/task/ResDexHand.yaml:187-235`, and gated off
under `test=true`. Resampled every `frequency: 32` steps.

| Parameter | Operation | Schedule |
|---|---|---|
| gravity | scaling | `linear_decay` over 1920 steps, from 0 |
| `manip_obj{,_rh,_lh}` shape friction | scaling, 250 buckets | `linear_decay` over 1920 steps, init 3, bounds [1.0, 6.0] |

Narrower than either reference paper — TeleDexter also randomizes hand mass, `Kp`/`Kd`, restitution,
mesh scale, external forces, sensing noise, and observation latency; VisualDexterity randomizes
object mass/friction/restitution/scale plus disturbance forces.

---

## Reference state initialization — `randomStateInit`

`True` for training, `False` for test. On reset, the env starts from a random frame in the first 98%
of that env's own trajectory rather than always frame 0 (`dexhandmanip_bih.py:2357-2380`). Sampling
is **phase-normalized per env** — a fraction of that env's `seq_len`, not a raw frame index — so
demos of different lengths are sampled comparably.

---

## What the current capping run uses

From `slurm/alps/train_maniptrans_inspire.run`:

```bash
useTrajAug=true \
RH_LH_Table_Center_Aug=true \
RH_LObj_Center_Aug=true \
RH_RObj_Center_Aug=true \
LH_LObj_Center_Aug=true \
numTrajAug=400 \
jointNoiseCm=0.5 \
failureThresholdNoiseCompensation=1.5 \
causal=true \
actionMaskProb=${ACTION_MASK_PROB} \
```

With both RH-center flags on, each of the 400 variants randomly picks one of them, then gets
LH-about-LH-obj and table-center applied on top, then a fixed uniform ±0.5 cm joint-noise draw.

---

## Corrections to `experiments.md`

Its augmentation section is out of date on three counts:

1. **Flag names are stale.** `useTableCenterAug` → `RH_LH_Table_Center_Aug`, `useLHObjCenterAug` →
   `RH_LObj_Center_Aug`, `useRHObjCenterAug` → `RH_RObj_Center_Aug`, `useLHAboutLHObjAug` →
   `LH_LObj_Center_Aug`.
2. **`jointNoiseCm` is uniform, not Gaussian**, and is applied per *variant at load time*, not per
   step.
3. **The RH-center augs do not chain.** They are mutually exclusive, one picked at random per
   variant.

It also predates `useObjRotationAug`, `rhLObjCenterAugMaxXShift`, the `causal*` knobs, and random
action masking.
