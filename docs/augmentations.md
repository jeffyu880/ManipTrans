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

## Consecutive subgoal tracking — `subgoalTracking`

Stops the demo pointer advancing one frame per control step. Instead it **holds** on the current
subgoal and **jumps** `Δk` frames only once both hands have held that subgoal inside tolerance for
`N_stay` consecutive frames (TeleDexter, arXiv 2607.11481 Sec. 3.1). Between subgoals the policy is
free to leave the reference and find its own contact strategy — which is what the paper credits for
in-hand reorientation, finger gaiting and regrasping. Its Tab. 4, on held-out references: episode
length 373–379 vs 89–131, subgoals reached 33–187 vs ~2.7, against dense frame-wise tracking.

Unlike everything else on this page this is not an *augmentation* — it changes the task the policy
is being asked to solve. It is documented here because it shares the action mask's curriculum
machinery and the same on/off discipline.

| Knob | Default | Meaning |
|---|---|---|
| `subgoalTracking` | `False` (off) | master switch; off reproduces dense tracking exactly |
| `subgoalTipTol` | `0.03` | m, per fingertip group |
| `subgoalObjPosTol` | `0.01` | m |
| `subgoalObjRotTolDeg` | `10.0` | deg |
| `subgoalStayMin` / `Max` | `5` / `15` | `N_stay ~ U{min,max}`, resampled after each hit |
| `subgoalStepMin` / `Max` | `8` / `24` | `Δk` floor and its curriculum ceiling, in reference frames |
| `subgoalRampSteps` | `64000` | control steps over which the `Δk` ceiling ramps |
| `subgoalFailTolerance` | `3.5` | `η_fail`: cumulative out-of-tolerance frames allowed per unit of `Δk` since the last hit |
| `subgoalScoreScale` | `1.5` | `α_s`, outer scale on the sparse bonus |
| `subgoalTipWeight` | `0.5` | `w_tip`, per fingertip group |
| `subgoalObjWeight` | `2.0` | `w_obj`, on object position and rotation |
| `subgoalTimePenalty` | `0.1` | constant per-step cost |
| `subgoalCrossTrajProb` | `0.0` (off) | chance a hit switches the env onto **another demo** instead of hopping `Δk` |
| `subgoalCrossTrajScope` | `aug` | candidate pool: `aug` = same augmentation variant, other demo; `any` = any other slot |
| `subgoalCrossTrajStepWeight` | `100.0` | flat `w_step` for a subgoal reached after a switch |
| `subgoalCrossTrajFailBudget` | `175` | flat `n_fail` frames for the transition after a switch |
| `subgoalMaxEpisodeSteps` | `600` | elapsed-step cap, since `progress_buf` no longer paces the episode |
| `denseRewardScale` | `1.0` (no-op) | `α_dense`; use **~0.05** with subgoal tracking on |

**Mechanism.** `advance_subgoals` (`dexhandmanip_bih.py`) runs at the *top* of `post_physics_step`,
before `compute_observations`, so the policy is shown the next subgoal on the same step it earned
the current one. It snapshots the pre-jump index into `subgoal_active_idx`, which is what the reward
scores against — so obs and reward stay pointed at the same target. `subgoal_reach_state` does the
tolerance test and the score in one pass; **both hands must pass simultaneously**, since a bimanual
subgoal is one synchronized hand-object configuration.

**Reward** (TeleDexter Eq. 5) becomes `1_reach · w_step · r_score + α_dense · r_dense − c_time`,
where `r_dense` is the existing imitation reward and `w_step = Δk + 5` — so a longer jump pays
proportionally more and the policy cannot farm the bonus off trivially close subgoals. `r_score`
reuses the repo's own reward kernels (thumb 100, index 90, middle 80, ring/pinky 60, obj pos 80,
obj rot 10) rather than the paper's flat 90, so the sparse and dense terms rank a pose the same way.

**Termination.** The per-finger instant-fail ladder is switched off — leaving the reference between
subgoals is the point. What replaces it is an `n_fail` budget of `η_fail · Δk` out-of-tolerance
frames since the last hit. The velocity sanity checks and a 15 cm object-position bail still cut an
episode immediately — except during a cross-trajectory transition, where the bail is lifted (below).

**Curriculum.** The `Δk` ceiling follows the same cubic σ curve as the action mask (`8 → 24` over
`subgoalRampSteps`), and the tolerances ride the existing `tightenFactor` schedule rather than
getting a second one. Both now read `curriculum_frames()`, which prefers `total_train_env_frames` —
checkpointed, so **the ramp no longer replays from scratch on resume** the way `control_steps` did.

### Cross-trajectory switching — `subgoalCrossTrajProb`

The second half of the paper's goal reset (TeleDexter Sec. C.4, Eq. 9), and the `traj:031 →
traj:014 → traj:005` row of its Fig. 2. On a subgoal hit:

```
(τ_next, k_next) = (τ, k + Δk)                with probability 1 − p     in-trajectory hop
                 = (τ' ~ Unif(demos), k)      with probability p         cross-trajectory switch
```

A switch **keeps the frame index and swaps the demo**. The env goes on manipulating its own object
actor; only the target trajectory is repointed, so the next subgoal is a *different demonstration's*
hand-object configuration at the same point in the clip. The policy therefore has to handle a target
that jumps off the trajectory it was following — which is the situation at deployment, where a live
operator's motion never stays inside any recorded clip.

**Mechanism.** A `demo_row` index sits between every per-step demo read and the packed
`[num_envs, nT, …]` buffers: normally row `e` for env `e`, and repointed at another row by a switch.
`resample_subgoal` does the draw; `subgoal_active_row` is its pre-jump snapshot, the mirror of
`subgoal_active_idx`, so the reward still scores the subgoal that was live when the switch happened.
`build_cross_traj_pool` precomputes the candidate rows once at startup — envs sharing a
(demo, augmentation variant) slot hold identical rows, so the pool stores one representative per
slot, not one per env.

**What a switch changes for that transition**, both mirroring the paper:

- `w_step` becomes the flat `subgoalCrossTrajStepWeight` (100) instead of `Δk + 5`. The jump is not
  measured in frames, and crossing between demos is the hard part worth paying for.
- `n_fail` becomes the flat `subgoalCrossTrajFailBudget` (175 frames) instead of `η_fail · Δk`, and
  **the 15 cm object bail is suspended** until the post-switch subgoal is reached. A switched-in
  target can legitimately sit further than 15 cm from where the object physically is; read as a
  dropped object it would end the episode on the step after every switch. The velocity checks and
  the 300-frame budget still bound the transition.

**Scope.** `aug` (default) only offers demos under the *same* augmentation variant — the paper's
case, and sound here because the loader re-centres every object onto the table centre, so two demos
of one task share a scene layout and differ only in how the human moved. `any` is the only scope
that does anything with a **single** demo plus `numTrajAug`, but it switches across whole-scene
rotations, and `_aug_demo_table_center` rotates the manipulated object's trajectory while leaving
`prop_trajectory` (the static bottle/cup) put — so the target can be defined against a scene layout
this env does not have. Much harder; reach for it deliberately. Either way a candidate must carry
the same object ids on both hands. With nothing switchable (one demo under `aug` scope) the
probability is zeroed at startup with a warning, rather than drawing coin flips that can never fire.

**Resets restore the env's own demo.** A switch lasts only the episode it happened in. Letting the
row persist would let `demo_row` random-walk, drifting the population off the balanced round-robin
`_create_envs` sets up and desynchronising `env_demo_idx`, which the per-demo reward/success logging
keys on. The spread TeleDexter gets from resetting to another trajectory is already baked into the
env ↔ demo assignment here.

`subgoal_cross_count` is logged per episode next to `subgoal_hit_count`.

**Four things to know before enabling:**

- `Δk` is **not** the paper's 40 → 80. Those are 600-frame free-play clips with the wrist IK-driven
  outside the policy; here demos run min 96 / median 346 / max 584 frames and the policy drives the
  wrist itself, so the defaults are fraction-matched and then halved.
- Leave `denseRewardScale` at 1.0 and the sparse bonus is buried — `reward_execute` peaks near 27.6
  per hand (~55 summed) against a paper balance of ~0.8 dense per step.
- **Success means something different.** Reaching the last subgoal, not running out the clip. An
  episode that hits every subgoal finishes in far fewer control steps than a dense run, so episode
  length and success rate are not comparable across modes.
- TeleDexter used SAPG with 6 independently exploring blocks and ~10¹⁰ env steps, and reports
  vanilla PPO dropping substantially on both reward and consecutive successes (its Fig. 5).
  `denseRewardScale` is the safety valve if the sparse signal stalls learning here.

Training only — forced off under `test=true` and under `--live`, where the wire overwrites the demo
slots every step and clamps the pointer itself.

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
