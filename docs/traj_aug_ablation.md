# Trajectory augmentation + joint noise: train-time ablation

**Question.** Does training the residual on augmented, joint-noised copies of the demos
(`useTrajAug=true`, `numTrajAug=10`, `jointNoiseCm=0.3`) improve held-out success?

**Answer.** Not in the mean: **0.509 → 0.540**, a paired **t of 0.57** over 50 demos. What it does
do is **halve the spread across folds** (sd 0.176 → 0.099) and lift reach distance 3 specifically
(**+0.254**, the only column that moves consistently). Read it as a regularizer that trades ceiling
for floor, not as a win.

That conclusion has now been measured twice, through two different success criteria, and both times
the mean effect came out at +0.03 with t < 0.6. The *absolute* levels moved a lot between the two
measurements; the comparison did not.

Status: 2026-08-11, jobs 79558 (non-aug) and 79560 (aug), all ten folds COMPLETED.

---

## Read this first: the success criterion was broken until 2026-08-11

Every number in the first version of this document was scored through a disabled failure gate.
`compute_imitation_reward`'s non-training branch built the gate and then threw it away:

```python
failed_execute = (
    (diff_thumb_tip_pos_dist > 0.06)
    | (diff_index_tip_pos_dist > 0.06)
    | (diff_obj_pos_dist > 0.03)
    | (diff_obj_rot_angle.abs() / np.pi * 180 > 45)
) & (running_progress_buf >= 8) | error_buf
failed_execute = failed_execute | error_buf
failed_execute = error_buf     ############## CHANGE MEE############   <-- discards all of it
```

Since `succeeded = (progress_buf + 1 >= max_length) & ~failed_execute`, "success" meant only *the
episode ran to the end of the demo without a sim error*. Cap position and cap orientation were never
tested. The line came in with commit `643b246` (2026-07-30, "added metrics for analysis of fingers
… on offline demos"), so it predates every matrix here.

The rollout data shows it plainly. Mean errors over rollouts counted **successful**, fold 1:

| Demo | succ | e_r (deg) | e_t (cm) |
|---|---|---|---|
| non-aug `m_141130` | 0.555 | 7.1 | 0.64 |
| non-aug `m_141843` | 0.859 | 6.9 | 0.53 |
| aug `m_141130` | 0.984 | 21.8 | 2.49 |
| aug `m_130957` | 0.922 | 16.6 | 2.04 |
| aug `m_140843` | 0.203 | **38.6** | **6.33** |

`m_140843` averages 6.33 cm of object position error among *successful* rollouts, against a gate
that is supposed to fail anything past 3 cm. The gate was not running.

**Effect of deleting that line**, same checkpoints, same pool, same eval arm otherwise:

| | gate disabled | gate enabled |
|---|---|---|
| non-aug | 0.568 | **0.509** |
| aug | 0.834 | **0.540** |
| delta | +0.266 | **+0.031** |

The aug family lost **0.294**; the non-aug family lost 0.059. The asymmetry is the point — the aug
policies drift far further from the demo (e_r 15-39° vs 4-8°) and were being credited for surviving
to the end of the episode anyway. The "augmentation wins big" result was almost entirely artifact.

**Corollary: the 30° vs 45° comparison never measured anything.** Both thresholds sat on the
discarded line, so changing 30 → 45 could not affect eval success. The 0.464 → 0.568 shift
attributed to it in the first version of this doc came from something else in the eval arm.

## Does the cap actually seat?

With the gate live, yes — and the 45° allowance turns out to be irrelevant. Across all 100
demo-evaluations (both families, 5 folds each), the worst mean error over successful rollouts is:

| | max mean `e_r` | max mean `e_t` |
|---|---|---|
| non-aug | **8.62°** | **0.78 cm** |
| aug | **8.48°** | **0.79 cm** |

Typical values are 3.7-8.6° and 0.33-0.79 cm. The successful set sits five times inside the 45°
rotation bound and four times inside the 3 cm position bound, and because the gate is evaluated per
timestep this is not just an endpoint average — no timestep of any successful rollout exceeded it.
The cap seats to roughly 8 mm and 9 degrees.

**So the rotation threshold is never the binding constraint.** Episodes fail on the 3 cm object
position gate or the 6 cm fingertip gates long before rotation approaches 45°. Changing it from 30
to 45 would have altered nothing even with the gate working, which is a second, independent reason
the "30° vs 45°" numbers in the first version of this doc measured something other than the
threshold. If the bar ever needs tightening, `diff_obj_pos_dist > 0.03` is the lever, not the angle.

## The two model families

Both are 5-fold leave-one-fold-out CV over the v3 reach pool (`data_stats/reach_demo_v3.csv`,
`make_kfold.py` seed 0, 2 demos per distance per fold), trained by
`slurm/alps/ALPS_train_kfold_v3_array.run`. They differ in the augmentation flags only.

| Family | Run dirs | Train flags |
|---|---|---|
| non-aug | `runs/reach_5foldcv_v3_holdout<k>_seed0__08-04-17-55-*` | `useTrajAug=false`, `jointNoiseCm=0.0` |
| aug | `runs/reach_5foldcv_v3_holdout<k>_seed0_AUG__08-04-*` | `useTrajAug=true`, `numTrajAug=10`, `jointNoiseCm=0.3` |

Everything else matches: `deterministicBaseAction=false`, `num_envs=8192`, `learning_rate=2e-4`,
`actionsMovingAverage=0.6`, `early_stop_epochs=1000`, `failureThresholdNoiseCompensation=1.5`. All
ten tasks early-stopped between 2:45:33 and 4:00:44; none hit `max_iterations`.

Note `jointNoiseCm` is only read **inside** the augmentation loop (`dexhandmanip_bih.py:616`,
guarded by `useTrajAug`), so with `useTrajAug=false` the noise is dead code. This ablation tests the
pair, not either flag alone.

## The eval arm

**Both families are scored on the CLEAN demo.** `eval_kfold.sh` pins every augmentation and noise
source off — `useTrajAug=false`, all four per-type aug flags false, `useObjRotationAug=false`,
`numTrajAug=1`, `jointNoiseCm=0.0`. Augmentation is a training-time treatment and is deliberately
not reproduced at eval; that is what makes the two families comparable.

`failureThresholdNoiseCompensation=1.0` at eval pins the **success criterion**, not a noise source.
Training loosens finger failure thresholds to 1.5 to tolerate injected noise; inheriting that would
score the two families against different bars.

Two more paths need no override: domain randomization is already off at test time
(`ResDexHand.yaml:163`, `randomize: ${if:${...test},False,True}`), and `obsHandNoise` /
`obsHandVelNoise` are commented out of `config.yaml`, so the env uses its own 0.0 defaults.

`DET_BASE=false` — the frozen imitators sample `Normal(mu, sigma)` at eval, matching how both
families trained. This differs from every v2 matrix in
[deterministic_base_ablation.md](deterministic_base_ablation.md), which were held at
`DET_BASE=true` to keep *that* ablation clean.

## Results

Non-augmented, overall CV **0.509**:

```
          dist 1   dist 2   dist 3   dist 4   dist 5   | Fold Mean
------------------------------------------------------------------
fold 0    0.125    0.398    0.098    0.672    0.508   |   0.360
fold 1    0.293    0.234    0.383    0.848    0.738   |   0.499
fold 2    0.859    0.371    0.613    0.926    0.996   |   0.753
fold 3    0.098    0.246    0.062    0.711    0.516   |   0.327
fold 4    0.695    0.539    0.523    0.418    0.856   |   0.606
------------------------------------------------------------------
D Mean    0.414    0.358    0.336    0.715    0.723   |   0.509
```

Augmented, overall CV **0.540**:

```
          dist 1   dist 2   dist 3   dist 4   dist 5   | Fold Mean
------------------------------------------------------------------
fold 0    0.219    0.617    0.012    0.656    0.789   |   0.459
fold 1    0.238    0.160    0.609    0.738    0.523   |   0.454
fold 2    0.273    0.547    0.883    0.746    0.738   |   0.638
fold 3    0.562    0.383    0.816    0.891    0.625   |   0.655
fold 4    0.441    0.312    0.629    0.477    0.605   |   0.493
------------------------------------------------------------------
D Mean    0.347    0.404    0.590    0.702    0.656   |   0.540
```

Delta (aug − non-aug):

```
          dist 1   dist 2   dist 3   dist 4   dist 5   |  Fold
------------------------------------------------------------------
fold 0    +0.094   +0.219   -0.086   -0.016   +0.281  |  +0.098
fold 1    -0.055   -0.074   +0.227   -0.109   -0.215  |  -0.045
fold 2    -0.586   +0.176   +0.269   -0.180   -0.258  |  -0.116
fold 3    +0.465   +0.137   +0.754   +0.180   +0.109  |  +0.329
fold 4    -0.254   -0.227   +0.105   +0.059   -0.250  |  -0.113
------------------------------------------------------------------
D Mean    -0.067   +0.046   +0.254   -0.013   -0.067  |  +0.031
```

## Why +0.031 is not a result

Per-demo deltas have **sd 0.380** around a mean of +0.031 — the noise is twelve times the effect.
Paired over the 50 demos that is **t = 0.57**, and the median demo moves **+0.004**. At demo level
it is 25 up, 22 down, 3 unchanged.

Two folds do nearly all the work in opposite directions and largely cancel: fold 3 **+0.329**, fold
2 **−0.116**. Reporting either alone would be a different paper.

The 30° measurement — through the broken gate, but over the same demos and split — gave
**+0.028, t = 0.52**. Two independent scorings agreeing on "no effect" is the strongest statement
in this document.

## What does move: fold variance

| | min | max | range | sd |
|---|---|---|---|---|
| non-aug fold means | 0.327 | 0.753 | 0.426 | 0.176 |
| aug fold means | 0.454 | 0.655 | 0.201 | **0.099** |

Augmentation lifts the floor (fold 3: 0.327 → 0.655) and pulls down the ceiling (fold 2:
0.753 → 0.638). That is what a regularizer does, and it held under both success criteria
(sd 0.176 → 0.101 at 30°, 0.176 → 0.099 here).

It also bounds how much any single-fold result is worth: a 0.43 fold range means a conclusion drawn
from one fold carries a ±0.2 error bar from fold choice alone.

## What does move: distance 3

Distance 3 gains **+0.254** and is the only column up in 4 of 5 folds (fold 3: 0.062 → 0.816,
fold 2: 0.613 → 0.883, fold 1: 0.383 → 0.609). It is the worst column in the non-aug family
(0.336) and was the worst in the v2 deterministic family too (0.295); augmented it reaches 0.590.

Alone that column is **t = 2.43 (n=10)**. But it is one of five columns tested and no other column
moves, so treat it as a hypothesis to confirm on more distance-3 demos, not an established effect.

Also: distance 3 is the group edited most heavily when the pool was rebuilt on 2026-08-04 (three
demos added, three removed). Whether the gain tracks augmentation or the new membership is not
separable from this data.

## Caveats

- **128 rollouts scored, not 2000.** `eval_kfold.sh` requests `num_rollouts_to_run=2000` but
  `num_rollouts_to_save=128` caps what reaches disk, and `stats.txt` reads the saved set. Each
  matrix cell is 2 demos x 128 episodes.
- **Saved-set bias is unverified.** `kfold_confusion_matrix.py` assumes the saved rollouts are the
  first 128 to complete "in arrival order, so it's unbiased". Arrival order correlates with episode
  length and failures terminate early, so the saved set may skew toward failures. It applies equally
  to both arms, so the *comparison* holds even if absolute levels are shifted.
- **Not comparable to the v2 matrices.** Different pool, `DET_BASE=false` instead of true, and a
  different `actionsMovingAverage`. Compare v3 to v3 only.
- **Cross-threshold comparisons are weak.** The 30° matrices were produced on 2026-08-05; since
  then `eval_kfold.sh` was reverted to HEAD and re-patched at least once. The gate fix is the one
  difference I can prove, so treat 30°-vs-45° levels as indicative and the within-run aug-vs-non-aug
  deltas as sound.
- **One seed.** Both families are seed 0. Nothing here separates the augmentation effect from
  seed-to-seed variation, and at fold sd 0.099 that is the obvious next control.

## Reproducing

```bash
# training (the script currently holds the aug family; the non-aug variant is in its git history)
sbatch --account=abs18 slurm/alps/ALPS_train_kfold_v3_array.run

# scoring, one job per fold -> dumps/v3_fold<k>_45_deg_err/ and dumps/v3aug_fold<k>_45_deg_err/
sbatch --array=0-4 --account=abs18 slurm/alps/ALPS_eval_kfold_sampled.run   # non-aug family
sbatch --array=0-4 --account=abs18 slurm/alps/ALPS_eval_kfold_detimit.run   # aug family

# matrices (add --per-demo to annotate each cell with its individual demo rates)
python data_stats/kfold_confusion_matrix.py --csv data_stats/reach_demo_v3.csv \
    --run-prefix reach_5foldcv_v3_holdout --dumps 'dumps/v3_fold*_45_deg_err' \
    --out data_stats/kfold_confusion_matrix_v3_45deg.png
python data_stats/kfold_confusion_matrix.py --csv data_stats/reach_demo_v3.csv \
    --run-prefix reach_5foldcv_v3_holdout --dumps 'dumps/v3aug_fold*_45_deg_err' \
    --out data_stats/kfold_confusion_matrix_v3aug_45deg.png
```

Both eval scripts are repurposed — they previously scored the v2 sampled and deterministic families
that [deterministic_base_ablation.md](deterministic_base_ablation.md) reports. Each carries a header
block with the settings to restore its old behaviour; `dumps/{sampled,detimit}_fold*` are untouched.

The aug family needs `--checkpoint` rather than `RUN_PREFIX`: `eval_kfold.sh` interpolates
`runs/<prefix><k>_seed0__<glob>`, and the `_AUG` infix sits between `_seed0` and `__`, so no prefix
can reach those dirs.

**Pin the pool.** `eval_kfold.sh` defaults `POOL_CSV` to `data_stats/reach_demo_v3.csv`.
`reach_demo_v2.csv` still exists and still lists `m_140827` / `m_141150` / `m_141254` / `m_133637` /
`m_133727`; it deals a different split, which puts demos the residual trained on into its own
held-out set. Two runs were lost to exactly that before the eval scripts learned to compare demo
ids rather than counts.

## Pending

1. **Seed control.** Retrain one family at seed 1. With fold sd at 0.099 and the family gap at
   0.031, seed variance plausibly explains the whole effect.
2. **Separate the two flags.** `jointNoiseCm` is dead unless `useTrajAug=true`, so "augmentation"
   and "0.3 cm joint noise" are confounded. Run `useTrajAug=true jointNoiseCm=0.0` to split them.
3. **Distance 3.** Confirm on demos outside the ten in this pool, to separate the augmentation gain
   from the 2026-08-04 membership change.
4. ~~**Re-examine what the gate should be.**~~ Measured 2026-08-11: the rotation threshold is never
   the binding constraint, so tightening it would change nothing. See
   [Does the cap actually seat?](#does-the-cap-actually-seat) below.
5. **`demoTargetFps` sweep.** `slurm/alps/ALPS_train_kfold_v3_50hz_array.run` is staged and
   unsubmitted. Note 50 Hz does not subsample a 60 Hz source (`skip = round(60/50) = 1`); it only
   changes `dt` to 1/50, i.e. 20% slower execution. Use 30 Hz for genuine subsampling.
6. **Check the saved-set bias** against `player.py` — open since the deterministic ablation.
