# Deterministic base action: train-time ablation

**Question.** Does freezing the frozen imitators to emit `mu` instead of sampling
`Normal(mu, sigma)` **during residual training** help or hurt?

**Answer so far.** It hurts, substantially. On the one fold measured both ways, held-out success
fell from **0.588 to 0.270** — a 54% relative drop. Training-time metrics did not reveal this.

Status: 2026-08-04. The five-fold confirmation (job 77234) was still running when this was
written; see [Pending](#pending).

---

## The two model families

Both are 5-fold leave-one-fold-out CV over the v2 reach pool
(`data_stats/reach_demo_v2.csv`, 50 demos, `make_kfold.py` seed 0, 2 demos per distance per fold).
They differ in exactly one training flag.

| Family | Run dirs | `deterministicBaseAction` at TRAIN |
|---|---|---|
| sampled | `runs/reach_5foldcv_v2_holdout<k>_seed0__07-29-15-26-*` | `false` |
| deterministic | `runs/reach_5foldcv_v2_deterministic_imitator_holdout<k>_seed0__08-03-*` | `true` |

**The eval arm is held fixed at `deterministicBaseAction=true` for both.** This is the point of
the design: if eval determinism tracked training determinism, train-time and eval-time effects
would be confounded and neither matrix would mean anything. Do not "fix" this to match the
training flag.

## Fold 4: the controlled comparison

Same 10 held-out demos, same eval config, 1280 rollouts per arm. Only the training differs.

| Demo | trained sampled | trained deterministic | delta |
|---|---|---|---|
| m_085551 | 0.5312 | 0.0000 | **-0.5312** |
| m_131154 | 0.5938 | 0.1094 | -0.4844 |
| m_131256 | 0.0000 | 0.0000 | +0.0000 |
| m_133917 | 1.0000 | 0.5469 | -0.4531 |
| m_140843 | 0.1719 | 0.1641 | -0.0078 |
| m_141027 | 0.9453 | 0.4375 | -0.5078 |
| m_141114 | 0.7578 | 0.1797 | **-0.5781** |
| m_141514 | 0.8828 | 0.3125 | -0.5703 |
| m_141642 | 1.0000 | 0.9453 | -0.0547 |
| m_141658 | 0.0000 | 0.0000 | +0.0000 |
| **POOLED** | **0.5883** | **0.2695** | **-0.3187** |

Seven of ten regressed, three held flat — and two of those three are 0.000 in both arms, i.e.
demos neither model ever solves. No demo improved.

The damage concentrates on demos the sampled model had nearly solved: `m_141114` 0.758 -> 0.180,
`m_141514` 0.883 -> 0.313, `m_141027` 0.945 -> 0.438. `m_085551` collapses from 0.531 to zero.

## Deterministic-trained family, all five folds

`python data_stats/kfold_confusion_matrix.py --csv data_stats/reach_demo_v2.csv
--run-prefix reach_5foldcv_v2_deterministic_imitator_holdout --dumps 'dumps/detimit_fold*'`

```
          dist 1   dist 2   dist 3   dist 4   dist 5   | Fold Mean
------------------------------------------------------------------
fold 0    0.195    0.477    0.488    0.121    0.211   |   0.298
fold 1    0.543    0.441    0.324    0.379    0.586   |   0.455
fold 2    0.656    0.219    0.340    0.512    0.535   |   0.452
fold 3    0.344    0.234    0.109    0.539    0.824   |   0.410
fold 4    0.082    0.309    0.211    0.473    0.273   |   0.270
------------------------------------------------------------------
D Mean     0.364    0.336    0.295    0.405    0.486   |   0.377  (overall CV)
```

Two things to read off it beyond the headline 0.377:

- **Reach distance does not drive difficulty monotonically.** Distance 5 is the *best* column
  (0.486) and distance 3 the worst (0.295). Whatever makes a demo hard is not how far the reach is.
- **Fold variance exceeds distance variance** (0.270-0.455 across folds vs 0.295-0.486 across
  distances). Which specific demos land in a fold matters more than the distance stratification,
  which argues against reading per-distance conclusions from a single fold.

## Why training-time metrics missed it

Mean training-time success across the five folds:

| | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | mean |
|---|---|---|---|---|---|---|
| sampled | 0.541 | 0.642 | 0.300 | 0.437 | 0.620 | **0.508** |
| deterministic | 0.541 | 0.584 | 0.349 | 0.555 | 0.536 | **0.513** |

Indistinguishable — the deterministic family looks marginally *better*. The regression is
invisible on the training distribution and appears only on held-out demos, so it is a
generalization failure, not an optimization one.

**Mechanism (hypothesis, not yet tested).** Sampling `Normal(mu, sigma)` from the frozen imitators
injects exploration noise the residual must stay robust to; it acts as a regularizer. Freeze the
base to `mu` and the residual co-adapts to a noiseless partner, then fails on held-out demos whose
required corrections fall outside what it saw. Testing this would mean sweeping the imitator sigma
rather than toggling the flag.

## Caveats

- **One fold.** The -0.3187 is a single controlled comparison. Job 77234 extends it to all five.
- **128 rollouts scored, not 2000.** `eval_kfold.sh` requests `num_rollouts_to_run=2000` but
  `num_rollouts_to_save=128` caps what reaches disk, and both `stats.txt` and `eval_score.py` read
  the saved set. Each matrix cell is 2 demos x 128 episodes = 256 rollouts.
- **Saved-set bias is unverified.** `kfold_confusion_matrix.py` asserts the saved rollouts are the
  first 128 to complete "in arrival order, so it's unbiased". Arrival order correlates with episode
  length, and failed episodes terminate early, so the saved set could skew toward failures. This
  has not been checked against `player.py`. It applies equally to both arms, so the *comparison*
  holds even if the absolute levels are shifted.
- **`reach_folds.csv` is stale.** It disagrees with `make_kfold.make_folds` over
  `reach_demo_v2.csv`, which is what training and eval actually use. Trust the generator.

## Reproducing

```bash
# deterministic-trained family (done -> dumps/detimit_fold{0..4}/)
sbatch --account=abs18 slurm/alps/ALPS_eval_kfold_detimit.run

# sampled-trained family (job 77234 -> dumps/sampled_fold{0..4}/)
sbatch --account=abs18 slurm/alps/ALPS_eval_kfold_sampled.run

# matrices
python data_stats/kfold_confusion_matrix.py --csv data_stats/reach_demo_v2.csv \
    --run-prefix reach_5foldcv_v2_deterministic_imitator_holdout \
    --dumps 'dumps/detimit_fold*' --out data_stats/kfold_confusion_matrix.png
python data_stats/kfold_confusion_matrix.py --csv data_stats/reach_demo_v2.csv \
    --run-prefix reach_5foldcv_v2_holdout \
    --dumps 'dumps/sampled_fold*' --out data_stats/kfold_confusion_matrix_sampled.png
```

`--dumps` takes a glob because the per-fold `DUMP_TAG` layout puts dumps one level deeper than the
script's default; the wildcard expands inside `glob.glob`, so no script change is needed.

Note `eval_kfold.sh` lives only on branch `final1` upstream; it was restored into `brush_cup` at
commit-time with its reward regex fixed (`_rew_+\K` instead of `(?<=_rew_)`, which silently skipped
every early-stop checkpoint because `base.py` writes those as `_rew__<value>__`).

## Pending

1. **Job 77234** — sampled-trained 5-fold eval. `sampled_fold4` re-scores the same checkpoint as
   `dumps/detbase_fold4/` and should land on ~0.588; if it does not, the pipeline is not
   deterministic enough to support the comparison and everything above needs re-examining.
2. Build the sampled matrix and diff the two cell-by-cell.
3. If the five-fold result confirms fold 4, revert residual training to
   `deterministicBaseAction=false` and treat the 08-03 deterministic family as a negative result.
