# ManipTrans Experiments

Authoritative reference for the current experiment setup: bimanual alcohol-burner capping under
leave-one-out (LOO) evaluation, trajectory augmentation, the imitator-only baseline, experiment
naming, and the SLURM submission workflow. (Relocated from `CLAUDE.md` to keep that file lean.)

---

## Current Experiment Setup (Alcohol Burner Capping, LOO)

### Task
Bimanual capping of an alcohol burner (`b5fa3@10_bih`). RH holds the pen/cap, LH holds the burner body.

### Leave-One-Out (LOO) Evaluation
Train on 8 demos, evaluate generalization on the held-out `b5fa3@10`:
```
Training: 3b1e6@12, d6fe3@0, 8e5df@13, a78a0@1, 0f900@10, f7d37@18, 85abe@4, e49f5@0
Eval:     b5fa3@10
```
`maxDemoLength=252` caps all training demos to the shortest useful length for balanced sampling.

### Trajectory Augmentation

Controlled by four flags (all require `useTrajAug=true` as master switch):

| Flag | What rotates | What is fixed |
|---|---|---|
| `useTableCenterAug` | everything | table center (XY plane) |
| `useLHObjCenterAug` | RH demo only | LH object position at each frame |
| `useRHObjCenterAug` | RH demo only | RH object position at each frame |
| `useLHAboutLHObjAug` | LH demo only (left hand + left object, rigidly) | LH object **position** at each frame (its orientation spins in place with the hand); RH demo untouched |

When multiple flags are enabled they **chain**: LH-obj-center is applied first, then RH-obj-center, then LH-about-LH-obj, then table-center — each operating on the already-transformed result from the previous step.

`numTrajAug=200` pre-generates 200 augmented versions of each demo at `create_envs` time. Envs cycle through these variants. During test mode the original (aug_k=0) is skipped so all envs use augmented variants, with a fixed RNG seed for reproducibility.

`jointNoiseCm` adds Gaussian noise (std in cm) to MANO wrist positions and joint keypoints, simulating hand pose estimator error. Applied after spatial augmentation via `_apply_joint_noise`.

### Baseline: Imitator-Only
Pass `zeroResidual=true` to zero out the residual delta, running only the frozen imitator. Used as a comparison baseline without retraining.

### Experiment Naming Convention
```
capping_alcohol_burner_9_<aug_type>_<noise?>_<ma>ma_<loo|single>_b5fa3@10
```
Examples:
- `capping_alcohol_burner_9_table_center_noise_0.6ma_loo_b5fa3@10` — table center aug + noise, 0.6 MA, LOO
- `capping_alcohol_burner_9_LH_center_0.4ma_loo_b5fa3@10` — LH-center aug, no noise, 0.4 MA, LOO
- `capping_alcohol_burner_9_LH_center_noise_single_0.4ma_b5fa3@10` — single demo (b5fa3@10 itself, not LOO)

### SLURM Submission Workflow
1. Edit `train_maniptrans_inspire.run` — set `DATA_INDICES`, `EXPERIMENT_NAME`, and aug flags
2. `sbatch train_maniptrans_inspire.run` → note the job ID
3. Log the run in `training_log.txt` with all params and job ID
4. Evaluate completed runs with `eval_capping.sh`, aggregate with `aggregate_results.py`

Logs go to `logs/inspire/slurm_<jobid>/slurm-<jobid>.{out,err}`.
Checkpoints go to `runs/<experiment>__<date>/nn/<experiment>.pth`.
