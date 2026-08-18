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

Full reference — every flag, the chaining order, and when each knob is applied — lives in
**[augmentations.md](augmentations.md)**. Summary for this experiment:

| Flag | What rotates | Pivot |
|---|---|---|
| `RH_LH_Table_Center_Aug` | everything | table center (XY plane) |
| `RH_LObj_Center_Aug` | RH demo only | LH object position at each frame |
| `RH_RObj_Center_Aug` | RH demo only | RH object position at each frame |
| `LH_LObj_Center_Aug` | LH demo only (left hand + left object, rigidly) | LH object **position** at each frame (its orientation spins in place with the hand); RH demo untouched |

All require `useTrajAug=true` as master switch. `RH_LH_Table_Center_Aug` additionally defaults to
**true** when none of the others is set.

Enabled augs **chain** — object-rotation, then an RH-center aug, then LH-about-LH-obj, then
table-center — except that `RH_LObj_Center_Aug` and `RH_RObj_Center_Aug` are **mutually exclusive**
(both rotate the RH demo, so applying both would double-rotate it). When both are set, one is chosen
at random per augmented variant.

`numTrajAug` (default `20`; this experiment uses `400`) pre-generates that many augmented versions
of each demo at `_create_envs` time. Env `i` is bound to variant `i % numTrajAug` for the whole run.
During test mode variant 0 (the original) is skipped so all envs use augmented variants, with a
fixed RNG seed for reproducibility.

`jointNoiseCm` adds **uniform** noise in `[−σ, +σ]` (σ in cm) to MANO wrist positions and joint
keypoints, simulating hand pose estimator error. Applied after spatial augmentation via
`_apply_joint_noise` — once per augmented variant at load time, **not** per step. Pair it with
`failureThresholdNoiseCompensation` so the failure thresholds tolerate the injected offset.

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

Submission scripts live under `slurm/<cluster>/` — `slurm/alps/train_maniptrans_inspire.run` and
`slurm/scitas/` (which also holds `train_cup_brush_individual.run`, a job array).

1. Edit `slurm/alps/train_maniptrans_inspire.run` — set `DATA_INDICES`, `EXPERIMENT_NAME`, and aug flags
2. `sbatch slurm/alps/train_maniptrans_inspire.run` → note the job ID
3. Evaluate completed runs with `eval_capping.sh` (repo root)

Logs go to `logs/inspire/slurm_<jobid>/slurm-<jobid>.{out,err}`.
Checkpoints go to `runs/<experiment>__<date>/nn/<experiment>.pth`.
