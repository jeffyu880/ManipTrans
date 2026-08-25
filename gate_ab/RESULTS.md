# Residual window on m_131154 (offline)

> **Superseded below.** Everything in this first section ran at `actionsMovingAverage=0.4`
> (taken from eval_capping.sh) and `maxDemoLength=252`, which cuts 40 of the capture's 292
> frames. CLAUDE.md and record_best_checkpoint.sh both use 0.6 for BiH, and 0.4 low-passes the
> commanded joint target — docs/baselines.md warns that lag "would be misread as retargeting
> error", which is exactly the arm being judged. See **Full length at 0.6** for the numbers to
> use.

Policy: `reach_5foldcv_v2_holdout4_seed0__07-29-15-26-26`, ckpt `ep_1300` (sr 0.620 on its own demos).
Demo `m_131154` (bottle capping) is **held out** — not among the 40 demos in that run's `demos.txt`.

Config: `num_envs=16`, `test=true`, `randomStateInit=false`, `actionsMovingAverage=0.4`,
`maxDemoLength=252`, all augmentation off (`useTrajAug=false`, `jointNoiseCm=0.0`), headless.
Augmentation is off deliberately: it perturbs `mano_joints` without recomputing `tips_distance`,
which is the signal the window keys off.

## Results

| Arm | Setting | Outcome | Reward | Steps |
|---|---|---|---|---|
| residual always | `residualGateDistance=-1` | 15/16 succ, 1 fail | 7544 | 252 |
| window (tight) | `0.03 / 0.045`, fade 12 | **16/16 succ** | 7421 | 252 |
| window (loose) | `0.05 / 0.075`, fade 12 | **16/16 succ** | 7512 | 252 |
| imitator only | `zeroResidual=true` | **0/16 succ** | 5277 | 183 |

## What the gate actually did

Computed offline from `tips_distance` (`gate_ab/gate_schedule.py`), full 292-frame capture:

| Hand | pinch dist range | window | residual on |
|---|---|---|---|
| RH | 0.04 – 5.37 cm | opens at frame 87, never closes | 70% of episode |
| LH | 0.21 – 1.81 cm | open from frame 0 (never above engage) | 100% — gate is a no-op |

## Conclusions

1. **The residual is essential here.** Imitator alone: 0/16, failing at step ~183 — well past the
   reach, i.e. during the manipulation. That is the division of labour the design assumes.
2. **Withholding the RH residual through the reach costs nothing.** 87 frames (30% of the episode)
   of RH residual removed; success unchanged.
3. **The LH gate was inert** — the left hand starts already on the bottle.
4. **The closing edge was never exercised.** On the full 292-frame capture neither hand retreats
   past the release threshold; the demo ends with both hands still on their objects. The third
   phase of imitator -> residual -> imitator is therefore UNTESTED by this demo.

## Caveats

- With `randomStateInit=false` and no augmentation every env is a near-identical replay, so the
  spread is sim nondeterminism, not sampling. 15/16 vs 16/16 is **not** a meaningful difference;
  treat each arm as n≈1.
- The policy was trained without a gate, so this is an off-policy intervention.

## Incidental bugs found

- `main/cfg/task/ResDexHand.yaml` defined `propScale` twice (commit `a76b68d`). omegaconf raises
  `ConstructorError`, so EVERY `task=ResDexHand` run died at config load. Fixed.
- `eval_capping.sh` passes `useTableCenterAug` / `useLHObjCenterAug` / `useRHObjCenterAug`, none of
  which exist in `config.yaml` any more. NOT fixed.
- The `save_rollouts` path records `success: 0 / fail: 0 / total: 0` even when the player itself
  reports 4 succ / 0 fail over full-length episodes, so `eval_score.py` sees an empty file. NOT
  fixed — results above are scored from the player's own count instead.
- `player.py` exits via `os._exit(0)`, which skips flushing block-buffered stdout. Redirect to a
  file without `python -u` and the entire episode summary is lost. Use `PYTHONUNBUFFERED=1`.

---

# Second option: reachController=dexret

`reachController=dexret` hands the reach to a per-frame dex-retargeting solve and crossfades the
BASE action to the frozen imitator on the same weight that fades the residual in, so the retargeter
owns exactly the span the residual is off for. Deliberately a separate knob from `dexRetBaseline`,
which makes `train.py:319` step the env directly and never build the policy — crossfading on top of
that would hand over to a hand nothing is driving. The two assert as mutually exclusive.

Same demo, checkpoint and knobs as above; `residualGateDistance=0.03`, 20 rollouts, video capture on.

| Arm | Individual failures | Mean reward | Notes |
|---|---|---|---|
| residual always | 0 | 7366 | |
| window 0.03, imitator reach | 0 | 7416 | |
| **window 0.03, dexret reach** | **2** | **5970** | −20% reward |

Cost: `1.41 ms` of a 16.7 ms control step for both hands (8% of budget), plus a one-off 0.56 s
calibration pre-pass offline. The wrist fit cut fingertip RMS 45.2 → 18.2 mm over 2016 solves.

## Reading

Driving the reach with dex-retargeting is **worse** on this demo — it is the only arm that fails at
all, and it gives up ~20% of the reward. The failures are informative in their own right: the two
imitator-reach arms were uniformly successful across every rollout, so sim jitter alone does not
flip outcomes there, whereas with the retargeter in the loop it does. That marks the dexret reach as
operating close to a margin the imitator reach is comfortably inside of.

One warning appeared repeatedly under dexret:
`wrist command saturated (|action|=1.02 > 1)`, ceiling `base_wrist_dt*translationScale*500 = 8.3 N`.
Raising `translationScale` would give the retargeted wrist headroom and is the obvious first thing
to try before concluding the dexret reach is inferior.

## Tooling caveat

`train.py` keys the video output directory off the **checkpoint's** run dir, not `experiment=`, so
two arms recorded against one checkpoint write identical filenames and the second silently
overwrites the first. Each arm here was recorded in its own invocation with its output moved aside
immediately after. Anything sweeping a knob against one checkpoint will lose all but the last run.


---

# Full length at 0.6 (authoritative)

Full 292-frame capture (no `maxDemoLength`), `actionsMovingAverage=0.6`, 20 rollouts, `num_envs=4`,
`randomStateInit=false`, no augmentation. Same checkpoint and demo.

| Arm | episodes | failures | mean reward | failure step |
|---|---|---|---|---|
| residual always | 5 | 1 | 8226 | 269 |
| window 0.03, imitator reach | 5 | 2 | 8091 | 283, 276 |
| window 0.03, dexret reach | 6 | 2 | 5786 | **69, 73** |

## Reading

**The imitator-reach window is on par with residual-always.** 8091 vs 8226 mean reward, both with
one or two late failures. Withholding the residual for the first 87 frames of the RH reach costs
nothing measurable — which is the claim the window was built to test.

**The dexret reach fails during the reach, not at the handover.** Both its failures land at steps
69 and 73, and the RH window does not open until frame 87. So the retargeter loses the task before
any handover happens; the crossfade is not implicated. That is a sharper result than the 0.4 pass
suggested, where the failures looked like a handover problem.

The wrist-saturation warning remains the first thing to chase: raising `translationScale` gives the
retargeted wrist headroom, and the failure being early and total is consistent with the wrist simply
not being able to follow the reach.

## Solve skipping

`reachController=dexret` stops solving once every hand is fully across, where the crossfade weight
is 1 and the solve is multiplied by zero. Measured on the same run:

| | dexret cost per control step |
|---|---|
| before | 1.41 ms (8% of a 16.7 ms budget) |
| after | **0.14 ms (1%)** |

Two guards make it safe. The solve carries state across control steps — the pd_ff causal velocity
divides a finite difference by a fixed `control_dt`, so an N-step gap reads as one step, and
`SeqRetargeting.last_qpos` is both the nlopt seed and the target of the objective's smoothness term.
`DexRetargetController.reset_step_history()` clears both, and `reset_idx` calls it. A live wrist-fit
calibration is never skipped: it accumulates its samples inside the solve, so skipping would leave
it short of its target forever.
