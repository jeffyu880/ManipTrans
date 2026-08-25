# Residual window on m_131154 (offline)

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
