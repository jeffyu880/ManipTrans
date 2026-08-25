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
