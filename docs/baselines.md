# Baselines: dex-retargeting (DexPilot)

The pure-retargeting comparison point for ManipTrans's learned two-stage policy. No RL, no
learned dynamics: every control step, `baselines/dexret_controller.py` reads the human hand target
the env already holds, solves robot joint angles with
[dex-retargeting](https://github.com/dexsuite/dex-retargeting), and substitutes that for the
policy's action inside `pre_physics_step`. The env, its physics, its termination logic and its
logging are untouched, so the baseline is scored by exactly the same machinery as a policy run.

It reads `demo_data_{rh,lh}` rather than a precomputed trajectory, which is what lets one code
path serve both offline demos and live teleoperation — `_inject_live` overwrites those buffers
every step, and the controller neither knows nor cares.

---

## Best known configuration

| Setting | Value | Why |
|---|---|---|
| `dexRetBaseline` | `true` | Turns the whole thing on |
| `dexRetType` | `dexpilot` | Bunny-VisionPro's choice. The only family that makes real contact — see [Which optimiser](#which-optimiser) |
| `dexRetWristMode` | `pd_ff` | Controller-owned PD gains, asset-drag compensation, causal velocity feedforward |
| `usePIDControl` | `false` | Required by `pd_ff`, which emits wrist force/torque rather than a position error |
| `dexRetWristFit` | `true` | 6-DOF wrist placement solved from the fingertips; supersedes the scalar pullback |
| `zeroResidual` | `true` | Belt-and-braces — the policy contributes nothing regardless |
| `actionsMovingAverage` | `1.0` | 0.6 would low-pass the commanded joint target, and the lag would be misread as retargeting error |
| `randomStateInit` | `false` | Test-mode reset |
| `num_envs` | 1–8 | Solves per env in Python, so cost scales linearly |
| `dexRetFitMode` | `constant` | Lifts the cap 114 mm against `per_frame`'s 81 — see [Constant vs per-frame](#constant-vs-per-frame) |

Plus these module constants in `baselines/utils/constants.py`, already set to these values:

| Constant | Value | Why |
|---|---|---|
| `DEXRET_SOLVE_URDF` | `"maniptrans"` | **Solve against the hand the simulator actually runs.** The single largest correction found — see [The URDF mismatch](#the-urdf-mismatch) |
| `DEXRET_FIT_MODE` | `"constant"` | Better lift and better RH contact than `per_frame`, despite worse fingertip RMS |
| `DEXRET_SCALING_FACTOR` | `1.00` | 2x the RH contact of dex-retargeting's shipped 1.15 (59.8% vs 29.8%) and 3.4x the cap lift, post-URDF-fix |
| `DEXRET_FIT_WEIGHTS` | `None` | Uniform across the five tips. Weighting was measured and is **not** a useful lever |
| `DEXRET_FIT_MAX_TRANSLATION` / `_ANGLE_DEG` | `0.15` / `60` | Outlier guard only. At the previous 80 mm / 30 deg it bound 24-81% of calibration samples and was silently doing tuning work |

### What this is measured against

Three demos (`m_085551`, `m_131154`, `m_131256`), dexpilot, constant fit. **Measured before the
URDF mismatch was found**, so these are the pullback-vs-fit comparison only — the absolute values no
longer reflect the current configuration:

| | old 0.38 pullback | fit @ 1.15 | **fit @ 1.00** |
|---|---|---|---|
| RH fingertip error | 35.85 mm | 27.73 mm | **29.36 mm** |
| RH frames in contact | 16.71% | 12.35% | **35.59%** |
| LH fingertip error | 27.33 mm | 22.77 mm | **23.56 mm** |
| LH frames in contact | 7.51% | 0.17% | **3.88%** |

`fit @ 1.00` is the first configuration that beats the pullback on **both** axes at once — every
other setting wins one and loses the other. Fingertip error and contact necessarily disagree here:
closing the fingers moves their tips off the human's by construction, so a lower fingertip error
past a point just means a hand hovering in free space. Contact is the metric that decides whether
the hand grips.

**LH contact remains the weak spot** (3.88% vs the pullback's 7.51%). RH pinches a cap while LH
power-grasps a bottle body, and `scaling_factor` is currently one scalar for both. Per-hand
scaling is the obvious untried next step — the two hands already load separate config files.

---

## The URDF mismatch

**The most consequential thing found in this baseline, and the one worth explaining first.**

The wrist fit solves against a URDF; the simulator then executes the resulting joint angles on a
*different* URDF. Until this was found, those were not the same hand:

* the fit solved against **dex-urdf's** inspire — the model dex-retargeting and Bunny-VisionPro
  were built for, and the natural choice for a faithful reproduction of the published method;
* the sim ran **ManipTrans's own** inspire.

Commanding identical joint values to both and measuring where the fingertips actually end up:

| | dex-urdf (before) | ManipTrans URDF (after) |
|---|---|---|
| RH commanded-vs-achieved \|err\| | 23.8 mm | **6.0 mm** |
| LH commanded-vs-achieved \|err\| | 19.8 mm | **5.2 mm** |
| RH achieved-minus-commanded z | −7.7 mm | **+0.4 mm** |
| LH achieved-minus-commanded z | **+14.4 mm** | **+1.5 mm** |

The residual ~5 mm is ordinary PD tracking error. What vanished is the *systematic, per-hand-signed*
bias — ~23 mm of swing between the two hands, which was essentially the whole left-vs-right
discrepancy this baseline carried for weeks.

Crucially, **no wrist fitting can reach this error.** Kabsch matches the fingertips of the model it
is handed; if the sim then runs different kinematics, the mismatch appears downstream of everything
the fit controls. That is why a long sequence of plausible fixes (below) all failed.

(The sharper form of that argument — that Kabsch determines translation *exactly* from centroids, so
it cannot leave a systematic offset — holds only for the raw **unclamped, per-frame** solve. Under
the recommended `constant` mode, and with clamping, the applied correction is an average that can
carry a bias. The empirical conclusion stands, but the proof is narrower than stated.)

Downstream effect, three demos, dexpilot:

| | dex-urdf | ManipTrans URDF |
|---|---|---|
| LH vertical offset | 20.5 mm | **10.1 mm** |
| LH fingertip error | 31.8 mm | **23.8 mm** |
| LH frames in contact | 16.67% | **31.16%** |

### How it is implemented

`DEXRET_SOLVE_URDF` selects the model. Two things follow from it:

1. **Config selection.** `default_config_path` appends an `_mt` suffix, so
   `baselines/configs/inspire_{side}_{type}_mt.yml` is used. `urdf_path` is byte-identical between the two sets — the model is switched by the search
   directory (`RetargetingConfig.set_default_urdf_dir(solve_urdf_dir())`). The `_mt` files differ only
   in the `R_`/`L_` link and joint prefixes, and in the root link name (`base` in dex-urdf vs
   `{R,L}_hand_base_link` in ManipTrans's).
2. **Frame composition.** ManipTrans's base frame sits 180 deg from the MANO frame the keypoints
   arrive in — which is exactly why this option was rejected in the original design, since a solver
   fed raw MANO-frame targets sees a hand pointing the wrong way. The controller now composes the
   hand-local transform with `base_align`, so keypoints reach the optimiser already in the target
   URDF's own base frame, and `solver_to_sim` collapses to identity. Setting
   `DEXRET_SOLVE_URDF="dex"` restores the solved model, the config file and the frame composition —
   but NOT the other constants that were retuned alongside it (`DEXRET_SCALING_FACTOR` 1.00 vs
   dex-retargeting's 1.15, the clamps, the fit mode, the calibration window), and it does not
   invalidate a stored live calibration: `load_calibration` checks robot/optimiser/scaling but not
   the solve URDF.

`MANIPTRANS_DEXRET_SOLVE_URDF=dex` reproduces the published-method configuration for comparison.

---

## Why the left hand looked broken

Worth recording because the symptom was misleading and the search was long.

**Symptom (measured BEFORE the fix; the pre-fix runs have since been overwritten, so these figures
are historical and not reproducible from what is on disk).** Over 16 demos the left hand's
fingertips sat **~+23 mm above** the human reference while the right sat near zero. The ManipTrans
imitator, on the *same* robot hand, *same* URDF, *same* physics and *same* demos, held LH to
+7.8 mm and RH to +9.6 mm — so the left-hand task is not intrinsically harder. Only dex-retargeting
struggled with it.

**Everything ruled out along the way.** Several of these tested knobs that have since been deleted
(they existed only as work-arounds for the bug), so the settings named are not all reproducible
today — they are recorded to show what the search covered, not as a recipe.

| Hypothesis | How it was killed |
|---|---|
| The Kabsch fit causes it | The hover exists with the old scalar pullback too, and the fit *reduces* it |
| Wrist control error | Wrist tracking is small and unbiased relative to a 20 mm hover: mean error 2.6–5.3 mm, per-axis signed bias under ~2 mm. (An earlier "<1 mm" claim referred to the signed bias only.) |
| The calibration window | Swapping the RH's narrow window onto the LH and vice versa moved each hand's offset by <3 mm. *(Historical: the windows are symmetric at 0.25/0.25 now.)* |
| Our custom `OPERATOR2AVP["left"]` | Substituting dex-retargeting's shipped matrix is catastrophic — 100% of fit samples clamped, LH tip error 128 mm. Ours is correct |
| The wrist reconstruction | Discounting the fit toward the human's measured wrist made it *worse* (+30 mm at zero correction vs +20.5 at full). *(Historical: tested with a `DEXRET_FIT_BLEND` knob that has since been removed.)* |
| Fingertip mirror / handedness | Each hand's robot layout matches its own human layout with the correct opposite y-ordering |
| The thumb | It is the worst-matched point in the **hand-local solve** (54 mm pre-fit residual — not a world-frame tip error, where it is one of the *better* fingers). Down-weighting it to 0.1 moved the hover 1.5 mm out of 20 |
| Hand size or joint limits | The imitator reaches +7.8 mm on the same hand and URDF |
| Ill-conditioned tip cloud | Kabsch determines translation from centroids, so the *unclamped per-frame* fit cannot leave a systematic offset — see the caveat under [The URDF mismatch](#the-urdf-mismatch) |

The last row is what finally pointed the right way: if the fit could not be responsible, the error
had to lie **downstream of the command**. Wrist tracking was already verified, so the
untested link was the fingers — and that measurement is the table in
[The URDF mismatch](#the-urdf-mismatch).

**A genuine defect found along the way, still present:** dex-urdf's `inspire_hand_left.urdf` has an
asymmetric thumb. FK at identical joint angles, left against the mirror of right, 500 random poses:
`thumb_tip` differs by 6.08 mm mean / 10.45 mm max while all four fingers mirror to 0.00 mm. It
traces to `thumb_proximal_pitch_joint`, whose origin is no clean sign flip of its counterpart.
**ManipTrans's own URDFs carry the identical defect** (5.27 mm mean / 9.68 mm max on the same test),
so switching the solve to ManipTrans's model did *not* remove it — both families presumably inherit
it from the same source. Too small to explain a 20 mm hover, but real, and it is the reason the
thumb is the worst-matched point in the hand-local solve.

---

## The remaining deficit: the lift

With the URDF fixed, the largest remaining gap is that **the hand does not lift as high as the
human's**. Decomposed over the lift phase (human fingertip z minimum to its following maximum),
demo `m_101716`:

| stage | rise | loss |
|---|---|---|
| human wrist | 103.3 mm | — |
| human fingertips | 135.5 mm | *human fingers contribute +32.2* |
| **commanded** wrist | 92.0 mm | **−11.3** ← the constant fit |
| **achieved** wrist | 90.4 mm | −1.6 ← PD tracking |
| robot fingertips | 96.7 mm | *robot fingers contribute only +6.3* → **−25.9** |
| **total vs human** | | **−38.8 mm** |

Reproduce with `videos_scaling/dexret_fitmode/constant/m_101716/{pinch.csv,wrist.csv}`; the lift
window is the human fingertip z minimum to its following maximum (frames 158–236).

**Finger retargeting dominates, roughly 2:1** — 25.9 mm of the 38.8 mm, against 11.3 mm from the
wrist fit and 1.6 mm from PD:

* **Fingers (67%).** DexPilot matches *scaled inter-finger vectors*; it never constrains where a
  fingertip sits **relative to the wrist**, which is exactly the quantity that is short. Inherent to
  the retargeting objective, not a bug.
* **The constant wrist fit (29%).** The correction is frozen in the hand-local frame, so as the
  wrist rotates through the lift its world-z projection changes (measured swinging +25.3 to
  +35.0 mm) and a fixed value cannot track it. `per_frame` does **not** fix this — it makes the lift
  worse; see [Constant vs per-frame](#constant-vs-per-frame).
* **PD tracking (4%)** — negligible.

### Every knob that raises the hand loses the cap

Three demos, constant fit, ManipTrans URDF:

| config | hand shortfall | cap lift | RH contact |
|---|---|---|---|
| **dexpilot @1.00** | −36.9 mm | **94.7 mm** | **59.8%** |
| dexpilot @1.15 | **−28.7** | 27.6 | 29.8 |
| vector @1.00 | −34.5 | 60.8 | 53.8 |
| vector @1.15 | **−26.5** | 60.1 | 29.7 |

Raising `scaling_factor` extends the fingers and recovers ~8 mm of hand lift on both optimisers —
and collapses the cap lift. The robot's tip span is ~0.92x the human's, so the only way to make its
fingers reach like the human's is to open them, and opening them releases a 2.2 g cap. **Reach and
grip are in direct competition**, and `dexpilot @1.00` — the worst hand lift, the best cap lift — is
still the right choice.

The one lever that improves both is making the object bigger, so the fingers meet it at a more open
pose (`objScaleRH`, 16 demos, dexpilot):

| cap scale | cap lift | RH contact |
|---|---|---|
| 1.0 | 54.6 mm | 37.0% |
| 1.15 | 63.3 | 40.0 |
| 1.3 | 56.4 | 42.1 |
| 1.45 | **68.6** | **48.7** |

That changes the task, though, so it belongs in the results as an axis rather than as a fix.

**The cap itself is not the problem.** It weighs 2.2 g against 0.2–0.3 N of grip — a 10x margin —
and follows the robot hand to 98% during the lift. It is not slipping; the hand simply does not go
as high.

---

## Running offline

```bash
python main/rl/train.py \
    task=ResDexHand dexhand=inspire side=BiH \
    test=true headless=true num_envs=1 \
    dataIndices=[m_085551] \
    rh_base_model_checkpoint=assets/imitator_rh_inspire.pth \
    lh_base_model_checkpoint=assets/imitator_lh_inspire.pth \
    dexRetBaseline=true dexRetType=dexpilot dexRetWristMode=pd_ff \
    dexRetWristFit=true zeroResidual=true actionsMovingAverage=1.0 \
    usePIDControl=false randomStateInit=false \
    num_rollouts_to_run=3 experiment=dexret_m_085551
```

Output lands in `dumps/test____demo_<idx>__<date>/`. **No `checkpoint=` is needed** — the imitator
checkpoints are loaded only to satisfy the env's construction and are never stepped. `train.py`
detects `test + dexRetBaseline` and steps the env directly rather than routing through rl_games,
which would build the residual network and load both frozen imitators purely to discard them.

### With video

```bash
    capture_video=true n_parallel_recorders=1 n_successful_videos_to_record=5
```

The recorder skips `num_envs * 2` warm-up episodes, so `num_rollouts_to_run=3` at `num_envs=1`
yields exactly one recorded episode. The *controller* is deterministic, but the physics is not
reproducible across resets — rollouts of the same demo diverge by up to ~117 mm in cap position and
can reach different outcomes — so extra rollouts are genuine samples, not duplicates, and one
recorded episode is a sample of one. Two views per episode: front, and `_top` (which is a *behind*
view despite the name). Both come from `maniptrans_envs/lib/envs/core/record_cameras.py`, shared with
`data_stats/playback_trajectory.py --record` so policy and playback footage are comparable.

### Metrics

```bash
MANIPTRANS_PINCH_CSV=pinch.csv MANIPTRANS_DEXRET_LOG=wrist_error.csv python main/rl/train.py ...
```

Only MANIPTRANS_DEXRET_LOG resolves a bare filename into the run directory. `pinch.csv` carries per-step reference (`*_avp_*`)
and achieved (`*_sim_*`) fingertip positions plus per-finger contact forces;
`MANIPTRANS_DEXRET_LOG` writes a wrist trace and auto-plots it via
`data_stats/plot_dexret_wrist.py`. Feed the pinch CSVs to `data_stats/plot_pinch_gap.py --compare`
to put the baseline alongside the imitator and residual series.

### Calibration, offline

The constant wrist correction is derived by a one-off pre-pass on first use over the frames where that hand's thumb and index tips are nearest its object (`tips_distance`, nearest 25%), strided to roughly 200 samples.8 s, ≤200 strided samples. No file is involved: offline the demo **is** the calibration source.

---

## Running online (live)

Two steps. Calibrate once; teleoperate freely thereafter.

```bash
# 1. Capture the wrist-fit constant. Stops itself and writes
#    data/dexret_calibration/inspire_dexpilot.json
python main/rl/train.py \
    task=ResDexHand dexhand=inspire side=BiH test=true headless=true num_envs=1 \
    dataIndices=[m_085551] \
    rh_base_model_checkpoint=assets/imitator_rh_inspire.pth \
    lh_base_model_checkpoint=assets/imitator_lh_inspire.pth \
    dexRetBaseline=true dexRetType=dexpilot dexRetWristMode=pd_ff \
    dexRetWristFit=true zeroResidual=true actionsMovingAverage=1.0 \
    usePIDControl=false randomStateInit=false \
    live=true liveAddr=<publisher-ip> livePort=5555 \
    dexRetCalibrate=true

# 2. Every session after: loads the file, correct from the first frame.
#    Same command, minus dexRetCalibrate.
python main/rl/train.py ... live=true liveAddr=<publisher-ip> dexRetCalibrate=false
```

**During step 1, move BOTH hands through the motion you intend to teleoperate** — reach out, close
into the grasp you will use, open again. The constant is a *median* over what it sees, so holding
one pose calibrates only that pose. It captures 120 frames per hand (2 s at 60 Hz;
`DEXRET_FIT_CALIB_FRAMES` overrides), prints progress every quarter, and ends the run itself.

A live run with **no** calibration file captures one before teleoperating and says so, so step 1
is a convenience rather than a hard prerequisite — but doing it deliberately means the hand is not
being driven by a correction that is still moving while you work.

`load_calibration` **refuses** a file taken against a different robot, optimiser or scaling factor
rather than adapting it. Silently reusing one would surface as a tracking problem nobody would
trace back to a stale calibration. Re-run with `dexRetCalibrate=true` after changing any of those.

A reference demo is still loaded via `dataIndices` — live mode needs it for assets, BPS and reset
init — but its target slots are overwritten in place every step. Transport, `liveBuffered`, and
debugging: [`../maniptrans_envs/lib/envs/live/README.md`](../maniptrans_envs/lib/envs/live/README.md).

> **Live is qualitative only.** `post_physics_step` skips `compute_reward` entirely and forces
> `reset_buf = 0` when `live=true`, so there are no episodes and no scores. All quantitative
> comparison must come from offline runs.

---

## Runtime cost

Measured at `num_envs=1`, `dexpilot`, constant fit:

| | per hand |
|---|---|
| dex-retargeting NLS solve | 0.97 ms |
| wrist fit (constant) | 0.04 ms |
| **both hands** | **2.02 ms of a 16.7 ms control step (12%)** |

Comfortably real-time at 60 Hz with ~14.7 ms of headroom. The fit is nearly free once the constant
is frozen — two matrix products, plus FK and two weighted norms purely for the RMS diagnostic. No
SVD, no optimiser. `per_frame` mode costs 0.12 ms instead, so the constant is 3x cheaper.

The 0.8 s calibration pre-pass is **offline only** — live loads the constant from file and never
pays it. The NLS solve dominates at 24x the fit's cost, so that is what to attack if more headroom
is ever needed, not the fit.

---

## How the wrist is placed

`vector` and `dexpilot` solve the fingers **alone**, in the wrist frame, and the wrist is then
supplied from the human. That leaves the grasp displaced for two reasons a wrist taken from the
human cannot fix:

- dex-retargeting scales the human's inter-finger vectors by `scaling_factor`, so the robot's
  fingertips deliberately do not sit where the human's did relative to the wrist;
- the solver assumes the robot's URDF base is aligned with the MANO frame the keypoints arrive in,
  which for dex-urdf's inspire holds only to ~25°.

`pull_wrist_back` (`DEXRET_WRIST_PULLBACK = 0.38`) was the previous answer and is strictly weaker:
it slides the wrist along **one** axis, wrist→middle-MCP, by a hand-tuned fraction, while the error
is a full 6-DOF rigid transform.

`baselines/utils/wrist_fit.py` replaces it. Because the finger angles are already fixed by the time
we ask where the hand goes, the placement is a closed-form orthogonal Procrustes (Kabsch) problem —
no optimiser, no gains, nothing to tune:

```
p_i = robot fingertips from FK on the solved pose  (hand base frame)
q_i = human MANO tips 4, 8, 12, 16, 20             (same frame)

H = Σ (p−p̄)(q−q̄)ᵀ ;   U, S, V = svd(H)
R = V · diag(1, 1, det(VUᵀ)) · Uᵀ ;   t = q̄ − R p̄
```

The `det` term excludes reflections. Five fingertips of a flat hand are very nearly planar, and
without that guard the SVD can return a mirrored "rotation" that turns the hand inside out.

**Frame algebra.** The solver's base orientation in world *is* `frame` from
`hand_local_transform`, and the sim's commanded wrist rotation is `frame @ C` for a constant
`C = OPERATOR2AVP.T @ loader_to_avp.T` (a 180° rotation, both hands). So the correction reaches the
sim as `frame @ R_fit @ C`, and **`R_fit = I` reproduces the previous command exactly** — verified
to 8.9e-16 over 500 random poses per hand. That invariant is what the whole wiring rests on.

Clamped at 15 cm and 60° against the human's own wrist pose, scaled back rather than rejected so a
degenerate frame degrades tracking smoothly instead of dropping out.

### Constant vs per-frame

Selected with `dexRetFitMode=constant|per_frame`. Three demos, dexpilot, ManipTrans URDF:

| | `constant` | `per_frame` |
|---|---|---|
| **cap lift** | **114.4 mm** | 81.3 mm |
| RH frames in contact | **59.8%** | 52.7% |
| RH fingertip error | **31.5 mm** | 32.6 mm |
| LH fingertip error | 36.0 mm | **24.9 mm** |
| LH frames in contact | 13.34% | **25.60%** |
| fit RMS (kinematic) | 15.0–21.0 mm | **9.0 mm** |

**`per_frame` has clearly better fingertip agreement and clearly worse task behaviour.** This is the
single most repeated pattern in this baseline and the one most likely to mislead: fingertip RMS is
*not* a proxy for whether the hand lifts the cap.

The mechanism is counter-intuitive. A per-frame fit re-solves every step to keep the robot's
fingertips on the human's; because the robot's fingers do not extend as far as the human's during a
lift, the only way to keep the tips matched is to **hold the wrist lower**. Measured on `m_101716`,
per-frame nearly halves the commanded wrist rise (92.0 → 48.6 mm). A constant cannot chase the
fingertips, and here that inability is protective. On one demo (`m_101747`) per-frame drops the cap
outright.

`per_frame` is still the better choice if the left hand is what matters — it roughly doubles LH
contact and cuts LH fingertip error by a third. It also needs **no calibration at all** in live
mode, which makes it the quickest way to try the fit on a live stream.

Averaging (for `constant`) uses the **per-axis median** for translation and the **chordal L2 mean**
for rotation. Median because degenerate frames — thumb on a joint limit, tips momentarily
near-collinear — produce large bad offsets that a mean would absorb; chordal mean because it has no
wrap-around failure, unlike averaging Euler or axis-angle vectors.

A caveat on smoothness numbers: the per-frame fit does jump — a genuine 73 mm single-frame jump was
found mid-grasp on `m_085551`, caused by a near-coplanar fingertip cloud (`s3/s1 ≈ 0.03`) making the
rotation ill-conditioned. But "worst jump" figures from multi-rollout runs are dominated by the
reset boundary between episodes, where the previous episode's correction is still cached, and should
not be read as jitter.

---

## Which optimiser

`dexRetType` selects among four configs in `baselines/configs/` (each with an `_mt` variant for
ManipTrans's URDF, all four verified to build at the default `DEXRET_SOLVE_URDF="maniptrans"`).

**Current numbers**, three demos, ManipTrans URDF, constant fit, `dexRetWristFit=true`:

| Type | RH contact | RH tip error | cap lift | Verdict |
|---|---|---|---|---|
| `dexpilot` | **59.8%** | 31.5 mm | **94.7 mm** | **Use this.** Bunny-VisionPro's method |
| `vector` | 53.8% | 36.5 mm | 60.8 mm | Close on contact, carries the cap noticeably worse |
| `position` (+Kabsch) | 16.3% | 33.7 mm | ~0 | Barely moves the cap |
| `position` (free joint) | 4.4% | 42.4 mm | ~0 | Barely moves the cap |

Both `position` variants leave the cap essentially motionless — 0.1–1.6 mm over an entire demo,
against 135–140 mm of hand travel. `position` is not a degraded baseline here, it is a
non-functional one. Note it *regressed* under the corrected URDF (16.3% RH contact, against 44%
when solved against dex-urdf) — its earlier apparent strength was partly the two models' errors
cancelling.

An older four-demo comparison, taken **before the wrist fit and before the URDF fix**, ranked them
`dexpilot` 18.33% / `vector` 17.94% / `position` 0.00% / `position_free` 0.00% RH contact. The
ordering has held; the magnitudes have not.

The `position` family scores *better* on fingertip error precisely because a hand hovering in free
space has nothing pushing back. Zero contact on the capping hand means it cannot do the task.
Fingertip error is a proxy; contact is the thing. By default the `position` family ignores `dexRetWristFit` — their `add_dummy_free_joint` already
solves placement, and applying both would displace the hand twice. Set
`MANIPTRANS_DEXRET_FIT_OVERRIDE_FREE=1` to keep the Kabsch fit and discard the free joint's answer
instead; that is the `position (+Kabsch)` row above.

---

## Environment variables

| Variable | Effect |
|---|---|
| `MANIPTRANS_PINCH_CSV` | Per-step fingertip/contact CSV. **Pass an explicit path.** train.py only resolves a bare name in LIVE mode, and then next to the checkpoint — offline a bare name lands in the process CWD |
| `MANIPTRANS_DEXRET_LOG` | Wrist trace CSV, auto-plotted by `data_stats/plot_dexret_wrist.py` |
| `MANIPTRANS_DEXRET_PLOT=0` | Suppress that auto-plot |
| `MANIPTRANS_DEXRET_DEBUG=N` | Print the wrist force breakdown for the first N steps |
| `MANIPTRANS_DEXRET_SCALING` | Override `scaling_factor` without editing a config |
| `MANIPTRANS_DEXRET_FIT_WEIGHTS` | Five per-finger weights, e.g. `3,3,1,1,1`. Needs ≥3 non-zero — Kabsch cannot pin a rotation from two points, and the assert says so |

---

## Shell prerequisites

Isaac Gym's `gym_38.so` links against `libpython3.8.so.1.0`, which `conda activate` does not put on
the library path:

```bash
ENV=/home/guest/miniconda3/envs/maniptrans
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:$LD_LIBRARY_PATH"
```

`pinocchio` must import **before** `isaacgym` or every call into dex-retargeting dies with
`No Python class registered for C++ class std::vector<std::string>`. `main/rl/train.py` does this
at the top for exactly that reason — use that entry point, or replicate the import order.

---

## Known limitations

- **Success rate is meaningless.** `maniptrans_envs/lib/envs/tasks/dexhandmanip_bih.py:3577` holds
  `failed_execute = error_buf ############## CHANGE MEE############`, which discards the entire
  eval failure criterion — every rollout reports success unless the sim explodes. No task-level
  comparison against the imitator or residual is valid until that is reverted to
  `failed_execute = failed_execute | error_buf`.
- **The live calibration capture loop has never run against a real stream.** The save/load round-trip,
  mismatch rejection and the averaging were checked by hand during development, but there is no
  test suite in this repo, so nothing guards them against regression.
- **Sample sizes are small.** The wrist-fit numbers are three demos; the optimiser comparison is
  four, with some very large standard deviations (LH contact ±38.71).
- **`scaling_factor` is one scalar for both hands** despite RH pinching a cap and LH power-grasping
  a bottle body. Per-hand scaling is untried.
- **The cap is never seated on the bottle.** Closest cap-to-bottle centre-to-centre approach is
  51–62 mm. The 21 mm lift shortfall is a plausible cause but is not proven to be the only one —
  the pinch CSV logs no object-object contact, so whether the cap and bottle actually intersect
  cannot be answered from the current logs. Adding a net contact force per object would settle it.
- **Three constants were tuned before the URDF mismatch was found** and have not been re-validated
  against the corrected solve: `DEXRET_SCALING_FACTOR = 1.00`, `DEXRET_FIT_MODE = "constant"`, and
  the 0.25 calibration window. They may no longer be optimal.
- **`position` regressed under the corrected URDF** — 16.26% RH contact with the Kabsch override,
  4.43% with its own free joint, against dexpilot's 59.8%. Its earlier apparent strength was
  partly the two models' errors cancelling. Both position variants leave the cap essentially
  motionless (0.1–1.6 mm over an entire demo).
