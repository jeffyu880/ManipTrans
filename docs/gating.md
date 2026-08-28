# Control Gating

Who drives each hand, and when. Three controllers are available at runtime — the frozen **imitator**,
the trained **residual**, and a per-frame **dex-retargeting** solve — and everything here is about
handing a hand between them.

All of it is per hand: RH and LH gate, arbitrate and hand over independently.

---

## The phase structure

With `residualGateDistance >= 0`, each hand runs the same three phases across one manipulation:

| phase | window | drives the hand | shown in the viewer |
|---|---|---|---|
| reach | shut (`w = 0`) | frozen imitator | **I** (amber) |
| manipulation | open (`w → 1`) | imitator + residual | **R** (green) |
| retreat / seated | shut again | frozen imitator | **I** (amber) |

The residual is spent on the grasp; the reach and the retreat are left to the imitator that was
trained for them. `residual_gate_weights()` produces the per-hand weight, and everything downstream
— the residual multiply, `reachController`, `switchModel`, `imitatorOnlyHands` — keys off it.

**A weight of zero means "the frozen imitator alone" everywhere except `reachController=dexret`,
which reads it as "the retargeter drives".** That inversion is why the two are mutually exclusive
with the flags below, and the asserts say so.

---

## The window

| knob | default | meaning |
|---|---|---|
| `residualGateDistance` | `-1.0` (off) | metres; where the window **opens** |
| `residualGateReleaseDistance` | `-1.0` (→ 1.5×) | where it **closes**; must exceed the open distance |
| `residualGateFadeSteps` | `12` | control steps the residual eases in/out over; `1` = instant |
| `residualGateMetric` | `surface` | what the distance is measured against — see below |

Two thresholds rather than one because a fingertip resting on a single boundary would flip the
residual every step, and each flip is a step discontinuity in the commanded action. Inside the band
neither edge fires and the window holds its state — hysteresis, not two independent thresholds.

### What the distance actually measures

The **operator's** thumb and index fingertips (not the robot's), taking the *minimum* of the two.
Middle, ring and pinky are ignored.

- **`surface`** (default) — distance to the nearest of **1000 points sampled off the object mesh**.
  Offline that is `sample_points_from_meshes(mesh, 1000)` then a Chamfer distance
  (`main/dataset/base.py:172, 200-210`); live it is recomputed each frame in
  `LiveTargetSource._tips_distance`. Note this is the nearest sampled **vertex**, not the true
  surface, so it reads slightly long wherever the sampling is coarse.
- **`origin`** — distance to the object's **mesh origin**. Cheap, and independent of mesh detail and
  of how the mesh happens to be sampled.

**Retune `residualGateDistance` when changing metric.** The origin sits inside the body, so the
distance is larger by roughly the object's radius — a 3 cm threshold meaning "about to touch" on
`surface` will never fire on `origin`. The offset is not even symmetric: `bottle_cap`'s origin is
2.6 cm off its own geometry (`OBJ_LOCAL_OFFSET_M` in the capture repo).

### Closing the window early

`liveResidualCutoff` (default `True`) zeroes the residual once the cap is seated on the bottle,
handing back to the imitator, which the residual otherwise fights. It is gated on
`live_object_set.seating_cutoff`, so it is a **no-op for `cup_brush`** — two props meeting means
nothing there, and that set closes its window on `residualGateReleaseDistance` instead.

---

## Pinning a hand — `imitatorOnlyHands`

`none` | `rh` | `lh` | `both`, default `none`.

Holds the named hand on the frozen imitator for the whole episode: no residual, no arbitration.
Implemented by zeroing that hand's window weight, so it composes with everything above without a
second code path, and the viewer shows **I** for that hand throughout.

Incompatible with `reachController=dexret` (asserted), for the inversion reason above.

---

## Who owns the *shut* window — `reachController`

`imitator` (default, a no-op) | `dexret`.

`dexret` hands the reach to a per-frame retargeting solve, crossfading to the imitator on the very
same weight that fades the residual in, so a hand rides the retargeter over exactly the span its
residual is off for. Mutually exclusive with `dexRetBaseline`, which replaces the action outright and
never loads the policy.

---

## Who owns the *open* window — `switchModel`

Runs **both** the residual policy and a dex-retargeting solve every frame the window is open, scores
them, and executes whichever follows the operator better. Per hand, per frame.

| knob | default | meaning |
|---|---|---|
| `switchModel` | `False` | master toggle |
| `switchModelObjWeight` | `0.7` | object share of the score; finger share is `1 - this` |
| `switchModelObjScale` | `0.02` | metres of object error normalising to 1.0 |
| `switchModelFingerScale` | `0.05` | metres of fingertip error normalising to 1.0 |
| `switchModelMargin` | `0.4` | score units the challenger must win by (hysteresis) |
| `switchModelDwellSteps` | `3` | minimum steps a choice is held |
| `switchModelLog` | `True` | per-step CSV of both scores and the selection |

### The score

```
f_res = ‖FK(residual targets) − operator_tips‖ / fingerScale     predictive, per candidate
f_dex = ‖FK(dexret targets)   − operator_tips‖ / fingerScale     predictive, per candidate
o     = ‖sim_object − operator_object‖         / objScale         observed, shared

use dexret  iff  (1−w)·(f_dex − f_res) + w·o  <  −margin
```

The finger term is a genuine one-step prediction: both controllers emit joint targets, the hand is
PD-driven toward them, so forward kinematics of the target says where the fingers are heading. It is
evaluated in the hand-base frame, so the wrist cancels out of the comparison.

The object term **cannot** be per-candidate — the object's response needs simulation, and is only
observable for whichever controller actually ran. It therefore shifts the threshold rather than
separating the candidates, and is charged to dex-retargeting: a drifting object means contact is the
difficulty, which is the residual's regime.

**That asymmetry is deliberate and load-bearing.** Dex-retargeting is *defined* as the argmin of
fingertip error, so on the finger term alone it wins nearly every frame by construction — and the
residual deliberately commands fingertips that differ from the operator's, since that offset is what
turns a PD position target into grip force (see `objScaleRH` / `objScaleLH`). Set `switchModelObjWeight=0` to see that
degeneracy directly; it is a diagnostic, not a working setting.

---

## Calibration, measured

From live runs on the capping demo (`*_switch_model.csv`, ~40 s each):

| quantity | RH | LH |
|---|---|---|
| fingertip error, both candidates (p50) | 0.045–0.048 m | 0.038–0.047 m |
| **finger advantage** `f_res − f_dex` (p50) | **+2 to +11 mm** | **+1 to −7 mm** |
| object error (p50) | **57 mm** | **6 mm** |

Two conclusions:

1. **`switchModelFingerScale = 0.05` is about right** — observed p50 normalises to ~0.8–1.4.
2. **`switchModelObjScale = 0.02` is 3–10× too tight.** A 57 mm median object error normalises to
   2.9 and p90 to 12.4, so the object term saturates and swamps the finger term regardless of the
   weights. **Try `0.1`.**

At the defaults (`switchModelObjWeight=0.7`, `switchModelObjScale=0.02`) dex-retargeting must overcome a ~136 mm-equivalent object
handicap plus a 66.7 mm margin, against a 2 mm finger advantage — measured selection was **0.0%**.

### The margin earns its place

The two controllers sit within a few mm of each other, so a bare comparison flips on noise. Replayed
against a recorded run at `switchModelObjWeight=0`:

| margin | dexret chosen (RH) | switches | mean hold |
|---|---|---|---|
| 0 mm | 57.7% | 7.92 /s | 0.13 s |
| 5 mm | 61.7% | 4.81 /s | 0.21 s |
| **20 mm** | **61.6%** | **1.03 /s** | **0.97 s** |

**8× fewer switches, same split** — it removes the chatter and not the preference. A longer dwell
does not substitute: 1 → 12 only takes flips from 7.9 to 3.4.

Note the margin's mm-equivalent depends on `switchModelObjWeight`, because it is in score units: **20 mm at
`w=0`, 66.7 mm at `w=0.7`**.

### Cost

While a window is open the arbiter runs both controllers every step: ~2.2 ms for the dex-retargeting
solve (warm; 13% of a 16.7 ms budget) plus ~3.8 ms of forward kinematics. Unlike
`reachController=dexret`, which latches the solve off once the windows open, `switchModel` needs it
for the whole manipulation phase.

---

## Reading the viewer

`liveRateOverlay` draws the achieved control rate, and under it one letter per hand, **RH then LH**:

| letter | meaning | colour |
|---|---|---|
| **I** | frozen imitator alone | amber |
| **R** | imitator + residual | green |
| **D** | dex-retargeting | cyan |

A hand shows **I** whenever its window is shut — reach, retreat, seated cutoff, or pinned by
`imitatorOnlyHands` — because downstream those are all the same condition.

---

## A known-good live configuration

Residual gating on, arbiter off, gating on distance to the mesh origin:

```
residualGateDistance=0.03
residualGateReleaseDistance=0.045
residualGateFadeSteps=4
residualGateMetric=origin
liveResidualCutoff=true
imitatorOnlyHands=none
switchModel=false
switchModelObjWeight=0
debugVis=false
```

`switchModelObjWeight` is inert while `switchModel=false`; it is carried here so that turning the
arbiter on gives the pure-kinematic diagnostic rather than the uncalibrated 0.7 default.

`debugVis=false` matters for live: the green demo skeletons and per-body contact colouring cost
several ms per step in Python at one env.
