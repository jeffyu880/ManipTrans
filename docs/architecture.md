# ManipTrans Architecture & Env Internals

Authoritative reference for how the two-stage policy and the Isaac Gym env work internally:
network structure, observation/target layouts, the runtime step loop, reward, hand configs, and
domain randomization. (Relocated from `CLAUDE.md` to keep that file lean.)

See also: [pipeline.md](pipeline.md) for the end-to-end flow, [datasets.md](datasets.md) for data
formats.

---

## Two-Stage Architecture

### Stage 1: Imitator

A per-hand imitation policy trained in `DexHandImitatorRH/LH` env. It learns to track a reference trajectory from demo data.

**Inputs** (per hand):
- `proprioception`: `[q, cos_q, sin_q, base_state]` — joint positions (raw, cos, sin) + wrist pose/vel (position zeroed, only orientation and velocity used)
- `privileged`: `[dq, manip_obj_pos (relative to wrist), manip_obj_quat, manip_obj_vel, manip_obj_ang_vel, tip_force (5×4), manip_obj_com, manip_obj_weight]`
- `target`: future demo state (see Target Observation below)

**Output** (action per hand): `[wrist_pos(3), wrist_rot(3 axis-angle), finger_dofs(n_dofs)]` = `6 + n_dofs` dims. With `useQuatRot=true`: `1 + 6 + n_dofs`. With `usePIDControl`: append 3 force dims.

Checkpoints saved to `assets/imitator_{rh,lh}_{dexhand}.pth`.

### Stage 2: Residual Policy

The residual policy (`ResDexHand`) is trained on top of **frozen** imitators. For bimanual tasks, both RH and LH imitators are loaded and kept in eval mode.

**Forward pass (bimanual):**
1. Full obs is `[rh_obs, lh_obs]` concatenated for each key.
2. RH imitator processes `obs[:, :rh_slice]` → `rh_base_action`
3. LH imitator processes `obs[:, half:half+lh_slice]` → `lh_base_action`
4. Residual MLP input: `[encoded_full_obs, rh_base_action, lh_base_action]`
5. Residual MLP output: `delta_action` (same dims as total action)
6. Final action sent to sim = `base_action + delta_action` (combined in agent)

The residual network is in [../lib/rl/network_builder_residual_bih.py](../lib/rl/network_builder_residual_bih.py) and model wrapper in [../lib/rl/res_models.py](../lib/rl/res_models.py).

---

## Target Observation (per hand, concatenated in this order)

These are look-ahead signals from the demo trajectory (at `progress_buf + 1`, clamped to seq end):

```
delta_wrist_pos         [3*K]   current→target wrist position delta
wrist_vel               [3*K]   target wrist linear velocity
delta_wrist_vel         [3*K]   current→target wrist velocity delta
wrist_quat              [4*K]   target wrist quaternion (xyzw)
delta_wrist_quat        [4*K]   relative rotation current→target
wrist_ang_vel           [3*K]   target wrist angular velocity
delta_wrist_ang_vel     [3*K]   current→target angular velocity delta
delta_joints_pos        [n_bodies*3*K]   per-joint position deltas (MANO keypoints)
joints_vel              [n_bodies*3*K]   target joint velocities
delta_joints_vel        [n_bodies*3*K]   current→target joint velocity delta
delta_manip_obj_pos     [3*K]   current→target object position delta
manip_obj_vel           [3*K]   target object linear velocity
delta_manip_obj_vel     [3*K]   current→target object velocity delta
manip_obj_quat          [4*K]   target object quaternion
delta_manip_obj_quat    [4*K]   relative rotation current→target
manip_obj_ang_vel       [3*K]   target object angular velocity
delta_manip_obj_ang_vel [3*K]   current→target object angular velocity delta
obj_to_joints           [n_bodies]   distance from object to each joint
gt_tips_distance        [K]     ground truth fingertip spread distance
bps                     [128]   BPS (Basis Point Set) encoding of object shape
```

`K = obsFutureLength` (default 1). For BiH, the full `target` vector is `[rh_target, lh_target]`.

---

## Observation Inputs (per network)

The BiH Stage-2 policy is **three networks**. All read from the same obs dict `{proprioception, privileged, target}`, but from different slices. The two imitators are frozen; only the residual MLP is trained. Per-hand **residual** dims for **inspire** (`n_dofs=12`, `n_bodies=18`): proprioception=49, privileged=49, target=350.

The frozen imitators do **not** read the full per-hand slice. Each key is truncated to
`base_model_obs_shape` (`main/cfg/rl_train/ResDexHandPPO.yaml:8-11`), which is the Stage-1
imitator's own, smaller observation — for inspire **49 / 12 / 176 = 237**. The residual's extra
channels (the object state appended to `privileged`, and the BPS + deltas appended to `target`)
exist only for the residual MLP.

| Network | Trainable? | Obs slice it reads | Input dims | Produces |
|---|---|---|---|---|
| **RH imitator** (frozen) | ✗ eval-only | **head of the RH half** of each key — `obs[:, :49]` / `[:, :12]` / `[:, :176]` ([res_models.py:296-298](../lib/rl/res_models.py#L296-L298)) | 49+12+176 = 237 | `rh_base_action` (6+n_dofs) |
| **LH imitator** (frozen) | ✗ eval-only | **head of the LH half** — `obs[:, half:half+49]` / `[:, half:half+12]` / `[:, half:half+176]`, `half = dim//2` ([res_models.py:300-303](../lib/rl/res_models.py#L300-L303)) | 237 | `lh_base_action` (6+n_dofs) |
| **Residual MLP** | ✓ **trained policy** | `encode(full obs)` — both halves: prop 98 + priv 98 + target 700 ([network_builder_residual_bih.py:160](../lib/rl/network_builder_residual_bih.py#L160)) | encoded + `rh_base_action` + `lh_base_action` ([:182](../lib/rl/network_builder_residual_bih.py#L182)) | `delta_action` |

Final sim action = `base_action + delta_action`, combined in `pre_physics_step`. Each frozen imitator sees only its own hand's half and is unaware of the other hand; only the residual MLP sees both hands jointly (plus both base actions), which is what lets it coordinate bimanual contact.

**Symbolic dims (per hand):**
- `proprioception` = `13 + n_dofs*3` (base_state + q/cos_q/sin_q)
- `privileged` = `n_dofs + 13 + 5*4 + 3 + 1` (dq + obj pose/vel + tip_force + obj com + obj weight)
- `target` = `128 (bps) + 5 (gt_tips) + (23 + (n_bodies-1)*9 + 23 + n_bodies)*K`

---

## Observation Sources (where each value comes from)

Three upstream sources: **① live PhysX sim** (refreshed every step in `_refresh` → `_update_states`), **② demo data** (SMPLX/MANO + raw anno, loaded at reset), **③ static precompute** (object constants). Built in `compute_observations_side` ([dexhandmanip_bih.py:1446](../maniptrans_envs/lib/envs/tasks/dexhandmanip_bih.py#L1446)).

| Obs key | Component | Source | Notes |
|---|---|---|---|
| **proprioception** | q, cos_q, sin_q | ① live DOF state | from `self._q` |
| | base_state | ① live root state | **position zeroed** in obs ([:1457](../maniptrans_envs/lib/envs/tasks/dexhandmanip_bih.py#L1457)) |
| **privileged** | dq | ① live DOF velocity | |
| | manip_obj_pos/quat/vel/ang_vel | ① live object root | pos made wrist-relative |
| | tip_force | ① live **net contact force** | `net_cf` at 5 fingertips |
| | manip_obj_com | ① live (obj quat) + ③ static com offset | |
| | manip_obj_weight | ③ static (mass × g) | |
| **target** | delta_wrist/_joints/_obj `*` | ② demo `@progress_buf+1` **minus** ① live state | the `delta_*` mix both sources |
| | wrist_vel, wrist_quat, manip_obj_* targets | ② demo (SMPLX wrist / obj_trajectory) | |
| | obj_to_joints | ① live (obj pos → live joint pos) | |
| | gt_tips_distance | ② demo (MANO fingertip → nearest obj surface point, Chamfer) | computed in [base.py:119-132](../main/dataset/base.py#L119-L132) |
| | bps | ③ static object-shape encoding | |
| *(residual only)* | rh_base_action, lh_base_action | frozen imitator outputs | [network_builder_residual_bih.py:165-176](../lib/rl/network_builder_residual_bih.py#L165-L176) |

`proprioception` + `privileged` are pure **live sim state** (where the robot is now); only `target` carries **demo intent** (where it should go), and even there `delta_*` terms are `demo − live`. Observations enter at runtime from exactly two places — the PhysX sim tensors and the demo buffers — plus a few static object constants.

---

## Step Loop (inputs → physics → outputs)

One `env.step` runs at 60 Hz (`dt = 1/60`, `substeps = 2`, `controlFrequencyInv = 1` → one `gym.simulate()` per action). The same dof/wrist targets are held while physics runs.

```
┌── A. POLICY INPUTS  (obs dict, BiH = [rh ‖ lh]) ── compute_observations [bih.py:1439]
│     proprioception (98)  ← ① live sim: q, cos_q, sin_q, base_state
│     privileged    (98)   ← ① live sim: dq, obj pose/vel, tip_force(net_cf), com, weight
│     target       (700)   ← ② demo @progress_buf+1 (deltas vs live) + ③ bps
│        ▲ reads refreshed GPU tensors (_root_state, _dof_state, net_cf, ...)
▼
┌── B. POLICY  (ResDexHand) ── network_builder_residual_bih.py / res_models.py
│     obs ─┬─► FROZEN RH imitator (rh slice) ─► rh_base_action
│          ├─► FROZEN LH imitator (lh slice) ─► lh_base_action
│          └─► encode(obs) ─► Residual MLP([enc, rh_base, lh_base]) ─► delta_action
│     OUTPUT = [ base_action ‖ residual_action ]   (concatenated)
▼
┌── C. DECODE ACTIONS  pre_physics_step [bih.py:1844]
│     split: base=actions[:, :half], residual=actions[:, half:]*2  (zeroResidual → 0)
│     FINGERS: dof = base_dof + residual_dof → clamp[-1,1] → scale to limits
│              → moving-average w/ prev (actionsMovingAverage) → _pos_control
│     WRIST:   base_wrist + residual_wrist → PID force/torque or direct target
▼
┌── D. APPLY TO SIM  [bih.py:2118-2127]
│     set_dof_position_target_tensor(_pos_control)        ← PD finger targets
│     apply_rigid_body_force_tensors(apply_forces, ...)   ← wrist
▼
╔══ E. PHYSICS TIMESTEP  vec_task.step [vec_task.py:486] ═══════════════════════════
║     for i in range(control_freq_inv):   # = 1
║         gym.simulate(sim)   ◄ PhysX: collision → solve contacts → integrate
║                               (dt=1/60, 2 substeps) WRITES net_cf
║     gym.fetch_results(sim, True)
▼
┌── F. READ BACK  post_physics_step → _refresh [bih.py:1200]
│     refresh dof/root/rigid_body/net_contact_force tensors  (now POST-step values)
│     • compute_observations()  → NEXT obs dict ─────────┐
│     • compute reward (track demo @progress_buf)         │
│     • check failure/success/timeout → reset_buf         │
└─────────────────────────────────────────────────────────┴──► back to A (next step)
```

**Notes:**
- Physics runs **once per policy step** here (`controlFrequencyInv=1`), so policy rate = sim rate = 60 Hz. If raised, the same targets are held across N `simulate` calls before the next obs.
- The base/residual split happens in `pre_physics_step` (stage C), **not** inside the network — the network emits `[base ‖ residual]` concatenated.
- `tip_force` in stage A is the *post-previous-step* `net_cf`, refreshed in stage F before the obs is built.

---

## Reward (Imitation)

Computed per hand independently, then summed for bimanual. The reward tracks the demo trajectory on:
- Wrist position and orientation
- Wrist linear and angular velocity
- Finger joint positions (mapped MANO keypoints)
- Object pose (position + rotation)
- Object velocity

Failure triggers when tracked quantities exceed per-joint thresholds scaled by `scale_factor`. Thresholds are tightened during training via a `tighten_method` schedule (`exp_decay` default): initially loose to allow exploration, then tightened over `tighten_steps`.

**Optional extra rewards (bimanual):**
- `usePenKeypointReward`: distance between pen tip and cap opening
- `useCoaxialReward`: alignment of pen and cap Z-axes when objects are close

---

## Dexterous Hand Config

Each hand is defined in `maniptrans_envs/lib/envs/dexhands/<hand>.py`. Key attributes used throughout:
- `n_dofs`: number of finger DOFs
- `n_bodies`: number of rigid bodies
- `body_names`: ordered list matching Isaac Gym DOF order
- `contact_body_names`: fingertip body names (5 fingers)
- `hand2dex_mapping`: maps MANO keypoint names to dex hand joint names
- `weight_idx`: `{level_1_joints, level_2_joints, thumb_tip, index_tip, ...}` — index sets for reward weighting
- `relative_rotation`, `relative_translation`: MANO wrist → dex wrist transform

---

## Domain Randomization

Implemented in `maniptrans_envs/lib/envs/core/vec_task.py` via `apply_randomizations()`. Driven entirely by `randomization_params` in the task YAML config. Three categories:

### 1. Non-physical (observations & actions)
Sets up a `noise_lambda` applied every step to `obs_buf` or `actions`. Supports:
- **Distributions**: `gaussian` or `uniform`
- **Operations**: `additive` (`tensor + noise`) or `scaling` (`tensor × noise`)
- **Two noise components**: correlated (same within an episode, resampled on reset) + uncorrelated (fresh each step)
- **Schedule**: ramp noise up linearly or via constant threshold over training steps

### 2. Sim parameters
Global physics properties (e.g. `gravity`). Applied globally at `rand_freq` step intervals.

### 3. Actor parameters
Per-env physics properties (e.g. `rigid_shape_properties` friction, `dof_properties` stiffness/damping, actor scale). Applied only to envs that just reset and have exceeded `rand_freq` steps since their last randomization. **Can only be changed on reset** (PhysX limitation).

### Timing
- Actor/sim params: on reset only
- Observation/action noise: every step

### Current config (ResDexHand.yaml)
- `gravity`: scaling, linear_decay schedule (ramps from 0 to full over 1920 steps)
- `manip_obj` friction: scaling, linear_decay schedule, 250 buckets, range [1–6×]
- No observation or action noise currently configured
