# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git
- Never add `Co-Authored-By: Claude` or any self-attribution to commit messages.

## Communication
- When making code changes, always show which file and line(s) are being modified and briefly describe what is changing before or alongside the edit.

# ManipTrans

**ManipTrans: Efficient Dexterous Bimanual Manipulation Transfer via Residual Learning** (CVPR 2025)

Transfers human hand-object manipulation demonstrations to dexterous robot hands using a two-stage residual learning approach in Isaac Gym.

---

## Project Structure

```
ManipTrans/
├── main/
│   ├── cfg/                        # Hydra configs (config.yaml, task/, rl_train/)
│   ├── dataset/                    # Dataset loaders (OakInk-V2, GRAB, FAVOR)
│   └── rl/                         # Training entry point, eval scoring
├── maniptrans_envs/lib/envs/
│   ├── tasks/                      # Isaac Gym environments
│   │   ├── dexhandimitator.py      # Stage 1: imitator env (single hand)
│   │   ├── dexhandmanip_bih.py     # Stage 2: bimanual residual env
│   │   └── dexhandmanip_sh.py      # Stage 2: single-hand residual env
│   └── dexhands/                   # Hand configs (inspire, shadow, allegro, ...)
├── lib/rl/
│   ├── network_builder_residual_bih.py   # BiH residual network
│   ├── network_builder_residual_sh.py    # Single-hand residual network
│   └── res_models.py                     # Residual model wrappers
├── DexManipNet/                    # DexManipNet dataset utilities
└── data/
    ├── OakInk-v2/                  # Raw OakInk-V2 annotations and meshes
    ├── retargeting/                # Preprocessed retargeted data (generated)
    ├── body_utils/body_models/smplx/
    ├── mano_v1_2/
    └── smplx_extra/body_upper_idx.pt
```

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

---

### Stage 2: Residual Policy

The residual policy (`ResDexHand`) is trained on top of **frozen** imitators. For bimanual tasks, both RH and LH imitators are loaded and kept in eval mode.

**Forward pass (bimanual):**
1. Full obs is `[rh_obs, lh_obs]` concatenated for each key.
2. RH imitator processes `obs[:, :rh_slice]` → `rh_base_action`
3. LH imitator processes `obs[:, half:half+lh_slice]` → `lh_base_action`
4. Residual MLP input: `[encoded_full_obs, rh_base_action, lh_base_action]`
5. Residual MLP output: `delta_action` (same dims as total action)
6. Final action sent to sim = `base_action + delta_action` (combined in agent)

The residual network is in [lib/rl/network_builder_residual_bih.py](lib/rl/network_builder_residual_bih.py) and model wrapper in [lib/rl/res_models.py](lib/rl/res_models.py).

---

### Target Observation (per hand, concatenated in this order)

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

### Observation Inputs (per network)

The BiH Stage-2 policy is **three networks**. All read from the same obs dict `{proprioception, privileged, target}`, but from different slices. The two imitators are frozen; only the residual MLP is trained. Per-hand dims shown for **inspire** (`n_dofs=12`, `n_bodies=18`): proprioception=49, privileged=49, target=350.

| Network | Trainable? | Obs slice it reads | Input dims | Produces |
|---|---|---|---|---|
| **RH imitator** (frozen) | ✗ eval-only | **RH half** of each key — `obs[:, :49]` / `[:, :49]` / `[:, :350]` ([res_models.py:296-298](lib/rl/res_models.py#L296-L298)) | 49+49+350 = 448 | `rh_base_action` (6+n_dofs) |
| **LH imitator** (frozen) | ✗ eval-only | **LH half** — `obs[:, half:half+49]` etc. ([res_models.py:300-303](lib/rl/res_models.py#L300-L303)) | 448 | `lh_base_action` (6+n_dofs) |
| **Residual MLP** | ✓ **trained policy** | `encode(full obs)` — both halves: prop 98 + priv 98 + target 700 ([network_builder_residual_bih.py:160](lib/rl/network_builder_residual_bih.py#L160)) | encoded + `rh_base_action` + `lh_base_action` ([:182](lib/rl/network_builder_residual_bih.py#L182)) | `delta_action` |

Final sim action = `base_action + delta_action`, combined in `pre_physics_step`. Each frozen imitator sees only its own hand's half and is unaware of the other hand; only the residual MLP sees both hands jointly (plus both base actions), which is what lets it coordinate bimanual contact.

**Symbolic dims (per hand):**
- `proprioception` = `13 + n_dofs*3` (base_state + q/cos_q/sin_q)
- `privileged` = `n_dofs + 13 + 5*4 + 3 + 1` (dq + obj pose/vel + tip_force + obj com + obj weight)
- `target` = `128 (bps) + 5 (gt_tips) + (23 + (n_bodies-1)*9 + 23 + n_bodies)*K`

### Observation Sources (where each value comes from)

Three upstream sources: **① live PhysX sim** (refreshed every step in `_refresh` → `_update_states`), **② demo data** (SMPLX/MANO + raw anno, loaded at reset), **③ static precompute** (object constants). Built in `compute_observations_side` ([dexhandmanip_bih.py:1446](maniptrans_envs/lib/envs/tasks/dexhandmanip_bih.py#L1446)).

| Obs key | Component | Source | Notes |
|---|---|---|---|
| **proprioception** | q, cos_q, sin_q | ① live DOF state | from `self._q` |
| | base_state | ① live root state | **position zeroed** in obs ([:1457](maniptrans_envs/lib/envs/tasks/dexhandmanip_bih.py#L1457)) |
| **privileged** | dq | ① live DOF velocity | |
| | manip_obj_pos/quat/vel/ang_vel | ① live object root | pos made wrist-relative |
| | tip_force | ① live **net contact force** | `net_cf` at 5 fingertips |
| | manip_obj_com | ① live (obj quat) + ③ static com offset | |
| | manip_obj_weight | ③ static (mass × g) | |
| **target** | delta_wrist/_joints/_obj `*` | ② demo `@progress_buf+1` **minus** ① live state | the `delta_*` mix both sources |
| | wrist_vel, wrist_quat, manip_obj_* targets | ② demo (SMPLX wrist / obj_trajectory) | |
| | obj_to_joints | ① live (obj pos → live joint pos) | |
| | gt_tips_distance | ② demo (MANO fingertip → nearest obj surface point, Chamfer) | computed in [base.py:119-132](main/dataset/base.py#L119-L132) |
| | bps | ③ static object-shape encoding | |
| *(residual only)* | rh_base_action, lh_base_action | frozen imitator outputs | [network_builder_residual_bih.py:165-176](lib/rl/network_builder_residual_bih.py#L165-L176) |

`proprioception` + `privileged` are pure **live sim state** (where the robot is now); only `target` carries **demo intent** (where it should go), and even there `delta_*` terms are `demo − live`. Observations enter at runtime from exactly two places — the PhysX sim tensors and the demo buffers — plus a few static object constants.

---

### Step Loop (inputs → physics → outputs)

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

## Dataset: OakInk-V2

Located at `data/OakInk-v2/`.

### Task Index Format

Tasks are referenced as `HHHHH@S` where:
- `HHHHH` = first 5 characters of the 20-char sequence hash (e.g. `20aed` from `20aed35da30d4b869590`)
- `S` = integer stage/primitive index within that sequence

Example: `20aed@0` = primitive 0 of sequence `scene_03__A004++seq__20aed35da30d4b869590__2023-04-22-18-45-27`

The `_bih` suffix (e.g. `20aed@0_bih`) is used in task lists and filenames but stripped when passing to `--data_idx`.

### Dataset Type Routing (`main/dataset/factory.py`)

| Index format | Dataset type |
|---|---|
| `HHHHH@S` | `oakink2` |
| `gN` | `grabdemo` |
| `vN` | `visionpro` |
| `#...` (contains `#`) | `mydataset` |
| `NM` (mirrored) | `oakink2_mirrored` |
| other | `favor` |

Factory appends `_rh` or `_lh` and looks up the registered dataset class. See [Dataset: MyDataset](#dataset-mydataset-optitrack--avp-capture) for the `#` index convention.

### Bimanual vs. Single-Hand Tasks

Check `data/OakInk-v2/program/program_info/<seq>.json`. Each key is `(str(lh_interval), str(rh_interval))`:
- Both intervals non-None → **bimanual**
- `lh_interval = None` → **right-hand-only** (LH dataset will raise `AssertionError`)
- `rh_interval = None` → **left-hand-only** (RH dataset will raise `AssertionError`)

Script to batch-check:
```python
import json, glob
PROG = "data/OakInk-v2/program/program_info"
hash_to_file = {f.split("__")[2][:5]: f for f in glob.glob(f"{PROG}/*.json")}
for h, stage in [("20aed", 0), ...]:
    raw = json.load(open(hash_to_file[h]))
    keys = [eval(k) for k in raw]
    lh, rh = keys[stage]
    print(h, stage, "BIMANUAL" if lh and rh else "LH-ONLY" if rh is None else "RH-ONLY")
```

### Confirmed Bimanual Tasks (burner_list.txt subset, sorted by intersected demo length)

```
667dd@0_bih    1124 frames
20aed@0_bih    1069
beb8f@0_bih     896
e49f5@0_bih     729
dee46@7_bih     724
11cb7@4_bih     693
a950e@0_bih     685
3b1e6@12_bih    680
d6fe3@0_bih     673
8e5df@13_bih    624
a78a0@1_bih     620
0f900@10_bih    608
db0a0@17_bih    597
81a95@7_bih     592
b5fa3@10_bih    504
f7d37@18_bih    489
85abe@4_bih     476
b0b13@11_bih    385
751fb@16_bih    276
```

### Raw Data Layout

```
data/OakInk-v2/
├── data/
│   └── scene_0x__y00z++<20-char-hash>__YYYY-mm-dd-HH-MM-SS/
│       ├── <camera_serial_0>/
│       │   ├── <frame_id>.jpg
│       │   └── ...
│       └── <camera_serial_3>/
│           └── ...
├── anno_preview/
│   └── scene_0x__...json.pkl     # per-sequence annotation pickle
├── object_preview/align_ds/
│   └── <obj_id>/<model>.obj|ply
├── coacd_object_preview/align_ds/ # COACD decomposed meshes + URDFs (generated)
│   └── <obj_id>/<model>.urdf
├── object_affordance/
│   ├── affordance_label.json
│   ├── instance_id.json
│   ├── object_affordance.json
│   ├── object_part_tree.json
│   └── part_desc.json
└── program/
    ├── program_info/              # task stage definitions
    ├── desc_info/                 # textual descriptions
    ├── initial_condition_info/    # initial conditions and recipes
    ├── pdg/                       # primitive dependency graphs
    ├── task_list.txt
    ├── task_list_filtered.txt
    └── burner_list.txt
```

### Annotation Pickle (`anno_preview/<seq>.pkl`)

```python
{
    'cam_def':          dict[str, str],                     # serial → camera name
    'cam_selection':    list[str],                          # selected camera names
    'frame_id_list':    list[int],                          # image frame ids (120Hz)
    'cam_intr':         dict[str, dict[int, np.ndarray]],   # [3,3] intrinsics per frame
    'cam_extr':         dict[str, dict[int, np.ndarray]],   # [4,4] extrinsics per frame
    'mocap_frame_id_list': list[int],                       # mocap frame ids (120Hz)
    'obj_list':         list[str],                          # object part ids in seq
    'obj_transf':       dict[str, dict[int, np.ndarray]],   # [4,4] object transforms
    'raw_smplx':        dict[int, dict[str, torch.Tensor]], # SMPLX body params per frame
    'raw_mano':         dict[int, dict[str, torch.Tensor]], # MANO hand params per frame
}
```

**raw_smplx** (per frame):
```python
{
    'body_shape':       Tensor[1, 300],
    'expr_shape':       Tensor[1, 10],
    'jaw_pose':         Tensor[1, 1, 4],     # quat [w,x,y,z]
    'leye_pose':        Tensor[1, 1, 4],
    'reye_pose':        Tensor[1, 1, 4],
    'world_rot':        Tensor[1, 4],        # quat [w,x,y,z]
    'world_tsl':        Tensor[1, 3],
    'body_pose':        Tensor[1, 21, 4],    # quat [w,x,y,z], lower body unused
    'left_hand_pose':   Tensor[1, 15, 4],   # quat [w,x,y,z]
    'right_hand_pose':  Tensor[1, 15, 4],
}
```

**raw_mano** (per frame):
```python
{
    'rh__pose_coeffs':  Tensor[1, 16, 4],   # quat [w,x,y,z]
    'lh__pose_coeffs':  Tensor[1, 16, 4],
    'rh__tsl':          Tensor[1, 3],
    'lh__tsl':          Tensor[1, 3],
    'rh__betas':        Tensor[1, 10],
    'lh__betas':        Tensor[1, 10],
}
```

### Program Info (`program/program_info/<seq>.json`)

Keys are `(str(lh_interval), str(rh_interval))` where each interval is `[start_frame, end_frame]` or `None`.

```python
{
    "(lh_interval, rh_interval)": {
        "primitive":        str,         # primitive id
        "obj_list":         list[str],
        "interaction_mode": str,         # "lh_main" | "rh_main" | "bh_main"
        "primitive_lh":     str,
        "primitive_rh":     str,
        "obj_list_lh":      list[str],
        "obj_list_rh":      list[str],
    }
}
```

When loading, the dataset intersects `lh_interval ∩ rh_interval` as the active frame range for bimanual stages. OakInk-V2 is 120Hz; the dataset loaders downsample by `skip=2` to 60Hz to match the sim.

### Desc Info / Initial Condition / PDG

```
program/desc_info/<seq>.json       # { interval_key: {"seg_desc": str} }
program/initial_condition_info/<seq>.json  # { interval_key: {"initial_condition": [...], "recipe": [...]} }
program/pdg/<seq>.json             # { "id_map": dict, "v": list[int], "e": list[list[int]] }
```

### Object Affordance

```
object_affordance/affordance_label.json
{
    'all_label':                        list[str],
    'affordance_label':                 list[str],   # part functions
    'affordance_instantiation_label':   list[str],   # interactions & primitive tasks
}

object_affordance/object_affordance.json
{
    obj_part_id: {
        "obj_part_id":              str,
        "is_instance":              bool,   # maps to full object instance
        "has_model":                bool,   # has segmented mesh
        "affordance":               list[str],
        "affordance_instantiation": list[str],
    }
}

object_affordance/object_part_tree.json
{ obj_part_id: list[str] }    # children of each part

object_affordance/instance_id.json
[ obj_part_id, ... ]          # part ids that are full instances

object_affordance/part_desc.json
{ obj_id: {"obj_id": str, "obj_name": str} }
```

### Object Models

```
object_preview/align_ds/<obj_id>/*.obj|ply        # raw meshes
coacd_object_preview/align_ds/<obj_id>/*.urdf     # COACD + URDF (must be generated)
```

Generate COACD for each object:
```bash
python maniptrans_envs/lib/utils/coacd_process.py \
    -i data/OakInk-v2/object_preview/align_ds/<id>/<model>.obj \
    -o data/OakInk-v2/coacd_object_preview/align_ds/<id>/<model>.obj \
    --max-convex-hull 32 --seed 1 -mi 2000 -md 5 -t 0.07
```

---

## Dataset: MyDataset (OptiTrack + AVP capture)

A custom bimanual capture: hand poses from **Apple Vision Pro** (AVP) hand tracking, object poses from **OptiTrack**. Already recorded at 60 Hz, so the loaders use `skip=1`. Pickles live flat in `data/my_dataset/*.pkl`. Loaders: [my_dataset_RH.py](main/dataset/my_dataset_RH.py) / [my_dataset_LH.py](main/dataset/my_dataset_LH.py), registered as `mydataset_rh` / `mydataset_lh`.

### Index convention (`#`)

The factory routes any index **containing `#`** to `mydataset` ([factory.py](main/dataset/factory.py) `dataset_type`). The index is a `#` marker plus a **trailing suffix of the pkl filename stem** — typically the last digits. The loaders strip the `#` and resolve to the unique pkl whose stem ends with the suffix:

```
file:  data/my_dataset/optitrack_recording_20260618_#160009.pkl
index: #160009   →  matches stem ending in "160009"  →  that file
```

If the suffix matches zero or >1 pkls the loader raises `AssertionError`. (Full stem also works.)

**Shell/Hydra quoting:** `#` starts a comment in *both* bash and Hydra/OmegaConf, so it must be protected. Pass the whole override double-quoted with inner single quotes:

```bash
"dataIndices=['#160009']"      # ✅ reaches Hydra as ['#160009']
dataIndices=[#160009]          # ❌ bash truncates / Hydra lexer error
dataIndices=['#160009']        # ❌ bash strips quotes → Hydra lexer error
```

### Pickle structure

```python
{
    'meta':         dict,   # fps, task_type, source, created, n_frames, obj_ids,
                            #   obj_frames, avp_to_opti_transform, note,
                            #   (optional) avp_to_mano_joints
    'obj_id':       list[str],                 # e.g. ['bottle_body', 'bottle_cap']
    'frame_id_list':list[int],
    'timestamps_s': ...,
    'obj_transf':   dict[str, ndarray[T,4,4]], # obj_id -> world transforms
    'sync':         ...,
    'hands': {
        'right': {'wrist_pos':[T,3], 'wrist_quat':[T,4] xyzw,
                  'joints_pos': dict[avp_joint_name -> [T,3]],
                  'wrist_mat':[T,3,3], 'joints_mat':...},
        'left':  { ... same ... },
        'avp_age_ms': ..., 'avp_sync_ok': ..., 'finger_names': ...,
    },
}
```

AVP joint names are mapped to ManipTrans `mano_joints` names via `meta['avp_to_mano_joints']` (fallback: the `AVP_TO_MANO_JOINTS` table in each loader).

### Object assets are hard-coded

The capture pkl stores `obj_mesh_path` / `obj_urdf_path` as `None`, so each loader hard-codes them in an `OBJ_ASSETS` dict keyed by the pkl's `obj_id`, reusing the OakInk-v2 **alcohol burner** body + cap meshes/urdfs:

| `obj_id` | Object | Mesh (`.ply`, verts/BPS) | URDF (`.urdf`, sim) |
|---|---|---|---|
| `bottle_body` | burner body | `object_preview/align_ds/O02@0206@00002/scan.ply` | `coacd_object_preview/align_ds/O02@0206@00002/scan.urdf` |
| `bottle_cap` | burner cap | `object_preview/align_ds/O02@0206@00001/scan.ply` | `coacd_object_preview/align_ds/O02@0206@00001/scan.urdf` |

**Hand ↔ object assignment:** RH holds the **cap** (`obj_id[-1]` = `bottle_cap`), LH holds the **body** (`obj_id[0]` = `bottle_body`) — matching the `b5fa3@10` capping convention.

### Cap mesh geometry (tracked pose vs. opening)

The tracked object pose is the cap mesh's **local origin `(0,0,0)`** — what `obj_trajectory` positions and the reward's `manip_obj_pos` follows. For the burner cap (`O02@0206@00001/scan.ply`), the cap's symmetry axis is **Y** (~4.8 cm tall, ~1.1–1.7 cm radius), and a hollowness analysis along Y shows:

- **Opening** = the **Y-min** end (rim at Y ≈ 0.016 m): slices there are rings (no central verts) → hollow mouth where the cap meets the bottle.
- **Closed top** = the **Y-max** end (Y ≈ 0.064 m): slices have central verts (a covering dome).

Relative to the opening, the tracked origin is **on the central axis** but **~1.6 cm below/outside the opening rim** (i.e. past the mouth, where the bottle neck would insert) — *not* at the cap's center or closed top. Closed top is 6.4 cm from the origin.

This is the geometry in the **OakInk cap mesh frame**. In a MyDataset capture the cap is positioned by the **OptiTrack** `obj_transf`, so whether the physical opening actually lands 1.6 cm from the tracked point depends on how the OptiTrack rigid-body origin was defined on the real cap — verify the OptiTrack→mesh alignment if the cap looks offset in sim.

<!-- ### Known blockers (before a run works) -->

<!-- 1. **numpy 2.x pickle** — the pkl was saved with numpy ≥2.0; the env has numpy 1.23.5, so `pickle.load` throws `ModuleNotFoundError: No module named 'numpy._core'`. Needs a `numpy._core → numpy.core` shim or re-saving the pkl.
2. **Retargeting** — loaders read `data/retargeting/my_dataset/mano2{dexhand}/{stem}_{rh,lh}.pkl` for the reset state; it must be generated first with `mano2dexhand.py` (both sides). `mano2dexhand.py` also lacks a `mydataset` save branch. -->

---

## Training

Entry point: `main/rl/train.py` (Hydra-based, rl_games framework).

### Step 1: Preprocessing (retargeting)

Optimizes a collision-free trajectory from MANO to the dexterous hand. Output saved to `data/retargeting/OakInk-v2/mano2{dexhand}/`.

### Retargeted PKL Format

Each retargeted file (e.g. `data/retargeting/OakInk-v2/mano2inspire_rh/<seq>@<stage>.pkl`) contains **only inspire hand data**:

```python
{
    'opt_wrist_pos':   ndarray[T, 3],     # inspire wrist position (world space, meters)
    'opt_wrist_rot':   ndarray[T, 3],     # inspire wrist rotation (axis-angle)
    'opt_dof_pos':     ndarray[T, n_dofs],# inspire finger joint angles
    'opt_joints_pos':  ndarray[T, 18, 3], # inspire rigid body positions (world space)
}
```

These are used **only to initialize the sim state at reset**. They are NOT used as policy tracking targets.

### Dataset Loading: Two Separate Sources

The dataset loader (`oakink2_dataset_dexhand_rh.py`) merges two sources into one `data` dict per sequence:

1. **Raw OakInk-V2 anno pkl** → runs SMPLX forward pass → extracts human fingertip/joint world positions → stored as `data["mano_joints"]` (dict of joint_name → `Tensor[T, 3]`) and `data["wrist_pos"]`, `data["wrist_rot"]`.

2. **Retargeted pkl** → loaded via `load_retargeted_data()` → adds `opt_wrist_pos`, `opt_wrist_rot`, `opt_dof_pos`, `opt_joints_pos` to the same `data` dict.

### What each source is used for

| Key | Source | Used for |
|---|---|---|
| `mano_joints` | SMPLX on raw anno | **Policy tracking targets** — `delta_joints_pos` in target obs, joint reward |
| `wrist_pos`, `wrist_rot` | SMPLX on raw anno | **Policy tracking targets** — `delta_wrist_pos` in target obs, wrist reward |
| `obj_trajectory` | Raw anno | **Policy tracking targets** — object pose reward |
| `opt_wrist_pos`, `opt_wrist_rot` | Retargeted pkl | **Sim reset only** — places inspire wrist at correct initial pose |
| `opt_dof_pos` | Retargeted pkl | **Sim reset only** — sets inspire finger joints at correct initial angles |

The policy learns to move the **inspire fingertips** (tracked from sim rigid body state) to match where the **human MANO fingertips** were in the demo. The retargeted data only provides a good initial configuration.

```bash
# Single hand
python main/dataset/mano2dexhand.py --data_idx g0 --dexhand inspire --headless --iter 2000

# Bimanual (run both sides)
python main/dataset/mano2dexhand.py --data_idx 20aed@0 --side right --dexhand inspire --headless --iter 7000
python main/dataset/mano2dexhand.py --data_idx 20aed@0 --side left  --dexhand inspire --headless --iter 7000
```

### Step 2: Train Imitator (Stage 1)

```bash
python main/rl/train.py task=DexHandImitator dexhand=inspire side=RH headless=true \
    num_envs=4096 dataIndices=[g0] experiment=imitator_rh_inspire
```

### Step 3: Train Residual Policy (Stage 2)

**Single hand:**
```bash
python main/rl/train.py task=ResDexHand dexhand=inspire side=RH headless=true \
    num_envs=4096 learning_rate=2e-4 test=false randomStateInit=true \
    rh_base_model_checkpoint=assets/imitator_rh_inspire.pth \
    lh_base_model_checkpoint=assets/imitator_lh_inspire.pth \
    dataIndices=[g0] early_stop_epochs=100 actionsMovingAverage=0.4 \
    experiment=cross_g0_inspire
```

**Bimanual:**
```bash
python main/rl/train.py task=ResDexHand dexhand=inspire side=BiH headless=true \
    num_envs=4096 learning_rate=2e-4 test=false randomStateInit=true \
    dataIndices=[20aed@0] \
    rh_base_model_checkpoint=assets/imitator_rh_inspire.pth \
    lh_base_model_checkpoint=assets/imitator_lh_inspire.pth \
    early_stop_epochs=1000 actionsMovingAverage=0.4 \
    experiment=cross_20aed@0_inspire
```

### Testing

```bash
python main/rl/train.py task=ResDexHand dexhand=inspire side=BiH headless=false \
    num_envs=4 test=true randomStateInit=false \
    dataIndices=[20aed@0] actionsMovingAverage=0.4 \
    rh_base_model_checkpoint=assets/imitator_rh_inspire.pth \
    lh_base_model_checkpoint=assets/imitator_lh_inspire.pth \
    "checkpoint='runs/cross_20aed@0_inspire__<date>/nn/cross_20aed@0_inspire.pth'"
```

Note: if the checkpoint path or experiment name contains commas (e.g. multi-demo names like `d6fe3@0,8e5df@13`), wrap the value in single quotes inside double quotes to avoid Hydra parsing it as a list: `"experiment='name_with,comma'"`.

### Key CLI Parameters

| Parameter | Default | Description |
|---|---|---|
| `task` | — | `DexHandImitator` or `ResDexHand` |
| `side` | — | `RH`, `LH`, `BiH` |
| `dexhand` | `inspire` | `inspire`, `shadow`, `allegro`, `artimano`, `xhand`, `inspireftp` |
| `dataIndices` | — | List of task indices, e.g. `[20aed@0]` or `[g0]`. Strip `_bih`/`_rh`/`_lh` suffixes. |
| `num_envs` | `8192` | Parallel envs (8192 typical for training, 4 for testing) |
| `maxDemoLength` | `None` | Cap all demos to this many frames (useful for balanced multi-demo training) |
| `early_stop_epochs` | `9999999` | Epochs without improvement before stopping (1000 for complex tasks) |
| `actionsMovingAverage` | `1.0` | Temporal smoothing on actions. **Prefer 0.6 for BiH** — empirically better than 0.4 or 1.0. |
| `randomStateInit` | `True` | RSI — start from random demo frame (true for train, false for test) |
| `usePIDControl` | `False` | Use PID wrist control instead of direct position |
| `headless` | `True` | Disable rendering (true for training) |
| `checkpoint` | `''` | Path to `.pth` to resume or test |
| `learning_rate` | `5e-4` | PPO learning rate (2e-4 typical for residual policy) |
| `max_iterations` | `9999999` | Hard cap on training iterations |
| `wandb_activate` | `False` | Enable Weights & Biases logging |
| `wandb_entity` | `None` | W&B entity (username or team) |
| `wandb_project` | `None` | W&B project name |
| `save_rollouts` | `False` | Save rollout episodes to `rollouts.hdf5` (for eval or distillation) |
| `num_rollouts_to_save` | `10000` | Max rollouts to write to HDF5 before stopping |
| `num_rollouts_to_run` | `1e10` | Max completed episodes before stopping; must be `> num_envs * 2` to pass warmup |
| `save_successful_rollouts_only` | `True` | If false, save both successful and failed rollouts |
| `useTrajAug` | `False` | Enable trajectory augmentation (random XY offset ±3cm, Z-rotation ±10°) at load time. **Must be `true` for any augmentation to occur** — it is the master switch; `useLHObjCenterAug` and `numTrajAug` have no effect without it. |
| `numTrajAug` | `20` | Number of pre-augmented demo versions per demo (envs cycle through these) |
| `useLHObjCenterAug` | `False` | Rotate augmentation around the left-hand object center instead of the table center. Requires `useTrajAug=true`. |
| `jointNoiseCm` | `0.0` | Gaussian noise std (cm) added to MANO joint keypoint positions — simulates hand pose estimator error |
| `useCoaxialReward` | `False` | Extra reward for pen/cap Z-axis alignment (pen capping tasks) |
| `usePenKeypointReward` | `False` | Extra reward for pen tip proximity to cap opening |
| `evalStartFrame` | `0` | Frame index to start evaluation rollouts from |

### Output Structure

```
runs/<experiment>__<MM-DD-HH-MM-SS>/
├── config.yaml
├── demos.txt
└── nn/
    └── <experiment>.pth
```

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

Controlled by three flags (all require `useTrajAug=true` as master switch):

| Flag | What rotates | What is fixed |
|---|---|---|
| `useTableCenterAug` | everything | table center (XY plane) |
| `useLHObjCenterAug` | RH demo only | LH object position at each frame |
| `useRHObjCenterAug` | RH demo only | RH object position at each frame |

When multiple flags are enabled they **chain**: LH-obj-center is applied first, then RH-obj-center, then table-center — each operating on the already-transformed result from the previous step.

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

---

## Adding a New Dataset

1. Subclass `ManipData` (`main/dataset/base.py`)
2. Implement `__getitem__` returning dict with keys: `data_path`, `obj_id`, `obj_verts`, `obj_urdf_path`, `obj_trajectory`, `wrist_pos`, `wrist_rot`, `mano_joints`
3. Decorate with `@register_manipdata("yourtype_rh")` / `@register_manipdata("yourtype_lh")`
4. Add index prefix detection in `ManipDataFactory.dataset_type()` in `main/dataset/factory.py`
