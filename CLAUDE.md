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
| `NM` (mirrored) | `oakink2_mirrored` |
| other | `favor` |

Factory appends `_rh` or `_lh` and looks up the registered dataset class.

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
| `actionsMovingAverage` | `1.0` | Temporal smoothing on actions (0.6 typical for BiH) |
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
| `useTrajAug` | `False` | Enable trajectory augmentation (random XY offset ±3cm, Z-rotation ±10°) at load time |
| `numTrajAug` | `20` | Number of pre-augmented demo versions per demo (envs cycle through these) |
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

## Adding a New Dataset

1. Subclass `ManipData` (`main/dataset/base.py`)
2. Implement `__getitem__` returning dict with keys: `data_path`, `obj_id`, `obj_verts`, `obj_urdf_path`, `obj_trajectory`, `wrist_pos`, `wrist_rot`, `mano_joints`
3. Decorate with `@register_manipdata("yourtype_rh")` / `@register_manipdata("yourtype_lh")`
4. Add index prefix detection in `ManipDataFactory.dataset_type()` in `main/dataset/factory.py`
