# ManipTrans Datasets

Authoritative reference for dataset formats, index routing, annotation/pickle layouts, the
MyDataset (OptiTrack + AVP) capture, the LH-cut trimming workflow, and how to add a new dataset.
(Relocated from `CLAUDE.md` to keep that file lean.)

See also: [pipeline.md](pipeline.md#step-0--data-sources--indexing) for how these feed the
pipeline.

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
| `m_...` (contains `m_`) | `mydataset` |
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

## Dataset: MyDataset (OptiTrack + AVP capture)

A custom bimanual capture: hand poses from **Apple Vision Pro** (AVP) hand tracking, object poses from **OptiTrack**. Already recorded at 60 Hz, so the loaders use `skip=1`. Pickles live flat in `data/my_dataset/*.pkl`. Loaders: [my_dataset_RH.py](../main/dataset/my_dataset_RH.py) / [my_dataset_LH.py](../main/dataset/my_dataset_LH.py), registered as `mydataset_rh` / `mydataset_lh`.

### Index convention (`m_`)

The factory routes any index **containing `m_`** to `mydataset` ([factory.py](../main/dataset/factory.py) `dataset_type`). The index is an `m_` marker plus a **trailing suffix of the pkl filename stem** — typically the last digits. The loaders strip the leading `m_` ([my_dataset_RH.py](../main/dataset/my_dataset_RH.py) `__getitem__`) and resolve to the unique pkl whose stem ends with the suffix:

```
file:  data/my_dataset/optitrack_recording_20260618_m_160009.pkl
index: m_160009   →  strip "m_" → "160009" → matches stem ending in "160009"  →  that file
```

Since the pkl stems already end in `..._m_<digits>`, the marker and the suffix coincide. If the suffix matches zero or >1 pkls the loader raises `AssertionError`. (Full stem also works.)

No special shell/Hydra quoting is needed — `m_160009` has no comment or lexer-special characters:

```bash
dataIndices=[m_160009]          # ✅ training / Hydra override
--data_idx m_160009             # ✅ mano2dexhand retargeting
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

### Object sets: assets and hand ↔ object assignment

The capture pkl stores `obj_mesh_path` / `obj_urdf_path` as `None`, so assets are resolved by name in [`main/dataset/object_sets.py`](../main/dataset/object_sets.py). Names in `OBJ_ASSETS` map to OakInk-v2 meshes; anything else is looked up under `data/my_dataset/obj_files/` (`obj_vis/<name>.{ply,obj}` for verts/BPS, `coacd/<name>.urdf` for sim), so a new prop only needs its two files dropped in.

| `obj_id` | Object | Mesh (verts/BPS) | URDF (sim) |
|---|---|---|---|
| `bottle_body` | burner body | `object_preview/align_ds/O02@0206@00002/scan.ply` | `coacd_object_preview/align_ds/O02@0206@00002/scan.urdf` |
| `bottle_cap` | burner cap | `object_preview/align_ds/O02@0206@00001/scan.ply` | `coacd_object_preview/align_ds/O02@0206@00001/scan.urdf` |
| `cup` | 3D-printed cup | `my_dataset/obj_files/obj_vis/cup.obj` | `my_dataset/obj_files/coacd/cup.urdf` |
| `square_brush` | 3D-printed brush | `my_dataset/obj_files/obj_vis/square_brush.obj` | `my_dataset/obj_files/coacd/square_brush.urdf` |

**Hand ↔ object assignment** comes from the capture's **object set**, inferred from the `obj_id`s it recorded (`infer_object_set`). The same registry serves live streaming, where `objectSet` names it explicitly because there is no pkl to infer from.

| Set | Capture tracks | RH scores | LH scores | Prop (never scored) | Recenter anchor |
|---|---|---|---|---|---|
| `bottle` | `bottle_body`, `bottle_cap` | `bottle_cap` | `bottle_body` | — | `bottle_body` |
| `cup_brush` | `d2_cup`, `d2_brush` | `square_brush` | `square_brush` | `cup` | `cup` |
| *(fallback)* | anything else | `obj_id[-1]` | `obj_id[0]` | — | `bottle_body` if present, else `obj_id[0]` |

The fallback reproduces the historical positional rule, which is what single-tracked-object captures like `SHARED_OBJ_m_170751.pkl` (`obj_id == ['bottle_cap']`) rely on — first and last are the same body, so both hands score it.

- **Scored** bodies drive each hand's obs, reward and failure check. Both hands scoring the *same* body (`cup_brush`, where the brush is manipulated bimanually) makes the env spawn **one** scored actor and alias the LH side to it. Pair with `sharedObject=true`, which is purely a reward toggle: it splits the object terms so the body is credited once in total instead of once per hand.
- A **prop** is spawned as a free rigid body and collided with, but is never a reward or failure target — the cup exists so the brush can be placed into it. The loaders emit `prop_obj_id` / `prop_urdf_path` / `prop_trajectory`; the env places it on reset only. The policy never *sees* it (BPS and `tips_distance` cover scored objects only).

### Cap mesh geometry (tracked pose vs. opening)

The tracked object pose is the cap mesh's **local origin `(0,0,0)`** — what `obj_trajectory` positions and the reward's `manip_obj_pos` follows. For the burner cap (`O02@0206@00001/scan.ply`), the cap's symmetry axis is **Y** (~4.8 cm tall, ~1.1–1.7 cm radius), and a hollowness analysis along Y shows:

- **Opening** = the **Y-min** end (rim at Y ≈ 0.016 m): slices there are rings (no central verts) → hollow mouth where the cap meets the bottle.
- **Closed top** = the **Y-max** end (Y ≈ 0.064 m): slices have central verts (a covering dome).

Relative to the opening, the tracked origin is **on the central axis** but **~1.6 cm below/outside the opening rim** (i.e. past the mouth, where the bottle neck would insert) — *not* at the cap's center or closed top. Closed top is 6.4 cm from the origin.

This is the geometry in the **OakInk cap mesh frame**. In a MyDataset capture the cap is positioned by the **OptiTrack** `obj_transf`, so whether the physical opening actually lands 1.6 cm from the tracked point depends on how the OptiTrack rigid-body origin was defined on the real cap — verify the OptiTrack→mesh alignment if the cap looks offset in sim.

### Trimming the terminal LH "retract" motion (LH cuts)

After the cap is placed, the operator's **left hand yanks the burner body away** in one quick direction at the end of every `cap_*` capping demo. That retract isn't part of the capping task and pollutes training (the policy would learn to fling the object), so it is **cut** off the end of each demo.

**Detection is directional, not speed-magnitude** (`detect_terminal_cut` in [data_stats/plot_lh_cut_analysis.py](../data_stats/plot_lh_cut_analysis.py)):
- `left_dir` = unit vector of the LH wrist's **net end-of-trajectory displacement** (median position over the first half → final position). This is the per-demo "left"/retract direction — data-driven, computed separately for each demo.
- Project the LH wrist velocity onto `left_dir` → `v_left`, smooth it (`v_s`).
- **Cut = the first frame** (in the latter part of the clip) where `v_s` rises above a low onset threshold (`onset_thr=0.15`) **and stays above it for `min_run` consecutive frames** (sustained motion in the left direction), within a run that peaks above `peak_min=0.5` m/s. The `min_run` guard prevents a single spike from causing a false cut; the cut sits **exactly on the threshold crossing** in the plot.
- Sideways capping wiggles project to ~0 on `left_dir`, so they don't trigger it. Works even when the hand **settles** before the clip ends (the burst need not reach the final frames).

**Scripts:**

| Script | What it does |
|---|---|
| [data_stats/plot_lh_cut_analysis.py](../data_stats/plot_lh_cut_analysis.py) | **Plot-only** (never cuts). Writes a 2×2 figure per demo to `vis_traj_outputs/lh_cut_analysis/<stem>.png` (wrist pos raw+retargeted, LH object pos, `v_left` with threshold+cut, LH object speed). Demos listed in its `DEMOS`. |
| [data_stats/apply_lh_cuts_all.py](../data_stats/apply_lh_cuts_all.py) | Detect + plot **all** `cap_*` demos; with `--apply`, also trims them. Default (no flag) = detect+plot only. |
| [data_stats/apply_lh_cuts.py](../data_stats/apply_lh_cuts.py) | Original hardcoded-`CUTS` version for the first 5 demos. |

**Applying a cut** (per demo, keep `[0, cut-1]`) truncates **all three frame-synced files** together so RH/LH/object stay in sync: the raw `data/my_dataset/cap_*.pkl` and both retargeting pkls `mano2inspire_{lh,rh}/<stem>_{lh,rh}.pkl`. Every time-indexed field (first dim == `T`) is sliced; non-temporal fields (`finger_names`, meta, calibration) are left alone and raw `meta.n_frames` is updated.

**Backups / idempotency:** the full original is preserved once as `<name>_original.pkl` (never overwritten), and trims always read **from** that backup — so re-running with different cut params reproduces from full data, and detection/plotting read `_original` to show the untrimmed trajectory. To revert, copy each `_original.pkl` back over its base name.

```bash
# review proposed cuts for all cap demos (no writes)
python data_stats/apply_lh_cuts_all.py
# then apply (trims raw + lh + rh, creating _original backups)
python data_stats/apply_lh_cuts_all.py --apply
```

### RH loader table-rotation fix + retargeting rotation

`my_dataset_RH.py` was missing the `TABLE_Z_ROT_DEG` (90°) table rotation that `my_dataset_LH.py`
applies — it now applies it (both hands consistent). **Consequence:** RH retargeted `opt_*`
generated *before* this fix lack the 90° and are misaligned with the now-rotated raw RH targets.
Fix without re-running `mano2dexhand`:
```bash
python data_stats/rotate_rh_retarget.py <file_rh.pkl>   # or no arg = all under data/retargeting/my_dataset/mano2*_rh/
```
It bakes `T = M · Ry(90°) · M⁻¹` into `opt_wrist_pos/rot` + `opt_joints_pos` (dof unchanged), is
idempotent (stamps `_table_rot_deg_applied`), and backs up to `<name>_prerot90.pkl`. **Do not run
it on a pkl regenerated with the fixed loader** — that already includes the rotation.

---

## Adding a New Dataset

1. Subclass `ManipData` (`main/dataset/base.py`)
2. Implement `__getitem__` returning dict with keys: `data_path`, `obj_id`, `obj_verts`, `obj_urdf_path`, `obj_trajectory`, `wrist_pos`, `wrist_rot`, `mano_joints`
3. Decorate with `@register_manipdata("yourtype_rh")` / `@register_manipdata("yourtype_lh")`
4. Add index prefix detection in `ManipDataFactory.dataset_type()` in `main/dataset/factory.py`

The codebase is designed for **60 FPS** data — preprocess higher/lower-FPS data first or the
transfer quality suffers. Reference implementations: `main/dataset/grab_dataset_dexhand.py` and
`main/dataset/oakink2_dataset_dexhand_rh.py`.
