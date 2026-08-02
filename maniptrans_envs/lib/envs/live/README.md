# Live streaming → ManipTrans (`--live`)

Feed the ManipTrans Isaac Gym policy with a **live** hand+object target stream (Apple Vision
Pro hands + OptiTrack/Motive objects) instead of a preloaded demo — for teleoperation, or for
replaying a recorded `.pkl` to test the pipeline. Enabled with `live=true`.

The publisher lives in the **Motion_Capture** repo (`src/live_streaming/`); the consumer +
env integration live here in **ManipTrans**. They agree only on the `wire` msgpack frame.

---

## Architecture

```
══════════════════════════════════════════════════════════════════════════════════════════
   PUBLISHER  (Motion_Capture repo — laptop for teleop, or desktop for replay testing)
══════════════════════════════════════════════════════════════════════════════════════════

  ┌─ REAL TELEOP ──────────────────────────┐     ┌─ REPLAY TEST ───────────────────────────┐
  │ src/live_streaming/live_publish.py     │     │ src/live_streaming/debug/mock_publish.py │
  │                                        │     │                                          │
  │ main() 60 Hz loop:                     │     │ main() loop (--once / loop, --rate-hz):  │
  │  • OptitrackPoseBuffer.snapshot()      │     │  • resolve_pkl(name)  → find the .pkl     │
  │  • AVPHandBuffer.snapshot()            │     │  • build_frame(raw,t) → one recorded step │
  │  • staleness gates (NaN if stale)      │     │    (dict→list joints, obj poses)          │
  │  • _build_frame():                     │     │                                          │
  │      align_frame()  ◄─── utils/        │     │  (pkl is already OptiTrack-frame,         │
  │        live_transform.py  (AVP+Motive  │     │   so no align_frame needed)               │
  │        → OptiTrack frame; SSOT w/ the  │     │                                          │
  │        offline recorder)               │     │                                          │
  └──────────────────┬─────────────────────┘     └───────────────────┬──────────────────────┘
                     │                                                │
                     └────────────► wire.pack()  ◄────────────────────┘
                        utils/wire.py  (msgpack frame: obj_transf, hands[left/right]
                        {wrist_pos, wrist_quat, joints_pos[25]}, finger_names, sync, seq)
                                            │
                                     ZMQ PUB .bind()
                                            │
════════════════════════════════════════════│═══════════════════════════════════════════════
     TRANSPORT   tcp://…:5555                │   teleop: CONFLATE (newest-only)
                                            │   replay: no CONFLATE + high HWM (every frame)
════════════════════════════════════════════│═══════════════════════════════════════════════
                                            │
                                     ZMQ SUB .connect()
                                            ▼
══════════════════════════════════════════════════════════════════════════════════════════
   CONSUMER — desktop sim (ManipTrans repo)
══════════════════════════════════════════════════════════════════════════════════════════

  maniptrans_envs/lib/envs/live/live_target_source.py  ── class LiveTargetSource
  ┌────────────────────────────────────────────────────────────────────────────────────┐
  │  start()        SUB connect + spawn rx thread; lock recenter anchor (first frame)    │
  │  _rx_loop()     recv → wire.unpack() → store newest  (or append to FIFO deque if     │
  │                 buffered=liveBuffered)                                                │
  │  latest()   ◄── called once per sim step. buffered? popleft() one : newest.          │
  │      ├─ _transform_side()   OptiTrack→gym: dexhand offset [+LH 180°] → recenter →     │
  │      │                      TABLE_Z_ROT → mujoco2gym   (mirrors my_dataset loaders)   │
  │      ├─ _update_velocity()  causal EMA, ÷ seq-gap (skip-robust)                       │
  │      └─ _tips_distance()    fingertip → obj-surface min dist                          │
  │  returns per-side {wrist_pos/rot(+vel), mano_joints(+vel), obj_traj(+vel), tips}      │
  └───────────────────────────────────────┬────────────────────────────────────────────┘
                                          │  one gym-frame target frame
                                          ▼
  maniptrans_envs/lib/envs/tasks/dexhandmanip_bih.py  ── DexHandManipBiHEnv
  ┌────────────────────────────────────────────────────────────────────────────────────┐
  │  __init__          read flags: live / liveAddr / livePort / liveBuffered; cap        │
  │                    max_demo_length=4                                                  │
  │  _create_envs()    build demo_data_{rh,lh} [num_envs, nT, …] from dataIndices        │
  │                    (assets/BPS/opt-init/shapes) — created once                        │
  │  post_physics_step()                                                                  │
  │      ├─ _ensure_live_source()   lazily construct + start() LiveTargetSource           │
  │      ├─ _inject_live()          latest() → OVERWRITE demo_data target slots,          │
  │      │                          broadcast across all envs (in place)                  │
  │      ├─ compute_observations()  reads demo_data slots (now = live)  → obs             │
  │      ├─ compute_reward()                                                              │
  │      ├─ reset_buf[:] = 0        (no auto-reset in live)                                │
  │      └─ clamp progress_buf ≤ seq_len-1  (tiny buffer, unclamped reward read)          │
  │  _reset_default_side()   live: init fingers = dexhand default (imitator takes over)   │
  └───────────────────────────────────────┬────────────────────────────────────────────┘
                                          ▼  obs dict (proprio, privileged, target)
  lib/rl/res_models.py + network_builder_residual_bih.py
  ┌────────────────────────────────────────────────────────────────────────────────────┐
  │  FROZEN RH imitator ─┐                                                                │
  │  FROZEN LH imitator ─┼─► base_action                                                  │
  │  Residual MLP ───────┘─► residual_action        action = base + residual              │
  └───────────────────────────────────────┬────────────────────────────────────────────┘
                                          ▼
              pre_physics_step() → PD finger targets + wrist force → gym.simulate()
                                          │
                                          └────► back to post_physics_step (next step)
```

---

## Main functions / where they live

| Function | File (repo) | Role |
|---|---|---|
| `align_frame()` | `utils/live_transform.py` (Motion_Capture) | AVP+Motive → **OptiTrack** frame; shared with offline recorder (SSOT) |
| `pack()` / `unpack()` / `SCHEMA_VERSION` | `utils/wire.py` (Motion_Capture) | msgpack wire frame (NaN-safe); the publisher↔consumer contract |
| `main()` / `_build_frame()` | `live_publish.py` (Motion_Capture) | real teleop publisher (OptiTrack+AVP → PUB) |
| `main()` / `build_frame()` / `resolve_pkl()` | `debug/mock_publish.py` (Motion_Capture) | replay a `.pkl` as a wire stream (test) |
| `LiveTargetSource.start()` / `_rx_loop()` | `live/live_target_source.py` (ManipTrans) | SUB connect, rx thread, newest-or-FIFO buffer, recenter anchor |
| `LiveTargetSource.latest()` | `live/live_target_source.py` | one gym-frame target/step; drives transform + velocity + tips |
| `_transform_side()` | `live/live_target_source.py` | **OptiTrack → gym** (mirrors `my_dataset_{RH,LH}.py`) |
| `_update_velocity()` | `live/live_target_source.py` | causal EMA velocity, ÷ seq-gap (skip-robust) |
| `_inject_live()` | `tasks/dexhandmanip_bih.py` (ManipTrans) | overwrite `demo_data` target slots with the live frame, broadcast to all envs |
| `post_physics_step()` | `tasks/dexhandmanip_bih.py` | live hook: inject → obs → reward, no-reset, clamp `progress_buf` |
| `_create_envs()` | `tasks/dexhandmanip_bih.py` | build `demo_data_{rh,lh}` from `dataIndices` (assets/shapes, once) |
| `_reset_default_side()` | `tasks/dexhandmanip_bih.py` | live reset: fingers = dexhand default, imitator takes over |

---

## The three seams (why live == offline)

1. **`wire.py`** — the only contract between publisher and consumer (msgpack schema + `SCHEMA_VERSION`).
2. **`align_frame`** (`live_transform.py`) — AVP↔OptiTrack, shared by `live_publish` **and** the
   offline recorder, so the recorded `.pkl` and the live stream carry identical content.
3. **`_transform_side`** (`live_target_source.py`) — OptiTrack→gym, mirrors the `my_dataset`
   loaders so live targets equal offline targets. **This is the one duplicated piece — keep it
   in sync with `main/dataset/my_dataset_{RH,LH}.py` + `base.process_data` (the LH wrist correction
   and `mujoco2gym`).** The constants themselves (`TABLE_Z_ROT_DEG`, `RECENTER_FINE`,
   `WRIST_PULLBACK`) and the obj↔hand assignment are no longer duplicated — both ends import them
   from `main/dataset/object_sets.py`.

---

## How the demo buffer is used in live mode

`demo_data_{rh,lh}` are still built once from the reference demo (`dataIndices`) for object
assets, BPS, `opt_*` reset init, and buffer shapes — but `nT` is capped to 4 and **their target
slots are overwritten in place every step** by `_inject_live()` with the latest live frame,
broadcast across all envs. `progress_buf` is frozen inside `[0, seq_len-1]` and auto-reset is
disabled, so it tracks continuously. `pack_data`'s `.squeeze()` requires **`num_envs ≥ 2`**.

---

## Config knobs

Top-level in `main/cfg/config.yaml`, plumbed through `main/cfg/task/ResDexHand.yaml`:

| Flag | Default | Meaning |
|---|---|---|
| `live` | `False` | master switch for the whole live path |
| `liveAddr` | `128.178.169.131` | address the desktop SUB connects to (laptop IP; `127.0.0.1` for local replay) |
| `livePort` | `5555` | ZMQ port |
| `liveBuffered` | `False` | `True` = FIFO, consume **every** frame in order (faithful replay); `False` = newest-only/CONFLATE (real-time teleop, may skip) |
| `objectSet` | `bottle` | which props are on the table — see **Object sets** below |
| `sharedObject` | `False` | **reward only**: split the object terms when both hands score one body |

---

## Object sets (scored objects vs props)

The reference demo passed as `dataIndices` only supplies buffer shapes and the retargeted reset
init in live mode, so **its** objects need not be the ones actually on the table. `objectSet` names
the real set. Offline the loaders infer the same set from what the capture recorded, so a set means
the same thing either way. Registry: [`main/dataset/object_sets.py`](../../../../main/dataset/object_sets.py).

A set declares, per prop, the Motive rigid-body names it may be published under (matched by **name**,
case-insensitively — never by position in `obj_ids`, which would silently swap the hands if Motive
reordered), which asset to spawn, and its role:

| Set | RH scores | LH scores | Prop (never scored) | Anchor |
|---|---|---|---|---|
| `bottle` | `bottle_cap` | `bottle_body` | — | `bottle_body` |
| `cup_brush` | `d2_brush` | `d2_brush` | `d2_cup` | `d2_cup` |

* **Scored** bodies are what each hand's observation, reward and failure check track. There are at
  most two. When both hands score the **same** body (`cup_brush`: the brush is moved and rotated in
  the air bimanually) the env spawns **one** scored actor and the LH side aliases it — two
  overlapping copies of one body is never a valid scene. That is inferred from the objects, not
  configured. Pair it with `sharedObject=true` so the body's reward is credited once in total
  rather than once per hand.
* A **prop** is spawned as a free rigid body and collided with, but is never a reward or failure
  target — the cup exists so the brush has something to be placed into and held upright by. It is
  placed from its tracked pose on **reset only**; it is deliberately not teleported every step,
  which would push force through its contact with the manipulated body.

> The policy does not *see* the prop: observations carry BPS shape and `tips_distance` only for
> scored objects. The prop exists to physics alone.

Adding a set: drop the mesh in `data/my_dataset/obj_files/obj_vis/<name>.{obj,ply}` and its COACD
decomposition + urdf in `coacd/`, then add one entry to `OBJECT_SETS`.

---

## Running it

**Prereqs:** `pyzmq` + `msgpack` in the `maniptrans` env; a reference demo passed as `dataIndices`;
frozen imitator checkpoints + a residual checkpoint.

### A. Replay a recorded `.pkl` (no rig — pipeline test)

```bash
# publisher (desktop): dump one pass into the buffer fast so the sim never starves
python <Motion_Capture>/src/live_streaming/debug/mock_publish.py \
    --pkl m_170805 --once --rate-hz 200 --addr 0.0.0.0 --port 5555

# sim (desktop): liveBuffered=true → consume every frame in order (like offline)
conda run -n maniptrans python main/rl/train.py \
    task=ResDexHand dexhand=inspire side=BiH headless=false \
    num_envs=16 test=true randomStateInit=false \
    live=true liveBuffered=true liveAddr=127.0.0.1 livePort=5555 \
    dataIndices=[m_170805] \
    rh_base_model_checkpoint=assets/imitator_rh_inspire.pth \
    lh_base_model_checkpoint=assets/imitator_lh_inspire.pth \
    "checkpoint='runs/<run>/nn/<run>.pth'"
```

### B. Real teleop (laptop publishes AVP+Motive)

```bash
# laptop:
python3 src/live_streaming/live_publish.py --addr 0.0.0.0 --port 5555
# desktop (default liveBuffered=false = newest-only, real-time):
conda run -n maniptrans python main/rl/train.py ... live=true liveAddr=<laptop_ip> livePort=5555 ...
```

Start the publisher **before** the sim — `LiveTargetSource.start()` waits ~10 s for the first frame.

---

## Debugging notes

- **Hands go crazy on a moving stream** → rate mismatch. `--static 0` on the publisher isolates
  it: stable ⇒ timing (use `liveBuffered=true`, or the seq-gap velocity fix handles skips);
  crazy ⇒ transform/data (check `_transform_side` vs the loaders).
- **CUDA index-out-of-bounds** → `progress_buf` past the tiny buffer (fixed by the clamp); re-run
  with `CUDA_LAUNCH_BLOCKING=1` to get the true line.
- **KeyError `'rh'`** → the wire frame keys hands by `left`/`right`; `_transform_side` maps
  `rh→right`, `lh→left`.
- **`.squeeze()` shape errors** → run with `num_envs ≥ 2`.
```
