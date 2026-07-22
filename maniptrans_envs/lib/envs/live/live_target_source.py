"""
LiveTargetSource — desktop consumer of the live AVP+Motive stream for `--live` mode.

Subscribes (ZMQ SUB + CONFLATE) to the laptop's `live_publish.py`, unpacks each `wire`
frame (OptiTrack frame), and maps it into the **gym/table frame** the ManipTrans env expects,
mirroring exactly what `main/dataset/my_dataset_{LH,RH}.py` + `base.process_data` do offline:

    raw OptiTrack frame
      → dexhand wrist offset (relative_translation/rotation) [+ LH 180°-about-Y correction]
      → recenter (subtract first-live-frame anchor + RECENTER_FINE)
      → table rotation (TABLE_Z_ROT_DEG about raw Y)
      → mujoco2gym_transf
    = gym-frame wrist_pos / wrist_rot(axis-angle) / mano_joints / obj_trajectory

Velocities are computed **causally** (consecutive live frames × fps/skip, EMA-smoothed) — the
offline path uses a non-causal Gaussian filter that looks ahead, which a live stream can't.
`tips_distance` is the per-fingertip nearest distance to the object surface (matches base.py).

The geometric constants (RECENTER_FINE, TABLE_Z_ROT_DEG, WRIST_PULLBACK, RECENTER_ANCHOR_OBJ,
obj-id↔side assignment) are imported from the loaders so live and offline can't drift.

`latest()` returns one frame of per-side targets (tensors on `device`, no env/time axes); the
env broadcasts them across all envs and overwrites the demo buffer slots.
"""

from __future__ import annotations

import collections
import os
import sys
import threading
import time
from typing import Optional

import numpy as np
import torch
from scipy.spatial.distance import cdist as scipy_cdist
from scipy.spatial.transform import Rotation as R

# wire.py (shared msgpack frame format) lives in the Motion_Capture live_streaming copy.
_WIRE_DIRS = [
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "live_streaming", "utils"),
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "live_streaming"),
]
for _d in _WIRE_DIRS:
    _d = os.path.abspath(_d)
    if os.path.isfile(os.path.join(_d, "wire.py")) and _d not in sys.path:
        sys.path.insert(0, _d)
import wire  # noqa: E402

# Shared geometric constants + the AVP→mano joint map — single source of truth with the loaders.
from main.dataset.my_dataset_LH import (  # noqa: E402
    AVP_TO_MANO_JOINTS,
    RECENTER_ANCHOR_OBJ,
    RECENTER_FINE,
    TABLE_Z_ROT_DEG,
    WRIST_PULLBACK,
)
from main.dataset.transform import rotmat_to_aa  # axis-angle, same helper the env/loaders use

# LH wrist needs an extra 180°-about-Y correction (AVP LH frame vs MANO); RH needs none.
# Mirrors AVP_LH_WRIST_CORRECTION inside my_dataset_LH.__getitem__.
_LH_WRIST_CORRECTION = R.from_rotvec([0.0, np.pi, 0.0]).as_matrix()

_TIP_KEYS = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]


class LiveTargetSource:
    def __init__(
        self,
        *,
        addr: str = "128.178.169.131",
        port: int = 5555,
        dexhand_rh,
        dexhand_lh,
        mujoco2gym_transf: torch.Tensor,
        obj_verts_rh: torch.Tensor,
        obj_verts_lh: torch.Tensor,
        device,
        skip: int = 1,
        fps: float = 60.0,  # native live-stream rate (Hz); OptiTrack+AVP is 60Hz
        ema_alpha: float = 0.4,
        stale_ms: float = 80.0,
        buffered: bool = False,
        causal_vel_mode: str = "pos_ema",
    ):
        # buffered=False: CONFLATE, always use the newest frame (real-time teleop; may skip
        #   frames if the sim is slower than the publisher).
        # buffered=True: FIFO queue, consume ONE frame per latest() call in publish order
        #   (faithful trajectory replay, like the offline 1-frame-per-step path). Lags if the
        #   sim is slower than the publisher, but loses nothing.
        self.buffered = buffered
        # causal_vel_mode: "pos_ema" (low-pass positions, then diff — LINEAR velocities only) or
        # "vel_ema" (diff, then EMA the velocity). Angular velocity always uses vel_ema. Must match
        # base.compute_velocity so live targets equal offline targets.
        self.causal_vel_mode = causal_vel_mode
        self.addr, self.port = addr, port
        self.device = device
        self.skip = skip
        self.ema_alpha = ema_alpha
        self.stale_ms = stale_ms
        self.mj2g = mujoco2gym_transf.to(device).float()

        self._dex = {"rh": dexhand_rh, "lh": dexhand_lh}
        # numpy: consumed by _tips_distance, which runs on CPU with the rest of the frame math
        self._obj_verts = {
            "rh": obj_verts_rh.float().cpu().numpy(),
            "lh": obj_verts_lh.float().cpu().numpy(),
        }
        # per-side dexhand wrist offsets as tensors
        self._rel_t = {s: torch.tensor(self._dex[s].relative_translation, device=device, dtype=torch.float32)
                       for s in ("rh", "lh")}
        self._rel_R = {s: torch.tensor(self._dex[s].relative_rotation, device=device, dtype=torch.float32)
                       for s in ("rh", "lh")}
        self._wrist_corr = {
            "lh": torch.tensor(_LH_WRIST_CORRECTION, device=device, dtype=torch.float32),
            "rh": torch.eye(3, device=device, dtype=torch.float32),
        }
        self._table_rot = torch.tensor(
            R.from_rotvec([0.0, np.deg2rad(TABLE_Z_ROT_DEG), 0.0]).as_matrix(),
            device=device, dtype=torch.float32,
        )
        self._vel_scale = fps / skip  # = 1/time_delta; fps = native stream rate (matches base.py fps/skip)

        # Mano joints are carried as a single [N,3] tensor (not a per-joint dict) so the whole hand
        # transforms in a handful of batched ops instead of ~N per-joint kernel launches per step.
        self.mano_names = list(AVP_TO_MANO_JOINTS.keys())    # canonical row order of the [N,3] tensor
        self._avp_names = list(AVP_TO_MANO_JOINTS.values())  # AVP source names, same order (for stacking)
        mano_row = {name: i for i, name in enumerate(self.mano_names)}
        self._tip_rows = np.array([mano_row[k] for k in _TIP_KEYS], dtype=np.int64)
        self._middle_proximal_row = mano_row["middle_proximal"]

        # streaming state
        self._lock = threading.Lock()
        self._raw: Optional[dict] = None      # newest wire frame
        self._raw_t: float = 0.0              # wall time it arrived
        self._anchor0: Optional[torch.Tensor] = None  # raw bottle_body pos of first frame (recenter anchor)
        self._queue = collections.deque()     # buffered mode: FIFO of unconsumed raw frames
        self._rx_last_seq: Optional[int] = None  # last seq the rx thread saw (backwards jump = publisher restart)
        self._reset_epoch = 0                    # ++ on each restart; watched by flush_and_wait_fresh
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # causal-velocity memory, batched: wrist_obj_* tensors are [2 sides (rh, lh),
        # 2 bodies (wrist, obj), ...] so one op updates all four wrist/obj quantities
        self._last_seq = -1
        self._prev_wrist_obj_rot = None    # [2,2,3,3] previous rotations (angular diff)
        self._prev_wrist_obj_pos = None    # [2,2,3] previous positions (vel_ema linear diff)
        self._prev_mano = None             # [2,N,3] previous mano joints (vel_ema)
        self._smooth_wrist_obj_pos = None  # [2,2,3] pos_ema low-passed positions
        self._smooth_mano = None           # [2,N,3] pos_ema low-passed mano joints
        self._vel_state = None             # {"wrist_obj_v","wrist_obj_w": [2,2,3], "mano_v": [2,N,3]}
        self._out_cache = None         # latest() result for _last_seq (reused while seq holds)
        self._avp_index = None         # numpy gather: wire finger_names order -> mano_names order
        self._avp_index_names = None   # finger_names tuple the cached gather was built for

    # ── networking ────────────────────────────────────────────────────────────────
    def start(self, wait_first_s: float = 20.0) -> None:
        import zmq
        ctx = zmq.Context.instance()
        self._sock = ctx.socket(zmq.SUB)
        if self.buffered:
            self._sock.setsockopt(zmq.RCVHWM, 100000)  # keep every frame (FIFO replay); no CONFLATE
        else:
            self._sock.setsockopt(zmq.CONFLATE, 1)      # newest-only (real-time teleop)
        self._sock.setsockopt(zmq.SUBSCRIBE, b"")
        self._sock.connect(f"tcp://{self.addr}:{self.port}")
        # control PUSH: signal the publisher to restart (mock_publish's control PULL on port+1)
        self._ctrl = ctx.socket(zmq.PUSH)
        self._ctrl.setsockopt(zmq.LINGER, 0)
        self._ctrl.connect(f"tcp://{self.addr}:{self.port + 1}")
        self._thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._thread.start()
        # wait for the first frame and lock the recenter anchor from it
        deadline = time.monotonic() + wait_first_s
        while time.monotonic() < deadline:
            with self._lock:
                raw = self._raw
            if raw is not None:
                self._set_anchor(raw)
                print(f"[LiveTargetSource] connected to tcp://{self.addr}:{self.port}, first frame received")
                return
            time.sleep(0.02)
        raise TimeoutError(f"No live frame from tcp://{self.addr}:{self.port} within {wait_first_s}s")

    def _rx_loop(self):
        import zmq
        poller = zmq.Poller()
        poller.register(self._sock, zmq.POLLIN)
        while not self._stop.is_set():
            if poller.poll(200):
                frame = wire.unpack(self._sock.recv())
                with self._lock:
                    seq = int(frame["seq"])
                    # seq jumping backwards = the publisher restarted its trajectory (viewer N). Bump
                    # the epoch so flush_and_wait_fresh knows the restart landed, and drop any stale
                    # pre-restart frames still queued so buffered replay resumes cleanly at frame 0.
                    if self._rx_last_seq is not None and seq < self._rx_last_seq:
                        self._reset_epoch += 1
                        self._queue.clear()
                    self._rx_last_seq = seq
                    self._raw = frame
                    self._raw_t = time.time()
                    if self.buffered:
                        self._queue.append((frame, self._raw_t))  # FIFO: consumed one per step

    def request_publisher_reset(self):
        """Signal the publisher (mock_publish) to restart its trajectory from frame 0."""
        ctrl = getattr(self, "_ctrl", None)
        if ctrl is None:
            return
        try:
            import zmq
            ctrl.send(b"reset", zmq.NOBLOCK)
        except Exception:
            pass  # no publisher control channel listening — ignore

    def flush_and_wait_fresh(self, timeout_s: float = 0.5) -> bool:
        """After requesting a publisher restart: drop stale buffered/held frames and block (bounded)
        until the publisher's seq jumps backwards (the actual restart), so a manual reset lands on the
        fresh frame 0 rather than an in-flight/stale frame. Waiting on the restart (not merely "any new
        seq") is what makes this correct with a continuously-streaming publisher / CONFLATE mode.
        Returns True if the restart was observed, False on timeout (caller then resets on the current
        frame — correct for live teleop, where there is no trajectory restart)."""
        with self._lock:
            self._queue.clear()  # discard buffered pre-restart frames (FIFO mode)
            start_epoch = self._reset_epoch
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                restarted = self._reset_epoch != start_epoch
            if restarted:
                return True
            time.sleep(0.005)
        return False

    def stop(self):
        self._stop.set()

    def _set_anchor(self, raw):
        anchor_pos = np.asarray(raw["obj_transf"][RECENTER_ANCHOR_OBJ])[:3, 3]
        self._anchor0 = torch.tensor(anchor_pos, device=self.device, dtype=torch.float32)
        self._precompute_frame_constants()

    def _precompute_frame_constants(self):
        """Fuse every per-frame-constant transform into a few matrices so _transform_frame is a
        handful of batched ops (it runs every control step). The compositions mirror the
        step-by-step loader math, e.g. positions:
            p_gym = mj2g_R @ (table_rot @ (p + rel_t + recenter)) + mj2g_t
                  = pre_R @ p + pos_const
        and object poses: obj_gym = (mj2g @ table_rot_hom @ recenter_hom) @ obj = obj_A @ obj.
        Stored as numpy: the per-frame math runs on CPU (tensors this small are dominated by
        CUDA launch overhead on GPU) and latest() uploads one packed result per frame."""
        device = self.device
        recenter = torch.tensor(RECENTER_FINE, device=device, dtype=torch.float32) - self._anchor0
        mj2g_R, mj2g_t = self.mj2g[:3, :3], self.mj2g[:3, 3]
        pre_R = mj2g_R @ self._table_rot
        side_rel_t = torch.stack([self._rel_t["rh"], self._rel_t["lh"]])    # [2,3]
        post_R = torch.stack(
            [self._rel_R["rh"] @ self._wrist_corr["rh"], self._rel_R["lh"] @ self._wrist_corr["lh"]]
        )  # [2,3,3]
        table_rot_hom = torch.eye(4, device=device, dtype=torch.float32)
        table_rot_hom[:3, :3] = self._table_rot
        recenter_hom = torch.eye(4, device=device, dtype=torch.float32)
        recenter_hom[:3, 3] = recenter

        self._pre_R = pre_R.cpu().numpy()                                            # [3,3]
        self._pre_R_T = np.ascontiguousarray(self._pre_R.T)
        self._mano_const = (pre_R @ recenter + mj2g_t).cpu().numpy()                 # [3]
        self._pos_const = ((side_rel_t + recenter) @ pre_R.T + mj2g_t).cpu().numpy() # [2,3]
        self._post_R = post_R.cpu().numpy()                                          # [2,3,3]
        self._obj_A = (self.mj2g @ table_rot_hom @ recenter_hom).cpu().numpy()       # [4,4]
        self._out_cache = None  # anchor moved: any cached targets are invalid

    # ── transform (mirror of my_dataset loaders + base.process_data) ──────────────
    @staticmethod
    def _rotmat_to_aa_np(matrices):
        """rotmat_to_aa on CPU numpy input/output — same helper (and thus the exact same
        axis-angle convention) as the offline loaders; scipy's as_rotvec disagrees near pi."""
        return rotmat_to_aa(torch.from_numpy(matrices)).numpy()

    def _transform_frame(self, raw):
        """Wire frame → gym-frame targets for BOTH sides at once (row 0 = rh, row 1 = lh):
        wrist_pos [2,3], wrist_aa [2,3], wrist_R [2,3,3], mano [2,N,3], obj [2,4,4].

        All numpy on CPU: it runs every control step, and at these tensor sizes GPU execution
        is pure launch overhead. latest() uploads the packed result in one copy."""
        # wire sends joints_pos in AVP `finger_names` order; cache the gather that reorders the
        # rows into self.mano_names order (finger_names is constant across frames in practice)
        finger_names = raw["finger_names"]
        if self._avp_index_names != tuple(finger_names):
            wire_row = {name: i for i, name in enumerate(finger_names)}
            self._avp_index = np.array([wire_row[n] for n in self._avp_names], dtype=np.int64)
            self._avp_index_names = tuple(finger_names)

        right, left = raw["hands"]["right"], raw["hands"]["left"]
        mano_raw = np.stack(
            [np.asarray(right["joints_pos"], dtype=np.float32), np.asarray(left["joints_pos"], dtype=np.float32)]
        )[:, self._avp_index]  # [2,N,3] in mano_names row order
        wrist_pos_raw = np.stack([right["wrist_pos"], left["wrist_pos"]]).astype(np.float32)  # [2,3]
        if WRIST_PULLBACK:  # pullback (0 for AVP) uses RAW joints, before any transform
            middle_proximal = mano_raw[:, self._middle_proximal_row]
            wrist_pos_raw = wrist_pos_raw - (middle_proximal - wrist_pos_raw) * WRIST_PULLBACK
        # both wrist quats → matrices in one scipy call
        wrist_R_raw = R.from_quat(np.stack([right["wrist_quat"], left["wrist_quat"]])).as_matrix().astype(np.float32)
        obj_ids = raw["obj_ids"]  # LH = first id, RH = last id (mirrors the loaders)
        obj_raw = np.stack(
            [np.asarray(raw["obj_transf"][obj_ids[-1]], dtype=np.float32),
             np.asarray(raw["obj_transf"][obj_ids[0]], dtype=np.float32)]
        )  # [2,4,4]

        mano = mano_raw @ self._pre_R_T + self._mano_const
        wrist_pos = wrist_pos_raw @ self._pre_R_T + self._pos_const
        wrist_R = self._pre_R @ (wrist_R_raw @ self._post_R)
        obj = self._obj_A @ obj_raw
        wrist_aa = self._rotmat_to_aa_np(wrist_R)
        return wrist_pos, wrist_aa, wrist_R, mano, obj

    def _tips_distance(self, mano, obj, side):
        """Nearest distance from each of the 5 fingertips to the object surface (matches base.py)."""
        verts = self._obj_verts[side] @ obj[:3, :3].T + obj[:3, 3]          # [N,3] world
        tips = mano[self._tip_rows]                                         # [5,3]
        return scipy_cdist(tips, verts).min(axis=1).astype(np.float32)      # [5]

    # ── public ────────────────────────────────────────────────────────────────────
    def latest(self) -> dict:
        """Per-side gym-frame targets for THIS step (single frame, tensors on device).

        buffered=False: newest received frame. buffered=True: the next frame in publish order
        (one popped per call → faithful replay); holds the last if the queue is momentarily empty.
        """
        with self._lock:
            if self.buffered and self._queue:
                self._raw, self._raw_t = self._queue.popleft()  # advance exactly one frame
            raw = self._raw
            raw_t = self._raw_t
        if raw is None or self._anchor0 is None:
            raise RuntimeError("LiveTargetSource.latest() called before start()/first frame")

        seq = int(raw["seq"])
        stale = (time.time() - raw_t) * 1e3 > self.stale_ms
        if seq == self._last_seq and self._out_cache is not None:
            # same frame as the previous call: targets are unchanged, only freshness can differ
            self._out_cache["stale"] = stale
            return self._out_cache
        if 0 <= seq < self._last_seq:
            # seq jumped backwards = the publisher restarted its trajectory. Differencing across
            # the restart would fabricate a huge one-frame velocity (last pre-restart pose ->
            # frame 0, x 60Hz) that the EMA then feeds the policy for ~0.5s — the post-reset
            # hand blow-up. Forget all motion history and re-seed from this frame with zero
            # velocity, matching the env's zeroed reset state.
            self._prev_wrist_obj_rot = None
            self._last_seq = -1
            self._out_cache = None
        # frames skipped since we last processed one (CONFLATE drops intermediate frames when
        # the sim consumes slower than the publisher). Divide the velocity by this so the delta
        # spanning N frames isn't read as an N x too-large one-frame velocity.
        seq_gap = max(1, seq - self._last_seq) if self._last_seq >= 0 else 1

        wrist_pos, wrist_aa, wrist_R, mano, obj = self._transform_frame(raw)
        wrist_obj_pos = np.stack([wrist_pos, obj[:, :3, 3]], axis=1)  # [2,2,3] (side, wrist|obj, xyz)
        wrist_obj_rot = np.stack([wrist_R, obj[:, :3, :3]], axis=1)   # [2,2,3,3]
        vel = self._update_velocity(wrist_obj_pos, mano, wrist_obj_rot, seq_gap)
        tips = np.stack(
            [self._tips_distance(mano[row], obj[row], side) for row, side in enumerate(("rh", "lh"))]
        )  # [2,5]

        # everything so far is CPU numpy — ship it to the device in ONE copy, hand out views
        pieces = (wrist_pos, wrist_aa, vel["wrist_obj_v"], vel["wrist_obj_w"], mano, vel["mano_v"], obj, tips)
        packed = torch.from_numpy(np.concatenate([piece.ravel() for piece in pieces])).to(self.device)
        views, offset = [], 0
        for piece in pieces:
            views.append(packed[offset : offset + piece.size].view(piece.shape))
            offset += piece.size
        wrist_pos_d, wrist_aa_d, wrist_obj_v_d, wrist_obj_w_d, mano_d, mano_v_d, obj_d, tips_d = views

        out = {"seq": seq, "stale": stale, "sync_ok": bool(raw["sync"]["sync_ok"])}
        for row, side in enumerate(("rh", "lh")):
            out[side] = {
                "wrist_pos": wrist_pos_d[row],
                "wrist_rot": wrist_aa_d[row],
                "wrist_velocity": wrist_obj_v_d[row, 0],
                "wrist_angular_velocity": wrist_obj_w_d[row, 0],
                "mano_joints": mano_d[row],
                "mano_joints_velocity": mano_v_d[row],
                "obj_trajectory": obj_d[row],
                "obj_velocity": wrist_obj_v_d[row, 1],
                "obj_angular_velocity": wrist_obj_w_d[row, 1],
                "tips_distance": tips_d[row],
            }
        self._last_seq = seq
        self._out_cache = out
        return out

    # ── causal velocity (consecutive frames × vel_scale, EMA) ─────────────────────
    def _update_velocity(self, wrist_obj_pos, mano, wrist_obj_rot, seq_gap=1):
        """Causal velocities for both sides in one batch. wrist_obj_pos [2,2,3] and
        wrist_obj_rot [2,2,3,3] are [side (rh, lh), body (wrist, obj), ...] so each update
        below is one kernel for all four wrist/obj quantities instead of four chains.

        pos_ema: LINEAR velocities from low-passed positions (EMA) then backward-diff — matches
        base.compute_velocity(causal_mode='pos_ema'); with seq_gap=1, velocity[t] =
        a*(cur - prev)*vel_scale, identical to the offline pos_ema.
        vel_ema: backward-diff raw positions, then EMA the velocity.
        ANGULAR velocity always uses vel_ema (backward-diff raw rotations, then EMA)."""
        if self._prev_wrist_obj_rot is None:
            # first frame: seed the state, emit zero velocities (the EMA then blends up from zero)
            self._vel_state = {
                "wrist_obj_v": np.zeros_like(wrist_obj_pos),
                "wrist_obj_w": np.zeros_like(wrist_obj_pos),
                "mano_v": np.zeros_like(mano),
            }
            self._smooth_wrist_obj_pos = wrist_obj_pos.copy()
            self._smooth_mano = mano.copy()
            self._prev_wrist_obj_pos = wrist_obj_pos
            self._prev_mano = mano
            self._prev_wrist_obj_rot = wrist_obj_rot
            return self._vel_state

        # divide the per-frame scale by the frame gap so skipped frames don't inflate velocity
        scale, alpha = self._vel_scale / max(1, seq_gap), self.ema_alpha
        prev_vel = self._vel_state

        # angular: backward-diff raw rotations, then EMA (both modes)
        relative_rot = wrist_obj_rot @ self._prev_wrist_obj_rot.swapaxes(-1, -2)  # [2,2,3,3]
        raw_w = self._rotmat_to_aa_np(relative_rot.reshape(-1, 3, 3)).reshape(2, 2, 3) * scale
        wrist_obj_w = alpha * raw_w + (1 - alpha) * prev_vel["wrist_obj_w"]

        if self.causal_vel_mode == "pos_ema":
            # low-pass the positions, then diff the smoothed signal (no further EMA on velocity)
            new_smooth_pos = alpha * wrist_obj_pos + (1 - alpha) * self._smooth_wrist_obj_pos
            new_smooth_mano = alpha * mano + (1 - alpha) * self._smooth_mano
            wrist_obj_v = (new_smooth_pos - self._smooth_wrist_obj_pos) * scale
            mano_v = (new_smooth_mano - self._smooth_mano) * scale
            self._smooth_wrist_obj_pos = new_smooth_pos
            self._smooth_mano = new_smooth_mano
        else:  # vel_ema
            wrist_obj_v = (
                alpha * ((wrist_obj_pos - self._prev_wrist_obj_pos) * scale)
                + (1 - alpha) * prev_vel["wrist_obj_v"]
            )
            mano_v = alpha * ((mano - self._prev_mano) * scale) + (1 - alpha) * prev_vel["mano_v"]

        self._prev_wrist_obj_pos = wrist_obj_pos
        self._prev_mano = mano
        self._prev_wrist_obj_rot = wrist_obj_rot
        self._vel_state = {"wrist_obj_v": wrist_obj_v, "wrist_obj_w": wrist_obj_w, "mano_v": mano_v}
        return self._vel_state
