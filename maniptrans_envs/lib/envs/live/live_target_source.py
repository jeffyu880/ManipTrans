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

Velocities are computed **causally** (consecutive live frames × 120/skip, EMA-smoothed) — the
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
        ema_alpha: float = 0.4,
        stale_ms: float = 80.0,
        buffered: bool = False,
    ):
        # buffered=False: CONFLATE, always use the newest frame (real-time teleop; may skip
        #   frames if the sim is slower than the publisher).
        # buffered=True: FIFO queue, consume ONE frame per latest() call in publish order
        #   (faithful trajectory replay, like the offline 1-frame-per-step path). Lags if the
        #   sim is slower than the publisher, but loses nothing.
        self.buffered = buffered
        self.addr, self.port = addr, port
        self.device = device
        self.skip = skip
        self.ema_alpha = ema_alpha
        self.stale_ms = stale_ms
        self.mj2g = mujoco2gym_transf.to(device).float()

        self._dex = {"rh": dexhand_rh, "lh": dexhand_lh}
        self._obj_verts = {"rh": obj_verts_rh.to(device).float(), "lh": obj_verts_lh.to(device).float()}
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
        self._vel_scale = 120.0 / skip  # matches base.compute_velocity time_delta = 1/(120/skip)

        # streaming state
        self._lock = threading.Lock()
        self._raw: Optional[dict] = None      # newest wire frame
        self._raw_t: float = 0.0              # wall time it arrived
        self._anchor0: Optional[torch.Tensor] = None  # raw bottle_body pos of first frame (recenter anchor)
        self._queue = collections.deque()     # buffered mode: FIFO of unconsumed raw frames
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # causal-velocity memory, keyed per side
        self._last_seq = -1
        self._prev = {"rh": None, "lh": None}   # previous transformed targets (for finite diff)
        self._vel = {"rh": None, "lh": None}     # EMA-smoothed velocities

    # ── networking ────────────────────────────────────────────────────────────────
    def start(self, wait_first_s: float = 10.0) -> None:
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

    def stop(self):
        self._stop.set()

    def _set_anchor(self, raw):
        anchor_pos = np.asarray(raw["obj_transf"][RECENTER_ANCHOR_OBJ])[:3, 3]
        self._anchor0 = torch.tensor(anchor_pos, device=self.device, dtype=torch.float32)

    # ── transform (mirror of my_dataset loaders + base.process_data) ──────────────
    def _obj_id_for_side(self, raw, side):
        ids = raw["obj_ids"]
        return ids[0] if side == "lh" else ids[-1]   # LH=body(first), RH=cap(last)

    # the wire frame keys hands by "left"/"right"; we use "lh"/"rh" internally
    _WIRE_SIDE = {"rh": "right", "lh": "left"}

    def _transform_side(self, raw, side):
        """raw wire frame → gym-frame (wrist_pos[3], wrist_rot_aa[3], mano_joints{[3]}, obj[4,4])."""
        dev = self.device
        hand = raw["hands"][self._WIRE_SIDE[side]]
        recenter = torch.tensor(RECENTER_FINE, device=dev, dtype=torch.float32) - self._anchor0

        # wire sends joints_pos as a list in AVP `finger_names` order; map AVP -> mano joint
        # names exactly as the loader does (mano_joints[mano_name] = joints_pos[avp_name]).
        finger_names = raw["finger_names"]
        jp = hand["joints_pos"]
        avp = {finger_names[i]: jp[i] for i in range(len(finger_names))}
        mano = {
            mano_name: torch.tensor(avp[avp_name], device=dev, dtype=torch.float32)
            for mano_name, avp_name in AVP_TO_MANO_JOINTS.items()
        }

        # wrist position: pullback (0 for AVP) + dexhand offset + recenter
        wrist_pos = torch.tensor(hand["wrist_pos"], device=dev, dtype=torch.float32)
        if WRIST_PULLBACK:
            wrist_pos = wrist_pos - (mano["middle_proximal"] - wrist_pos) * WRIST_PULLBACK
        wrist_pos = wrist_pos + self._rel_t[side] + recenter
        mano = {k: v + recenter for k, v in mano.items()}

        # wrist rotation: quat → R @ dexhand offset [@ LH correction]
        wrist_R = torch.tensor(R.from_quat(hand["wrist_quat"]).as_matrix(), device=dev, dtype=torch.float32)
        wrist_R = wrist_R @ self._rel_R[side] @ self._wrist_corr[side]

        # object pose (OptiTrack frame) + recenter
        obj = torch.tensor(raw["obj_transf"][self._obj_id_for_side(raw, side)], device=dev, dtype=torch.float32)
        obj[:3, 3] = obj[:3, 3] + recenter

        # table rotation about raw Y
        tr = self._table_rot
        wrist_pos = tr @ wrist_pos
        mano = {k: tr @ v for k, v in mano.items()}
        wrist_R = tr @ wrist_R
        obj = obj.clone()
        obj[:3, 3] = tr @ obj[:3, 3]
        obj[:3, :3] = tr @ obj[:3, :3]

        # mujoco2gym
        Rg, tg = self.mj2g[:3, :3], self.mj2g[:3, 3]
        wrist_pos = Rg @ wrist_pos + tg
        mano = {k: Rg @ v + tg for k, v in mano.items()}
        wrist_R = Rg @ wrist_R
        obj = self.mj2g @ obj

        wrist_aa = rotmat_to_aa(wrist_R)
        return wrist_pos, wrist_aa, wrist_R, mano, obj

    def _tips_distance(self, mano, obj, side):
        """Nearest distance from each of the 5 fingertips to the object surface (matches base.py)."""
        verts = (obj[:3, :3] @ self._obj_verts[side].T).T + obj[:3, 3]      # [N,3] world
        tips = torch.stack([mano[k] for k in _TIP_KEYS], dim=0)            # [5,3]
        d = torch.cdist(tips, verts)                                        # [5,N]
        return d.min(dim=1).values                                         # [5]

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
        new_frame = seq != self._last_seq
        # frames skipped since we last processed one (CONFLATE drops intermediate frames when
        # the sim consumes slower than the publisher). Divide the velocity by this so the delta
        # spanning N frames isn't read as an N x too-large one-frame velocity.
        seq_gap = max(1, seq - self._last_seq) if self._last_seq >= 0 else 1
        stale = (time.time() - raw_t) * 1e3 > self.stale_ms
        out = {"seq": seq, "stale": stale, "sync_ok": bool(raw["sync"]["sync_ok"])}

        for side in ("rh", "lh"):
            wrist_pos, wrist_aa, wrist_R, mano, obj = self._transform_side(raw, side)
            cur = {"wrist_pos": wrist_pos, "wrist_R": wrist_R, "mano": mano, "obj": obj}

            if new_frame:
                self._vel[side] = self._update_velocity(side, cur, self._vel[side], seq_gap)
                self._prev[side] = cur
            vel = self._vel[side] or self._zero_velocity(mano)

            out[side] = {
                "wrist_pos": wrist_pos,
                "wrist_rot": wrist_aa,
                "wrist_velocity": vel["wrist_v"],
                "wrist_angular_velocity": vel["wrist_w"],
                "mano_joints": mano,
                "mano_joints_velocity": vel["mano_v"],
                "obj_trajectory": obj,
                "obj_velocity": vel["obj_v"],
                "obj_angular_velocity": vel["obj_w"],
                "tips_distance": self._tips_distance(mano, obj, side),
            }
        if new_frame:
            self._last_seq = seq
        return out

    # ── causal velocity (consecutive frames × vel_scale, EMA) ─────────────────────
    def _update_velocity(self, side, cur, prev_vel, seq_gap=1):
        prev = self._prev[side]
        if prev is None:
            return self._zero_velocity(cur["mano"])
        # divide the per-frame scale by the frame gap so skipped frames don't inflate velocity
        s, a = self._vel_scale / max(1, seq_gap), self.ema_alpha

        def lin(p_now, p_prev):
            return (p_now - p_prev) * s

        wrist_v = lin(cur["wrist_pos"], prev["wrist_pos"])
        obj_v = lin(cur["obj"][:3, 3], prev["obj"][:3, 3])
        wrist_w = self._ang(cur["wrist_R"], prev["wrist_R"], s)
        obj_w = self._ang(cur["obj"][:3, :3], prev["obj"][:3, :3], s)
        mano_v = {k: lin(cur["mano"][k], prev["mano"][k]) for k in cur["mano"]}

        new = {"wrist_v": wrist_v, "wrist_w": wrist_w, "obj_v": obj_v, "obj_w": obj_w, "mano_v": mano_v}
        if prev_vel is None:
            return new
        # EMA smoothing
        out = {
            "wrist_v": a * wrist_v + (1 - a) * prev_vel["wrist_v"],
            "wrist_w": a * wrist_w + (1 - a) * prev_vel["wrist_w"],
            "obj_v": a * obj_v + (1 - a) * prev_vel["obj_v"],
            "obj_w": a * obj_w + (1 - a) * prev_vel["obj_w"],
            "mano_v": {k: a * mano_v[k] + (1 - a) * prev_vel["mano_v"][k] for k in mano_v},
        }
        return out

    def _ang(self, R_now, R_prev, scale):
        rel = R_now @ R_prev.transpose(-1, -2)
        return rotmat_to_aa(rel) * scale

    def _zero_velocity(self, mano):
        z3 = torch.zeros(3, device=self.device)
        return {"wrist_v": z3.clone(), "wrist_w": z3.clone(), "obj_v": z3.clone(),
                "obj_w": z3.clone(), "mano_v": {k: z3.clone() for k in mano}}
