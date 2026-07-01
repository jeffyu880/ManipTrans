#!/usr/bin/env python3
"""
Wire format for the live AVP+Motive → ManipTrans stream.

Single definition of the on-the-wire frame, imported by BOTH ends:
  * laptop  `live_publish.py`  — packs aligned frames and PUBs them
  * desktop `LiveTargetSource` (Step 3) — unpacks them for the policy

msgpack payload, one dict per timestep. NaN survives the round trip (msgpack encodes it as
a float64 NaN), so stale/missing fields stay NaN exactly as in the offline `.pkl` — the
desktop must still gate on the `sync` flags before using a frame.

Frame schema (v1)
-----------------
{
  "v": 1,                       # schema version
  "seq": int,                   # monotonic frame counter (gap => dropped publish)
  "t_capture_s": float,         # laptop wall clock (time.time()) at snapshot
  "obj_ids": [str, ...],        # e.g. ["bottle_body", "bottle_cap"]
  "obj_transf": {oid: 4x4},     # object pose in the OptiTrack frame; NaN 4x4 if stale/missing
  "finger_names": [str * 25],   # AVP joint order
  "hands": {
    "left" | "right": {
      "wrist_pos":  [3],        # OptiTrack frame
      "wrist_quat": [4],        # xyzw (scipy)
      "joints_pos": [[3] * 25], # in finger_names order
    },
  },
  "sync": {
    "optitrack_age_ms": float, "optitrack_sync_ok": bool, "optitrack_max_age_ms": float,
    "avp_age_ms": float,       "avp_sync_ok": bool,       "avp_max_age_ms": float,
    "sync_ok": bool,           # optitrack_sync_ok AND avp_sync_ok
  },
}
"""

from __future__ import annotations

import struct

import msgpack
import numpy as np

SCHEMA_VERSION = 1


def _to_py(o):
    """Recursively convert numpy arrays/scalars to plain Python so msgpack can encode them."""
    if isinstance(o, np.ndarray):
        return o.astype(float).tolist()
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, dict):
        return {k: _to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_py(v) for v in o]
    return o


def pack(frame: dict) -> bytes:
    """Serialize a frame dict (numpy arrays allowed) to msgpack bytes."""
    return msgpack.packb(_to_py(frame), use_bin_type=True)


def unpack(buf: bytes) -> dict:
    """Deserialize msgpack bytes back to a frame dict (lists, not numpy)."""
    return msgpack.unpackb(buf, raw=False)


# ── optional length-prefixed file tee (for raw recording → offline reconstruction) ──

def write_framed(fh, buf: bytes) -> None:
    """Append one length-prefixed msgpack frame to an open binary file."""
    fh.write(struct.pack("<I", len(buf)))
    fh.write(buf)


def read_framed(fh):
    """Yield successive frame dicts from a file written by write_framed()."""
    while True:
        hdr = fh.read(4)
        if len(hdr) < 4:
            return
        (n,) = struct.unpack("<I", hdr)
        buf = fh.read(n)
        if len(buf) < n:
            return
        yield unpack(buf)
