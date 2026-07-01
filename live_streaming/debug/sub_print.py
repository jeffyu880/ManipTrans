#!/usr/bin/env python3
"""
Step 2 verification subscriber — runs on the DESKTOP. Skeleton for Step 3's LiveTargetSource.

SUBscribes to the live stream through the reverse SSH tunnel, unpacks each frame with the
shared `wire` format, and prints a throttled status line: frame rate, dropped frames (from
`seq` gaps), sync flags, and a couple of sample values so you can eyeball that real data is
flowing and aligned. No sim, no policy — just confirms the publish path.

Run (desktop), after the laptop's live_publish.py is up and the tunnel is open:

    python3 sub_print.py --port <desktopPort>

`<desktopPort>` is the listen port of the `ssh -R <desktopPort>:localhost:<laptopPort>` forward
(the desktop connects to its own localhost, tunneled back to the laptop's PUB).

Note on latency: `age_ms` here is desktop_now − laptop_t_capture, so it includes any clock
offset between the two machines. Drop count (from seq gaps) is clock-free and is the reliable
health metric. For a true one-way latency number, sync clocks (chrony/NTP) first.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# wire.py lives in live_streaming/utils/; this script is in live_streaming/debug/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import zmq  # noqa: E402

from utils import wire  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Live stream verification subscriber (desktop side).")
    ap.add_argument("--addr", default="127.0.0.1", help="Connect address (desktop end of the tunnel).")
    ap.add_argument("--port", type=int, default=5555, help="Connect port (ssh -R listen port). Default 5555.")
    ap.add_argument("--print-every", type=int, default=60, help="Print a status line every N frames. Default 60.")
    args = ap.parse_args()

    endpoint = f"tcp://{args.addr}:{args.port}"
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.CONFLATE, 1)          # keep only the newest frame (latest-sample semantics)
    sock.setsockopt(zmq.SUBSCRIBE, b"")        # all messages
    sock.connect(endpoint)
    print(f"[sub_print] SUB connected to {endpoint}; waiting for frames (Ctrl+C to stop)")

    n = 0
    last_seq = None
    dropped = 0
    t_first = None
    try:
        while True:
            frame = wire.unpack(sock.recv())
            now = time.time()
            if t_first is None:
                t_first = now

            seq = frame["seq"]
            if last_seq is not None and seq > last_seq + 1:
                dropped += seq - last_seq - 1   # CONFLATE legitimately skips; this counts the skips
            last_seq = seq
            n += 1

            if args.print_every and n % args.print_every == 0:
                s = frame["sync"]
                age_ms = (now - frame["t_capture_s"]) * 1e3
                # show every object's position (translation column of its 4x4), e.g. bottle_body + bottle_cap
                objs = "  ".join(
                    f"{oid}=[{M[0][3]:.3f},{M[1][3]:.3f},{M[2][3]:.3f}]"
                    for oid, M in ((o, frame["obj_transf"][o]) for o in frame["obj_ids"])
                )
                rwp = frame["hands"]["right"]["wrist_pos"]
                fps = n / (now - t_first) if now > t_first else 0.0
                print(f"[{seq:06d}] {fps:5.1f} fps  skips={dropped}  sync_ok={s['sync_ok']} "
                      f"(ot={s['optitrack_sync_ok']}/avp={s['avp_sync_ok']})  age~{age_ms:7.1f}ms  "
                      f"{objs}  R.wrist=[{rwp[0]:.3f},{rwp[1]:.3f},{rwp[2]:.3f}]")
    except KeyboardInterrupt:
        print(f"\n[sub_print] stopped after {n} frames (seq skips: {dropped})")
    finally:
        sock.close(linger=0)
        ctx.term()


if __name__ == "__main__":
    main()
