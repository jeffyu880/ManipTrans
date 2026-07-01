#!/usr/bin/env python3
"""
Step 0 echo server — runs on the REMOTE DESKTOP (the Isaac Gym / ManipTrans machine).

Validates the live-streaming transport before any real data flows. It connects a ZMQ
REP socket back through the reverse SSH tunnel to the laptop's bound REQ socket and
echoes every frame verbatim, so the laptop can measure true end-to-end RTT.

Transport / tunnel direction
----------------------------
The laptop opens the reverse tunnel and BINDS; the desktop CONNECTs:

    # on the laptop (capture machine):
    ssh -R 5555:localhost:5555 <user>@<desktop>

    # on the desktop (this script):
    python3 echo_server.py --port 5555

A connection to the desktop's localhost:5555 is forwarded by `ssh -R` back to the
laptop's localhost:5555, where rtt_probe.py is bound. So this side always CONNECTs to
its own localhost — never change that to bind().

This mirrors the real stream's bind/connect direction (laptop PUB binds, desktop SUB
connects), so if this works, the data path's tunnel direction is validated too.
"""

from __future__ import annotations

import argparse

import zmq


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 0 ZMQ echo server (desktop side).")
    ap.add_argument("--addr", default="127.0.0.1",
                    help="Address to connect to (the tunnel endpoint on THIS host). Default 127.0.0.1.")
    ap.add_argument("--port", type=int, default=5555, help="Port (must match the ssh -R forward). Default 5555.")
    args = ap.parse_args()

    endpoint = f"tcp://{args.addr}:{args.port}"
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.connect(endpoint)  # connect, not bind — see module docstring
    print(f"[echo_server] REP connected to {endpoint}; echoing frames (Ctrl+C to stop)")

    n = 0
    try:
        while True:
            payload = sock.recv()       # raw bytes; we do not deserialize, just bounce
            sock.send(payload)          # echo verbatim — keeps RTT measurement on one clock
            n += 1
            if n % 600 == 0:
                print(f"[echo_server] echoed {n} frames")
    except KeyboardInterrupt:
        print(f"\n[echo_server] stopped after {n} frames")
    finally:
        sock.close(linger=0)
        ctx.term()


if __name__ == "__main__":
    main()
