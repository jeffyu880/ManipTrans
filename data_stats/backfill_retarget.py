"""Backfill the `retarget_fps` stamp into pre-Jul-9 mydataset retarget pkls.

These pkls were produced before commit 16dc096 added the stamp, so they hold full
60 Hz frames (1:1 with the raw capture) but no `retarget_fps` key. Without it the
loader assumes retarget_skip=1 and skips subsampling, breaking the length assertion
under demoTargetFps=30. Native mydataset rate is 60 Hz, so stamp 60.0 -- but only
after verifying len(opt_wrist_pos) actually equals the raw frame count.
"""
import os
import pickle

RAW_DIR = "../data/my_dataset"
RET_DIRS = {
    "lh": ("../data/retargeting/my_dataset/mano2inspire_lh", "left"),
    "rh": ("../data/retargeting/my_dataset/mano2inspire_rh", "right"),
}
NATIVE_FPS = 60.0

def raw_frame_count(stem, side):
    raw = pickle.load(open(os.path.join(RAW_DIR, f"{stem}.pkl"), "rb"))
    return len(raw["hands"][side]["wrist_pos"])

for suffix, (ret_dir, side) in RET_DIRS.items():
    stamped = skipped_present = skipped_mismatch = missing_raw = 0
    for f in sorted(os.listdir(ret_dir)):
        if not f.endswith(f"_{suffix}.pkl"):
            continue
        stem = f[: -len(f"_{suffix}.pkl")]
        path = os.path.join(ret_dir, f)
        p = pickle.load(open(path, "rb"))
        if "retarget_fps" in p:
            skipped_present += 1
            continue
        if not os.path.exists(os.path.join(RAW_DIR, f"{stem}.pkl")):
            print(f"  [no raw]   {f}: raw pkl missing, skipping")
            missing_raw += 1
            continue
        t_ret, t_raw = len(p["opt_wrist_pos"]), raw_frame_count(stem, side)
        if t_ret != t_raw:
            print(f"  [MISMATCH] {f}: opt={t_ret} raw={t_raw} -> NOT stamping (needs re-retarget)")
            skipped_mismatch += 1
            continue
        p["retarget_fps"] = NATIVE_FPS
        with open(path, "wb") as fh:
            pickle.dump(p, fh)
        stamped += 1
    print(f"[{suffix}] stamped={stamped}  already_had={skipped_present}  "
          f"mismatch={skipped_mismatch}  no_raw={missing_raw}")
