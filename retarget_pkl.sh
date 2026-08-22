#!/bin/bash
# Retarget the b_c_4 0822 capping (alcohol-burner) captures for the Inspire hand.
# These captures are bimanual (both hands finite in every pkl), so each index is run
# for BOTH right and left, matching the original single-index version of this script.
#
# Index is the bare m_<id>; the mydataset loader resolves it to the unique pkl whose
# stem ends in those 6 digits (e.g. m_101613 -> cap_1_0803_m_101613.pkl).
#
# All of these track ['bottle_body', 'bottle_cap'], so mano2dexhand -> ManipDataFactory -> the
# mydataset loader infers the default `bottle` object set (LH holds the bottle_body, RH brings the
# bottle_cap down onto it; bottle_body is the recentering anchor, both bodies scored). Nothing to
# pass on the command line -- the set comes from what the capture recorded, and `objectSet` is
# live-only. See main/dataset/object_sets.py.
#
# Only m_142506 had a retarget already (it is redone here). Retargeted output lands in:
#   data/retargeting/my_dataset/mano2inspire_rh/<stem>_rh.pkl
#   data/retargeting/my_dataset/mano2inspire_lh/<stem>_lh.pkl
#
# After both sides of an index are retargeted, playback_trajectory.py renders the retargeted
# trajectory to an mp4 for a quick visual sanity check (set RECORD=0 to skip), once per camera
# view in $VIEWS. The views are the shared record_cameras.py poses the BiH env records through,
# so a playback and a capture_video run of the same demo can be put side by side:
#   data_stats/vis_traj_outputs/retarget_playback/<data_idx>_both_<view>.mp4
#
# Usage: bash retarget_pkl.sh                       (RECORD=0 ... to skip the videos)
#        VIEWS="front behind overhead" bash retarget_pkl.sh   (pick the camera poses)
set -u
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

PYTHON="${PYTHON:-python}"
DEXHAND="${DEXHAND:-inspire}"
ITER="${ITER:-7000}"
RECORD="${RECORD:-1}"          # 1 = also record a playback video per index; 0 = skip
GID="${GID:-0}"                # GPU graphics device id for the off-screen recording camera
VIEWS="${VIEWS:-front overhead}"   # camera poses to record, space separated: front | behind | overhead
OUT_DIR="data_stats/vis_traj_outputs/retarget_playback"

# IsaacGym's gym_38.so links libpython3.8.so.1.0 from the conda env's lib/; `conda activate`
# does not put it on LD_LIBRARY_PATH, so the import dies without this. Derive + prepend it
# (identical approach to playback_0721.sh).
py_bin=$(command -v "$PYTHON") || { echo "[ERROR] python not found: $PYTHON"; exit 1; }
py_lib="$(dirname "$(dirname "$(readlink -f "$py_bin")")")/lib"
[[ -e "$py_lib/libpython3.8.so.1.0" ]] && export LD_LIBRARY_PATH="$py_lib:${LD_LIBRARY_PATH:-}"

# All 11 b_c_4 captures from 0822. Verified before submitting that every 6-digit id resolves to
# exactly one pkl under data/my_dataset (no suffix collisions).
INDICES=(
    # b_c_4_0822
    m_142506
    m_144612
    m_144630
    m_144709
    m_144729
    m_144748
    m_144812
    m_144831
    m_144852
    m_144914
    m_144933
)

fails=()
for idx in "${INDICES[@]}"; do
    for side in right left; do
        echo "=== Retargeting $idx ($side) [$(date +%H:%M:%S)] ==="
        if ! "$PYTHON" main/dataset/mano2dexhand.py \
                --data_idx "$idx" --side "$side" --dexhand "$DEXHAND" --headless --iter "$ITER"; then
            echo "[FAIL] $idx ($side) exited non-zero"
            fails+=("$idx/$side")
        fi
    done

    # Record the just-retargeted bimanual trajectory to an mp4 (off-screen GPU camera), one pass
    # per view. --record_path is explicit because playback's default name carries no view, so
    # several views would otherwise overwrite each other.
    if [[ "$RECORD" == "1" ]]; then
        for view in $VIEWS; do
            echo "=== Recording $idx trajectory ($view) [$(date +%H:%M:%S)] ==="
            if ! "$PYTHON" data_stats/playback_trajectory.py \
                    --data_idx "$idx" --side both --dexhand "$DEXHAND" --record --view "$view" \
                    --record_path "$OUT_DIR/${idx}_both_${view}.mp4" --graphics_device_id "$GID"; then
                echo "[FAIL] $idx recording ($view) exited non-zero"
                fails+=("$idx/record/$view")
            fi
        done
    fi
done

echo "=== ALL DONE [$(date +%H:%M:%S)] ==="
if ((${#fails[@]})); then
    echo "[SUMMARY] ${#fails[@]} failed: ${fails[*]}"
else
    echo "[SUMMARY] all ${#INDICES[@]} indices retargeted (both hands)"
fi
