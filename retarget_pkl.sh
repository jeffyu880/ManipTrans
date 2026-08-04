#!/bin/bash
# Retarget the 0724 pen-capping captures for the Inspire hand.
# These captures are bimanual (both hands finite in every pkl), so each index is run
# for BOTH right and left, matching the original single-index version of this script.
#
# Index is the bare m_<id>; the mydataset loader resolves it to the unique pkl whose
# stem ends in those 6 digits (e.g. m_133607 -> cap_3_0724_m_133607.pkl).
#
# m_133607 is listed FIRST because its retarget needed updating; the other 14 had no
# retarget yet. Retargeted output lands in:
#   data/retargeting/my_dataset/mano2inspire/<stem>_{rh,lh}.pkl
#
# After both sides of an index are retargeted, playback_trajectory.py renders the retargeted
# trajectory to an mp4 for a quick visual sanity check (set RECORD=0 to skip):
#   data_stats/vis_traj_outputs/retarget_playback/<data_idx>_both.mp4
#
# Usage: bash retarget_pkl.sh          (RECORD=0 bash retarget_pkl.sh to skip the videos)
set -u
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

PYTHON="${PYTHON:-python}"
DEXHAND="${DEXHAND:-inspire}"
ITER="${ITER:-7000}"
RECORD="${RECORD:-1}"          # 1 = also record a playback video per index; 0 = skip
GID="${GID:-0}"                # GPU graphics device id for the off-screen recording camera

# IsaacGym's gym_38.so links libpython3.8.so.1.0 from the conda env's lib/; `conda activate`
# does not put it on LD_LIBRARY_PATH, so the import dies without this. Derive + prepend it
# (identical approach to playback_0721.sh).
py_bin=$(command -v "$PYTHON") || { echo "[ERROR] python not found: $PYTHON"; exit 1; }
py_lib="$(dirname "$(dirname "$(readlink -f "$py_bin")")")/lib"
[[ -e "$py_lib/libpython3.8.so.1.0" ]] && export LD_LIBRARY_PATH="$py_lib:${LD_LIBRARY_PATH:-}"

# All fifteen 0724 capture indices; m_133607 first (needed updating).
INDICES=(
    m_153706 m_153723 m_153738 m_153753 m_153810
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

    # Record the just-retargeted bimanual trajectory to an mp4 (off-screen GPU camera).
    if [[ "$RECORD" == "1" ]]; then
        echo "=== Recording $idx trajectory [$(date +%H:%M:%S)] ==="
        if ! "$PYTHON" data_stats/playback_trajectory.py \
                --data_idx "$idx" --side both --dexhand "$DEXHAND" --record --graphics_device_id "$GID"; then
            echo "[FAIL] $idx recording exited non-zero"
            fails+=("$idx/record")
        fi
    fi
done

echo "=== ALL DONE [$(date +%H:%M:%S)] ==="
if ((${#fails[@]})); then
    echo "[SUMMARY] ${#fails[@]} failed: ${fails[*]}"
else
    echo "[SUMMARY] all ${#INDICES[@]} indices retargeted (both hands)"
fi
