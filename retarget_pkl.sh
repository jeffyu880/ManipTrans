#!/bin/bash
# Retarget the 0803 capping (alcohol-burner) captures for the Inspire hand.
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
# None of these had a retarget yet. Retargeted output lands in:
#   data/retargeting/my_dataset/mano2inspire_rh/<stem>_rh.pkl
#   data/retargeting/my_dataset/mano2inspire_lh/<stem>_lh.pkl
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

# All 15 capping captures from 0803: cap_1 x3, cap_2 x4, cap_3 x6, cap_5 x2. Verified before
# submitting that every one resolves to a unique pkl and to the `bottle` object set, both hands finite.
INDICES=(
    # cap_1_0803
    m_101613 m_101629 m_101644
    # cap_2_0803
    m_101705 m_101720 m_101752 m_101807
    # cap_3_0803
    m_101901 m_101919 m_101933 m_102009 m_102023 m_102056
    # cap_5_0803
    m_102126 m_102140
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
