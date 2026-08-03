#!/bin/bash
# Retarget the 0803 cup+brush captures for the Inspire hand.
# These captures are bimanual (both hands finite in every pkl), so each index is run
# for BOTH right and left, matching the original single-index version of this script.
#
# Index is the bare m_<id>; the mydataset loader resolves it to the unique pkl whose
# stem ends in those 6 digits (e.g. m_095523 -> brush_cap_0803_m_095523.pkl).
#
# All of these track ['d2_cup', 'd2_brush'], so mano2dexhand -> ManipDataFactory -> the mydataset
# loader infers the `cup_brush` object set (cup as the recentering anchor, recenter_fine=(0,0,0),
# the brush scored by BOTH hands). Nothing to pass on the command line -- the set comes from what
# the capture recorded, and `objectSet` is live-only. See main/dataset/object_sets.py.
#
# None of these had a retarget yet. Retargeted output lands in:
#   data/retargeting/my_dataset/mano2inspire_rh/<stem>_rh.pkl
#   data/retargeting/my_dataset/mano2inspire_lh/<stem>_lh.pkl
#
# Usage: bash retarget_pkl.sh
set -u
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

PYTHON="${PYTHON:-python}"
DEXHAND="${DEXHAND:-inspire}"
ITER="${ITER:-7000}"

# IsaacGym's gym_38.so links libpython3.8.so.1.0 from the conda env's lib/; `conda activate`
# does not put it on LD_LIBRARY_PATH, so the import dies without this. Derive + prepend it
# (identical approach to playback_0721.sh).
py_bin=$(command -v "$PYTHON") || { echo "[ERROR] python not found: $PYTHON"; exit 1; }
py_lib="$(dirname "$(dirname "$(readlink -f "$py_bin")")")/lib"
[[ -e "$py_lib/libpython3.8.so.1.0" ]] && export LD_LIBRARY_PATH="$py_lib:${LD_LIBRARY_PATH:-}"

# All 21 cup+brush captures from 0803: brush_cap x15, bc_cmplx x6. Verified before submitting that
# every one resolves to a unique pkl and to the cup_brush object set, with both hands finite.
INDICES=(
    # brush_cap_0803
    m_095523 m_095549 m_095610 m_095647 m_095702
    m_095718 m_095733 m_095749 m_095817 m_095834
    m_095900 m_095916 m_095928 m_095943 m_095959
    # bc_cmplx_0803
    m_101123 m_101140 m_101158 m_101215 m_101231 m_101249
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
done

echo "=== ALL DONE [$(date +%H:%M:%S)] ==="
if ((${#fails[@]})); then
    echo "[SUMMARY] ${#fails[@]} failed: ${fails[*]}"
else
    echo "[SUMMARY] all ${#INDICES[@]} indices retargeted (both hands)"
fi
