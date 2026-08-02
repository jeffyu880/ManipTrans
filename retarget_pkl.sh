#!/bin/bash
# Retarget the 0728 pen-capping captures for the Inspire hand.
# These captures are bimanual (both hands finite in every pkl), so each index is run
# for BOTH right and left, matching the original single-index version of this script.
#
# Index is the bare m_<id>; the mydataset loader resolves it to the unique pkl whose
# stem ends in those 6 digits (e.g. m_085551 -> cap_1_0728_m_085551.pkl).
#
# None of the 0728 captures had a retarget yet. Retargeted output lands in:
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

# All twenty 0728 capture indices (cap_1 x3, cap_2 x4, cap_3 x3, cap_4 x8, cap_5 x2).
INDICES=(
    m_191123
    m_191211
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
