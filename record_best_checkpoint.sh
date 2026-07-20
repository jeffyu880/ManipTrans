#!/bin/bash
# Record demos from the best (highest-reward) checkpoint of each MyDataset capping run.
#
# For every data index m_<id> there is one (or more) runs/capping_md_m_<id>__<date>/ folders.
# Per index, this finds the single .pth with the highest reward (parsed from the filename
# field _rew_<float>_) across ALL matching run folders, then records rollouts in test mode
# using the SAME data index the policy was trained on (capping_md_m_<id> -> dataIndices=[m_<id>]).
#
# Videos are written by train.py next to the checkpoint:
#     runs/capping_md_m_<id>__<date>/videos/m_<id>/
#
# Usage:
#   bash record_best_checkpoint.sh                 # record ALL discovered capping_md_m_* indices
#   bash record_best_checkpoint.sh m_170541 m_170401   # record only these indices
#   bash record_best_checkpoint.sh --list          # just print the best checkpoint per index, don't run
#
# Multi-demo mode (a single policy trained on several trajectories at once):
#   bash record_best_checkpoint.sh --run runs/reach_dist_1-5_AUG_noise__07-16-18-58-33
#       -> picks the best .pth in that run's nn/ and evaluates it on every dataIndices
#          entry from that run's config.yaml, ONE eval run per demo (default).
#   bash record_best_checkpoint.sh --run <dirA> <dirB> <dirC>
#       -> several runs back to back; best checkpoint auto-found in each.
#          Bare run names are resolved under runs/ automatically.
#   bash record_best_checkpoint.sh --combined --run <dir>
#       -> all of the run's demos in a single multi-demo eval instead of one per demo.
#   bash record_best_checkpoint.sh --run <dir> --checkpoint <path.pth>
#       -> use this exact checkpoint instead of the best one (single --run only).
#   bash record_best_checkpoint.sh --run <dir> --indices m_130824 m_130919
#       -> restrict evaluation to the given subset of indices.

shopt -s nullglob

RH_CKPT="assets/imitator_rh_inspire.pth"
LH_CKPT="assets/imitator_lh_inspire.pth"

LIST_ONLY=0
RUN_DIRS=()
IDX_OVERRIDE=()
CKPT_OVERRIDE=""
PER_INDEX=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --list)       LIST_ONLY=1; shift ;;
        --combined)   PER_INDEX=0; shift ;;
        --checkpoint) CKPT_OVERRIDE="$2"; shift 2 ;;
        # --run and --indices each swallow every following non-flag argument.
        --run|--indices)
            flag="$1"; shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                if [[ "$flag" == "--run" ]]; then
                    d="${1%/}"
                    # bare run names are resolved under runs/
                    [[ -d "$d" ]] || d="runs/$d"
                    RUN_DIRS+=("$d")
                else
                    IDX_OVERRIDE+=("$1")
                fi
                shift
            done ;;
        *)            break ;;
    esac
done

# Find the .pth with the highest _rew_<float>_ among all paths passed in.
find_best_checkpoint() {
    local best_ckpt="" best_rew=-9999999 rew
    for pth in "$@"; do
        [[ -f "$pth" ]] || continue
        rew=$(basename "$pth" | grep -oP '(?<=_rew_)[0-9.]+' | head -1)
        [[ -z "$rew" ]] && continue
        if awk "BEGIN{exit !($rew > $best_rew)}"; then
            best_rew="$rew"
            best_ckpt="$pth"
        fi
    done
    echo "$best_ckpt"
}

# Discover all m_<id> indices that have at least one capping_md run folder.
discover_indices() {
    for d in runs/capping_md_m_*/; do
        [[ -d "$d" ]] || continue
        basename "$d" | sed -E 's/^capping_md_(m_[0-9]+)__.*/\1/'
    done | sort -u
}

# Read the `dataIndices:` yaml list out of a run's config.yaml.
indices_from_config() {
    sed -n '/^dataIndices:/,/^[^ -]/p' "$1" | grep -oP '(?<=^- ).*'
}

# Evaluate one checkpoint on one or more data indices in a single test run.
run_eval() {
    local ckpt="$1"; shift
    local idx_csv
    idx_csv=$(IFS=,; echo "$*")
    python main/rl/train.py \
        task=ResDexHand \
        dexhand=inspire \
        side=BiH \
        headless=true \
        num_envs=$(( $# * 4 )) \
        learning_rate=2e-4 \
        test=true \
        randomStateInit=false \
        "dataIndices=[${idx_csv}]" \
        rh_base_model_checkpoint=${RH_CKPT} \
        lh_base_model_checkpoint=${LH_CKPT} \
        actionsMovingAverage=0.6 \
        num_rollouts_to_run=20 \
        capture_video=true \
        plot_trajectories=false \
        n_traj_episodes=10 \
        n_parallel_recorders=4 \
        jointNoiseCm=0.0 \
        causal=true \
        "checkpoint='${ckpt}'"   # single-quoted: run names may contain commas
}

# ---- Multi-demo mode: one policy, all trajectories it was trained on. ----
if (( ${#RUN_DIRS[@]} )); then
    if [[ -n "$CKPT_OVERRIDE" && ${#RUN_DIRS[@]} -gt 1 ]]; then
        echo "[ERROR] --checkpoint only makes sense with a single --run"; exit 1
    fi

    for run_dir in "${RUN_DIRS[@]}"; do
        cfg="$run_dir/config.yaml"
        if [[ ! -f "$cfg" ]]; then
            echo "[SKIP] $run_dir : no config.yaml"; continue
        fi

        if [[ -n "$CKPT_OVERRIDE" ]]; then
            best="$CKPT_OVERRIDE"
        else
            best=$(find_best_checkpoint "$run_dir"/nn/*.pth)
        fi
        if [[ ! -f "$best" ]]; then
            echo "[SKIP] $run_dir : no checkpoint with _rew_ under nn/"; continue
        fi

        if (( ${#IDX_OVERRIDE[@]} )); then
            INDICES=("${IDX_OVERRIDE[@]}")
        else
            mapfile -t INDICES < <(indices_from_config "$cfg")
        fi
        if (( ! ${#INDICES[@]} )); then
            echo "[SKIP] $run_dir : no dataIndices in config.yaml"; continue
        fi

        echo "========================================"
        echo "Run:        $run_dir"
        echo "Checkpoint: $best"
        echo "Indices (${#INDICES[@]}): ${INDICES[*]}"
        echo "========================================"
        [[ "$LIST_ONLY" == "1" ]] && continue

        if [[ "$PER_INDEX" == "1" ]]; then
            # one eval run per demo, so videos/metrics stay separated per trajectory
            for idx in "${INDICES[@]}"; do
                echo "--- $run_dir : $idx ---"
                run_eval "$best" "$idx"
            done
        else
            run_eval "$best" "${INDICES[@]}"
        fi
    done
    exit 0
fi

# Indices to record: CLI args if given, else all discovered.
if [[ $# -gt 0 ]]; then
    INDICES=("$@")
else
    mapfile -t INDICES < <(discover_indices)
fi

echo "Indices to process (${#INDICES[@]}): ${INDICES[*]}"
echo

for idx in "${INDICES[@]}"; do
    # Best checkpoint for this index across every matching run folder.
    best=$(find_best_checkpoint runs/capping_md_${idx}__*/nn/*.pth)

    if [[ -z "$best" ]]; then
        echo "[SKIP] $idx : no checkpoint with _rew_ found under runs/capping_md_${idx}__*/nn/"
        continue
    fi

    rew=$(basename "$best" | grep -oP '(?<=_rew_)[0-9.]+' | head -1)
    echo "========================================"
    echo "Index:      $idx"
    echo "Reward:     $rew"
    echo "Checkpoint: $best"
    echo "========================================"

    [[ "$LIST_ONLY" == "1" ]] && continue

    run_eval "$best" "$idx"
done
