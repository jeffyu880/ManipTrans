#!/bin/bash
# Held-out validation for the 5-fold leave-one-fold-out CV trained by
# slurm/scitas/train_kfold_v2_array.run (v2 pool: data_stats/reach_demo_v2.csv, incl. 0724/0728).
#
# For each fold k: find the best checkpoint of runs/reach_5foldcv_v2_holdout<k>_seed0__*/, then
# evaluate it on each of fold k's 10 HELD-OUT demos (demos the policy never trained on) using the
# quantitative path -- train.py test=true save_rollouts=true -> eval_score.py (writes results.txt
# with succ rate + er/et/ej/eft). The held-out ids come from data_stats/make_kfold.py (seed=0) with
# --csv $POOL_CSV, the SAME generator+pool training used, so train/eval agree on the split.
#
# NOTE: this is the QUANTITATIVE path (like eval_capping.sh). record_best_checkpoint.sh only
# records videos and computes NO metrics -- do not use it for CV scoring.
#
# After scoring, run:  python data_stats/aggregate_kfold.py --csv data_stats/reach_demo_v2.csv \
#     --run-prefix reach_5foldcv_v2_holdout   (honest CV mean; counts held-out demos with zero
#     successes as succ_rate=0, unlike aggregate_results.py which drops them).
#
# Usage:
#   bash eval_kfold.sh                 # all 5 folds
#   bash eval_kfold.sh 0 3             # only folds 0 and 3
#   bash eval_kfold.sh --checkpoint <path> 0   # force a checkpoint for a single fold
#
# Two env vars, both defaulting to the original behavior:
#   RUN_GLOB  pins checkpoint selection to specific run dirs (see below).
#   DET_BASE  true -> deterministicBaseAction=true, i.e. the frozen imitators emit mu instead of
#             sampling Normal(mu, sigma) every step. Pin RUN_GLOB when A/B-ing this, or a newer
#             training run can change the checkpoint at the same time and confound the comparison:
#                 DET_BASE=true RUN_GLOB='07-25-16-42-*' bash eval_kfold.sh

set -u

RH_CKPT="assets/imitator_rh_inspire.pth"
LH_CKPT="assets/imitator_lh_inspire.pth"
SEED=0
POOL_CSV="${POOL_CSV:-data_stats/reach_demo_v2.csv}"   # v2 pool (52 demos incl. 0724/0728 captures)

# Object-size robustness knobs. objScale scales the object GEOMETRY the fingers close on while the
# mass is held at the unscaled value (dexhandmanip_bih.py:722-727), so e.g. OBJ_SCALE_RH=1.10 tests a
# 10%-bigger CAP on checkpoints trained at 1.0. DUMP_TAG, when set, moves each scored dump into
# dumps/<tag>/ so a scaled sweep doesn't collide with the baseline dumps during aggregation.
OBJ_SCALE_RH="${OBJ_SCALE_RH:-1.0}"   # cap  (RH object) geometry scale
OBJ_SCALE_LH="${OBJ_SCALE_LH:-1.0}"   # body (LH object) geometry scale
DUMP_TAG="${DUMP_TAG:-}"

# COMMON is built from scratch and never reads the scored run's config.yaml, so any knob it omits
# falls back to the main/cfg/config.yaml default -- which for an arm trained with the residual
# window (residualGateDistance et al), causal velocities or a scaled object means scoring the policy
# under a control structure it was never trained on. ACT_MA and EXTRA_OVERRIDES let an arm restate
# what it trained with; both default to the values this script has always used, so an eval that sets
# neither composes exactly the command it did before. Put only knobs COMMON does not already carry
# in EXTRA_OVERRIDES -- anything listed twice is a duplicate Hydra override.
ACT_MA="${ACT_MA:-0.4}"                 # actionsMovingAverage; training arms typically use 0.6
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"  # e.g. "causal=true residualGateDistance=0.03"

# Eval knobs mirror eval_capping.sh's COMMON block (the repo's established scoring setup).
COMMON="\
    task=ResDexHand \
    dexhand=inspire \
    side=BiH \
    headless=true \
    num_envs=256 \
    learning_rate=2e-4 \
    test=true \
    randomStateInit=false \
    rh_base_model_checkpoint=${RH_CKPT} \
    lh_base_model_checkpoint=${LH_CKPT} \
    actionsMovingAverage=${ACT_MA} \
    save_rollouts=true \
    num_rollouts_to_save=128 \
    num_rollouts_to_run=2000 \
    useTrajAug=false \
    RH_LH_Table_Center_Aug=true \
    RH_LObj_Center_Aug=true \
    RH_RObj_Center_Aug=true \
    LH_LObj_Center_Aug=true \
    jointNoiseCm=0.0 \
    save_successful_rollouts_only=false \
    zeroResidual=${ZERO_RESIDUAL:-false} \
    evalStartFrame=0 \
    deterministicBaseAction=${DET_BASE:-false} \
    objScaleRH=${OBJ_SCALE_RH} \
    objScaleLH=${OBJ_SCALE_LH} \
    ${EXTRA_OVERRIDES} \
    "

CKPT_OVERRIDE=""
FOLDS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint) CKPT_OVERRIDE="$2"; shift 2 ;;
        *)            FOLDS+=("$1"); shift ;;
    esac
done
[[ ${#FOLDS[@]} -eq 0 ]] && FOLDS=(0 1 2 3 4)

# Highest-reward checkpoint among the args (reward parsed from the _rew_<float>_ filename field).
# Same selection logic as record_best_checkpoint.sh:65-77.
find_best_checkpoint() {
    local best_ckpt="" best_rew=-9999999 rew
    for pth in "$@"; do
        [[ -f "$pth" ]] || continue
        rew=$(basename "$pth" | grep -oP '(?<=_rew_)[0-9.]+' | head -1)
        [[ -z "$rew" ]] && continue
        if awk "BEGIN{exit !($rew > $best_rew)}"; then
            best_rew="$rew"; best_ckpt="$pth"
        fi
    done
    echo "$best_ckpt"
}

for k in "${FOLDS[@]}"; do
    echo "########################################"
    echo "# FOLD ${k} held-out validation"
    echo "########################################"

    if [[ -n "$CKPT_OVERRIDE" ]]; then
        ckpt="$CKPT_OVERRIDE"
    else
        shopt -s nullglob
        # RUN_GLOB pins checkpoint selection to a specific run (e.g. '07-25-16-42-*' = the finished
        # job 3094924), so a concurrently-running retrain in another timestamped dir can't be scored
        # by accident. Defaults to '*' = all runs for this fold (original behavior).
        candidates=(runs/reach_5foldcv_v2_holdout${k}_seed0__${RUN_GLOB:-*}/nn/*.pth)
        shopt -u nullglob
        ckpt=$(find_best_checkpoint "${candidates[@]}")
    fi
    if [[ -z "$ckpt" ]]; then
        echo "  no checkpoint found for fold ${k} (runs/reach_5foldcv_v2_holdout${k}_seed0__*/nn/) -- SKIPPED"
        continue
    fi
    echo "  best checkpoint: $ckpt"

    heldout=$(python data_stats/make_kfold.py --csv "$POOL_CSV" --fold "$k" --which heldout --seed "$SEED")
    ckpt_stem=$(basename "$ckpt" .pth)

    for demo in ${heldout//,/ }; do
        echo "  ---- eval fold ${k} held-out demo ${demo} ----"
        python main/rl/train.py $COMMON \
            "dataIndices=[${demo}]" \
            experiment=eval_kfold_f${k}_${demo} \
            "checkpoint=${ckpt}"

        # Dump dir written by train.py test mode: dumps/dump__<ckpt_stem>__demo_<demo>__<ts>
        dump_dir=$(ls -td dumps/dump__${ckpt_stem}__demo_${demo}__* 2>/dev/null | head -1)
        if [[ -z "$dump_dir" ]]; then
            echo "    WARNING: no dump dir for ${demo} (0 saved rollouts?); counts as 0 success in aggregate."
            continue
        fi
        echo "    scoring $dump_dir"
        python main/rl/eval_score.py \
            --path "${dump_dir}/rollouts.hdf5" \
            --data_id "${demo}" \
            --dexhand inspire \
            --side bih

        # Isolate a scaled sweep's dumps so aggregate/matrix (newest-by-mtime) can't mix them with
        # the baseline. Moves after scoring, so stats.txt/results.txt travel with the dir.
        if [[ -n "$DUMP_TAG" ]]; then
            mkdir -p "dumps/${DUMP_TAG}"
            mv "$dump_dir" "dumps/${DUMP_TAG}/" \
                && echo "    moved -> dumps/${DUMP_TAG}/$(basename "$dump_dir")"
        fi
    done
done

echo "========================================"
echo "Done. Aggregate the CV metrics with:"
echo "    python data_stats/aggregate_kfold.py --csv data_stats/reach_demo_v2.csv --run-prefix reach_5foldcv_v2_holdout"
