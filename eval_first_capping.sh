#!/bin/bash
set -e

BASE="runs/first_capping_train"
RH_CKPT="assets/imitator_rh_inspire.pth"
LH_CKPT="assets/imitator_lh_inspire.pth"
COMMON="task=ResDexHand dexhand=inspire side=BiH headless=true num_envs=256 learning_rate=2e-4 test=true randomStateInit=false rh_base_model_checkpoint=${RH_CKPT} lh_base_model_checkpoint=${LH_CKPT} actionsMovingAverage=0.6 save_rollouts=true num_rollouts_to_save=128 num_rollouts_to_run=700 save_successful_rollouts_only=false"

run_eval() {
    local demo=$1
    local ckpt=$2
    echo "========================================"
    echo "Evaluating $demo"
    echo "Checkpoint: $ckpt"
    echo "========================================"
    python main/rl/train.py $COMMON \
        "dataIndices=[${demo}]" \
        "checkpoint=${ckpt}"
}

run_eval "3b1e6@12" "${BASE}/capping_alcohol_burner_3b1e6@12_large_envs__05-28-20-17-12/nn/last_capping_alcohol_burner_3b1e6@12_large_envs_ep_1100_rew_6066.9966_sr_0.9943330883979797_fr_0.00566680496558547.pth"

run_eval "d6fe3@0" "${BASE}/capping_alcohol_burner_d6fe3@0_large_envs__05-28-22-02-15/nn/last_capping_alcohol_burner_d6fe3@0_large_envs_ep_1400_rew_2384.084_sr_0.5322580337524414_fr_0.46774187684059143.pth"

run_eval "8e5df@13" "${BASE}/capping_alcohol_burner_8e5df@13__05-28-19-54-33/nn/last_capping_alcohol_burner_8e5df@13_ep_900_rew_2385.949_sr_0.46484556794166565_fr_0.535154402256012.pth"

run_eval "a78a0@1" "${BASE}/capping_alcohol_burner_a78a0@1__05-28-18-32-02/nn/last_capping_alcohol_burner_a78a0@1_ep_800_rew_3913.1843_sr_0.7978141903877258_fr_0.20218585431575775.pth"

run_eval "0f900@10" "${BASE}/capping_alcohol_burner_0f900@10_large_envs__05-28-22-16-41/nn/last_capping_alcohol_burner_0f900@10_large_envs_ep_800_rew_4987.5366_sr_0.9603081345558167_fr_0.03969183191657066.pth"

run_eval "b5fa3@10" "${BASE}/capping_alcohol_burner_b5fa3@10_large_envs__05-28-23-00-17/nn/last_capping_alcohol_burner_b5fa3@10_large_envs_ep_800_rew_4486.405_sr_0.9786320328712463_fr_0.021367954090237617.pth"

run_eval "f7d37@18" "${BASE}/capping_alcohol_burner_f7d37@18__05-28-19-51-54/nn/last_capping_alcohol_burner_f7d37@18_ep_900_rew_3607.6824_sr_0.9324186444282532_fr_0.06758135557174683.pth"

run_eval "85abe@4" "${BASE}/capping_alcohol_burner_85abe@4__05-28-19-51-37/nn/last_capping_alcohol_burner_85abe@4_ep_1100_rew_3466.8496_sr_0.9743828177452087_fr_0.02561710774898529.pth"

run_eval "b0b13@11" "${BASE}/capping_alcohol_burner_b0b13@11__05-28-19-49-28/nn/last_capping_alcohol_burner_b0b13@11_ep_1100_rew_2300.813_sr_0.8725490570068359_fr_0.12745098769664764.pth"

run_eval "751fb@16" "${BASE}/capping_alcohol_burner_751fb@16__05-28-18-31-55/nn/last_capping_alcohol_burner_751fb@16_ep_900_rew_2311.4792_sr_1.0_fr_0.0.pth"
