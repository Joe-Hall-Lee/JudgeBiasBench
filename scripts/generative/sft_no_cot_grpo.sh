#!/bin/bash
# 消融 SFT w/o teacher reasoning：SFT 目标只保留 [[A]]/[[B]]，不含 CoT -> GRPO -> 评测。
# 用法：bash scripts/generative/sft_no_cot_grpo.sh
EXPERIMENT=generative/sft_no_cot_grpo
source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"
source "$REPO_ROOT/scripts/lib/generative.sh"

SFT_CONFIG=train/generative/sft/qwen2_5_7b_sft_no_cot.yaml
SFT_OUTPUT=${SFT_OUTPUT:-$CKPT_ROOT/sft_no_cot}
GRPO_SAVE=${GRPO_SAVE:-$CKPT_ROOT/sft_no_cot_grpo}
REWARD_FN=$REPO_ROOT/train/generative/grpo/reward_function/bias.py
MODEL_NAME=${MODEL_NAME:-Qwen2.5-7B-Instruct-sft-no-cot-grpo-s$STEP}
EVAL_CONFIG=$REPO_ROOT/evaluate/configs/Qwen2.5-7B-Instruct-sft-no-cot-grpo.yaml

run_generative_pipeline
