#!/bin/bash
# 主实验，生成式 judge：SFT 冷启动，teacher 为 GPT-4o CoT -> GRPO -> 评测。
# 用法：bash scripts/generative/sft_cot_grpo.sh
# 可覆盖：BASE_MODEL CKPT_ROOT STAGE N_GPU CUDA_VISIBLE_DEVICES FORCE_RETRAIN（lib/common.sh）；STEP GRPO_DATA_DIR GRPO_EXTRA_ARGS（lib/generative.sh）
EXPERIMENT=generative/sft_cot_grpo
source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"
source "$REPO_ROOT/scripts/lib/generative.sh"

SFT_CONFIG=train/generative/sft/qwen2_5_7b_sft_cot_gpt4o.yaml
SFT_OUTPUT=${SFT_OUTPUT:-$CKPT_ROOT/sft_cot_gpt4o}
GRPO_SAVE=${GRPO_SAVE:-$CKPT_ROOT/sft_cot_grpo}
REWARD_FN=$REPO_ROOT/train/generative/grpo/reward_function/bias.py
MODEL_NAME=${MODEL_NAME:-Qwen2.5-7B-Instruct-sft-grpo-s$STEP}
EVAL_CONFIG=$REPO_ROOT/evaluate/configs/Qwen2.5-7B-Instruct-sft-grpo.yaml

run_generative_pipeline
