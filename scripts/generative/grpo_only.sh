#!/bin/bash
# 消融 GRPO only：不做 SFT 冷启动，从基座直接 GRPO -> 评测。
# 用法：bash scripts/generative/grpo_only.sh
EXPERIMENT=generative/grpo_only
source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"
source "$REPO_ROOT/scripts/lib/generative.sh"

SFT_CONFIG=""                       # 无 SFT，GRPO 从 BASE_MODEL 起跑
GRPO_SAVE=${GRPO_SAVE:-$CKPT_ROOT/grpo_only}
REWARD_FN=$REPO_ROOT/train/generative/grpo/reward_function/bias.py
MODEL_NAME=${MODEL_NAME:-Qwen2.5-7B-Instruct-grpo-only-s$STEP}
EVAL_CONFIG=$REPO_ROOT/evaluate/configs/Qwen2.5-7B-Instruct-grpo-only.yaml

run_generative_pipeline
