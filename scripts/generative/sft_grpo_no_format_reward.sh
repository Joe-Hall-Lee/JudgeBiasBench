#!/bin/bash
# 消融 w/o format reward：与主实验相同，奖励函数换成 bias_no_format.py。SFT ckpt 与 sft_cot_grpo.sh 共用。
# 用法：bash scripts/generative/sft_grpo_no_format_reward.sh
EXPERIMENT=generative/sft_grpo_no_format_reward
source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"
source "$REPO_ROOT/scripts/lib/generative.sh"

SFT_CONFIG=train/generative/sft/qwen2_5_7b_sft_cot_gpt4o.yaml
SFT_OUTPUT=${SFT_OUTPUT:-$CKPT_ROOT/sft_cot_gpt4o}
GRPO_SAVE=${GRPO_SAVE:-$CKPT_ROOT/sft_grpo_no_format_reward}
REWARD_FN=$REPO_ROOT/train/generative/grpo/reward_function/bias_no_format.py
MODEL_NAME=${MODEL_NAME:-Qwen2.5-7B-Instruct-sft-grpo-no-format-reward-s$STEP}
EVAL_CONFIG=$REPO_ROOT/evaluate/configs/Qwen2.5-7B-Instruct-sft-grpo-no-format-reward.yaml

run_generative_pipeline
