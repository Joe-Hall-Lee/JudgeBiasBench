#!/bin/bash
# 消融 Hinge w/ bias-aware data：Hinge + 仅偏见增强负样本。
# 用法：bash scripts/discriminative/hinge_bias_negatives.sh
EXPERIMENT=discriminative/hinge_bias_negatives
source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"
source "$REPO_ROOT/scripts/lib/discriminative.sh"

TRAIN_MODE=pairwise
DATA_FILE=unified_feedback_data_4biased_eval_no_value_1biased_biasedneg_only.jsonl
MAKE_SCRIPT=make_biasedneg_only.py
OUTPUT_PATH=${OUTPUT_PATH:-$CKPT_ROOT/rm_hinge_bias_negatives}
MODEL_NAME=${MODEL_NAME:-Qwen2.5-7B-Instruct-hinge-bias-negatives}

run_discriminative_pipeline
