#!/bin/bash
# 消融 InfoNCE w/o bias-aware data：InfoNCE + 仅原始负样本。
# 用法：bash scripts/discriminative/contrast_original_negatives.sh
EXPERIMENT=discriminative/contrast_original_negatives
source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"
source "$REPO_ROOT/scripts/lib/discriminative.sh"

TRAIN_MODE=contrast
DATA_FILE=unified_feedback_data_4biased_eval_no_value_1biased_origneg_only.jsonl
MAKE_SCRIPT=make_origneg_only.py
OUTPUT_PATH=${OUTPUT_PATH:-$CKPT_ROOT/rm_contrast_original_negatives}
MODEL_NAME=${MODEL_NAME:-Qwen2.5-7B-Instruct-contrast-original-negatives}

run_discriminative_pipeline
