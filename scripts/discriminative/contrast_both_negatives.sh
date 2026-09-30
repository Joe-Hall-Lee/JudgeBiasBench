#!/bin/bash
# 主实验，判别式 judge：InfoNCE + 原始负样本 + 偏见增强负样本。
# 用法：bash scripts/discriminative/contrast_both_negatives.sh
# 可覆盖：BASE_MODEL CKPT_ROOT STAGE N_GPU CUDA_VISIBLE_DEVICES FORCE_RETRAIN（lib/common.sh）；DS_CONFIG（train/discriminative/run_training.sh）
EXPERIMENT=discriminative/contrast_both_negatives
source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"
source "$REPO_ROOT/scripts/lib/discriminative.sh"

TRAIN_MODE=contrast
DATA_FILE=unified_feedback_data_4biased_eval_no_value_1biased.jsonl
MAKE_SCRIPT=""                      # 主数据，见 README.md Training data
OUTPUT_PATH=${OUTPUT_PATH:-$CKPT_ROOT/rm_contrast_both_negatives}
MODEL_NAME=${MODEL_NAME:-Qwen2.5-7B-Instruct-contrast-both-negatives}

run_discriminative_pipeline
