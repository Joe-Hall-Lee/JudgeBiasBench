#!/bin/bash
# 消融 Hinge w/ orig + bias negatives：Hinge + 原始负样本 + 偏见增强负样本，每行拆成两对。
# 用法：bash scripts/discriminative/hinge_both_negatives.sh
EXPERIMENT=discriminative/hinge_both_negatives
source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"
source "$REPO_ROOT/scripts/lib/discriminative.sh"

TRAIN_MODE=pairwise
DATA_FILE=unified_feedback_data_4biased_eval_no_value_1biased_bothneg_pairs.jsonl
MAKE_SCRIPT=make_bothneg_pairs.py
OUTPUT_PATH=${OUTPUT_PATH:-$CKPT_ROOT/rm_hinge_both_negatives}
MODEL_NAME=${MODEL_NAME:-Qwen2.5-7B-Instruct-hinge-both-negatives}

run_discriminative_pipeline
