#!/bin/bash
# Build the bias-augmented discriminative judge training set (data/train/discriminative/):
#   unified_feedback_data.jsonl (GRAM-fine-tuning-65K pairs) -> gptout.py (GPT-4o generates bias-augmented rejected)
#   -> verify_correctness.py (GPT-4o verifies chosen is still the best)
# The benchmark itself is built with filter_helpsteer3.py + process_bias.py, see README.md "Benchmark construction"
# Usage: API_KEY=... bash construct/run_construction.sh
# Optional: MODEL_NAME (default gpt-4o), DATA_PATH, LIMIT (debug: only first N rows)
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
DATA_PATH=${DATA_PATH:-${JBB_TRAIN_DATA_DIR:-$REPO_ROOT/data/train}/discriminative}
MODEL_NAME=${MODEL_NAME:-gpt-4o}
LIMIT_ARG=${LIMIT:+--limit $LIMIT}
: "${API_KEY:?export API_KEY (and optionally API_BASE_URL) first — never commit keys}"

echo "步骤1: 注入偏见回答（gptout.py，多线程调用 LLM，为每条 chosen 生成偏见增强的 rejected）..."
python3 -u "$HERE/gptout.py" \
    --input_file "$DATA_PATH/unified_feedback_data.jsonl" \
    --output_file "$DATA_PATH/unified_feedback_data_4biased.jsonl" \
    --model_name "$MODEL_NAME" $LIMIT_ARG

echo "步骤2: 验证 rejected 回答的正确性（verify_correctness.py，GPT-4o 判断 chosen 仍优于注入后的 rejected）..."
python3 "$HERE/verify_correctness.py" \
    --input_file "$DATA_PATH/unified_feedback_data_4biased.jsonl" \
    --output_file "$DATA_PATH/unified_feedback_data_4biased_eval.jsonl" \
    --model_name "$MODEL_NAME"
