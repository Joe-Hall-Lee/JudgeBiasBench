#!/usr/bin/env bash
# Evaluate a generative judge on the 12 JudgeBiasBench bias sets with vLLM.
# Usage: MODEL_PATH=/path/to/model MODEL_NAME=qwen2.5-7b MODEL_TYPE=default bash scripts/run_evaluation_judge.sh
#   MODEL_TYPE: default | think | direct | judgelm | auto-j | selene | prometheus
#   Optional: GPU_NUM, MAX_NEW_TOKEN, ENABLE_THINKING=1, OUTPUT_DIR
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH}
MODEL_NAME=${MODEL_NAME:-$(basename "$MODEL_PATH")}
MODEL_TYPE=${MODEL_TYPE:-default}
GPU_NUM=${GPU_NUM:-1}
MAX_NEW_TOKEN=${MAX_NEW_TOKEN:-2048}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_ROOT/results/baselines}
THINK_FLAG=""; [ "${ENABLE_THINKING:-0}" = "1" ] && THINK_FLAG="--enable_thinking"

cd "$REPO_ROOT"
python evaluate/eval_judge.py all \
  --model_path "$MODEL_PATH" \
  --model_name "$MODEL_NAME" \
  --model_type "$MODEL_TYPE" \
  --temperature 0.0 \
  --max_new_token "$MAX_NEW_TOKEN" \
  --tensor_parallel_size "$GPU_NUM" \
  --gpu_memory_utilization 0.9 \
  --output_dir "$OUTPUT_DIR" \
  $THINK_FLAG
