#!/usr/bin/env bash
# Evaluate a discriminative judge on the JudgeBiasBench bias sets and, with WITH_BENCHMARKS=1,
# on RewardBench / JudgeBench / RM-Bench / RMB.
# Usage: MODEL_PATH=/path/to/judge bash scripts/run_evaluation_rm.sh
#   Optional: MODEL_NAME, WITH_BENCHMARKS=1, BATCH_SIZE, OUTPUT_DIR
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH}
MODEL_NAME=${MODEL_NAME:-$(basename "$MODEL_PATH")}
BATCH_SIZE=${BATCH_SIZE:-32}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_ROOT/results/discriminative}

cd "$REPO_ROOT"
python evaluate/eval_rm.py all --model_path "$MODEL_PATH" --model_name "$MODEL_NAME" \
    --batch_size "$BATCH_SIZE" --output_dir "$OUTPUT_DIR"
if [ "${WITH_BENCHMARKS:-0}" = "1" ]; then
    for bench in rewardbench judgebench rmbench rmb; do
        python evaluate/eval_rm_$bench.py all --model_path "$MODEL_PATH" --model_name "$MODEL_NAME" \
            --batch_size "$BATCH_SIZE" --output_dir "$OUTPUT_DIR"
    done
fi
