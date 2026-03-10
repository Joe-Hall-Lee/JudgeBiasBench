#!/usr/bin/env bash

# ===============================
# Config
# ===============================
MODEL_PATH="/root/autodl-tmp/RewardBiasBench/models/Qwen2.5-7B-Instruct"
MODEL_NAME="qwen2.5-7b"
MODEL_TYPE="chateval"        # default | cot | reference | rules | metrics | chateval | judgelm | auto-j | selene | prometheus
GPU_NUM=1

# ???? thinking?Qwen3 ??
# ENABLE_THINKING="--enable_thinking"
ENABLE_THINKING=""

# ===============================
# Run all biases
# ===============================
CUDA_VISIBLE_DEVICES=0 \
python src/evaluate/eval_judge.py all \
  --model_path ${MODEL_PATH} \
  --model_name ${MODEL_NAME} \
  --model_type ${MODEL_TYPE} \
  --temperature 0.0 \
  --max_new_token 2048 \
  --tensor_parallel_size ${GPU_NUM} \
  --gpu_memory_utilization 0.9 \
  ${ENABLE_THINKING}
