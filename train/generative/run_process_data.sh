#!/bin/bash
# Build generative-judge training data from the discriminative jsonl (data/train/discriminative/) :
#   1) classify_data.py       split unified_feedback_data_4biased_eval*.jsonl per bias type -> data/train/generative/intermediate/
#   2) generate_cot.py        query a teacher model for <think> reasoning traces (needs API_KEY / API_BASE_URL)
#   3) prepare_train_data.py  keep verified traces -> data/train/generative/sft/train_cot_<teacher>.json
#   4) process_sft_data.py    no-CoT SFT labels;  process_grpo_data.py  EasyR1 parquet -> data/train/generative/grpo/
# Usage: API_KEY=... MODEL_NAME=gpt-4o bash train/generative/run_process_data.sh
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL_NAME=${MODEL_NAME:-gpt-4o}
: "${API_KEY:?export API_KEY (and optionally API_BASE_URL) first — never commit keys}"

echo "步骤1: 将训练数据按 bias 类型分类（classify_data.py）..."
python3 "$HERE/process_data/classify_data.py"
echo "步骤2: 获取教师模型 CoT 数据（generate_cot.py, model=$MODEL_NAME）..."
python3 -u "$HERE/process_data/generate_cot.py" --model_name "$MODEL_NAME"
echo "步骤3: 筛选 CoT 数据（prepare_train_data.py：只保留判断正确的推理链）-> SFT json..."
python3 -u "$HERE/process_data/prepare_train_data.py" --model_name "$MODEL_NAME"
echo "步骤4: 无 CoT 的 SFT json（process_sft_data.py）+ GRPO parquet（process_grpo_data.py）..."
python3 "$HERE/process_data/process_sft_data.py"
python3 "$HERE/process_data/process_grpo_data.py"
