#!/bin/bash
# Discriminative judge training (Accelerate + DeepSpeed ZeRO-3).
#
#   TRAIN_MODE=contrast  -> InfoNCE over chosen vs. all rejected (paper: bias-aware discriminative judge; rejected[1] = bias-augmented negative)
#   TRAIN_MODE=pairwise  -> Hinge/margin loss over chosen vs. rejected[0] only
#
# Env overrides: MODEL_PATH, OUTPUT_PATH, DATA_PATH, TRAIN_MODE, N_GPU, MAIN_PORT, DS_CONFIG, TEMPERATURE, MARGIN,
#                BATCH_SIZE, GRAD_ACCUM, EPOCHS, LR, STATE_SAVE_STEPS, EXTRA_ARGS.
# Resumable: --state_save_steps writes optimizer/scheduler/dataloader state under OUTPUT_PATH/resume_state every
# N steps and --auto_resume picks it up, so re-running the same command after preemption continues training.

set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

TRAIN_MODE=${TRAIN_MODE:-contrast}
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH (e.g. /path/to/Qwen2.5-7B-Instruct)}
OUTPUT_PATH=${OUTPUT_PATH:?set OUTPUT_PATH}
DATA_PATH=${DATA_PATH:-${JBB_TRAIN_DATA_DIR:-$REPO_ROOT/data/train}/discriminative/unified_feedback_data_4biased_eval_no_value_1biased.jsonl}
DS_CONFIG=${DS_CONFIG:-$HERE/configs/ds_z3_config.json}
N_GPU=${N_GPU:-8}
TEMPERATURE=${TEMPERATURE:-0.5}
MARGIN=${MARGIN:-1.0}
EPOCHS=${EPOCHS:-2}
LR=${LR:-5e-6}
STATE_SAVE_STEPS=${STATE_SAVE_STEPS:-500}
if [ "$TRAIN_MODE" = "contrast" ]; then
    BATCH_SIZE=${BATCH_SIZE:-1}; GRAD_ACCUM=${GRAD_ACCUM:-8}
    MODE_ARGS="--temperature $TEMPERATURE"
else
    BATCH_SIZE=${BATCH_SIZE:-4}; GRAD_ACCUM=${GRAD_ACCUM:-2}
    MODE_ARGS="--margin $MARGIN"
fi

[ -d "$MODEL_PATH" ] || echo "[note] MODEL_PATH=$MODEL_PATH is not a local directory; it will be resolved as a HF hub id"
[ -f "$DATA_PATH" ] || { echo "Error: training data $DATA_PATH not found (see README.md, Training data)"; exit 1; }
mkdir -p "$OUTPUT_PATH"

echo "=== discriminative judge training: mode=$TRAIN_MODE model=$MODEL_PATH data=$DATA_PATH -> $OUTPUT_PATH ==="
accelerate launch \
    --num_processes "$N_GPU" \
    --use_deepspeed \
    --deepspeed_config_file "$DS_CONFIG" \
    --main_process_port "${MAIN_PORT:-19500}" \
    "$HERE/train_reward_model.py" \
    --model_name_or_path "$MODEL_PATH" \
    --train_data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_PATH" \
    --training_mode "$TRAIN_MODE" \
    $MODE_ARGS \
    --num_epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate "$LR" \
    --max_length 1024 \
    --warmup_steps 100 \
    --eval_steps 500 \
    --save_steps 0 \
    --max_grad_norm 1.0 \
    --weight_decay 0.01 \
    --seed 42 \
    --state_save_steps "$STATE_SAVE_STEPS" \
    --auto_resume \
    --bf16 \
    ${EXTRA_ARGS:-}

echo "Training completed! Model saved to: $OUTPUT_PATH"
