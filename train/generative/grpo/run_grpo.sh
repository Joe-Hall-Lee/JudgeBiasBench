#!/bin/bash
# GRPO training for the generative judge (EasyR1). Run from anywhere; cds into train/generative/EasyR1.
#
# Required:  MODEL_PATH   policy init (SFT ckpt for the main config; base Qwen2.5-7B-Instruct for GRPO-only)
#            SAVE_PATH    checkpoint dir (trainer.save_checkpoint_path; auto-resumes from the latest global_step_*)
# Optional:  EXP_NAME     trainer.experiment_name            (default: basename of SAVE_PATH)
#            REWARD_FN    reward function file               (default: reward_function/bias.py, format_weight=0.1;
#                                                             use reward_function/bias_no_format.py for format_weight=0)
#            DATA_DIR     EasyR1 parquet dir                 (default: data/train/generative/grpo/bias_0p25 = bias:orig 1:4)
#            N_GPU        trainer.n_gpus_per_node            (default: 8)
#            SAVE_FREQ / SAVE_LIMIT                          (default: 100 / 15 — frequent saves for preemptible jobs)
#            EXTRA_ARGS   appended verbatim to the command
#
# Example (main config):
#   MODEL_PATH=/path/to/qwen2.5-7b_sft_cot_gpt4o SAVE_PATH=/path/to/ckpts/grpo_main bash train/generative/grpo/run_grpo.sh

set -e
set -x
export PYTHONUNBUFFERED=1

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
EASYR1_DIR="$REPO_ROOT/train/generative/EasyR1"

MODEL_PATH=${MODEL_PATH:?set MODEL_PATH}
SAVE_PATH=${SAVE_PATH:?set SAVE_PATH}
EXP_NAME=${EXP_NAME:-$(basename "$SAVE_PATH")}
REWARD_FN=${REWARD_FN:-$HERE/reward_function/bias.py}
DATA_DIR=${DATA_DIR:-${JBB_TRAIN_DATA_DIR:-$REPO_ROOT/data/train}/generative/grpo/bias_0p25}
N_GPU=${N_GPU:-8}
SAVE_FREQ=${SAVE_FREQ:-100}
SAVE_LIMIT=${SAVE_LIMIT:-15}

[ -f "$EASYR1_DIR/examples/config.yaml" ] || { echo "EasyR1 not found at $EASYR1_DIR"; exit 1; }
[ -f "$DATA_DIR/train-orig.parquet" ] || { echo "GRPO data not found in $DATA_DIR (see README.md, Training data)"; exit 1; }

cd "$EASYR1_DIR"
python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files="$DATA_DIR@train" \
    data.val_files="$DATA_DIR@test" \
    data.format_prompt="$HERE/bias.jinja" \
    data.max_prompt_length=10240 \
    data.max_response_length=1024 \
    data.rollout_batch_size=32 \
    data.val_batch_size=128 \
    worker.actor.global_batch_size=8 \
    worker.actor.micro_batch_size_per_device_for_experience=8 \
    worker.actor.model.model_path="$MODEL_PATH" \
    worker.actor.optim.lr_warmup_ratio=0.05 \
    worker.actor.optim.lr_scheduler_type=cosine \
    worker.rollout.n=4 \
    worker.rollout.max_num_batched_tokens=20480 \
    worker.reward.reward_function="$REWARD_FN:compute_score" \
    trainer.total_epochs=1 \
    trainer.n_gpus_per_node="$N_GPU" \
    trainer.logger=[file] \
    trainer.experiment_name="$EXP_NAME" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.save_limit="$SAVE_LIMIT" \
    trainer.save_checkpoint_path="$SAVE_PATH" \
    ${EXTRA_ARGS:-}
