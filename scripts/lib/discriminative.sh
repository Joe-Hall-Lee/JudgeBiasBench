#!/bin/bash
# 判别式 judge 流水线：准备训练数据 -> train/discriminative/run_training.sh -> eval_rm*.py。
# 由 scripts/discriminative/*.sh 设置变量后 source 并调用 run_discriminative_pipeline。
#
# 需设置：
#   TRAIN_MODE   contrast 即 InfoNCE，pairwise 即 Hinge
#   DATA_FILE    $JBB_TRAIN_DATA_DIR/discriminative/ 下的训练文件名
#   MAKE_SCRIPT  DATA_FILE 缺失时用 train/discriminative/data_prep/<MAKE_SCRIPT> 生成，可为空
#   OUTPUT_PATH  模型输出目录
#   MODEL_NAME   结果目录名 results/discriminative/<MODEL_NAME>/
# 超参由 run_training.sh 按 TRAIN_MODE 取默认值。训练每 500 步存状态并自动续训，已完成则只跑评测。

DATA_DIR="$JBB_TRAIN_DATA_DIR/discriminative"
RESULTS_DIR="$REPO_ROOT/results/discriminative"
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-32}

# OUTPUT_PATH 下有最终权重且无 resume_state 即视为训练已完成
train_already_done() {
    [ "${FORCE_RETRAIN:-0}" != "1" ] \
        && ls "$OUTPUT_PATH"/*.bin >/dev/null 2>&1 \
        && [ -f "$OUTPUT_PATH/config.json" ] \
        && [ ! -d "$OUTPUT_PATH/resume_state" ]
}

run_discriminative_pipeline() {
    : "${TRAIN_MODE:?}" "${DATA_FILE:?}" "${OUTPUT_PATH:?}" "${MODEL_NAME:?}"
    local data_path="$DATA_DIR/$DATA_FILE"
    use_env "$ENV_EVAL"
    cd "$REPO_ROOT"

    if [ "$STAGE" = "all" ] || [ "$STAGE" = "train" ]; then
        [ -d "$BASE_MODEL" ] || echo "[note] BASE_MODEL=$BASE_MODEL is not a local directory; it will be resolved as a HF hub id"
        if [ ! -f "$data_path" ]; then
            [ -n "$MAKE_SCRIPT" ] || { echo "Error: training data $data_path missing (see README.md, Training data)"; exit 1; }
            log "training data missing, generating with data_prep/$MAKE_SCRIPT"
            python train/discriminative/data_prep/"$MAKE_SCRIPT"
        fi
        if train_already_done; then
            log "final model already in $OUTPUT_PATH, skip training (FORCE_RETRAIN=1 to override)"
        else
            log "training $TRAIN_MODE discriminative judge on $DATA_FILE -> $OUTPUT_PATH"
            TRAIN_MODE="$TRAIN_MODE" MODEL_PATH="$BASE_MODEL" OUTPUT_PATH="$OUTPUT_PATH" DATA_PATH="$data_path" \
                bash train/discriminative/run_training.sh
        fi
    fi

    if [ "$STAGE" = "all" ] || [ "$STAGE" = "eval" ]; then
        # 已有 summary 的 bias 自动跳过，--overwrite 重跑
        log "JudgeBiasBench bias eval (Acc_ori / Acc_inj / BSR) -> $RESULTS_DIR/$MODEL_NAME"
        python evaluate/eval_rm.py all --model_path "$OUTPUT_PATH" --model_name "$MODEL_NAME" \
            --batch_size "$EVAL_BATCH_SIZE" --output_dir "$RESULTS_DIR"

        log "general benchmarks: RewardBench / JudgeBench / RM-Bench / RMB"
        local bench
        for bench in rewardbench judgebench rmbench rmb; do
            python evaluate/eval_rm_$bench.py all --model_path "$OUTPUT_PATH" --model_name "$MODEL_NAME" \
                --batch_size "$EVAL_BATCH_SIZE" --output_dir "$RESULTS_DIR"
        done
        log "results in $RESULTS_DIR/$MODEL_NAME/"
    fi
    leave_env
    log "$EXPERIMENT done"
}
