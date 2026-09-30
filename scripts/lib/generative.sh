#!/bin/bash
# 生成式 judge 流水线：SFT -> GRPO -> 合并 ckpt -> JudgeBiasBench 评测 -> 通用基准。
# 由 scripts/generative/*.sh 设置变量后 source 并调用 run_generative_pipeline。
#
# 需设置：
#   SFT_CONFIG   train/generative/sft/*.yaml，为空则不做 SFT
#   SFT_OUTPUT   SFT 输出目录
#   GRPO_SAVE    GRPO checkpoint 目录
#   REWARD_FN    train/generative/grpo/reward_function/*.py
#   MODEL_NAME   结果目录名 results/generative/<MODEL_NAME>/
#   EVAL_CONFIG  evaluate/configs/*.yaml
# 可选：STEP 评测的 global_step，默认 1800；GRPO_DATA_DIR；GRPO_EXTRA_ARGS 追加给 EasyR1 的参数

STEP=${STEP:-1800}
RESULTS_DIR="$REPO_ROOT/results/generative"

# ---- SFT，LLaMA-Factory。yaml 内路径相对仓库根，须在仓库根执行。
run_sft() {
    cd "$REPO_ROOT"
    [ -f "$SFT_CONFIG" ] || { echo "Error: SFT config $SFT_CONFIG not found"; exit 1; }
    local dataset
    dataset=$(awk '/^dataset:/ {print $2}' "$SFT_CONFIG")
    python3 - "$dataset" "$JBB_TRAIN_DATA_DIR/generative/sft" <<'PY'
import json, os, sys
root = sys.argv[2]
info = json.load(open(os.path.join(root, "dataset_info.json")))
f = os.path.join(root, info[sys.argv[1]]["file_name"])
if not os.path.isfile(f):
    sys.exit(f"Error: SFT data missing: {f} (see README.md, Training data)")
PY
    # output_dir 根目录已有 safetensors 说明 SFT 已完成，跳过；中断时 LLaMA-Factory 自动从最新 checkpoint 续训
    if [ "${FORCE_RETRAIN:-0}" != "1" ] && ls "$SFT_OUTPUT"/*.safetensors >/dev/null 2>&1; then
        log "SFT already finished ($SFT_OUTPUT has final weights), skip (FORCE_RETRAIN=1 to override)"
        return 0
    fi
    use_env "$ENV_SFT"
    log "SFT: $SFT_CONFIG  model=$BASE_MODEL -> $SFT_OUTPUT"
    # 命令行覆盖 yaml 中的 model_name_or_path / output_dir
    FORCE_TORCHRUN=1 llamafactory-cli train "$SFT_CONFIG" \
        model_name_or_path="$BASE_MODEL" output_dir="$SFT_OUTPUT"
    leave_env
}

# ---- GRPO，EasyR1。重跑时自动从 GRPO_SAVE 最新 global_step 续训。
run_grpo() {
    local init_model
    if [ -n "$SFT_CONFIG" ]; then
        init_model="$SFT_OUTPUT"
        # SFT 未完成则不起 GRPO
        ls "$init_model"/*.safetensors >/dev/null 2>&1 \
            || { echo "Error: SFT ckpt $init_model has no final weights yet"; exit 1; }
    else
        init_model="$BASE_MODEL"     # GRPO only
    fi
    use_env "$ENV_GRPO"
    log "GRPO: init=$init_model reward=$(basename "$REWARD_FN") -> $GRPO_SAVE"
    MODEL_PATH="$init_model" SAVE_PATH="$GRPO_SAVE" EXP_NAME="$MODEL_NAME" REWARD_FN="$REWARD_FN" \
        DATA_DIR="${GRPO_DATA_DIR:-$JBB_TRAIN_DATA_DIR/generative/grpo/bias_0p25}" \
        EXTRA_ARGS="${GRPO_EXTRA_ARGS:-}" \
        bash "$REPO_ROOT/train/generative/grpo/run_grpo.sh"
    leave_env
}

# ---- 评测：合并 FSDP 分片 -> eval_judge.py -> eval_general_benchmarks.py。已有结果自动跳过。
run_eval_generative() {
    local actor_dir="$GRPO_SAVE/global_step_${STEP}/actor"
    local hf_dir="$actor_dir/huggingface"
    [ -d "$actor_dir" ] || { echo "Error: checkpoint $actor_dir not found; check STEP or ls $GRPO_SAVE"; exit 1; }

    if ! ls "$hf_dir"/*.safetensors >/dev/null 2>&1; then
        log "merging FSDP shards -> $hf_dir"
        use_env "$ENV_GRPO"
        python "$REPO_ROOT/train/generative/EasyR1/scripts/model_merger.py" --local_dir "$actor_dir"
        leave_env
    else
        log "merged weights already exist, skip merging"
    fi

    use_env "$ENV_EVAL"
    cd "$REPO_ROOT"
    # --model_type think：训练时使用的 <think> + [[A]]/[[B]] 提示格式
    log "JudgeBiasBench bias eval (Acc_ori / Acc_inj / BSR) -> $RESULTS_DIR/$MODEL_NAME"
    python evaluate/eval_judge.py all \
        --model_path "$hf_dir" \
        --model_name "$MODEL_NAME" \
        --model_type think \
        --temperature 0.0 \
        --max_new_token 1024 \
        --output_dir "$RESULTS_DIR"

    log "general benchmarks: RewardBench / JudgeBench / RM-Bench / RMB"
    python evaluate/eval_general_benchmarks.py \
        --config "$EVAL_CONFIG" \
        --model_path "$hf_dir" \
        --name "$MODEL_NAME" \
        --benchmarks rewardbench,judgebench,rm-bench,rmb_pairwise \
        --output_dir "$RESULTS_DIR"
    leave_env
    log "results in $RESULTS_DIR/$MODEL_NAME/"
}

run_generative_pipeline() {
    : "${GRPO_SAVE:?}" "${REWARD_FN:?}" "${MODEL_NAME:?}" "${EVAL_CONFIG:?}"
    [ -d "$BASE_MODEL" ] || echo "[note] BASE_MODEL=$BASE_MODEL is not a local directory; it will be resolved as a HF hub id"
    if [ "$STAGE" = "all" ] || [ "$STAGE" = "train" ]; then
        [ -n "$SFT_CONFIG" ] && run_sft
        run_grpo
    fi
    if [ "$STAGE" = "all" ] || [ "$STAGE" = "eval" ]; then
        run_eval_generative
    fi
    log "$EXPERIMENT done"
}
