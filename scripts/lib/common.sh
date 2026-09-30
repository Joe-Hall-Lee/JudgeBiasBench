#!/bin/bash
# scripts/ 下实验脚本共享的初始化。每个实验脚本是一条完整 job：切换环境 -> 训练 -> 评测，
# 重跑同一条命令即可续跑。
#
# 可覆盖变量：
#   BASE_MODEL   基座模型，本地目录或 HF hub id
#   CKPT_ROOT    checkpoint 根目录，默认 <repo>/checkpoints
#   STAGE        all|train|eval
#   N_GPU / CUDA_VISIBLE_DEVICES   只设其一时另一个随之推导
#   JBB_TRAIN_DATA_DIR   训练数据根目录，默认 <repo>/data/train
#   SETUP_ENV    1 按阶段切换 conda env；0 表示已在正确 env 内
#   ENV_EVAL / ENV_SFT / ENV_GRPO   三个 conda env 名，见 README Setup
#   FORCE_RETRAIN=1  强制重训
#   EVAL_BATCH_SIZE  判别式 judge 评测 batch，默认 32

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [ -f "$REPO_ROOT/scripts/env_lib.sh" ]; then
    source "$REPO_ROOT/scripts/env_lib.sh"
else
    ensure_conda() {
        if command -v conda >/dev/null 2>&1; then
            source "$(conda info --base)/etc/profile.d/conda.sh"
        else
            echo "Error: conda not found; create the envs described in README.md Setup first, or run with SETUP_ENV=0 inside an activated env"; exit 1
        fi
    }
fi

BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}
CKPT_ROOT=${CKPT_ROOT:-$REPO_ROOT/checkpoints}
STAGE=${STAGE:-all}
SETUP_ENV=${SETUP_ENV:-1}
ENV_EVAL=${ENV_EVAL:-jbb-eval}     # requirements.txt
ENV_SFT=${ENV_SFT:-jbb-sft}        # requirements-sft.txt
ENV_GRPO=${ENV_GRPO:-jbb-grpo}     # requirements-grpo.txt
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ -z "${N_GPU:-}" ]; then
    N_GPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
fi
export N_GPU=${N_GPU:-8}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((N_GPU-1)))}
export JBB_TRAIN_DATA_DIR=${JBB_TRAIN_DATA_DIR:-$REPO_ROOT/data/train}

case "$STAGE" in all|train|eval) ;; *) echo "STAGE must be all|train|eval (got $STAGE)"; exit 1 ;; esac
mkdir -p "$CKPT_ROOT"

# 激活 conda env，SETUP_ENV=0 时跳过
use_env() {
    [ "$SETUP_ENV" = "1" ] || return 0
    [ -f "$REPO_ROOT/scripts/setup_envs.sh" ] && bash "$REPO_ROOT/scripts/setup_envs.sh" "$1"
    ensure_conda
    conda activate "$1"
}
leave_env() {
    [ "$SETUP_ENV" = "1" ] || return 0
    conda deactivate || true
}

log() { echo "=== [$(date '+%m-%d %H:%M:%S')] $* ==="; }

log "experiment=$EXPERIMENT stage=$STAGE node=$(hostname) gpus=$CUDA_VISIBLE_DEVICES base=$BASE_MODEL ckpt_root=$CKPT_ROOT"
