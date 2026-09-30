#!/usr/bin/env python3
"""判别式消融数据预处理：InfoNCE + 仅原始负样本（scripts/discriminative/contrast_original_negatives.sh）。

pairwise（Hinge）模式的加载器只取 rejected[0]（见 train/discriminative/data_utils.py 的
collate_fn），主实验数据里 rejected[0] 是原始负样本、rejected[1] 是偏见增强负样本。
本脚本把每行的 rejected 替换为 [rejected[0]]（原始负样本），contrast 模式下组内
只含 (chosen, original-negative)，其余字段不动。
"""
import json
import sys
import os

# Training-data roots (see README.md, "Training data"). Override with JBB_TRAIN_DATA_DIR.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
_TRAIN_DATA = os.environ.get("JBB_TRAIN_DATA_DIR", os.path.join(_REPO_ROOT, "data", "train"))
DISC_DIR = os.path.join(_TRAIN_DATA, "discriminative")            # unified_feedback_*.jsonl
INTER_DIR = os.path.join(_TRAIN_DATA, "generative", "intermediate")  # per-bias splits, teacher CoT
SFT_DIR = os.path.join(_TRAIN_DATA, "generative", "sft")          # LLaMA-Factory json + dataset_info.json
GRPO_DIR = os.path.join(_TRAIN_DATA, "generative", "grpo")        # EasyR1 parquet (bias_0p00/0p25/0p50)
for _d in (DISC_DIR, INTER_DIR, SFT_DIR, GRPO_DIR):
    os.makedirs(_d, exist_ok=True)

SRC = os.path.join(DISC_DIR, "unified_feedback_data_4biased_eval_no_value_1biased.jsonl")
DST = os.path.join(DISC_DIR, "unified_feedback_data_4biased_eval_no_value_1biased_origneg_only.jsonl")


def main():
    n_in, n_out, n_skip = 0, 0, 0
    with open(SRC, encoding="utf-8") as fin, open(DST, "w", encoding="utf-8") as fout:
        for line in fin:
            n_in += 1
            d = json.loads(line)
            rejected = d.get("rejected", [])
            if len(rejected) < 2:
                n_skip += 1
                continue
            d["rejected"] = [rejected[0]]
            fout.write(json.dumps(d, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"read {n_in}, wrote {n_out}, skipped (len(rejected)<2) {n_skip}")
    if n_skip:
        print("WARNING: some rows lacked a bias-augmented negative", file=sys.stderr)


if __name__ == "__main__":
    main()
