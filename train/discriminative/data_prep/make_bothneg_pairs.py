#!/usr/bin/env python3
"""判别式消融数据预处理：Hinge + 原始负样本 + 偏见增强负样本（scripts/discriminative/hinge_both_negatives.sh）。

pairwise（Hinge）模式的加载器只取 rejected[0]（见 train/discriminative/data_utils.py 的
collate_fn），因此把每行拆成两行：
  行1 rejected=[rejected[0]]（原始负样本），行2 rejected=[rejected[1]]（偏见负样本），
其余字段不动（verify_correctness 保留，训练加载器按 CHOSEN_IS_BEST 过滤）。
训练时 dataloader 混洗，两类配对自然交错。
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
DST = os.path.join(DISC_DIR, "unified_feedback_data_4biased_eval_no_value_1biased_bothneg_pairs.jsonl")


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
            for neg in rejected[:2]:
                row = dict(d)
                row["rejected"] = [neg]
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_out += 1
    print(f"read {n_in}, wrote {n_out} (2 pairs/row), skipped (len(rejected)<2) {n_skip}")
    if n_skip:
        print("WARNING: some rows lacked a bias-augmented negative", file=sys.stderr)


if __name__ == "__main__":
    main()
