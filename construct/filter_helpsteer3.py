"""基准构建第一步：从 HelpSteer3-Preference 抽样每类偏见的原始偏好对。
只保留单轮、general 域、有明确偏好、且 len(rejected)/len(chosen) <= 2 的样本（长度控制），再随机抽 SAMPLE_SIZE 条。
每类偏见换一个 seed 各抽一次，输出给 process_bias.py 作为 <bias>_orig.jsonl。
用法：python construct/filter_helpsteer3.py --input helpsteer3.jsonl --output orig_data/race_orig.jsonl --seed 9
"""
import argparse
import json
import random
import os

parser = argparse.ArgumentParser(description="Filter and sample HelpSteer3-Preference pairs for one bias type.")
parser.add_argument("--input", default="orig_data/helpsteer3.jsonl", help="HelpSteer3-Preference jsonl")
parser.add_argument("--output", default="orig_data/race_orig.jsonl", help="sampled pairs, one per line")
parser.add_argument("--sample_size", type=int, default=500)
parser.add_argument("--seed", type=int, default=9)
args = parser.parse_args()

INPUT_FILE_NAME = args.input
OUTPUT_FILE_NAME = args.output
SAMPLE_SIZE = args.sample_size
RANDOM_SEED = args.seed

random.seed(RANDOM_SEED)

print("--- 开始执行数据筛选和抽样任务 ---")
print(f"已设置随机种子为: {RANDOM_SEED}")

# 检查输入文件是否存在
if not os.path.exists(INPUT_FILE_NAME):
    print(f"错误：输入文件 '{INPUT_FILE_NAME}' 未在当前目录找到，请检查文件名。")
else:
    passed_filter_data = []
    total_lines = 0

    print(f"读取文件: {INPUT_FILE_NAME}")

    with open(INPUT_FILE_NAME, 'r', encoding='utf-8') as f:
        for line in f:
            total_lines += 1
            try:
                data = json.loads(line)

                # 检查 context 是否为单轮对话（即只包含一个 user 角色）
                context = data.get("context", [])
                if not isinstance(context, list):  # 确保 context 是列表
                    continue

                user_turn_count = sum(
                    1 for turn in context if turn.get("role") == "user")
                if user_turn_count != 1:
                    continue

                if data.get("domain") != "general":
                    continue

                if data.get("overall_preference") < 0:
                    chosen_text = data.get("response1")
                    rejected_text = data.get("response2")
                elif data.get("overall_preference") > 0:
                    chosen_text = data.get("response2")
                    rejected_text = data.get("response1")
                else:
                    continue

                # 确保两个关键字段都存在
                if not all([chosen_text, rejected_text]):
                    continue

                len_chosen = len(chosen_text)
                len_rejected = len(rejected_text)

                # 避免长度为 0 导致除零错误
                if min(len_chosen, len_rejected) == 0:
                    continue

                # 计算长度比，过滤掉大于 2 的
                if len_rejected / len_chosen > 2:
                    continue

                # 如果通过了所有筛选，则将原始行数据存入列表
                passed_filter_data.append(line)

            except (json.JSONDecodeError, TypeError) as e:
                continue

    num_passed = len(passed_filter_data)
    print(f"\n筛选完成。总共读取 {total_lines} 条数据，共有 {num_passed} 条数据通过筛选。")

    # --- 执行抽样 ---
    if num_passed == 0:
        print("没有数据通过筛选，无法进行抽样。")
        final_sample = []
    elif num_passed < SAMPLE_SIZE:
        print(
            f"警告: 通过筛选的数据（{num_passed}条）少于目标抽样数（{SAMPLE_SIZE}条）。将保存所有通过筛选的数据。")
        final_sample = passed_filter_data
    else:
        print(f"从 {num_passed} 条数据中随机抽样 {SAMPLE_SIZE} 条...")
        final_sample = random.sample(passed_filter_data, SAMPLE_SIZE)

    # --- 保存到新文件 ---
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_FILE_NAME)), exist_ok=True)
    with open(OUTPUT_FILE_NAME, 'w', encoding='utf-8') as f_out:
        for item_line in final_sample:
            f_out.write(item_line)

    print(f"\n任务完成！已将 {len(final_sample)} 条数据保存至新文件: '{OUTPUT_FILE_NAME}'")
