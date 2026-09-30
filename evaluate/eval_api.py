import json
import os
import sys
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # repo root
from common.api import query_model
from build_prompt import create_eval_prompt
from model_utils import (
    bias_input_file, result_file, set_data_dir, set_results_dir,
    parse_pairwise, apply_bias_injection,
)

MAX_WORKERS = 10
MODEL_TYPE = "default"   # API models use the general prompt; injection markers must match it
ALL_BIASES = ['length', 'authority', 'beauty', 'assertiveness', 'sycophancy',
              'sentiment', 'concreteness', 'gender', 'race', 'bandwagon',
              'superficial-reflection', 'position']

EVAL_BIAS_ONLY_APPLICABLE = [
    'length', 'authority', 'beauty', 'assertiveness', 'sycophancy',
    'sentiment', 'concreteness', 'gender', 'race'
]


def evaluate_pairwise_item(item_and_bias_type, model_name, eval_bias_only=False):
    """Evaluate one record; the record layout and result fields follow eval_judge.py."""
    data_item, bias_type = item_and_bias_type
    try:
        question = data_item.get("question")
        eval_prompt_template = create_eval_prompt(MODEL_TYPE)
        label = data_item.get("label")

        # position: same pair in both orders, the judge should pick the same response twice
        if bias_type == 'position':
            label_is_response1 = label == "response1"
            orig_chosen = data_item.get("response1") if label_is_response1 else data_item.get("response2")
            orig_rejected = data_item.get("response2") if label_is_response1 else data_item.get("response1")
            prompt_orig = eval_prompt_template.format(
                question=question, answer_a=orig_chosen, answer_b=orig_rejected)
            prompt_bias = eval_prompt_template.format(
                question=question, answer_a=orig_rejected, answer_b=orig_chosen)
            result_orig_raw = query_model(prompt_orig, model_name=model_name, temperature=0)
            result_bias_raw = query_model(prompt_bias, model_name=model_name, temperature=0)
            parsed_orig = parse_pairwise(result_orig_raw, MODEL_TYPE)
            parsed_bias = parse_pairwise(result_bias_raw, MODEL_TYPE)
            is_orig_eval_correct = (parsed_orig == 'A') if parsed_orig not in ["Error", "Tie"] else parsed_orig
            is_bias_eval_correct = "eval_error" if parsed_bias in ["Error", "Tie"] else (parsed_bias == 'B')
            return {"question": question, "response1": data_item.get("response1"),
                    "response2": data_item.get("response2"), "label": label,
                    "original_evaluation_correct": is_orig_eval_correct,
                    "bias_evaluation_correct": is_bias_eval_correct,
                    "raw_evaluation_original": result_orig_raw, "raw_evaluation_bias": result_bias_raw}

        correct_position_in_prompt = 'A' if label == "response1" else 'B'

        # bandwagon / superficial-reflection keep the responses and manipulate the prompt instead
        is_prompt_bias = bias_type in ['bandwagon', 'superficial-reflection']
        if is_prompt_bias:
            response1_orig = data_item.get("response1")
            response2_orig = data_item.get("response2")
            response1_bias, response2_bias = response1_orig, response2_orig
        else:
            response1_orig = data_item.get("original_response1")
            response2_orig = data_item.get("original_response2")
            response1_bias = data_item.get("rewritten_response1")
            response2_bias = data_item.get("rewritten_response2")

        prompt_orig = eval_prompt_template.format(
            question=question, answer_a=response1_orig, answer_b=response2_orig)
        if is_prompt_bias:
            prompt_bias = apply_bias_injection(prompt_orig, bias_type, MODEL_TYPE, correct_position_in_prompt)
        else:
            prompt_bias = eval_prompt_template.format(
                question=question, answer_a=response1_bias, answer_b=response2_bias)

        result_orig_raw, parsed_result_orig = "skipped", "skipped"
        if not (eval_bias_only and bias_type in EVAL_BIAS_ONLY_APPLICABLE):
            result_orig_raw = query_model(prompt_orig, model_name=model_name, temperature=0)
            parsed_result_orig = parse_pairwise(result_orig_raw, MODEL_TYPE)

        result_bias_raw = query_model(prompt_bias, model_name=model_name, temperature=0)
        parsed_result_bias = parse_pairwise(result_bias_raw, MODEL_TYPE)

        is_orig_eval_correct = (parsed_result_orig == correct_position_in_prompt) if parsed_result_orig not in [
            "Error", "skipped", "Tie"] else parsed_result_orig
        is_bias_eval_correct = "eval_error" if parsed_result_bias in ["Error", "Tie"] else (
            parsed_result_bias == correct_position_in_prompt)

        record = {"question": question, "label": label,
                  "original_evaluation_correct": is_orig_eval_correct,
                  "bias_evaluation_correct": is_bias_eval_correct,
                  "raw_evaluation_original": result_orig_raw, "raw_evaluation_bias": result_bias_raw}
        if is_prompt_bias:
            record.update({"response1": response1_orig, "response2": response2_orig})
        else:
            record.update({"original_response1": response1_orig, "original_response2": response2_orig,
                           "rewritten_response1": response1_bias, "rewritten_response2": response2_bias})
        return record
    except Exception as e:
        print(f"处理 pairwise 数据时发生错误: {e}")
        return None


def _compute_summary(results, model_name):
    """Same statistics and keys as eval_judge.py so the summaries can be merged."""
    total_evaluated = len(results)
    num_orig_evaluated = total_evaluated - sum(
        1 for r in results if r['original_evaluation_correct'] == 'skipped')
    orig_correct_count = sum(1 for r in results if r['original_evaluation_correct'] is True)
    bias_correct_count = sum(1 for r in results if r['bias_evaluation_correct'] is True)
    bias_sensitive_count = sum(
        1 for r in results
        if r['original_evaluation_correct'] is True and r['bias_evaluation_correct'] is False)

    orig_accuracy = orig_correct_count / num_orig_evaluated if num_orig_evaluated > 0 else 0
    bias_accuracy = bias_correct_count / total_evaluated if total_evaluated > 0 else 0
    accuracy_drop = orig_accuracy - bias_accuracy
    bias_sensitive_rate = bias_sensitive_count / orig_correct_count if orig_correct_count > 0 else 0

    summary_stats = {
        "eval_model": model_name,
        "total_records_evaluated": total_evaluated,
        "num_orig_evaluated": num_orig_evaluated,
        "orig_correct_count": orig_correct_count,
        "bias_correct_count": bias_correct_count,
        "bias_sensitive_count": bias_sensitive_count,
        "original_pair_accuracy": f"{orig_accuracy * 100:.2f}%",
        "biased_pair_accuracy": f"{bias_accuracy * 100:.2f}%",
        "accuracy_drop_due_to_bias": f"{accuracy_drop * 100:.2f}%",
        "bias_sensitive_rate": {
            "count": bias_sensitive_count,
            "rate": f"{bias_sensitive_rate * 100:.2f}%"
        }
    }
    print("\n--- Pairwise Evaluation Summary ---")
    print(f"Original Pair Accuracy: {summary_stats['original_pair_accuracy']}")
    print(f"Biased Pair Accuracy: {summary_stats['biased_pair_accuracy']}")
    print(f"Accuracy Drop due to Bias: {summary_stats['accuracy_drop_due_to_bias']}")
    print(f"Bias Sensitive Rate (BSR): {summary_stats['bias_sensitive_rate']['rate']} "
          f"({bias_sensitive_count}/{orig_correct_count})")
    return summary_stats


def evaluate_single_bias(bias_type, model_name, eval_bias_only=False):
    """评估单个偏见类型的辅助函数"""
    input_file = bias_input_file(bias_type)
    output_file = result_file(model_name, f"{bias_type}_{model_name}.jsonl")
    summary_file = result_file(model_name, f"{bias_type}_summary_{model_name}.json")

    if not os.path.exists(input_file):
        print(f"错误: 输入文件 '{input_file}' 未找到。")
        return None

    with open(input_file, 'r', encoding='utf-8') as f:
        source_data = [json.loads(line) for line in f]

    print(f"\n评估偏见类型: {bias_type}")
    print(
        f"成功加载 {len(source_data)} 条数据。使用 {model_name} 模型和 {MAX_WORKERS} 个线程进行评估。")
    if eval_bias_only:
        print("模式：仅评估偏见数据对。")

    target_func = partial(
        evaluate_pairwise_item, model_name=model_name, eval_bias_only=eval_bias_only)
    items_to_process = [(item, bias_type) for item in source_data]
    print("启动 Pairwise 评估模式...")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results_iterator = executor.map(target_func, items_to_process)
        results = [res for res in tqdm(results_iterator, total=len(
            source_data), desc=f"评估 {bias_type}") if res is not None]

    print(f"\n{bias_type} 处理完成，成功评估 {len(results)} 条新数据。正在写入文件...")

    # 确保结果目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for record in results:
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"详细评估结果已保存至: '{output_file}'")

    total_evaluated = len(results)
    if total_evaluated > 0:
        if eval_bias_only:
            bias_correct_count = sum(
                1 for r in results if r['bias_evaluation_correct'] is True)
            bias_accuracy = bias_correct_count / total_evaluated if total_evaluated > 0 else 0
            summary_stats = {"eval_model": model_name, "total_records_evaluated": total_evaluated,
                             "bias_correct_count": bias_correct_count,
                             "biased_pair_accuracy": f"{bias_accuracy * 100:.2f}%"}
            print("\n--- Pairwise 评估结果统计 (仅偏见数据) ---")
            print(
                f"偏见数据对评估准确率 (Biased Accuracy): {summary_stats['biased_pair_accuracy']}")
        else:
            summary_stats = _compute_summary(results, model_name)

        with open(summary_file, 'w', encoding='utf-8') as f_summary:
            json.dump(summary_stats, f_summary, indent=4, ensure_ascii=False)
        print(f"统计摘要已保存至: '{summary_file}'")

        return summary_stats
    return None


def main():
    parser = argparse.ArgumentParser(description="评估不同类型偏见对模型的影响。")
    parser.add_argument("bias_type", type=str, choices=ALL_BIASES + ['all'],
                        help=f"要评估的偏见类型。可用选项: {ALL_BIASES + ['all']}")
    parser.add_argument("--model", type=str,
                        default="gpt-4o", help="用于评估的 judge 模型名称。")
    parser.add_argument("--eval_bias_only", action='store_true',
                        help="仅评估偏见数据对（仅适用于部分数据改写型偏见）。")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory with <bias>_bias.jsonl files (default: <repo>/data/eval).")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Root results directory (default: <repo>/results).")
    parser.add_argument("--overwrite", action='store_true',
                        help="Re-evaluate bias types whose summary already exists (default: resume).")
    args = parser.parse_args()
    set_data_dir(args.data_dir)
    set_results_dir(args.output_dir)

    bias_type = args.bias_type
    model_name = args.model
    eval_bias_only = args.eval_bias_only

    if eval_bias_only and bias_type == 'all':
        print("警告: --eval_bias_only 参数在评估所有偏见类型时将被忽略。")
    elif eval_bias_only and bias_type not in EVAL_BIAS_ONLY_APPLICABLE:
        print(f"错误: --eval_bias_only 参数不适用于 '{bias_type}' 偏见类型。")
        print(f"适用类型: {EVAL_BIAS_ONLY_APPLICABLE}")
        return

    if bias_type == 'all':
        print(f"将评估所有偏见类型: {ALL_BIASES}")
        all_summaries = {}
        overall = {"total_records_evaluated": 0, "num_orig_evaluated": 0,
                   "orig_correct_count": 0, "bias_correct_count": 0, "bias_sensitive_count": 0}
        start_time = time.time()

        for bias in ALL_BIASES:
            summary_file = result_file(model_name, f"{bias}_summary_{model_name}.json")
            if not args.overwrite and os.path.exists(summary_file):
                with open(summary_file, 'r', encoding='utf-8') as f:
                    summary = json.load(f)
                print(f"[resume] {bias}: reusing existing summary {summary_file}")
            else:
                summary = evaluate_single_bias(bias, model_name, False)
            if summary:
                all_summaries[bias] = summary
                for key in overall:
                    overall[key] += int(summary.get(key, 0))
            print("\n" + "="*80 + "\n")

        overall_orig_acc = overall["orig_correct_count"] / overall["num_orig_evaluated"] if overall["num_orig_evaluated"] else 0.0
        overall_bias_acc = overall["bias_correct_count"] / overall["total_records_evaluated"] if overall["total_records_evaluated"] else 0.0
        overall_bsr = overall["bias_sensitive_count"] / overall["orig_correct_count"] if overall["orig_correct_count"] else 0.0
        all_summaries["_overall"] = {
            **overall,
            "original_pair_accuracy": f"{overall_orig_acc * 100:.2f}%",
            "biased_pair_accuracy": f"{overall_bias_acc * 100:.2f}%",
            "bias_sensitive_rate": {
                "count": overall["bias_sensitive_count"],
                "rate": f"{overall_bsr * 100:.2f}%"
            }
        }

        # 保存所有评估的汇总结果
        all_summary_file = result_file(model_name, f"all_biases_summary_{model_name}.json")
        with open(all_summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_summaries, f, indent=4, ensure_ascii=False)

        elapsed_time = time.time() - start_time
        print(f"\n全部偏见类型评估完成！总耗时: {elapsed_time:.2f}秒")
        print(f"汇总结果已保存至: '{all_summary_file}'")
    else:
        evaluate_single_bias(bias_type, model_name, eval_bias_only)
        print(f"\n处理完成！")


if __name__ == "__main__":
    main()
