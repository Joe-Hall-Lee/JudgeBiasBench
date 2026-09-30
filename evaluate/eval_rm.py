import argparse
import json
import os
import time
from tqdm import tqdm
import torch
from rm_utils import load_reward_model
from model_utils import bias_input_file, result_file, set_data_dir, set_results_dir

# --- Constants ---
EVAL_BIAS_ONLY_APPLICABLE = [
    'length', 'authority', 'beauty', 'assertiveness', 'sycophancy',
    'sentiment', 'concreteness', 'gender', 'race'
]

# --- Main Evaluation Logic ---


def evaluate_single_bias_rm(bias_type, model, tokenizer, model_name, batch_size=32):
    """
    Evaluates a single bias type using a discriminative judge.
    """
    input_file = bias_input_file(bias_type)
    output_file = result_file(model_name, f"{bias_type}_{model_name}.jsonl")
    summary_file = result_file(model_name, f"{bias_type}_summary_{model_name}.json")

    if not os.path.exists(input_file):
        print(f"错误：输入文件 '{input_file}' 未找到。")
        return None

    with open(input_file, 'r', encoding='utf-8') as f:
        source_data = [json.loads(line) for line in f]

    print(f"\n正在评估偏见类型: {bias_type}")
    print(f"成功加载 {len(source_data)} 条数据。正在使用判别式 judge: {model_name}")

    # 1. Prepare all text pairs using the tokenizer's chat template
    all_texts_to_score = []
    print("正在使用聊天模板准备输入...")
    for item in tqdm(source_data, desc="格式化输入 (pairwise)"):
        q = item["question"]
        dialogues = [
            [{"role": "user", "content": q}, {"role": "assistant",
                                              "content": item['original_response1']}],
            [{"role": "user", "content": q}, {"role": "assistant",
                                              "content": item['original_response2']}],
            [{"role": "user", "content": q}, {"role": "assistant",
                                              "content": item['rewritten_response1']}],
            [{"role": "user", "content": q}, {"role": "assistant",
                                              "content": item['rewritten_response2']}],
        ]
        formatted_texts = [
            tokenizer.apply_chat_template(
                dialogue, tokenize=False, add_generation_prompt=False)
            for dialogue in dialogues
        ]
        all_texts_to_score.extend(formatted_texts)

    # 2. Run inference in batches
    all_scores = []
    print(f"正在使用判别式 judge为 {len(all_texts_to_score)} 个文本对打分...")
    for i in tqdm(range(0, len(all_texts_to_score), batch_size), desc=f"评估 {bias_type}"):
        batch_texts = all_texts_to_score[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            scores = outputs.cpu().tolist()
            all_scores.extend(scores)

    # 3. Process results and calculate accuracy
    results = []
    for i, item in enumerate(source_data):
        score_idx = i * 4
        score_orig1, score_orig2, score_bias1, score_bias2 = all_scores[
            score_idx: score_idx + 4]
        is_resp1_better = item['label'] == 'response1'
        is_orig_eval_correct = (score_orig1 > score_orig2) if is_resp1_better else (
            score_orig2 > score_orig1)
        is_bias_eval_correct = (score_bias1 > score_bias2) if is_resp1_better else (
            score_bias2 > score_bias1)
        results.append({
            "question": item["question"],
            "original_response1": item["original_response1"],
            "original_response2": item["original_response2"],
            "rewritten_response1": item["rewritten_response1"],
            "rewritten_response2": item["rewritten_response2"],
            "label": item["label"],
            "scores_original": [score_orig1, score_orig2],
            "scores_biased": [score_bias1, score_bias2],
            "original_evaluation_correct": is_orig_eval_correct,
            "bias_evaluation_correct": is_bias_eval_correct,
        })

    # 4. Calculate and save summary statistics
    total_evaluated = len(source_data)
    orig_correct_count = sum(
        1 for r in results if r['original_evaluation_correct'])
    bias_correct_count = sum(
        1 for r in results if r['bias_evaluation_correct'])

    # BSR: samples where the original pair is judged correctly but the biased pair is not
    bias_sensitive_count = sum(
        1 for r in results if r['original_evaluation_correct'] and not r['bias_evaluation_correct'])

    orig_accuracy = orig_correct_count / total_evaluated if total_evaluated > 0 else 0
    bias_accuracy = bias_correct_count / total_evaluated if total_evaluated > 0 else 0
    accuracy_drop = orig_accuracy - bias_accuracy
    bias_sensitive_rate = bias_sensitive_count / orig_correct_count if orig_correct_count > 0 else 0

    # same keys as eval_judge.py / eval_api.py
    summary_stats = {
        "eval_model": model_name,
        "total_records_evaluated": total_evaluated,
        "num_orig_evaluated": total_evaluated,
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
    print("\n--- 判别式 judge 评估摘要 (Pairwise) ---")
    print(f"原始数据对准确率: {summary_stats['original_pair_accuracy']}")
    print(f"偏见数据对准确率: {summary_stats['biased_pair_accuracy']}")
    print(f"偏见敏感率 (BSR): {summary_stats['bias_sensitive_rate']['rate']} ({bias_sensitive_count}/{orig_correct_count})")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for record in results:
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"\n详细评估结果已保存至: '{output_file}'")

    with open(summary_file, 'w', encoding='utf-8') as f_summary:
        json.dump(summary_stats, f_summary, indent=4, ensure_ascii=False)
    print(f"统计摘要已保存至: '{summary_file}'")

    return summary_stats


def main():
    parser = argparse.ArgumentParser(description="使用判别式 judge评估不同类型的偏见。")
    parser.add_argument("bias_type", type=str, choices=EVAL_BIAS_ONLY_APPLICABLE + ['all'],
                        help=f"要评估的偏见类型。可用选项: {EVAL_BIAS_ONLY_APPLICABLE + ['all']}")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Hugging Face 模型目录的路径 (必须是 ForSequenceClassification 模型)。")
    parser.add_argument("--model_name", type=str,
                        help="模型的名称，用于创建结果目录。默认为 model_path 的最后一部分。")
    parser.add_argument("--batch_size", type=int,
                        default=32, help="推理时的批处理大小。")

    parser.add_argument("--overwrite", action='store_true',
                        help="Re-run biases whose summary file already exists (default: skip, for resumable jobs).")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory with <bias>_bias.jsonl files (default: <repo>/data/eval).")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Root results directory (default: <repo>/results).")
    args = parser.parse_args()
    set_data_dir(args.data_dir)
    set_results_dir(args.output_dir)

    if not args.model_name:
        args.model_name = os.path.basename(args.model_path)

    # Load Model and Tokenizer with model parallelism
    print(f"正在从以下路径加载判别式 judge: {args.model_path}")
    print("注意：将启用模型并行，自动将模型分配到所有可用的 GPU 上。")
    try:
        model, tokenizer = load_reward_model(args.model_path)
    except Exception as e:
        print(f"从路径加载模型失败: {args.model_path}")
        print(f"错误: {e}")
        return

    start_time = time.time()
    if args.bias_type == 'all':
        print(f"将开始评估所有适用的偏见类型: {EVAL_BIAS_ONLY_APPLICABLE}")
        all_summaries = {}
        for bias in EVAL_BIAS_ONLY_APPLICABLE:
            summary_file = result_file(args.model_name, f"{bias}_summary_{args.model_name}.json")
            if not args.overwrite and os.path.exists(summary_file):
                with open(summary_file, 'r', encoding='utf-8') as f:
                    summary = json.load(f)
                print(f"[resume] {bias}: reusing existing summary {summary_file}")
            else:
                summary = evaluate_single_bias_rm(
                    bias, model, tokenizer, args.model_name, args.batch_size)
            if summary:
                all_summaries[bias] = summary
            print("\n" + "="*80 + "\n")

        # overall metrics pooled over all biases, same block as eval_judge.py
        overall = {"total_records_evaluated": 0, "num_orig_evaluated": 0,
                   "orig_correct_count": 0, "bias_correct_count": 0, "bias_sensitive_count": 0}
        for summary in all_summaries.values():
            for key in overall:
                overall[key] += int(summary.get(key, 0))
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

        all_summary_file = result_file(args.model_name, f"all_biases_summary_{args.model_name}.json")
        with open(all_summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_summaries, f, indent=4, ensure_ascii=False)
        elapsed_time = time.time() - start_time
        print(f"\n🎉 所有偏见类型评估完成！总耗时: {elapsed_time:.2f} 秒")
        print(f"总体摘要已保存至: '{all_summary_file}'")
        
        overall_block = all_summaries["_overall"]
        print(f"\n--- 所有偏见类型的总体指标 ---")
        print(f"总评估记录数: {overall_block['total_records_evaluated']}")
        print(f"原始数据对准确率: {overall_block['original_pair_accuracy']}")
        print(f"偏见数据对准确率: {overall_block['biased_pair_accuracy']}")
        print(f"偏见敏感率 (BSR): {overall_block['bias_sensitive_rate']['rate']}")
    else:
        evaluate_single_bias_rm(args.bias_type, model,
                                tokenizer, args.model_name, args.batch_size)
        elapsed_time = time.time() - start_time
        print(f"\n🎉 处理完成！总耗时: {elapsed_time:.2f} 秒")


if __name__ == "__main__":
    main()
