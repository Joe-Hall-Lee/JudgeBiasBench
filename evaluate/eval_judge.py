# -*- coding:utf-8 -*-
import json
import os
import argparse
import time
from tqdm import tqdm
import vllm
from transformers import AutoTokenizer
from model_utils import (
    bias_input_file, result_file, set_data_dir, set_results_dir,
    parse_pairwise,
    apply_bias_injection
)
from build_prompt import create_eval_prompt, MODEL_TYPES
from rewardbench_handler import evaluate_rewardbench
from rmbench_handler import evaluate_rmbench
from generic_benchmark_handler import evaluate_judgebench, evaluate_rmb

# --- Constants ---
ALL_BIASES = ['length', 'authority', 'beauty', 'assertiveness', 'sycophancy',
              'sentiment', 'concreteness', 'gender', 'race', 'bandwagon',
              'superficial-reflection', 'position', 'rewardbench', 'rmbench', 'judgebench', 'rmb']
PAIRWISE_BIASES = ['length', 'authority', 'beauty', 'assertiveness', 'sycophancy',
                   'sentiment', 'concreteness', 'gender', 'race', 'bandwagon',
                   'superficial-reflection', 'position']

EVAL_BIAS_ONLY_APPLICABLE = [
    'length', 'authority', 'beauty', 'assertiveness', 'sycophancy',
    'sentiment', 'concreteness', 'gender', 'race'
]

# --- Prompt Generation and Result Processing ---


def process_evaluation_results(source_data, vllm_outputs, bias_type, model_type, eval_bias_only):
    """
    将 VLLM 的批量输出映射回原始数据项，并根据模型类型解析结果。
    """
    results = []
    output_idx = 0

    for item in tqdm(source_data, desc=f"解析 {bias_type} 结果"):
        try:
            num_prompts = 2 if not (
                eval_bias_only and bias_type in EVAL_BIAS_ONLY_APPLICABLE) else 1
            result_orig_raw, parsed_result_orig = "skipped", "skipped"
            if num_prompts == 2:
                result_orig_raw = vllm_outputs[output_idx].outputs[0].text
                parsed_result_orig = parse_pairwise(
                    result_orig_raw, model_type=model_type)
                output_idx += 1

            result_bias_raw = vllm_outputs[output_idx].outputs[0].text
            parsed_result_bias = parse_pairwise(
                result_bias_raw, model_type=model_type)
            output_idx += 1

            label = item.get("label")
            if bias_type == 'position':
                is_orig_eval_correct = (parsed_result_orig == 'A') if parsed_result_orig not in [
                    "Error", "skipped", "Tie"] else parsed_result_orig
                is_bias_eval_correct = "eval_error" if parsed_result_bias in [
                    "Error", "Tie"] else (parsed_result_bias == 'B')
            else:
                correct_choice = 'A' if label == "response1" else 'B'
                is_orig_eval_correct = (parsed_result_orig == correct_choice) if parsed_result_orig not in [
                    "Error", "skipped", "Tie"] else parsed_result_orig
                is_bias_eval_correct = "eval_error" if parsed_result_bias in [
                    "Error", "Tie"] else (parsed_result_bias == correct_choice)

            if bias_type in ['bandwagon', 'superficial-reflection', 'position']:
                results.append({
                    "question": item.get("question"),
                    "response1": item.get("response1"),
                    "response2": item.get("response2"),
                    "label": label,
                    "original_evaluation_correct": is_orig_eval_correct,
                    "bias_evaluation_correct": is_bias_eval_correct,
                    "raw_evaluation_original": result_orig_raw,
                    "raw_evaluation_bias": result_bias_raw
                })
            else:
                results.append({
                    "question": item.get("question"),
                    "original_response1": item.get("original_response1"),
                    "original_response2": item.get("original_response2"),
                    "rewritten_response1": item.get("rewritten_response1"),
                    "rewritten_response2": item.get("rewritten_response2"),
                    "label": label,
                    "original_evaluation_correct": is_orig_eval_correct,
                    "bias_evaluation_correct": is_bias_eval_correct,
                    "raw_evaluation_original": result_orig_raw,
                    "raw_evaluation_bias": result_bias_raw
                })
        except Exception as e:
            print(f"处理结果时出错: {e} | 项目: {item.get('question', 'N/A')}")

    return results



# --- Main Evaluation Logic ---



def evaluate_single_bias(bias_type, model_name, model_type, model, tokenizer, sampling_params, eval_bias_only=False, enable_thinking=False):
    """
    Evaluates a single bias type by batching all prompts and using VLLM for inference.
    """
    # 如果是 rewardbench 类型，使用专门的函数
    if bias_type == "rewardbench":
        return evaluate_rewardbench(model_name, model, tokenizer, sampling_params, model_type, enable_thinking)

    # 如果是 rmbench 类型，使用专门的函数
    if bias_type == "rmbench":
        return evaluate_rmbench(model_name, model, tokenizer, sampling_params, model_type, enable_thinking)

    # 如果是 judgebench 类型
    if bias_type == "judgebench":
        return evaluate_judgebench(model_name, model, tokenizer, sampling_params, model_type, enable_thinking)

    # 如果是 rmb 类型
    if bias_type == "rmb":
        return evaluate_rmb(model_name, model, tokenizer, sampling_params, model_type, enable_thinking)

    input_file = bias_input_file(bias_type)
    output_file = result_file(model_name, f"{bias_type}_{model_name}.jsonl")
    summary_file = result_file(model_name, f"{bias_type}_summary_{model_name}.json")

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return None

    with open(input_file, 'r', encoding='utf-8') as f:
        source_data = [json.loads(line) for line in f]

    print(f"\nEvaluating bias type: {bias_type}")
    print(
        f"Successfully loaded {len(source_data)} items. Using VLLM model for evaluation.")
    if eval_bias_only:
        print("Mode: Evaluating biased data pairs only.")

    all_prompts = []
    eval_prompt_template = create_eval_prompt(model_type)

    has_chat_template = tokenizer is not None and tokenizer.chat_template is not None
    if has_chat_template:
        print("检测到聊天模板，将使用 apply_chat_template。")
    else:
        print("未检测到聊天模板，将直接使用格式化后的 prompt 字符串。")

    # Prepare keyword arguments for the chat template conditionally.
    template_kwargs = {
        'tokenize': False,
        'add_generation_prompt': True
    }
    if 'qwen3' in model_name.lower():
        template_kwargs['enable_thinking'] = enable_thinking
        if enable_thinking:
            print("Info: 'enable_thinking=True' is active for Qwen3 model.")
        else:
            print("Info: 'enable_thinking=False' is active for Qwen3 model.")

    def format_with_chat_template(prompt_text):
        """统一的聊天模板应用函数"""
        if not has_chat_template:
            return prompt_text
        messages = [{"role": "user", "content": prompt_text}]
        return tokenizer.apply_chat_template(messages, **template_kwargs)

    for item in tqdm(source_data, desc=f"Generating prompts for {bias_type}"):
        question = item.get("question")

        if bias_type == 'position':
            label_is_response1 = item.get("label") == "response1"
            orig_chosen = item.get(
                "response1") if label_is_response1 else item.get("response2")
            orig_rejected = item.get(
                "response2") if label_is_response1 else item.get("response1")

            prompt_orig_text = eval_prompt_template.format(
                question=question, answer_a=orig_chosen, answer_b=orig_rejected)
            prompt_bias_text = eval_prompt_template.format(
                question=question, answer_a=orig_rejected, answer_b=orig_chosen)
        else:
            correct_position_in_prompt = 'A' if item.get(
                "label") == "response1" else 'B'

            if bias_type in ['bandwagon', 'superficial-reflection']:
                response1_orig, response2_orig = item.get(
                    "response1"), item.get("response2")
                response1_bias, response2_bias = response1_orig, response2_orig
            else:
                response1_orig, response2_orig = item.get(
                    "original_response1"), item.get("original_response2")
                response1_bias, response2_bias = item.get(
                    "rewritten_response1"), item.get("rewritten_response2")

            prompt_orig_text = eval_prompt_template.format(
                question=question, answer_a=response1_orig, answer_b=response2_orig)

            if bias_type in ['bandwagon', 'superficial-reflection']:
                prompt_bias_text = apply_bias_injection(
                    prompt_orig_text, bias_type, model_type, correct_position_in_prompt
                )
            else:
                prompt_bias_text = eval_prompt_template.format(
                    question=question, answer_a=response1_bias, answer_b=response2_bias)

        # 应用聊天模板
        prompt_orig_text = format_with_chat_template(prompt_orig_text)
        prompt_bias_text = format_with_chat_template(prompt_bias_text)

        if not (eval_bias_only and bias_type in EVAL_BIAS_ONLY_APPLICABLE):
            all_prompts.append(prompt_orig_text)
        all_prompts.append(prompt_bias_text)

    print(f"Sending {len(all_prompts)} prompts to VLLM for generation...")
    vllm_outputs = model.generate(all_prompts, sampling_params)
    print("VLLM generation complete.")

    results = process_evaluation_results(
        source_data, vllm_outputs, bias_type, model_type, eval_bias_only)
    print(f"\n{bias_type} processing finished, successfully evaluated {len(results)} new items. Writing to file...")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for record in results:
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"Detailed evaluation results saved to: '{output_file}'")

    # Summary statistics logic
    summary_stats = compute_summary_statistics(
        results, bias_type, model_name, eval_bias_only
    )

    if summary_stats:
        with open(summary_file, 'w', encoding='utf-8') as f_summary:
            json.dump(summary_stats, f_summary, indent=4, ensure_ascii=False)
        print(f"Summary statistics saved to: '{summary_file}'")

    return summary_stats


def compute_summary_statistics(results, bias_type, model_name, eval_bias_only):
    """
    计算并打印汇总统计信息
    """
    total_evaluated = len(results)
    if total_evaluated == 0:
        return None

    return _compute_pairwise_summary(results, model_name, total_evaluated, eval_bias_only)


def _compute_pairwise_summary(results, model_name, total_evaluated, eval_bias_only):
    """计算 pairwise 类型的汇总"""
    if eval_bias_only:
        bias_correct_count = sum(
            1 for r in results if r['bias_evaluation_correct'] is True)
        bias_accuracy = bias_correct_count / total_evaluated if total_evaluated > 0 else 0

        summary_stats = {
            "eval_model": model_name,
            "total_records_evaluated": total_evaluated,
            "biased_pair_accuracy": f"{bias_accuracy * 100:.2f}%"
        }

        print("\n--- Pairwise Evaluation Summary (Biased Only) ---")
        print(f"Biased Pair Accuracy: {summary_stats['biased_pair_accuracy']}")
        return summary_stats

    # Full evaluation
    num_orig_evaluated = total_evaluated - sum(
        1 for r in results if r['original_evaluation_correct'] == 'skipped'
    )

    orig_correct_count = sum(
        1 for r in results if r['original_evaluation_correct'] is True)
    bias_correct_count = sum(
        1 for r in results if r['bias_evaluation_correct'] is True)

    bias_sensitive_count = sum(
        1 for r in results
        if r['original_evaluation_correct'] is True
        and r['bias_evaluation_correct'] is False
    )

    orig_accuracy = orig_correct_count / \
        num_orig_evaluated if num_orig_evaluated > 0 else 0
    bias_accuracy = bias_correct_count / total_evaluated if total_evaluated > 0 else 0
    accuracy_drop = orig_accuracy - bias_accuracy
    bias_sensitive_rate = bias_sensitive_count / \
        orig_correct_count if orig_correct_count > 0 else 0

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
    print(
        f"Accuracy Drop due to Bias: {summary_stats['accuracy_drop_due_to_bias']}")
    print(f"Bias Sensitive Rate (BSR): {summary_stats['bias_sensitive_rate']['rate']} "
          f"({bias_sensitive_count}/{orig_correct_count})")

    return summary_stats


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model biases using VLLM for local inference.")
    parser.add_argument("bias_type", type=str, choices=ALL_BIASES + ['all'],
                        help=f"Bias type to evaluate. Options: {ALL_BIASES + ['all']}")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the VLLM-compatible model directory.")
    parser.add_argument("--model_name", type=str, default="vllm_model",
                        help="A name for the model, used for creating result directories.")
    parser.add_argument("--model_type", type=str, default="default", choices=MODEL_TYPES,
                        help="Prompt format. 'default' for general LLMs; 'think'/'direct' are the <think>-CoT / label-only "
                             "prompts of our trained judges; the rest are the native formats of fine-tuned judge baselines.")

    parser.add_argument("--eval_bias_only", action='store_true',
                        help="Only evaluate biased data pairs (for applicable bias types).")
    parser.add_argument("--temperature", type=float,
                        default=0.0, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Sampling top_p.")
    parser.add_argument("--top_k", type=int, default=-1,
                        help="The number of highest probability vocabulary tokens to keep for top-k-filtering. -1 disables it.")
    parser.add_argument("--min_p", type=float, default=0.0,
                        help="Minimum probability for nucleus sampling (min_p).")
    parser.add_argument("--max_new_token", type=int,
                        default=512, help="Maximum new tokens to generate.")
    parser.add_argument("--tensor_parallel_size", type=int,
                        default=1, help="Tensor parallel size for VLLM.")
    parser.add_argument("--gpu_memory_utilization", type=float,
                        default=0.9, help="GPU memory utilization for VLLM.")
    parser.add_argument("--enable_thinking", action='store_true',
                        help="Enable 'thinking' tokens in chat template, for models like Qwen2/Qwen3.")

    parser.add_argument("--overwrite", action='store_true',
                        help="Re-run biases whose summary file already exists (default: skip them, for resumable jobs).")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory with <bias>_bias.jsonl files (default: <repo>/data/eval).")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Root results directory; outputs go to <output_dir>/<model_name>/ (default: <repo>/results).")
    args = parser.parse_args()
    set_data_dir(args.data_dir)
    set_results_dir(args.output_dir)

    if args.eval_bias_only and args.bias_type == 'all':
        print("Warning: --eval_bias_only is ignored when evaluating 'all' bias types.")
    elif args.eval_bias_only and args.bias_type not in EVAL_BIAS_ONLY_APPLICABLE:
        print(
            f"Error: --eval_bias_only is not applicable to '{args.bias_type}' bias type.")
        print(f"Applicable types: {EVAL_BIAS_ONLY_APPLICABLE}")
        return

    # Load VLLM model and tokenizer
    print("Loading VLLM model...")
    try:
        model = vllm.LLM(
            model=args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype="bfloat16",
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        sampling_params = vllm.SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_new_token,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p
        )
        print("VLLM model and tokenizer loaded successfully!")
    except Exception as e:
        print(f"Failed to load VLLM model from path: {args.model_path}")
        print("Full error:", repr(e))
        return

    start_time = time.time()
    if args.bias_type == 'all':
        print("Starting evaluation for all orig/bias test sets (excluding benchmarks).")
        biases_to_run = [b for b in ALL_BIASES if b not in ['rewardbench', 'rmbench', 'judgebench', 'rmb']]
        print(f"Bias types to evaluate: {biases_to_run}")
        all_summaries = {}
        # Aggregation counters for overall metrics (pairwise only)
        overall_num_orig_evaluated = 0
        overall_total_evaluated = 0
        overall_orig_correct = 0
        overall_bias_correct = 0
        overall_bsr_sensitive = 0

        for bias in biases_to_run:
            # Resume support for preemptible jobs: reuse a finished per-bias summary unless --overwrite
            summary_file = result_file(args.model_name, f"{bias}_summary_{args.model_name}.json")
            if not args.overwrite and os.path.exists(summary_file):
                with open(summary_file, 'r', encoding='utf-8') as f:
                    summary = json.load(f)
                print(f"[resume] {bias}: reusing existing summary {summary_file}")
            else:
                # Force full evaluation (orig + bias) for 'all'
                summary = evaluate_single_bias(
                    bias, args.model_name, args.model_type, model, tokenizer,
                    sampling_params, False, args.enable_thinking
                )
            if summary:
                all_summaries[bias] = summary
                # Accumulate overall metrics for pairwise biases only
                if bias in PAIRWISE_BIASES and "num_orig_evaluated" in summary:
                    overall_num_orig_evaluated += int(summary.get("num_orig_evaluated", 0))
                    overall_total_evaluated += int(summary.get("total_records_evaluated", 0))
                    overall_orig_correct += int(summary.get("orig_correct_count", 0))
                    overall_bias_correct += int(summary.get("bias_correct_count", 0))
                    overall_bsr_sensitive += int(summary.get("bias_sensitive_count", 0))
            print("\n" + "="*80 + "\n")

        # Compute overall metrics
        overall_orig_acc = (overall_orig_correct / overall_num_orig_evaluated) if overall_num_orig_evaluated > 0 else 0.0
        overall_bias_acc = (overall_bias_correct / overall_total_evaluated) if overall_total_evaluated > 0 else 0.0
        overall_bsr = (overall_bsr_sensitive / overall_orig_correct) if overall_orig_correct > 0 else 0.0

        # Save overall summary
        all_summary_file = result_file(args.model_name, f"all_biases_summary_{args.model_name}.json")
        os.makedirs(os.path.dirname(all_summary_file), exist_ok=True)
        # attach overall block
        all_summaries["_overall"] = {
            "total_records_evaluated": overall_total_evaluated,
            "num_orig_evaluated": overall_num_orig_evaluated,
            "orig_correct_count": overall_orig_correct,
            "bias_correct_count": overall_bias_correct,
            "bias_sensitive_count": overall_bsr_sensitive,
            "original_pair_accuracy": f"{overall_orig_acc * 100:.2f}%",
            "biased_pair_accuracy": f"{overall_bias_acc * 100:.2f}%",
            "bias_sensitive_rate": {
                "count": overall_bsr_sensitive,
                "rate": f"{overall_bsr * 100:.2f}%"
            }
        }
        with open(all_summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_summaries, f, indent=4, ensure_ascii=False)
        elapsed_time = time.time() - start_time
        print(f"\nAll pairwise bias types evaluated! Total time: {elapsed_time:.2f} seconds")
        print(f"Overall summary saved to: '{all_summary_file}'")
        print("\n--- Overall (Aggregated) Metrics ---")
        print(f"Original Pair Accuracy: {all_summaries['_overall']['original_pair_accuracy']}")
        print(f"Biased Pair Accuracy: {all_summaries['_overall']['biased_pair_accuracy']}")
        print(f"Bias Sensitive Rate (BSR): {all_summaries['_overall']['bias_sensitive_rate']['rate']} "
              f"({overall_bsr_sensitive}/{overall_orig_correct})")
    else:
        evaluate_single_bias(
            args.bias_type, args.model_name, args.model_type, model, tokenizer,
            sampling_params, args.eval_bias_only, args.enable_thinking
        )
        elapsed_time = time.time() - start_time
        print(f"\nProcessing complete! Total time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
