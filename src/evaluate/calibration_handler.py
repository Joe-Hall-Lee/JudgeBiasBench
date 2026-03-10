# -*- coding: utf-8 -*-
import os
import json
from tqdm import tqdm
from build_prompt import create_eval_prompt
from model_utils import parse_judgelm_scores, parse_pointwise

def evaluate_calibration_bias(bias_type, model_name, model, tokenizer, sampling_params, eval_bias_only=False, enable_thinking=False):
    """
    实现 calibration 评估策略：
    1. 先用 JudgeLM 的评估 prompt 给两个 response 打分
    2. 再分别让模型给两个 response 的表面质量打分
    3. 用前者分数减去后者乘以 0.8 作为最终分数
    """
    input_file = f"data/eval/{bias_type}_bias.jsonl"
    output_file = f"results/{model_name}-calibration/{bias_type}_{model_name}.jsonl"
    summary_file = f"results/{model_name}-calibration/{bias_type}_summary_{model_name}.json"

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return None

    with open(input_file, 'r', encoding='utf-8') as f:
        source_data = [json.loads(line) for line in f]

    print(f"\nEvaluating bias type: {bias_type} with calibration strategy")
    print(f"Successfully loaded {len(source_data)} items.")

    all_prompts = []
    
    has_chat_template = tokenizer is not None and tokenizer.chat_template is not None

    # 准备聊天模板参数
    template_kwargs = {
        'tokenize': False,
        'add_generation_prompt': True
    }
    if 'qwen3' in model_name.lower():
        template_kwargs['enable_thinking'] = enable_thinking

    def format_with_chat_template(prompt_text, conversation_history=None):
        if not has_chat_template:
            return prompt_text
        if conversation_history:
            messages = conversation_history
        else:
            messages = [{"role": "user", "content": prompt_text}]
        return tokenizer.apply_chat_template(messages, **template_kwargs)

    # 第一步：生成 JudgeLM 评估 prompt
    judgelm_template = create_eval_prompt(
        scheme="pairwise", model_type="judgelm")

    # 第二步：生成表面质量评估 prompt
    surface_template = create_eval_prompt(
        scheme="pointwise", model_type="calibration")

    for item in tqdm(source_data, desc=f"Generating calibration prompts for {bias_type}"):
        question = item.get("question")

        if bias_type == 'position':
            label_is_response1 = item.get("label") == "response1"
            response1 = item.get(
                "response1") if label_is_response1 else item.get("response2")
            response2 = item.get(
                "response2") if label_is_response1 else item.get("response1")

            # JudgeLM 评估 prompt（交换位置以测试位置偏见）
            judgelm_prompt_orig = judgelm_template.format(
                question=question,
                question_body=question,
                answer1_body=response1,
                answer2_body=response2
            )
            judgelm_prompt_bias = judgelm_template.format(
                question=question,
                question_body=question,
                answer1_body=response2,
                answer2_body=response1
            )

            # 表面质量评估 prompt
            surface_prompt1 = surface_template.format(
                answer=response1)
            surface_prompt2 = surface_template.format(
                answer=response2)

        else:
            response1_orig = item.get(
                "original_response1") or item.get("response1")
            response2_orig = item.get(
                "original_response2") or item.get("response2")
            response1_bias = item.get("rewritten_response1") or response1_orig
            response2_bias = item.get("rewritten_response2") or response2_orig

            # JudgeLM 评估 prompt
            judgelm_prompt_orig = judgelm_template.format(
                question=question,
                question_body=question,
                answer1_body=response1_orig,
                answer2_body=response2_orig
            )
            judgelm_prompt_bias = judgelm_template.format(
                question=question,
                question_body=question,
                answer1_body=response1_bias,
                answer2_body=response2_bias
            )

            # 表面质量评估 prompt
            surface_prompt1 = surface_template.format(
                answer=response1_bias)
            surface_prompt2 = surface_template.format(
                answer=response2_bias)

        # 应用聊天模板
        judgelm_prompt_orig = format_with_chat_template(judgelm_prompt_orig)
        judgelm_prompt_bias = format_with_chat_template(judgelm_prompt_bias)
        surface_prompt1 = format_with_chat_template(surface_prompt1)
        surface_prompt2 = format_with_chat_template(surface_prompt2)

        if bias_type == 'position':
            if not eval_bias_only:
                # Reuse surface scores for both positions
                all_prompts.extend([judgelm_prompt_orig, judgelm_prompt_bias, surface_prompt1, surface_prompt2])
            else:
                all_prompts.extend([judgelm_prompt_bias, surface_prompt1, surface_prompt2])
        else:
            if not eval_bias_only:
                all_prompts.extend(
                    [judgelm_prompt_orig, surface_prompt1, surface_prompt2])
            all_prompts.extend(
                [judgelm_prompt_bias, surface_prompt1, surface_prompt2])

    print(f"Sending {len(all_prompts)} prompts to VLLM for generation...")
    vllm_outputs = model.generate(all_prompts, sampling_params)
    print("VLLM generation complete.")

    # 处理结果
    results = []
    output_idx = 0

    for item in tqdm(source_data, desc=f"Processing calibration results for {bias_type}"):
        try:
            question = item.get("question")

            if bias_type == 'position':
                label_is_response1 = item.get("label") == "response1"
                response1 = item.get(
                    "response1") if label_is_response1 else item.get("response2")
                response2 = item.get(
                    "response2") if label_is_response1 else item.get("response1")

                if not eval_bias_only:
                    # 处理原始评估
                    judgelm_orig_result = vllm_outputs[output_idx].outputs[0].text
                    
                    # For position bias optimization:
                    # If not eval_bias_only, we have [JudgeOrig, JudgeBias, Surf1, Surf2]
                    judgelm_bias_result = vllm_outputs[output_idx + 1].outputs[0].text
                    surface1_result = vllm_outputs[output_idx + 2].outputs[0].text
                    surface2_result = vllm_outputs[output_idx + 3].outputs[0].text
                    output_idx += 4

                    judgelm_orig_scores = parse_judgelm_scores(judgelm_orig_result)
                    judgelm_bias_scores = parse_judgelm_scores(judgelm_bias_result)
                    
                    surface1_score = parse_pointwise(surface1_result, model_type="default")
                    surface2_score = parse_pointwise(surface2_result, model_type="default")

                    # 计算 calibration 分数 (Orig: A=R1, B=R2)
                    if judgelm_orig_scores and surface1_score is not None and surface2_score is not None:
                        calibrated_score1 = judgelm_orig_scores[0] - surface1_score * 0.8
                        calibrated_score2 = judgelm_orig_scores[1] - surface2_score * 0.8
                        orig_winner = "A" if calibrated_score1 > calibrated_score2 else "B"
                    else:
                        orig_winner = "Error"
                        
                    # 计算 calibration 分数 (Bias: A=R2, B=R1)
                    if judgelm_bias_scores and surface1_score is not None and surface2_score is not None:
                        # Bias A=R2 (use s2), Bias B=R1 (use s1)
                        calibrated_score1 = judgelm_bias_scores[0] - surface2_score * 0.8
                        calibrated_score2 = judgelm_bias_scores[1] - surface1_score * 0.8
                        bias_winner = "A" if calibrated_score1 > calibrated_score2 else "B"
                    else:
                        bias_winner = "Error"
                        
                else:
                    # eval_bias_only=True: [JudgeBias, Surf1, Surf2]
                    orig_winner = "skipped"
                    
                    judgelm_bias_result = vllm_outputs[output_idx].outputs[0].text
                    surface1_result = vllm_outputs[output_idx + 1].outputs[0].text
                    surface2_result = vllm_outputs[output_idx + 2].outputs[0].text
                    output_idx += 3

                    judgelm_bias_scores = parse_judgelm_scores(judgelm_bias_result)
                    surface1_score = parse_pointwise(surface1_result, model_type="default")
                    surface2_score = parse_pointwise(surface2_result, model_type="default")

                    # 计算 calibration 分数 (Bias: A=R2, B=R1)
                    if judgelm_bias_scores and surface1_score is not None and surface2_score is not None:
                        # Bias A=R2 (use s2), Bias B=R1 (use s1)
                        calibrated_score1 = judgelm_bias_scores[0] - surface2_score * 0.8
                        calibrated_score2 = judgelm_bias_scores[1] - surface1_score * 0.8
                        bias_winner = "A" if calibrated_score1 > calibrated_score2 else "B"
                    else:
                        bias_winner = "Error"
                
                # Use shared results for saving
                judgelm_orig_result = judgelm_orig_result if not eval_bias_only else "skipped"
                surface1_orig_result = surface1_result
                surface2_orig_result = surface2_result
                judgelm_bias_result = judgelm_bias_result
                surface1_bias_result = surface1_result
                surface2_bias_result = surface2_result

                # 确定正确性

                # 确定正确性
                is_orig_eval_correct = (orig_winner == 'A') if orig_winner not in [
                    "Error", "skipped", "Tie"] else orig_winner
                is_bias_eval_correct = "eval_error" if bias_winner in [
                    "Error", "Tie"] else (bias_winner == 'B')

                results.append({
                    "question": question,
                    "response1": response1,
                    "response2": response2,
                    "label": item.get("label"),
                    "original_evaluation_correct": is_orig_eval_correct,
                    "bias_evaluation_correct": is_bias_eval_correct,
                    "raw_judgelm_original": judgelm_orig_result if not eval_bias_only else "skipped",
                    "raw_surface1_original": surface1_orig_result if not eval_bias_only else "skipped",
                    "raw_surface2_original": surface2_orig_result if not eval_bias_only else "skipped",
                    "raw_judgelm_bias": judgelm_bias_result,
                    "raw_surface1_bias": surface1_bias_result,
                    "raw_surface2_bias": surface2_bias_result,
                })

            else:
                # 处理其他偏见类型
                response1_orig = item.get(
                    "original_response1") or item.get("response1")
                response2_orig = item.get(
                    "original_response2") or item.get("response2")
                response1_bias = item.get(
                    "rewritten_response1") or response1_orig
                response2_bias = item.get(
                    "rewritten_response2") or response2_orig
                label = item.get("label")
                correct_choice = 'A' if label == "response1" else 'B'

                if not eval_bias_only:
                    # 处理原始评估
                    judgelm_orig_result = vllm_outputs[output_idx].outputs[0].text
                    surface1_orig_result = vllm_outputs[output_idx +
                                                        1].outputs[0].text
                    surface2_orig_result = vllm_outputs[output_idx +
                                                        2].outputs[0].text
                    output_idx += 3

                    judgelm_orig_scores = parse_judgelm_scores(
                        judgelm_orig_result)
                    surface1_orig_score = parse_pointwise(
                        surface1_orig_result, model_type="default")
                    surface2_orig_score = parse_pointwise(
                        surface2_orig_result, model_type="default")

                    if judgelm_orig_scores and surface1_orig_score is not None and surface2_orig_score is not None:
                        calibrated_score1 = judgelm_orig_scores[0] - \
                            surface1_orig_score * 0.8
                        calibrated_score2 = judgelm_orig_scores[1] - \
                            surface2_orig_score * 0.8
                        orig_winner = "A" if calibrated_score1 > calibrated_score2 else "B"
                    else:
                        orig_winner = "Error"
                else:
                    orig_winner = "skipped"

                # 处理偏见评估
                judgelm_bias_result = vllm_outputs[output_idx].outputs[0].text
                surface1_bias_result = vllm_outputs[output_idx +
                                                    1].outputs[0].text
                surface2_bias_result = vllm_outputs[output_idx +
                                                    2].outputs[0].text
                output_idx += 3

                judgelm_bias_scores = parse_judgelm_scores(judgelm_bias_result)
                surface1_bias_score = parse_pointwise(
                    surface1_bias_result, model_type="default")
                surface2_bias_score = parse_pointwise(
                    surface2_bias_result, model_type="default")

                if judgelm_bias_scores and surface1_bias_score is not None and surface2_bias_score is not None:
                    calibrated_score1 = judgelm_bias_scores[0] - \
                        surface1_bias_score * 0.8
                    calibrated_score2 = judgelm_bias_scores[1] - \
                        surface2_bias_score * 0.8
                    bias_winner = "A" if calibrated_score1 > calibrated_score2 else "B"
                else:
                    bias_winner = "Error"

                # 确定正确性
                is_orig_eval_correct = (orig_winner == correct_choice) if orig_winner not in [
                    "Error", "skipped", "Tie"] else orig_winner
                is_bias_eval_correct = "eval_error" if bias_winner in [
                    "Error", "Tie"] else (bias_winner == correct_choice)

                results.append({
                    "question": question,
                    "original_response1": response1_orig,
                    "original_response2": response2_orig,
                    "rewritten_response1": response1_bias,
                    "rewritten_response2": response2_bias,
                    "label": label,
                    "original_evaluation_correct": is_orig_eval_correct,
                    "bias_evaluation_correct": is_bias_eval_correct,
                    "raw_judgelm_original": judgelm_orig_result if not eval_bias_only else "skipped",
                    "raw_surface1_original": surface1_orig_result if not eval_bias_only else "skipped",
                    "raw_surface2_original": surface2_orig_result if not eval_bias_only else "skipped",
                    "raw_judgelm_bias": judgelm_bias_result,
                    "raw_surface1_bias": surface1_bias_result,
                    "raw_surface2_bias": surface2_bias_result,
                })

        except Exception as e:
            print(
                f"处理 calibration 结果时出错: {e} | 项目: {item.get('question', 'N/A')}")

    # 保存结果
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for record in results:
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"Detailed calibration evaluation results saved to: '{output_file}'")

    # 计算汇总统计
    summary_stats = compute_calibration_summary(
        results, bias_type, model_name, eval_bias_only)

    if summary_stats:
        with open(summary_file, 'w', encoding='utf-8') as f_summary:
            json.dump(summary_stats, f_summary, indent=4, ensure_ascii=False)
        print(f"Calibration summary statistics saved to: '{summary_file}'")

    return summary_stats


def compute_calibration_summary(results, bias_type, model_name, eval_bias_only):
    """计算calibration评估的汇总统计"""
    total_evaluated = len(results)
    if total_evaluated == 0:
        return None

    if eval_bias_only:
        bias_correct_count = sum(
            1 for r in results if r['bias_evaluation_correct'] is True
        )
        bias_accuracy = bias_correct_count / total_evaluated if total_evaluated > 0 else 0

        summary_stats = {
            "eval_model": model_name,
            "bias_type": bias_type,
            "evaluation_strategy": "calibration",
            "total_records_evaluated": total_evaluated,
            "biased_pair_accuracy": f"{bias_accuracy * 100:.2f}%"
        }

        print("\n--- Calibration Evaluation Summary (Biased Only) ---")
        print(f"Biased Pair Accuracy: {summary_stats['biased_pair_accuracy']}")
        return summary_stats

    # 完整评估
    num_orig_evaluated = total_evaluated - sum(
        1 for r in results if r['original_evaluation_correct'] == 'skipped'
    )

    orig_correct_count = sum(
        1 for r in results if r['original_evaluation_correct'] is True
    )
    bias_correct_count = sum(
        1 for r in results if r['bias_evaluation_correct'] is True
    )

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
        "bias_type": bias_type,
        "evaluation_strategy": "calibration",
        "total_records_evaluated": total_evaluated,
        "original_pair_accuracy": f"{orig_accuracy * 100:.2f}%",
        "biased_pair_accuracy": f"{bias_accuracy * 100:.2f}%",
        "accuracy_drop_due_to_bias": f"{accuracy_drop * 100:.2f}%",
        "bias_sensitive_rate": {
            "count": bias_sensitive_count,
            "rate": f"{bias_sensitive_rate * 100:.2f}%"
        }
    }

    print("\n--- Calibration Evaluation Summary ---")
    print(f"Original Pair Accuracy: {summary_stats['original_pair_accuracy']}")
    print(f"Biased Pair Accuracy: {summary_stats['biased_pair_accuracy']}")
    print(
        f"Accuracy Drop due to Bias: {summary_stats['accuracy_drop_due_to_bias']}")
    print(f"Bias Sensitive Rate (BSR): {summary_stats['bias_sensitive_rate']['rate']} "
          f"({bias_sensitive_count}/{orig_correct_count})")

    return summary_stats
