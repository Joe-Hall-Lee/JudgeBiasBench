import json
import os
import time
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from model_utils import (
    RESOURCE_CONFIG,
    parse_pairwise,
    load_eval_resources,
    needs_resource,
    format_prompt_with_resources,
    parse_judgelm_scores,
    parse_pointwise,
    parse_chateval_scores,
    run_chateval_debate,
    run_calibration_eval
)
from build_prompt import create_eval_prompt

# RewardBench Categories
REWARD_BENCH_CATEGORIES = {
    'Chat': [
        'alpacaeval-easy', 'alpacaeval-length', 'alpacaeval-hard',
        'mt-bench-easy', 'mt-bench-med'
    ],
    'Chat Hard': [
        'mt-bench-hard', 'llmbar-natural', 'llmbar-adver-neighbor',
        'llmbar-adver-GPTInst', 'llmbar-adver-GPTOut', 'llmbar-adver-manual'
    ],
    'Safety': [
        'refusals-dangerous', 'refusals-offensive', 'xstest-should-refuse',
        'xstest-should-respond', 'donotanswer'
    ],
    'Reasoning': [
        'math-prm', 'hep-cpp', 'hep-go', 'hep-js', 'hep-java', 'hep-python', 'hep-rust'
    ]
}

# Reasoning Subsets Split
REASONING_MATH_SUBSETS = {'math-prm'}
REASONING_CODE_SUBSETS = {'hep-cpp', 'hep-go', 'hep-js', 'hep-java', 'hep-python', 'hep-rust'}

SUBSET_TO_CATEGORY = {}
for category, subsets in REWARD_BENCH_CATEGORIES.items():
    for subset in subsets:
        SUBSET_TO_CATEGORY[subset] = category

def evaluate_rewardbench(model_name, model, tokenizer, sampling_params, model_type, enable_thinking=False):
    """
    Evaluates the model on RewardBench dataset using Generative Judge approach.
    """
    input_file = "data/eval/reward_bench/rewardbench.jsonl"
    output_file = f"results/{model_name}-{model_type}/rewardbench_{model_name}.jsonl"
    summary_file = f"results/{model_name}-{model_type}/rewardbench_summary_{model_name}.json"

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return None

    print(f"\nLoading RewardBench data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        source_data = [json.loads(line) for line in f]
    
    print(f"Successfully loaded {len(source_data)} items.")

    # Load resources if needed
    resources = None
    if needs_resource(model_type):
        try:
            resources = load_eval_resources("rewardbench", model_name, model_type)
            resource_type = RESOURCE_CONFIG[model_type]['dir']
            print(f"Loaded {len(resources)} {resource_type}")
        except Exception as e:
            print(f"Warning: Failed to load resources for {model_type}: {e}")
            resources = {}

    # Handle Chat Template
    has_chat_template = tokenizer is not None and tokenizer.chat_template is not None
    
    template_kwargs = {
        'tokenize': False,
        'add_generation_prompt': True
    }
    if 'qwen3' in model_name.lower():
        template_kwargs['enable_thinking'] = enable_thinking

    def format_with_chat_template(prompt_text):
        if not has_chat_template:
            return prompt_text
        messages = [{"role": "user", "content": prompt_text}]
        return tokenizer.apply_chat_template(messages, **template_kwargs)

    results = []

    # --- ChatEval Strategy ---
    if model_type == "chateval":
        print("Evaluating with ChatEval strategy (Multi-round Debate)...")
        
        # Prepare eval items
        eval_items = []
        item_map = []
        
        for idx, item in enumerate(source_data):
            # Position 1: A=Chosen, B=Rejected
            eval_items.append({
                'id': f"{idx}_pos1",
                'question': item['prompt'],
                'answer_a': item['chosen'],
                'answer_b': item['rejected'],
                'correct_pos': 'A'
            })
            item_map.append({'idx': idx, 'pos': 1})
            
            # Position 2: A=Rejected, B=Chosen
            eval_items.append({
                'id': f"{idx}_pos2",
                'question': item['prompt'],
                'answer_a': item['rejected'],
                'answer_b': item['chosen'],
                'correct_pos': 'B'
            })
            item_map.append({'idx': idx, 'pos': 2})
            
        results_list = run_chateval_debate(
            model, tokenizer, sampling_params, eval_items, 
            bias_type=None, enable_thinking=enable_thinking
        )
        
        # Map results back
        temp_results = defaultdict(dict)
        for i, res in enumerate(results_list):
            meta = item_map[i]
            temp_results[meta['idx']][meta['pos']] = res
            
        for idx, item in enumerate(source_data):
            res1 = temp_results[idx][1]['final_vote']
            res2 = temp_results[idx][2]['final_vote']
            
            # Correctness
            is_correct_1 = (res1 == 'A')
            is_correct_2 = (res2 == 'B')
            
            subset = item.get("subset", "Unknown")
            category = SUBSET_TO_CATEGORY.get(subset, "Unknown")
            
            results.append({
                "prompt": item.get("prompt"),
                "chosen": item.get("chosen"),
                "rejected": item.get("rejected"),
                "subset": subset,
                "category": category,
                "position_1": {"correct": is_correct_1, "parsed_result": res1},
                "position_2": {"correct": is_correct_2, "parsed_result": res2},
                "average_accuracy": (int(is_correct_1) + int(is_correct_2)) / 2.0
            })

    # --- Calibration Strategy ---
    elif model_type == "calibration":
        print("Evaluating with Calibration strategy...")
        
        # Prepare eval items
        eval_items = []
        item_map = []
        
        for idx, item in enumerate(source_data):
            # Position 1: A=Chosen, B=Rejected
            eval_items.append({
                'id': f"{idx}_pos1",
                'question': item['prompt'],
                'answer_a': item['chosen'],
                'answer_b': item['rejected']
            })
            item_map.append({'idx': idx, 'pos': 1})
            
            # Position 2: A=Rejected, B=Chosen
            eval_items.append({
                'id': f"{idx}_pos2",
                'question': item['prompt'],
                'answer_a': item['rejected'],
                'answer_b': item['chosen']
            })
            item_map.append({'idx': idx, 'pos': 2})
            
        results_list = run_calibration_eval(
            model, tokenizer, sampling_params, eval_items, 
            resources=resources, enable_thinking=enable_thinking
        )
        
        temp_results = defaultdict(dict)
        for i, res in enumerate(results_list):
            meta = item_map[i]
            temp_results[meta['idx']][meta['pos']] = res
            
        for idx, item in enumerate(source_data):
            res1 = temp_results[idx][1]['final_vote']
            res2 = temp_results[idx][2]['final_vote']
            
            is_correct_1 = (res1 == 'A')
            is_correct_2 = (res2 == 'B')
            
            subset = item.get("subset", "Unknown")
            category = SUBSET_TO_CATEGORY.get(subset, "Unknown")
            
            results.append({
                "prompt": item.get("prompt"),
                "chosen": item.get("chosen"),
                "rejected": item.get("rejected"),
                "subset": subset,
                "category": category,
                "position_1": {"correct": is_correct_1, "parsed_result": res1},
                "position_2": {"correct": is_correct_2, "parsed_result": res2},
                "average_accuracy": (int(is_correct_1) + int(is_correct_2)) / 2.0
            })

    # --- Standard Strategy ---
    else:
        # Prepare Prompts
        eval_prompt_template = create_eval_prompt(scheme="pairwise", model_type=model_type)
        all_prompts = []
        
        print("Preparing prompts...")
        for item in tqdm(source_data):
            question = item.get("prompt")
            chosen = item.get("chosen")
            rejected = item.get("rejected")
            
            prompt_text_1 = format_prompt_with_resources(
                eval_prompt_template, question, resources, model_type,
                answer_a=chosen, answer_b=rejected
            )
            prompt_text_2 = format_prompt_with_resources(
                eval_prompt_template, question, resources, model_type,
                answer_a=rejected, answer_b=chosen
            )
            
            all_prompts.append(format_with_chat_template(prompt_text_1))
            all_prompts.append(format_with_chat_template(prompt_text_2))

        print(f"Sending {len(all_prompts)} prompts to VLLM...")
        vllm_outputs = model.generate(all_prompts, sampling_params)
        print("Generation complete.")

        for idx, item in enumerate(tqdm(source_data, desc="Processing results")):
            output_1 = vllm_outputs[2 * idx].outputs[0].text
            output_2 = vllm_outputs[2 * idx + 1].outputs[0].text
            
            parsed_result_1 = parse_pairwise(output_1, model_type=model_type)
            is_correct_1 = (parsed_result_1 == 'A')
            
            parsed_result_2 = parse_pairwise(output_2, model_type=model_type)
            is_correct_2 = (parsed_result_2 == 'B')
            
            subset = item.get("subset", "Unknown")
            category = SUBSET_TO_CATEGORY.get(subset, "Unknown")
            
            results.append({
                "prompt": item.get("prompt"),
                "chosen": item.get("chosen"),
                "rejected": item.get("rejected"),
                "subset": subset,
                "category": category,
                "position_1": {
                    "raw_output": output_1,
                    "parsed_result": parsed_result_1,
                    "correct": is_correct_1
                },
                "position_2": {
                    "raw_output": output_2,
                    "parsed_result": parsed_result_2,
                    "correct": is_correct_2
                },
                "average_accuracy": (int(is_correct_1) + int(is_correct_2)) / 2.0
            })

    # --- Summary Calculation (Shared) ---
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for record in results:
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"Detailed results saved to {output_file}")

    # Compute Summary
    category_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    subset_stats = defaultdict(lambda: {'total': 0, 'correct': 0})

    for res in results:
        cat = res['category']
        sub = res['subset']
        if cat != "Unknown":
            category_stats[cat]['total'] += 2
            subset_stats[sub]['total'] += 2
            
            if res['position_1']['correct']:
                category_stats[cat]['correct'] += 1
                subset_stats[sub]['correct'] += 1
            
            if res['position_2']['correct']:
                category_stats[cat]['correct'] += 1
                subset_stats[sub]['correct'] += 1

    summary = {
        "model_name": model_name,
        "model_type": model_type,
        "total_samples": len(results),
        "category_results": {}
    }
    
    total_correct = 0
    total_count = 0
    
    print("\n--- RewardBench Evaluation Summary ---")
    for category, stats in category_stats.items():
        if category == 'Reasoning':
             # Special handling for Reasoning: Average of Math and Code
             math_correct = 0
             math_total = 0
             code_correct = 0
             code_total = 0
             
             for sub in REASONING_MATH_SUBSETS:
                 math_correct += subset_stats[sub]['correct']
                 math_total += subset_stats[sub]['total']
             
             for sub in REASONING_CODE_SUBSETS:
                 code_correct += subset_stats[sub]['correct']
                 code_total += subset_stats[sub]['total']
             
             math_acc = math_correct / math_total if math_total > 0 else 0
             code_acc = code_correct / code_total if code_total > 0 else 0
             
             # Average of the two domains
             final_acc = (math_acc + code_acc) / 2
             
             summary['category_results'][category] = {
                "accuracy": f"{final_acc*100:.2f}%",
                "math_accuracy": f"{math_acc*100:.2f}%",
                "code_accuracy": f"{code_acc*100:.2f}%",
                "correct": stats['correct'], 
                "total": stats['total']
            }
             print(f"{category}: {final_acc*100:.2f}% (Math: {math_acc*100:.2f}%, Code: {code_acc*100:.2f}%)")
             
        else:
            acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            summary['category_results'][category] = {
                "accuracy": f"{acc*100:.2f}%",
                "correct": stats['correct'],
                "total": stats['total']
            }
            print(f"{category}: {acc*100:.2f}% ({stats['correct']}/{stats['total']})")
        
        total_correct += stats['correct']
        total_count += stats['total']
        
    overall_acc = total_correct / total_count if total_count > 0 else 0
    summary["overall_accuracy"] = f"{overall_acc*100:.2f}%"
    summary["total_correct"] = total_correct
    
    print(f"Overall Accuracy: {overall_acc*100:.2f}%")
    
    with open(summary_file, 'w', encoding='utf-8') as f_sum:
        json.dump(summary, f_sum, indent=4, ensure_ascii=False)
    print(f"Summary saved to {summary_file}")
    
    return summary
