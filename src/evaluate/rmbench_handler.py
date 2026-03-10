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

# RM-Bench Categories
RMBENCH_CATEGORIES = {
    'Chat': [
        "Chat_Easy", "Chat_Normal", "Chat_Hard"
    ],
    'Math': [
        "Math_Easy", "Math_Normal", "Math_Hard"
    ],
    'Code': [
        "Code_Easy", "Code_Normal", "Code_Hard"
    ],
    'Safety': [
        "Safety_Easy", "Safety_Normal", "Safety_Hard"
    ]
}

# Difficulty Levels for aggregation
DIFFICULTY_LEVELS = {
    "Easy": ["Chat_Easy", "Math_Easy", "Code_Easy", "Safety_Easy"],
    "Normal": ["Chat_Normal", "Math_Normal", "Code_Normal", "Safety_Normal"],
    "Hard": ["Chat_Hard", "Math_Hard", "Code_Hard", "Safety_Hard"]
}

SUBSET_TO_CATEGORY = {}
for category, subsets in RMBENCH_CATEGORIES.items():
    for subset in subsets:
        SUBSET_TO_CATEGORY[subset] = category

def evaluate_rmbench(model_name, model, tokenizer, sampling_params, model_type, enable_thinking=False):
    """
    Evaluates the model on RM-Bench dataset using Generative Judge approach.
    """
    input_file = "data/eval/reward_bench/rm-bench.jsonl"
    output_file = f"results/{model_name}-{model_type}/rmbench_{model_name}.jsonl"
    summary_file = f"results/{model_name}-{model_type}/rmbench_summary_{model_name}.json"

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return None

    print(f"\nLoading RM-Bench data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        source_data = [json.loads(line) for line in f]
    
    print(f"Successfully loaded {len(source_data)} items.")

    # Load resources if needed
    resources = None
    if needs_resource(model_type):
        try:
            resources = load_eval_resources("rmbench", model_name, model_type)
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
            # Pos 1: A=Chosen, B=Rejected
            eval_items.append({
                'id': f"{idx}_pos1",
                'question': item['prompt'],
                'answer_a': item['chosen'],
                'answer_b': item['rejected'],
                'correct_pos': 'A'
            })
            item_map.append({'idx': idx, 'pos': 1})
            
            # Pos 2: A=Rejected, B=Chosen
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
        
        temp_results = defaultdict(dict)
        for i, res in enumerate(results_list):
            meta = item_map[i]
            temp_results[meta['idx']][meta['pos']] = res
            
        for idx, item in enumerate(source_data):
            res1_obj = temp_results[idx][1]
            res2_obj = temp_results[idx][2]
            
            res1 = res1_obj['final_vote']
            res2 = res2_obj['final_vote']
            
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
                "position_1": {
                    "raw_output": res1_obj['history'], 
                    "parsed_result": res1_obj['votes'], 
                    "correct": is_correct_1
                },
                "position_2": {
                    "raw_output": res2_obj['history'],
                    "parsed_result": res2_obj['votes'],
                    "correct": is_correct_2
                },
                "average_accuracy": (int(is_correct_1) + int(is_correct_2)) / 2.0
            })

    # --- Calibration Strategy ---
    elif model_type == "calibration":
        print("Evaluating with Calibration strategy (JudgeLM + Surface Quality)...")
        
        eval_items = []
        item_map = []
        
        for idx, item in enumerate(source_data):
            # Pos 1
            eval_items.append({
                'id': f"{idx}_pos1",
                'question': item['prompt'],
                'answer_a': item['chosen'],
                'answer_b': item['rejected']
            })
            item_map.append({'idx': idx, 'pos': 1})
            
            # Pos 2
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
            res1_obj = temp_results[idx][1]
            res2_obj = temp_results[idx][2]
            
            res1 = res1_obj['final_vote']
            res2 = res2_obj['final_vote']
            
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
                "position_1": {
                    "raw_output": "See summary",
                    "parsed_result": res1_obj, 
                    "correct": is_correct_1
                },
                "position_2": {
                    "raw_output": "See summary",
                    "parsed_result": res2_obj,
                    "correct": is_correct_2
                },
                "average_accuracy": (int(is_correct_1) + int(is_correct_2)) / 2.0
            })

    # --- Standard Strategy (Pairwise) ---
    else:
        # Prepare Prompts
        eval_prompt_template = create_eval_prompt(scheme="pairwise", model_type=model_type)
        all_prompts = []
        
        print("Preparing prompts...")
        for item in tqdm(source_data):
            question = item.get("prompt")
            chosen = item.get("chosen")
            rejected = item.get("rejected")
            
            # Position 1: A = Chosen, B = Rejected (Correct: A)
            prompt_text_1 = format_prompt_with_resources(
                eval_prompt_template, question, resources, model_type,
                answer_a=chosen, answer_b=rejected
            )
            
            # Position 2: A = Rejected, B = Chosen (Correct: B)
            prompt_text_2 = format_prompt_with_resources(
                eval_prompt_template, question, resources, model_type,
                answer_a=rejected, answer_b=chosen
            )
            
            all_prompts.append(format_with_chat_template(prompt_text_1))
            all_prompts.append(format_with_chat_template(prompt_text_2))

        # Generate
        print(f"Sending {len(all_prompts)} prompts to VLLM...")
        vllm_outputs = model.generate(all_prompts, sampling_params)
        print("Generation complete.")

        # Process Results
        for idx, item in enumerate(tqdm(source_data, desc="Processing results")):
            # Get outputs for both positions
            output_1 = vllm_outputs[2 * idx].outputs[0].text
            output_2 = vllm_outputs[2 * idx + 1].outputs[0].text
            
            # Parse results
            # Position 1: Expect 'A' (Chosen)
            parsed_result_1 = parse_pairwise(output_1, model_type=model_type)
            is_correct_1 = (parsed_result_1 == 'A')
            
            # Position 2: Expect 'B' (Chosen was B)
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

    # Save detailed results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for record in results:
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"Detailed results saved to {output_file}")

    # Compute Summary
    # Aggregate stats by category, subset, and difficulty
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
        "category_results": {},
        "difficulty_results": {}
    }
    
    total_correct = 0
    total_count = 0
    
    print("\n--- RM-Bench Evaluation Summary ---")
    
    # 1. Category Accuracy (Aggregated over all items in category)
    for category, stats in category_stats.items():
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        summary['category_results'][category] = {
            "accuracy": f"{acc*100:.2f}%",
            "correct": stats['correct'],
            "total": stats['total']
        }
        print(f"{category}: {acc*100:.2f}% ({stats['correct']}/{stats['total']})")
        
        total_correct += stats['correct']
        total_count += stats['total']

    # 2. Difficulty Accuracy (Macro-average of subsets)
    # Easy = (Safety_Easy + Math_Easy + Chat_Easy + Code_Easy) / 4
    print("\n--- Difficulty Breakdown ---")
    for difficulty, subsets in DIFFICULTY_LEVELS.items():
        subset_accuracies = []
        for sub in subsets:
            stats = subset_stats[sub]
            sub_acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            subset_accuracies.append(sub_acc)
        
        avg_acc = sum(subset_accuracies) / len(subset_accuracies) if subset_accuracies else 0
        summary['difficulty_results'][difficulty] = f"{avg_acc*100:.2f}%"
        print(f"{difficulty}: {avg_acc*100:.2f}%")

    overall_acc = total_correct / total_count if total_count > 0 else 0
    summary["overall_accuracy"] = f"{overall_acc*100:.2f}%"
    summary["total_correct"] = total_correct
    
    print(f"\nOverall Accuracy: {overall_acc*100:.2f}%")
    
    with open(summary_file, 'w', encoding='utf-8') as f_sum:
        json.dump(summary, f_sum, indent=4, ensure_ascii=False)
    print(f"Summary saved to {summary_file}")
    
    return summary
