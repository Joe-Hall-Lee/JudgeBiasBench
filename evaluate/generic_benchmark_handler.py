import json
import os
from tqdm import tqdm
from collections import defaultdict
from model_utils import benchmark_input_file, result_file, parse_pairwise
from build_prompt import create_eval_prompt

# --- Constants ---

JUDGEBENCH_CATEGORIES = {
    "Knowledge": [
        "mmlu-pro-biology", "mmlu-pro-business", "mmlu-pro-chemistry",
        "mmlu-pro-computer science", "mmlu-pro-economics", "mmlu-pro-engineering",
        "mmlu-pro-health", "mmlu-pro-history", "mmlu-pro-law", "mmlu-pro-math",
        "mmlu-pro-other", "mmlu-pro-philosophy", "mmlu-pro-physics", "mmlu-pro-psychology"
    ],
    "Reasoning": ["livebench-reasoning"],
    "Math (JudgeBench)": ["livebench-math"],
    "Coding": ["livecodebench"]
}

RMB_CATEGORIES = {
    "Harmlessness": [
        "Violent Crimes", "Non-Violent Crimes", "Sex-Related Crimes",
        "Child Sexual Exploitation", "Specialized Advice", "Privacy",
        "Intellectual Property", "Indiscriminate Weapons", "Hate",
        "Suicide & Self-Harm", "Sexual Content", "Multi"
    ],
    "Helpfulness": [
        "Brainstorming", "Chat", "Classification", "Closed QA", "Code",
        "Generation", "Open QA", "Reasoning", "Rewrite", "Role Playing",
        "Summarization", "Translation"
    ]
}

def get_subset_mapping(categories):
    mapping = {}
    for cat, subsets in categories.items():
        for sub in subsets:
            mapping[sub] = cat
    return mapping

# --- Generic Evaluator ---

def evaluate_generic_benchmark(
    benchmark_name,
    input_file,
    categories,
    model_name,
    model,
    tokenizer,
    sampling_params,
    model_type,
    enable_thinking=False
):
    """
    Generic evaluation logic for RewardBench-style datasets (Prompt, Chosen, Rejected)
    with dual-position (swap) evaluation.
    """
    output_file = result_file(model_name, f"{benchmark_name}_{model_name}.jsonl")
    summary_file = result_file(model_name, f"{benchmark_name}_summary_{model_name}.json")
    
    subset_to_category = get_subset_mapping(categories)

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return None

    print(f"\nLoading {benchmark_name} data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        source_data = [json.loads(line) for line in f]
    
    print(f"Successfully loaded {len(source_data)} items.")

    # Handle Chat Template
    has_chat_template = tokenizer is not None and tokenizer.chat_template is not None
    template_kwargs = {'tokenize': False, 'add_generation_prompt': True}
    if 'qwen3' in model_name.lower():
        template_kwargs['enable_thinking'] = enable_thinking

    def format_with_chat_template(prompt_text):
        if not has_chat_template:
            return prompt_text
        messages = [{"role": "user", "content": prompt_text}]
        return tokenizer.apply_chat_template(messages, **template_kwargs)

    results = []
    eval_prompt_template = create_eval_prompt(model_type)
    all_prompts = []
    
    for item in tqdm(source_data):
        question = item.get("prompt")
        chosen = item.get("chosen")
        rejected = item.get("rejected")
        
        prompt_text_1 = eval_prompt_template.format(question=question, answer_a=chosen, answer_b=rejected)
        prompt_text_2 = eval_prompt_template.format(question=question, answer_a=rejected, answer_b=chosen)
        
        all_prompts.append(format_with_chat_template(prompt_text_1))
        all_prompts.append(format_with_chat_template(prompt_text_2))
        
    print(f"Sending {len(all_prompts)} prompts to VLLM...")
    vllm_outputs = model.generate(all_prompts, sampling_params)
    
    temp_results = defaultdict(dict)
    for idx in range(len(source_data)):
        output_1 = vllm_outputs[2 * idx].outputs[0].text
        output_2 = vllm_outputs[2 * idx + 1].outputs[0].text
        
        parsed_1 = parse_pairwise(output_1, model_type=model_type)
        parsed_2 = parse_pairwise(output_2, model_type=model_type)
        
        temp_results[idx][1] = {'final_vote': parsed_1, 'raw_output': output_1}
        temp_results[idx][2] = {'final_vote': parsed_2, 'raw_output': output_2}

    # --- Aggregate Results ---
    for idx, item in enumerate(source_data):
        res1_data = temp_results[idx][1]
        res2_data = temp_results[idx][2]
        
        res1 = res1_data.get('final_vote')
        res2 = res2_data.get('final_vote')
        
        is_correct_1 = (res1 == 'A')
        is_correct_2 = (res2 == 'B')
        
        subset = item.get("subset", "Unknown")
        category = subset_to_category.get(subset, "Unknown")
        
        record = {
            "prompt": item.get("prompt"),
            "chosen": item.get("chosen"),
            "rejected": item.get("rejected"),
            "subset": subset,
            "category": category,
            "position_1": {
                "correct": is_correct_1, 
                "parsed_result": res1,
                "raw_output": res1_data.get('raw_output')
            },
            "position_2": {
                "correct": is_correct_2, 
                "parsed_result": res2,
                "raw_output": res2_data.get('raw_output')
            },
            "average_accuracy": (int(is_correct_1) + int(is_correct_2)) / 2.0
        }
        results.append(record)

    # --- Save Results ---
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for record in results:
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"Detailed results saved to {output_file}")

    # --- Compute Summary ---
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
        "benchmark": benchmark_name,
        "total_items": len(source_data),
        "overall_accuracy": 0.0,
        "category_accuracies": {},
        "subset_accuracies": {}
    }

    # Overall
    total_correct = sum(s['correct'] for s in category_stats.values())
    total_count = sum(s['total'] for s in category_stats.values())
    summary['overall_accuracy'] = total_correct / total_count if total_count > 0 else 0

    # By Category
    for cat, stats in category_stats.items():
        summary['category_accuracies'][cat] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0

    # By Subset
    for sub, stats in subset_stats.items():
        summary['subset_accuracies'][sub] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        
    # Save Summary
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)
        
    print(f"Summary saved to {summary_file}")
    print(f"Overall Accuracy: {summary['overall_accuracy']:.2%}")
    return summary

# --- Specific Entry Points ---

def evaluate_judgebench(model_name, model, tokenizer, sampling_params, model_type, enable_thinking=False):
    return evaluate_generic_benchmark(
        benchmark_name="judgebench",
        input_file=benchmark_input_file("judgebench"),
        categories=JUDGEBENCH_CATEGORIES,
        model_name=model_name,
        model=model,
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        model_type=model_type,
        enable_thinking=enable_thinking
    )

def evaluate_rmb(model_name, model, tokenizer, sampling_params, model_type, enable_thinking=False):
    return evaluate_generic_benchmark(
        benchmark_name="rmb",
        input_file=benchmark_input_file("rmb_pairwise"),
        categories=RMB_CATEGORIES,
        model_name=model_name,
        model=model,
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        model_type=model_type,
        enable_thinking=enable_thinking
    )
