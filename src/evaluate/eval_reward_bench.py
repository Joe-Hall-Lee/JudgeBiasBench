import argparse
import json
import os
import time
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from collections import defaultdict

# --- Constants ---
REWARD_BENCH_CATEGORIES = {
    'Chat': [
        'alpacaeval-easy', 'alpacaeval-length', 'alpacaeval-hard',
        'mt-bench-easy', 'mt-bench-medium'
    ],
    'Chat Hard': [
        'mt-bench-hard', 'llmbar-natural', 'llmbar-adver-neighbor',
        'llmbar-adver-gptinst', 'llmbar-adver-gptout', 'llmbar-adver-manual'
    ],
    'Safety': [
        'refusals-dangerous', 'refusals-offensive', 'xstest-should-refuse',
        'xstest-should-respond', 'donotanswer'
    ],
    'Reasoning': [
        'prm-math', 'hep-cpp', 'hep-go', 'hep-js', 'hep-java', 'hep-python', 'hep-rust'
    ]
}

# Create reverse mapping from subset to category
SUBSET_TO_CATEGORY = {}
for category, subsets in REWARD_BENCH_CATEGORIES.items():
    for subset in subsets:
        SUBSET_TO_CATEGORY[subset] = category

def load_reward_bench_data():
    """
    Load the reward-bench dataset and organize by categories.
    """
    print("Loading reward-bench dataset...")
    ds = load_dataset("allenai/reward-bench")
    data = ds['filtered']
    
    # Group data by category
    category_data = defaultdict(list)
    subset_counts = defaultdict(int)
    
    for item in data:
        subset = item['subset']
        subset_counts[subset] += 1
        
        # Map subset to main category
        category = SUBSET_TO_CATEGORY.get(subset, 'Unknown')
        if category != 'Unknown':
            category_data[category].append(item)
    
    print(f"Loaded {len(data)} samples from reward-bench dataset")
    print("\nSubset distribution:")
    for subset, count in sorted(subset_counts.items()):
        category = SUBSET_TO_CATEGORY.get(subset, 'Unknown')
        print(f"  {subset}: {count} samples -> {category}")
    
    print("\nCategory distribution:")
    for category, items in category_data.items():
        print(f"  {category}: {len(items)} samples")
    
    return category_data

def evaluate_reward_bench_category(category, data, model, tokenizer, model_name, batch_size=32):
    """
    Evaluate a single category using pairwise comparison.
    """
    print(f"\n正在评估类别: {category}")
    print(f"样本数量: {len(data)}")
    
    # Prepare all text pairs using the tokenizer's chat template
    all_texts_to_score = []
    print("正在使用聊天模板准备输入...")
    
    for item in tqdm(data, desc=f"格式化输入 ({category})"):
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]
        
        # Create dialogues for chosen and rejected responses
        dialogues = [
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": chosen}],
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": rejected}]
        ]
        
        formatted_texts = [
            tokenizer.apply_chat_template(
                dialogue, tokenize=False, add_generation_prompt=False)
            for dialogue in dialogues
        ]
        all_texts_to_score.extend(formatted_texts)
    
    # Run RM inference in batches
    all_scores = []
    print(f"正在使用奖励模型为 {len(all_texts_to_score)} 个文本打分...")
    for i in tqdm(range(0, len(all_texts_to_score), batch_size), desc=f"评估 {category}"):
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
            scores = outputs.logits.squeeze(-1).cpu().tolist()
            all_scores.extend(scores)
    
    # Process results and calculate accuracy
    results = []
    correct_count = 0
    
    for i, item in enumerate(data):
        score_idx = i * 2
        chosen_score, rejected_score = all_scores[score_idx: score_idx + 2]
        
        # The model should prefer chosen over rejected
        is_correct = chosen_score > rejected_score
        if is_correct:
            correct_count += 1
        
        results.append({
            "category": category,
            "subset": item["subset"],
            "prompt": item["prompt"],
            "chosen": item["chosen"],
            "rejected": item["rejected"],
            "chosen_score": chosen_score,
            "rejected_score": rejected_score,
            "correct": is_correct,
            "id": item.get("id", "")
        })
    
    accuracy = correct_count / len(data) if len(data) > 0 else 0.0
    
    category_summary = {
        "category": category,
        "total_samples": len(data),
        "correct_predictions": correct_count,
        "accuracy": f"{accuracy * 100:.2f}%"
    }
    
    print(f"类别 {category} 准确率: {category_summary['accuracy']}")
    
    return results, category_summary

def main():
    parser = argparse.ArgumentParser(description="使用奖励模型评估 Reward-Bench 数据集。")
    parser.add_argument("category", type=str, 
                        choices=list(REWARD_BENCH_CATEGORIES.keys()) + ['all'],
                        help=f"要评估的类别。可用选项: {list(REWARD_BENCH_CATEGORIES.keys()) + ['all']}")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Hugging Face 模型目录的路径 (必须是 ForSequenceClassification 模型)。")
    parser.add_argument("--model_name", type=str,
                        help="模型的名称，用于创建结果目录。默认为 model_path 的最后一部分。")
    parser.add_argument("--batch_size", type=int,
                        default=32, help="推理时的批处理大小。")
    
    args = parser.parse_args()
    
    if not args.model_name:
        args.model_name = os.path.basename(args.model_path)
    
    # Load Model and Tokenizer
    print(f"正在从以下路径加载奖励模型: {args.model_path}")
    print("注意：将启用模型并行，自动将模型分配到所有可用的 GPU 上。")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        model.eval()
        print("模型和分词器加载成功，并已分配到多个 GPU。")
    except Exception as e:
        print(f"从路径加载模型失败: {args.model_path}")
        print(f"错误: {e}")
        return
    
    # Load reward-bench data
    category_data = load_reward_bench_data()
    
    # Prepare output files
    output_file = f"results/{args.model_name}/reward_bench_evaluation_{args.model_name}.jsonl"
    summary_file = f"results/{args.model_name}/reward_bench_summary_{args.model_name}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    start_time = time.time()
    all_results = []
    all_summaries = {}
    
    if args.category == 'all':
        print(f"将开始评估所有类别: {list(REWARD_BENCH_CATEGORIES.keys())}")
        
        for category in REWARD_BENCH_CATEGORIES.keys():
            if category in category_data:
                results, summary = evaluate_reward_bench_category(
                    category, category_data[category], model, tokenizer, 
                    args.model_name, args.batch_size
                )
                all_results.extend(results)
                all_summaries[category] = summary
                print("\n" + "="*80 + "\n")
        
        # Calculate weighted average accuracy
        total_samples = sum(summary['total_samples'] for summary in all_summaries.values())
        total_correct = sum(summary['correct_predictions'] for summary in all_summaries.values())
        weighted_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        # Create overall summary
        overall_summary = {
            "model_name": args.model_name,
            "total_samples": total_samples,
            "total_correct": total_correct,
            "weighted_average_accuracy": f"{weighted_accuracy * 100:.2f}%",
            "category_results": all_summaries
        }
        
        print(f"\n--- Reward-Bench 总体评估摘要 ---")
        print(f"总样本数: {total_samples}")
        print(f"总正确数: {total_correct}")
        print(f"加权平均准确率: {overall_summary['weighted_average_accuracy']}")
        print(f"\n各类别准确率:")
        for category, summary in all_summaries.items():
            print(f"  {category}: {summary['accuracy']} ({summary['total_samples']} samples)")
        
    else:
        # Evaluate single category
        if args.category in category_data:
            results, summary = evaluate_reward_bench_category(
                args.category, category_data[args.category], model, tokenizer,
                args.model_name, args.batch_size
            )
            all_results.extend(results)
            all_summaries[args.category] = summary
            
            overall_summary = {
                "model_name": args.model_name,
                "category": args.category,
                "total_samples": summary['total_samples'],
                "total_correct": summary['correct_predictions'],
                "accuracy": summary['accuracy']
            }
        else:
            print(f"错误：类别 '{args.category}' 在数据集中未找到。")
            return
    
    # Save detailed results
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for record in all_results:
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"\n详细评估结果已保存至: '{output_file}'")
    
    # Save summary
    with open(summary_file, 'w', encoding='utf-8') as f_summary:
        json.dump(overall_summary, f_summary, indent=4, ensure_ascii=False)
    print(f"统计摘要已保存至: '{summary_file}'")
    
    elapsed_time = time.time() - start_time
    print(f"\n🎉 处理完成！总耗时: {elapsed_time:.2f} 秒")

if __name__ == "__main__":
    main()
