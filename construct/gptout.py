import json
import random
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from build_prompt import create_gptout_prompt
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # repo root
from common.api import query_model

# 支持的 bias 类型
BIAS_TYPES = [
    'length', 'authority', 'beauty',
    'assertiveness', 'sycophancy', 'sentiment', 'concreteness',
    'gender', 'race'
]


def generate_with_retry(prompt, model_name, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = query_model(prompt, model_name)
            if not response or response.startswith("Error:"):
                raise Exception(response)
            return response.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                continue
            else:
                return f"GENERATION_FAILED: {str(e)}"


def generate_bias_answer(bias_type, question_text, model_name):
    """Generate a single bias answer for the given question and bias type."""
    prompt_template = create_gptout_prompt(bias_type)
    prompt_filled = prompt_template.format(question=question_text)
    return generate_with_retry(prompt_filled, model_name), bias_type


def process_batch_items(batch_items, model_name, pbar=None):
    """Process a batch of items with their pre-selected biases using ThreadPoolExecutor."""
    # 收集所有需要生成的任务
    generation_tasks = []
    for item_idx, item, selected_biases in batch_items:
        question_text = item.get("question")
        if question_text:
            for bias_type in selected_biases:
                generation_tasks.append((item_idx, bias_type, question_text))
    
    # 使用 ThreadPoolExecutor 并行处理所有生成任务
    results = {}
    completed_items = set()  # 跟踪已完成的item
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        # 提交所有生成任务
        future_to_task = {
            executor.submit(generate_bias_answer, bias_type, question_text, model_name): (item_idx, bias_type)
            for item_idx, bias_type, question_text in generation_tasks
        }
        
        # 收集结果
        for future in future_to_task:
            item_idx, bias_type = future_to_task[future]
            try:
                answer, returned_bias_type = future.result()
                if item_idx not in results:
                    results[item_idx] = []
                results[item_idx].append((answer, returned_bias_type))
                
                # 检查该item的所有bias是否都已完成
                item_biases = [biases for idx, item, biases in batch_items if idx == item_idx]
                if item_biases and len(results[item_idx]) == len(item_biases[0]):
                    # 该item的所有bias都已完成，更新进度条
                    if pbar and item_idx not in completed_items:
                        pbar.update(1)
                        completed_items.add(item_idx)
                        
            except Exception as e:
                print(f"Error generating bias answer for item {item_idx}, bias {bias_type}: {e}")
                if item_idx not in results:
                    results[item_idx] = []
                results[item_idx].append((f"GENERATION_FAILED: {str(e)}", bias_type))
                
                # 检查该item的所有bias是否都已完成（包括失败的）
                item_biases = [biases for idx, item, biases in batch_items if idx == item_idx]
                if item_biases and len(results[item_idx]) == len(item_biases[0]):
                    # 该item的所有bias都已完成，更新进度条
                    if pbar and item_idx not in completed_items:
                        pbar.update(1)
                        completed_items.add(item_idx)
    
    return results


def inject_bias(input_path, output_path, model_name, limit=None, batch_size=100):
    # 第一步：检查输出文件是否存在，如果存在则读取已处理的行数
    processed_count = 0
    processed_items = []
    
    try:
        with open(output_path, 'r', encoding='utf-8') as f_out_check:
            for line in f_out_check:
                if line.strip():
                    processed_items.append(json.loads(line))
                    processed_count += 1
        print(f"Found existing output file with {processed_count} processed items. Resuming from line {processed_count}...")
    except FileNotFoundError:
        print("No existing output file found. Starting from the beginning...")
    
    # 第二步：读取所有数据并预选择biases
    print("Step 1: Loading data and pre-selecting biases...")
    all_items = []
    
    with open(input_path, 'r', encoding='utf-8') as f_in:
        for idx, line in enumerate(f_in):
            if limit and idx >= limit:
                break
            
            item = json.loads(line)
            
            # 如果这一行已经处理过，直接使用已处理的数据
            if idx < processed_count:
                all_items.append((idx, processed_items[idx], []))  # 已处理的项目，不需要再生成
                continue
            
            question_text = item.get("question")
            if not question_text:
                all_items.append((idx, item, []))  # 没有问题的项目，不生成bias
                continue

            # 从所有 bias 类型中随机选择 4 个
            selected_biases = random.sample(BIAS_TYPES, 4)
            all_items.append((idx, item, selected_biases))

    total_items = len(all_items)
    print(f"Loaded {total_items} items, {processed_count} already processed, {total_items - processed_count} remaining")
    print(f"Processing in batches of {batch_size}")

    # 第三步：按批次处理数据（追加模式）
    file_mode = 'a' if processed_count > 0 else 'w'
    
    # 创建总体进度条，显示所有样本的处理进度
    with tqdm(total=total_items, initial=processed_count, desc="Processing samples") as pbar:
        with open(output_path, file_mode, encoding='utf-8') as f_out:
            # 计算需要处理的批次范围
            start_batch = processed_count // batch_size
            if processed_count % batch_size != 0:
                start_batch += 1
            
            for batch_idx in range(start_batch, (total_items + batch_size - 1) // batch_size):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, total_items)
                batch_items = all_items[batch_start:batch_end]
                
                # 分离需要生成bias的项目和不需要生成的项目
                items_to_process = [(idx, item, biases) for idx, item, biases in batch_items if biases and idx >= processed_count]
                
                # 处理需要生成bias的项目
                if items_to_process:
                    pbar.set_description(f"Processing batch {batch_idx + 1}: {len(items_to_process)} items need bias generation")
                    batch_results = process_batch_items(items_to_process, model_name, pbar)
                    
                    # 更新项目数据
                    for item_idx, item, selected_biases in items_to_process:
                        if item_idx in batch_results:
                            # 确保 rejected 是列表
                            if not isinstance(item.get("rejected"), list):
                                item["rejected"] = []
                            
                            # 确保 bias_type 是列表
                            if not isinstance(item.get("bias_type"), list):
                                item["bias_type"] = []
                            
                            # 添加生成的结果
                            for answer, bias_type in batch_results[item_idx]:
                                item["rejected"].append(answer)
                                item["bias_type"].append(bias_type)
                
                # 写入当前批次的所有项目到文件（只写入未处理的项目）
                for item_idx, item, _ in batch_items:
                    if item_idx >= processed_count:
                        f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                
                # 对于没有bias生成需求的项目，也需要更新进度条
                items_without_bias = [(idx, item, biases) for idx, item, biases in batch_items if not biases and idx >= processed_count]
                if items_without_bias:
                    pbar.update(len(items_without_bias))
                
                pbar.set_description(f"Processing samples")

    print(f"Bias injection complete. Output saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inject bias answers into JSONL file using LLM")
    parser.add_argument("--input_file", type=str,
                        required=True)
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to save new file with bias answers")
    parser.add_argument("--model_name", type=str,
                        default="gemini-2.0-flash", help="Model name to use")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of entries to process")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Batch size for processing items (default: 100)")
    args = parser.parse_args()

    inject_bias(args.input_file, args.output_file, args.model_name, args.limit, args.batch_size)
