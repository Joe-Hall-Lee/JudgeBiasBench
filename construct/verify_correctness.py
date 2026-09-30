import json
import time
import argparse
import logging
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # repo root
from common.api import query_model

# 配置日志记录
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 用于选择最佳回答的模板
CHOOSE_BEST_RESPONSE_TEMPLATE = """Given a question and multiple candidate responses, please choose the best response that most accurately and helpfully answers the question.

# Question: {question}

# Candidate Responses:
{candidates}

Please respond with only the letter (A, B, C, etc.) of the best response."""


def parse_best_choice_response(response: str) -> str:
    """
    解析模型的回复，提取选择的字母 (A, B, C, etc.)。
    """
    if not response:
        return "UNKNOWN"
    cleaned_response = response.strip().upper()
    
    # 查找单个字母选择
    import re
    match = re.search(r'\b([A-Z])\b', cleaned_response)
    if match:
        return match.group(1)
    
    logging.warning(
        f"Could not parse letter choice from response: '{response}'")
    return "UNKNOWN"


def choose_best_response_with_retry(question: str, chosen: str, rejected_list: list, model_name: str, max_retries: int = 5, initial_delay: int = 2) -> str:
    """
    带重试逻辑的 API 调用函数，用于从候选回答中选择最佳回答。
    """
    # 构建候选回答列表，chosen 作为选项 A
    candidates = [chosen] + rejected_list
    candidates_text = ""
    for i, candidate in enumerate(candidates):
        letter = chr(ord('A') + i)
        candidates_text += f"{letter}: {candidate}\n\n"
    
    prompt = CHOOSE_BEST_RESPONSE_TEMPLATE.format(
        question=question, candidates=candidates_text.strip())
    
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            api_response = query_model(prompt, model_name)
            if api_response.startswith("Error:"):
                raise Exception(api_response)
            parsed_result = parse_best_choice_response(api_response)
            if parsed_result != "UNKNOWN":
                return parsed_result
            else:
                raise ValueError(
                    f"Failed to parse valid letter choice from response: {api_response}")
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                logging.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # 指数退避
            else:
                logging.error(
                    "Max retries reached. Choosing best response failed.")
                return "FAILED"


def verify_single_item(item_idx, item, model_name):
    """验证单个样本的正确性"""
    question = item.get("question")
    chosen = item.get("chosen")
    rejected_list = item.get("rejected")

    if not question or not chosen or not isinstance(rejected_list, list) or len(rejected_list) == 0:
        return item_idx, "SKIPPED"

    # 选择最佳回答
    best_choice = choose_best_response_with_retry(
        question, chosen, rejected_list, model_name)
    
    if best_choice == "FAILED":
        return item_idx, "FAILED"
    elif best_choice == "A":  # chosen 是选项 A
        return item_idx, "CHOSEN_IS_BEST"
    else:
        return item_idx, "CHOSEN_NOT_BEST"


def process_batch_items(batch_items, model_name, pbar=None):
    """使用多线程处理一批验证任务"""
    results = {}
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        # 提交所有验证任务
        future_to_item = {
            executor.submit(verify_single_item, item_idx, item, model_name): item_idx
            for item_idx, item in batch_items
        }
        
        # 收集结果
        for future in future_to_item:
            item_idx = future_to_item[future]
            try:
                idx, verification_result = future.result()
                results[idx] = verification_result
                
                # 更新进度条
                if pbar:
                    pbar.update(1)
                    
            except Exception as e:
                logging.error(f"Error verifying item {item_idx}: {e}")
                results[item_idx] = "FAILED"
                if pbar:
                    pbar.update(1)
    
    return results


def verify_correctness_batch(input_path: str, output_path: str, model_name: str, limit: int = None, batch_size: int = 100):
    """
    主处理函数，支持断点续传和批处理。
    """
    logging.info(f"Starting batch verification with model: {model_name}")
    
    # 第一步：检查输出文件是否存在，如果存在则读取已处理的行数
    processed_count = 0
    processed_items = []
    
    try:
        with open(output_path, 'r', encoding='utf-8') as f_out_check:
            for line in f_out_check:
                if line.strip():
                    processed_items.append(json.loads(line))
                    processed_count += 1
        logging.info(f"Found existing output file with {processed_count} processed items. Resuming from line {processed_count}...")
    except FileNotFoundError:
        logging.info("No existing output file found. Starting from the beginning...")
    
    # 第二步：读取所有数据
    logging.info("Loading data...")
    all_items = []
    
    with open(input_path, 'r', encoding='utf-8') as f_in:
        for idx, line in enumerate(f_in):
            if limit and idx >= limit:
                break
            
            item = json.loads(line)
            
            # 如果这一行已经处理过，直接使用已处理的数据
            if idx < processed_count:
                all_items.append((idx, processed_items[idx]))
                continue
            
            all_items.append((idx, item))

    total_items = len(all_items)
    logging.info(f"Loaded {total_items} items, {processed_count} already processed, {total_items - processed_count} remaining")
    logging.info(f"Processing in batches of {batch_size}")

    # 第三步：按批次处理数据
    file_mode = 'a' if processed_count > 0 else 'w'
    
    stats = {'CHOSEN_IS_BEST': 0, 'CHOSEN_NOT_BEST': 0, 'FAILED': 0, 'SKIPPED': 0}
    
    # 创建总体进度条，显示所有样本的处理进度
    with tqdm(total=total_items, initial=processed_count, desc="Verifying samples") as pbar:
        with open(output_path, file_mode, encoding='utf-8') as f_out:
            # 计算需要处理的批次范围
            start_batch = processed_count // batch_size
            if processed_count % batch_size != 0:
                start_batch += 1
            
            for batch_idx in range(start_batch, (total_items + batch_size - 1) // batch_size):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, total_items)
                batch_items = all_items[batch_start:batch_end]
                
                # 分离需要处理的项目和已处理的项目
                items_to_process = [(idx, item) for idx, item in batch_items if idx >= processed_count]
                items_already_processed = [(idx, item) for idx, item in batch_items if idx < processed_count]
                
                # 处理需要验证的项目
                batch_results = {}
                if items_to_process:
                    pbar.set_description(f"Processing batch {batch_idx + 1}: {len(items_to_process)} items")
                    batch_results = process_batch_items(items_to_process, model_name, pbar)
                
                # 更新已处理项目的统计（从文件中读取的）
                for idx, item in items_already_processed:
                    verify_result = item.get('verify_correctness', 'UNKNOWN')
                    if verify_result in stats:
                        stats[verify_result] += 1
                    if pbar.initial <= idx:  # 避免重复计数
                        pbar.update(1)
                
                # 写入当前批次的所有项目到文件（只写入未处理的项目）
                for item_idx, item in items_to_process:
                    if item_idx in batch_results:
                        # 添加验证结果字段
                        item['verify_correctness'] = batch_results[item_idx]
                        stats[batch_results[item_idx]] += 1
                    else:
                        item['verify_correctness'] = 'FAILED'
                        stats['FAILED'] += 1
                    
                    f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                
                pbar.set_description("Verifying samples")

    logging.info("Processing complete.")
    logging.info("--- Final Statistics ---")
    total_processed = sum(stats.values())
    logging.info(f"Total items processed: {total_processed}")
    for key, value in stats.items():
        percentage = (value / total_processed) * 100 if total_processed > 0 else 0
        logging.info(f"{key}: {value} ({percentage:.2f}%)")
    logging.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify correctness by selecting the best response from chosen and rejected answers, with batch processing and resume support.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--input_file', type=str, required=True,
                        help="Path to the input JSONL file.")
    parser.add_argument('--output_file', type=str, required=True,
                        help="Path to save the output JSONL file with verification results.")
    parser.add_argument('--model_name', type=str, default="gemini-2.0-flash",
                        help="The name of the model to use for judging.")
    parser.add_argument('--limit', type=int, default=None,
                        help="Limit the number of entries to process (for testing purposes).")
    parser.add_argument('--batch_size', type=int, default=100,
                        help="Batch size for processing items (default: 100).")

    args = parser.parse_args()
    verify_correctness_batch(args.input_file, args.output_file, args.model_name, args.limit, args.batch_size)
