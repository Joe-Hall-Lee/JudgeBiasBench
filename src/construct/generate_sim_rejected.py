import json
import time
import argparse
import logging
import os
from tqdm import tqdm

from api import query_model

# 配置日志记录
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def generate_answer_with_retry(question: str, model_name: str, max_retries: int = 5, initial_delay: int = 2) -> str:
    """
    带重试逻辑的 API 调用函数，用于生成回答。

    Args:
        question (str): 要回答的问题。
        model_name (str): 要调用的模型名称。
        max_retries (int): 最大重试次数。
        initial_delay (int): 初始重试等待时间（秒）。

    Returns:
        str: 模型的回复内容，如果最终失败则返回 None。
    """
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            # 直接将问题作为 prompt
            response = query_model(question, model_name)
            if response.startswith("Error:"):
                raise Exception(response)
            return response.strip()
        except Exception as e:
            logging.error(
                f"Attempt {attempt + 1} for question '{question[:50]}...' failed: {e}")
            if attempt < max_retries - 1:
                logging.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # 指数退避
            else:
                logging.error(
                    "Max retries reached. Generation failed for this question.")
                return None


def main(input_path: str, output_path: str, model_name: str, limit: int = None):
    """
    主处理函数。

    Args:
        input_path (str): 输入的、已判断相似性的 JSONL 文件路径。
        output_path (str): 最终输出的 JSONL 文件路径。
        model_name (str): 用于生成回答的模型名称。
        limit (int, optional): 处理的最大条目数，用于测试。
    """
    logging.info(
        f"Starting to generate new rejected answers with model: {model_name}")

    # 检查输出文件，确定需要跳过的数量
    num_to_skip = 0
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f_check:
                num_to_skip = sum(1 for line in f_check)
            logging.info(
                f"Output file '{output_path}' found with {num_to_skip} existing entries. Will resume from this point.")
        except Exception as e:
            logging.error(
                f"Could not read existing output file: {e}. Exiting.")
            return

    # 以追加模式打开文件
    file_mode = 'a' if num_to_skip > 0 else 'w'

    # 读取所有输入数据
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            all_data = [json.loads(line) for line in f]
    except FileNotFoundError:
        logging.error(f"Input file not found: {input_path}")
        return

    # 筛选出 is_same 为 'NO' 的数据
    no_data = [item for item in all_data if item.get('is_same') == 'NO']
    logging.info(
        f"Found a total of {len(no_data)} entries where 'is_same' is 'NO'.")

    # 检查是否已全部完成
    if num_to_skip >= len(no_data):
        logging.info(
            "All required entries have already been processed. Nothing to do.")
        return

    # 创建需要处理的数据列表
    data_to_process = no_data[num_to_skip:]
    logging.info(
        f"Skipping {num_to_skip} already processed entries. Now processing {len(data_to_process)} new entries.")

    if limit:
        data_to_process = data_to_process[:limit]
        logging.info(
            f"Processing a limited number of {limit} entries from the remaining set.")

    # 以追加模式打开输出文件并开始处理
    with open(output_path, file_mode, encoding='utf-8') as f_out:
        for item in tqdm(data_to_process, desc="Generating new rejected answers"):
            similar_question = item.get("similar_question")

            if not similar_question:
                logging.warning(
                    f"Skipping item due to missing 'similar_question' field: {item}")
                continue

            # 调用模型回答 similar_question
            new_rejected_answer = generate_answer_with_retry(
                similar_question, model_name)

            # --- 将新的回答插入到 rejected 列表的索引 1 位置 ---
            if new_rejected_answer:
                if not isinstance(item.get('rejected'), list):
                    item['rejected'] = []
                # 使用 insert(index, value)
                item['rejected'].insert(1, new_rejected_answer)
            else:
                if not isinstance(item.get('rejected'), list):
                    item['rejected'] = []
                # 即使失败，也插入到指定位置以保持结构一致
                item['rejected'].insert(1, "GENERATION_FAILED")

            # 写入增强后的数据
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

    logging.info("Processing complete.")
    final_count = num_to_skip + len(data_to_process)
    logging.info(
        f"A total of {final_count} entries have been processed and saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate new 'rejected' answers for instructions judged as not the same.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--input_file', type=str, required=True,
                        help="Path to the input JSONL file.")
    parser.add_argument('--output_file', type=str, required=True,
                        help="Path to save the final output JSONL file.")
    parser.add_argument('--model_name', type=str, default="gemini-2.0-flash",
                        help="The name of the model to use for generating answers.")
    parser.add_argument('--limit', type=int, default=None,
                        help="Limit the number of entries to process (for testing purposes).")

    args = parser.parse_args()
    main(args.input_file, args.output_file, args.model_name, args.limit)
