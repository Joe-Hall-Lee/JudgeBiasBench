import json
import time
import argparse
import logging
from tqdm import tqdm

from api import query_model

# 配置日志记录
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 用于判断指令是否相同的模板
JUDGE_PROMPT_TEMPLATE = """There are two instructions, Instruction A and Instruction B. Are the two instructions asking the same thing? Please answer in `YES` or `NO`.

# Instruction A: {instruction_A}

# Instruction B: {instruction_B}

# Are the two instructions asking the same thing?"""


def parse_judge_response(response: str) -> str:
    """
    解析模型的回复，严格提取 'YES' 或 'NO'。

    Args:
        response (str): 模型的原始回复字符串。

    Returns:
        str: 'YES', 'NO', 或 'UNKNOWN'。
    """
    if not response:
        return "UNKNOWN"

    cleaned_response = response.strip().upper()

    # 优先判断否定的情况，因为它更关键
    if 'NO' in cleaned_response:
        return 'NO'
    if 'YES' in cleaned_response:
        return 'YES'

    logging.warning(
        f"Could not parse 'YES' or 'NO' from response: '{response}'")
    return "UNKNOWN"


def judge_with_retry(instruction_A: str, instruction_B: str, model_name: str, max_retries: int = 5, initial_delay: int = 2) -> str:
    """
    带重试逻辑的 API 调用函数，用于判断两个指令是否相同。

    Args:
        instruction_A (str): 第一个指令。
        instruction_B (str): 第二个指令。
        model_name (str): 要调用的模型名称。
        max_retries (int): 最大重试次数。
        initial_delay (int): 初始重试等待时间（秒）。

    Returns:
        str: 'YES', 'NO', 或 'FAILED'。
    """
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        instruction_A=instruction_A, instruction_B=instruction_B)
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            response = query_model(prompt, model_name)
            if response.startswith("Error:"):
                raise Exception(response)

            parsed_result = parse_judge_response(response)
            if parsed_result in ['YES', 'NO']:
                return parsed_result
            else:
                # 如果解析失败，也视为一种需要重试的错误
                raise ValueError(
                    f"Failed to parse valid YES/NO from response: {response}")

        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                logging.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # 指数退避
            else:
                logging.error(
                    "Max retries reached. Judging failed for this pair.")
                return "FAILED"


def main(input_path: str, output_path: str, model_name: str, limit: int = None):
    """
    主处理函数。

    Args:
        input_path (str): 输入的 JSONL 文件路径。
        output_path (str): 输出的 JSONL 文件路径。
        model_name (str): 用于判断的模型名称。
        limit (int, optional): 处理的最大条目数，用于测试。
    """
    logging.info(f"Starting judging process with model: {model_name}")

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            all_data = [json.loads(line) for line in f]
    except FileNotFoundError:
        logging.error(f"Input file not found: {input_path}")
        return

    if limit:
        all_data = all_data[:limit]
        logging.info(f"Processing a limited number of {limit} entries.")

    # 初始化统计计数器
    stats = {'YES': 0, 'NO': 0, 'FAILED': 0, 'SKIPPED': 0}

    with open(output_path, 'w', encoding='utf-8') as f_out:
        for item in tqdm(all_data, desc="Judging instruction similarity"):
            original_question = item.get("question")
            similar_question = item.get("similar_question")

            # 跳过无效数据
            if not all or not similar_question or similar_question == "GENERATION_FAILED":
                item['is_same'] = 'SKIPPED'
                stats['SKIPPED'] += 1
            else:
                # 调用 API 进行判断
                judgement = judge_with_retry(
                    original_question, similar_question, model_name)
                item['is_same'] = judgement
                stats[judgement] += 1

            # 写入新文件
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

    logging.info("Processing complete.")
    logging.info("--- Final Statistics ---")
    total_processed = len(all_data)
    logging.info(f"Total items processed: {total_processed}")
    for key, value in stats.items():
        percentage = (value / total_processed) * \
            100 if total_processed > 0 else 0
        logging.info(f"{key}: {value} ({percentage:.2f}%)")
    logging.info(f"Output saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Judge if original and similar instructions are the same, and add the result to a new file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--input_file', type=str, required=True,
                        help="Path to the input JSONL file with similar_question.")
    parser.add_argument('--output_file', type=str, required=True,
                        help="Path to save the output JSONL file with the 'is_same' field.")
    parser.add_argument('--model_name', type=str, default="gemini-2.0-flash",
                        help="The name of the model to use for judging.")
    parser.add_argument('--limit', type=int, default=None,
                        help="Limit the number of entries to process (for testing purposes).")

    args = parser.parse_args()
    main(args.input_file, args.output_file, args.model_name, args.limit)
