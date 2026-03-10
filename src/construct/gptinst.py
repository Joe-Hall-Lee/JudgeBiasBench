import json
import time
import argparse
import logging
from tqdm import tqdm

from api import query_model

# 配置日志记录
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 用于生成相似指令的模板
PROMPT_TEMPLATE = """Given an user input (called "given input"), please generate a new user input (called "generated input") such that: 
(1) The generated input is highly relevant to but different from the given input.
(2) The correct response to the generated input superficially resembles the correct response to the given input as much as possible.
(3) But actually, the correct response to the generated input should not be a correct response to the given input.

Given input:
{instruction}

Generated input:"""


def generate_with_retry(instruction: str, model_name: str, max_retries: int = 5, initial_delay: int = 2):

    prompt = PROMPT_TEMPLATE.format(instruction=instruction)
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            response = query_model(prompt, model_name)
            # 检查 query_model 是否返回了错误字符串
            if response.startswith("Error:"):
                raise Exception(response)

            # 成功获取响应
            return response.strip()
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                logging.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # 指数退避
            else:
                logging.error(
                    "Max retries reached. Skipping this instruction.")
                return None


def main(input_path: str, output_path: str, model_name: str, limit: int = None):
    """
    主处理函数。

    Args:
        input_path (str): 输入的 JSONL 文件路径。
        output_path (str): 输出的 JSONL 文件路径。
        model_name (str): 用于生成内容的模型名称。
        limit (int, optional): 处理的最大条目数，用于测试。
    """
    logging.info(f"Starting generation process with model: {model_name}")

    # 读取输入文件
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            all_data = [json.loads(line) for line in f]
    except FileNotFoundError:
        logging.error(f"Input file not found: {input_path}")
        return
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from file: {input_path}")
        return

    # 如果设置了限制，则截取部分数据
    if limit:
        all_data = all_data[:limit]
        logging.info(f"Processing a limited number of {limit} entries.")

    # 打开输出文件并开始处理
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for item in tqdm(all_data, desc="Generating similar questions"):
            original_question = item.get("question")
            if not original_question:
                logging.warning(
                    f"Skipping item due to missing 'question' field: {item}")
                continue

            # 生成相似问题
            similar_question = generate_with_retry(
                original_question, model_name)

            # 更新数据项
            if similar_question:
                item['similar_question'] = similar_question
            else:
                item['similar_question'] = "GENERATION_FAILED"

            # 写入新文件
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

    logging.info(f"Processing complete. Output saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate similar but tricky instructions based on an existing dataset.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help="Path to the input JSONL file."
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help="Path to save the output JSONL file with the new 'similar_question' field."
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default="gemini-2.0-flash",
        help="The name of the mode l to use for generation."
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help="Limit the number of entries to process (for testing purposes)."
    )

    args = parser.parse_args()
    main(args.input_file, args.output_file, args.model_name, args.limit)
