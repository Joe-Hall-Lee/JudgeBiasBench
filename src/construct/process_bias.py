import json
import os
import time
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from api import query_model
from build_prompt import create_bias_prompt

MAX_WORKERS = 10
API_DELAY_SECONDS = 0.1
REWRITE_BIASES = ['length', 'authority', 'beauty',
                  'assertiveness', 'sycophancy', 'sentiment', 'concreteness']
PREFIX_BIASES = ['gender', 'race']
PASSTHROUGH_BIASES = ['bandwagon', 'superficial-reflection', 'position']
REFINEMENT_BIASES = ['refinement-aware']
ALL_BIASES = REWRITE_BIASES + PREFIX_BIASES + \
    PASSTHROUGH_BIASES + REFINEMENT_BIASES


def process_single_item(item_and_bias_type):
    """
    Processes a single data item to generate a biased version based on the bias type,
    and formats the output with a 'label' field.
    """
    data_item, bias_type = item_and_bias_type
    try:
        original_question = data_item.get("context", [{}])[
            0].get("content", "")
        response1 = data_item.get("response1", "")
        response2 = data_item.get("response2", "")
        preference = data_item.get("overall_preference")

        if not all([original_question, response1, response2, preference is not None]):
            return None

        # Determine the chosen/rejected responses and the corresponding label
        if preference < 0:
            source_chosen = response1
            source_rejected = response2
            label = "response1"
        elif preference > 0:
            source_chosen = response2
            source_rejected = response1
            label = "response2"
        else:  # preference == 0, no clear winner
            return None

        # Initialize rewritten responses with the original ones
        new_chosen = source_chosen
        new_rejected = source_rejected

        # Apply bias modification logic
        if bias_type in REWRITE_BIASES:
            prompt_for_chosen = create_bias_prompt(
                bias_type=bias_type, response_type='chosen')
            formatted_prompt_chosen = prompt_for_chosen.format(
                question=original_question, response=source_chosen)
            new_chosen = query_model(
                formatted_prompt_chosen, "gemini-2.0-flash")
            time.sleep(API_DELAY_SECONDS)

            prompt_for_rejected = create_bias_prompt(
                bias_type=bias_type, response_type='rejected')
            formatted_prompt_rejected = prompt_for_rejected.format(
                question=original_question, response=source_rejected)
            new_rejected = query_model(
                formatted_prompt_rejected, "gemini-2.0-flash")

        elif bias_type in PREFIX_BIASES:
            if bias_type == 'gender':
                prefix = "I am a woman. "
                new_chosen = prefix + source_chosen
            elif bias_type == 'race':
                prefix = "I am black. "
                new_chosen = prefix + source_chosen
            # The rejected response remains unchanged for prefix biases
            new_rejected = source_rejected

        elif bias_type in REFINEMENT_BIASES:
            # Only the rejected response is rewritten
            prompt_for_rejected = create_bias_prompt(
                bias_type=bias_type, response_type='rejected')
            formatted_prompt_rejected = prompt_for_rejected.format(
                question=original_question, response=source_rejected)
            new_rejected = query_model(
                formatted_prompt_rejected, "gemini-2.0-flash")
            # The chosen response remains unchanged
            new_chosen = source_chosen

        elif bias_type in PASSTHROUGH_BIASES:
            # Both responses remain unchanged
            new_chosen = source_chosen
            new_rejected = source_rejected

        if not all([new_chosen, new_rejected]):
            return None

        # Map the rewritten chosen/rejected back to response1/response2
        if label == "response1":
            rewritten_response1 = new_chosen
            rewritten_response2 = new_rejected
        else:  # label == "response2"
            rewritten_response1 = new_rejected
            rewritten_response2 = new_chosen

        # Construct the final output record in the desired format
        output_record = {
            "question": original_question,
            "original_response1": response1,
            "original_response2": response2,
            "rewritten_response1": rewritten_response1,
            "rewritten_response2": rewritten_response2,
            "label": label
        }
        return output_record

    except Exception as e:
        print(f"处理数据时发生错误: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="为不同类型的偏见构建数据集。")
    parser.add_argument("bias_type", type=str, choices=ALL_BIASES,
                        help=f"要生成的偏见类型。可用选项: {ALL_BIASES}")
    args = parser.parse_args()

    bias_type = args.bias_type
    input_file = f"orig_data/{bias_type}_orig.jsonl"
    output_file = f"data/{bias_type}_bias.jsonl"

    if not os.path.exists(input_file):
        print(f"错误: 输入文件 '{input_file}' 未找到。")
        return
    with open(input_file, 'r', encoding='utf-8') as f:
        source_data = [json.loads(line) for line in f]

    print(f"成功加载 {len(source_data)} 条 '{bias_type}' 类型的数据。")

    if bias_type in REWRITE_BIASES:
        print(f"将使用 API 改写模式（双边），配置 {MAX_WORKERS} 个线程并发处理。")
    elif bias_type in PREFIX_BIASES:
        print("将使用本地前缀添加模式处理。")
    elif bias_type in REFINEMENT_BIASES:
        print(f"将使用 API 改写模式（仅 Rejected），配置 {MAX_WORKERS} 个线程并发处理。")
    elif bias_type in PASSTHROUGH_BIASES:
        print("将使用直通模式（无 API 调用，仅转换格式）处理。")

    items_to_process = [(item, bias_type) for item in source_data]
    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results_iterator = executor.map(process_single_item, items_to_process)
        results = [res for res in tqdm(results_iterator, total=len(
            source_data), desc=f"处理 '{bias_type}' 数据中") if res is not None]

    print(f"\n处理完成，成功生成 {len(results)} 条新数据。正在写入文件...")
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for record in results:
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"全部处理完成！改写后的数据已保存至 '{output_file}'。")


if __name__ == "__main__":
    main()
