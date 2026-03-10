#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm

# ======================
# 配置区
# ======================

# 输入：包含多个 jsonl 文件的目录
DATA_DIR = "data/eval"

# 输出目录
OUTPUT_DIR = "responses"

# 单一模型
MODEL_NAME = "qwen3-14b"
MODEL_PATH = "models/Qwen3-14B"

# 模型参数
MAX_MODEL_LEN = 8192
MAX_NEW_TOKENS = 1024


# ======================
# 工具函数
# ======================

def load_prompts_from_jsonl(file_path):
    """从单个 JSONL 文件加载 prompts"""
    prompts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if "question" in data:
                    prompts.append(data["question"])
                else:
                    print(f"警告：缺少 question 字段，跳过：{line.strip()}")
            except json.JSONDecodeError as e:
                print(f"JSON 解析失败：{e}")
    return prompts


def format_and_truncate_prompt(
    prompt: str,
    tokenizer,
    max_model_len: int,
    max_new_tokens: int,
):
    """
    apply_chat_template + token 级截断
    """
    messages = [{"role": "user", "content": prompt}]

    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    input_ids = tokenizer(
        formatted,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]

    max_prompt_len = max_model_len - max_new_tokens
    if max_prompt_len <= 0:
        raise ValueError("max_model_len 太小，无法生成")

    if len(input_ids) > max_prompt_len:
        input_ids = input_ids[-max_prompt_len:]

    return tokenizer.decode(
        input_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


# ======================
# 推理主逻辑
# ======================

def generate_for_file(llm, tokenizer, jsonl_path):
    """对单个 jsonl 文件生成 responses"""
    prompts = load_prompts_from_jsonl(jsonl_path)
    if not prompts:
        print(f"跳过空文件：{jsonl_path}")
        return

    formatted_prompts = [
        format_and_truncate_prompt(
            prompt,
            tokenizer,
            MAX_MODEL_LEN,
            MAX_NEW_TOKENS,
        )
        for prompt in prompts
    ]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_NEW_TOKENS,
    )

    outputs = llm.generate(formatted_prompts, sampling_params)

    results = []
    for i, output in enumerate(tqdm(outputs, desc=f"生成 {os.path.basename(jsonl_path)}")):
        results.append({
            "question": prompts[i],
            "model": MODEL_NAME,
            "response": output.outputs[0].text.strip(),
        })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_name = os.path.basename(jsonl_path).replace(
        ".jsonl", f"_{MODEL_NAME}.jsonl")
    out_path = os.path.join(OUTPUT_DIR, out_name)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"已保存：{out_path}")


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"模型路径不存在：{MODEL_PATH}")
        return

    if not os.path.isdir(DATA_DIR):
        print(f"输入目录不存在：{DATA_DIR}")
        return

    print(f"加载模型：{MODEL_NAME}")
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        max_model_len=MAX_MODEL_LEN,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )

    jsonl_files = [
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.endswith(".jsonl")
    ]

    if not jsonl_files:
        print("目录下没有 jsonl 文件")
        return

    print(f"发现 {len(jsonl_files)} 个 jsonl 文件")

    for jsonl_path in jsonl_files:
        generate_for_file(llm, tokenizer, jsonl_path)


if __name__ == "__main__":
    main()
