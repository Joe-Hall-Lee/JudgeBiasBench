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

# 包含各个 bias 的 jsonl
DATA_DIR = "data/eval"

# metrics 输出目录
MODEL_NAME = "qwen2.5-7b"
MODEL_PATH = "/root/autodl-tmp/RewardBiasBench/models/Qwen2.5-7B-Instruct"

# 模型参数
MAX_MODEL_LEN = 8192
MAX_NEW_TOKENS = 256

# ======================
# Prompt 模板
# ======================

METRICS_PROMPT_TEMPLATE = """You are a helpful assistant in evaluating the quality of the outputs for a given instruction.
Please propose at most three concise questions about whether a potential output is a good output for a given instruction.
Another assistant will evaluate different aspects of the output by answering all the questions.

Here are some rules of the evaluation:
(1) You should prioritize evaluating whether the output honestly, precisely, and closely executes the instruction.
(2) Outputs should NOT contain more or less than what the instruction asks for.

Instruction:
{instruction}

Requirements for Your Output:
(1) The questions should specifically target the given instruction instead of general standards.
(2) Directly give the questions without any other words.
(3) Questions are presented from most important to least important.
"""

# ======================
# 工具函数
# ======================


def load_questions_from_jsonl(file_path):
    """从 jsonl 文件中读取 question 字段"""
    questions = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if "question" in data:
                questions.append(data["question"])
    return questions


def format_prompt(prompt_text, tokenizer):
    """统一使用 chat template"""
    messages = [{"role": "user", "content": prompt_text}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

# ======================
# 主逻辑
# ======================


def generate_metrics_for_file(llm, tokenizer, jsonl_path):
    bias_name = os.path.basename(jsonl_path).replace(".jsonl", "")
    questions = load_questions_from_jsonl(jsonl_path)

    if not questions:
        print(f"跳过空文件：{jsonl_path}")
        return

    prompts = []
    for q in questions:
        prompt_text = METRICS_PROMPT_TEMPLATE.format(instruction=q)
        prompts.append(format_prompt(prompt_text, tokenizer))

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_NEW_TOKENS,
    )

    print(f"生成 metrics：{bias_name}（{len(prompts)} 条）")
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for i, output in enumerate(tqdm(outputs, desc=f"metrics {bias_name}")):
        results.append({
            "question": questions[i],
            "metrics": output.outputs[0].text.strip()
        })

    out_dir = os.path.join(
        os.path.dirname(jsonl_path),
        "metrics"
    )
    os.makedirs(out_dir, exist_ok=True)

    base_name = os.path.basename(jsonl_path).replace(".jsonl", "")
    out_name = f"{base_name}_{MODEL_NAME}.jsonl"
    out_path = os.path.join(out_dir, out_name)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"已保存 metrics：{out_path}")


def main():
    if not os.path.isdir(DATA_DIR):
        print(f"DATA_DIR 不存在：{DATA_DIR}")
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
        print("未发现 jsonl 文件")
        return

    print(f"发现 {len(jsonl_files)} 个 bias 数据集")

    for jsonl_path in jsonl_files:
        generate_metrics_for_file(llm, tokenizer, jsonl_path)


if __name__ == "__main__":
    main()
