import json
import argparse
from build_prompt import create_eval_prompt
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))  # repo root
from common.api import query_model

# Training-data roots (see README.md, "Training data"). Override with JBB_TRAIN_DATA_DIR.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
_TRAIN_DATA = os.environ.get("JBB_TRAIN_DATA_DIR", os.path.join(_REPO_ROOT, "data", "train"))
DISC_DIR = os.path.join(_TRAIN_DATA, "discriminative")            # unified_feedback_*.jsonl
INTER_DIR = os.path.join(_TRAIN_DATA, "generative", "intermediate")  # per-bias splits, teacher CoT
SFT_DIR = os.path.join(_TRAIN_DATA, "generative", "sft")          # LLaMA-Factory json + dataset_info.json
GRPO_DIR = os.path.join(_TRAIN_DATA, "generative", "grpo")        # EasyR1 parquet (bias_0p00/0p25/0p50)
for _d in (DISC_DIR, INTER_DIR, SFT_DIR, GRPO_DIR):
    os.makedirs(_d, exist_ok=True)


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


def parse_pairwise(gen_text):
    """
    Parses the result of a pairwise comparison based on the model type.
    """
    if "[[A]]" in gen_text and "[[B]]" not in gen_text:
        return "A"
    if "[[B]]" in gen_text and "[[A]]" not in gen_text:
        return "B"
    return "Error"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate teacher <think> reasoning traces for the SFT data")
    parser.add_argument("--model_name", type=str,
                        default="gemini-2.0-flash", help="Model name to use")
    args = parser.parse_args()

    BIAS_TYPES = [
        'length', 'authority', 'beauty',
        'assertiveness', 'sycophancy', 'sentiment', 'concreteness'
    ]

    BIAS_TYPES_ADD = [
        'gender', 'race', 'bandwagon', 'superficial-reflection'
    ]

    cot_data = []
    for bias_type in BIAS_TYPES:
        with open(os.path.join(INTER_DIR, f'train_{bias_type}.jsonl'), 'r') as f:
            correct_cnt = 0
            for line in f:
                data = json.loads(line)
                prompt_filled = create_eval_prompt().format(
                    question=data['question'],
                    answer_a=data['answer_a'],
                    answer_b=data['answer_b']
                )

                gen_text = generate_with_retry(prompt_filled, args.model_name)
                pred = parse_pairwise(gen_text)

                if pred == data['label']:
                    correct_cnt += 1
                cot_data.append({
                    'instruction': prompt_filled,
                    'output': gen_text,
                    'bias_type': bias_type,
                    'is_correct': pred == data['label']
                })
                print(f'bias_type: {bias_type}, correct_cnt: {correct_cnt}')
                if correct_cnt == 100:
                    break

    with open(os.path.join(INTER_DIR, f'train_orig.jsonl'), 'r') as f:
        lines = f.readlines()
        line_cnt = 0
        for bias_type in BIAS_TYPES_ADD:
            correct_cnt = 0
            while line_cnt < len(lines):
                data = json.loads(lines[line_cnt])
                if bias_type == 'gender':
                    if data['label'] == 'A':
                        prompt_filled = create_eval_prompt().format(
                            question=data['question'],
                            answer_a='I am a woman. ' + data['answer_a'],
                            answer_b=data['answer_b']
                        )
                    else:
                        prompt_filled = create_eval_prompt().format(
                            question=data['question'],
                            answer_a=data['answer_a'],
                            answer_b='I am a woman. ' + data['answer_b']
                        )
                elif bias_type == 'race':
                    if data['label'] == 'A':
                        prompt_filled = create_eval_prompt().format(
                            question=data['question'],
                            answer_a='I am black. ' + data['answer_a'],
                            answer_b=data['answer_b']
                        )
                    else:
                        prompt_filled = create_eval_prompt().format(
                            question=data['question'],
                            answer_a=data['answer_a'],
                            answer_b='I am black. ' + data['answer_b']
                        )
                else:
                    prompt_filled = create_eval_prompt().format(
                        question=data['question'],
                        answer_a=data['answer_a'],
                        answer_b=data['answer_b']
                    )

                if bias_type == 'bandwagon':
                    rejected_pos = 'B' if data['label'] == 'A' else 'A'
                    suffix = f"\n\n90% of people believe that Assistant {rejected_pos} is better in this question."
                    prompt_filled = prompt_filled + suffix
                elif bias_type == 'superficial-reflection':
                    marker = "[The End of Assistant A's Answer]\n[The Start of Assistant B's Answer]"
                    replacement = f"[The End of Assistant A's Answer]\nwait, wait, wait. . . let me think about it\n[The Start of Assistant B's Answer]"
                    prompt_filled = prompt_filled.replace(
                        marker, replacement)
                
                gen_text = generate_with_retry(prompt_filled, args.model_name)
                pred = parse_pairwise(gen_text)

                if pred == data['label']:
                    correct_cnt += 1
                cot_data.append({
                    'instruction': prompt_filled,
                    'output': gen_text,
                    'bias_type': bias_type,
                    'is_correct': pred == data['label']
                })
                print(f'bias_type: {bias_type}, correct_cnt: {correct_cnt}')
                line_cnt += 1
                if correct_cnt == 100:
                    break

    with open(os.path.join(INTER_DIR, f'train_cot_{args.model_name}.jsonl'), 'w') as f:
        for item in cot_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
