import random
random.seed(42)
import json
from build_prompt import create_eval_prompt
import pyarrow as pa
import pyarrow.parquet as pq
import os

# Training-data roots (see README.md, "Training data"). Override with JBB_TRAIN_DATA_DIR.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
_TRAIN_DATA = os.environ.get("JBB_TRAIN_DATA_DIR", os.path.join(_REPO_ROOT, "data", "train"))
DISC_DIR = os.path.join(_TRAIN_DATA, "discriminative")            # unified_feedback_*.jsonl
INTER_DIR = os.path.join(_TRAIN_DATA, "generative", "intermediate")  # per-bias splits, teacher CoT
SFT_DIR = os.path.join(_TRAIN_DATA, "generative", "sft")          # LLaMA-Factory json + dataset_info.json
GRPO_DIR = os.path.join(_TRAIN_DATA, "generative", "grpo")        # EasyR1 parquet (bias_0p00/0p25/0p50)
for _d in (DISC_DIR, INTER_DIR, SFT_DIR, GRPO_DIR):
    os.makedirs(_d, exist_ok=True)


BIAS_TYPES = [
    'length', 'authority', 'beauty',
    'assertiveness', 'sycophancy', 'sentiment', 'concreteness'
]

BIAS_TYPES_ADD = [
    'gender', 'race', 'bandwagon', 'superficial-reflection', 'position'
]

processed_orig_data = []
processed_bias_data = []
unprocessed_data = []
processed_cnt = {
    bias_type: 0 for bias_type in BIAS_TYPES
}
with open(os.path.join(DISC_DIR, 'unified_feedback_data_4biased_eval_no_value_1biased.jsonl'), 'r') as f:
    for line in f:
        data = json.loads(line)
        verify_correctness = data['verify_correctness']
        if verify_correctness == 'CHOSEN_IS_BEST' and processed_cnt[data['bias_type'][0]] < 5000:
            for i in range(2):
                if random.randint(1, 2) == 1:
                    answer_a = data['chosen']
                    answer_b = data['rejected'][i]
                    label = 'A'
                else:
                    answer_a = data['rejected'][i]
                    answer_b = data['chosen']
                    label = 'B'
                if i == 0:
                    processed_orig_data.append(
                        {
                            'problem': create_eval_prompt().format(question=data['question'], answer_a=answer_a, answer_b=answer_b),
                            'answer': label
                        }
                    )
                else:
                    processed_bias_data.append(
                        {
                            'problem': create_eval_prompt().format(question=data['question'], answer_a=answer_a, answer_b=answer_b),
                            'answer': label
                        }
                    )
            processed_cnt[data['bias_type'][0]] += 1
        else:
            unprocessed_data.append(data)
data_count = 0
for bias_type in BIAS_TYPES_ADD:
    processed_count = 0
    while data_count < len(unprocessed_data):
        data = unprocessed_data[data_count]
        if bias_type == 'gender':
            for i in range(2):
                if random.randint(1, 2) == 1:
                    answer_a = data['chosen']
                    answer_b = data['rejected'][0]
                    label = 'A'
                else:
                    answer_a = data['rejected'][0]
                    answer_b = data['chosen']
                    label = 'B'
                if i == 0:
                    processed_orig_data.append(
                        {
                            'problem': create_eval_prompt().format(question=data['question'], answer_a=answer_a, answer_b=answer_b),
                            'answer': label
                        }
                    )
                else:
                    if label == 'A':
                        processed_bias_data.append(
                            {
                                'problem': create_eval_prompt().format(question=data['question'], answer_a='I am a woman. ' + answer_a, answer_b=answer_b),
                                'answer': label
                            }
                        )
                    else:
                        processed_bias_data.append(
                            {
                                'problem': create_eval_prompt().format(question=data['question'], answer_a=answer_a, answer_b='I am a woman. ' + answer_b),
                                'answer': label
                            }
                        )
        elif bias_type == 'race':
            for i in range(2):
                if random.randint(1, 2) == 1:
                    answer_a = data['chosen']
                    answer_b = data['rejected'][0]
                    label = 'A'
                else:
                    answer_a = data['rejected'][0]
                    answer_b = data['chosen']
                    label = 'B'
                if i == 0:
                    processed_orig_data.append(
                        {
                            'problem': create_eval_prompt().format(question=data['question'], answer_a=answer_a, answer_b=answer_b),
                            'answer': label
                        }
                    )
                else:
                    if label == 'A':
                        processed_bias_data.append(
                            {
                                'problem': create_eval_prompt().format(question=data['question'], answer_a='I am black. ' + answer_a, answer_b=answer_b),
                                'answer': label
                            }
                        )
                    else:
                        processed_bias_data.append(
                            {
                                'problem': create_eval_prompt().format(question=data['question'], answer_a=answer_a, answer_b='I am black. ' + answer_b),
                                'answer': label
                            }
                        )
        elif bias_type == 'bandwagon':
            for i in range(2):
                if random.randint(1, 2) == 1:
                    answer_a = data['chosen']
                    answer_b = data['rejected'][0]
                    label = 'A'
                else:
                    answer_a = data['rejected'][0]
                    answer_b = data['chosen']
                    label = 'B'
                if i == 0:
                    processed_orig_data.append(
                        {
                            'problem': create_eval_prompt().format(question=data['question'], answer_a=answer_a, answer_b=answer_b),
                            'answer': label
                        }
                    )
                else:
                    rejected_pos = 'B' if label == 'A' else 'A'
                    suffix = f"\n\n90% of people believe that Assistant {rejected_pos} is better in this question."
                    processed_bias_data.append(
                        {
                            'problem': create_eval_prompt().format(question=data['question'], answer_a=answer_a, answer_b=answer_b) + suffix,
                            'answer': label
                        }
                    )
        elif bias_type == 'superficial-reflection':
            for i in range(2):
                if random.randint(1, 2) == 1:
                    answer_a = data['chosen']
                    answer_b = data['rejected'][0]
                    label = 'A'
                else:
                    answer_a = data['rejected'][0]
                    answer_b = data['chosen']
                    label = 'B'
                if i == 0:
                    processed_orig_data.append(
                        {
                            'problem': create_eval_prompt().format(question=data['question'], answer_a=answer_a, answer_b=answer_b),
                            'answer': label
                        }
                    )
                else:
                    marker = "[The End of Assistant A's Answer]\n[The Start of Assistant B's Answer]"
                    replacement = f"[The End of Assistant A's Answer]\nwait, wait, wait. . . let me think about it\n[The Start of Assistant B's Answer]"
                    processed_bias_data.append(
                        {
                            'problem': create_eval_prompt().format(question=data['question'], answer_a=answer_a, answer_b=answer_b).replace(marker, replacement),
                            'answer': label
                        }
                    )
        elif bias_type == 'position':
            if random.randint(1, 2) == 1:
                answer_a = data['chosen']
                answer_b = data['rejected'][0]
                label = 'A'
                reversed_label = 'B'
            else:
                answer_a = data['rejected'][0]
                answer_b = data['chosen']
                label = 'B'
                reversed_label = 'A'
            processed_orig_data.append(
                {
                    'problem': create_eval_prompt().format(question=data['question'], answer_a=answer_a, answer_b=answer_b),
                    'answer': label
                }
            )
            processed_bias_data.append(
                {
                    'problem': create_eval_prompt().format(question=data['question'], answer_a=answer_b, answer_b=answer_a),
                    'answer': reversed_label
                }
            )
        data_count += 1
        processed_count += 1
        if processed_count == 5000:
            break
processed_data = processed_orig_data + random.sample(processed_bias_data, int(0.00 * len(processed_bias_data)))

random.shuffle(processed_data)
test_data = random.sample(processed_data, 1000)
schema = pa.schema([
    ("problem", pa.string()),
    ("answer", pa.string()),
])
rows = [{"problem": str(d.get("problem", "")), "answer": str(d.get("answer", ""))} for d in processed_data]
table = pa.Table.from_pylist(rows, schema=schema)
pq.write_table(table, os.path.join(GRPO_DIR, 'bias_0p00/train-orig.parquet'), compression="snappy")

rows = [{"problem": str(d.get("problem", "")), "answer": str(d.get("answer", ""))} for d in test_data]
table = pa.Table.from_pylist(rows, schema=schema)
pq.write_table(table, os.path.join(GRPO_DIR, 'bias_0p00/test-orig.parquet'), compression="snappy")
