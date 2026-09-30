import random
random.seed(42)
import json
from collections import defaultdict
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


with open(os.path.join(DISC_DIR, 'unified_feedback_data_4biased_eval_no_value_1biased.jsonl'), 'r') as f:
    processed_data = defaultdict(list)
    for line in f:
        data = json.loads(line)
        verify_correctness = data['verify_correctness']
        if verify_correctness == 'CHOSEN_IS_BEST':
            if random.randint(1, 2) == 1:
                answer_a = data['chosen']
                answer_b = data['rejected'][1]
                label = 'A'
            else:
                answer_a = data['rejected'][1]
                answer_b = data['chosen']
                label = 'B'
            processed_data[data['bias_type'][0]].append(
                {
                    'question': data['question'],
                    'answer_a': answer_a,
                    'answer_b': answer_b,
                    'label': label
                }
            )
        else:
            if random.randint(1, 2) == 1:
                answer_a = data['chosen']
                answer_b = data['rejected'][0]
                label = 'A'
            else:
                answer_a = data['rejected'][0]
                answer_b = data['chosen']
                label = 'B'
            processed_data['orig'].append(
                {
                    'question': data['question'],
                    'answer_a': answer_a,
                    'answer_b': answer_b,
                    'label': label
                }
            )

for bias_type, data in processed_data.items():
    with open(os.path.join(INTER_DIR, f'train_{bias_type}.jsonl'), 'w') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
