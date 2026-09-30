import json
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


with open(os.path.join(DISC_DIR, 'unified_feedback_data_4biased_eval.jsonl'), 'r') as f:
    lines = f.readlines()

all_data = []
for line in lines:
    data = json.loads(line)
    all_data.append(
        {
            'question': data['question'],
            'chosen': data['chosen'],
            'rejected': data['rejected'],
            'bias_type': data['bias_type'],
            'verify_correctness': data['verify_correctness']
        }
    )

with open(os.path.join(INTER_DIR, 'train_offsetbias.jsonl'), 'r') as f:
    for line in f:
        data = json.loads(line)
        all_data.append(
            {
                'question': data['instruction'],
                'chosen': data['output_1'],
                'rejected': [data['output_2']],
                'bias_type': [],
                'verify_correctness': "CHOSEN_IS_BEST"
            }
        )
with open(os.path.join(DISC_DIR, f'unified_feedback_data_4biased_eval_with_offsetbias.jsonl'), 'w') as f:
    for data in all_data:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')
