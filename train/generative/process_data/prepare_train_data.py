import json
import argparse
import random
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
random.seed(42)
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Keep the teacher traces whose verdict is correct and write the SFT json")
    parser.add_argument("--model_name", type=str,
                        default="gemini-2.0-flash", help="Model name to use")
    args = parser.parse_args()

    with open(os.path.join(INTER_DIR, f'train_cot_{args.model_name}.jsonl'), 'r') as f:
        lines = f.readlines()
        json_data = []
        for line in lines:
            data = json.loads(line)
            if data['is_correct']:
                json_data.append(
                    {
                        "instruction": data['instruction'],
                        "output": data['output']
                    }
                )
        
        random.shuffle(json_data)

    with open(os.path.join(SFT_DIR, f'train_cot_{args.model_name}.json'), 'w') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
