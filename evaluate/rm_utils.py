"""
Unified loader for discriminative judges.

Two kinds of checkpoints are supported and auto-detected from the directory:

* ``RewardModel`` trained by ``train/discriminative/train_reward_model.py``
  (a causal LM backbone + linear reward head saved as ``reward_head.pt``);
* any Hugging Face ``AutoModelForSequenceClassification`` model
  (Skywork-Reward, GRM, Llama-3.1 RM-RB2, ...).

``load_reward_model`` returns a :class:`RewardScorer` whose call operator maps
tokenized inputs to a 1-D tensor of scalar rewards, so evaluation scripts do
not need to know which architecture they are dealing with.
"""
import os
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_DISC_TRAIN_DIR = os.path.join(REPO_ROOT, "train", "discriminative")


class RewardScorer:
    """Callable wrapper returning a 1-D tensor of rewards for a batch."""

    def __init__(self, model, kind):
        self.model = model
        self.kind = kind  # "custom" | "seq_cls"
        self.model.eval()

    @property
    def device(self):
        return self.model.device

    @torch.no_grad()
    def __call__(self, **inputs):
        if self.kind == "custom":
            return self.model(**inputs).reshape(-1)
        return self.model(**inputs).logits.reshape(-1)


def is_custom_reward_model(model_path):
    return os.path.exists(os.path.join(model_path, "reward_head.pt"))


def load_reward_model(model_path, dtype=torch.bfloat16):
    """Return ``(scorer, tokenizer)`` for a discriminative judge checkpoint directory."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if is_custom_reward_model(model_path):
        if _DISC_TRAIN_DIR not in sys.path:
            sys.path.insert(0, _DISC_TRAIN_DIR)
        from reward_model import RewardModel  # noqa: WPS433 (repo-local import)

        model = RewardModel.from_pretrained(model_path).to(dtype)
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[rm_utils] loaded custom RewardModel (reward_head.pt) from {model_path}")
        return RewardScorer(model, "custom"), tokenizer

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, device_map="auto", torch_dtype=dtype
    )
    print(f"[rm_utils] loaded AutoModelForSequenceClassification from {model_path}")
    return RewardScorer(model, "seq_cls"), tokenizer
