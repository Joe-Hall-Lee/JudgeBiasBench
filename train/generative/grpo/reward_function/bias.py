# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import random
from typing import Any, Dict, List


def exact_match_reward(response: str, ground_truth: str) -> float:
    pattern = r"\[\[([AB])\]\]"
    ans = re.findall(pattern, response, re.DOTALL | re.MULTILINE)
    if ans:
        ans = ans[0]
        reward = 1.0 if ans == ground_truth else 0.0
    else:
        reward = 0.0
    if random.random() < 0.1:
        print("response: ", response)
        print("answer: ", ans)
        print("ground_truth: ", ground_truth)
    return reward


def format_reward(response: str) -> float:
    pattern = re.compile(r"^<think>.*?</think>\s*\[\[([AB])\]\]\s*$", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def compute_score(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.1) -> List[Dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    for reward_input in reward_inputs:
        response = reward_input["response"]  # handle qwen2.5vl-32b format
        format_score = format_reward(response)
        exact_match_score = exact_match_reward(response, reward_input["ground_truth"])
        scores.append(
            {
                "overall": (1 - format_weight) * exact_match_score + format_weight * format_score,
                "format": format_score,
                "exact match": exact_match_score,
            }
        )

    return scores
