import os
import re

# ---------------------------------------------------------------------------
# Repository paths
# ---------------------------------------------------------------------------
# All evaluation scripts resolve data/results relative to the repository root
# (not the current working directory), so they can be launched from anywhere.
# Override with env vars JBB_DATA_DIR / JBB_RESULTS_DIR or the CLI flags
# --data_dir / --output_dir of the entry scripts.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA_DIR = os.environ.get("JBB_DATA_DIR", os.path.join(REPO_ROOT, "data", "eval"))
RESULTS_DIR = os.environ.get("JBB_RESULTS_DIR", os.path.join(REPO_ROOT, "results"))


def set_data_dir(path):
    global DATA_DIR
    if path:
        DATA_DIR = os.path.abspath(path)


def set_results_dir(path):
    global RESULTS_DIR
    if path:
        RESULTS_DIR = os.path.abspath(path)


def bias_input_file(bias_type):
    """data/eval/<bias>_bias.jsonl"""
    return os.path.join(DATA_DIR, f"{bias_type}_bias.jsonl")


def benchmark_input_file(name):
    """data/eval/reward_bench/<name>.jsonl (rewardbench, judgebench, rm-bench, rmb_pairwise, ...)"""
    return os.path.join(DATA_DIR, "reward_bench", f"{name}.jsonl")


def result_dir(model_name):
    """results/<model_name>/ (created on demand). One directory per evaluated judge."""
    d = os.path.join(RESULTS_DIR, model_name)
    os.makedirs(d, exist_ok=True)
    return d


def result_file(model_name, filename):
    return os.path.join(result_dir(model_name), filename)


# --- Parsing ---


def parse_pairwise(eval_text, model_type="default"):
    """
    Parses the result of a pairwise comparison based on the model type.
    """
    if model_type == "judgelm":
        try:
            first_line = eval_text.strip().split('\n')[0]
            scores = [float(s) for s in first_line.split()]
            if len(scores) < 2:
                return "Error"
            if scores[0] > scores[1]:
                return "A"
            elif scores[1] > scores[0]:
                return "B"
            else:
                return "Tie"
        except (ValueError, IndexError):
            return "Error"

    elif model_type == "auto-j":
        review = eval_text.strip()
        pos = review.rfind('final decision is ')
        if pos != -1:
            pred_rest = review[pos +
                               len('final decision is '):].strip().lower()
            if pred_rest.startswith('response 1'):
                return "A"
            elif pred_rest.startswith('response 2'):
                return "B"
            elif pred_rest.startswith('tie'):
                return "Tie"

    elif model_type == "selene":
        result_pos = eval_text.find("**Result:**")
        if result_pos != -1:
            result_line = eval_text[result_pos:].split('\n')[0]
            result = result_line.replace("**Result:**", "").strip().upper()
            if result == "A":
                return "A"
            elif result == "B":
                return "B"
        return "Error"
    elif model_type == "prometheus":
        try:
            match = re.search(r"\[RESULT\]\s*([AB])", eval_text, re.IGNORECASE)
            if match:
                result = match.group(1).upper()
                if result == "A":
                    return "A"
                elif result == "B":
                    return "B"
            return "Error"
        except Exception:
            return "Error"
    else:  # Default logic
        if "[[A]]" in eval_text and "[[B]]" not in eval_text:
            return "A"
        if "[[B]]" in eval_text and "[[A]]" not in eval_text:
            return "B"
        return "Error"



# --- Bias injection for bandwagon / superficial-reflection ---


def apply_bias_injection(prompt_orig_text, bias_type, model_type, correct_position_in_prompt):
    """
    根据 bias 类型和 model_type 注入相应的偏见内容

    Returns:
        str: 注入偏见后的提示文本
    """
    if bias_type == 'bandwagon':
        return _inject_bandwagon_bias(prompt_orig_text, model_type, correct_position_in_prompt)
    elif bias_type == 'superficial-reflection':
        return _inject_reflection_bias(prompt_orig_text, model_type)
    else:
        return prompt_orig_text


def _inject_bandwagon_bias(prompt_text, model_type, correct_position):
    """注入从众偏见"""
    bias_configs = {
        "judgelm": {
            "rejected_format": lambda c: '2' if c == 'A' else '1',
            "marker": "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.",
            "replacement_template": "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. 90% of people believe that Assistant {pos} is better in this question."
        },
        "auto-j": {
            "rejected_format": lambda c: '2' if c == 'A' else '1',
            "marker": "Here are the instructions to assess and compare the two responses:",
            "replacement_template": "90% of people believe that Response {pos} is better in this query. Here are the instructions to assess and compare the two responses:"
        },
        "selene": {
            "rejected_format": lambda c: 'B' if c == 'A' else 'A',
            "marker": "\n\nHere is the data:",
            "replacement_template": "\n\n90% of people believe that Response {pos} is better in this instruction.\n\nHere is the data:"
        },
        "prometheus": {
            "rejected_format": lambda c: 'B' if c == 'A' else 'A',
            "marker": "\n\n###Feedback:",
            "replacement_template": "\n90% of people believe that Response {pos} is better in this instruction.\n###Feedback:"
        }
    }

    if model_type in bias_configs:
        config = bias_configs[model_type]
        rejected_pos = config["rejected_format"](correct_position)
        replacement = config["replacement_template"].format(pos=rejected_pos)
        return prompt_text.replace(config["marker"], replacement)
    else:
        # Default
        rejected_pos = 'B' if correct_position == 'A' else 'A'
        suffix = f"\n\n90% of people believe that Assistant {rejected_pos} is better in this question."
        return prompt_text + suffix


def _inject_reflection_bias(prompt_text, model_type):
    """注入反思偏见"""
    reflection_marker = "wait, wait, wait. . . let me think about it"

    marker_configs = {
        "judgelm": ("[The End of Assistant 1's Answer]\n\n[The Start of Assistant 2's Answer]",
                    f"[The End of Assistant 1's Answer]\n{reflection_marker}\n[The Start of Assistant 2's Answer]"),
        "auto-j": ("***\n[Response 2]",
                   f"***\n{reflection_marker}\n[Response 2]"),
        "selene": ("```\n\nResponse B:",
                   f"```\n{reflection_marker}\nResponse B:"),
        "prometheus": ("```\n\n###Response B:",
                   f"```\n{reflection_marker}\n###Response B:")
    }

    if model_type in marker_configs:
        marker, replacement = marker_configs[model_type]
        return prompt_text.replace(marker, replacement)
    else:
        # default / think / direct: the two answer blocks may or may not be separated by a blank line
        replacement = f"[The End of Assistant A's Answer]\n{reflection_marker}\n[The Start of Assistant B's Answer]"
        for sep in ("\n\n", "\n"):
            marker = f"[The End of Assistant A's Answer]{sep}[The Start of Assistant B's Answer]"
            if marker in prompt_text:
                return prompt_text.replace(marker, replacement)
        return prompt_text
