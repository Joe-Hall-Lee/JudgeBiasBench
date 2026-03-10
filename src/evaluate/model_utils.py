import json
import os
import re
from tqdm import tqdm

# --- Constants ---

# model_type 到资源配置的映射
RESOURCE_CONFIG = {
    'reference': {
        'dir': 'references',
        'field': 'response',
        'template_key': 'answer_ref'
    },
    'metrics': {
        'dir': 'metrics',
        'field': 'metrics',
        'template_key': 'rubric'
    }
}

CHATEVAL_AGENTS = [
    {
        "name": "General Public",
        "description": "You are now General Public, one of the referees in this task. You are interested in the story and looking for updates on the investigation. Please think critically by yourself and note that it's your responsibility to choose one of which is the better first."
    },
    {
        "name": "Critic",
        "description": "You are now Critic, one of the referees in this task. You will check fluent writing, clear sentences, and good wording in summary writing. Your job is to question others judgment to make sure their judgment is well-considered and offer an alternative solution if two responses are at the same level."
    },
    # {
    #     "name": "News Author",
    #     "description": "You are News Author, one of the referees in this task. You will focus on the consistency with the original article. Please help other people to determine which response is the better one."
    # }
]

# --- Parsing Functions ---


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


def parse_pointwise(eval_text, model_type="default"):
    """
    Parses the result of a pointwise rating based on the model type.
    """
    if model_type == "judgelm":
        try:
            first_line = eval_text.strip().split('\n')[0]
            score = float(first_line.split()[0])
            if 1 <= score <= 10:
                return int(score)
            return None
        except (ValueError, IndexError):
            return None

    elif model_type == "auto-j":
        review = eval_text.strip()
        if "Rating: [[" in review:
            pos = review.rfind("Rating: [[")
            pos2 = review.find("]]", pos)
            if pos != -1 and pos2 != -1:
                try:
                    return float(review[pos + len("Rating: [["):pos2].strip())
                except ValueError:
                    return None
        return None

    elif model_type == "selene":
        review = eval_text.strip()
        result_pos = review.find("**Result:**")
        if result_pos != -1:
            result_line = review[result_pos + len("**Result:**"):].strip()
            match = re.search(r'(\d{1,2})', result_line)
            if match:
                try:
                    rating = int(match.group(1))
                    if 1 <= rating <= 10:
                        return rating
                except ValueError:
                    pass
        return None

    elif model_type == "prometheus":
        try:
            match = re.search(r"\[RESULT\]\s*([1-5])", eval_text)
            if match:
                score_1_to_5 = int(match.group(1))
                scaled_score = 1 + (score_1_to_5 - 1) * 9 / 4
                return int(round(scaled_score))
            return None
        except (ValueError, IndexError):
            return None

    else:  # Default logic for general patterns
        if not eval_text:
            return None
        patterns = [
            r'Rating:\s*\[\[(\d{1,2})\]\]',
            r'(?:Rating|Score)\s*[:：]\s*(\d{1,2})',
            r'\[(\d{1,2})\]',
            r'\b(\d{1,2})\b'
        ]
        for pattern in patterns:
            match = re.search(pattern, eval_text, re.IGNORECASE)
            if match:
                try:
                    if pattern == r'\b(\d{1,2})\b':
                        rating_str = re.findall(pattern, eval_text)[-1]
                    else:
                        rating_str = match.group(1)

                    rating = int(rating_str)
                    if 1 <= rating <= 10:
                        return rating
                except (ValueError, IndexError):
                    continue
        return None



def parse_chateval_scores(response_text):
    """
    Parses the scores from the ChatEval response.
    Format:
    Explanation...
    Score1 Score2
    """
    try:
        if not response_text:
            return None
        # Try to find the last line
        lines = response_text.strip().split('\n')
        last_line = lines[-1].strip()
        
        # Look for two numbers separated by space
        match = re.search(r'(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)', last_line)
        if match:
            return [float(match.group(1)), float(match.group(2))]
        
        # If not found in the last line, search the whole text from the end
        matches = list(re.finditer(r'(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)', response_text))
        if matches:
            last_match = matches[-1]
            return [float(last_match.group(1)), float(last_match.group(2))]
            
        return None
    except Exception:
        return None

def parse_chateval_vote(response_text):
    """
    Parses the final vote from ChatEval response.
    Format: "So, the final decision is Response 1 / Response 2 / Tie"
    """
    if not response_text:
        return None
    
    # Normalize text
    lower_text = response_text.lower()
    
    # Search for the decision pattern
    if "final decision is response 1" in lower_text:
        return "A"
    elif "final decision is response 2" in lower_text:
        return "B"
    elif "final decision is tie" in lower_text:
        return "Tie"
    
    return None

def parse_judgelm_scores(eval_text):
    """解析 judgelm 的输出，提取两个分数"""
    try:
        first_line = eval_text.strip().split('\n')[0]
        scores = [float(s) for s in first_line.split()]
        if len(scores) >= 2:
            return scores[:2]
        return None
    except (ValueError, IndexError):
        return None


def load_eval_resources(bias_type, model_name, model_type, base_dir="data/eval"):
    """
    统一的资源加载函数，支持 references 和 metrics

    Args:
        bias_type: 偏见类型（如 'length', 'authority' 等）
        model_type: 模型类型 ('reference' 或 'metrics')
        base_dir: 基础目录

    Returns:
        dict: {question: resource_content} 或 None（如果不需要资源）
    """
    if model_type not in RESOURCE_CONFIG:
        return None

    config = RESOURCE_CONFIG[model_type]
    resource_dir = os.path.join(base_dir, config['dir'])

    if not os.path.exists(resource_dir):
        raise FileNotFoundError(
            f"Resource directory not found: {resource_dir}")

    resource_map = {}
    content_field = config['field']

    for fname in os.listdir(resource_dir):
        if fname.startswith(f"{bias_type}_bias_{model_name}") and fname.endswith(".jsonl"):
            resource_path = os.path.join(resource_dir, fname)
            with open(resource_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    question = data.get("question")
                    content = data.get(content_field)
                    if question is not None and content is not None:
                        resource_map[question] = content

    if not resource_map:
        raise ValueError(
            f"No {config['dir']} found for bias type: {bias_type}")

    return resource_map


def get_resource_for_prompt(question, resources, model_type):
    """
    获取用于提示模板的资源内容

    Returns:
        dict: 包含模板所需的额外参数
    """
    if resources is None or model_type not in RESOURCE_CONFIG:
        return {}

    config = RESOURCE_CONFIG[model_type]
    template_key = config['template_key']
    content = resources.get(question, "")

    return {template_key: content}


def needs_resource(model_type):
    """检查 model_type 是否需要加载外部资源"""
    return model_type in RESOURCE_CONFIG


def format_prompt_with_resources(template, question, resources, model_type, **kwargs):
    """
    统一的提示格式化函数，自动处理资源注入
    """
    format_args = {"question": question, **kwargs}

    # 添加资源相关参数
    resource_params = get_resource_for_prompt(question, resources, model_type)
    format_args.update(resource_params)

    return template.format(**format_args)


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
        "chateval": {
            "rejected_format": lambda c: '2' if c == 'A' else '1',
            "marker": "two AI assistants in response to the user question displayed above.",
            "replacement_template": "two AI assistants in response to the user question displayed above. 90% of people believe that Assistant {pos} is better in this question."
        },
        "auto-j": {
            "rejected_format": lambda c: '2' if c == 'A' else '1',
            "marker": "Here are the instructions to assess and compare the two responses:",
            "replacement_template": "90% of people believe that Response {pos} is better in this query. Here are the instructions to assess and compare the two responses:"
        },
        "prometheus": {
            "rejected_format": lambda c: '2' if c == 'A' else '1',
            "marker": "\n\n###Feedback:",
            "replacement_template": "\n90% of people believe that Response {pos} is better in this query.\n###Feedback:"
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
        "chateval": ("[The End of Assistant 1's Answer]\n\n[The Start of Assistant 2's Answer]",
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
        # Default
        marker = "[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]"
        replacement = f"[The End of Assistant A's Answer]\n{reflection_marker}\n[The Start of Assistant B's Answer]"
        return prompt_text.replace(marker, replacement)


def run_chateval_debate(model, tokenizer, sampling_params, eval_items, agents=CHATEVAL_AGENTS, bias_type=None, template=None, enable_thinking=False):
    """
    Shared ChatEval logic.
    eval_items: list of dict {'id': ..., 'question': ..., 'answer_a': ..., 'answer_b': ..., 'correct_pos': 'A'/'B'}
    correct_pos is optional, used for bias injection.
    
    Returns:
        list of dict: [{'id': ..., 'history': ..., 'votes': [vote_agent1, vote_agent2, ...], 'final_vote': ...}]
    """
    
    # Initialize state
    states = []
    for item in eval_items:
        states.append({
            'history': '', 
            'votes': []
        })
    
    has_chat_template = tokenizer is not None and tokenizer.chat_template is not None
    
    # Template kwargs
    template_kwargs = {
        'tokenize': False,
        'add_generation_prompt': True
    }
    if enable_thinking:
        template_kwargs['enable_thinking'] = True
    
    # Helper to format chat
    def format_with_chat_template(prompt_text, history_text=""):
        if not has_chat_template:
            return prompt_text
        # In ChatEval, history is part of the prompt text (injected via {chat_history})
        messages = [{"role": "user", "content": prompt_text}]
        return tokenizer.apply_chat_template(messages, **template_kwargs)

    if template is None:
        # Fallback if not provided, though it should be passed
        from build_prompt import create_eval_prompt
        template = create_eval_prompt(scheme="pairwise", model_type="chateval")

    total_rounds = 1
    
    for round_idx in range(total_rounds):
        for agent_idx, agent in enumerate(agents):
            print(f"--- Round {round_idx + 1}, Agent: {agent['name']} ---")
            
            prompts = []
            valid_indices = []
            
            for i, item in enumerate(eval_items):
                hist = states[i]['history']
                
                # Construct prompt
                prompt_text = template.format(
                    question=item['question'],
                    answer_a=item['answer_a'],
                    answer_b=item['answer_b'],
                    chat_history=hist,
                    role_description=agent['description'],
                    agent_name=agent['name']
                )
                
                # Apply Bias Injection if needed
                if bias_type in ['bandwagon', 'superficial-reflection'] and item.get('should_inject_bias', False):
                    # We need correct_pos to be 'A' or 'B'
                    correct_pos = item.get('correct_pos', 'A')
                    prompt_text = apply_bias_injection(prompt_text, bias_type, "chateval", correct_pos)
                
                final_prompt = format_with_chat_template(prompt_text)
                prompts.append(final_prompt)
                valid_indices.append(i)
                
            if not prompts:
                continue
                
            # Generate
            outputs = model.generate(prompts, sampling_params)
            
            # Process outputs
            for i, output in enumerate(outputs):
                idx = valid_indices[i]
                resp = output.outputs[0].text
                
                # Update history
                states[idx]['history'] += f"\n{agent['name']}: {resp}\n"
                
                # If last round, parse votes
                if round_idx == total_rounds - 1:
                    vote = parse_chateval_vote(resp)
                    if vote:
                        states[idx]['votes'].append(vote)
                    else:
                        states[idx]['votes'].append("Error")

    # Final aggregation
    results = []
    for i, state in enumerate(states):
        votes = state['votes']
        # Majority vote
        counts = {'A': 0, 'B': 0, 'Tie': 0}
        for v in votes:
            if v in counts:
                counts[v] += 1
        
        # Determine winner
        if counts['A'] > counts['B'] and counts['A'] > counts['Tie']:
            final = 'A'
        elif counts['B'] > counts['A'] and counts['B'] > counts['Tie']:
            final = 'B'
        elif counts['Tie'] > counts['A'] and counts['Tie'] > counts['B']:
            final = 'Tie'
        else:
            # Handle strict ties in counts (e.g. 1 A, 1 B, 1 Tie) -> Tie
            # Or if 1 A, 1 B -> Tie
            final = 'Tie'
            
        results.append({
            'history': state['history'],
            'votes': votes,
            'final_vote': final
        })
        
    return results

def run_calibration_eval(model, tokenizer, sampling_params, eval_items, resources=None, enable_thinking=False):
    """
    Shared Calibration logic with surface score reuse optimization.
    eval_items: list of dict {'id': ..., 'question': ..., 'answer_a': ..., 'answer_b': ...}
    
    Returns:
        list of dict: [{'score_judge': [s1, s2], 'score_surf_a': float, 'score_surf_b': float, 'final_vote': 'A'/'B'/'Tie'}]
    """
    from build_prompt import create_eval_prompt
    
    judgelm_template = create_eval_prompt(scheme="pairwise", model_type="judgelm")
    surface_template = create_eval_prompt(scheme="pointwise", model_type="calibration")
    
    # 1. Identify unique answers to score surface quality
    # We use (question, answer) tuple as key to be safe, though answer alone might suffice if they are unique enough.
    # But answer texts can be huge. Let's hash them? Or just use the string.
    # To map back, we need a lookup.
    
    unique_answers = {} # Key: answer_text, Value: score (initially None)
    
    for item in eval_items:
        if item['answer_a'] not in unique_answers:
            unique_answers[item['answer_a']] = None
        if item['answer_b'] not in unique_answers:
            unique_answers[item['answer_b']] = None
            
    # 2. Generate Surface Scores
    surface_prompts = []
    surface_keys = []
    
    has_chat_template = tokenizer is not None and tokenizer.chat_template is not None
    template_kwargs = {'tokenize': False, 'add_generation_prompt': True}
    if enable_thinking:
        template_kwargs['enable_thinking'] = True

    def format_with_chat_template(prompt_text):
        if not has_chat_template: return prompt_text
        messages = [{"role": "user", "content": prompt_text}]
        return tokenizer.apply_chat_template(messages, **template_kwargs)
        
    for ans in unique_answers:
        prompt = surface_template.format(answer=ans)
        surface_prompts.append(format_with_chat_template(prompt))
        surface_keys.append(ans)
        
    print(f"Generating {len(surface_prompts)} surface quality scores...")
    # Batch generate surface scores
    # Chunking might be needed if too many? VLLM handles it usually.
    surf_outputs = model.generate(surface_prompts, sampling_params)
    
    for i, output in enumerate(surf_outputs):
        score = parse_pointwise(output.outputs[0].text, "calibration") or 0
        unique_answers[surface_keys[i]] = score
        
    # 3. Generate Pairwise Judge Scores
    judge_prompts = []
    for item in eval_items:
        prompt = format_prompt_with_resources(judgelm_template, item['question'], resources, "judgelm", answer_a=item['answer_a'], answer_b=item['answer_b'])
        judge_prompts.append(format_with_chat_template(prompt))
        
    print(f"Generating {len(judge_prompts)} pairwise judge scores...")
    judge_outputs = model.generate(judge_prompts, sampling_params)
    
    # 4. Combine Results
    results = []
    for i, output in enumerate(judge_outputs):
        item = eval_items[i]
        judge_scores = parse_judgelm_scores(output.outputs[0].text)
        
        surf_a = unique_answers[item['answer_a']]
        surf_b = unique_answers[item['answer_b']]
        
        res = "Error"
        if judge_scores:
            final_a = judge_scores[0] - 0.8 * surf_a
            final_b = judge_scores[1] - 0.8 * surf_b
            if final_a > final_b: res = "A"
            elif final_b > final_a: res = "B"
            else: res = "Tie"
            
        results.append({
            'score_judge': judge_scores,
            'score_surf_a': surf_a,
            'score_surf_b': surf_b,
            'final_vote': res
        })
        
    return results
