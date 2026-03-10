import os
import json
from build_prompt import create_eval_prompt
from model_utils import run_chateval_debate

def evaluate_chateval_bias(bias_type, model_name, model, tokenizer, sampling_params, eval_bias_only=False, enable_thinking=False):
    """
    Implements ChatEval evaluation strategy using shared run_chateval_debate.
    """
    input_file = f"data/eval/{bias_type}_bias.jsonl"
    output_file = f"results/{model_name}-chateval/{bias_type}_{model_name}.jsonl"
    summary_file = f"results/{model_name}-chateval/{bias_type}_summary_{model_name}.json"

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return None

    with open(input_file, 'r', encoding='utf-8') as f:
        source_data = [json.loads(line) for line in f]

    print(f"\nEvaluating bias type: {bias_type} with ChatEval strategy")
    print(f"Successfully loaded {len(source_data)} items.")

    # Prepare eval items
    eval_items = []
    # To map results back to source_data
    # item_map = [ {source_idx, type='orig'|'bias'} ]
    item_map = []

    for idx, item in enumerate(source_data):
        question = item.get("question")
        
        # Determine answers for orig and bias scenarios
        if bias_type == 'position':
            label_is_response1 = item.get("label") == "response1"
            response1 = item.get("response1") if label_is_response1 else item.get("response2")
            response2 = item.get("response2") if label_is_response1 else item.get("response1")
            
            # Orig: A=response1, B=response2. Correct: A
            # Bias: A=response2, B=response1. Correct: B
            
            if not eval_bias_only:
                eval_items.append({
                    'id': f"{idx}_orig",
                    'question': question,
                    'answer_a': response1,
                    'answer_b': response2,
                    'correct_pos': 'A'
                })
                item_map.append({'idx': idx, 'type': 'orig'})
            
            eval_items.append({
                'id': f"{idx}_bias",
                'question': question,
                'answer_a': response2,
                'answer_b': response1,
                'correct_pos': 'B'
            })
            item_map.append({'idx': idx, 'type': 'bias'})

        else:
            response1_orig = item.get("original_response1") or item.get("response1")
            response2_orig = item.get("original_response2") or item.get("response2")
            response1_bias = item.get("rewritten_response1") or response1_orig
            response2_bias = item.get("rewritten_response2") or response2_orig
            
            label = item.get("label")
            # If label is response1, then response1 is better (A)
            correct_pos = 'A' if label == "response1" else 'B'
            
            if not eval_bias_only:
                eval_items.append({
                    'id': f"{idx}_orig",
                    'question': question,
                    'answer_a': response1_orig,
                    'answer_b': response2_orig,
                    'correct_pos': correct_pos,
                    'should_inject_bias': False
                })
                item_map.append({'idx': idx, 'type': 'orig'})
                
            eval_items.append({
                'id': f"{idx}_bias",
                'question': question,
                'answer_a': response1_bias,
                'answer_b': response2_bias,
                'correct_pos': correct_pos,
                'should_inject_bias': True
            })
            item_map.append({'idx': idx, 'type': 'bias'})

    # Run Evaluation
    results_list = run_chateval_debate(
        model, tokenizer, sampling_params, eval_items, 
        bias_type=bias_type, enable_thinking=enable_thinking
    )
    
    # Process Results
    final_records = []
    
    # Organize results by source index
    records_by_idx = {}
    
    for i, res in enumerate(results_list):
        meta = item_map[i]
        idx = meta['idx']
        typ = meta['type']
        
        if idx not in records_by_idx:
            records_by_idx[idx] = {}
        records_by_idx[idx][typ] = res
        
    for idx, item in enumerate(source_data):
        if idx not in records_by_idx:
            continue
            
        res_map = records_by_idx[idx]
        
        orig_winner = "skipped"
        if not eval_bias_only:
            orig_res = res_map.get('orig')
            if orig_res:
                orig_winner = orig_res['final_vote']
        
        bias_res = res_map.get('bias')
        bias_winner = bias_res['final_vote'] if bias_res else "Error"
        
        # Determine correctness
        if bias_type == 'position':
            label_is_response1 = item.get("label") == "response1"
            correct_choice_orig = 'A' if label_is_response1 else 'B'
            correct_choice_bias = 'B' if label_is_response1 else 'A'
            
            is_orig_eval_correct = (orig_winner == correct_choice_orig) if orig_winner not in ["Error", "skipped", "Tie"] else orig_winner
            is_bias_eval_correct = "eval_error" if bias_winner in ["Error", "Tie"] else (bias_winner == correct_choice_bias)
            
        else:
            label = item.get("label")
            correct_choice = 'A' if label == "response1" else 'B'
            
            is_orig_eval_correct = (orig_winner == correct_choice) if orig_winner not in ["Error", "skipped", "Tie"] else orig_winner
            is_bias_eval_correct = "eval_error" if bias_winner in ["Error", "Tie"] else (bias_winner == correct_choice)

        final_records.append({
            "question": item.get("question"),
            "label": item.get("label"),
            "original_evaluation_correct": is_orig_eval_correct,
            "bias_evaluation_correct": is_bias_eval_correct,
            "chat_history_orig": res_map.get('orig', {}).get('history', "skipped") if not eval_bias_only else "skipped",
            "chat_history_bias": res_map.get('bias', {}).get('history', "")
        })

    # Save Results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for record in final_records:
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"Detailed ChatEval results saved to: '{output_file}'")
    
    # Compute Summary
    total = len(final_records)
    if eval_bias_only:
        correct = sum(1 for r in final_records if r['bias_evaluation_correct'] is True)
        acc = correct / total if total > 0 else 0
        print(f"Biased Pair Accuracy: {acc*100:.2f}%")
        summary = {"biased_pair_accuracy": f"{acc*100:.2f}%"}
    else:
        orig_correct = sum(1 for r in final_records if r['original_evaluation_correct'] is True)
        bias_correct = sum(1 for r in final_records if r['bias_evaluation_correct'] is True)
        
        # BSR
        bsr_numerator = sum(1 for r in final_records if r['original_evaluation_correct'] is True and r['bias_evaluation_correct'] is False)
        bsr = bsr_numerator / orig_correct if orig_correct > 0 else 0
        
        orig_acc = orig_correct / total if total > 0 else 0
        bias_acc = bias_correct / total if total > 0 else 0
        
        print(f"Original Accuracy: {orig_acc*100:.2f}%")
        print(f"Biased Accuracy: {bias_acc*100:.2f}%")
        print(f"Bias Sensitive Rate (BSR): {bsr*100:.2f}%")
        
        summary = {
            "original_accuracy": f"{orig_acc*100:.2f}%",
            "biased_accuracy": f"{bias_acc*100:.2f}%",
            "bias_sensitive_rate": f"{bsr*100:.2f}%"
        }
        
    with open(summary_file, 'w', encoding='utf-8') as f_summary:
        json.dump(summary, f_summary, indent=4, ensure_ascii=False)
        
    return summary
