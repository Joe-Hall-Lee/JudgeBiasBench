import json
import re
from datasets import load_dataset
from tqdm import tqdm
import os

def clean_answer(answer):
    """
    Clean up the answer by removing special tokens and artifacts.
    
    Args:
        answer (str): The raw answer text
        
    Returns:
        str: Cleaned answer text
    """
    if not answer:
        return answer
    
    # Remove common special tokens
    answer = answer.strip()
    
    # Remove <|im_end|> and similar tokens
    answer = re.sub(r'<\|im_end\|>', '', answer)
    answer = re.sub(r'<\|.*?\|>', '', answer)  # Remove any other special tokens
    
    # Remove trailing whitespace and newlines
    answer = answer.strip()
    
    return answer

def parse_instruction(instruction):
    """
    Parse the instruction field to extract user question and assistant answers.
    
    Args:
        instruction (str): The instruction containing evaluation prompt and assistant answers
        
    Returns:
        tuple: (user_question, assistant_a_answer, assistant_b_answer) or (None, None, None) if parsing fails
    """
    try:
        # Extract user question
        user_question_match = re.search(r'\[User Question\]\n(.*?)\n\n\[The Start of Assistant A\'s Answer\]', instruction, re.DOTALL)
        if not user_question_match:
            return None, None, None
        user_question = user_question_match.group(1).strip()
        
        # Extract Assistant A's answer
        assistant_a_match = re.search(r'\[The Start of Assistant A\'s Answer\]\n(.*?)\n\[The End of Assistant A\'s Answer\]', instruction, re.DOTALL)
        if not assistant_a_match:
            return None, None, None
        assistant_a_answer = clean_answer(assistant_a_match.group(1))
        
        # Extract Assistant B's answer
        assistant_b_match = re.search(r'\[The Start of Assistant B\'s Answer\]\n(.*?)\n\[The End of Assistant B\'s Answer\]', instruction, re.DOTALL)
        if not assistant_b_match:
            return None, None, None
        assistant_b_answer = clean_answer(assistant_b_match.group(1))
        
        return user_question, assistant_a_answer, assistant_b_answer
    except Exception as e:
        print(f"Error parsing instruction: {e}")
        return None, None, None

def process_dataset():
    """
    Load and process the GRAM-fine-tuning-65k dataset.
    Convert judgment data to preference learning format.
    """
    print("Loading dataset...")
    # Login using e.g. `huggingface-cli login` to access this dataset
    ds = load_dataset("NiuTrans/GRAM-fine-tuning-65k")['train']
    
    output_file = "data/train-unified-feedback/unified_feedback_data.jsonl"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    processed_count = 0
    skipped_count = 0
    
    print(f"Processing {len(ds)} samples...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in tqdm(ds, desc="Processing samples"):
            # Extract fields
            instruction = sample.get('instruction', '')
            output = sample.get('output', '')
            
            # Parse instruction to get user question and assistant answers
            user_question, assistant_a_answer, assistant_b_answer = parse_instruction(instruction)
            
            if user_question is None or assistant_a_answer is None or assistant_b_answer is None:
                skipped_count += 1
                continue
            
            # Determine chosen and rejected based on output
            if output == 'A':
                chosen_answer = assistant_a_answer
                rejected_answer = assistant_b_answer
            elif output == 'B':
                chosen_answer = assistant_b_answer
                rejected_answer = assistant_a_answer
            else:
                # Skip samples with invalid output
                skipped_count += 1
                continue
            
            # Create output record
            record = {
                "question": user_question,
                "chosen": chosen_answer,
                "rejected": [rejected_answer],
            }
            
            # Write to JSONL file
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            processed_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Processed samples: {processed_count}")
    print(f"Skipped samples: {skipped_count}")
    print(f"Output file: {output_file}")

if __name__ == "__main__":
    process_dataset()
