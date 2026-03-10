import json
import torch
import random
import logging
from typing import List, Dict, Any, Tuple
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ContrastiveExample:
    """Single example for contrastive learning."""
    question: str
    chosen: str
    rejected: List[str]
    similar_question: str = None
    bias_type: List[str] = None


class RewardModelDataset(Dataset):
    """
    Dataset for reward model training with contrastive learning.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        max_rejected: int = None
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the JSON data file
            tokenizer: Tokenizer for processing text
            max_length: Maximum sequence length
            max_rejected: Maximum number of rejected answers to use (None for all)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_rejected = max_rejected
        
        # Load and process data
        self.examples = self._load_data(data_path)
        logger.info(f"Loaded {len(self.examples)} examples from {data_path}")
    
    def _load_data(self, data_path: str) -> List[ContrastiveExample]:
        """Load data from JSON file."""
        examples = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    
                    # Extract required fields
                    question = data.get('question', '')
                    chosen = data.get('chosen', '')
                    rejected = data.get('rejected', [])
                    similar_question = data.get('similar_question', None)
                    bias_type = data.get('bias_type', None)
                    
                    # Validate data
                    if not question or not chosen or not rejected:
                        logger.warning(f"Skipping incomplete example at line {line_idx + 1}")
                        continue
                    
                    # Limit number of rejected answers if specified
                    if self.max_rejected and len(rejected) > self.max_rejected:
                        rejected = random.sample(rejected, self.max_rejected)
                    
                    examples.append(ContrastiveExample(
                        question=question,
                        chosen=chosen,
                        rejected=rejected,
                        similar_question=similar_question,
                        bias_type=bias_type
                    ))
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON at line {line_idx + 1}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Error processing line {line_idx + 1}: {e}")
                    continue
        
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single example."""
        example = self.examples[idx]
        
        # Format conversations using chat template
        chosen_conversation = [
            {"role": "user", "content": example.question},
            {"role": "assistant", "content": example.chosen}
        ]
        
        rejected_conversations = [
            [
                {"role": "user", "content": example.question},
                {"role": "assistant", "content": rejected_answer}
            ]
            for rejected_answer in example.rejected
        ]
        
        # Tokenize chosen conversation
        chosen_text = self.tokenizer.apply_chat_template(
            chosen_conversation,
            tokenize=False,
            add_generation_prompt=False
        )
        chosen_encoding = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize rejected conversations
        rejected_encodings = []
        for rejected_conv in rejected_conversations:
            rejected_text = self.tokenizer.apply_chat_template(
                rejected_conv,
                tokenize=False,
                add_generation_prompt=False
            )
            rejected_encoding = self.tokenizer(
                rejected_text,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt"
            )
            rejected_encodings.append(rejected_encoding)
        
        return {
            'chosen_input_ids': chosen_encoding['input_ids'].squeeze(0),
            'chosen_attention_mask': chosen_encoding['attention_mask'].squeeze(0),
            'rejected_input_ids': [enc['input_ids'].squeeze(0) for enc in rejected_encodings],
            'rejected_attention_mask': [enc['attention_mask'].squeeze(0) for enc in rejected_encodings],
            'num_rejected': len(rejected_encodings)
        }


def collate_fn(batch: List[Dict[str, Any]], tokenizer: PreTrainedTokenizer, training_mode: str = "contrast") -> Dict[str, torch.Tensor]:
    """
    Collate function for batching examples.
    Optimized version: combines chosen and rejected into unified input_ids and attention_mask.
    
    Args:
        batch: List of examples
        tokenizer: Tokenizer for padding
        training_mode: "contrast" or "pairwise"
            - contrast: chosen + all rejected (for contrastive learning)
            - pairwise: chosen + first rejected only (for pairwise comparison)
    
    Returns:
        Dict with unified input_ids, attention_mask, and example_lengths
        - For contrast: [chosen, rejected1, rejected2, ...] per example
        - For pairwise: [chosen, rejected1] per example
    """
    all_input_ids = []
    all_attention_mask = []
    example_lengths = []  # Number of examples per batch item
    
    for item in batch:
        # Add chosen example first (index 0)
        all_input_ids.append(item['chosen_input_ids'])
        all_attention_mask.append(item['chosen_attention_mask'])
        
        if training_mode == "contrast":
            # # Add all rejected examples for contrastive learning
            # all_input_ids.extend(item['rejected_input_ids'])
            # all_attention_mask.extend(item['rejected_attention_mask'])
            # # Record total length: 1 chosen + all rejected
            # example_lengths.append(1 + item['num_rejected'])

            # Add all rejected examples for contrastive learning
            all_input_ids.extend([item['rejected_input_ids'][0]] + item['rejected_input_ids'][2:])
            all_attention_mask.extend([item['rejected_attention_mask'][0]] + item['rejected_attention_mask'][2:])
            # Record total length: 1 chosen + all rejected
            example_lengths.append(item['num_rejected'])
            
        elif training_mode == "pairwise":
            # Add only the first rejected example for pairwise comparison
            if item['num_rejected'] > 0:
                all_input_ids.append(item['rejected_input_ids'][0])
                all_attention_mask.append(item['rejected_attention_mask'][0])
                # Record total length: 1 chosen + 1 rejected
                example_lengths.append(2)
            else:
                # If no rejected examples, skip this item (shouldn't happen in practice)
                example_lengths.append(1)
        else:
            raise ValueError(f"Unknown training_mode: {training_mode}")
    
    # Pad all sequences
    input_ids = torch.nn.utils.rnn.pad_sequence(
        all_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        all_attention_mask, batch_first=True, padding_value=0
    )
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'example_lengths': torch.tensor(example_lengths, dtype=torch.long)
    }


def create_dataloader(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 4,
    max_length: int = 512,
    max_rejected: int = None,
    shuffle: bool = True,
    num_workers: int = 0,
    training_mode: str = "contrast"
) -> DataLoader:
    """
    Create a DataLoader for reward model training.
    
    Args:
        data_path: Path to the JSON data file
        tokenizer: Tokenizer for processing text
        batch_size: Batch size
        max_length: Maximum sequence length
        max_rejected: Maximum number of rejected answers per example
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        training_mode: "contrast" or "pairwise" training mode
        
    Returns:
        DataLoader instance
    """
    dataset = RewardModelDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        max_rejected=max_rejected
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, training_mode),
        pin_memory=True
    )