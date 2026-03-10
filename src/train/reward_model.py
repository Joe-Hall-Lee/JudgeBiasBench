import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig
)
from typing import Optional


class RewardModel(nn.Module):
    """
    Reward Model using AutoModelForSequenceClassification for scoring responses.
    
    This model uses the standard Hugging Face classification approach with num_labels=1
    to output scalar reward scores for contrastive learning.
    """
    
    def __init__(self, model_name_or_path: str):
        super().__init__()
        
        # Load config and modify for regression (num_labels=1)
        self.config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.config.num_labels = 1  # For scalar reward output
        
        # Set pad_token_id in config to avoid batch processing issues
        if not hasattr(self.config, 'pad_token_id') or self.config.pad_token_id is None:
            # Use eos_token_id as pad_token_id if not set
            if hasattr(self.config, 'eos_token_id') and self.config.eos_token_id is not None:
                self.config.pad_token_id = self.config.eos_token_id
            else:
                # Fallback to a common pad token id
                self.config.pad_token_id = 0
        
        # Use AutoModelForSequenceClassification with num_labels=1
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            config=self.config,
            trust_remote_code=True
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass of the reward model.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            
        Returns:
            reward_scores: Scalar rewards of shape (batch_size,)
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # The logits from AutoModelForSequenceClassification with num_labels=1
        # will have shape (batch_size, 1), so we squeeze to get (batch_size,)
        reward_scores = outputs.logits.squeeze(-1)
        
        return reward_scores
    
    def save_pretrained(self, save_directory: str):
        """Save the model to a directory."""
        self.model.save_pretrained(save_directory)
    
    @classmethod
    def from_pretrained(cls, model_path: str):
        """Load a trained reward model from a directory."""
        # Create instance with the saved model
        instance = cls.__new__(cls)
        super(RewardModel, instance).__init__()
        
        # Load the saved model
        instance.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        instance.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            config=instance.config,
            trust_remote_code=True
        )
        
        return instance


def create_reward_model(model_name_or_path: str) -> RewardModel:
    """
    Create a reward model from a pretrained model.
    
    Args:
        model_name_or_path: Path to the pretrained model or model name
        
    Returns:
        RewardModel instance
    """
    return RewardModel(model_name_or_path)


def load_reward_model(model_path: str) -> RewardModel:
    """
    Load a trained reward model from a directory.
    
    Args:
        model_path: Path to the saved reward model directory
        
    Returns:
        RewardModel instance
    """
    return RewardModel.from_pretrained(model_path)
