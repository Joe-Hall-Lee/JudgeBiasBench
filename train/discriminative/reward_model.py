import os
import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Optional


class RewardModel(nn.Module):
    """
    Discriminative judge using a custom scoring head on top of a pretrained transformer.
    """
    
    def __init__(self, base_model: nn.Module):
        super().__init__()
        
        # Use the provided base model
        self.model = base_model
        self.config = self.model.config
        
        # Define the reward head
        self.reward_head = nn.Linear(self.config.hidden_size, 1, bias=False)
        
        # Set pad_token_id in config to be used for finding the last token
        if not hasattr(self.config, 'pad_token_id') or self.config.pad_token_id is None:
            eos_token_id = getattr(self.config, 'eos_token_id', None)
            if eos_token_id is not None:
                # eos_token_id is a list for Llama-3.1-style configs but a plain
                # int for Qwen2.5; the trainer overrides this with the
                # tokenizer's real pad id right after construction
                if isinstance(eos_token_id, (list, tuple)):
                    eos_token_id = eos_token_id[0]
                self.config.pad_token_id = eos_token_id
            else:
                # Fallback to a common pad token id
                self.config.pad_token_id = 0

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass of the discriminative judge.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            
        Returns:
            reward_scores: Scalar rewards of shape (batch_size,)
        """
        # Get hidden states from the base model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs
        )
        
        # Get the hidden state of the last token
        last_hidden_state = outputs.last_hidden_state
        
        # Find the index of the last non-padding token for each sequence
        if self.config.pad_token_id is not None:
            sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
            # Ensure indices are not negative
            sequence_lengths = torch.max(sequence_lengths, torch.zeros_like(sequence_lengths))
        else:
            # If no pad token, assume last token is the one to use
            sequence_lengths = input_ids.shape[1] - 1

        # Gather the hidden states of the last non-pad token
        last_token_states = last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device), sequence_lengths]

        # Calculate reward scores
        reward_scores = self.reward_head(last_token_states).squeeze(-1)
        
        return reward_scores

    @property
    def device(self):
        """Returns the device of the base model."""
        return self.model.device
    
    def save_pretrained(self, save_directory: str, state_dict: Optional[dict] = None, **kwargs):
        """
        Save the base model and the reward head.
        
        If a state_dict is provided (e.g., from accelerator.get_state_dict),
        it's split and used to save the base model and reward head separately.
        """
        os.makedirs(save_directory, exist_ok=True)

        if state_dict is None:
            # Default behavior: save the current state of the model components
            self.model.save_pretrained(save_directory, **kwargs)
            torch.save(self.reward_head.state_dict(), os.path.join(save_directory, "reward_head.pt"))
        else:
            # When a full state_dict is provided, we need to separate it
            base_model_state_dict = {}
            reward_head_state_dict = {}
            
            for key, value in state_dict.items():
                if key.startswith("model."):
                    base_model_state_dict[key.replace("model.", "", 1)] = value
                elif key.startswith("reward_head."):
                    reward_head_state_dict[key.replace("reward_head.", "", 1)] = value

            # Save the base model using its part of the state dict
            self.model.save_pretrained(save_directory, state_dict=base_model_state_dict, **kwargs)
            
            # Save the reward head's state dict
            torch.save(reward_head_state_dict, os.path.join(save_directory, "reward_head.pt"))
    
    @classmethod
    def from_pretrained(cls, model_path: str):
        """Load a trained discriminative judge for evaluation."""
        base_model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        model = cls(base_model)
        
        reward_head_path = os.path.join(model_path, "reward_head.pt")
        if os.path.exists(reward_head_path):
            model.reward_head.load_state_dict(torch.load(reward_head_path, map_location=torch.device("cpu")))
        
        return model


def create_reward_model(model_name_or_path: str) -> RewardModel:
    """Create a discriminative judge for training, starting from a pretrained model."""
    base_model = AutoModel.from_pretrained(
        model_name_or_path, 
        trust_remote_code=True
    )
    return RewardModel(base_model)


def load_reward_model(model_path: str) -> RewardModel:
    """
    Load a trained discriminative judge from a directory.
    """
    return RewardModel.from_pretrained(model_path)
