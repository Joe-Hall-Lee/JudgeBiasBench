import os
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
from typing import Dict, Any
from torch import nn
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed
)
from accelerate import Accelerator

from reward_model import create_reward_model
from data_utils import create_dataloader

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class RewardModelTrainer:
    """Trainer for reward model with contrastive learning and pairwise training."""
    
    def __init__(self, args):
        self.args = args
        
        # Initialize accelerator with mixed precision
        self.accelerator = Accelerator(
            mixed_precision="fp16" if args.fp16 else ("bf16" if args.bf16 else "no")
        )
        
        # Set up logging
        logger.info(f"Accelerator state: {self.accelerator.state}")
        
        # Set random seed
        if args.seed is not None:
            set_seed(args.seed)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True
        )
        
        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize model
        self.model = create_reward_model(args.model_name_or_path)
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Create data loaders
        self.train_dataloader = create_dataloader(
            data_path=args.train_data_path,
            tokenizer=self.tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            max_rejected=args.max_rejected,
            shuffle=True,
            num_workers=args.num_workers,
            training_mode=args.training_mode
        )
        
        if args.eval_data_path:
            self.eval_dataloader = create_dataloader(
                data_path=args.eval_data_path,
                tokenizer=self.tokenizer,
                batch_size=args.batch_size,
                max_length=args.max_length,
                max_rejected=args.max_rejected,
                shuffle=False,
                num_workers=args.num_workers,
                training_mode=args.training_mode
            )
        else:
            self.eval_dataloader = None
        
        # Calculate total training steps BEFORE accelerator.prepare()
        # This ensures we get the correct total across all processes
        total_batches_per_epoch = len(self.train_dataloader)
        self.total_steps = total_batches_per_epoch * args.num_epochs
        
        # Initialize scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=self.total_steps
        )
        
        # Prepare everything with accelerator
        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.scheduler
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.scheduler
        )
        
        # After accelerator.prepare(), calculate steps per process for progress bar
        # Each process will handle len(self.train_dataloader) batches per epoch
        self.steps_per_process = len(self.train_dataloader) * args.num_epochs
        
        if self.eval_dataloader:
            self.eval_dataloader = self.accelerator.prepare(self.eval_dataloader)
        
        # Initialize best metrics tracking
        self.best_eval_loss = float('inf')
        self.global_step = 0
    
    def compute_contrastive_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute contrastive loss with optimized unified input processing.
        
        The batch now contains unified input_ids and attention_mask where:
        - Index 0 for each example is the chosen answer
        - Remaining indices are rejected answers
        """
        device = self.accelerator.device
        
        # Get all scores in one forward pass
        all_scores = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )  # (total_examples,)
        
        # Reconstruct the contrastive groups
        example_lengths = batch['example_lengths']
        batch_size = len(example_lengths)
        
        all_contrastive_losses = []
        start_idx = 0
        
        for i in range(batch_size):
            example_length = example_lengths[i].item()  # 1 chosen + num_rejected
            
            # Get scores for this example group
            example_scores = all_scores[start_idx:start_idx + example_length]
            
            # The first score is chosen, the rest are rejected
            chosen_score = example_scores[0:1]  # Keep as tensor
            rejected_scores = example_scores[1:]  # All rejected scores
            
            # Combine rejected scores (first) with chosen score (last)
            # This follows the pattern: [rejected_1, rejected_2, ..., chosen]
            contrastive_scores = torch.cat([rejected_scores, chosen_score], dim=0)
            
            # Apply temperature scaling
            contrastive_scores = contrastive_scores / self.args.temperature
            
            # The target is the last position (chosen answer)
            num_rejected = example_length - 1
            target = torch.tensor(num_rejected, dtype=torch.long, device=device)
            
            # Compute cross-entropy loss
            loss = nn.CrossEntropyLoss()(contrastive_scores.unsqueeze(0), target.unsqueeze(0))
            all_contrastive_losses.append(loss)
            
            start_idx += example_length
        
        # Average the losses
        contrastive_loss = torch.stack(all_contrastive_losses).mean()
        
        return contrastive_loss
    
    def compute_pairwise_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute pairwise margin loss with optimized unified input processing.
        
        The batch now contains unified input_ids and attention_mask where:
        - Index 0 for each example is the chosen answer
        - Index 1 for each example is the first rejected answer
        
        This implements margin loss: max(0, margin - (chosen_score - rejected_score))
        """
        # Get all scores in one forward pass
        all_scores = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )  # (total_examples,) where total_examples = batch_size * 2
        
        # Reconstruct pairs: each example has 2 scores (chosen, rejected)
        example_lengths = batch['example_lengths']
        batch_size = len(example_lengths)
        
        chosen_scores = []
        rejected_scores = []
        start_idx = 0
        
        for i in range(batch_size):
            example_length = example_lengths[i].item()  # Should be 2 for pairwise
            
            if example_length == 2:
                # First score is chosen, second is rejected
                chosen_scores.append(all_scores[start_idx])
                rejected_scores.append(all_scores[start_idx + 1])
            else:
                # Fallback (shouldn't happen in practice)
                chosen_scores.append(all_scores[start_idx])
                rejected_scores.append(all_scores[start_idx])
            
            start_idx += example_length
        
        # Stack scores
        chosen_scores = torch.stack(chosen_scores)
        rejected_scores = torch.stack(rejected_scores)
        
        # Compute margin loss: max(0, margin - (chosen_score - rejected_score))
        margin_loss = torch.clamp(
            self.args.margin - (chosen_scores - rejected_scores), 
            min=0.0
        ).mean()
        
        return margin_loss
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Perform a single training step."""
        self.model.train()
        
        # Compute loss based on training mode
        if self.args.training_mode == "contrast":
            loss = self.compute_contrastive_loss(batch)
        elif self.args.training_mode == "pairwise":
            loss = self.compute_pairwise_loss(batch)
        else:
            raise ValueError(f"Unknown training mode: {self.args.training_mode}")
        
        # Backward pass
        self.accelerator.backward(loss)
        
        # Gradient clipping
        if self.args.max_grad_norm > 0:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        return loss.item()
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model."""
        if self.eval_dataloader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating", disable=not self.accelerator.is_local_main_process):
                # Use the same loss function as training for evaluation
                if self.args.training_mode == "contrast":
                    loss = self.compute_contrastive_loss(batch)
                elif self.args.training_mode == "pairwise":
                    loss = self.compute_pairwise_loss(batch)
                else:
                    raise ValueError(f"Unknown training mode: {self.args.training_mode}")
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Gather results across processes
        avg_loss = self.accelerator.gather(torch.tensor(avg_loss, device=self.accelerator.device)).mean().item()
        
        return {"eval_loss": avg_loss}
    
    def save_model(self, output_dir: str, is_best: bool = False):
        """Save the model."""
        if self.accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
            
            # Unwrap model for saving
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            if is_best:
                logger.info(f"Saved best model to {output_dir}")
            else:
                logger.info(f"Saved model checkpoint to {output_dir}")
    
    def train(self):
        """Main training loop."""
        logger.info("***** Running training *****")
        logger.info(f"  Training mode = {self.args.training_mode}")
        if self.args.training_mode == "contrast":
            logger.info(f"  Temperature = {self.args.temperature}")
        elif self.args.training_mode == "pairwise":
            logger.info(f"  Margin = {self.args.margin}")
        logger.info(f"  Num examples = {len(self.train_dataloader.dataset)}")
        logger.info(f"  Num Epochs = {self.args.num_epochs}")
        logger.info(f"  Batch size = {self.args.batch_size}")
        logger.info(f"  Total optimization steps = {self.total_steps}")
        
        progress_bar = tqdm(
            range(self.steps_per_process),
            desc="Training",
            disable=not self.accelerator.is_local_main_process
        )
        
        for epoch in range(self.args.num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in self.train_dataloader:
                loss = self.train_step(batch)
                epoch_loss += loss
                num_batches += 1
                self.global_step += 1
                
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": f"{loss:.4f}"})
                
                # Evaluation
                if (self.args.eval_steps > 0 and 
                    self.global_step % self.args.eval_steps == 0):
                    
                    eval_results = self.evaluate()
                    
                    if eval_results and self.accelerator.is_main_process:
                        logger.info(f"Step {self.global_step}: {eval_results}")
                        
                        # Save best model
                        if eval_results["eval_loss"] < self.best_eval_loss:
                            self.best_eval_loss = eval_results["eval_loss"]
                            best_model_dir = os.path.join(self.args.output_dir, "best_model")
                            self.save_model(best_model_dir, is_best=True)
                
                # Save checkpoint
                if (self.args.save_steps > 0 and 
                    self.global_step % self.args.save_steps == 0):
                    
                    checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-{self.global_step}")
                    self.save_model(checkpoint_dir)
            
            # End of epoch logging
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            if self.accelerator.is_main_process:
                logger.info(f"Epoch {epoch + 1}/{self.args.num_epochs} - Average loss: {avg_epoch_loss:.4f}")
        
        # Save final model
        self.save_model(self.args.output_dir)
        
        logger.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(description="Train Reward Model with Contrastive Learning")
    
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default="/workspace/HFModels/Qwen2.5-3B-Instruct",
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    
    # Data arguments
    parser.add_argument("--train_data_path", type=str, default="./data/train_data.jsonl",
                        help="Path to training data (JSONL format)")
    parser.add_argument("--eval_data_path", type=str, default=None,
                        help="Path to evaluation data (JSONL format)")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--max_rejected", type=int, default=None,
                        help="Maximum number of rejected answers per example")
    
    # Training arguments
    parser.add_argument("--training_mode", type=str, default="contrast", choices=["contrast", "pairwise"],
                        help="Training mode: 'contrast' for contrastive learning with all rejected answers, "
                             "'pairwise' for margin loss with only first rejected answer")
    parser.add_argument("--margin", type=float, default=1.0,
                        help="Margin for pairwise margin loss (only used when training_mode='pairwise')")
    parser.add_argument("--output_dir", type=str, default="./output/Qwen2.5-3B-Instruct-contrast",
                        help="Output directory for model checkpoints")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Number of warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for clipping")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Temperature for contrastive learning (only used when training_mode='contrast')")
    
    # Evaluation and saving
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Evaluation frequency (0 to disable)")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Save frequency (0 to disable)")
    
    # Mixed precision arguments
    parser.add_argument("--fp16", action="store_true",
                        help="Enable FP16 mixed precision training")
    parser.add_argument("--bf16", action="store_true",
                        help="Enable BF16 mixed precision training")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of data loading workers")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize trainer and start training
    trainer = RewardModelTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
