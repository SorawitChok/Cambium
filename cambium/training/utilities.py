"""
Training utilities for integration with HF Trainer, TRL, and custom loops.
"""

from typing import Any, Callable, Dict, List, Optional, Union
import logging

import torch
from torch import nn
from torch.optim import Optimizer

from cambium.core.freezing import FreezingManager

logger = logging.getLogger(__name__)


class TrainingUtilities:
    """
    Integration helpers for various training frameworks.

    Provides utilities for:
    - Discriminative learning rates
    - PEFT/LoRA compatibility
    - Memory optimizations
    - Integration with HF Trainer and TRL
    """

    @staticmethod
    def get_optimizer_with_discriminative_lr(
        model: nn.Module,
        lr_config: Dict[str, float],
        optimizer_class: type = torch.optim.AdamW,
        weight_decay: float = 0.01,
    ) -> Optimizer:
        """
        Create an optimizer with different learning rates for different parameter groups.

        This is useful for applying different learning rates to:

        - Embeddings (very low LR, should change minimally)
        - Original layers (low LR, small changes)
        - New layers (higher LR, more aggressive learning)
        - LM head (low LR, preserve output distribution)

        Args:
            model: The model to optimize
            lr_config: Dict mapping regex patterns to learning rates.
                Example value::

                    {
                        "embed_tokens|lm_head": 1e-7,
                        "model\\.layers\\.\\d+": 1e-6,
                        "new_|expanded_": 1e-4,
                    }
            optimizer_class: Optimizer class to use
            weight_decay: Weight decay coefficient

        Returns:
            Configured optimizer
        """
        import re

        param_groups: List[Dict[str, Any]] = []
        assigned_params: set = set()

        # Sort patterns by specificity (more specific first)
        sorted_patterns = sorted(
            lr_config.items(),
            key=lambda x: len(x[0]),
            reverse=True,
        )

        for pattern, lr in sorted_patterns:
            params = []
            regex = re.compile(pattern)

            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue

                if regex.search(name) and id(param) not in assigned_params:
                    params.append(param)
                    assigned_params.add(id(param))

            if params:
                param_groups.append(
                    {
                        "params": params,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "name": pattern,
                    }
                )
                logger.debug(f"LR group '{pattern}': {len(params)} params, lr={lr}")

        # Add remaining parameters with default LR
        remaining = []
        for name, param in model.named_parameters():
            if param.requires_grad and id(param) not in assigned_params:
                remaining.append(param)
                assigned_params.add(id(param))

        if remaining:
            default_lr = 1e-4
            param_groups.append(
                {
                    "params": remaining,
                    "lr": default_lr,
                    "weight_decay": weight_decay,
                    "name": "default",
                }
            )
            logger.debug(f"Default group: {len(remaining)} params, lr={default_lr}")

        return optimizer_class(param_groups)

    @staticmethod
    def prepare_for_peft(
        model: nn.Module,
        lora_config: Optional[Dict] = None,
    ) -> nn.Module:
        """
        Prepare an expanded model for PEFT/LoRA fine-tuning.

        This ensures the model is compatible with PEFT library.

        Args:
            model: The expanded model
            lora_config: Optional LoRA configuration

        Returns:
            Model ready for PEFT
        """
        try:
            from peft import get_peft_model, LoraConfig
        except ImportError:
            logger.warning("PEFT library not available. Install with: pip install peft")
            return model

        if lora_config is None:
            # Default config for expanded models
            lora_config = {
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
                "lora_dropout": 0.05,
                "bias": "none",
                "task_type": "CAUSAL_LM",
            }

        peft_config = LoraConfig(**lora_config)
        model = get_peft_model(model, peft_config)

        logger.info("Applied PEFT/LoRA configuration")
        return model

    @staticmethod
    def enable_memory_optimizations(
        model: nn.Module,
        gradient_checkpointing: bool = True,
        cpu_offload: bool = False,
        mixed_precision: str = "fp16",
    ) -> None:
        """
        Apply memory optimizations for training large expanded models.

        Args:
            model: The model to optimize
            gradient_checkpointing: Enable gradient checkpointing
            cpu_offload: Offload optimizer states to CPU
            mixed_precision: Mixed precision mode ('fp16', 'bf16', 'none')
        """
        # Enable gradient checkpointing
        if gradient_checkpointing:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                logger.info("Enabled gradient checkpointing")
            elif hasattr(model, "model") and hasattr(model.model, "gradient_checkpointing_enable"):
                model.model.gradient_checkpointing_enable()
                logger.info("Enabled gradient checkpointing")

        # Mixed precision (requires Accelerate or PyTorch AMP)
        if mixed_precision in ["fp16", "bf16"]:
            logger.info(
                f"Mixed precision ({mixed_precision}) should be configured in training loop"
            )

    @staticmethod
    def get_staged_lr_schedule(
        base_lr: float = 1e-4,
        warmup_steps: int = 100,
        total_steps: int = 1000,
        min_lr_ratio: float = 0.1,
    ) -> Callable:
        """
        Get a learning rate scheduler with warmup and cosine decay.

        Args:
            base_lr: Peak learning rate
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
            min_lr_ratio: Minimum LR as ratio of base_lr

        Returns:
            Lambda function for LR scheduling
        """
        import math

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine decay
                progress = float(current_step - warmup_steps) / float(
                    max(1, total_steps - warmup_steps)
                )
                return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (
                    1.0 + math.cos(math.pi * progress)
                )

        return lr_lambda

    @staticmethod
    def integrate_with_hf_trainer(
        model: nn.Module,
        training_args: Any,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
        **kwargs,
    ) -> Any:
        """
        Integrate expanded model with Hugging Face Trainer.

        Args:
            model: The expanded model
            training_args: HF TrainingArguments
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            **kwargs: Additional arguments for Trainer

        Returns:
            HF Trainer instance
        """
        try:
            from transformers import Trainer
        except ImportError:
            raise ImportError(
                "transformers library required. Install with: pip install transformers"
            )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **kwargs,
        )

        return trainer

    @staticmethod
    def integrate_with_trl(
        model: nn.Module,
        tokenizer: Any,
        training_args: Any,
        train_dataset: Any,
    ) -> Any:
        """
        Integrate expanded model with TRL (Transformer Reinforcement Learning).

        Args:
            model: The expanded model
            tokenizer: Tokenizer for the model
            training_args: TRL training arguments
            train_dataset: Training dataset

        Returns:
            TRL trainer instance
        """
        try:
            from trl import SFTTrainer
        except ImportError:
            raise ImportError("trl library required. Install with: pip install trl")

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
        )

        return trainer

    @staticmethod
    def create_optimizer_groups_for_staged_training(
        model: nn.Module,
        phase: int = 1,
        base_lr: float = 1e-4,
    ) -> List[Dict[str, Any]]:
        """
        Create optimizer parameter groups for different training phases.

        Phase 1: Only new layers (high LR)
        Phase 2: Last N layers + new layers (medium LR)
        Phase 3: All layers (low LR with discriminative)

        Args:
            model: The expanded model
            phase: Which phase (1, 2, or 3)
            base_lr: Base learning rate

        Returns:
            List of parameter groups
        """
        import re

        groups = []

        if phase == 1:
            # Phase 1: Only new layers
            new_params = []
            for name, param in model.named_parameters():
                if param.requires_grad and re.search(r"new_|expanded_", name):
                    new_params.append(param)

            groups.append(
                {
                    "params": new_params,
                    "lr": base_lr,
                    "name": "new_layers",
                }
            )

        elif phase == 2:
            # Phase 2: New layers + last quarter of original
            new_params = []
            tail_params = []

            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue

                if re.search(r"new_|expanded_", name):
                    new_params.append(param)
                elif re.search(r"model\.layers\.(1[8-9]|2[0-9])", name):
                    # Assuming 24 layers, last 6
                    tail_params.append(param)

            groups.append(
                {
                    "params": new_params,
                    "lr": base_lr,
                    "name": "new_layers",
                }
            )
            groups.append(
                {
                    "params": tail_params,
                    "lr": base_lr * 0.5,
                    "name": "tail_original",
                }
            )

        else:
            # Phase 3: All layers with discriminative LR
            lr_config = {
                r"embed|lm_head": base_lr * 0.1,
                r"model\\.layers": base_lr * 0.5,
                r"new_|expanded_": base_lr,
            }
            return TrainingUtilities.get_optimizer_with_discriminative_lr(model, lr_config)

        return groups

    @staticmethod
    def print_model_info(model: nn.Module) -> None:
        """Print useful information about the model."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("=" * 60)
        print("Model Information")
        print("=" * 60)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {total_params - trainable_params:,}")
        print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")

        # Layer count
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            print(f"Number of layers: {len(model.model.layers)}")

        # Model size estimate
        param_size_mb = total_params * 4 / (1024**2)  # Assuming fp32
        print(f"Model size (fp32): ~{param_size_mb:.1f} MB")
        print("=" * 60)
