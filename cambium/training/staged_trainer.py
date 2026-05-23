"""
StagedTrainer - Orchestrates multi-phase training with progressive unfreezing.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Union
import logging

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from cambium.core.freezing import FreezingManager

logger = logging.getLogger(__name__)


@dataclass
class TrainingPhase:
    """
    Configuration for a single training phase.

    Each phase can have different freezing patterns and learning rates,
    enabling progressive training strategies.
    """

    name: str
    """Name of the phase for logging."""

    epochs: int = 1
    """Number of epochs for this phase."""

    freeze: Optional[str] = None
    """What to freeze: 'original', 'all', 'none', or None (no change)."""

    unfreeze_groups: Optional[List[int]] = None
    """Layer group indices to unfreeze (for progressive unfreezing)."""

    lr: float = 1e-4
    """Learning rate for this phase."""

    discriminative_lr: Optional[Dict[str, float]] = field(default_factory=dict)
    """
    Discriminative LR config mapping patterns to learning rates.
    Example: {"embeddings": 1e-7, "original_layers": 1e-6, "new_layers": 1e-4}
    """

    batch_size: int = 8
    """Batch size for this phase."""

    gradient_accumulation_steps: int = 1
    """Gradient accumulation steps."""

    warmup_steps: int = 100
    """Warmup steps for this phase."""

    max_grad_norm: float = 1.0
    """Maximum gradient norm for clipping."""

    eval_every: int = 100
    """Evaluate every N steps."""

    save_every: int = 500
    """Save checkpoint every N steps."""

    callbacks: List[Callable] = field(default_factory=list)
    """Optional callbacks to run at phase start/end."""


class StagedTrainer:
    """
    Orchestrates multi-phase training with automatic freezing/unfreezing.

    Supports the standard expansion training recipe:

    - Phase 1: Train only new layers (original frozen)
    - Phase 2: Progressive unfreezing of original layers
    - Phase 3: Full fine-tuning

    Example::

        trainer = StagedTrainer(model)

        # Phase 1: Train only new layers
        trainer.add_phase(
            name="warmup_new_layers",
            freeze="original",
            lr=1e-4,
            epochs=2,
        )

        # Phase 2: Unfreeze last 4 layers
        trainer.add_phase(
            name="unfreeze_tail",
            freeze=None,  # Keep previous freeze state
            unfreeze_groups=[-4, -3, -2, -1],
            lr=5e-5,
            epochs=1,
        )

        # Phase 3: Full fine-tuning
        trainer.add_phase(
            name="full_finetune",
            freeze="none",
            lr=1e-6,
            epochs=1,
        )

        trainer.train(train_dataloader, eval_dataloader)
    """

    def __init__(
        self,
        model: Union[nn.Module, Any],
        freezing_manager: Optional[FreezingManager] = None,
        optimizer_class: type = torch.optim.AdamW,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the staged trainer.

        Args:
            model: The model to train (or ExpandableModel)
            freezing_manager: FreezingManager instance
            optimizer_class: Optimizer class to use
            device: Device to train on (auto-detected if None)
        """
        # Handle both raw models and ExpandableModel wrappers
        if hasattr(model, "get_model"):
            self.model = model.get_model()
            self.freezing_manager = freezing_manager or model.freezing_manager
        else:
            self.model = model
            self.freezing_manager = freezing_manager or FreezingManager(model)

        self.optimizer_class = optimizer_class
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.phases: List[TrainingPhase] = []
        self.current_phase_idx = 0

        # Training state
        self.global_step = 0
        self.current_epoch = 0

    def add_phase(self, **kwargs) -> None:
        """
        Add a training phase.

        Args:
            **kwargs: TrainingPhase parameters
        """
        phase = TrainingPhase(**kwargs)
        self.phases.append(phase)
        logger.info(f"Added phase: {phase.name}")

    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Run all training phases.

        Args:
            train_dataloader: Training data
            eval_dataloader: Optional evaluation data
            optimizer: Optional pre-configured optimizer
            scheduler: Optional learning rate scheduler

        Returns:
            Training history dictionary
        """
        history = {
            "phases": [],
            "final_loss": None,
        }

        for phase_idx, phase in enumerate(self.phases):
            self.current_phase_idx = phase_idx
            logger.info(f"\n{'='*60}")
            logger.info(f"Starting Phase {phase_idx + 1}: {phase.name}")
            logger.info(f"{'='*60}")

            # Configure phase
            self._apply_phase_config(phase)

            # Setup optimizer
            if optimizer is None or phase.discriminative_lr:
                optimizer = self._create_optimizer(phase)

            # Train phase
            phase_history = self._train_phase(
                phase,
                train_dataloader,
                eval_dataloader,
                optimizer,
                scheduler,
            )

            history["phases"].append(phase_history)

        logger.info("\n" + "="*60)
        logger.info("Training Complete!")
        logger.info("="*60)

        return history

    def _apply_phase_config(self, phase: TrainingPhase) -> None:
        """Configure model for this phase."""
        # Apply freezing
        if phase.freeze == "original":
            self.freezing_manager.freeze_original_layers()
        elif phase.freeze == "all":
            self.freezing_manager.freeze_all()
        elif phase.freeze == "none":
            self.freezing_manager.unfreeze_all()
        # phase.freeze = None means no change

        # Apply unfreeze groups
        if phase.unfreeze_groups:
            for group_idx in phase.unfreeze_groups:
                self.freezing_manager.unfreeze_group(group_idx)

        # Log status
        info = self.freezing_manager.get_trainable_params()
        logger.info(f"Phase config applied:")
        logger.info(f"  Trainable params: {info['trainable_params']:,} ({info['percent_trainable']:.2f}%)")
        logger.info(f"  Frozen params: {info['frozen_params']:,}")

    def _create_optimizer(self, phase: TrainingPhase) -> Optimizer:
        """Create optimizer with appropriate LR settings."""
        if phase.discriminative_lr:
            param_groups = self.freezing_manager.get_parameter_groups_for_discriminative_lr(
                phase.discriminative_lr
            )
        else:
            param_groups = [{
                "params": [p for p in self.model.parameters() if p.requires_grad],
                "lr": phase.lr,
            }]

        return self.optimizer_class(param_groups, lr=phase.lr)

    def _train_phase(
        self,
        phase: TrainingPhase,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader],
        optimizer: Optimizer,
        scheduler: Optional[Any],
    ) -> Dict[str, Any]:
        """Train a single phase."""
        self.model.to(self.device)
        self.model.train()

        history = {
            "name": phase.name,
            "losses": [],
            "steps": 0,
        }

        step_in_phase = 0

        for epoch in range(phase.epochs):
            self.current_epoch = epoch

            for batch_idx, batch in enumerate(train_dataloader):
                # Move batch to device
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}
                elif isinstance(batch, (list, tuple)):
                    batch = [v.to(self.device) if isinstance(v, torch.Tensor) else v
                            for v in batch]

                # Forward pass
                loss = self._compute_loss(batch)

                # Scale loss for gradient accumulation
                loss = loss / phase.gradient_accumulation_steps

                # Backward pass
                loss.backward()

                # Update weights
                if (batch_idx + 1) % phase.gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        phase.max_grad_norm,
                    )

                    optimizer.step()
                    optimizer.zero_grad()

                    if scheduler is not None:
                        scheduler.step()

                    self.global_step += 1
                    step_in_phase += 1

                # Logging
                if step_in_phase % 10 == 0:
                    history["losses"].append(loss.item() * phase.gradient_accumulation_steps)
                    logger.debug(f"Step {step_in_phase}: loss={loss.item():.4f}")

                # Evaluation
                if eval_dataloader and step_in_phase % phase.eval_every == 0:
                    eval_loss = self._evaluate(eval_dataloader)
                    logger.info(f"Step {step_in_phase}: eval_loss={eval_loss:.4f}")

        history["steps"] = step_in_phase
        return history

    def _compute_loss(self, batch: Any) -> torch.Tensor:
        """Compute loss for a batch."""
        if isinstance(batch, dict):
            outputs = self.model(**batch)
        elif isinstance(batch, (list, tuple)) and len(batch) == 2:
            input_ids, labels = batch
            outputs = self.model(input_ids=input_ids, labels=labels)
        else:
            outputs = self.model(batch)

        # Handle different output formats
        if hasattr(outputs, "loss"):
            return outputs.loss
        elif isinstance(outputs, torch.Tensor):
            return outputs.mean()
        else:
            raise ValueError(f"Cannot determine loss from output: {type(outputs)}")

    def _evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}
                elif isinstance(batch, (list, tuple)):
                    batch = [v.to(self.device) if isinstance(v, torch.Tensor) else v
                            for v in batch]

                loss = self._compute_loss(batch)
                total_loss += loss.item()
                num_batches += 1

        self.model.train()
        return total_loss / num_batches if num_batches > 0 else float('inf')

    def save_checkpoint(self, path: str, metadata: Optional[Dict] = None) -> None:
        """Save training checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "current_phase": self.current_phase_idx,
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "metadata": metadata or {},
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.current_phase_idx = checkpoint.get("current_phase", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.current_epoch = checkpoint.get("current_epoch", 0)
        logger.info(f"Loaded checkpoint from {path}")
        return checkpoint.get("metadata", {})
