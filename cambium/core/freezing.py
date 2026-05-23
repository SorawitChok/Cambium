"""
Advanced freezing and unfreezing utilities for staged training.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any
import torch
from torch import nn

logger = logging.getLogger(__name__)


class FreezingManager:
    """
    Manages parameter freezing with fine-grained control.

    Supports freezing by layer indices, patterns, or groups for
    progressive unfreezing strategies.
    """

    def __init__(self, model: Optional[nn.Module] = None):
        """
        Initialize the freezing manager.

        Args:
            model: Optional model to manage. Can be set later with set_model().
        """
        self.model: nn.Module
        if model is None:
            # Temporary placeholder until set_model is called
            self.model = nn.Identity()
        else:
            self.model = model
        self.original_requires_grad: Dict[str, bool] = {}
        if model is not None:
            self._record_original_state()

    def set_model(self, model: nn.Module) -> None:
        """Set or change the model being managed."""
        self.model = model
        self._record_original_state()

    def _record_original_state(self) -> None:
        """Record the original requires_grad state of all parameters."""
        self.original_requires_grad = {
            name: param.requires_grad
            for name, param in self.model.named_parameters()
        }

    def freeze_all(self) -> None:
        """Freeze all parameters in the model."""
        for param in self.model.parameters():
            param.requires_grad = False
        logger.debug("Froze all parameters")

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters in the model."""
        for param in self.model.parameters():
            param.requires_grad = True
        logger.debug("Unfroze all parameters")

    def freeze_by_pattern(self, pattern: str) -> List[str]:
        """
        Freeze parameters matching a regex pattern.

        Args:
            pattern: Regex pattern to match parameter names

        Returns:
            List of frozen parameter names
        """
        frozen = []
        regex = re.compile(pattern)

        for name, param in self.model.named_parameters():
            if regex.search(name):
                param.requires_grad = False
                frozen.append(name)

        logger.debug(f"Froze {len(frozen)} parameters matching pattern '{pattern}'")
        return frozen

    def unfreeze_by_pattern(self, pattern: str) -> List[str]:
        """
        Unfreeze parameters matching a regex pattern.

        Args:
            pattern: Regex pattern to match parameter names

        Returns:
            List of unfrozen parameter names
        """
        unfrozen = []
        regex = re.compile(pattern)

        for name, param in self.model.named_parameters():
            if regex.search(name):
                param.requires_grad = True
                unfrozen.append(name)

        logger.debug(f"Unfroze {len(unfrozen)} parameters matching pattern '{pattern}'")
        return unfrozen

    def freeze_original_layers(self, new_layer_prefix: str = "new_") -> None:
        """
        Freeze all original pretrained weights, leaving only new layers trainable.

        This is useful for Phase 1 training where only newly added layers are trained.

        Args:
            new_layer_prefix: Prefix used to identify new layers (default: ``"new_"``)
        """
        # Freeze everything
        self.freeze_all()

        # Unfreeze new layers
        new_layers = self.unfreeze_by_pattern(f".*{new_layer_prefix}.*")

        logger.info(f"Phase 1 setup: Froze original weights, {len(new_layers)} new layers trainable")

    def freeze_embeddings(self) -> None:
        """Freeze embedding layers (typically want to keep these fixed)."""
        self.freeze_by_pattern(r".*embed.*")
        self.freeze_by_pattern(r".*lm_head.*")
        logger.debug("Froze embedding and lm_head layers")

    def unfreeze_layer_range(self, start_idx: int, end_idx: int, layer_pattern: str = r"model\.layers\.(\d+)") -> None:
        """
        Unfreeze a specific range of transformer layers.

        Args:
            start_idx: Starting layer index (inclusive)
            end_idx: Ending layer index (inclusive)
            layer_pattern: Regex pattern to extract layer indices from parameter names
        """
        regex = re.compile(layer_pattern)

        for name, param in self.model.named_parameters():
            match = regex.search(name)
            if match:
                layer_idx = int(match.group(1))
                if start_idx <= layer_idx <= end_idx:
                    param.requires_grad = True

        logger.info(f"Unfroze layers {start_idx} to {end_idx}")

    def unfreeze_group(self, group_idx: int, num_groups: int = 4) -> None:
        """
        Unfreeze a specific group of layers (for progressive unfreezing).

        Divides the model into num_groups equal groups and unfreezes one group.

        Args:
            group_idx: Which group to unfreeze (0 to num_groups-1)
            num_groups: Total number of groups to divide the model into
        """
        # Find all transformer layers
        layer_indices = []
        pattern = re.compile(r".*model\.layers\.(\d+).*")

        for name, _ in self.model.named_parameters():
            match = pattern.search(name)
            if match:
                idx = int(match.group(1))
                if idx not in layer_indices:
                    layer_indices.append(idx)

        if not layer_indices:
            logger.warning("No transformer layers found with pattern 'model.layers'")
            return

        layer_indices.sort()
        total_layers = len(layer_indices)
        layers_per_group = total_layers // num_groups

        # Calculate range for this group
        start_idx = layer_indices[group_idx * layers_per_group]
        if group_idx == num_groups - 1:
            end_idx = layer_indices[-1]
        else:
            end_idx = layer_indices[(group_idx + 1) * layers_per_group - 1]

        self.unfreeze_layer_range(start_idx, end_idx)
        logger.info(f"Unfroze group {group_idx}/{num_groups} (layers {start_idx}-{end_idx})")

    def get_trainable_params(self) -> Dict[str, Any]:
        """
        Get information about trainable parameters.

        Returns:
            Dictionary with trainable parameter counts and names
        """
        trainable = []
        frozen = []
        trainable_count = 0
        frozen_count = 0

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable.append(name)
                trainable_count += param.numel()
            else:
                frozen.append(name)
                frozen_count += param.numel()

        return {
            "trainable_params": trainable_count,
            "frozen_params": frozen_count,
            "trainable_names": trainable,
            "frozen_names": frozen,
            "percent_trainable": 100 * trainable_count / (trainable_count + frozen_count)
            if (trainable_count + frozen_count) > 0 else 0,
        }

    def get_parameter_groups_for_discriminative_lr(
        self,
        lr_config: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """
        Create parameter groups with different learning rates.

        Args:
            lr_config: Dict mapping patterns to learning rates.
                      Example: {"embeddings": 1e-7, "original": 1e-6, "new": 1e-4}

        Returns:
            List of parameter group dicts for optimizer
        """
        groups = []
        assigned_params = set()

        for pattern, lr in lr_config.items():
            params = []
            regex = re.compile(pattern)

            for name, param in self.model.named_parameters():
                if param.requires_grad and regex.search(name) and name not in assigned_params:
                    params.append(param)
                    assigned_params.add(name)

            if params:
                groups.append({"params": params, "lr": lr, "name": pattern})

        # Add remaining parameters with default LR
        remaining = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and name not in assigned_params:
                remaining.append(param)

        if remaining:
            groups.append({"params": remaining, "lr": 1e-4, "name": "default"})

        return groups

    def print_trainable_status(self) -> None:
        """Print a summary of trainable vs frozen parameters."""
        info = self.get_trainable_params()

        print("=" * 60)
        print("Parameter Freezing Status")
        print("=" * 60)
        print(f"Trainable parameters: {info['trainable_params']:,} ({info['percent_trainable']:.2f}%)")
        print(f"Frozen parameters: {info['frozen_params']:,}")
        print(f"Trainable parameter groups:")
        for name in info["trainable_names"][:10]:  # Show first 10
            print(f"  - {name}")
        if len(info["trainable_names"]) > 10:
            print(f"  ... and {len(info['trainable_names']) - 10} more")
        print("=" * 60)

    def save_state(self, path: str) -> None:
        """Save the current freezing state to a file."""
        state = {
            name: param.requires_grad
            for name, param in self.model.named_parameters()
        }
        torch.save(state, path)
        logger.info(f"Saved freezing state to {path}")

    def load_state(self, path: str) -> None:
        """Load a freezing state from a file."""
        state = torch.load(path)

        for name, param in self.model.named_parameters():
            if name in state:
                param.requires_grad = state[name]
            else:
                logger.warning(f"Parameter {name} not found in saved state")

        logger.info(f"Loaded freezing state from {path}")


def freeze_model(model: nn.Module, freeze_embeddings: bool = True) -> FreezingManager:
    """
    Convenience function to freeze a model and return a FreezingManager.

    Args:
        model: The model to freeze
        freeze_embeddings: Whether to also freeze embeddings and lm_head

    Returns:
        FreezingManager configured with the frozen model
    """
    manager = FreezingManager(model)
    manager.freeze_all()

    if freeze_embeddings:
        # Keep embeddings frozen (they're already frozen by freeze_all)
        pass

    return manager
