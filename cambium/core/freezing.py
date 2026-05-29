"""
Advanced freezing and unfreezing utilities for staged training.
"""

import logging
import re
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


class FreezingManager:
    """
    Manages parameter freezing with fine-grained control.

    Supports freezing by layer indices, patterns, or groups for
    progressive unfreezing strategies.
    """

    def __init__(self, model: nn.Module | None = None):
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
        self.original_requires_grad: dict[str, bool] = {}
        if model is not None:
            self._record_original_state()

    def set_model(self, model: nn.Module) -> None:
        """Set or change the model being managed."""
        self.model = model
        self._record_original_state()

    def _record_original_state(self) -> None:
        """Record the original requires_grad state of all parameters."""
        self.original_requires_grad = {
            name: param.requires_grad for name, param in self.model.named_parameters()
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

    def freeze_by_pattern(self, pattern: str) -> list[str]:
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

    def unfreeze_by_pattern(self, pattern: str) -> list[str]:
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

    def freeze_original_layers(self) -> None:
        """
        Freeze all original pretrained weights, leaving only new layers trainable.

        This is useful for Phase 1 training where only newly added layers are trained.

        New layers are identified by the ``_cambium_new`` attribute set during
        block insertion, rather than by name patterns.
        """
        # Freeze everything
        self.freeze_all()

        # Unfreeze parameters belonging to layers marked as new by the engine
        new_params_count = 0
        for name, module in self.model.named_modules():
            if getattr(module, "_cambium_new", False):
                for param in module.parameters(recurse=True):
                    param.requires_grad = True
                    new_params_count += param.numel()

        logger.info(
            f"Phase 1 setup: Froze original weights, "
            f"{new_params_count} new parameters trainable"
        )

    def freeze_embeddings(self) -> None:
        """Freeze embedding layers (typically want to keep these fixed)."""
        self.freeze_by_pattern(r".*embed.*")
        self.freeze_by_pattern(r".*lm_head.*")
        logger.debug("Froze embedding and lm_head layers")

    def unfreeze_layer_range(
        self, start_idx: int, end_idx: int, layer_pattern: str = r"model\.layers\.(\d+)"
    ) -> None:
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

    def get_trainable_params(self) -> dict[str, Any]:
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
            if (trainable_count + frozen_count) > 0
            else 0,
        }

    def get_parameter_groups_for_discriminative_lr(
        self,
        lr_config: dict[str | tuple[int, int], float],
    ) -> list[dict[str, Any]]:
        """
        Create parameter groups with different learning rates.

        Supports three key types for maximum flexibility:

        1. **Layer index tuples** ``(start_idx, end_idx)`` — most intuitive::

            lr_config = {
                (0, 19): 1e-6,          # Layers 0-19
                (20, 31): 5e-6,         # Layers 20-31
            }

        2. **Semantic names** — special strings with built-in meaning::

            lr_config = {
                "embeddings": 1e-8,       # embed_tokens + lm_head
                "new_layers": 1e-4,     # Layers marked _cambium_new
                "original_layers": 1e-6, # All original transformer layers
            }

        3. **Regex patterns** — full control via regex::

            lr_config = {
                r"embed_tokens|lm_head": 1e-8,
                r"model\.layers\.\d+": 1e-6,
            }

        Args:
            lr_config: Dict mapping patterns/tuples/names to learning rates.

        Returns:
            List of parameter group dicts for optimizer
        """
        groups = []
        assigned_params = set()

        for key, lr in lr_config.items():
            params = []
            group_name = str(key)

            if isinstance(key, tuple) and len(key) == 2:
                # Layer index range: (start_idx, end_idx)
                start_idx, end_idx = key
                layer_pattern = re.compile(r".*model\.layers\.(\d+)\.")

                for name, param in self.model.named_parameters():
                    if param.requires_grad and name not in assigned_params:
                        match = layer_pattern.search(name)
                        if match:
                            layer_idx = int(match.group(1))
                            if start_idx <= layer_idx <= end_idx:
                                params.append(param)
                                assigned_params.add(name)

            elif isinstance(key, str):
                if key == "embeddings":
                    # Match embed_tokens and lm_head
                    regex = re.compile(r".*(embed_tokens|lm_head).*")
                elif key == "new_layers":
                    # Match parameters belonging to modules marked as new
                    for module_name, module in self.model.named_modules():
                        if getattr(module, "_cambium_new", False):
                            for param in module.parameters(recurse=True):
                                # Find the full parameter name
                                for full_name, model_param in self.model.named_parameters():
                                    if model_param is param and full_name not in assigned_params:
                                        params.append(param)
                                        assigned_params.add(full_name)
                                        break
                    continue  # Already built params list
                elif key == "original_layers":
                    # Match all transformer layers (model.layers.N) that are NOT new
                    for name, param in self.model.named_parameters():
                        if param.requires_grad and name not in assigned_params:
                            layer_match = re.search(r".*model\.layers\.(\d+)\.", name)
                            if layer_match:
                                layer_idx = int(layer_match.group(1))
                                # Check if this layer is original (not marked _cambium_new)
                                is_new = False
                                for mod_name, mod in self.model.named_modules():
                                    mod_layer_match = re.search(r"layers\.(\d+)", mod_name)
                                    if (
                                        mod_layer_match
                                        and int(mod_layer_match.group(1)) == layer_idx
                                    ):
                                        if getattr(mod, "_cambium_new", False):
                                            is_new = True
                                            break
                                if not is_new:
                                    params.append(param)
                                    assigned_params.add(name)
                    continue  # Already built params list
                else:
                    # Treat as regex pattern (legacy behavior)
                    regex = re.compile(key)

                for name, param in self.model.named_parameters():
                    if param.requires_grad and regex.search(name) and name not in assigned_params:
                        params.append(param)
                        assigned_params.add(name)

            if params:
                groups.append({"params": params, "lr": lr, "name": group_name})

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
        print(
            f"Trainable parameters: {info['trainable_params']:,} ({info['percent_trainable']:.2f}%)"
        )
        print(f"Frozen parameters: {info['frozen_params']:,}")
        print(f"Trainable parameter groups:")
        for name in info["trainable_names"]:
            print(f"  - {name}")
        print("=" * 60)

    def save_state(self, path: str) -> None:
        """Save the current freezing state to a file."""
        state = {name: param.requires_grad for name, param in self.model.named_parameters()}
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
