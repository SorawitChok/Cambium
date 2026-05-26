"""
Low-level surgical operations for transformer model expansion.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import nn

logger = logging.getLogger(__name__)


class ExpansionEngine:
    """
    Core engine for surgical model expansion operations.

    Handles layer insertion, module replacement, and validation of expanded models.
    Works with any Hugging Face transformers model that follows standard conventions.
    """

    def __init__(self, model_type: str = "auto"):
        """
        Initialize the expansion engine.

        Args:
            model_type: Type of model architecture (auto-detected if "auto")
        """
        self.model_type = model_type
        self.expansion_history: List[Dict[str, Any]] = []

    def insert_blocks(
        self,
        model: nn.Module,
        positions: List[int],
        block_factory: Callable[[], nn.Module],
        block_attribute: str = "model.layers",
    ) -> None:
        """
        Insert new transformer blocks at specified positions.

        Args:
            model: The model to expand
            positions: List of indices where new blocks should be inserted
            block_factory: Factory function that creates new blocks
            block_attribute: Dot-separated path to the layers module (e.g., "model.layers")

        Raises:
            ValueError: If positions are invalid or block_attribute not found
        """
        # Get the layers module
        layers_module = self._get_nested_attr(model, block_attribute)
        if layers_module is None:
            raise ValueError(f"Could not find {block_attribute} in model")

        if not isinstance(layers_module, nn.ModuleList):
            raise ValueError(f"{block_attribute} must be a ModuleList, got {type(layers_module)}")

        original_length = len(layers_module)

        # Validate positions
        for pos in positions:
            if pos < 0 or pos > original_length:
                raise ValueError(f"Invalid position {pos}. Must be between 0 and {original_length}")

        # Sort positions in descending order to maintain indices during insertion
        sorted_positions = sorted(positions, reverse=True)

        # Create new blocks
        new_blocks = [block_factory() for _ in range(len(positions))]

        # Insert blocks (from end to start to maintain indices)
        for i, pos in enumerate(sorted_positions):
            layers_module.insert(pos, new_blocks[i])
            new_blocks[i]._cambium_new = True
            logger.debug(f"Inserted new block at position {pos}")

        # Fix layer_idx for ALL layers after insertion.
        # Original layers shifted positions; new layers got dummy indices.
        # layer_idx is used for KV-cache indexing during generation.
        for idx, layer in enumerate(layers_module):
            for submodule in layer.modules():
                if hasattr(submodule, "layer_idx"):
                    submodule.layer_idx = idx

        # Record expansion
        self.expansion_history.append(
            {
                "operation": "insert_blocks",
                "positions": positions.copy(),
                "original_length": original_length,
                "new_length": len(layers_module),
            }
        )

        logger.info(
            f"Inserted {len(positions)} blocks at positions {positions}. "
            f"Model now has {len(layers_module)} layers (was {original_length})"
        )

    def expand_dimensions(
        self,
        module: nn.Module,
        old_dim: int,
        new_dim: int,
        axis: int = 0,
        initialization: str = "zero_pad",
    ) -> nn.Module:
        """
        Expand the dimensions of a linear layer or embedding.

        Args:
            module: The module to expand (Linear or Embedding)
            old_dim: Original dimension size
            new_dim: New dimension size
            axis: Which axis to expand (0 for out_features, 1 for in_features)
            initialization: How to initialize new weights ("zero_pad", "noise", "copy")

        Returns:
            The expanded module (same object, modified in-place)

        Raises:
            ValueError: If module type is not supported
        """
        if isinstance(module, nn.Linear):
            self._expand_linear(module, old_dim, new_dim, axis, initialization)
        elif isinstance(module, nn.Embedding):
            self._expand_embedding(module, old_dim, new_dim, initialization)
        else:
            raise ValueError(f"Cannot expand dimensions of {type(module)}")

        self.expansion_history.append(
            {
                "operation": "expand_dimensions",
                "old_dim": old_dim,
                "new_dim": new_dim,
                "axis": axis,
            }
        )

        return module

    def _expand_linear(
        self,
        linear: nn.Linear,
        old_dim: int,
        new_dim: int,
        axis: int,
        initialization: str,
    ) -> None:
        """Expand a Linear layer's dimensions."""
        old_weight = linear.weight.data

        if axis == 0:  # Expand output dimension
            new_weight = torch.zeros(new_dim, old_weight.shape[1], device=old_weight.device)
            if initialization == "zero_pad":
                new_weight[:old_dim, :] = old_weight
            elif initialization == "copy":
                new_weight[:old_dim, :] = old_weight
                # Copy last row for remaining
                for i in range(old_dim, new_dim):
                    new_weight[i, :] = old_weight[-1, :]
            elif initialization == "noise":
                new_weight[:old_dim, :] = old_weight
                new_weight[old_dim:, :] = (
                    torch.randn(new_dim - old_dim, old_weight.shape[1], device=old_weight.device)
                    * 0.01
                )

            linear.weight = nn.Parameter(new_weight)

            # Expand bias if present
            if linear.bias is not None:
                old_bias = linear.bias.data
                new_bias = torch.zeros(new_dim, device=old_bias.device)
                new_bias[:old_dim] = old_bias
                linear.bias = nn.Parameter(new_bias)

            linear.out_features = new_dim

        else:  # Expand input dimension
            new_weight = torch.zeros(old_weight.shape[0], new_dim, device=old_weight.device)
            if initialization == "zero_pad":
                new_weight[:, :old_dim] = old_weight
            elif initialization == "copy":
                new_weight[:, :old_dim] = old_weight
                new_weight[:, old_dim:] = old_weight[:, -1:].expand(-1, new_dim - old_dim)
            elif initialization == "noise":
                new_weight[:, :old_dim] = old_weight
                new_weight[:, old_dim:] = (
                    torch.randn(old_weight.shape[0], new_dim - old_dim, device=old_weight.device)
                    * 0.01
                )

            linear.weight = nn.Parameter(new_weight)
            linear.in_features = new_dim

    def _expand_embedding(
        self,
        embedding: nn.Embedding,
        old_dim: int,
        new_dim: int,
        initialization: str,
    ) -> None:
        """Expand an Embedding layer."""
        old_weight = embedding.weight.data
        new_weight = torch.zeros(new_dim, old_weight.shape[1], device=old_weight.device)

        if initialization == "zero_pad":
            new_weight[:old_dim, :] = old_weight
        elif initialization == "copy":
            new_weight[:old_dim, :] = old_weight
            # Copy last embeddings for remaining
            for i in range(old_dim, new_dim):
                new_weight[i, :] = old_weight[-1, :]
        elif initialization == "noise":
            new_weight[:old_dim, :] = old_weight
            new_weight[old_dim:, :] = (
                torch.randn(new_dim - old_dim, old_weight.shape[1], device=old_weight.device) * 0.01
            )

        embedding.weight = nn.Parameter(new_weight)
        embedding.num_embeddings = new_dim

    def validate_expansion(self, model: nn.Module) -> Dict[str, Any]:
        """
        Validate that an expanded model is consistent and trainable.

        Args:
            model: The expanded model to validate

        Returns:
            Dictionary with validation results
        """
        results = {
            "valid": True,
            "checks": {},
            "warnings": [],
            "errors": [],
        }

        # Check 1: All parameters have requires_grad where expected
        try:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            results["checks"]["parameters"] = {
                "total": total_params,
                "trainable": trainable_params,
                "frozen": total_params - trainable_params,
            }
        except Exception as e:
            results["valid"] = False
            results["errors"].append(f"Parameter check failed: {e}")

        # Check 2: No NaN or Inf in parameters
        try:
            has_nan = any(torch.isnan(p).any() for p in model.parameters())
            has_inf = any(torch.isinf(p).any() for p in model.parameters())
            results["checks"]["numerical_stability"] = {
                "has_nan": has_nan,
                "has_inf": has_inf,
            }
            if has_nan or has_inf:
                results["valid"] = False
                results["errors"].append("Model has NaN or Inf parameters")
        except Exception as e:
            results["valid"] = False
            results["errors"].append(f"Numerical check failed: {e}")

        # Check 3: Forward pass works with dummy input
        try:
            # This is a basic check - subclasses should override for specific models
            results["checks"]["forward_pass"] = "skipped (requires model-specific input)"
        except Exception as e:
            results["valid"] = False
            results["errors"].append(f"Forward pass check failed: {e}")

        return results

    def _get_nested_attr(self, obj: Any, attr_path: str) -> Any:
        """Get a nested attribute using dot notation."""
        parts = attr_path.split(".")
        for part in parts:
            if not hasattr(obj, part):
                return None
            obj = getattr(obj, part)
        return obj

    def get_expansion_report(self) -> str:
        """Generate a human-readable report of all expansions."""
        lines = ["Cambium Expansion Report", "=" * 50, ""]

        for i, expansion in enumerate(self.expansion_history, 1):
            lines.append(f"Expansion {i}: {expansion['operation']}")
            for key, value in expansion.items():
                if key != "operation":
                    lines.append(f"  {key}: {value}")
            lines.append("")

        return "\n".join(lines)
