"""
ExpandableModel - PyTorch-native API for model expansion.

Provides an explicit, object-oriented interface for loading models
and applying expansion strategies.
"""

import json
import logging
import os
from typing import Any

import torch
from torch import nn

from cambium.core.expansion import ExpansionEngine
from cambium.core.freezing import FreezingManager
from cambium.core.initialization import Initializer

logger = logging.getLogger(__name__)


class ExpandableModel:
    """
    Wrapper for Hugging Face models that enables surgical expansion.

    Provides a PyTorch-native API for loading models and applying
    expansion strategies.

    Example::

        from cambium import ExpandableModel, InterleavedExpansion
        import torch

        # Load model
        model = ExpandableModel.from_pretrained("HuggingFaceTB/SmolLM2-135M", dtype=torch.float32)

        # Expand
        expander = InterleavedExpansion(num_layers=4)
        model.expand(expander)

        # Get the expanded model for training
        expanded = model.get_model()
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: str | None = None,
        config: Any | None = None,
    ):
        """
        Initialize an ExpandableModel wrapper.

        Args:
            model: The base PyTorch model
            model_name: Name/path of the model (for metadata)
            config: Model configuration
        """
        self.model = model
        self.model_name = model_name or "unknown"
        self.config = config or model.config

        # Expansion tracking
        self.engine = ExpansionEngine()
        self.expansions: list[dict[str, Any]] = []
        self.is_expanded = False

        # Training utilities
        self.freezing_manager = FreezingManager(model)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        **kwargs,
    ) -> "ExpandableModel":
        """
        Load a pretrained model and wrap it.

        Args:
            model_name_or_path: Hugging Face model name or path
            **kwargs: Additional arguments for from_pretrained

        Returns:
            ExpandableModel instance
        """
        try:
            from transformers import AutoConfig, AutoModelForCausalLM
        except ImportError:
            raise ImportError(
                "transformers library required. Install with: pip install transformers"
            )

        logger.info(f"Loading model: {model_name_or_path}")

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **kwargs,
        )

        config = model.config

        return cls(model, model_name=model_name_or_path, config=config)

    @staticmethod
    def _sanitize_config(config: dict[str, Any]) -> dict[str, Any]:
        """Remove non-JSON-serializable values from a dict."""
        sanitized: dict[str, Any] = {}
        for k, v in config.items():
            try:
                json.dumps(v)
                sanitized[k] = v
            except (TypeError, ValueError):
                sanitized[k] = f"<{type(v).__name__}>"
        return sanitized

    def expand(self, expander: Any) -> "ExpandableModel":
        """
        Apply an expansion strategy.

        Args:
            expander: Expansion strategy (e.g., InterleavedExpansion)

        Returns:
            self (for method chaining)
        """
        logger.info(f"Applying expansion: {type(expander).__name__}")

        # Apply the expansion
        expander.expand(self.model, self.engine)

        # Track expansion
        self.expansions.append(
            {
                "strategy": type(expander).__name__,
                "config": self._sanitize_config(getattr(expander, "__dict__", {})),
            }
        )

        self.is_expanded = True

        # Update freezing manager with the modified model
        self.freezing_manager = FreezingManager(self.model)

        logger.info("Expansion complete")
        return self

    def get_model(self) -> nn.Module:
        """
        Get the underlying PyTorch model.

        Returns:
            The model (expanded if expand() was called)
        """
        return self.model

    def get_config(self) -> Any:
        """
        Get the model configuration.

        Returns:
            Model config (updated after expansion)
        """
        return self.config

    def freeze_original(self) -> "ExpandableModel":
        """
        Freeze original pretrained weights.

        Newly added layers remain trainable.

        Returns:
            self (for method chaining)
        """
        self.freezing_manager.freeze_original_layers()
        return self

    def unfreeze_all(self) -> "ExpandableModel":
        """
        Unfreeze all parameters.

        Returns:
            self (for method chaining)
        """
        self.freezing_manager.unfreeze_all()
        return self

    def print_trainable(self) -> None:
        """Print a summary of trainable vs frozen parameters."""
        self.freezing_manager.print_trainable_status()

    def save_expanded(
        self,
        save_directory: str,
        safe_serialization: bool = True,
    ) -> None:
        """
        Save the expanded model and expansion metadata.

        Args:
            save_directory: Directory to save to
            safe_serialization: Use safetensors format
        """
        os.makedirs(save_directory, exist_ok=True)

        # Save model weights and config via Hugging Face's native method.
        # This correctly handles shared tensors (e.g., tied embeddings/lm_head)
        # for safetensors by cloning them during save and retieing on load.
        self.model.save_pretrained(
            save_directory,
            safe_serialization=safe_serialization,
        )

        # Save expansion metadata
        metadata = {
            "original_model": self.model_name,
            "expansions": self.expansions,
            "is_expanded": self.is_expanded,
        }
        with open(os.path.join(save_directory, "cambium_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved expanded model to {save_directory}")

    @classmethod
    def load_expanded(
        cls,
        load_directory: str,
        **kwargs,
    ) -> "ExpandableModel":
        """
        Load a previously expanded model.

        Re-applies any expansions that were active at save time, so that
        structural side-effects (e.g. ``cambium_adapters`` attributes on
        transformer layers, patched forward closures) are restored before
        the weights are returned to the caller.

        Args:
            load_directory: Directory to load from
            **kwargs: Additional arguments for model loading

        Returns:
            ExpandableModel instance
        """
        # Load metadata
        metadata_path = os.path.join(load_directory, "cambium_metadata.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Load base model
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            load_directory,
            **kwargs,
        )

        expandable = cls(model, model_name=metadata.get("original_model"))
        stored_expansions = metadata.get("expansions", [])

        # Re-apply any expansions whose dataclass config can be
        # reconstructed from primitives alone. This is required for
        # strategies like ParallelAdapterExpansion that attach side-modules
        # as plain attributes (not registered submodules) and patch
        # forward closures, neither of which survive a save/load cycle.
        from cambium.strategies import STRATEGY_REGISTRY

        for record in stored_expansions:
            strategy_name = record.get("strategy")
            config = record.get("config", {})
            strategy_cls = STRATEGY_REGISTRY.get(strategy_name)
            if strategy_cls is None:
                # Not in registry (e.g. CustomBlockExpansion carries a
                # block_class callable that can't round-trip through JSON).
                # Leave a placeholder so callers can detect this state.
                expandable.expansions.append(record)
                continue
            try:
                expander = strategy_cls(**config)
            except TypeError as e:
                # Stored config no longer matches the dataclass signature
                # (e.g. field renamed/removed between save and load). Skip
                # rather than crashing the load entirely.
                logger.warning(
                    f"Could not reconstruct {strategy_name} from saved "
                    f"config: {e}. Skipping re-application."
                )
                expandable.expansions.append(record)
                continue
            expandable.expand(expander)

        if stored_expansions and not expandable.is_expanded:
            expandable.is_expanded = True

        # Some strategies (e.g. ParallelAdapterExpansion) attach
        # parameters as plain attributes on submodules. Those tensors
        # are saved by save_pretrained, but the base HF model class has
        # no declared mapping for them, so they are loaded as
        # "UNEXPECTED" and dropped. We pull the safetensors file
        # ourselves and copy any keys that now match a live parameter
        # (the re-applied expansion makes the parameter paths valid).
        expandable._reload_orphan_weights(load_directory)

        return expandable

    def _reload_orphan_weights(self, load_directory: str) -> None:
        """
        Copy weights from ``model.safetensors`` into the live model
        parameters, for any key the base HF loader would have dropped.

        This is specifically needed when an expansion re-attaches
        side-modules (like ``cambium_adapters``) to layers after
        ``from_pretrained`` has finished. The re-attached parameters
        did not exist when HF built the base model, so their tensors
        were not loaded; this method picks them up from disk.
        """
        from pathlib import Path

        weights_file = Path(load_directory) / "model.safetensors"
        if not weights_file.exists():
            # Non-safetensors checkpoints (e.g. PyTorch bin) are not
            # re-loaded here; the caller would need to handle those.
            return

        try:
            from safetensors.torch import load_file
        except ImportError:
            logger.debug("safetensors not available; skipping orphan weight reload")
            return

        saved = load_file(str(weights_file))
        live = dict(self.model.named_parameters())

        loaded, missing = 0, 0
        for key, tensor in saved.items():
            if key in live and live[key].shape == tensor.shape:
                live[key].data.copy_(tensor.to(live[key].dtype))
                loaded += 1
            elif key in live:
                logger.warning(
                    f"Shape mismatch for '{key}': saved {tuple(tensor.shape)} "
                    f"vs model {tuple(live[key].shape)}; skipping"
                )
                missing += 1
        if loaded:
            logger.info(f"Reloaded {loaded} expansion-specific weight tensor(s)")

    def validate(self) -> dict[str, Any]:
        """
        Validate the expanded model.

        Returns:
            Validation results dictionary
        """
        return self.engine.validate_expansion(self.model)

    def get_expansion_report(self) -> str:
        """Get a human-readable report of expansions."""
        lines = [
            f"Model: {self.model_name}",
            f"Expanded: {self.is_expanded}",
            "",
            "Expansion History:",
            "-" * 40,
        ]

        for i, exp in enumerate(self.expansions, 1):
            lines.append(f"{i}. {exp['strategy']}")
            for key, value in exp.get("config", {}).items():
                lines.append(f"   {key}: {value}")

        lines.append("")
        lines.append(self.engine.get_expansion_report())

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ExpandableModel("
            f"model_name='{self.model_name}', "
            f"expanded={self.is_expanded}, "
            f"num_expansions={len(self.expansions)})"
        )
