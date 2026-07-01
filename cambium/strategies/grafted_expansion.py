"""
Graft a single pretrained block from a remote HuggingFace model into a target model.

Only the source model config and the shards holding the requested block are
downloaded; the rest of the source weights are left on the hub.
"""

import inspect
import logging
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from cambium.blocks.base import ResidualWrapper
from cambium.core.expansion import ExpansionEngine
from cambium.core.grafting import load_grafted_block
from cambium.exceptions import BlockValidationError, GraftingError

logger = logging.getLogger(__name__)


class _GraftedBlockWrapper(nn.Module):
    """
    Wrap a source decoder layer so it fits the target model.

    Handles two mismatches:
    1. Source decoder layers may return ``(hidden_states, ...)`` tuples.
    2. Source and target hidden sizes may differ.
    """

    def __init__(self, source_block: nn.Module, source_hidden_size: int, target_hidden_size: int):
        super().__init__()
        self.source_block = source_block
        if source_hidden_size != target_hidden_size:
            self.input_proj = nn.Linear(target_hidden_size, source_hidden_size)
            self.output_proj = nn.Linear(source_hidden_size, target_hidden_size)
        else:
            self.input_proj = None
            self.output_proj = None

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        x = hidden_states
        if self.input_proj is not None:
            x = self.input_proj(x)

        out = self.source_block(x, **kwargs)
        if isinstance(out, tuple):
            out = out[0]

        if self.output_proj is not None:
            out = self.output_proj(out)
        return out


@dataclass
class GraftedBlockExpansion:
    """
    Insert a pretrained block from a remote model into the target model.

    Only the requested block's weights are downloaded from the source repo.
    If the source and target hidden sizes differ, small learnable projection
    layers are added before and after the grafted block.

    Example::

        from cambium import ExpandableModel, GraftedBlockExpansion

        model = ExpandableModel.from_pretrained("HuggingFaceTB/SmolLM2-135M")
        model.expand(GraftedBlockExpansion(
            source_model_id="HuggingFaceTB/SmolLM2-135M",
            source_block_idx=5,
            positions=[3],
        ))

    Args:
        source_model_id: HuggingFace repo id of the source model.
        source_block_idx: Integer index of the source layer.
        source_block_name: Exact source layer name/prefix (e.g. ``model.layers.5``).
            Overrides ``source_block_idx`` when provided.
        positions: Target insertion positions.
        projection: Add projection layers when hidden sizes differ. Defaults to True.
        freeze: Freeze the grafted block after insertion. Defaults to False.
        source_layer_attribute: Dot-separated path to source layers in the checkpoint.
        layer_attribute: Dot-separated path to target layers module.
        validate: Run forward/shape validation before insertion.
        residual_connection: Wrap the grafted block in a residual connection.
        cache_dir: Optional HuggingFace cache directory.
    """

    source_model_id: str
    source_block_idx: int | None = None
    source_block_name: str | None = None
    positions: list[int] | None = None
    projection: bool = True
    freeze: bool = False
    source_layer_attribute: str = "model.layers"
    layer_attribute: str = "model.layers"
    validate: bool = True
    residual_connection: bool = True
    cache_dir: str | None = None

    def expand(self, model: nn.Module, engine: ExpansionEngine) -> nn.Module:
        """Apply grafted block expansion to the target model."""
        logger.info(f"Starting grafted block expansion from {self.source_model_id}")

        self._validate_inputs(model)

        positions = self._compute_positions(model)
        if len(positions) != 1:
            raise GraftingError(
                "GraftedBlockExpansion currently supports grafting exactly one block"
            )

        target_dtype = next(model.parameters()).dtype
        source_block = load_grafted_block(
            source_model_id=self.source_model_id,
            source_layer_attribute=self.source_layer_attribute,
            source_block_idx=self.source_block_idx,
            source_block_name=self.source_block_name,
            target_dtype=target_dtype,
            cache_dir=self.cache_dir,
        )

        wrapped = self._wrap_source_block(model, source_block)

        if self.validate:
            self._validate_block(wrapped, model)

        if self.residual_connection:
            wrapped = ResidualWrapper(wrapped)

        if self.freeze:
            for param in wrapped.parameters(recurse=True):
                param.requires_grad = False

        created_blocks = [wrapped]

        def capturing_factory() -> nn.Module:
            return created_blocks.pop(0)

        engine.insert_blocks(
            model,
            positions,
            capturing_factory,
            block_attribute=self.layer_attribute,
        )

        self._update_config(model, positions)

        logger.info(f"Grafted block expansion complete: inserted at position {positions[0]}")
        return model

    def _validate_inputs(self, model: nn.Module) -> None:
        """Ensure the user provided exactly one way to identify the source block."""
        if self.source_block_idx is None and self.source_block_name is None:
            raise GraftingError("Must provide source_block_idx or source_block_name")
        if self.source_block_idx is not None and self.source_block_name is not None:
            raise GraftingError("Provide only one of source_block_idx or source_block_name")
        if self.positions is None or len(self.positions) == 0:
            raise GraftingError("Must provide at least one target position")

        layers_module = self._get_layers_module(model)
        current_layers = len(layers_module)
        for pos in self.positions:
            if pos < 0 or pos > current_layers:
                raise GraftingError(
                    f"Invalid target position {pos}. Must be between 0 and {current_layers}"
                )

    def _get_layers_module(self, model: nn.Module) -> nn.ModuleList:
        """Get the target layers ModuleList."""
        parts = self.layer_attribute.split(".")
        module = model
        for part in parts:
            module = getattr(module, part)
        return module

    def _compute_positions(self, model: nn.Module) -> list[int]:
        """Return sorted target insertion positions."""
        return sorted(self.positions)

    def _wrap_source_block(self, model: nn.Module, source_block: nn.Module) -> nn.Module:
        """Optionally wrap the source block with projection layers."""
        target_hidden_size = model.config.hidden_size
        source_hidden_size = getattr(source_block, "hidden_size", None)
        if source_hidden_size is None and hasattr(source_block, "config"):
            source_hidden_size = getattr(source_block.config, "hidden_size", None)
        if source_hidden_size is None:
            raise GraftingError("Could not determine source block hidden_size")

        if not self.projection and source_hidden_size != target_hidden_size:
            raise GraftingError(
                f"Source hidden_size ({source_hidden_size}) != target hidden_size "
                f"({target_hidden_size}). Set projection=True or choose a compatible source block."
            )

        return _GraftedBlockWrapper(source_block, source_hidden_size, target_hidden_size)

    def _validate_block(self, block: nn.Module, model: nn.Module) -> None:
        """Check that the grafted block returns the right shape on a dummy input."""
        hidden_size = model.config.hidden_size
        block.eval()
        dummy = torch.randn(1, 1, hidden_size)

        # HF decoder layers require position_ids/attention_mask etc. Inspect the
        # actual source block (inside the wrapper if present) so we build the
        # right kwargs for the underlying architecture.
        source_block = block
        if isinstance(block, _GraftedBlockWrapper):
            source_block = block.source_block
        elif isinstance(block, ResidualWrapper):
            inner = getattr(block, "block", None)
            if isinstance(inner, _GraftedBlockWrapper):
                source_block = inner.source_block

        forward_kwargs = self._build_validation_kwargs(source_block, model)

        with torch.no_grad():
            try:
                output = block(dummy, **forward_kwargs)
                if isinstance(output, tuple):
                    output = output[0]
                if output.shape != dummy.shape:
                    raise BlockValidationError(
                        block_idx=-1,
                        reason=(
                            f"Grafted block output shape {tuple(output.shape)} does not match "
                            f"target hidden shape {tuple(dummy.shape)}"
                        ),
                    )
            except Exception as e:
                raise BlockValidationError(
                    block_idx=-1,
                    reason=f"Grafted block forward failed on dummy input: {e}",
                ) from e
        block.train()

    def _build_validation_kwargs(self, block: nn.Module, model: nn.Module) -> dict[str, Any]:
        """Build kwargs needed to run a single-step validation forward."""
        hidden_size = model.config.hidden_size
        forward_kwargs: dict[str, Any] = {}
        try:
            sig = inspect.signature(block.forward)
            params = set(sig.parameters.keys())
        except (ValueError, TypeError):
            params = set()

        if "position_ids" in params:
            forward_kwargs["position_ids"] = torch.zeros(1, 1, dtype=torch.long)
        if "attention_mask" in params:
            forward_kwargs["attention_mask"] = torch.ones(1, 1, dtype=torch.long)
        if "position_embeddings" in params:
            head_dim = getattr(
                model.config,
                "head_dim",
                hidden_size // getattr(model.config, "num_attention_heads", 1),
            )
            cos = sin = torch.zeros(1, 1, head_dim)
            forward_kwargs["position_embeddings"] = (cos, sin)
        if "cache_position" in params:
            forward_kwargs["cache_position"] = torch.zeros(1, dtype=torch.long)

        return forward_kwargs

    def _update_config(self, model: nn.Module, positions: list[int]) -> None:
        """Update model config to reflect the inserted grafted block."""
        num_new = len(positions)

        if hasattr(model.config, "num_hidden_layers"):
            model.config.num_hidden_layers += num_new

        if hasattr(model.config, "_name_or_path"):
            model.config._name_or_path += f"_cambium_graft_{num_new}L"

        if hasattr(model.config, "layer_types") and model.config.layer_types is not None:
            original_layer_types = list(model.config.layer_types)
            new_layer_types = list(original_layer_types)

            for offset, pos in enumerate(sorted(positions)):
                insert_pos = pos + offset
                layer_type = (
                    original_layer_types[pos]
                    if pos < len(original_layer_types)
                    else "full_attention"
                )
                new_layer_types.insert(insert_pos, layer_type)

            model.config.layer_types = new_layer_types
            logger.info(
                f"Updated layer_types: {len(original_layer_types)} -> {len(new_layer_types)} entries"
            )

        grafted_blocks = getattr(model.config, "_cambium_grafted_blocks", [])
        grafted_blocks.append(
            {
                "source_model_id": self.source_model_id,
                "source_block_idx": self.source_block_idx,
                "source_block_name": self.source_block_name,
                "positions": positions.copy(),
                "projection": self.projection,
                "freeze": self.freeze,
            }
        )
        model.config._cambium_grafted_blocks = grafted_blocks
