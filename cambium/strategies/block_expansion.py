"""
Interleaved block expansion strategy (LLaMA-Pro style).

Inserts new transformer blocks between existing ones,
enabling increased capacity while preserving original weights.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch import nn

from cambium.core.expansion import ExpansionEngine
from cambium.core.initialization import InitializationStrategy, Initializer

logger = logging.getLogger(__name__)


@dataclass
class InterleavedExpansion:
    """
    LLaMA-Pro style expansion: Insert new blocks between existing ones.

    Original: [Block0] → [Block1] → [Block2] → [Block3]
    Expanded: [Block0] → [New0] → [Block1] → [New1] → [Block2] → [New2] → [Block3]

    This is the recommended strategy for most use cases as it:
    - Preserves all original weights exactly
    - Allows gradual capacity increase
    - Maintains the original model's behavior initially (with identity init)

    Args:
        num_layers: Number of new blocks to insert.
        positions: Specific positions to insert blocks. Auto-distributed if None.
        initialization: Initialization strategy. One of ``'identity'``,
            ``'small_random'``, ``'noise'``, ``'zero'``.
        block_config: Override configuration for new blocks.
        layer_attribute: Dot-separated path to the layers ModuleList.
    """

    num_layers: int
    positions: list[int] | None = None
    initialization: str = "identity"
    block_config: dict[str, Any] | None = field(default_factory=dict)
    layer_attribute: str = "model.layers"

    def expand(self, model: nn.Module, engine: ExpansionEngine) -> nn.Module:
        """
        Apply interleaved expansion to a model.

        Args:
            model: The model to expand
            engine: ExpansionEngine instance

        Returns:
            The expanded model (modified in-place)
        """
        logger.info(
            f"Starting interleaved expansion: {self.num_layers} layers, "
            f"init={self.initialization}"
        )

        # Get current number of layers
        layers_module = self._get_layers_module(model)
        current_layers = len(layers_module)

        # Determine insertion positions
        positions = self._compute_positions(current_layers)
        logger.info(f"Will insert blocks at positions: {positions}")

        # Capture block instances so we can initialize them directly
        # instead of looking them up by (now-stale) positions afterwards.
        created_blocks: list[nn.Module] = []
        factory = self._create_block_factory(model)

        def capturing_factory() -> nn.Module:
            block = factory()
            created_blocks.append(block)
            return block

        engine.insert_blocks(
            model,
            positions,
            capturing_factory,
            block_attribute=self.layer_attribute,
        )

        self._apply_initialization(model, created_blocks)

        # Update model config
        self._update_config(model, positions=positions)

        logger.info("Interleaved expansion complete")
        return model

    def _get_layers_module(self, model: nn.Module) -> nn.ModuleList:
        """Get the layers ModuleList from the model."""
        parts = self.layer_attribute.split(".")
        module = model
        for part in parts:
            module = getattr(module, part)
        return module

    def _compute_positions(self, current_layers: int) -> list[int]:
        """
        Compute insertion positions for new blocks.

        Distributes new blocks evenly among existing ones.
        """
        if self.positions is not None:
            # Validate provided positions
            for pos in self.positions:
                if pos < 0 or pos > current_layers:
                    raise ValueError(
                        f"Invalid position {pos}. Must be between 0 and {current_layers}"
                    )
            return sorted(self.positions)

        # Auto-distribute: place new blocks between existing ones
        if self.num_layers >= current_layers:
            # More new layers than existing - insert after each existing
            return list(range(1, current_layers + 1))

        # Distribute evenly
        step = current_layers / (self.num_layers + 1)
        positions = []
        for i in range(self.num_layers):
            pos = int((i + 1) * step)
            # Ensure we don't collide
            if positions and pos == positions[-1]:
                pos += 1
            positions.append(pos)

        return positions

    def _create_block_factory(self, model: nn.Module) -> Callable[[], nn.Module]:
        """
        Create a factory function that produces new blocks.

        This uses the model's config to create compatible blocks.
        """
        config = model.config

        # Apply any overrides
        if self.block_config:
            for key, value in self.block_config.items():
                setattr(config, key, value)

        # Determine the block class based on model type
        model_type = getattr(config, "model_type", "llama")

        def create_block() -> nn.Module:
            if model_type == "llama":
                from transformers.models.llama.modeling_llama import LlamaDecoderLayer

                return LlamaDecoderLayer(config, layer_idx=0)
            elif model_type == "mistral":
                from transformers.models.mistral.modeling_mistral import MistralDecoderLayer

                return MistralDecoderLayer(config, layer_idx=0)
            elif model_type == "gemma":
                from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer

                return GemmaDecoderLayer(config, layer_idx=0)
            elif model_type in ["gemma3", "gemma3_text"]:
                from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer

                return Gemma3DecoderLayer(config, layer_idx=0)
            elif model_type == "qwen2":
                from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

                return Qwen2DecoderLayer(config, layer_idx=0)
            elif model_type == "qwen3":
                from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

                return Qwen3DecoderLayer(config, layer_idx=0)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

        return create_block

    def _apply_initialization(self, model: nn.Module, blocks: list[nn.Module]) -> None:
        """Apply initialization strategy to the newly inserted blocks.

        Args:
            model: The expanded model. Unused; kept for API symmetry with
                ``CustomBlockExpansion``.
            blocks: The block instances that were just inserted.
        """
        # Map initialization string to strategy
        strategy_map = {
            "identity": InitializationStrategy.IDENTITY_MAPPING,
            "small_random": InitializationStrategy.SMALL_RANDOM,
            "noise": InitializationStrategy.NOISE_INJECTION,
            "zero": InitializationStrategy.ZERO_INIT,
            "xavier": InitializationStrategy.XAVIER_UNIFORM,
            "kaiming": InitializationStrategy.KAIMING_NORMAL,
        }

        strategy = strategy_map.get(self.initialization, InitializationStrategy.IDENTITY_MAPPING)

        initializer = Initializer(strategy)

        # Apply to each new block
        for i, block in enumerate(blocks):
            # Get all modules in the block
            modules = list(block.modules())[1:]  # Skip the block itself
            initializer.apply(modules)
            logger.debug(f"Applied {self.initialization} initialization to new block {i}")

    def _update_config(self, model: nn.Module, positions: list[int]) -> None:
        """Update model config to reflect the new architecture."""
        if hasattr(model.config, "num_hidden_layers"):
            model.config.num_hidden_layers += self.num_layers
        if hasattr(model.config, "_name_or_path"):
            model.config._name_or_path += f"_cambium_expanded_{self.num_layers}L"

        # Update layer_types if present (e.g. Gemma3 uses this to select
        # sliding vs full attention per layer). The list must stay in sync
        # with num_hidden_layers or the forward pass will IndexError.
        if hasattr(model.config, "layer_types") and model.config.layer_types is not None:
            original_layer_types = list(model.config.layer_types)
            new_layer_types = list(original_layer_types)

            for offset, pos in enumerate(sorted(positions)):
                insert_pos = pos + offset  # Account for earlier insertions shifting indices
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


@dataclass
class AppendExpansion:
    """
    Append new blocks at the end of the model.

    Less common than interleaved expansion as it changes the
    representation space significantly, but useful for certain tasks.
    """

    num_layers: int
    initialization: str = "identity"
    layer_attribute: str = "model.layers"

    def expand(self, model: nn.Module, engine: ExpansionEngine) -> nn.Module:
        """Append new blocks at the end."""
        layers_module = self._get_layers_module(model)
        current_layers = len(layers_module)

        # Positions are all at the end
        positions = [current_layers] * self.num_layers

        # Create and insert blocks
        block_factory = self._create_block_factory(model)
        engine.insert_blocks(model, positions, block_factory, self.layer_attribute)

        return model

    def _get_layers_module(self, model: nn.Module) -> nn.ModuleList:
        parts = self.layer_attribute.split(".")
        module = model
        for part in parts:
            module = getattr(module, part)
        return module

    def _create_block_factory(self, model: nn.Module) -> Callable[[], nn.Module]:
        """Create block factory (same as InterleavedExpansion)."""
        config = model.config
        model_type = getattr(config, "model_type", "llama")

        def create_block() -> nn.Module:
            if model_type == "llama":
                from transformers.models.llama.modeling_llama import LlamaDecoderLayer

                return LlamaDecoderLayer(config, layer_idx=0)
            elif model_type == "mistral":
                from transformers.models.mistral.modeling_mistral import MistralDecoderLayer

                return MistralDecoderLayer(config, layer_idx=0)
            elif model_type == "gemma":
                from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer

                return GemmaDecoderLayer(config, layer_idx=0)
            elif model_type in ["gemma3", "gemma3_text"]:
                from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer

                return Gemma3DecoderLayer(config, layer_idx=0)
            elif model_type == "qwen2":
                from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

                return Qwen2DecoderLayer(config, layer_idx=0)
            elif model_type == "qwen3":
                from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

                return Qwen3DecoderLayer(config, layer_idx=0)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

        return create_block
