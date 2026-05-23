"""
Width expansion strategy - increase hidden dimensions.

Expands the hidden size of transformer models, requiring careful
weight re-mapping to preserve behavior.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

import torch
from torch import nn

from cambium.core.expansion import ExpansionEngine

logger = logging.getLogger(__name__)


@dataclass
class WidthExpansion:
    """
    Expand the hidden dimensions of a model.

    This is more aggressive than block expansion as it changes the
    representation space. Requires re-mapping existing weights and
    careful initialization.

    Example: 768 -> 1152 hidden dimensions
    """

    hidden_dim_multiplier: float = 1.5
    """Multiplier for hidden dimension (e.g., 1.5 means 768 -> 1152)."""

    initialization: str = "copy"
    """How to initialize new dimensions: 'copy', 'zero', 'noise'."""

    freeze_original_dims: bool = True
    """Whether to freeze the original dimension weights during initial training."""

    def expand(self, model: nn.Module, engine: ExpansionEngine) -> nn.Module:
        """
        Apply width expansion to a model.

        Args:
            model: The model to expand
            engine: ExpansionEngine instance

        Returns:
            The expanded model (modified in-place)
        """
        config = model.config
        old_hidden = config.hidden_size
        new_hidden = int(old_hidden * self.hidden_dim_multiplier)

        logger.info(f"Expanding width: {old_hidden} -> {new_hidden}")

        # Expand key modules
        self._expand_embeddings(model, old_hidden, new_hidden)
        self._expand_lm_head(model, old_hidden, new_hidden)
        self._expand_transformer_layers(model, old_hidden, new_hidden)

        # Update config
        config.hidden_size = new_hidden
        if hasattr(config, "intermediate_size"):
            # Also expand FFN dimension proportionally
            old_intermediate = config.intermediate_size
            new_intermediate = int(old_intermediate * self.hidden_dim_multiplier)
            self._expand_ffn_layers(model, old_intermediate, new_intermediate)
            config.intermediate_size = new_intermediate

        logger.info("Width expansion complete")
        return model

    def _expand_embeddings(self, model: nn.Module, old_dim: int, new_dim: int) -> None:
        """Expand token embeddings."""
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            embed = model.model.embed_tokens
            if embed.weight.shape[1] == old_dim:
                self._expand_linear_weight(embed, old_dim, new_dim, axis=1)
                logger.debug(f"Expanded embeddings: {old_dim} -> {new_dim}")

    def _expand_lm_head(self, model: nn.Module, old_dim: int, new_dim: int) -> None:
        """Expand LM head."""
        if hasattr(model, "lm_head"):
            lm_head = model.lm_head
            if lm_head.weight.shape[1] == old_dim:
                self._expand_linear_weight(lm_head, old_dim, new_dim, axis=1)
                logger.debug(f"Expanded lm_head: {old_dim} -> {new_dim}")

    def _expand_transformer_layers(self, model: nn.Module, old_dim: int, new_dim: int) -> None:
        """Expand all transformer layers."""
        if not hasattr(model, "model") or not hasattr(model.model, "layers"):
            logger.warning("Could not find transformer layers to expand")
            return

        layers = model.model.layers

        for layer_idx, layer in enumerate(layers):
            # Expand self-attention
            self._expand_attention_layer(layer, old_dim, new_dim)

            # Expand MLP
            self._expand_mlp_layer(layer, old_dim, new_dim)

            # Expand layer norm
            self._expand_layer_norm(layer, old_dim, new_dim)

            logger.debug(f"Expanded layer {layer_idx}")

    def _expand_attention_layer(self, layer: nn.Module, old_dim: int, new_dim: int) -> None:
        """Expand attention projection layers."""
        # Common patterns: q_proj, k_proj, v_proj, o_proj
        for proj_name in ["q_proj", "k_proj", "v_proj"]:
            if hasattr(layer.self_attn, proj_name):
                proj = getattr(layer.self_attn, proj_name)
                if proj.weight.shape[0] == old_dim:  # output dimension
                    self._expand_linear_weight(proj, old_dim, new_dim, axis=0)

        # o_proj (output projection) - only expand input dimension
        if hasattr(layer.self_attn, "o_proj"):
            o_proj = layer.self_attn.o_proj
            if o_proj.weight.shape[1] == old_dim:  # input dimension
                self._expand_linear_weight(o_proj, old_dim, new_dim, axis=1)

    def _expand_mlp_layer(self, layer: nn.Module, old_dim: int, new_dim: int) -> None:
        """Expand MLP layers (up_proj and down_proj)."""
        mlp = layer.mlp

        # up_proj (expands hidden dim -> intermediate dim)
        for proj_name in ["up_proj", "gate_proj"]:
            if hasattr(mlp, proj_name):
                proj = getattr(mlp, proj_name)
                if proj.weight.shape[1] == old_dim:
                    self._expand_linear_weight(proj, old_dim, new_dim, axis=1)

        # down_proj (reduces intermediate dim -> hidden dim)
        if hasattr(mlp, "down_proj"):
            down_proj = mlp.down_proj
            if down_proj.weight.shape[0] == old_dim:
                self._expand_linear_weight(down_proj, old_dim, new_dim, axis=0)

    def _expand_ffn_layers(self, model: nn.Module, old_intermediate: int, new_intermediate: int) -> None:
        """Expand FFN intermediate dimensions."""
        if not hasattr(model, "model") or not hasattr(model.model, "layers"):
            return

        for layer in model.model.layers:
            mlp = layer.mlp

            # Expand the output dimension of up_proj and gate_proj
            for proj_name in ["up_proj", "gate_proj"]:
                if hasattr(mlp, proj_name):
                    proj = getattr(mlp, proj_name)
                    if proj.weight.shape[0] == old_intermediate:
                        self._expand_linear_weight(proj, old_intermediate, new_intermediate, axis=0)

            # Expand the input dimension of down_proj
            if hasattr(mlp, "down_proj"):
                down_proj = mlp.down_proj
                if down_proj.weight.shape[1] == old_intermediate:
                    self._expand_linear_weight(down_proj, old_intermediate, new_intermediate, axis=1)

    def _expand_layer_norm(self, layer: nn.Module, old_dim: int, new_dim: int) -> None:
        """Expand LayerNorm/RMSNorm weights."""
        # Input norm
        if hasattr(layer, "input_layernorm"):
            norm = layer.input_layernorm
            if hasattr(norm, "weight") and norm.weight.shape[0] == old_dim:
                self._expand_norm_weight(norm, old_dim, new_dim)

        # Post-attention norm
        if hasattr(layer, "post_attention_layernorm"):
            norm = layer.post_attention_layernorm
            if hasattr(norm, "weight") and norm.weight.shape[0] == old_dim:
                self._expand_norm_weight(norm, old_dim, new_dim)

    def _expand_linear_weight(
        self,
        linear: nn.Linear,
        old_dim: int,
        new_dim: int,
        axis: int,
    ) -> None:
        """Expand a linear layer's weight along specified axis."""
        old_weight = linear.weight.data

        if axis == 0:  # Expand output dimension
            new_weight = torch.zeros(new_dim, old_weight.shape[1], device=old_weight.device, dtype=old_weight.dtype)

            if self.initialization == "copy":
                new_weight[:old_dim, :] = old_weight
                # Copy the last rows for new dimensions
                new_weight[old_dim:, :] = old_weight[-1:, :].expand(new_dim - old_dim, -1)
            elif self.initialization == "zero":
                new_weight[:old_dim, :] = old_weight
                # New dimensions are already zeros
            elif self.initialization == "noise":
                new_weight[:old_dim, :] = old_weight
                new_weight[old_dim:, :] = torch.randn(
                    new_dim - old_dim, old_weight.shape[1],
                    device=old_weight.device, dtype=old_weight.dtype
                ) * 0.01

            linear.weight = nn.Parameter(new_weight)

            # Expand bias if present
            if linear.bias is not None:
                old_bias = linear.bias.data
                new_bias = torch.zeros(new_dim, device=old_bias.device, dtype=old_bias.dtype)
                new_bias[:old_dim] = old_bias
                if self.initialization == "copy":
                    new_bias[old_dim:] = old_bias[-1]
                linear.bias = nn.Parameter(new_bias)

            linear.out_features = new_dim

        else:  # axis == 1, expand input dimension
            new_weight = torch.zeros(old_weight.shape[0], new_dim, device=old_weight.device, dtype=old_weight.dtype)

            if self.initialization == "copy":
                new_weight[:, :old_dim] = old_weight
                # Copy the last columns for new dimensions
                new_weight[:, old_dim:] = old_weight[:, -1:].expand(-1, new_dim - old_dim)
            elif self.initialization == "zero":
                new_weight[:, :old_dim] = old_weight
            elif self.initialization == "noise":
                new_weight[:, :old_dim] = old_weight
                new_weight[:, old_dim:] = torch.randn(
                    old_weight.shape[0], new_dim - old_dim,
                    device=old_weight.device, dtype=old_weight.dtype
                ) * 0.01

            linear.weight = nn.Parameter(new_weight)
            linear.in_features = new_dim

    def _expand_norm_weight(self, norm: nn.Module, old_dim: int, new_dim: int) -> None:
        """Expand LayerNorm/RMSNorm weight."""
        old_weight = norm.weight.data
        new_weight = torch.ones(new_dim, device=old_weight.device, dtype=old_weight.dtype)
        new_weight[:old_dim] = old_weight
        norm.weight = nn.Parameter(new_weight)

        if hasattr(norm, "bias") and norm.bias is not None:
            old_bias = norm.bias.data
            new_bias = torch.zeros(new_dim, device=old_bias.device, dtype=old_bias.dtype)
            new_bias[:old_dim] = old_bias
            norm.bias = nn.Parameter(new_bias)

        if hasattr(norm, "normalized_shape"):
            norm.normalized_shape = (new_dim,)

        if hasattr(norm, "num_features"):
            norm.num_features = new_dim
