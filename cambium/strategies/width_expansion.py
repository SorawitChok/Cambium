"""
Width expansion strategy - increase hidden dimensions.

Expands the hidden size of transformer models, requiring careful
weight re-mapping to preserve behavior.
"""

import logging
from dataclasses import dataclass, field

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

    layer_indices: list[int] | None = None
    """Which transformer layers to expand internally. If None, all layers are expanded."""

    expand_attention: bool = True
    """Whether to expand attention head dimensions (applies to ALL layers because rotary embeddings are shared)."""

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

        # Validate that new_hidden is divisible by num_attention_heads
        num_heads = getattr(config, "num_attention_heads", None)
        if num_heads and new_hidden % num_heads != 0:
            raise ValueError(
                f"New hidden_size {new_hidden} is not divisible by num_attention_heads {num_heads}. "
                f"Choose a multiplier that yields an integer head_dim."
            )

        # Determine which layers to expand internally
        total_layers = self._get_num_layers(model)
        if self.layer_indices is None:
            target_layers = set(range(total_layers))
            logger.info(
                f"Expanding width: {old_hidden} -> {new_hidden} (all {total_layers} layers)"
            )
        else:
            # Validate indices
            invalid = [i for i in self.layer_indices if i < 0 or i >= total_layers]
            if invalid:
                raise ValueError(
                    f"Invalid layer_indices {invalid} for model with {total_layers} layers"
                )
            target_layers = set(self.layer_indices)
            logger.info(
                f"Expanding width: {old_hidden} -> {new_hidden} "
                f"(cross-layer all {total_layers} layers, internal layers {sorted(target_layers)})"
            )

        # Expand key modules
        self._expand_embeddings(model, old_hidden, new_hidden)
        self._expand_lm_head(model, old_hidden, new_hidden)
        self._expand_transformer_layers(model, old_hidden, new_hidden, target_layers, total_layers)
        self._expand_final_norm(model, old_hidden, new_hidden)

        # Update cached head_dim on attention modules so .view() shapes stay valid
        # Only update when attention dimensions are actually expanded.
        if self.expand_attention:
            self._update_cached_attributes(model, old_hidden, new_hidden)

        # Update config
        config.hidden_size = new_hidden
        if self.expand_attention and hasattr(config, "head_dim") and num_heads:
            config.head_dim = new_hidden // num_heads
        if hasattr(config, "intermediate_size"):
            # Also expand FFN dimension proportionally
            old_intermediate = config.intermediate_size
            new_intermediate = int(old_intermediate * self.hidden_dim_multiplier)
            self._expand_ffn_layers(
                model, old_intermediate, new_intermediate, target_layers, total_layers
            )
            config.intermediate_size = new_intermediate

        logger.info("Width expansion complete")
        return model

    def _get_num_layers(self, model: nn.Module) -> int:
        """Get the number of transformer layers."""
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return len(model.model.layers)
        return 0

    def _expand_embeddings(self, model: nn.Module, old_dim: int, new_dim: int) -> None:
        """Expand token embedding dimension (not vocab size)."""
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            embed = model.model.embed_tokens
            if embed.weight.shape[1] == old_dim:
                self._expand_embedding_dim(embed, old_dim, new_dim)
                embed._cambium_new = True
                logger.debug(f"Expanded embeddings: {old_dim} -> {new_dim}")

    def _expand_lm_head(self, model: nn.Module, old_dim: int, new_dim: int) -> None:
        """Expand LM head input dimension."""
        if hasattr(model, "lm_head"):
            lm_head = model.lm_head
            if lm_head.weight.shape[1] == old_dim:
                self._expand_linear_weight(lm_head, old_dim, new_dim, axis=1)
                lm_head._cambium_new = True
                logger.debug(f"Expanded lm_head: {old_dim} -> {new_dim}")

    def _expand_transformer_layers(
        self,
        model: nn.Module,
        old_dim: int,
        new_dim: int,
        target_layers: set,
        total_layers: int,
    ) -> None:
        """Expand transformer layers."""
        if not hasattr(model, "model") or not hasattr(model.model, "layers"):
            logger.warning("Could not find transformer layers to expand")
            return

        layers = model.model.layers

        for layer_idx, layer in enumerate(layers):
            is_target = layer_idx in target_layers

            # Expand self-attention cross-layer dimensions for ALL layers
            self._expand_attention_layer(layer, old_dim, new_dim, is_target)

            # Expand MLP: full internal expansion for target layers,
            # cross-layer only for non-target layers
            self._expand_mlp_layer(layer, old_dim, new_dim, is_target)

            # Expand layer norm (cross-layer, always)
            self._expand_layer_norm(layer, old_dim, new_dim)

            # Tag target layers so freeze_original() knows they're trainable
            if is_target:
                layer._cambium_new = True

            logger.debug(f"Expanded layer {layer_idx} (internal={is_target})")

    def _expand_attention_layer(
        self, layer: nn.Module, old_dim: int, new_dim: int, is_target: bool
    ) -> None:
        """Expand attention projection layers."""
        # Detect GQA models using num_key_value_groups (available on all LlamaAttention)
        num_kv_groups = getattr(layer.self_attn, "num_key_value_groups", 1)
        if num_kv_groups > 1:
            raise NotImplementedError(
                "WidthExpansion does not yet support GQA models. "
                "Please use a model with standard multi-head attention "
                "(num_attention_heads == num_key_value_heads)."
            )

        # For standard MHA all projections are (hidden_size, hidden_size).
        # Cross-layer dimensions (input to q/k/v, output from o) must always
        # match the new hidden size so adjacent layers can connect.
        for proj_name in ["q_proj", "k_proj", "v_proj"]:
            if hasattr(layer.self_attn, proj_name):
                proj = getattr(layer.self_attn, proj_name)
                # Input dimension always expands
                if proj.weight.shape[1] == old_dim:
                    self._expand_linear_weight(proj, old_dim, new_dim, axis=1)
                # Output dimension expands only when attention is expanded
                if self.expand_attention and proj.weight.shape[0] == old_dim:
                    self._expand_linear_weight(proj, old_dim, new_dim, axis=0)

        if hasattr(layer.self_attn, "o_proj"):
            o_proj = layer.self_attn.o_proj
            # Input dimension expands only when attention is expanded
            if self.expand_attention and o_proj.weight.shape[1] == old_dim:
                self._expand_linear_weight(o_proj, old_dim, new_dim, axis=1)
            # Output dimension always expands (cross-layer)
            if o_proj.weight.shape[0] == old_dim:
                self._expand_linear_weight(o_proj, old_dim, new_dim, axis=0)

    def _expand_mlp_layer(
        self, layer: nn.Module, old_dim: int, new_dim: int, is_target: bool
    ) -> None:
        """Expand MLP layers (up_proj and down_proj)."""
        mlp = layer.mlp

        # up_proj / gate_proj: input (cross-layer) always expands,
        # output (intermediate) expands only for target layers
        for proj_name in ["up_proj", "gate_proj"]:
            if hasattr(mlp, proj_name):
                proj = getattr(mlp, proj_name)
                if proj.weight.shape[1] == old_dim:
                    self._expand_linear_weight(proj, old_dim, new_dim, axis=1)
                if is_target and proj.weight.shape[0] == self._get_old_intermediate(layer):
                    # Will be handled by _expand_ffn_layers later
                    pass

        # down_proj: input (intermediate) expands only for target layers,
        # output (cross-layer) always expands
        if hasattr(mlp, "down_proj"):
            down_proj = mlp.down_proj
            if down_proj.weight.shape[0] == old_dim:
                self._expand_linear_weight(down_proj, old_dim, new_dim, axis=0)

    def _get_old_intermediate(self, layer: nn.Module) -> int:
        """Get the original intermediate size from a layer's MLP."""
        mlp = layer.mlp
        if hasattr(mlp, "up_proj"):
            return mlp.up_proj.weight.shape[0]
        if hasattr(mlp, "gate_proj"):
            return mlp.gate_proj.weight.shape[0]
        return 0

    def _expand_ffn_layers(
        self,
        model: nn.Module,
        old_intermediate: int,
        new_intermediate: int,
        target_layers: set,
        total_layers: int,
    ) -> None:
        """Expand FFN intermediate dimensions for target layers only."""
        if not hasattr(model, "model") or not hasattr(model.model, "layers"):
            return

        for layer_idx, layer in enumerate(model.model.layers):
            if layer_idx not in target_layers:
                continue

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
                    self._expand_linear_weight(
                        down_proj, old_intermediate, new_intermediate, axis=1
                    )

    def _expand_final_norm(self, model: nn.Module, old_dim: int, new_dim: int) -> None:
        """Expand the final RMSNorm/LayerNorm after the transformer layers."""
        if hasattr(model, "model") and hasattr(model.model, "norm"):
            norm = model.model.norm
            if hasattr(norm, "weight") and norm.weight.shape[0] == old_dim:
                self._expand_norm_weight(norm, old_dim, new_dim)
                logger.debug(f"Expanded final norm: {old_dim} -> {new_dim}")

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

    def _update_cached_attributes(self, model: nn.Module, old_dim: int, new_dim: int) -> None:
        """Update cached head_dim, scaling, and rotary embeddings after hidden_size changes."""
        if not hasattr(model, "model") or not hasattr(model.model, "layers"):
            return

        # Update per-attention-layer cached attributes
        for layer in model.model.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is None or not hasattr(attn, "head_dim"):
                continue

            old_head_dim = attn.head_dim
            # Compute num_heads from the old geometry since it is not stored
            # directly on the module in recent transformers versions.
            num_heads = old_dim // old_head_dim
            if num_heads == 0:
                continue

            new_head_dim = new_dim // num_heads
            if new_head_dim * num_heads != new_dim:
                raise ValueError(
                    f"New hidden_size {new_dim} is not divisible by num_attention_heads {num_heads}. "
                    f"Choose a multiplier that yields an integer head_dim."
                )

            attn.head_dim = new_head_dim
            logger.debug(f"Updated head_dim: {old_head_dim} -> {new_head_dim}")

            # Update attention scaling (head_dim**-0.5)
            if hasattr(attn, "scaling"):
                attn.scaling = new_head_dim**-0.5
                logger.debug(f"Updated scaling: {attn.scaling:.4f}")

        # Update the single model-level rotary embedding
        if hasattr(model.model, "rotary_emb"):
            rope = model.model.rotary_emb
            if hasattr(rope, "inv_freq"):
                config = getattr(rope, "config", None) or getattr(model, "config", None)
                if config is not None:
                    base = getattr(config, "rope_theta", 10000.0)
                    if hasattr(config, "rope_parameters"):
                        base = config.rope_parameters.get("rope_theta", base)

                    device = rope.inv_freq.device
                    inv_freq = 1.0 / (
                        base
                        ** (
                            torch.arange(0, new_head_dim, 2, dtype=torch.int64).to(
                                device=device, dtype=torch.float
                            )
                            / new_head_dim
                        )
                    )
                    rope.inv_freq = inv_freq
                    logger.debug(f"Updated rotary inv_freq for dim {new_head_dim}")

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
            new_weight = torch.zeros(
                new_dim, old_weight.shape[1], device=old_weight.device, dtype=old_weight.dtype
            )

            if self.initialization == "copy":
                new_weight[:old_dim, :] = old_weight
                # Copy the last rows for new dimensions
                new_weight[old_dim:, :] = old_weight[-1:, :].expand(new_dim - old_dim, -1)
            elif self.initialization == "zero":
                new_weight[:old_dim, :] = old_weight
                # New dimensions are already zeros
            elif self.initialization == "noise":
                new_weight[:old_dim, :] = old_weight
                new_weight[old_dim:, :] = (
                    torch.randn(
                        new_dim - old_dim,
                        old_weight.shape[1],
                        device=old_weight.device,
                        dtype=old_weight.dtype,
                    )
                    * 0.01
                )

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
            new_weight = torch.zeros(
                old_weight.shape[0], new_dim, device=old_weight.device, dtype=old_weight.dtype
            )

            if self.initialization == "copy":
                new_weight[:, :old_dim] = old_weight
                # Copy the last columns for new dimensions
                new_weight[:, old_dim:] = old_weight[:, -1:].expand(-1, new_dim - old_dim)
            elif self.initialization == "zero":
                new_weight[:, :old_dim] = old_weight
            elif self.initialization == "noise":
                new_weight[:, :old_dim] = old_weight
                new_weight[:, old_dim:] = (
                    torch.randn(
                        old_weight.shape[0],
                        new_dim - old_dim,
                        device=old_weight.device,
                        dtype=old_weight.dtype,
                    )
                    * 0.01
                )

            linear.weight = nn.Parameter(new_weight)
            linear.in_features = new_dim

    def _expand_embedding_dim(self, embed: nn.Embedding, old_dim: int, new_dim: int) -> None:
        """Expand an Embedding layer's embedding dimension (shape[1])."""
        old_weight = embed.weight.data  # (num_embeddings, old_dim)
        new_weight = torch.zeros(
            old_weight.shape[0], new_dim, device=old_weight.device, dtype=old_weight.dtype
        )

        if self.initialization == "copy":
            new_weight[:, :old_dim] = old_weight
            new_weight[:, old_dim:] = old_weight[:, -1:].expand(-1, new_dim - old_dim)
        elif self.initialization == "zero":
            new_weight[:, :old_dim] = old_weight
        elif self.initialization == "noise":
            new_weight[:, :old_dim] = old_weight
            new_weight[:, old_dim:] = (
                torch.randn(
                    old_weight.shape[0],
                    new_dim - old_dim,
                    device=old_weight.device,
                    dtype=old_weight.dtype,
                )
                * 0.01
            )

        embed.weight = nn.Parameter(new_weight)

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
