"""
Parallel adapter expansion strategy.

Adds parallel pathways alongside existing transformer blocks,
similar to MoE-lite or parallel adapter architectures.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn

from cambium.core.expansion import ExpansionEngine
from cambium.core.initialization import InitializationStrategy, Initializer

logger = logging.getLogger(__name__)


class ParallelBottleneckAdapter(nn.Module):
    """
    A parallel bottleneck adapter module.

    Processes input through a bottleneck projection and adds to the main path.
    Similar to LoRA but with learned gating.
    """

    def __init__(
        self,
        hidden_dim: int,
        bottleneck_dim: int,
        init_scale: float = 0.01,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.init_scale = init_scale

        self.down_proj = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(bottleneck_dim, hidden_dim, bias=False)
        self.gate = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize with small values for near-identity behavior."""
        nn.init.normal_(self.down_proj.weight, std=self.init_scale)
        nn.init.normal_(self.up_proj.weight, std=self.init_scale)
        nn.init.zeros_(self.gate.weight)
        if self.gate.bias is not None:
            nn.init.zeros_(self.gate.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Process input and return adapter output.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]

        Returns:
            Adapter output [batch, seq_len, hidden_dim]
        """
        # Compute gate value (soft gating)
        gate_val = torch.sigmoid(self.gate(hidden_states))

        # Bottleneck transformation
        down = self.down_proj(hidden_states)
        activated = self.activation(down)
        up = self.up_proj(activated)

        # Apply gating
        return gate_val * up


class ParallelAttentionAdapter(nn.Module):
    """
    A parallel cross-attention adapter.

    Adds a learnable cross-attention mechanism alongside the self-attention.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        init_scale: float = 0.01,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.init_scale = init_scale

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize with small values."""
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.normal_(proj.weight, std=self.init_scale)
        nn.init.zeros_(self.gate.weight)
        if self.gate.bias is not None:
            nn.init.zeros_(self.gate.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply parallel attention.

        Args:
            hidden_states: [batch, seq_len, hidden_dim]

        Returns:
            Adapter output [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Compute projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and output projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)
        output = self.o_proj(attn_output)

        # Apply gating
        gate_val = torch.sigmoid(self.gate(hidden_states))
        return gate_val * output


@dataclass
class ParallelAdapterExpansion:
    """
    Add parallel adapter pathways to transformer blocks.

    Unlike block expansion which inserts new blocks between existing ones,
    this adds parallel modules alongside existing blocks.
    """

    adapter_type: str = "bottleneck"
    """Type of adapter: 'bottleneck', 'attention', or 'mlp'."""

    bottleneck_dim: int = 256
    """Dimension for bottleneck adapters."""

    num_heads: int = 4
    """Number of attention heads for attention adapters."""

    initialization: str = "zero"
    """Initialization strategy."""

    layer_attribute: str = "model.layers"
    """Path to transformer layers."""

    target_layers: list[int] | None = field(default_factory=list)
    """Specific layer indices to add adapters to. If empty, add to all."""

    def expand(self, model: nn.Module, engine: ExpansionEngine) -> nn.Module:
        """
        Add parallel adapters to the model.

        Args:
            model: The model to augment
            engine: ExpansionEngine instance

        Returns:
            The augmented model
        """
        logger.info(
            f"Adding {self.adapter_type} adapters with " f"bottleneck_dim={self.bottleneck_dim}"
        )

        layers_module = self._get_layers_module(model)
        num_layers = len(layers_module)

        # Determine which layers to target
        if self.target_layers:
            target_indices = self.target_layers
        else:
            target_indices = list(range(num_layers))

        # Add adapters to each target layer
        for layer_idx in target_indices:
            if layer_idx >= num_layers:
                logger.warning(f"Layer index {layer_idx} out of range, skipping")
                continue

            layer = layers_module[layer_idx]
            adapter = self._create_adapter(model)

            # Store adapter in the layer
            if not hasattr(layer, "cambium_adapters"):
                layer.cambium_adapters = nn.ModuleList()
            layer.cambium_adapters.append(adapter)

            # Wrap the forward method to include adapter
            self._wrap_layer_forward(layer)

            logger.debug(f"Added adapter to layer {layer_idx}")

        logger.info(f"Added adapters to {len(target_indices)} layers")
        return model

    def _get_layers_module(self, model: nn.Module) -> nn.ModuleList:
        """Get the layers ModuleList."""
        parts = self.layer_attribute.split(".")
        module = model
        for part in parts:
            module = getattr(module, part)
        return module

    def _create_adapter(self, model: nn.Module) -> nn.Module:
        """Create an adapter module based on configuration."""
        hidden_dim = model.config.hidden_size

        if self.adapter_type == "bottleneck":
            return ParallelBottleneckAdapter(
                hidden_dim=hidden_dim,
                bottleneck_dim=self.bottleneck_dim,
            )
        elif self.adapter_type == "attention":
            return ParallelAttentionAdapter(
                hidden_dim=hidden_dim,
                num_heads=self.num_heads,
            )
        else:
            raise ValueError(f"Unknown adapter type: {self.adapter_type}")

    def _wrap_layer_forward(self, layer: nn.Module) -> None:
        """
        Wrap the layer's forward method to include adapter outputs.

        This modifies the layer to call adapters after the original forward.
        """
        original_forward = layer.forward

        def new_forward(hidden_states, *args, **kwargs):
            # Call original forward
            output = original_forward(hidden_states, *args, **kwargs)

            # Handle both tuple outputs (hidden_states, ...) and single tensors
            if isinstance(output, tuple):
                hidden_out = output[0]
            else:
                hidden_out = output

            # Apply adapters
            if hasattr(layer, "cambium_adapters"):
                for adapter in layer.cambium_adapters:
                    adapter_out = adapter(hidden_out)
                    hidden_out = hidden_out + adapter_out

            # Reconstruct output
            if isinstance(output, tuple):
                return (hidden_out,) + output[1:]
            else:
                return hidden_out

        layer.forward = new_forward


@dataclass
class MixtureOfExpertsExpansion:
    """
    Add MoE-style parallel experts to the model.

    This is a more advanced parallel expansion that adds multiple expert
    pathways with a learned routing mechanism.
    """

    num_experts: int = 8
    """Number of expert pathways."""

    expert_dim: int = 512
    """Hidden dimension for each expert."""

    top_k: int = 2
    """Number of experts to route to for each token."""

    target_layers: list[int] | None = field(default_factory=list)
    """Layers to add MoE to. If empty, add to FFN layers."""

    def expand(self, model: nn.Module, engine: ExpansionEngine) -> nn.Module:
        """Add MoE layers to the model."""
        logger.info(f"Adding MoE with {self.num_experts} experts (top-{self.top_k})")
        # Implementation would go here
        # This is a placeholder for the full MoE implementation
        raise NotImplementedError("MoE expansion not yet implemented")
