"""
Pre-built block templates for common architecture patterns.

Users can import these directly, subclass them, or use them as
reference when building their own custom blocks.
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from cambium.blocks.base import CambiumBlock

__all__ = [
    "SwiGLUBlock",
    "MultiQueryAttentionBlock",
    "GatedResidualBlock",
    "CrossAttentionBlock",
    "RetentionBlock",
]


class SwiGLUBlock(CambiumBlock):
    """
    SwiGLU MLP block (used in LLaMA 2, PaLM).

    SwiGLU is a variant of the standard FFN that uses a gated
    linear unit with SiLU activation:

        gate = SiLU(gate_proj(x))
        up = up_proj(x)
        output = down_proj(gate * up)

    This block returns a delta (output of the MLP only).
    Use with residual_connection=True in CustomBlockExpansion.
    """

    required_config_keys = ["hidden_size"]

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = getattr(config, "intermediate_size", self.hidden_size * 4)
        self.layer_idx = layer_idx

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        gate = self.act(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        output = self.down_proj(gate * up)
        return output


class MultiQueryAttentionBlock(CambiumBlock):
    """
    Multi-Query Attention block (used in PaLM, Falcon).

    Uses a single KV head with multiple query heads for
    efficient inference with lower KV-cache memory.

    This block returns a delta (attention output only).
    Use with residual_connection=True in CustomBlockExpansion.
    """

    required_config_keys = ["hidden_size", "num_attention_heads"]

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.layer_idx = layer_idx

        # Multiple query heads, single KV head
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.layer_norm = nn.LayerNorm(self.hidden_size)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        residual = hidden_states
        x = self.layer_norm(hidden_states)

        batch_size, seq_len, _ = x.shape

        # Q: [batch, num_heads, seq, head_dim]
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # K, V: [batch, 1, seq, head_dim] (single KV head)
        k = self.k_proj(x).unsqueeze(1)
        v = self.v_proj(x).unsqueeze(1)

        # Expand KV to match query heads
        k = k.expand(-1, self.num_heads, -1, -1)
        v = v.expand(-1, self.num_heads, -1, -1)

        # Scaled dot-product attention
        scale = self.head_dim**-0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

        return self.o_proj(attn_output)


class GatedResidualBlock(CambiumBlock):
    """
    Gated Residual block — simple and effective for capacity expansion.

    Uses a single projection with a learned gate:

        x = proj_up(hidden_states)
        x, gate = split(x)
        x = SiLU(x) * sigmoid(gate)
        output = proj_down(x)

    This block returns a delta. Use with residual_connection=True.
    """

    required_config_keys = ["hidden_size"]

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = getattr(config, "intermediate_size", hidden_size * 2)
        self.layer_idx = layer_idx

        self.proj_up = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.proj_down = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.proj_up(hidden_states)
        x, gate = x.chunk(2, dim=-1)
        x = self.act(x) * torch.sigmoid(gate)
        output = self.proj_down(x)
        return output


class CrossAttentionBlock(CambiumBlock):
    """
    Cross-attention block with learned gating.

    Applies self-attention with a learned gate that controls how
    much of the attention output is mixed into the residual stream.
    The gate starts near zero (identity-like) and learns to open.

    This block returns a delta. Use with residual_connection=True.
    """

    required_config_keys = ["hidden_size", "num_attention_heads"]

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = hidden_size // num_heads
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate = nn.Linear(hidden_size, 1, bias=True)

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Initialize gate to near-zero for identity-like behavior
        nn.init.zeros_(self.gate.weight)
        if self.gate.bias is not None:
            nn.init.zeros_(self.gate.bias)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        residual = hidden_states
        x = self.layer_norm(hidden_states)

        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim**-0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, x.shape[-1])

        attn_output = self.o_proj(attn_output)

        # Learned gating
        gate_val = torch.sigmoid(self.gate(x))
        return gate_val * attn_output


class RetentionBlock(CambiumBlock):
    """
    Retention block — alternative to attention with linear complexity.

    Based on "Retentive Network: A Successor to Transformer for Large
    Language Models" (Sun et al., 2023). Uses a retention mechanism
    instead of softmax attention, achieving O(1) inference and O(n)
    training complexity.

    This block returns a delta. Use with residual_connection=True.
    """

    required_config_keys = ["hidden_size", "num_attention_heads"]

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = hidden_size // num_heads
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate = nn.Linear(hidden_size, 1, bias=True)

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Decay factor for retention
        self.decay = 0.5 ** (8 / num_heads)

        # Initialize gate near zero
        nn.init.zeros_(self.gate.weight)
        if self.gate.bias is not None:
            nn.init.zeros_(self.gate.bias)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.layer_norm(hidden_states)
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Retention: Q * (K^T * decay_matrix) * V
        # Build decay matrix
        positions = torch.arange(seq_len, device=x.device)
        distance = positions.unsqueeze(0) - positions.unsqueeze(1)
        decay_matrix = self.decay ** distance.float()
        # Causal mask: future positions get zero
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        decay_matrix = decay_matrix * causal_mask

        # Retention computation
        retention = torch.matmul(q, k.transpose(-2, -1))
        retention = retention * decay_matrix.unsqueeze(0).unsqueeze(0)
        # No softmax — retention uses element-wise scaling
        retention = retention / (self.head_dim**0.5)

        output = torch.matmul(retention, v)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, x.shape[-1])

        output = self.o_proj(output)

        # Learned gating
        gate_val = torch.sigmoid(self.gate(x))
        return gate_val * output
