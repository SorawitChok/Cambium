"""
Base classes for custom Cambium blocks.

Provides the contract that custom blocks must follow and
utility wrappers for inserting them into models.
"""

from abc import ABC, abstractmethod
from typing import ClassVar, List
import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


class CambiumBlock(nn.Module, ABC):
    """
    Base class for all custom blocks inserted by Cambium.

    Every custom block MUST:

    - Accept ``hidden_states`` as first positional arg in ``forward()``
    - Return a tensor of the same shape as ``hidden_states``
    - Accept ``**kwargs`` that it can safely ignore (``attention_mask``, etc.)

    This mirrors the HF ``DecoderLayer`` contract so custom blocks
    drop into the model without signature mismatches.

    Subclasses can optionally declare what config keys they need
    via ``required_config_keys``. ``CustomBlockExpansion`` will validate
    these at expansion time.

    Example::

        class MyBlock(CambiumBlock):
            required_config_keys = ["hidden_size"]

            def __init__(self, config, layer_idx=0):
                super().__init__()
                self.proj = nn.Linear(config.hidden_size, config.hidden_size)

            def forward(self, hidden_states, **kwargs):
                return hidden_states + self.proj(hidden_states)
    """

    required_config_keys: ClassVar[List[str]] = []

    @abstractmethod
    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Process hidden_states and return output of the same shape.

        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            **kwargs: Additional arguments from the model's forward pass
                      (attention_mask, position_ids, etc.). Custom blocks
                      should accept and ignore any they don't need.

        Returns:
            torch.Tensor of shape [batch, seq_len, hidden_dim]
        """
        ...


class ResidualWrapper(nn.Module):
    """
    Wraps a block so output = input + block(input).

    Most custom blocks produce a delta (small adjustment) that
    should be added to the input. This wrapper handles that
    automatically.

    If the user's block already includes a residual connection
    internally, they should set ``residual_connection=False`` when
    calling ``CustomBlockExpansion`` to avoid double-residual.
    """

    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return hidden_states + self.block(hidden_states, **kwargs)

    def __repr__(self) -> str:
        return f"ResidualWrapper({self.block})"


class BlockOutputWrapper(nn.Module):
    """
    Wraps a block so that its output replaces the hidden_states.

    Use this when the block returns the full output (not a delta)
    and you want to replace the hidden_states directly.

    Used internally when residual_connection=False.
    """

    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.block(hidden_states, **kwargs)

    def __repr__(self) -> str:
        return f"BlockOutputWrapper({self.block})"