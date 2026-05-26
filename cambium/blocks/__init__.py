"""
Custom block definitions and templates for Cambium.
"""

from cambium.blocks.base import BlockOutputWrapper, CambiumBlock, ResidualWrapper
from cambium.blocks.templates import (
    CrossAttentionBlock,
    GatedResidualBlock,
    MultiQueryAttentionBlock,
    RetentionBlock,
    SwiGLUBlock,
)

__all__ = [
    "CambiumBlock",
    "ResidualWrapper",
    "BlockOutputWrapper",
    "SwiGLUBlock",
    "MultiQueryAttentionBlock",
    "GatedResidualBlock",
    "CrossAttentionBlock",
    "RetentionBlock",
]
