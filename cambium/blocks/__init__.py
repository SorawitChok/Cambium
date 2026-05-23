"""
Custom block definitions and templates for Cambium.
"""

from cambium.blocks.base import CambiumBlock, ResidualWrapper, BlockOutputWrapper
from cambium.blocks.templates import (
    SwiGLUBlock,
    MultiQueryAttentionBlock,
    GatedResidualBlock,
    CrossAttentionBlock,
    RetentionBlock,
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