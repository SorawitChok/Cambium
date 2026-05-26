"""Expansion strategies for model augmentation."""

from cambium.strategies.block_expansion import InterleavedExpansion
from cambium.strategies.custom_expansion import CustomBlockExpansion
from cambium.strategies.parallel_adapters import ParallelAdapterExpansion
from cambium.strategies.width_expansion import WidthExpansion

__all__ = [
    "InterleavedExpansion",
    "WidthExpansion",
    "ParallelAdapterExpansion",
    "CustomBlockExpansion",
]
