"""Expansion strategies for model augmentation."""

from cambium.strategies.block_expansion import AppendExpansion, InterleavedExpansion
from cambium.strategies.custom_expansion import CustomBlockExpansion
from cambium.strategies.parallel_adapters import ParallelAdapterExpansion
from cambium.strategies.width_expansion import WidthExpansion

__all__ = [
    "InterleavedExpansion",
    "AppendExpansion",
    "WidthExpansion",
    "ParallelAdapterExpansion",
    "CustomBlockExpansion",
]

# Registry of strategy class name -> dataclass class. Used by
# ExpandableModel.load_expanded to reconstruct a strategy from its
# stored config so the model can be re-expanded in-place after load.
#
# Only strategies whose configs are pure JSON-serializable primitives
# can be round-tripped automatically. Strategies that capture callables
# (block_class, block_factory, custom_init_fn) cannot be reconstructed
# from the JSON metadata alone; the user must re-apply those manually.
STRATEGY_REGISTRY = {
    cls.__name__: cls
    for cls in (InterleavedExpansion, AppendExpansion, WidthExpansion, ParallelAdapterExpansion)
}
