"""
Cambium - Advanced LLM Architecture Augmentation Library

A library for surgical model expansion, allowing developers to add new layers
and architecture blocks to existing LLMs while leveraging pretrained weights.
"""

__version__ = "0.1.1"

from cambium.blocks.base import CambiumBlock, ResidualWrapper
from cambium.blocks.templates import (
    CrossAttentionBlock,
    GatedResidualBlock,
    MultiQueryAttentionBlock,
    RetentionBlock,
    SwiGLUBlock,
)
from cambium.core.expansion import ExpansionEngine
from cambium.core.freezing import FreezingManager
from cambium.core.initialization import InitializationStrategy, Initializer
from cambium.exceptions import (
    BlockValidationError,
    CambiumError,
    ConfigMismatchError,
    DataError,
    ExpansionError,
    GraftingError,
    ShapeMismatchError,
)
from cambium.models.expandable import ExpandableModel
from cambium.strategies.block_expansion import AppendExpansion, InterleavedExpansion
from cambium.strategies.custom_expansion import CustomBlockExpansion
from cambium.strategies.grafted_expansion import GraftedBlockExpansion
from cambium.strategies.parallel_adapters import ParallelAdapterExpansion
from cambium.strategies.width_expansion import WidthExpansion
from cambium.training.data import (
    AlpacaFormatter,
    ChatFormatter,
    DataConfig,
    TextFormatter,
    build_text_dataloader,
)
from cambium.training.staged_trainer import StagedTrainer, TrainingPhase
from cambium.training.utilities import TrainingUtilities

__all__ = [
    # Core components
    "ExpansionEngine",
    "FreezingManager",
    "Initializer",
    "InitializationStrategy",
    # Expansion strategies
    "InterleavedExpansion",
    "AppendExpansion",
    "WidthExpansion",
    "ParallelAdapterExpansion",
    "CustomBlockExpansion",
    "GraftedBlockExpansion",
    # Custom blocks
    "CambiumBlock",
    "ResidualWrapper",
    "SwiGLUBlock",
    "MultiQueryAttentionBlock",
    "GatedResidualBlock",
    "CrossAttentionBlock",
    "RetentionBlock",
    # Models
    "ExpandableModel",
    # Training
    "StagedTrainer",
    "TrainingPhase",
    "TrainingUtilities",
    "DataConfig",
    "build_text_dataloader",
    "TextFormatter",
    "AlpacaFormatter",
    "ChatFormatter",
    # Exceptions
    "CambiumError",
    "BlockValidationError",
    "ShapeMismatchError",
    "ConfigMismatchError",
    "ExpansionError",
    "GraftingError",
    "DataError",
]
