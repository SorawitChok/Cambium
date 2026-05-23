"""
Cambium - Advanced LLM Architecture Augmentation Library

A library for surgical model expansion, allowing developers to add new layers
and architecture blocks to existing LLMs while leveraging pretrained weights.
"""

__version__ = "0.1.0"

from cambium.core.expansion import ExpansionEngine
from cambium.core.freezing import FreezingManager
from cambium.core.initialization import Initializer, InitializationStrategy
from cambium.strategies.block_expansion import InterleavedExpansion
from cambium.strategies.width_expansion import WidthExpansion
from cambium.strategies.parallel_adapters import ParallelAdapterExpansion
from cambium.strategies.custom_expansion import CustomBlockExpansion
from cambium.blocks.base import CambiumBlock, ResidualWrapper
from cambium.blocks.templates import (
    SwiGLUBlock,
    MultiQueryAttentionBlock,
    GatedResidualBlock,
    CrossAttentionBlock,
    RetentionBlock,
)
from cambium.models.expandable import ExpandableModel
from cambium.training.staged_trainer import StagedTrainer, TrainingPhase
from cambium.training.utilities import TrainingUtilities
from cambium.exceptions import (
    CambiumError,
    BlockValidationError,
    ShapeMismatchError,
    ConfigMismatchError,
    ExpansionError,
)

__all__ = [
    # Core components
    "ExpansionEngine",
    "FreezingManager",
    "Initializer",
    "InitializationStrategy",
    # Expansion strategies
    "InterleavedExpansion",
    "WidthExpansion",
    "ParallelAdapterExpansion",
    "CustomBlockExpansion",
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
    # Exceptions
    "CambiumError",
    "BlockValidationError",
    "ShapeMismatchError",
    "ConfigMismatchError",
    "ExpansionError",
]