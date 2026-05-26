"""Core components for model expansion and surgical modification."""

from cambium.core.expansion import ExpansionEngine
from cambium.core.freezing import FreezingManager
from cambium.core.initialization import InitializationStrategy, Initializer

__all__ = ["ExpansionEngine", "FreezingManager", "Initializer", "InitializationStrategy"]
