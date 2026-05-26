"""
Initialization strategies for new model components.
"""

import logging
from enum import Enum
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)


class InitializationStrategy(Enum):
    """Available initialization strategies for new layers."""

    IDENTITY_MAPPING = "identity"
    SMALL_RANDOM = "small_random"
    NOISE_INJECTION = "noise"
    KNOWLEDGE_DISTILL = "distill"
    ZERO_INIT = "zero"
    XAVIER_UNIFORM = "xavier_uniform"
    KAIMING_NORMAL = "kaiming_normal"


class Initializer:
    """
    Manages initialization of new modules to ensure stable training.

    Different strategies help preserve the original model's behavior
    while allowing the new components to learn effectively.
    """

    def __init__(self, strategy: InitializationStrategy = InitializationStrategy.IDENTITY_MAPPING):
        """
        Initialize the initializer with a default strategy.

        Args:
            strategy: Default initialization strategy to use
        """
        self.default_strategy = strategy

    def apply(
        self,
        new_modules: List[nn.Module],
        strategy: Optional[InitializationStrategy] = None,
        reference_modules: Optional[List[nn.Module]] = None,
        scale: float = 1.0,
    ) -> None:
        """
        Apply initialization to new modules.

        Args:
            new_modules: List of modules to initialize
            strategy: Initialization strategy to use (overrides default)
            reference_modules: Optional reference modules for strategies
                             that require copying (e.g., identity mapping)
            scale: Scaling factor for some initialization strategies
        """
        strategy = strategy or self.default_strategy

        if strategy == InitializationStrategy.IDENTITY_MAPPING:
            self._init_identity(new_modules, reference_modules, scale)
        elif strategy == InitializationStrategy.SMALL_RANDOM:
            self._init_small_random(new_modules, scale)
        elif strategy == InitializationStrategy.NOISE_INJECTION:
            self._init_noise(new_modules, scale)
        elif strategy == InitializationStrategy.ZERO_INIT:
            self._init_zero(new_modules)
        elif strategy == InitializationStrategy.XAVIER_UNIFORM:
            self._init_xavier(new_modules)
        elif strategy == InitializationStrategy.KAIMING_NORMAL:
            self._init_kaiming(new_modules)
        elif strategy == InitializationStrategy.KNOWLEDGE_DISTILL:
            self._init_distill(new_modules, reference_modules)
        else:
            raise ValueError(f"Unknown initialization strategy: {strategy}")

        logger.info(f"Applied {strategy.value} initialization to {len(new_modules)} modules")

    def _init_identity(
        self,
        new_modules: List[nn.Module],
        reference_modules: Optional[List[nn.Module]],
        scale: float,
    ) -> None:
        """
        Initialize to approximate identity mapping.

        For transformer blocks, this uses zero initialization for output projections
        and small random init for other layers, making the new block behave
        approximately as identity (preserving original model behavior initially).
        """
        for module in new_modules:
            if isinstance(module, nn.Linear):
                # For output projections (usually the last linear in a block),
                # use near-zero initialization
                if module.out_features == module.in_features:
                    # Could be an identity-like layer
                    nn.init.eye_(module.weight)
                    module.weight.data *= scale
                else:
                    nn.init.xavier_uniform_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

            elif isinstance(module, nn.RMSNorm):
                nn.init.ones_(module.weight)

    def _init_small_random(self, modules: List[nn.Module], scale: float) -> None:
        """Initialize with small random values (Gaussian noise)."""
        for module in modules:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02 * scale)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02 * scale)

    def _init_noise(self, modules: List[nn.Module], scale: float) -> None:
        """Initialize with larger noise for more diversity."""
        for module in modules:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.1 * scale)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.1 * scale)

    def _init_zero(self, modules: List[nn.Module]) -> None:
        """Zero initialization (useful for output layers in residual connections)."""
        for module in modules:
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.zeros_(module.weight)

    def _init_xavier(self, modules: List[nn.Module]) -> None:
        """Xavier/Glorot initialization."""
        for module in modules:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)

    def _init_kaiming(self, modules: List[nn.Module]) -> None:
        """Kaiming/He initialization (good for ReLU/LeakyReLU)."""
        for module in modules:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _init_distill(
        self,
        new_modules: List[nn.Module],
        reference_modules: Optional[List[nn.Module]],
    ) -> None:
        """
        Initialize from a reference (teacher) model.

        This copies weights from corresponding reference modules,
        useful when you have a larger teacher model to initialize from.
        """
        if reference_modules is None:
            raise ValueError("reference_modules required for knowledge_distill strategy")

        if len(new_modules) != len(reference_modules):
            raise ValueError(
                f"Module count mismatch: {len(new_modules)} new vs {len(reference_modules)} reference"
            )

        for new_mod, ref_mod in zip(new_modules, reference_modules):
            self._copy_weights(new_mod, ref_mod)

    def _copy_weights(self, target: nn.Module, source: nn.Module) -> None:
        """Copy weights from source to target, handling dimension mismatches."""
        for (name_tgt, param_tgt), (name_src, param_src) in zip(
            target.named_parameters(), source.named_parameters()
        ):
            if param_tgt.shape == param_src.shape:
                param_tgt.data.copy_(param_src.data)
            else:
                # Handle dimension mismatch by copying what we can
                min_shape = tuple(min(s1, s2) for s1, s2 in zip(param_tgt.shape, param_src.shape))
                slices_tgt = tuple(slice(0, s) for s in min_shape)
                slices_src = tuple(slice(0, s) for s in min_shape)
                param_tgt.data[slices_tgt].copy_(param_src.data[slices_src])
                logger.warning(
                    f"Copied partial weights for {name_tgt}: "
                    f"{param_src.shape} -> {param_tgt.shape}"
                )

    def smart_init_for_block(self, block: nn.Module, model_type: str = "llama") -> None:
        """
        Apply model-specific smart initialization to a transformer block.

        Args:
            block: The transformer block to initialize
            model_type: Model architecture type (llama, gemma, mistral, etc.)
        """
        # Find output projections in the block
        output_projections = []

        # Common patterns for output projections
        for name, module in block.named_modules():
            if isinstance(module, nn.Linear):
                # Heuristics: last linear in attention MLP is usually output
                if "o_proj" in name or "out_proj" in name or "down_proj" in name:
                    output_projections.append((name, module))

        # Initialize output projections to near-zero (identity-like)
        for name, module in output_projections:
            nn.init.normal_(module.weight, mean=0.0, std=0.001)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            logger.debug(f"Smart init: Near-zero init for {name}")

        # Initialize other layers with standard initialization
        for name, module in block.named_modules():
            if isinstance(module, nn.Linear) and (name, module) not in output_projections:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
                nn.init.ones_(module.weight)
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.zeros_(module.bias)


class IdentityInitializer:
    """
    Convenience class for identity-mapping initialization.

    This is the recommended default for most expansions as it preserves
    the original model's behavior initially, allowing the new layers
    to warm up gradually during training.
    """

    def __init__(self, output_scale: float = 0.001):
        """
        Args:
            output_scale: Scale factor for output projection initialization.
                         Smaller values make the block more identity-like.
        """
        self.output_scale = output_scale

    def __call__(self, module: nn.Module) -> None:
        """Apply identity initialization to a module."""
        initializer = Initializer(InitializationStrategy.IDENTITY_MAPPING)
        initializer.apply([module], scale=self.output_scale)
