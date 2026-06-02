"""
Custom block expansion strategy.

Inserts user-defined blocks into an existing model.
Supports three input modes: block class, block factory, or pre-created instances.
"""

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Type

import torch
from torch import nn

from cambium.blocks.base import CambiumBlock, ResidualWrapper
from cambium.core.expansion import ExpansionEngine
from cambium.core.initialization import InitializationStrategy, Initializer
from cambium.exceptions import BlockValidationError, ShapeMismatchError

logger = logging.getLogger(__name__)


@dataclass
class CustomBlockExpansion:
    """
    Insert user-defined blocks into a model.

    Supports three modes:

    1. ``block_class``: Pass a class that takes ``(config, layer_idx)`` and returns ``nn.Module``
    2. ``block_factory``: Pass a callable that returns an ``nn.Module`` instance
    3. ``block_instances``: Pass pre-created ``nn.Module`` instances (one per position)

    The block must:

    - Accept ``hidden_states`` as first arg in ``forward()``
    - Return a tensor of the same shape as input
    - Accept ``**kwargs`` for arguments it doesn't use

    Example (using a template)::

        from cambium import ExpandableModel, CustomBlockExpansion
        from cambium.blocks import SwiGLUBlock

        model = ExpandableModel.from_pretrained("google/gemma-2b")
        model.expand(CustomBlockExpansion(
            block_class=SwiGLUBlock,
            num_layers=4,
            residual_connection=True,
        ))

    Example (using a custom block)::

        class MyBlock(CambiumBlock):
            required_config_keys = ["hidden_size"]

            def __init__(self, config, layer_idx=0):
                super().__init__()
                self.proj = nn.Linear(config.hidden_size, config.hidden_size)

            def forward(self, hidden_states, **kwargs):
                return self.proj(hidden_states)

        model.expand(CustomBlockExpansion(
            block_class=MyBlock,
            positions=[8, 16],
            residual_connection=True,
        ))
    """

    # Exactly one of these must be provided
    block_class: Type[nn.Module] | None = None
    """Block class to instantiate. Called as block_class(config, layer_idx=i)."""

    block_factory: Callable[[], nn.Module] | None = None
    """Factory callable that returns a new nn.Module instance."""

    block_instances: list[nn.Module] | None = None
    """Pre-created block instances. Must match len(positions) or num_layers."""

    # Where and how many
    num_layers: int | None = None
    """Number of blocks to insert. Auto-distributes if positions is None."""

    positions: list[int] | None = None
    """Specific positions to insert blocks at. Overrides num_layers distribution."""

    # Initialization
    initialization: str = "smart"
    """Initialization strategy: 'smart', 'identity', 'small_random', 'noise',
       'zero', 'xavier', 'kaiming', 'custom'."""

    custom_init_fn: Callable[[nn.Module], None] | None = None
    """Custom initialization function. Called with each new block.
       Only used when initialization='custom'."""

    # Validation
    validate: bool = True
    """Whether to run shape and signature validation before insertion."""

    # Adapter wrapping
    residual_connection: bool = True
    """Whether to wrap the block in a residual connection (output = input + block(input)).
       Set to False if the block already includes its own residual."""

    # Model traversal
    layer_attribute: str = "model.layers"
    """Dot-separated path to the layers ModuleList."""

    def expand(self, model: nn.Module, engine: ExpansionEngine) -> nn.Module:
        """
        Apply custom block expansion to a model.

        Args:
            model: The model to expand
            engine: ExpansionEngine instance

        Returns:
            The expanded model (modified in-place)
        """
        logger.info("Starting custom block expansion")

        # 1. Validate inputs
        self._validate_inputs()

        # 2. Determine positions
        layers_module = self._get_layers_module(model)
        current_layers = len(layers_module)
        positions = self._compute_positions(current_layers)
        logger.info(f"Will insert {len(positions)} custom blocks at positions: {positions}")

        # 3. Create blocks
        blocks = self._create_blocks(model, len(positions))

        # 4. Validate blocks (if enabled)
        if self.validate:
            self._validate_blocks(blocks, model)

        # 5. Optionally wrap in residual connection
        if self.residual_connection:
            blocks = [ResidualWrapper(b) for b in blocks]
            logger.debug(f"Wrapped {len(blocks)} blocks in ResidualWrapper")

        # 6. Insert blocks via engine
        # We need a factory that returns our pre-created blocks
        block_iter = iter(blocks)

        def block_factory():
            return next(block_iter)

        engine.insert_blocks(
            model,
            positions,
            block_factory,
            block_attribute=self.layer_attribute,
        )

        # Initialize the newly inserted blocks directly (no position lookups).
        self._apply_initialization(model, blocks)

        self._update_config(model, positions)

        logger.info(f"Custom block expansion complete: inserted {len(positions)} blocks")
        return model

    def _validate_inputs(self) -> None:
        """Validate that exactly one block source is provided."""
        sources = [
            self.block_class is not None,
            self.block_factory is not None,
            self.block_instances is not None,
        ]
        provided = sum(sources)

        if provided == 0:
            raise ValueError(
                "Must provide exactly one of: block_class, block_factory, or block_instances"
            )
        if provided > 1:
            raise ValueError(
                "Must provide exactly one of: block_class, block_factory, or block_instances. "
                f"Got {provided}."
            )

        if self.block_instances is not None:
            if self.positions is not None and len(self.block_instances) != len(self.positions):
                raise ValueError(
                    f"Number of block_instances ({len(self.block_instances)}) must match "
                    f"number of positions ({len(self.positions)})"
                )
            if self.num_layers is not None and len(self.block_instances) != self.num_layers:
                raise ValueError(
                    f"Number of block_instances ({len(self.block_instances)}) must match "
                    f"num_layers ({self.num_layers})"
                )

        if self.positions is None and self.num_layers is None and self.block_instances is None:
            raise ValueError(
                "Must provide either num_layers or positions to determine where to insert blocks"
            )

    def _get_layers_module(self, model: nn.Module) -> nn.ModuleList:
        """Get the layers ModuleList from the model."""
        parts = self.layer_attribute.split(".")
        module = model
        for part in parts:
            module = getattr(module, part)
        return module

    def _compute_positions(self, current_layers: int) -> list[int]:
        """
        Compute insertion positions.

        If positions provided, use those.
        If block_instances provided, use num_layers or len(block_instances).
        Otherwise auto-distribute num_layers blocks evenly.
        """
        if self.positions is not None:
            for pos in self.positions:
                if pos < 0 or pos > current_layers:
                    raise ValueError(
                        f"Invalid position {pos}. Must be between 0 and {current_layers}"
                    )
            return sorted(self.positions)

        # Determine how many blocks to insert
        if self.block_instances is not None:
            n = len(self.block_instances)
        elif self.num_layers is not None:
            n = self.num_layers
        else:
            raise ValueError("Cannot determine number of blocks to insert")

        # Auto-distribute evenly
        if n >= current_layers:
            return list(range(1, current_layers + 1))

        step = current_layers / (n + 1)
        positions = []
        for i in range(n):
            pos = int((i + 1) * step)
            if positions and pos == positions[-1]:
                pos += 1
            positions.append(pos)

        return positions

    def _create_blocks(self, model: nn.Module, count: int) -> list[nn.Module]:
        """Create block instances based on which input mode was used."""
        config = model.config

        if self.block_instances is not None:
            logger.debug(f"Using {len(self.block_instances)} pre-created block instances")
            return list(self.block_instances)

        if self.block_class is not None:
            block_class = self.block_class
            blocks = []
            for i in range(count):
                try:
                    block = block_class(config, layer_idx=i)
                except TypeError:
                    # Block class might not accept layer_idx
                    try:
                        block = block_class(config)
                    except TypeError:
                        # Block class might not accept config either
                        block = block_class()
                blocks.append(block)
                logger.debug(f"Created block {i} from class {block_class.__name__}")
            return blocks

        if self.block_factory is not None:
            blocks = [self.block_factory() for i in range(count)]
            logger.debug(f"Created {count} blocks from factory")
            return blocks

        # Should not reach here after _validate_inputs
        raise ValueError("No block source provided")

    def _validate_blocks(self, blocks: list[nn.Module], model: nn.Module) -> None:
        """
        Validate that blocks are compatible with the model.

        Checks:
        1. Output shape matches input shape
        2. Forward signature accepts hidden_states and **kwargs
        3. Required config keys are present (for CambiumBlock subclasses)
        4. No NaN in parameters after initialization
        """
        hidden_size = model.config.hidden_size
        errors: list[str] = []

        for i, block in enumerate(blocks):
            # Check 1: Forward signature
            try:
                sig = inspect.signature(block.forward)
                params = list(sig.parameters.keys())
                if len(params) == 0:
                    errors.append(
                        f"Block {i}: forward() has no parameters. "
                        f"Must accept hidden_states as first arg."
                    )
                else:
                    first_param = params[0]
                    if first_param not in ("hidden_states", "x", "input", "args"):
                        logger.warning(
                            f"Block {i}: first parameter is '{first_param}', "
                            f"expected 'hidden_states'. This may work but is unconventional."
                        )

                    # Check if **kwargs is accepted
                    has_var_keyword = any(
                        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
                    )
                    if not has_var_keyword:
                        logger.warning(
                            f"Block {i}: forward() doesn't accept **kwargs. "
                            f"Model may pass additional arguments (attention_mask, etc.) "
                            f"that this block won't handle. Consider adding **kwargs."
                        )
            except (ValueError, TypeError) as e:
                logger.warning(f"Block {i}: Could not inspect forward signature: {e}")

            # Check 2: Shape validation
            block_eval = block
            if isinstance(block, ResidualWrapper):
                block_eval = block.block

            block_eval.eval()
            dummy = torch.randn(1, 1, hidden_size)
            with torch.no_grad():
                try:
                    output = block_eval(dummy)
                    if isinstance(output, tuple):
                        # Some HF layers return (hidden_states, ...)
                        output = output[0]
                    if output.shape != dummy.shape:
                        if self.residual_connection:
                            errors.append(
                                f"Block {i}: output shape {tuple(output.shape)} "
                                f"doesn't match input shape {tuple(dummy.shape)}. "
                                f"With residual_connection=True, block output must "
                                f"match input shape for proper addition."
                            )
                        else:
                            logger.warning(
                                f"Block {i}: output shape {tuple(output.shape)} "
                                f"differs from input shape {tuple(dummy.shape)}. "
                                f"Ensure this is intentional."
                            )
                except TypeError as e:
                    logger.warning(
                        f"Block {i}: forward() failed with dummy input: {e}. "
                        f"The block may require additional arguments."
                    )
                except Exception as e:
                    logger.warning(f"Block {i}: forward() raised {type(e).__name__}: {e}")
            block_eval.train()

            # Check 3: Config key validation (CambiumBlock subclasses)
            if isinstance(block, CambiumBlock):
                missing = []
                for key in block.required_config_keys:
                    if not hasattr(model.config, key):
                        missing.append(key)
                if missing:
                    available = (
                        list(model.config.to_dict().keys())
                        if hasattr(model.config, "to_dict")
                        else []
                    )
                    errors.append(
                        f"Block {i}: requires config keys {missing} "
                        f"not found in model config. Available: {available[:20]}"
                    )

            # Check 4: NaN detection
            has_nan = any(torch.isnan(p).any().item() for p in block.parameters() if p.numel() > 0)
            if has_nan:
                errors.append(f"Block {i}: has NaN parameters after initialization")

        if errors:
            error_msg = "\n".join(f"  - {e}" for e in errors)
            raise BlockValidationError(
                block_idx=-1,
                reason=f"Validation failed with {len(errors)} error(s):\n{error_msg}",
            )

        logger.info(f"All {len(blocks)} blocks passed validation")

    def _apply_initialization(self, model: nn.Module, blocks: list[nn.Module]) -> None:
        """Apply initialization strategy to the newly inserted blocks.

        Args:
            model: The expanded model. Only used for ``model_type``-aware smart init.
            blocks: The block instances that were just inserted (after optional
                ``ResidualWrapper`` wrapping).
        """
        if self.initialization == "custom":
            if self.custom_init_fn is None:
                raise ValueError("custom_init_fn must be provided when initialization='custom'")
            for i, block in enumerate(blocks):
                # If wrapped in ResidualWrapper, initialize the inner block
                inner = block.block if isinstance(block, ResidualWrapper) else block
                self.custom_init_fn(inner)
                logger.debug(f"Applied custom initialization to block {i}")
            return

        # Map initialization string to strategy
        strategy_map = {
            "smart": None,  # Handled separately
            "identity": InitializationStrategy.IDENTITY_MAPPING,
            "small_random": InitializationStrategy.SMALL_RANDOM,
            "noise": InitializationStrategy.NOISE_INJECTION,
            "zero": InitializationStrategy.ZERO_INIT,
            "xavier": InitializationStrategy.XAVIER_UNIFORM,
            "kaiming": InitializationStrategy.KAIMING_NORMAL,
        }

        if self.initialization not in strategy_map:
            raise ValueError(
                f"Unknown initialization strategy: {self.initialization}. "
                f"Must be one of: {list(strategy_map.keys())}, 'custom'"
            )

        initializer = Initializer()

        for i, block in enumerate(blocks):
            inner = block.block if isinstance(block, ResidualWrapper) else block

            if self.initialization == "smart":
                # Smart init: near-zero for output projections, standard for rest
                initializer.smart_init_for_block(inner, model_type="custom")
            else:
                strategy = strategy_map[self.initialization]
                modules = list(inner.modules())[1:]  # Skip the block itself
                initializer.apply(modules, strategy=strategy)

            logger.debug(f"Applied {self.initialization} initialization to block {i}")

    def _update_config(self, model: nn.Module, positions: list[int]) -> None:
        """Update model config to reflect the custom blocks."""
        num_new = len(positions)

        if hasattr(model.config, "num_hidden_layers"):
            model.config.num_hidden_layers += num_new

        if hasattr(model.config, "_name_or_path"):
            model.config._name_or_path += f"_cambium_custom_{num_new}L"

        # Track custom block metadata
        custom_blocks = getattr(model.config, "_cambium_custom_blocks", [])

        block_name = "unknown"
        if self.block_class is not None:
            block_name = self.block_class.__name__
        elif self.block_factory is not None:
            block_name = getattr(self.block_factory, "__name__", "factory")
        elif self.block_instances is not None:
            inner = self.block_instances[0]
            if isinstance(inner, ResidualWrapper):
                inner = inner.block
            block_name = type(inner).__name__

        custom_blocks.append(
            {
                "block_type": block_name,
                "positions": positions.copy(),
                "residual": self.residual_connection,
                "initialization": self.initialization,
            }
        )

        model.config._cambium_custom_blocks = custom_blocks
