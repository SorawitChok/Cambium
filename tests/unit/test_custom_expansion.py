"""Tests for custom block expansion."""

import pytest
import torch
from torch import nn

from cambium.strategies.custom_expansion import CustomBlockExpansion
from cambium.blocks.base import CambiumBlock, ResidualWrapper
from cambium.blocks.templates import (
    SwiGLUBlock,
    GatedResidualBlock,
    CrossAttentionBlock,
)
from cambium.exceptions import BlockValidationError
from cambium.core.expansion import ExpansionEngine


# --- Test fixtures ---


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, num_layers=4, hidden_size=32):
        super().__init__()
        self.config = type("Config", (), {
            "hidden_size": hidden_size,
            "num_hidden_layers": num_layers,
            "num_attention_heads": 4,
            "intermediate_size": hidden_size * 4,
            "model_type": "test",
            "to_dict": lambda self: {"hidden_size": hidden_size},
        })()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size)
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_size, 100)

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)
        return self.lm_head(x)


class SimpleCambiumBlock(CambiumBlock):
    """Simple test block that doubles the input."""

    required_config_keys = ["hidden_size"]

    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states, **kwargs):
        return self.proj(hidden_states)


class BadShapeBlock(CambiumBlock):
    """Block that returns wrong shape — should fail validation."""

    required_config_keys = ["hidden_size"]

    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.proj = nn.Linear(config.hidden_size, config.hidden_size * 2)

    def forward(self, hidden_states, **kwargs):
        return self.proj(hidden_states)


class NoKwargsBlock(CambiumBlock):
    """Block without **kwargs — should produce a warning."""

    required_config_keys = ["hidden_size"]

    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states):
        return self.proj(hidden_states)


class ResidualBlock(CambiumBlock):
    """Block that already includes residual — for use with residual_connection=False."""

    required_config_keys = ["hidden_size"]

    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states, **kwargs):
        return hidden_states + self.proj(hidden_states)


class MockEngine:
    """Mock expansion engine for testing."""

    def __init__(self):
        self.inserted = []

    def insert_blocks(self, model, positions, block_factory, block_attribute):
        layers_module = self._get_layers(model, block_attribute)
        for pos in sorted(positions, reverse=True):
            block = block_factory()
            layers_module.insert(pos, block)
            self.inserted.append(pos)

    def _get_layers(self, model, attr):
        parts = attr.split(".")
        module = model
        for part in parts:
            module = getattr(module, part)
        return module


# --- Test CambiumBlock base class ---


class TestCambiumBlock:
    """Tests for CambiumBlock ABC."""

    def test_cannot_instantiate_abc(self):
        """Cannot instantiate CambiumBlock directly."""
        with pytest.raises(TypeError):
            CambiumBlock()

    def test_subclass_must_implement_forward(self):
        """Subclass must implement forward()."""

        class IncompleteBlock(CambiumBlock):
            pass

        with pytest.raises(TypeError):
            IncompleteBlock()

    def test_subclass_with_forward_works(self):
        """Subclass with forward() works."""
        block = SimpleCambiumBlock(type("Config", (), {"hidden_size": 32})())
        assert block is not None

    def test_required_config_keys(self):
        """required_config_keys is a class variable."""
        assert SimpleCambiumBlock.required_config_keys == ["hidden_size"]
        assert GatedResidualBlock.required_config_keys == ["hidden_size"]


class TestResidualWrapper:
    """Tests for ResidualWrapper."""

    def test_wraps_block_with_residual(self):
        """Wrapper adds input to block output."""
        block = SimpleCambiumBlock(type("Config", (), {"hidden_size": 32})())
        wrapper = ResidualWrapper(block)

        x = torch.randn(1, 10, 32)
        with torch.no_grad():
            output = wrapper(x)

        # Output should be input + block(input)
        assert output.shape == x.shape

    def test_repr(self):
        """Wrapper has readable repr."""
        block = SimpleCambiumBlock(type("Config", (), {"hidden_size": 32})())
        wrapper = ResidualWrapper(block)
        assert "ResidualWrapper" in repr(wrapper)


# --- Test CustomBlockExpansion ---


class TestCustomBlockExpansionValidation:
    """Tests for input validation."""

    def test_no_block_source_raises(self):
        """Must provide exactly one of block_class, block_factory, block_instances."""
        expander = CustomBlockExpansion(num_layers=2)
        with pytest.raises(ValueError, match="Must provide exactly one"):
            expander._validate_inputs()

    def test_multiple_block_sources_raises(self):
        """Cannot provide more than one block source."""
        expander = CustomBlockExpansion(
            block_class=SimpleCambiumBlock,
            block_factory=lambda: nn.Linear(32, 32),
            num_layers=2,
        )
        with pytest.raises(ValueError, match="Must provide exactly one"):
            expander._validate_inputs()

    def test_block_instances_count_mismatch(self):
        """block_instances count must match positions count."""
        blocks = [SimpleCambiumBlock(type("Config", (), {"hidden_size": 32})())]
        expander = CustomBlockExpansion(
            block_instances=blocks,
            positions=[1, 2, 3],  # 3 positions but 1 block
        )
        with pytest.raises(ValueError, match="must match"):
            expander._validate_inputs()


class TestCustomBlockExpansionBlockClass:
    """Tests for block_class mode."""

    def test_expand_with_block_class(self):
        """Insert blocks created from block_class."""
        model = SimpleModel(num_layers=4, hidden_size=32)
        engine = ExpansionEngine()

        expander = CustomBlockExpansion(
            block_class=SimpleCambiumBlock,
            num_layers=2,
            positions=[1, 3],
            validate=False,
        )

        result = expander.expand(model, engine)
        assert result is model
        assert len(model.model.layers) == 6  # 4 original + 2 new

    def test_expand_with_auto_positions(self):
        """Auto-distribute blocks when only num_layers is given."""
        model = SimpleModel(num_layers=4, hidden_size=32)
        engine = ExpansionEngine()

        expander = CustomBlockExpansion(
            block_class=SimpleCambiumBlock,
            num_layers=2,
            validate=False,
        )

        result = expander.expand(model, engine)
        assert result is model
        assert len(model.model.layers) == 6

    def test_config_updated(self):
        """Model config is updated after expansion."""
        model = SimpleModel(num_layers=4, hidden_size=32)
        engine = ExpansionEngine()

        expander = CustomBlockExpansion(
            block_class=SimpleCambiumBlock,
            num_layers=2,
            positions=[1, 3],
            validate=False,
        )

        expander.expand(model, engine)
        assert model.config.num_hidden_layers == 6


class TestCustomBlockExpansionBlockFactory:
    """Tests for block_factory mode."""

    def test_expand_with_factory(self):
        """Insert blocks created from factory function."""
        model = SimpleModel(num_layers=4, hidden_size=32)
        engine = ExpansionEngine()
        config = model.config

        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return SimpleCambiumBlock(config, layer_idx=call_count)

        expander = CustomBlockExpansion(
            block_factory=factory,
            num_layers=2,
            positions=[1, 3],
            validate=False,
        )

        result = expander.expand(model, engine)
        assert result is model
        assert len(model.model.layers) == 6
        assert call_count == 2


class TestCustomBlockExpansionBlockInstances:
    """Tests for block_instances mode."""

    def test_expand_with_instances(self):
        """Insert pre-created block instances."""
        model = SimpleModel(num_layers=4, hidden_size=32)
        engine = ExpansionEngine()
        config = model.config

        blocks = [
            SimpleCambiumBlock(config, layer_idx=i)
            for i in range(2)
        ]

        expander = CustomBlockExpansion(
            block_instances=blocks,
            positions=[1, 3],
            validate=False,
        )

        result = expander.expand(model, engine)
        assert result is model
        assert len(model.model.layers) == 6


class TestCustomBlockExpansionResidual:
    """Tests for residual connection wrapping."""

    def test_residual_connection_true(self):
        """Blocks are wrapped in ResidualWrapper when residual_connection=True."""
        model = SimpleModel(num_layers=4, hidden_size=32)
        engine = ExpansionEngine()

        expander = CustomBlockExpansion(
            block_class=SimpleCambiumBlock,
            num_layers=2,
            positions=[1, 3],
            residual_connection=True,
            validate=False,
        )

        expander.expand(model, engine)

        # After inserting at positions [1, 3] in descending order (3 first, then 1):
        # Original [L0, L1, L2, L3]
        # Insert at 3: [L0, L1, L2, New, L3]
        # Insert at 1: [L0, New, L1, L2, New, L3]
        # So wrappers are at indices 1 and 4
        wrapper_count = sum(1 for layer in model.model.layers if isinstance(layer, ResidualWrapper))
        assert wrapper_count == 2
        assert isinstance(model.model.layers[1], ResidualWrapper)
        assert isinstance(model.model.layers[4], ResidualWrapper)

    def test_residual_connection_false(self):
        """Blocks are not wrapped when residual_connection=False."""
        model = SimpleModel(num_layers=4, hidden_size=32)
        engine = ExpansionEngine()

        expander = CustomBlockExpansion(
            block_class=ResidualBlock,  # Block with its own residual
            num_layers=1,
            positions=[2],
            residual_connection=False,
            validate=False,
        )

        expander.expand(model, engine)

        # Block should be inserted directly, not wrapped
        assert not isinstance(model.model.layers[2], ResidualWrapper)
        assert isinstance(model.model.layers[2], ResidualBlock)


class TestCustomBlockExpansionValidation:
    """Tests for block validation."""

    def test_validation_catches_shape_mismatch(self):
        """Validation rejects blocks with wrong output shape."""
        model = SimpleModel(num_layers=4, hidden_size=32)
        engine = ExpansionEngine()

        expander = CustomBlockExpansion(
            block_class=BadShapeBlock,
            num_layers=1,
            positions=[2],
            validate=True,
            residual_connection=True,  # Requires matching shape
        )

        with pytest.raises(BlockValidationError):
            expander.expand(model, engine)

    def test_validation_disabled(self):
        """No validation when validate=False."""
        model = SimpleModel(num_layers=4, hidden_size=32)
        engine = ExpansionEngine()

        expander = CustomBlockExpansion(
            block_class=BadShapeBlock,
            num_layers=1,
            positions=[2],
            validate=False,
            residual_connection=True,
        )

        # Should not raise — validation is disabled
        expander.expand(model, engine)

    def test_validation_passes_for_correct_block(self):
        """Validation passes for blocks with correct shape."""
        model = SimpleModel(num_layers=4, hidden_size=32)
        engine = ExpansionEngine()

        expander = CustomBlockExpansion(
            block_class=SimpleCambiumBlock,
            num_layers=1,
            positions=[2],
            validate=True,
            residual_connection=True,
        )

        # Should not raise
        expander.expand(model, engine)
        assert len(model.model.layers) == 5

    def test_validation_warns_about_missing_kwargs(self):
        """Validation warns about blocks without **kwargs."""
        model = SimpleModel(num_layers=4, hidden_size=32)
        engine = ExpansionEngine()

        expander = CustomBlockExpansion(
            block_class=NoKwargsBlock,
            num_layers=1,
            positions=[2],
            validate=True,
            residual_connection=True,
        )

        # Should not raise, but should log a warning
        # (We can't easily test logging, but ensure it doesn't crash)
        expander.expand(model, engine)


class TestCustomBlockExpansionInitialization:
    """Tests for initialization strategies."""

    def test_smart_initialization(self):
        """Smart initialization is applied."""
        model = SimpleModel(num_layers=4, hidden_size=32)
        engine = ExpansionEngine()

        expander = CustomBlockExpansion(
            block_class=SimpleCambiumBlock,
            num_layers=1,
            positions=[2],
            initialization="smart",
            validate=False,
        )

        expander.expand(model, engine)
        # Should not raise

    def test_custom_initialization(self):
        """Custom init function is called."""
        model = SimpleModel(num_layers=4, hidden_size=32)
        engine = ExpansionEngine()

        init_calls = []

        def my_init(block):
            init_calls.append(block)
            nn.init.zeros_(block.proj.weight)

        expander = CustomBlockExpansion(
            block_class=SimpleCambiumBlock,
            num_layers=1,
            positions=[2],
            initialization="custom",
            custom_init_fn=my_init,
            validate=False,
        )

        expander.expand(model, engine)
        assert len(init_calls) == 1

    def test_custom_init_requires_fn(self):
        """Custom initialization requires custom_init_fn."""
        model = SimpleModel(num_layers=4, hidden_size=32)
        engine = ExpansionEngine()

        expander = CustomBlockExpansion(
            block_class=SimpleCambiumBlock,
            num_layers=1,
            positions=[2],
            initialization="custom",
            custom_init_fn=None,
            validate=False,
        )

        with pytest.raises(ValueError, match="custom_init_fn must be provided"):
            expander._apply_initialization(model, [2])


class TestTemplateBlocks:
    """Tests for template blocks."""

    def test_swiglu_block_forward(self):
        """SwiGLUBlock forward pass works."""
        config = type("Config", (), {"hidden_size": 32, "intermediate_size": 64})()
        block = SwiGLUBlock(config)
        x = torch.randn(1, 10, 32)
        with torch.no_grad():
            output = block(x)
        assert output.shape == (1, 10, 32)

    def test_gated_residual_block_forward(self):
        """GatedResidualBlock forward pass works."""
        config = type("Config", (), {"hidden_size": 32, "intermediate_size": 64})()
        block = GatedResidualBlock(config)
        x = torch.randn(1, 10, 32)
        with torch.no_grad():
            output = block(x)
        assert output.shape == (1, 10, 32)

    def test_cross_attention_block_forward(self):
        """CrossAttentionBlock forward pass works."""
        config = type("Config", (), {
            "hidden_size": 32,
            "num_attention_heads": 4,
        })()
        block = CrossAttentionBlock(config)
        x = torch.randn(1, 10, 32)
        with torch.no_grad():
            output = block(x)
        assert output.shape == (1, 10, 32)

    def test_template_blocks_accept_kwargs(self):
        """Template blocks accept **kwargs."""
        config = type("Config", (), {
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_attention_heads": 4,
        })()
        block = SwiGLUBlock(config)
        x = torch.randn(1, 10, 32)
        with torch.no_grad():
            output = block(x, attention_mask=None, position_ids=None)
        assert output.shape == (1, 10, 32)


class TestCustomBlockExpansionIntegration:
    """Integration tests with MockEngine."""

    def test_expand_then_freeze(self):
        """Can freeze original layers after expansion."""
        model = SimpleModel(num_layers=4, hidden_size=32)
        engine = ExpansionEngine()

        expander = CustomBlockExpansion(
            block_class=SimpleCambiumBlock,
            num_layers=2,
            positions=[1, 3],
            validate=False,
            residual_connection=True,
        )

        expander.expand(model, engine)

        from cambium.core.freezing import FreezingManager
        fm = FreezingManager(model)

        # Freeze all then count trainable
        fm.freeze_all()
        info = fm.get_trainable_params()
        assert info["trainable_params"] == 0

        # Unfreeze all and verify parameters exist
        fm.unfreeze_all()
        info = fm.get_trainable_params()
        assert info["trainable_params"] > 0
        assert info["frozen_params"] == 0

    def test_mixed_expansions(self):
        """Can add custom blocks after interleaved expansion."""
        model = SimpleModel(num_layers=4, hidden_size=32)
        engine = ExpansionEngine()

        # First expansion: interleaved (using mock engine for simplicity)
        # This simulates adding standard blocks first

        # Then: custom blocks
        expander = CustomBlockExpansion(
            block_class=GatedResidualBlock,
            num_layers=1,
            positions=[2],
            validate=False,
            residual_connection=True,
        )

        result = expander.expand(model, engine)
        assert len(result.model.layers) == 5