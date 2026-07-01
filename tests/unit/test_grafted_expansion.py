"""Tests for grafted block expansion."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from cambium.core.expansion import ExpansionEngine
from cambium.exceptions import BlockValidationError, GraftingError
from cambium.strategies.grafted_expansion import GraftedBlockExpansion, _GraftedBlockWrapper


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, num_layers=4, hidden_size=32):
        super().__init__()
        self.config = type(
            "Config",
            (),
            {
                "hidden_size": hidden_size,
                "num_hidden_layers": num_layers,
                "num_attention_heads": 4,
                "intermediate_size": hidden_size * 4,
                "model_type": "test",
                "layer_types": ["full_attention"] * num_layers,
                "to_dict": lambda self: {"hidden_size": hidden_size},
            },
        )()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
        )
        self.lm_head = nn.Linear(hidden_size, 100)


class MockDecoderLayer(nn.Module):
    """Source-like decoder layer returned by mocked loader."""

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, **kwargs):
        return self.proj(hidden_states)


class MockDecoderLayerWithTuple(nn.Module):
    """Source-like decoder layer that returns a tuple like HF layers."""

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, **kwargs):
        return (self.proj(hidden_states), None)


class TestGraftedBlockWrapper:
    """Tests for the projection/tuple wrapper."""

    def test_no_projection_when_sizes_match(self):
        """Wrapper is transparent when hidden sizes match."""
        source = MockDecoderLayer(32)
        wrapped = _GraftedBlockWrapper(source, 32, 32)

        assert wrapped.input_proj is None
        assert wrapped.output_proj is None

        x = torch.randn(1, 2, 32)
        with torch.no_grad():
            out = wrapped(x)
        assert out.shape == x.shape

    def test_projection_when_sizes_differ(self):
        """Wrapper adds linear projections when hidden sizes differ."""
        source = MockDecoderLayer(16)
        wrapped = _GraftedBlockWrapper(source, 16, 32)

        assert wrapped.input_proj is not None
        assert wrapped.output_proj is not None

        x = torch.randn(1, 2, 32)
        with torch.no_grad():
            out = wrapped(x)
        assert out.shape == x.shape

    def test_tuple_return_unwrapped(self):
        """Wrapper extracts the first element from tuple returns."""
        source = MockDecoderLayerWithTuple(32)
        wrapped = _GraftedBlockWrapper(source, 32, 32)

        x = torch.randn(1, 2, 32)
        with torch.no_grad():
            out = wrapped(x)
        assert out.shape == x.shape


class TestGraftedBlockExpansionValidation:
    """Tests for input validation."""

    def test_no_source_identifier_raises(self):
        """Must provide source_block_idx or source_block_name."""
        model = SimpleModel()
        engine = ExpansionEngine()
        expander = GraftedBlockExpansion(
            source_model_id="foo/bar",
            positions=[1],
        )
        with pytest.raises(GraftingError, match="source_block_idx or source_block_name"):
            expander.expand(model, engine)

    def test_both_source_identifiers_raises(self):
        """Cannot provide both source_block_idx and source_block_name."""
        model = SimpleModel()
        engine = ExpansionEngine()
        expander = GraftedBlockExpansion(
            source_model_id="foo/bar",
            source_block_idx=0,
            source_block_name="model.layers.0",
            positions=[1],
        )
        with pytest.raises(GraftingError, match="only one of"):
            expander.expand(model, engine)

    def test_missing_positions_raises(self):
        """Must provide target positions."""
        model = SimpleModel()
        engine = ExpansionEngine()
        expander = GraftedBlockExpansion(
            source_model_id="foo/bar",
            source_block_idx=0,
            positions=None,
        )
        with pytest.raises(GraftingError, match="target position"):
            expander.expand(model, engine)

    def test_invalid_target_position_raises(self):
        """Target position must be within bounds."""
        model = SimpleModel(num_layers=4)
        engine = ExpansionEngine()
        expander = GraftedBlockExpansion(
            source_model_id="foo/bar",
            source_block_idx=0,
            positions=[10],
        )
        with pytest.raises(GraftingError, match="Invalid target position"):
            expander.expand(model, engine)


class TestGraftedBlockExpansion:
    """Tests for successful grafting."""

    def test_graft_same_hidden_size(self):
        """Graft a block with matching hidden size."""
        model = SimpleModel(num_layers=4, hidden_size=32)
        engine = ExpansionEngine()
        source_block = MockDecoderLayer(32)
        # Use a deterministic weight value so we can verify it is preserved.
        with torch.no_grad():
            source_block.proj.weight.fill_(0.5)

        expander = GraftedBlockExpansion(
            source_model_id="foo/bar",
            source_block_idx=2,
            positions=[2],
            validate=False,
        )

        with patch(
            "cambium.strategies.grafted_expansion.load_grafted_block",
            return_value=source_block,
        ):
            expander.expand(model, engine)

        assert len(model.model.layers) == 5
        assert model.config.num_hidden_layers == 5
        assert len(model.config.layer_types) == 5

    def test_graft_different_hidden_size_adds_projection(self):
        """Graft a block with mismatched hidden size creates projection layers."""
        model = SimpleModel(num_layers=4, hidden_size=32)
        engine = ExpansionEngine()
        source_block = MockDecoderLayer(16)

        expander = GraftedBlockExpansion(
            source_model_id="foo/bar",
            source_block_idx=2,
            positions=[2],
            projection=True,
            validate=False,
        )

        with patch(
            "cambium.strategies.grafted_expansion.load_grafted_block",
            return_value=source_block,
        ):
            expander.expand(model, engine)

        assert len(model.model.layers) == 5
        grafted = model.model.layers[2]
        assert hasattr(grafted, "block")
        assert hasattr(grafted.block, "input_proj")
        assert hasattr(grafted.block, "output_proj")

    def test_graft_different_hidden_size_without_projection_raises(self):
        """Mismatched hidden sizes require projection=True."""
        model = SimpleModel(num_layers=4, hidden_size=32)
        engine = ExpansionEngine()
        source_block = MockDecoderLayer(16)

        expander = GraftedBlockExpansion(
            source_model_id="foo/bar",
            source_block_idx=2,
            positions=[2],
            projection=False,
            validate=False,
        )

        with patch(
            "cambium.strategies.grafted_expansion.load_grafted_block",
            return_value=source_block,
        ):
            with pytest.raises(GraftingError, match="projection"):
                expander.expand(model, engine)

    def test_graft_validation_catches_bad_shape(self):
        """Validation rejects a block that changes the hidden shape."""
        model = SimpleModel(num_layers=4, hidden_size=32)
        engine = ExpansionEngine()

        class BadBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden_size = 32
                self.proj = nn.Linear(32, 64)

            def forward(self, hidden_states, **kwargs):
                return self.proj(hidden_states)

        expander = GraftedBlockExpansion(
            source_model_id="foo/bar",
            source_block_idx=2,
            positions=[2],
            validate=True,
        )

        with patch(
            "cambium.strategies.grafted_expansion.load_grafted_block",
            return_value=BadBlock(),
        ):
            with pytest.raises(BlockValidationError):
                expander.expand(model, engine)

    def test_freeze_option(self):
        """freeze=True makes grafted parameters non-trainable."""
        model = SimpleModel(num_layers=4, hidden_size=32)
        engine = ExpansionEngine()
        source_block = MockDecoderLayer(32)

        expander = GraftedBlockExpansion(
            source_model_id="foo/bar",
            source_block_idx=2,
            positions=[2],
            freeze=True,
            validate=False,
        )

        with patch(
            "cambium.strategies.grafted_expansion.load_grafted_block",
            return_value=source_block,
        ):
            expander.expand(model, engine)

        grafted = model.model.layers[2]
        for param in grafted.parameters(recurse=True):
            assert not param.requires_grad


class TestGraftingLoader:
    """Tests for the lightweight weight loader."""

    def test_resolve_prefix_from_index(self):
        """resolve_block_prefix builds the correct prefix from an index."""
        from cambium.core.grafting import resolve_block_prefix

        prefix = resolve_block_prefix("model.layers", source_block_idx=3, source_block_name=None)
        assert prefix == "model.layers.3."

    def test_resolve_prefix_from_name(self):
        """resolve_block_prefix uses the provided name."""
        from cambium.core.grafting import resolve_block_prefix

        prefix = resolve_block_prefix(
            "model.layers", source_block_idx=None, source_block_name="model.layers.5"
        )
        assert prefix == "model.layers.5."

    def test_strip_prefix(self):
        """strip_prefix removes the block prefix from state-dict keys."""
        from cambium.core.grafting import strip_prefix

        state_dict = {
            "model.layers.5.self_attn.q_proj.weight": torch.randn(4, 4),
            "model.layers.5.mlp.gate_proj.weight": torch.randn(4, 4),
        }
        stripped = strip_prefix(state_dict, "model.layers.5.")
        assert set(stripped.keys()) == {"self_attn.q_proj.weight", "mlp.gate_proj.weight"}

    def test_strip_prefix_from_name_with_trailing_dot(self):
        """resolve_block_prefix normalizes a trailing dot."""
        from cambium.core.grafting import resolve_block_prefix

        prefix = resolve_block_prefix(
            "model.layers", source_block_idx=None, source_block_name="model.layers.2."
        )
        assert prefix == "model.layers.2."
