"""Tests for expansion strategies."""

import pytest
import torch
from torch import nn

from cambium.strategies.block_expansion import InterleavedExpansion


def _make_config(**overrides):
    """Create a minimal config compatible with HF LlamaDecoderLayer."""
    from transformers import LlamaConfig

    defaults = dict(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    defaults.update(overrides)
    return LlamaConfig(**defaults)


class MockModel:
    """Mock model for testing strategies."""

    def __init__(self, num_layers=4, hidden_size=32):
        self.config = _make_config(
            num_hidden_layers=num_layers,
            hidden_size=hidden_size,
        )
        self.model = type("Model", (), {})()
        self.model.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
        )


class MockEngine:
    """Mock expansion engine for testing."""

    def __init__(self):
        self.inserted_positions = []
        self.expansion_history = []

    def insert_blocks(self, model, positions, block_factory, block_attribute):
        for pos in sorted(positions, reverse=True):
            new_block = block_factory()
            model.model.layers.insert(pos, new_block)
            self.inserted_positions.append(pos)

        self.expansion_history.append(
            {
                "operation": "insert_blocks",
                "positions": positions,
                "original_length": len(model.model.layers) - len(positions),
                "new_length": len(model.model.layers),
            }
        )


class TestInterleavedExpansion:
    """Tests for InterleavedExpansion strategy."""

    def test_compute_positions_auto(self):
        """Test auto-computing positions."""
        expander = InterleavedExpansion(num_layers=4)

        positions = expander._compute_positions(12)

        assert len(positions) == 4
        # Should be evenly distributed
        assert all(0 <= p <= 12 for p in positions)

    def test_compute_positions_manual(self):
        """Test manual position specification."""
        expander = InterleavedExpansion(
            num_layers=2,
            positions=[2, 5],
        )

        positions = expander._compute_positions(8)

        assert positions == [2, 5]

    def test_compute_positions_invalid(self):
        """Test invalid position specification."""
        expander = InterleavedExpansion(
            num_layers=1,
            positions=[100],  # Invalid position
        )

        with pytest.raises(ValueError, match="Invalid position"):
            expander._compute_positions(8)

    def test_expand(self):
        """Test full expansion."""
        model = MockModel(num_layers=4)
        engine = MockEngine()

        expander = InterleavedExpansion(
            num_layers=2,
            positions=[1, 3],
            initialization="identity",
        )

        result = expander.expand(model, engine)

        # Should have 6 layers now
        assert len(model.model.layers) == 6
        assert len(engine.inserted_positions) == 2

    def test_update_config(self):
        """Test config update after expansion."""
        model = MockModel(num_layers=4)
        engine = MockEngine()

        expander = InterleavedExpansion(num_layers=2)
        expander._update_config(model, positions=[1, 3])

        assert model.config.num_hidden_layers == 6


class TestWidthExpansion:
    """Tests for WidthExpansion strategy (placeholder)."""

    def test_placeholder(self):
        """Placeholder test."""
        # Full tests would require actual transformer models
        pass
