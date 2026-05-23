"""Tests for expansion engine."""

import pytest
import torch
from torch import nn

from cambium.core.expansion import ExpansionEngine


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, num_layers=4, hidden_size=32):
        super().__init__()
        self.config = type("Config", (), {
            "hidden_size": hidden_size,
            "num_hidden_layers": num_layers,
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


class TestExpansionEngine:
    """Tests for ExpansionEngine."""

    def test_initialization(self):
        """Test engine initialization."""
        engine = ExpansionEngine()
        assert engine.model_type == "auto"
        assert engine.expansion_history == []

    def test_insert_blocks(self):
        """Test block insertion."""
        model = SimpleModel(num_layers=4)
        engine = ExpansionEngine()

        def block_factory():
            return nn.Linear(32, 32)

        # Insert 2 blocks at positions 1 and 3
        engine.insert_blocks(
            model,
            positions=[1, 3],
            block_factory=block_factory,
            block_attribute="model.layers",
        )

        # Should now have 6 layers
        assert len(model.model.layers) == 6

        # Check history
        assert len(engine.expansion_history) == 1
        assert engine.expansion_history[0]["operation"] == "insert_blocks"
        assert engine.expansion_history[0]["original_length"] == 4
        assert engine.expansion_history[0]["new_length"] == 6

    def test_insert_blocks_invalid_position(self):
        """Test block insertion with invalid position."""
        model = SimpleModel(num_layers=4)
        engine = ExpansionEngine()

        def block_factory():
            return nn.Linear(32, 32)

        with pytest.raises(ValueError, match="Invalid position"):
            engine.insert_blocks(
                model,
                positions=[10],  # Invalid position
                block_factory=block_factory,
                block_attribute="model.layers",
            )

    def test_insert_blocks_invalid_attribute(self):
        """Test block insertion with invalid attribute path."""
        model = SimpleModel(num_layers=4)
        engine = ExpansionEngine()

        def block_factory():
            return nn.Linear(32, 32)

        with pytest.raises(ValueError, match="Could not find"):
            engine.insert_blocks(
                model,
                positions=[1],
                block_factory=block_factory,
                block_attribute="invalid.path",
            )

    def test_validate_expansion(self):
        """Test expansion validation."""
        model = SimpleModel(num_layers=4)
        engine = ExpansionEngine()

        results = engine.validate_expansion(model)

        assert results["valid"] is True
        assert "checks" in results
        assert "parameters" in results["checks"]
        assert results["checks"]["numerical_stability"]["has_nan"] is False
        assert results["checks"]["numerical_stability"]["has_inf"] is False

    def test_get_expansion_report(self):
        """Test expansion report generation."""
        model = SimpleModel(num_layers=4)
        engine = ExpansionEngine()

        def block_factory():
            return nn.Linear(32, 32)

        engine.insert_blocks(
            model,
            positions=[1],
            block_factory=block_factory,
            block_attribute="model.layers",
        )

        report = engine.get_expansion_report()
        assert "Expansion Report" in report
        assert "insert_blocks" in report


class TestWidthExpansion:
    """Tests for width expansion functionality."""

    def test_expand_linear_output(self):
        """Test expanding linear layer output dimension."""
        engine = ExpansionEngine()
        linear = nn.Linear(32, 32)

        engine._expand_linear(linear, 32, 48, axis=0, initialization="copy")

        assert linear.out_features == 48
        assert linear.weight.shape == (48, 32)

    def test_expand_linear_input(self):
        """Test expanding linear layer input dimension."""
        engine = ExpansionEngine()
        linear = nn.Linear(32, 32)

        engine._expand_linear(linear, 32, 48, axis=1, initialization="copy")

        assert linear.in_features == 48
        assert linear.weight.shape == (32, 48)

    def test_expand_linear_with_bias(self):
        """Test expanding linear layer with bias."""
        engine = ExpansionEngine()
        linear = nn.Linear(32, 32, bias=True)

        engine._expand_linear(linear, 32, 48, axis=0, initialization="copy")

        assert linear.bias.shape == (48,)
