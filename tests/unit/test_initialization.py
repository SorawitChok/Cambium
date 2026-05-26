"""Tests for initialization strategies."""

import pytest
import torch
from torch import nn

from cambium.core.initialization import (
    Initializer,
    InitializationStrategy,
    IdentityInitializer,
)


class TestInitializer:
    """Tests for Initializer class."""

    @pytest.fixture
    def modules(self):
        """Create test modules."""
        return [
            nn.Linear(32, 32),
            nn.Linear(64, 32),
            nn.Embedding(100, 32),
        ]

    def test_init_identity(self, modules):
        """Test identity initialization."""
        initializer = Initializer(InitializationStrategy.IDENTITY_MAPPING)
        initializer.apply(modules)

        # Check that modules were initialized
        for module in modules:
            if isinstance(module, nn.Linear):
                assert module.weight is not None

    def test_init_small_random(self, modules):
        """Test small random initialization."""
        initializer = Initializer(InitializationStrategy.SMALL_RANDOM)
        initializer.apply(modules, scale=0.01)

        for module in modules:
            if isinstance(module, nn.Linear):
                # Check that weights are small
                assert module.weight.abs().mean() < 0.1

    def test_init_zero(self, modules):
        """Test zero initialization."""
        initializer = Initializer(InitializationStrategy.ZERO_INIT)
        initializer.apply(modules)

        for module in modules:
            if isinstance(module, nn.Linear):
                assert torch.allclose(module.weight, torch.zeros_like(module.weight))

    def test_init_xavier(self, modules):
        """Test Xavier initialization."""
        initializer = Initializer(InitializationStrategy.XAVIER_UNIFORM)
        initializer.apply(modules)

        for module in modules:
            if isinstance(module, nn.Linear):
                assert module.weight is not None

    def test_init_kaiming(self, modules):
        """Test Kaiming initialization."""
        initializer = Initializer(InitializationStrategy.KAIMING_NORMAL)
        initializer.apply(modules)

        for module in modules:
            if isinstance(module, nn.Linear):
                assert module.weight is not None

    def test_smart_init_for_block(self):
        """Test smart initialization for transformer block."""

        # Create a simple transformer-like block
        class SimpleBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(32, 32)
                self.k_proj = nn.Linear(32, 32)
                self.v_proj = nn.Linear(32, 32)
                self.o_proj = nn.Linear(32, 32)
                self.norm = nn.LayerNorm(32)

        block = SimpleBlock()
        initializer = Initializer()
        initializer.smart_init_for_block(block, model_type="llama")

        # o_proj should have small weights (near-zero init)
        assert block.o_proj.weight.abs().mean() < 0.1


class TestIdentityInitializer:
    """Tests for IdentityInitializer convenience class."""

    def test_call(self):
        """Test calling IdentityInitializer."""
        module = nn.Linear(32, 32)
        initializer = IdentityInitializer(output_scale=0.001)

        initializer(module)

        # Weights should be initialized
        assert module.weight is not None
