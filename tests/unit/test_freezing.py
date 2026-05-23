"""Tests for freezing manager."""

import pytest
import torch
from torch import nn

from cambium.core.freezing import FreezingManager


class SimpleModel(nn.Module):
    """Simple model for testing freezing."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 32)
        self.layers = nn.ModuleList([
            nn.Linear(32, 32) for _ in range(4)
        ])
        self.new_layers = nn.ModuleList([
            nn.Linear(32, 32) for _ in range(2)
        ])
        self.lm_head = nn.Linear(32, 100)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        for layer in self.new_layers:
            x = layer(x)
        return self.lm_head(x)


class TestFreezingManager:
    """Tests for FreezingManager."""

    @pytest.fixture
    def model(self):
        return SimpleModel()

    @pytest.fixture
    def manager(self, model):
        return FreezingManager(model)

    def test_initialization(self, model, manager):
        """Test manager initialization."""
        assert manager.model is model
        assert len(manager.original_requires_grad) > 0

    def test_freeze_all(self, model, manager):
        """Test freezing all parameters."""
        manager.freeze_all()

        for param in model.parameters():
            assert not param.requires_grad

    def test_unfreeze_all(self, model, manager):
        """Test unfreezing all parameters."""
        manager.freeze_all()
        manager.unfreeze_all()

        for param in model.parameters():
            assert param.requires_grad

    def test_freeze_by_pattern(self, model, manager):
        """Test freezing by pattern."""
        frozen = manager.freeze_by_pattern(r"layers\.[0-1]")

        # Check that layers 0 and 1 are frozen
        for name, param in model.named_parameters():
            if "layers.0" in name or "layers.1" in name:
                assert not param.requires_grad

        assert len(frozen) > 0

    def test_unfreeze_by_pattern(self, model, manager):
        """Test unfreezing by pattern."""
        manager.freeze_all()
        unfrozen = manager.unfreeze_by_pattern(r"new_layers")

        # Check that new_layers are unfrozen
        for name, param in model.named_parameters():
            if "new_layers" in name:
                assert param.requires_grad

        assert len(unfrozen) > 0

    def test_get_trainable_params(self, model, manager):
        """Test getting trainable parameters info."""
        manager.freeze_all()
        manager.unfreeze_by_pattern(r"new_layers")

        info = manager.get_trainable_params()

        assert "trainable_params" in info
        assert "frozen_params" in info
        assert info["trainable_params"] > 0
        assert info["frozen_params"] > 0
        assert 0 < info["percent_trainable"] < 100

    def test_get_parameter_groups_for_discriminative_lr(self, model, manager):
        """Test getting parameter groups for discriminative LR."""
        lr_config = {
            r"embed": 1e-5,
            r"layers": 1e-4,
            r"new_layers": 1e-3,
        }

        groups = manager.get_parameter_groups_for_discriminative_lr(lr_config)

        assert len(groups) > 0
        for group in groups:
            assert "params" in group
            assert "lr" in group
            assert len(group["params"]) > 0

    def test_save_load_state(self, model, manager, tmp_path):
        """Test saving and loading freezing state."""
        # Freeze some layers
        manager.freeze_all()
        manager.unfreeze_by_pattern(r"new_layers")

        # Save state
        state_path = tmp_path / "freeze_state.pt"
        manager.save_state(str(state_path))

        # Unfreeze all
        manager.unfreeze_all()

        # Load state
        manager.load_state(str(state_path))

        # Verify state restored
        for name, param in model.named_parameters():
            if "new_layers" in name:
                assert param.requires_grad
            else:
                assert not param.requires_grad

    def test_freeze_original_layers(self, model, manager):
        """Test freezing original layers while keeping new layers trainable."""
        # Tag new layers with _cambium_new as the engine would
        for layer in model.new_layers:
            layer._cambium_new = True

        manager.freeze_original_layers()

        for name, param in model.named_parameters():
            if "new_layers" in name:
                assert param.requires_grad
            else:
                assert not param.requires_grad
