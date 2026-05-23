# Parallel Adapter Expansion

Parallel adapters add additional processing pathways alongside existing transformer blocks, similar to LoRA or MoE-lite architectures.

## The Concept

```
Original:  Input → [Block] → Output

With Adapter:  Input → [Block] ─┬→ Output
                               │
                               └→ [Adapter] ─┘ (added)
```

## Basic Adapter Expansion

```python
from cambium import ExpandableModel
from cambium.strategies import ParallelAdapterExpansion

# Load base model
model = ExpandableModel.from_pretrained("google/gemma-2b")

# Add bottleneck adapters to all layers
expander = ParallelAdapterExpansion(
    adapter_type="bottleneck",
    bottleneck_dim=256,  # Hidden dim of the bottleneck
    initialization="zero",
)

model.expand(expander)
```

## Targeting Specific Layers

```python
from cambium.strategies import ParallelAdapterExpansion

# Add adapters only to later layers (often more effective)
expander = ParallelAdapterExpansion(
    adapter_type="bottleneck",
    bottleneck_dim=256,
    target_layers=[16, 17, 18, 19, 20, 21, 22, 23],  # Last 8 layers only
)

model.expand(expander)
```

## Attention Adapters

```python
from cambium.strategies import ParallelAdapterExpansion

# Add parallel cross-attention adapters
expander = ParallelAdapterExpansion(
    adapter_type="attention",
    num_heads=4,  # Fewer heads for efficiency
    initialization="zero",
)

model.expand(expander)
```

## Adapter Architecture Details

### Bottleneck Adapter

```python
# The bottleneck adapter:
# 1. Down-project: hidden_dim -> bottleneck_dim
# 2. Apply GELU activation
# 3. Up-project: bottleneck_dim -> hidden_dim
# 4. Gate: Multiply by sigmoid-learned gate

from cambium.strategies.parallel_adapters import ParallelBottleneckAdapter
import torch

adapter = ParallelBottleneckAdapter(
    hidden_dim=2048,
    bottleneck_dim=256,
)

# Test forward
test_input = torch.randn(1, 10, 2048)  # batch, seq, hidden
output = adapter(test_input)
print(f"Output shape: {output.shape}")  # [1, 10, 2048]
```

### Attention Adapter

```python
from cambium.strategies.parallel_adapters import ParallelAttentionAdapter

adapter = ParallelAttentionAdapter(
    hidden_dim=2048,
    num_heads=4,
)

# Test forward
test_input = torch.randn(1, 10, 2048)
output = adapter(test_input)
print(f"Output shape: {output.shape}")  # [1, 10, 2048]
```

## Training Adapters

Adapters are designed to be trained with the base model frozen:

```python
from cambium import ExpandableModel
from cambium.strategies import ParallelAdapterExpansion
from cambium.training import StagedTrainer

model = ExpandableModel.from_pretrained("google/gemma-2b")

# Add adapters
model.expand(ParallelAdapterExpansion(
    adapter_type="bottleneck",
    bottleneck_dim=256,
))

# Freeze everything except adapters
fm = model.freezing_manager
fm.freeze_all()
fm.unfreeze_by_pattern(r"cambium_adapters")

# Train only adapters
trainer = StagedTrainer(model)
trainer.add_phase(
    name="adapter_training",
    freeze=None,  # Keep current freeze state
    lr=1e-3,      # Higher LR since we're training small adapter params
    epochs=3,
)

history = trainer.train(train_dataloader)
```

## When to Use Parallel Adapters

| Scenario | Recommendation |
|----------|---------------|
| Limited compute | Adapters (fewer new params) |
| Quick experiments | Adapters (faster training) |
| Domain adaptation | Adapters (effective for transfer) |
| Maximum capacity | Block expansion |
| Architectural changes | Block expansion |

## Adapter vs LoRA

| Aspect | Cambium Adapters | LoRA |
|--------|-----------------|------|
| Architecture | Bottleneck/Attention | Low-rank decomposition |
| Parameters | ~0.5-1% of model | ~0.1-0.5% of model |
| Training | Adapter layers only | Selected linear layers |
| Inference | Slightly slower | Same speed as base |
| Cambium | Native integration | Via PEFT integration |

## Combining with Other Expansions

```python
from cambium import ExpandableModel, InterleavedExpansion
from cambium.strategies import ParallelAdapterExpansion

model = ExpandableModel.from_pretrained("google/gemma-2b")

# Add new blocks
model.expand(InterleavedExpansion(num_layers=4))

# Add adapters to new blocks only
# (Note: This would require custom logic to target specific blocks)
```

## Tips for Adapter Expansion

1. **Use bottleneck_dim of 64-512** (typically 1/8 to 1/4 of hidden_dim)
2. **Target later layers** for better performance
3. **Use zero initialization** for stable training
4. **Higher learning rates** work well (1e-3 to 1e-4)
5. **Fewer epochs** needed compared to block expansion
