# Width Expansion

Width expansion increases the hidden dimensions of the model (e.g., 768 → 1152), providing more capacity through wider representations.

## The Concept

Unlike block expansion which adds more layers, width expansion makes existing layers wider:

```
Original:  hidden_size = 768
Expanded:  hidden_size = 1152 (768 * 1.5)

This affects:
- Attention projections
- MLP intermediate dimensions
- Embeddings
- LayerNorm weights
```

## Basic Width Expansion

```python
from cambium import ExpandableModel
from cambium.strategies import WidthExpansion

# Load base model
model = ExpandableModel.from_pretrained("google/gemma-2b")

# Expand width by 1.5x
expander = WidthExpansion(
    hidden_dim_multiplier=1.5,  # 2048 -> 3072
    initialization="copy",       # Copy existing weights
)

model.expand(expander)

print(f"New hidden size: {model.get_model().config.hidden_size}")
```

## Initialization Strategies

```python
from cambium.strategies import WidthExpansion

# Copy existing weights to new dimensions
expander = WidthExpansion(
    hidden_dim_multiplier=1.5,
    initialization="copy",
)

# Zero initialization for new dimensions
expander = WidthExpansion(
    hidden_dim_multiplier=1.5,
    initialization="zero",
)

# Noise injection for new dimensions
expander = WidthExpansion(
    hidden_dim_multiplier=1.5,
    initialization="noise",
)
```

## Combined Expansion (Blocks + Width)

```python
from cambium import ExpandableModel, InterleavedExpansion
from cambium.strategies import WidthExpansion

model = ExpandableModel.from_pretrained("google/gemma-2b")

# First expand width
model.expand(WidthExpansion(hidden_dim_multiplier=1.25))

# Then add blocks
model.expand(InterleavedExpansion(num_layers=4))

print(f"Model now has {len(model.get_model().model.layers)} layers")
print(f"Hidden size: {model.get_model().config.hidden_size}")
```

## When to Use Width Expansion

| Scenario | Recommendation |
|----------|---------------|
| Need more representation capacity | Width expansion |
| Want to preserve layer depth | Width expansion |
| Memory constrained | Width expansion (fewer activations than blocks) |
| Need new processing steps | Block expansion |
| Want progressive training | Block expansion (easier to freeze selectively) |

## Training Considerations

Width expansion requires more careful training than block expansion:

```python
from cambium import ExpandableModel
from cambium.strategies import WidthExpansion
from cambium.training import StagedTrainer

model = ExpandableModel.from_pretrained("google/gemma-2b")
model.expand(WidthExpansion(hidden_dim_multiplier=1.25))

trainer = StagedTrainer(model)

# Use lower learning rates for width expansion
trainer.add_phase(
    name="width_expansion_training",
    freeze="none",
    lr=5e-6,  # Lower than block expansion
    discriminative_lr={
        r"embed|lm_head": 1e-7,     # Very low for embeddings
        r"original_dims": 5e-6,      # Low for existing dims
        r"new_dims": 1e-5,           # Higher for new dims
    },
    epochs=3,
)
```

## Validation

```python
from cambium import ExpandableModel
from cambium.strategies import WidthExpansion
from cambium.utils import validate_model_output
import torch

model = ExpandableModel.from_pretrained("google/gemma-2b")
model.expand(WidthExpansion(hidden_dim_multiplier=1.25))

# Test forward pass
test_input = torch.randint(0, 32000, (1, 10))  # Batch 1, seq 10

results = validate_model_output(
    model.get_model(),
    test_input,
)

print(f"Output shape: {results['output_shape']}")
print(f"No NaN: {not results['has_nan']}")
print(f"No Inf: {not results['has_inf']}")
```

## Tips for Width Expansion

1. **Start with smaller multipliers** (1.25-1.5x)
2. **Use "copy" initialization** to preserve behavior
3. **Lower learning rates** compared to block expansion
4. **Monitor more carefully** - width expansion changes representations more
5. **Consider combining** with block expansion for maximum effect
