# Cambium Quickstart Guide

This guide shows you how to get started with Cambium for expanding LLMs.

## Installation

```bash
pip install cambium
# Or with training dependencies
pip install "cambium[train]"
```

## Basic Usage: Expanding a Model

```python
from cambium import ExpandableModel, InterleavedExpansion

# Load a base model
model = ExpandableModel.from_pretrained("google/gemma-2b")

# Expand with 4 new transformer blocks (LLaMA-Pro style)
expander = InterleavedExpansion(
    num_layers=4,
    initialization="identity"  # Near-identity initialization
)

# Apply expansion
model.expand(expander)

# Check what changed
print(model.get_expansion_report())
```

## Freezing and Training Setup

```python
# Freeze original weights, train only new layers
model.freeze_original()

# Print trainable parameter summary
model.print_trainable()

# Get the expanded PyTorch model
torch_model = model.get_model()
```

## Saving and Loading

```python
# Save the expanded model
model.save_expanded("./gemma-2b-expanded-4L")

# Later, load it back
from cambium import ExpandableModel

model = ExpandableModel.load_expanded("./gemma-2b-expanded-4L")
```

## Validation

```python
# Validate the expansion
results = model.validate()
print(f"Valid: {results['valid']}")
print(f"Total params: {results['checks']['parameters']['total']:,}")
print(f"Trainable params: {results['checks']['parameters']['trainable']:,}")
```

## Next Steps

- See [02_interleaved_expansion.md](02_interleaved_expansion.md) for detailed interleaved expansion
- See [03_staged_training.md](03_staged_training.md) for training workflows
- See [04_complete_workflow.md](04_complete_workflow.md) for end-to-end examples
