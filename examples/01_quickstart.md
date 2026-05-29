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
import torch
from cambium import ExpandableModel, InterleavedExpansion

# Load a small base model (HuggingFaceTB/SmolLM2-135M)
model_name = "HuggingFaceTB/SmolLM2-135M"
wrapper = ExpandableModel.from_pretrained(model_name, dtype=torch.float32)
print(f"Loaded: {wrapper}")

# Check layers before expansion
original_layers = wrapper.config.num_hidden_layers
print(f"Original layer count: {original_layers}")

# Expand with 2 new transformer blocks (LLaMA-Pro style)
expander = InterleavedExpansion(
    num_layers=2,
    initialization="identity",  # Near-identity initialization
)
wrapper.expand(expander)
print(f"Expanded: {wrapper}")

# Verify layer count grew
new_layers = wrapper.config.num_hidden_layers
assert new_layers == original_layers + 2
print(f"New layer count: {new_layers} (expected {original_layers + 2})")
```

## Freezing and Training Setup

```python
# Freeze original weights, train only new layers
wrapper.freeze_original()

# Print trainable parameter summary
wrapper.print_trainable()

# Get the expanded PyTorch model
torch_model = wrapper.get_model()
```

## Forward Pass Sanity Check

```python
# Run a forward pass to verify the expanded model works
inputs = torch.randint(0, wrapper.config.vocab_size, (1, 10))
with torch.no_grad():
    out = torch_model(inputs)
print(f"Output logits shape: {out.logits.shape}")
```

## Save and Reload

```python
# Save the expanded model
wrapper.save_expanded("./test-expanded")

# Later, load it back
from cambium import ExpandableModel

reloaded = ExpandableModel.load_expanded("./test-expanded")
print(f"Reloaded: {reloaded}")
```

## Validation

```python
# Validate the expansion
report = wrapper.validate()
print(f"Valid: {report['valid']}")
print(f"Total params: {report['checks']['parameters']['total']:,}")
print(f"Trainable: {report['checks']['parameters']['trainable']:,}")
print(f"NaN: {report['checks']['numerical_stability']['has_nan']}")
print(f"Inf: {report['checks']['numerical_stability']['has_inf']}")
```

## Generation Example

Because identity-initialized new layers act as near pass-throughs, the
expanded model's output (before any training) should look very similar to the
original model's output:

```python
import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

prompt = "Artificial intelligence is"
model = wrapper.get_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    expanded_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
expanded_text = tokenizer.decode(expanded_ids[0], skip_special_tokens=True)
print(f"Prompt: {prompt}")
print(f"Output: {expanded_text}")
```

## Next Steps

- See [02_interleaved_expansion.md](02_interleaved_expansion.md) for detailed interleaved expansion
- See [03_staged_training.md](03_staged_training.md) for training workflows
- See [04_complete_workflow.md](04_complete_workflow.md) for end-to-end examples
