# Interleaved Block Expansion

This example demonstrates LLaMA-Pro style interleaved expansion where new transformer blocks are inserted between existing ones.

## The Concept

```
Original:  [Block 0] → [Block 1] → [Block 2] → [Block 3]
Expanded:  [Block 0] → [New 0] → [Block 1] → [New 1] → [Block 2] → [New 2] → [Block 3] → [New 3]
```

## Basic Interleaved Expansion

```python
from cambium import ExpandableModel, InterleavedExpansion

# Load base model
model = ExpandableModel.from_pretrained("google/gemma-2b")

# Expand with 4 new blocks
expander = InterleavedExpansion(
    num_layers=4,
    initialization="identity",  # Near-identity initialization
    layer_attribute="model.layers"  # Path to transformer layers
)

model.expand(expander)
```

## Auto-Distributed vs Manual Positions

```python
from cambium import ExpandableModel, InterleavedExpansion

model = ExpandableModel.from_pretrained("mistralai/Mistral-7B-v0.1")

# Option 1: Auto-distribute (recommended)
# Cambium automatically spaces new blocks evenly
expander_auto = InterleavedExpansion(num_layers=4)
model.expand(expander_auto)
# New blocks inserted at positions: [4, 9, 13, 18] for a 32-layer model

# Option 2: Manual positions (advanced)
expander_manual = InterleavedExpansion(
    num_layers=4,
    positions=[4, 12, 20, 28]  # Insert at specific indices
)
model.expand(expander_manual)
```

## Initialization Strategies

```python
from cambium import InterleavedExpansion

# Identity mapping (default) - new blocks act as identity initially
expander = InterleavedExpansion(
    num_layers=4,
    initialization="identity"
)

# Small random noise - slight variations for diversity
expander = InterleavedExpansion(
    num_layers=4,
    initialization="small_random"
)

# Larger noise - more aggressive initialization
expander = InterleavedExpansion(
    num_layers=4,
    initialization="noise"
)

# Zero initialization - output projections initialized to zero
expander = InterleavedExpansion(
    num_layers=4,
    initialization="zero"
)
```

## Custom Block Configuration

```python
from cambium import InterleavedExpansion

# Override default config for new blocks
expander = InterleavedExpansion(
    num_layers=4,
    block_config={
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_attention_heads": 8,
        # ... other config options
    }
)
```

## Chained Expansions

```python
from cambium import ExpandableModel, InterleavedExpansion

model = ExpandableModel.from_pretrained("google/gemma-2b")

# First expansion: add 2 blocks
model.expand(InterleavedExpansion(num_layers=2))

# Second expansion: add 4 more blocks
model.expand(InterleavedExpansion(num_layers=4))

print(f"Total expansions: {len(model.expansions)}")
```

## Memory Estimation

```python
from cambium import ExpandableModel, InterleavedExpansion
from cambium.utils import estimate_memory_usage

model = ExpandableModel.from_pretrained("google/gemma-2b")
model.expand(InterleavedExpansion(num_layers=4))

# Estimate memory for training
memory_estimate = estimate_memory_usage(
    model.get_model(),
    batch_size=4,
    sequence_length=512,
    dtype="fp16",
    gradient_checkpointing=True
)

print(f"Estimated memory: {memory_estimate['total_gb']:.2f} GB")
print(f"Recommended: {memory_estimate['recommended_gb']:.2f} GB")
```

## Validation

```python
from cambium import ExpandableModel, InterleavedExpansion
from transformers import AutoTokenizer

model = ExpandableModel.from_pretrained("google/gemma-2b")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")

# Expand
model.expand(InterleavedExpansion(num_layers=4))

# Test that the model still works
model.get_model().eval()

text = "The capital of France is"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model.get_model()(**inputs)

# Check logits shape
print(f"Logits shape: {outputs.logits.shape}")

# Sample next token
next_token_logits = outputs.logits[:, -1, :]
next_token = next_token_logits.argmax(dim=-1)
print(f"Next token: {tokenizer.decode(next_token)}")
```

## Tips for Interleaved Expansion

1. **Number of Layers**: Start with 20-30% of original layers
2. **Initialization**: "identity" is safest, "noise" for more diversity
3. **Positions**: Let Cambium auto-distribute unless you have specific needs
4. **Training**: Always freeze original weights in Phase 1
