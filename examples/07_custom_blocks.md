# Custom Block Expansion

The most powerful feature of Cambium — define your own architecture blocks and insert them into any existing LLM.

## The Concept

```
Original:  [Block0] → [Block1] → [Block2] → [Block3]
Custom:    [Block0] → [YOUR BLOCK] → [Block1] → [Block2] → [Block3]
```

Instead of inserting copies of the model's own block type, you can insert **any** PyTorch module — novel attention mechanisms, custom MLPs, retrieval layers, or anything else you can dream up.

## Using Template Blocks

Cambium ships with 5 ready-made block templates:

```python
from cambium import ExpandableModel, CustomBlockExpansion
from cambium.blocks import SwiGLUBlock, GatedResidualBlock

model = ExpandableModel.from_pretrained("google/gemma-2b")

# Insert SwiGLU MLP blocks
model.expand(CustomBlockExpansion(
    block_class=SwiGLUBlock,
    num_layers=4,
    residual_connection=True,  # output = input + block(input)
    initialization="smart",
))
```

### Available Templates

| Block | Description | Use Case |
|-------|-------------|----------|
| **SwiGLUBlock** | SwiGLU MLP (LLaMA-style) | Capacity expansion |
| **GatedResidualBlock** | Gated projection with SiLU | Lightweight expansion |
| **MultiQueryAttentionBlock** | Multi-query attention | Efficient inference |
| **CrossAttentionBlock** | Gated self-attention | Attention augmentation |
| **RetentionBlock** | Retention mechanism (linear) | Long-sequence models |

## Defining Custom Blocks

### Simple Custom Block

```python
from cambium import ExpandableModel, CustomBlockExpansion
from cambium.blocks import CambiumBlock
import torch.nn as nn

class MyBlock(CambiumBlock):
    """My custom architecture block."""

    # Declare what config keys your block needs
    required_config_keys = ["hidden_size"]

    def __init__(self, config, layer_idx=0):
        super().__init__()
        hidden = config.hidden_size
        self.proj = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.act = nn.GELU()

    def forward(self, hidden_states, **kwargs):
        # Return a delta — residual_connection=True will add input back
        x = self.proj(self.norm(hidden_states))
        return self.act(x)

model = ExpandableModel.from_pretrained("google/gemma-2b")

model.expand(CustomBlockExpansion(
    block_class=MyBlock,
    num_layers=2,
    positions=[8, 16],
    residual_connection=True,
    initialization="smart",
))
```

### Block with Internal Residual

If your block already includes a residual connection:

```python
class ResidualMLP(CambiumBlock):
    required_config_keys = ["hidden_size"]

    def __init__(self, config, layer_idx=0):
        super().__init__()
        hidden = config.hidden_size
        self.up = nn.Linear(hidden, hidden * 4)
        self.down = nn.Linear(hidden * 4, hidden)
        self.act = nn.GELU()

    def forward(self, hidden_states, **kwargs):
        # This block already handles residual internally
        return hidden_states + self.down(self.act(self.up(hidden_states)))

model.expand(CustomBlockExpansion(
    block_class=ResidualMLP,
    num_layers=2,
    residual_connection=False,  # Don't double-wrap
))
```

### Block Without CambiumBlock Base

You don't have to subclass `CambiumBlock`. Any `nn.Module` works as long as it:
- Accepts `hidden_states` as the first argument
- Returns a tensor of the same shape
- Accepts `**kwargs` (or at least doesn't crash on extra args)

```python
import torch.nn as nn

class PlainBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, **kwargs):
        return self.linear(hidden_states)

model.expand(CustomBlockExpansion(
    block_class=PlainBlock,
    num_layers=2,
    # block_class called as PlainBlock(config) — must accept config
))
```

## Three Ways to Provide Blocks

### 1. block_class (Recommended)

```python
model.expand(CustomBlockExpansion(
    block_class=MyBlock,       # Called as MyBlock(config, layer_idx=i)
    num_layers=4,
    residual_connection=True,
))
```

### 2. block_factory (Full Control)

```python
def my_factory():
    """Create a block with custom logic."""
    config = model.get_model().config
    block = MyBlock(config)
    # Custom setup here
    nn.init.xavier_uniform_(block.proj.weight)
    return block

model.expand(CustomBlockExpansion(
    block_factory=my_factory,
    num_layers=4,
    residual_connection=True,
))
```

### 3. block_instances (Pre-created)

```python
config = model.get_model().config
blocks = [MyBlock(config, layer_idx=i) for i in range(4)]

model.expand(CustomBlockExpansion(
    block_instances=blocks,
    positions=[4, 8, 12, 16],
    residual_connection=True,
))
```

## Initialization Strategies

```python
# Smart init (default) — near-zero for output projections
model.expand(CustomBlockExpansion(
    block_class=MyBlock,
    num_layers=2,
    initialization="smart",
))

# Custom initialization function
def my_init(block):
    nn.init.xavier_uniform_(block.proj.weight)
    nn.init.zeros_(block.proj.bias)

model.expand(CustomBlockExpansion(
    block_class=MyBlock,
    num_layers=2,
    initialization="custom",
    custom_init_fn=my_init,
))
```

## Validation

By default, Cambium validates your blocks before insertion:

- **Shape check**: Dummy forward pass to verify output matches input
- **Signature check**: Warns if `forward()` doesn't accept `**kwargs`
- **Config check**: Verifies `required_config_keys` exist in model config
- **NaN check**: Detects NaN parameters after initialization

```python
# Validation enabled (default)
model.expand(CustomBlockExpansion(
    block_class=MyBlock,
    num_layers=2,
    validate=True,   # Catch issues early
))

# Skip validation (if you know what you're doing)
model.expand(CustomBlockExpansion(
    block_class=MyBlock,
    num_layers=2,
    validate=False,
))
```

If validation fails, you get a clear error:

```
BlockValidationError: Block 0 validation failed:
  - Block 0: output shape (1, 1, 64) doesn't match input shape (1, 1, 32).
    With residual_connection=True, block output must match input shape.
```

## Mixing with Other Strategies

```python
from cambium import ExpandableModel, InterleavedExpansion, CustomBlockExpansion
from cambium.blocks import SwiGLUBlock, GatedResidualBlock

model = ExpandableModel.from_pretrained("google/gemma-2b")

# First: insert standard blocks at even positions
model.expand(InterleavedExpansion(
    num_layers=2,
    positions=[6, 18],
))

# Then: insert custom SwiGLU blocks at specific positions
model.expand(CustomBlockExpansion(
    block_class=SwiGLUBlock,
    num_layers=2,
    positions=[12, 24],
    residual_connection=True,
))

# Then: add gated residual blocks
model.expand(CustomBlockExpansion(
    block_class=GatedResidualBlock,
    num_layers=2,
    residual_connection=True,
))

print(model.get_expansion_report())
```

## Advanced: Retrieval-Augmented Block

```python
from cambium.blocks import CambiumBlock
import torch.nn as nn

class RAGBlock(CambiumBlock):
    """Block that can retrieve from an external knowledge store."""

    required_config_keys = ["hidden_size"]

    def __init__(self, config, layer_idx=0):
        super().__init__()
        hidden = config.hidden_size
        self.query_proj = nn.Linear(hidden, hidden)
        self.retriever_proj = nn.Linear(hidden, hidden)
        self.fusion = nn.Linear(hidden * 2, hidden)
        self.gate = nn.Linear(hidden, 1)

    def forward(self, hidden_states, **kwargs):
        # In practice, retrieval would happen here
        # For now, this is a placeholder architecture
        query = self.query_proj(hidden_states)

        # Simulated retrieval output (replace with actual retrieval)
        retrieved = self.retriever_proj(hidden_states)

        # Fuse query and retrieved
        fused = torch.cat([query, retrieved], dim=-1)
        output = self.fusion(fused)

        # Learned gating
        gate = torch.sigmoid(self.gate(hidden_states))
        return gate * output

model.expand(CustomBlockExpansion(
    block_class=RAGBlock,
    num_layers=2,
    residual_connection=True,
    initialization="smart",
))
```

## Tips for Custom Blocks

1. **Always accept `**kwargs`** — HF models pass attention_mask, position_ids, etc.
2. **Match the hidden_size** — Your block must return the same shape as input
3. **Use `residual_connection=True`** for blocks that return deltas
4. **Use `required_config_keys`** to catch config mismatches early
5. **Initialize output projections near zero** for identity-like behavior
6. **Start with a template** — Subclass SwiGLUBlock or GatedResidualBlock and customize
7. **Test with `validate=True`** — It catches shape and config issues before training