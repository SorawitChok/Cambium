# Grafted Block Expansion

Take a single pretrained transformer block from one HuggingFace model and insert it into another.

## The Concept

```
Source model:  [B0] [B1] [B2] [B3] [B4] [B5] ...
                                  |
                                  v
Target model:  [B0] [B1] [B2] [GRAFTED B5] [B3] [B4] ...
```

Instead of adding a randomly initialized block, you reuse a block that has already been
trained. Cambium downloads only the source model config and the safetensors shards
that contain the requested block — the rest of the source weights stay on the hub.

## Basic Usage

This example uses the same model as both source and target so it is fully runnable
on a laptop. In practice the real value is grafting from a *different* model.

```python
from cambium import ExpandableModel, GraftedBlockExpansion

model = ExpandableModel.from_pretrained("HuggingFaceTB/SmolLM2-135M", dtype="float32")

model.expand(GraftedBlockExpansion(
    source_model_id="HuggingFaceTB/SmolLM2-135M",
    source_block_idx=5,
    positions=[3],
))

print(f"Model now has {model.config.num_hidden_layers} layers")
```

- `source_model_id`: HuggingFace repo id of the model to graft from.
- `source_block_idx`: Which layer to copy.
- `positions`: Where to insert the block in the target model.

## Select by Layer Name

If you know the exact checkpoint key prefix, you can use it directly:

```python
model.expand(GraftedBlockExpansion(
    source_model_id="HuggingFaceTB/SmolLM2-135M",
    source_block_name="model.layers.5",
    positions=[3],
))
```

## Cross-Model Grafting with Projection

When the source and target models have different hidden sizes, Cambium automatically
adds learnable projection layers around the grafted block:

```python
# Use any supported causal-LM repo as the source.
model.expand(GraftedBlockExpansion(
    source_model_id="unsloth/Llama-3.2-1B",  # different hidden size than target
    source_block_idx=7,
    positions=[10],
    projection=True,  # default; required when hidden sizes differ
))
```

Set `projection=False` to raise an error instead of adding projections when the
hidden sizes do not match.

## Freezing the Grafted Block

By default the grafted block is trainable. To keep it frozen:

```python
model.expand(GraftedBlockExpansion(
    source_model_id="HuggingFaceTB/SmolLM2-135M",
    source_block_idx=5,
    positions=[3],
    freeze=True,
))
```

## How the Lightweight Download Works

1. Load the source model `config.json`.
2. Build the architecture for a single decoder layer from that config.
3. Read `model.safetensors.index.json` to find which shards contain the block.
4. Download only those shards.
5. Load only the matching keys into the standalone block.

If the repo has a single `model.safetensors` file, only that file is downloaded.
Grafting currently requires safetensors checkpoints.

## Notes and Limitations

- Only one block can be grafted per `GraftedBlockExpansion` call.
- Supported architectures follow Cambium's decoder-layer support: `llama`, `mistral`,
  `gemma`, `gemma3`, `qwen2`, `qwen3`, and compatible variants.
- The source block must accept `hidden_states` as its first forward argument and
  accept `**kwargs` like standard HuggingFace decoder layers.
- KV-cache tuples returned by the source block are discarded by the wrapper.
