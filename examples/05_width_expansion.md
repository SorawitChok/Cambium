# Width Expansion

Width expansion increases the hidden dimensions of the model (e.g., 768 → 960), providing more capacity through wider representations.

## The Concept

Unlike block expansion which adds more layers, width expansion makes existing layers wider:

```
Original:  hidden_size = 768
Expanded:  hidden_size = 960 (768 * 1.25)

This affects:
- Attention projections
- MLP intermediate dimensions
- Embeddings
- LayerNorm weights
```

## Basic Width Expansion

```python
import torch
from cambium import ExpandableModel
from cambium.strategies import WidthExpansion

# Load a small base model (tested: JackFram/llama-160m)
model_name = "JackFram/llama-160m"
wrapper = ExpandableModel.from_pretrained(model_name, dtype=torch.float32)
print(f"Original hidden_size: {wrapper.config.hidden_size}")

# Expand width by 1.25x
expander = WidthExpansion(
    hidden_dim_multiplier=1.25,
    initialization="zero",
)
wrapper.expand(expander)
print(f"New hidden_size: {wrapper.config.hidden_size}")
```

## Initialization Strategies

```python
from cambium.strategies import WidthExpansion

# Zero initialization for new dimensions
expander_zero = WidthExpansion(
    hidden_dim_multiplier=1.25,
    initialization="zero",
)

# Copy existing weights to new dimensions
expander_copy = WidthExpansion(
    hidden_dim_multiplier=1.25,
    initialization="copy",
)

# Noise injection for new dimensions
expander_noise = WidthExpansion(
    hidden_dim_multiplier=1.25,
    initialization="noise",
)
```

## Full Width Expansion with Training

This complete example loads a model, expands width, compares generation before
and after training, then saves the result:

```python
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from cambium import ExpandableModel
from cambium.strategies import WidthExpansion
from cambium.training.staged_trainer import StagedTrainer
from cambium.utils.validation import validate_model_output

MODEL_NAME = "JackFram/llama-160m"
MULTIPLIER = 1.25
PROMPT = "The future of artificial intelligence is"
GEN_KWARGS = {"max_new_tokens": 30, "do_sample": False}


class ToyTextDataset(Dataset):
    """Tiny in-memory dataset for quick training demos."""

    def __init__(self, tokenizer, num_samples=64, seq_length=64):
        self.samples = []
        texts = [
            "The proliferation of large language models has precipitated a paradigm shift in how we conceptualize intelligence, blurring the once-distinct boundary between statistical pattern matching and genuine cognitive reasoning.",
            "In the philosophy of mind, the hard problem of consciousness asks why subjective experience arises from physical processes, a question that remains stubbornly resistant to reductionist explanation despite centuries of inquiry.",
            "Contemporary geopolitical dynamics are increasingly shaped by the asymmetric distribution of computational resources, wherein nation-states and corporate entities that control advanced semiconductor fabrication exert disproportionate influence over global information ecosystems.",
            "The second law of thermodynamics, while often misconstrued as a principle of universal decay, more accurately describes the statistical tendency of isolated systems to evolve toward macrostates with the greatest number of corresponding microstates.",
            "Epistemologically, Bayesian inference offers a coherent framework for updating beliefs in light of new evidence, yet its practical application demands careful scrutiny of prior assumptions that may encode unrecognized biases.",
            "During the European Renaissance, the recovery of classical Greek and Arabic manuscripts catalyzed intellectual movements that fundamentally reconceptualized humanity's relationship to nature, authority, and the limits of knowledge.",
            "Climate feedback mechanisms, including albedo reduction from melting ice and methane release from thawing permafrost, introduce nonlinearities into atmospheric models that complicate precise long-term predictions.",
            "The architecture of transformer-based neural networks leverages self-attention mechanisms to compute contextualized representations, enabling the modeling of long-range dependencies that recurrent architectures struggle to capture efficiently.",
            "In constitutional democracies, the tension between majoritarian impulses and minority protections necessitates institutional safeguards, such as judicial review and supermajoritarian thresholds, that deliberately slow the pace of political change.",
            "Emergent phenomena in complex systems, from ant colonies to financial markets, demonstrate how localized interactions among simple agents can generate collective behaviors that are not obviously derivable from the properties of individual components.",
        ]
        for i in range(num_samples):
            text = texts[i % len(texts)] + tokenizer.eos_token
            tokens = tokenizer(
                text,
                truncation=True,
                max_length=seq_length,
                padding="max_length",
            )
            input_ids = torch.tensor(tokens["input_ids"])
            attention_mask = torch.tensor(tokens["attention_mask"])
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            self.samples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def generate_text(model, tokenizer, prompt):
    """Greedy-decode a continuation for the prompt."""
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            pad_token_id=tokenizer.pad_token_id,
            **GEN_KWARGS,
        )
    return tokenizer.decode(gen_ids[0], skip_special_tokens=True)


# Shared tokenizer and dataset
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

train_data = ToyTextDataset(tokenizer, num_samples=64)
train_loader = DataLoader(train_data, batch_size=4, shuffle=True)

# Original model (baseline)
orig_wrapper = ExpandableModel.from_pretrained(MODEL_NAME, dtype=torch.float32)
orig_model = orig_wrapper.get_model()
orig_text = generate_text(orig_model, tokenizer, PROMPT)
print(f"Original: '{orig_text}'")

# Expand width
exp_wrapper = ExpandableModel.from_pretrained(MODEL_NAME, dtype=torch.float32)
exp_model = exp_wrapper.get_model()
exp_wrapper.expand(WidthExpansion(hidden_dim_multiplier=MULTIPLIER, initialization="zero"))
print(f"New hidden_size: {exp_wrapper.config.hidden_size}")
exp_text_before = generate_text(exp_model, tokenizer, PROMPT)
print(f"Expanded (before train): '{exp_text_before}'")

# Validate forward pass
with torch.no_grad():
    dummy = torch.randint(0, exp_wrapper.config.vocab_size, (1, 10))
    out_logits = exp_model(dummy).logits
assert not torch.isnan(out_logits).any(), "NaN in output"
assert not torch.isinf(out_logits).any(), "Inf in output"
results = validate_model_output(exp_model, dummy)
print(f"validate_model_output: success={results['success']}")

# Train with StagedTrainer
exp_wrapper.freeze_original()
exp_wrapper.freezing_manager.freeze_embeddings()
exp_wrapper.print_trainable()

trainer = StagedTrainer(exp_wrapper)
trainer.add_phase(
    name="full_expansion",
    freeze=None,  # Keep manual freeze config
    lr=1e-5,
    epochs=10,
)
trainer.train(train_loader)
exp_text_after = generate_text(exp_model, tokenizer, PROMPT)
print(f"Expanded (after train): '{exp_text_after}'")

# Save and reload
exp_wrapper.save_expanded("test-width-expanded")
reloaded = ExpandableModel.from_pretrained("test-width-expanded")
assert reloaded.get_model().config.hidden_size == exp_wrapper.config.hidden_size
print("Reload OK")
```

## Selective Layer Expansion

You can also expand width only for specific layers and choose whether to expand attention.
This example reuses the tokenizer, dataset, and `generate_text` helper from the previous
section; run that first or include the same setup code:

```python
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from cambium import ExpandableModel
from cambium.strategies import WidthExpansion
from cambium.training.staged_trainer import StagedTrainer

MODEL_NAME = "JackFram/llama-160m"
MULTIPLIER = 1.25
PROMPT = "The future of artificial intelligence is"
GEN_KWARGS = {"max_new_tokens": 30, "do_sample": False}

# Rebuild tokenizer and dataset (same as previous example)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# ...build ToyTextDataset and DataLoader here (see previous block)...
# For brevity, assume train_loader is available.


def generate_text(model, tokenizer, prompt):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            pad_token_id=tokenizer.pad_token_id,
            **GEN_KWARGS,
        )
    return tokenizer.decode(gen_ids[0], skip_special_tokens=True)


# Expand only layers 4-7, MLP only (skip attention)
sel_wrapper = ExpandableModel.from_pretrained(MODEL_NAME, dtype=torch.float32)
sel_model = sel_wrapper.get_model()
sel_wrapper.expand(
    WidthExpansion(
        hidden_dim_multiplier=MULTIPLIER,
        initialization="zero",
        layer_indices=list(range(4, 8)),
        expand_attention=False,
    )
)
for i in [0, 4, 7, 11]:
    layer = sel_model.model.layers[i]
    print(f"layer {i}: up_proj.out={layer.mlp.up_proj.weight.shape[0]}")

# Train selectively expanded model
sel_wrapper.freeze_original()
sel_wrapper.freezing_manager.freeze_embeddings()
sel_wrapper.print_trainable()

trainer_sel = StagedTrainer(sel_wrapper)
trainer_sel.add_phase(
    name="selective_expansion",
    freeze=None,
    lr=1e-5,
    epochs=10,
)
trainer_sel.train(train_loader)
sel_text_after = generate_text(sel_model, tokenizer, PROMPT)
print(f"Selective (after train): '{sel_text_after}'")
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

Width expansion requires more careful training than block expansion. The example
above freezes original weights and embeddings/LM head to preserve pretrained
token representations while training only the new expanded dimensions.

## Tips for Width Expansion

1. **Start with smaller multipliers** (1.25-1.5x)
2. **Use "zero" or "copy" initialization** to preserve behavior
3. **Lower learning rates** compared to block expansion
4. **Monitor more carefully** - width expansion changes representations more
5. **Consider combining** with block expansion for maximum effect
