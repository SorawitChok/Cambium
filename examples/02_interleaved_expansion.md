# Interleaved Block Expansion

This example demonstrates LLaMA-Pro style interleaved expansion where new transformer blocks are inserted between existing ones.

## The Concept

```
Original:  [Block 0] → [Block 1] → [Block 2] → [Block 3]
Expanded:  [Block 0] → [New 0] → [Block 1] → [New 1] → [Block 2] → [New 2] → [Block 3] → [New 3]
```

## Basic Interleaved Expansion

```python
import torch
from cambium import ExpandableModel, InterleavedExpansion

# Load a small base model (tested: HuggingFaceTB/SmolLM2-135M)
model_name = "HuggingFaceTB/SmolLM2-135M"
wrapper = ExpandableModel.from_pretrained(model_name, dtype=torch.float32)
original_layers = wrapper.config.num_hidden_layers
print(f"Original layers: {original_layers}")

# Expand with 2 new blocks
expander = InterleavedExpansion(
    num_layers=2,
    initialization="identity",
)
wrapper.expand(expander)
print(f"Layers after expansion: {wrapper.config.num_hidden_layers}")
```

## Initialization Strategies

```python
from cambium import InterleavedExpansion

# Identity mapping (default) - new blocks act as identity initially
expander_identity = InterleavedExpansion(
    num_layers=2,
    initialization="identity",
)

# Small random noise - slight variations for diversity
expander_small_random = InterleavedExpansion(
    num_layers=2,
    initialization="small_random",
)

# Larger noise - more aggressive initialization
expander_noise = InterleavedExpansion(
    num_layers=2,
    initialization="noise",
)

# Zero initialization - output projections initialized to zero
expander_zero = InterleavedExpansion(
    num_layers=2,
    initialization="zero",
)
```

## Generation Comparison (Identity Initialization)

Identity-initialized new layers behave like near pass-throughs, so the expanded
model's output (before any training) should be very close to the original:

```python
import torch
from transformers import AutoTokenizer
from cambium import ExpandableModel, InterleavedExpansion

model_name = "HuggingFaceTB/SmolLM2-135M"
prompt = "Artificial intelligence is"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Expanded model
wrapper = ExpandableModel.from_pretrained(model_name, dtype=torch.float32)
wrapper.expand(InterleavedExpansion(num_layers=2, initialization="identity"))
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
print(f"Expanded (untrained): {expanded_text}")

# Original model for comparison
original_wrapper = ExpandableModel.from_pretrained(model_name, dtype=torch.float32)
original_model = original_wrapper.get_model()
original_model.to(device)
original_model.eval()
with torch.no_grad():
    original_ids = original_model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
original_text = tokenizer.decode(original_ids[0], skip_special_tokens=True)
print(f"Original (baseline): {original_text}")
```

## Training Warmup (New Layers Only)

After expanding, the standard recipe is to freeze the original weights and train
only the newly added layers:

```python
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from cambium import ExpandableModel, InterleavedExpansion
from cambium.training.staged_trainer import StagedTrainer

model_name = "HuggingFaceTB/SmolLM2-135M"

# Load model and expand
wrapper = ExpandableModel.from_pretrained(model_name, dtype=torch.float32)
wrapper.expand(InterleavedExpansion(num_layers=2, initialization="identity"))

# Build a tiny toy dataset
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

class ToyTextDataset(Dataset):
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

train_dataset = ToyTextDataset(tokenizer, num_samples=64)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Freeze original layers and train only new ones
wrapper.freeze_original()
wrapper.print_trainable()

trainer = StagedTrainer(wrapper)
trainer.add_phase(
    name="warmup_new_layers",
    freeze="original",
    lr=1e-5,
    epochs=10,
    gradient_accumulation_steps=1,
)

history = trainer.train(train_loader)
for phase_hist in history["phases"]:
    print(
        f"Phase '{phase_hist['name']}' -> steps: {phase_hist['steps']}, "
        f"final loss: {phase_hist['losses'][-1]:.4f}"
    )

# Save the warmed-up model
wrapper.save_expanded("./warmed-up-model")

# Quick generation sample after warmup
model = wrapper.get_model()
model.to(trainer.device)
model.eval()
prompt = "Artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt").to(trainer.device)
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
print(f"Prompt: {prompt}")
print(f"Output: {tokenizer.decode(out[0], skip_special_tokens=True)}")
```

## Chained Expansions

```python
from cambium import ExpandableModel, InterleavedExpansion

wrapper = ExpandableModel.from_pretrained("HuggingFaceTB/SmolLM2-135M")

# First expansion: add 2 blocks
wrapper.expand(InterleavedExpansion(num_layers=2))

# Second expansion: add 2 more blocks
wrapper.expand(InterleavedExpansion(num_layers=2))

print(f"Total expansions: {len(wrapper.expansions)}")
print(f"Total layers: {wrapper.config.num_hidden_layers}")
```

## Validation

```python
import torch
from cambium import ExpandableModel, InterleavedExpansion

wrapper = ExpandableModel.from_pretrained("HuggingFaceTB/SmolLM2-135M")
wrapper.expand(InterleavedExpansion(num_layers=2, initialization="identity"))
wrapper = wrapper.load_expanded("./warmed-up-model")

# Validate forward pass
model = wrapper.get_model()
model.eval()
inputs = torch.randint(0, wrapper.config.vocab_size, (1, 10))
with torch.no_grad():
    outputs = model(inputs)
print(f"Logits shape: {outputs.logits.shape}")

# Run library validation
report = wrapper.validate()
print(f"Valid: {report['valid']}")
print(f"Total params: {report['checks']['parameters']['total']:,}")
print(f"Trainable: {report['checks']['parameters']['trainable']:,}")
print(f"NaN: {report['checks']['numerical_stability']['has_nan']}")
print(f"Inf: {report['checks']['numerical_stability']['has_inf']}")
```

## Tips for Interleaved Expansion

1. **Number of Layers**: Start with 20-30% of original layers
2. **Initialization**: "identity" is safest, "noise" for more diversity
3. **Positions**: Let Cambium auto-distribute unless you have specific needs
4. **Training**: Always freeze original weights in Phase 1
