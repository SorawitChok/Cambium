# Staged Training

Cambium's `StagedTrainer` orchestrates multi-phase training with progressive unfreezing, which is crucial for successfully training expanded models.

## The Standard Recipe

The recommended training approach for expanded models:

```
Phase 1: Warmup New Layers
  - Freeze all original weights
  - Train only new layers at higher LR
  - Goal: New layers learn to work with frozen base

Phase 2: Progressive Unfreezing
  - Gradually unfreeze original layers (last layers first)
  - Use lower LR for original layers
  - Goal: Adapt original layers to work with new layers

Phase 3: Full Fine-tuning
  - Unfreeze all layers
  - Low learning rate with discriminative LR
  - Goal: Joint optimization of full model
```

## Basic Staged Training

```python
import torch
from cambium import ExpandableModel, InterleavedExpansion
from cambium.training import StagedTrainer, TrainingPhase
from torch.utils.data import DataLoader

# Load and expand model
model = ExpandableModel.from_pretrained("HuggingFaceTB/SmolLM2-135M", dtype=torch.float32)
model.expand(InterleavedExpansion(num_layers=4))

# Create trainer
trainer = StagedTrainer(model)

# Phase 1: Train only new layers
trainer.add_phase(
    name="warmup_new_layers",
    freeze="original",  # Freeze all original weights
    lr=1e-4,
    epochs=2,
    batch_size=4,
)

# Phase 2: Unfreeze last 2 groups (progressive unfreezing)
trainer.add_phase(
    name="unfreeze_tail",
    freeze=None,  # Keep current freeze state
    unfreeze_groups=[-2, -1],  # Unfreeze last 2 of 4 groups
    lr=5e-5,
    epochs=1,
)

# Phase 3: Full fine-tuning
trainer.add_phase(
    name="full_finetune",
    freeze="none",  # Unfreeze all
    lr=1e-6,
    epochs=1,
)

# Train
# Assuming you have train_dataloader and eval_dataloader
history = trainer.train(train_dataloader, eval_dataloader)
```

## Discriminative Learning Rates

```python
import torch
from cambium import ExpandableModel, InterleavedExpansion
from cambium.training import StagedTrainer

model = ExpandableModel.from_pretrained("HuggingFaceTB/SmolLM2-135M", dtype=torch.float32)
model.expand(InterleavedExpansion(num_layers=2))

trainer = StagedTrainer(model)

# Phase with discriminative learning rates
trainer.add_phase(
    name="discriminative_training",
    freeze="none",
    discriminative_lr={
        "embeddings": 1e-8,       # Semantic name: embed_tokens + lm_head
        (0, 19): 1e-6,             # Layers 0-19: low
        (20, 29): 5e-6,            # Layers 20-29: medium
        "new_layers": 1e-4,        # Semantic name: new expanded layers
    },
    epochs=2,
)

history = trainer.train(train_dataloader, eval_dataloader)
```

**Supported key types for `discriminative_lr`:**

| Key Type | Example | Description |
|----------|---------|-------------|
| Layer tuple | `(0, 19): 1e-6` | Apply LR to layers 0-19 (inclusive). Matches `model.layers.N.*` |
| Semantic name | `"embeddings": 1e-8` | Built-in alias for `embed_tokens` and `lm_head` |
| Semantic name | `"new_layers": 1e-4` | Matches layers marked as new during expansion |
| Semantic name | `"original_layers": 1e-6` | Matches all original (non-new) transformer layers |
| Regex string | `r"model\.layers\.\d+": 1e-6` | Full regex control (fallback) |

## Manual Freezing Control

```python
import torch
from cambium import ExpandableModel, InterleavedExpansion

model = ExpandableModel.from_pretrained("HuggingFaceTB/SmolLM2-135M", dtype=torch.float32)
model.expand(InterleavedExpansion(num_layers=2))

# Get the freezing manager
fm = model.freezing_manager

# Freeze specific patterns
fm.freeze_by_pattern(r"model\.layers\.[0-9]\.")  # Layers 0-9

# Unfreeze specific range
fm.unfreeze_layer_range(20, 29)  # Last 10 layers

# Unfreeze by groups
fm.unfreeze_group(3, num_groups=4)  # Unfreeze last quarter

# Print current status
fm.print_trainable_status()

# Now train with your own loop
# ...
```

## Integration with Hugging Face Trainer

```python
import torch
from cambium import ExpandableModel, InterleavedExpansion
from transformers import TrainingArguments, Trainer
from cambium.training import TrainingUtilities

# Load and expand model
model = ExpandableModel.from_pretrained("HuggingFaceTB/SmolLM2-135M", dtype=torch.float32)
model.expand(InterleavedExpansion(num_layers=2))

# Phase 1: Freeze original
model.freeze_original()

# Get optimizer with discriminative LR
optimizer = TrainingUtilities.get_optimizer_with_discriminative_lr(
    model.get_model(),
    lr_config={
        r"embed|lm_head": 1e-8,
        r"model\.layers": 0,  # Frozen
        r"new_": 1e-4,
    }
)

# Setup HF Trainer
training_args = TrainingArguments(
    output_dir="./results-phase1",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    learning_rate=1e-4,
    # ... other args
)

trainer = Trainer(
    model=model.get_model(),
    args=training_args,
    train_dataset=train_dataset,
    optimizers=(optimizer, None),  # Use our custom optimizer
)

# Train Phase 1
trainer.train()

# Save Phase 1
trainer.save_model("./phase1-checkpoint")

# Phase 2: Unfreeze and continue
model.unfreeze_all()
# ... continue with lower LR
```

## Integration with TRL (SFT)

```python
import torch
from cambium import ExpandableModel, InterleavedExpansion
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer

model = ExpandableModel.from_pretrained("HuggingFaceTB/SmolLM2-135M", dtype=torch.float32)
model.expand(InterleavedExpansion(num_layers=2))
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Freeze for Phase 1
model.freeze_original()

# Setup SFT Trainer
sft_config = SFTConfig(
    output_dir="./sft-results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    learning_rate=1e-4,
)

trainer = SFTTrainer(
    model=model.get_model(),
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    args=sft_config,
)

# Train
trainer.train()
```

## Memory Optimization

```python
import torch
from cambium import ExpandableModel, InterleavedExpansion
from cambium.training import TrainingUtilities

model = ExpandableModel.from_pretrained("HuggingFaceTB/SmolLM2-135M", dtype=torch.float32)
model.expand(InterleavedExpansion(num_layers=2))

# Enable memory optimizations
TrainingUtilities.enable_memory_optimizations(
    model.get_model(),
    gradient_checkpointing=True,
    cpu_offload=False,
    mixed_precision="fp16",
)

# Now train as usual
# ...
```

## Checkpointing and Resuming

```python
import torch
from cambium import ExpandableModel, InterleavedExpansion
from cambium.training import StagedTrainer

model = ExpandableModel.from_pretrained("HuggingFaceTB/SmolLM2-135M", dtype=torch.float32)
model.expand(InterleavedExpansion(num_layers=2))

trainer = StagedTrainer(model)

# Add phases
trainer.add_phase(name="phase1", freeze="original", lr=1e-4, epochs=2)
trainer.add_phase(name="phase2", freeze="none", lr=1e-6, epochs=1)

# Train for a bit
history = trainer.train(train_dataloader)

# Save checkpoint
trainer.save_checkpoint("./checkpoint-step-1000.pt")

# Later, resume
metadata = trainer.load_checkpoint("./checkpoint-step-1000.pt")
print(f"Resuming from phase {trainer.current_phase_idx}, step {trainer.global_step}")

# Continue training
history = trainer.train(train_dataloader)
```

## Monitoring Catastrophic Forgetting

```python
import torch
from cambium import ExpandableModel, InterleavedExpansion
from cambium.utils.validation import CatastrophicForgettingDetector
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load original model for comparison
original_model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM2-135M", dtype=torch.float32
)

# Create detector
detector = CatastrophicForgettingDetector(
    base_model=original_model,
    threshold=0.1,  # KL divergence threshold
)

# Load and expand model
model = ExpandableModel.from_pretrained("HuggingFaceTB/SmolLM2-135M", dtype=torch.float32)
model.expand(InterleavedExpansion(num_layers=2))

# Check divergence during training
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
text = "The capital of France is Paris"
inputs = tokenizer(text, return_tensors="pt")

is_ok, kl_div = detector.check(model.get_model(), inputs)
print(f"KL divergence: {kl_div:.4f}, Acceptable: {is_ok}")

# Get report
report = detector.get_report()
print(f"Violations: {report['violations']}/{report['num_checks']}")
```

## Recommended Hyperparameters

| Phase | Freeze | Learning Rate | Epochs | Notes |
|-------|--------|---------------|--------|-------|
| 1 | original | 1e-4 | 2-3 | Higher LR for new layers |
| 2 | last N layers | 5e-5 | 1-2 | Gradual unfreezing |
| 3 | none | 1e-6 | 1 | Full fine-tune, very low LR |

## Tips for Staged Training

1. **Always start with frozen base** - Let new layers warm up first
2. **Monitor validation loss** - Watch for divergence
3. **Use discriminative LR** - Different LRs for different layer types
4. **Save checkpoints** - Save after each phase
5. **Watch memory** - Expanded models need more memory
