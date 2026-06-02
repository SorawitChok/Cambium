# Complete Workflow Example

This example shows a complete end-to-end workflow: loading a model, expanding it, training in stages, and saving the result.

## Full Example: Expanding and Fine-tuning

```python
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Requires: pip install datasets
from datasets import load_dataset

from cambium import ExpandableModel, InterleavedExpansion
from cambium.training import StagedTrainer
from cambium.training import TrainingUtilities
from cambium.utils import estimate_memory_usage

# ============================================
# Step 1: Load Base Model
# ============================================
print("Step 1: Loading base model...")
model = ExpandableModel.from_pretrained(
    "HuggingFaceTB/SmolLM2-135M",
    dtype=torch.float32,
)
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Check original model info
n_layers = len(model.get_model().model.layers)
print(f"Original layers: {n_layers}")

# ============================================
# Step 2: Estimate Memory
# ============================================
print("\nStep 2: Memory estimation...")
estimate = estimate_memory_usage(
    model.get_model(),
    batch_size=4,
    sequence_length=512,
    dtype="fp16",
    gradient_checkpointing=True,
)
print(f"Estimated memory: {estimate['total_gb']:.2f} GB")
print(f"Recommended: {estimate['recommended_gb']:.2f} GB")

# ============================================
# Step 3: Expand Model
# ============================================
print("\nStep 3: Expanding model...")
expander = InterleavedExpansion(
    num_layers=2,               # Add 2 new layers
    initialization="identity",  # Near-identity initialization
)

model.expand(expander)

print(f"Expanded layers: {len(model.get_model().model.layers)}")
print(f"Expansion report:\n{model.get_expansion_report()}")

# ============================================
# Step 4: Prepare Dataset
# ============================================
print("\nStep 4: Preparing dataset...")

# Load a small dataset for demonstration
# (Use your own dataset in practice)
dataset = load_dataset("json", data_files="train_data.jsonl", split="train")

# Tokenize
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# Split train/val
train_data = tokenized_dataset.select(range(900))
val_data = tokenized_dataset.select(range(900, 1000))

train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
val_loader = DataLoader(val_data, batch_size=4)

# ============================================
# Step 5: Enable Memory Optimizations
# ============================================
print("\nStep 5: Enabling memory optimizations...")

TrainingUtilities.enable_memory_optimizations(
    model.get_model(),
    gradient_checkpointing=True,
    mixed_precision="fp16",
)

# ============================================
# Step 6: Setup Staged Training
# ============================================
print("\nStep 6: Setting up staged training...")

# Setup trainer
trainer = StagedTrainer(
    model,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

# Phase 1: Warmup new layers only
# Freeze everything except new layers
trainer.add_phase(
    name="phase1_warmup",
    freeze="original",  # Freeze all original weights
    lr=1e-4,
    epochs=2,
    batch_size=4,
    gradient_accumulation_steps=2,
    warmup_steps=100,
)

# Phase 2: Unfreeze last half of the model (groups 2 and 3 of 4)
trainer.add_phase(
    name="phase2_unfreeze_tail",
    freeze=None,  # Don't change freeze state
    unfreeze_groups=[-2, -1],
    lr=5e-5,
    epochs=1,
    batch_size=4,
    warmup_steps=50,
)

# Phase 3: Progressive unfreezing - unfreeze first half (groups 0 and 1)
trainer.add_phase(
    name="phase3_unfreeze_middle",
    freeze=None,
    unfreeze_groups=[-4, -3],
    lr=2e-5,
    epochs=1,
    batch_size=4,
    warmup_steps=50,
)

# Phase 4: Full fine-tuning
# Now everything trains with low LR
trainer.add_phase(
    name="phase4_full_finetune",
    freeze="none",  # Unfreeze all
    lr=1e-6,
    discriminative_lr={
        r"embed|lm_head": 1e-8,
        r"model\.layers": 1e-6,
        r"new_": 1e-5,
    },
    epochs=1,
    batch_size=4,
    warmup_steps=50,
)

print(f"Configured {len(trainer.phases)} training phases")

# ============================================
# Step 7: Train
# ============================================
print("\nStep 7: Starting training...")
print("Training phases:")
for i, phase in enumerate(trainer.phases, 1):
    print(f"  Phase {i}: {phase.name} ({phase.epochs} epochs, lr={phase.lr})")

# Run training
history = trainer.train(train_loader, val_loader)

print("\nTraining complete!")
print(f"History: {history}")

# ============================================
# Step 8: Save
# ============================================
print("\nStep 8: Saving model...")

# Save expanded model with metadata
model.save_expanded("./smollm-expanded-2L-finetuned")

# Also save a checkpoint
trainer.save_checkpoint("./final-checkpoint.pt")

print("Saved to: ./smollm-expanded-2L-finetuned")

# ============================================
# Step 9: Test Inference
# ============================================
print("\nStep 9: Testing inference...")

test_prompt = "The capital of France is"
inputs = tokenizer(test_prompt, return_tensors="pt").to(model.freezing_manager.model.device)

model.get_model().eval()
with torch.no_grad():
    outputs = model.get_model().generate(
        **inputs,
        max_new_tokens=20,
        do_sample=True,
        temperature=0.7,
    )

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Prompt: {test_prompt}")
print(f"Output: {result}")

print("\n=== Complete Workflow Finished ===")
```

## Resuming from Checkpoint

```python
from cambium import ExpandableModel
from cambium.training import StagedTrainer

# Load previously saved model
model = ExpandableModel.load_expanded("./smollm-expanded-2L-finetuned")

# Create trainer
trainer = StagedTrainer(model)

# Load checkpoint
trainer.load_checkpoint("./final-checkpoint.pt")
print(f"Resuming from phase {trainer.current_phase_idx}, step {trainer.global_step}")

# Add remaining phases and continue
# ...
```

## Using with PEFT/LoRA on Top

```python
import torch
from cambium import ExpandableModel, InterleavedExpansion
from cambium.training import TrainingUtilities

# Requires: pip install peft
from peft import LoraConfig, get_peft_model

# Load and expand
model = ExpandableModel.from_pretrained("HuggingFaceTB/SmolLM2-135M", dtype=torch.float32)
model.expand(InterleavedExpansion(num_layers=2))

# Add LoRA on top of expanded model
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

peft_model = get_peft_model(model.get_model(), lora_config)
peft_model.print_trainable_parameters()

# Now train with standard PEFT workflow
# ...
```

## Loading and Using in Production

```python
import torch
from cambium import ExpandableModel
from transformers import AutoTokenizer, pipeline

# Load expanded model
model_wrapper = ExpandableModel.load_expanded("./smollm-expanded-2L-finetuned")
tokenizer = AutoTokenizer.from_pretrained("./smollm-expanded-2L-finetuned")

# Use with transformers pipeline
text_gen = pipeline(
    "text-generation",
    model=model_wrapper.get_model(),
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
)

# Generate
result = text_gen(
    "Write a poem about machine learning:",
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
)
print(result[0]["generated_text"])
```

## Tips for Complete Workflows

1. **Start Small**: Test with a small model and subset of data first
2. **Monitor Memory**: Use `estimate_memory_usage()` before training
3. **Save Often**: Save checkpoints after each phase
4. **Validate**: Test generation at each phase
5. **Compare**: Track metrics vs original model to detect forgetting
