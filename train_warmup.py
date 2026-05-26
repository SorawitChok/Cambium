"""Manual training test for Cambium expanded model.

Trains only the newly added layers (warmup phase) on a toy dataset.
This demonstrates the standard post-expansion training recipe.
"""
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from cambium import ExpandableModel, InterleavedExpansion
from cambium.training.staged_trainer import StagedTrainer, TrainingPhase

print("=" * 50)
print("Manual Warmup Training Test")
print("=" * 50)

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"

# 1. Load model and expand
print(f"\n[1] Loading + expanding model: {MODEL_NAME}")
wrapper = ExpandableModel.from_pretrained(MODEL_NAME, dtype=torch.float32)
original_layers = wrapper.config.num_hidden_layers
wrapper.expand(InterleavedExpansion(num_layers=2, initialization="identity"))
print(f"    -> Layers: {original_layers} -> {wrapper.config.num_hidden_layers}")

# 2. Create a tiny in-memory dataset (random token sequences)
# In a real workflow you'd use a real text dataset (e.g., datasets.load_dataset)
print(f"\n[2] Creating toy dataset")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
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
            text = texts[i % len(texts)]
            # Append EOS so the model learns when to stop generating
            text_with_eos = text + tokenizer.eos_token
            tokens = tokenizer(
                text_with_eos,
                truncation=True,
                max_length=seq_length,
                padding="max_length",
            )
            input_ids = torch.tensor(tokens["input_ids"])
            attention_mask = torch.tensor(tokens["attention_mask"])
            # Mask padding positions in labels so they don't compute loss
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            self.samples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# 3. Build dataloaders
print(f"\n[3] Building dataloaders")
train_dataset = ToyTextDataset(tokenizer, num_samples=64)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
print(f"    Train batches: {len(train_loader)}")

# 4. Setup trainer with a warmup phase (only new layers trainable)
print(f"\n[4] Configuring StagedTrainer")
trainer = StagedTrainer(wrapper)
trainer.add_phase(
    name="warmup_new_layers",
    freeze="original",
    lr=1e-5,
    epochs=10,
    gradient_accumulation_steps=1,
)
print(f"    Phase: warmup_new_layers | freeze=original | lr=1e-5 | epochs=10")

# 5. Print trainable status before training
print(f"\n[5] Trainable parameters BEFORE training")
wrapper.freeze_original()  # apply the same freeze config the trainer will use
wrapper.print_trainable()

# 6. Run training
print(f"\n[6] Running training (this may take a minute on CPU)...")
history = trainer.train(train_loader)

# 7. Print results
print(f"\n[7] Training complete")
for phase_hist in history["phases"]:
    print(
        f"    Phase '{phase_hist['name']}' -> steps: {phase_hist['steps']}, final loss: {phase_hist['losses'][-1]:.4f}"
    )

# 8. Save the warmed-up model
print(f"\n[8] Saving warmed-up model")
wrapper.save_expanded("./warmed-up-model")

# 9. Generate a quick sample to see if output changed
print(f"\n[9] Quick generation sample after warmup")
model = wrapper.get_model()
device = trainer.device
model.to(device)
model.eval()
prompt = "Artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
print(f"    Prompt: {prompt}")
print(f"    Output: {tokenizer.decode(out[0], skip_special_tokens=True)}")

print("\n" + "=" * 50)
print("Warmup training test complete!")
print("=" * 50)
