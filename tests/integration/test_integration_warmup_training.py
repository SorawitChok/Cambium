"""Integration warmup training test for Cambium.

Trains only the newly added layers (warmup phase) on a toy dataset.
This demonstrates the standard post-expansion training recipe.
"""
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from cambium import ExpandableModel, InterleavedExpansion
from cambium.training.staged_trainer import StagedTrainer

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"


class ToyTextDataset(Dataset):
    """Tiny in-memory dataset for quick training demos."""

    def __init__(self, tokenizer, num_samples=64, seq_length=64):
        self.samples = []
        texts = [
            "The proliferation of large language models has precipitated a paradigm shift in how we conceptualize intelligence.",
            "In the philosophy of mind, the hard problem of consciousness asks why subjective experience arises from physical processes.",
            "Contemporary geopolitical dynamics are increasingly shaped by the asymmetric distribution of computational resources.",
            "The second law of thermodynamics describes the statistical tendency of isolated systems to evolve toward macrostates.",
            "Epistemologically, Bayesian inference offers a coherent framework for updating beliefs in light of new evidence.",
            "During the European Renaissance, the recovery of classical manuscripts catalyzed intellectual movements.",
            "Climate feedback mechanisms introduce nonlinearities into atmospheric models that complicate precise predictions.",
            "The architecture of transformer-based neural networks leverages self-attention mechanisms.",
            "In constitutional democracies, the tension between majoritarian impulses and minority protections necessitates safeguards.",
            "Emergent phenomena in complex systems demonstrate how localized interactions can generate collective behaviors.",
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


def main():
    print("=" * 70)
    print("Integration Warmup Training Test")
    print("=" * 70)

    # 1. Load and expand
    print(f"\n[1] Loading + expanding model: {MODEL_NAME}")
    wrapper = ExpandableModel.from_pretrained(MODEL_NAME, dtype=torch.float32)
    original_layers = wrapper.config.num_hidden_layers
    wrapper.expand(InterleavedExpansion(num_layers=2, initialization="identity"))
    print(f"    -> Layers: {original_layers} -> {wrapper.config.num_hidden_layers}")

    # 2. Create dataset
    print(f"\n[2] Creating toy dataset")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = ToyTextDataset(tokenizer, num_samples=64)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    print(f"    -> Train batches: {len(train_loader)}")

    # 3. Setup trainer
    print(f"\n[3] Configuring StagedTrainer")
    trainer = StagedTrainer(wrapper)
    trainer.add_phase(
        name="warmup_new_layers",
        freeze="original",
        lr=1e-5,
        epochs=10,
        gradient_accumulation_steps=1,
    )
    print(f"    -> Phase: warmup_new_layers | freeze=original | lr=1e-5 | epochs=10")

    # 4. Print trainable status
    print(f"\n[4] Trainable parameters BEFORE training")
    wrapper.freeze_original()
    wrapper.print_trainable()

    # 5. Run training
    print(f"\n[5] Running training (this may take a minute on CPU)...")
    history = trainer.train(train_loader)

    # 6. Print results
    print(f"\n[6] Training complete")
    for phase_hist in history["phases"]:
        final_loss = phase_hist["losses"][-1] if phase_hist["losses"] else None
        print(
            f"    -> Phase '{phase_hist['name']}' | steps: {phase_hist['steps']} | final loss: {final_loss}"
        )

    # 7. Save
    print(f"\n[7] Saving warmed-up model")
    wrapper.save_expanded("./warmed-up-model")

    # 8. Generation sample
    print(f"\n[8] Quick generation sample after warmup")
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
    print(f"    -> Prompt: {prompt}")
    print(f"    -> Output: {tokenizer.decode(out[0], skip_special_tokens=True)}")

    print("\n" + "=" * 70)
    print("Integration Warmup Training Test: PASSED")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    import sys

    try:
        sys.exit(main())
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        sys.exit(1)
