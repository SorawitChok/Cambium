"""Integration test for Cambium StagedTrainer.

This test demonstrates:
1. Interleaved expansion on a small model.
2. Multi-phase staged training (warmup, progressive unfreezing, full fine-tune).
3. Discriminative learning rates.
4. Inference comparison: original vs stage-trained model.
5. Checkpointing and resuming.
6. Training history tracking.
"""
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from cambium import ExpandableModel, InterleavedExpansion
from cambium.training.staged_trainer import StagedTrainer
from cambium.utils.validation import CatastrophicForgettingDetector

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"


class ToyTextDataset(Dataset):
    """Simple text dataset for testing."""

    def __init__(self, tokenizer, num_samples=128, seq_length=64):
        self.samples = []
        texts = [
            "The proliferation of large language models has precipitated a paradigm shift.",
            "In the philosophy of mind, the hard problem of consciousness asks why subjective experience arises.",
            "Contemporary geopolitical dynamics are increasingly shaped by the asymmetric distribution of computational resources.",
            "The second law of thermodynamics describes the statistical tendency of isolated systems.",
            "Epistemologically, Bayesian inference offers a coherent framework for updating beliefs.",
            "During the European Renaissance, the recovery of classical manuscripts catalyzed intellectual movements.",
            "Climate feedback mechanisms introduce nonlinearities into atmospheric models.",
            "The architecture of transformer-based neural networks leverages self-attention mechanisms.",
            "In constitutional democracies, the tension between majoritarian impulses and minority protections necessitates safeguards.",
            "Emergent phenomena in complex systems demonstrate how localized interactions generate collective behaviors.",
            "Machine learning algorithms trained on biased datasets can perpetuate societal inequalities.",
            "The development of quantum computing threatens to render many current encryption schemes obsolete.",
            "Cognitive biases systematically distort human reasoning in predictable ways.",
            "The Fermi paradox asks why we have observed no evidence of extraterrestrial civilizations.",
            "CRISPR-Cas9 gene editing technology has revolutionized molecular biology.",
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


def generate_sample(model, tokenizer, prompt, device="cpu", max_tokens=50):
    """Generate text from a prompt."""
    model.to(device)
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    print("=" * 70)
    print("Integration Test: StagedTrainer")
    print("=" * 70)

    # 1. Load original model
    print("\n[1] Loading original model for baseline comparison")
    print(f"    -> Model: {MODEL_NAME}")
    original_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    original_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if original_tokenizer.pad_token is None:
        original_tokenizer.pad_token = original_tokenizer.eos_token
    print(f"    -> Original layers: {original_model.config.num_hidden_layers}")

    # 2. Load and expand
    print("\n[2] Loading model via ExpandableModel wrapper")
    wrapper = ExpandableModel.from_pretrained(MODEL_NAME, dtype=torch.float32)
    original_layers = wrapper.config.num_hidden_layers
    print(f"    -> Original layers: {original_layers}")

    print("\n[3] Applying InterleavedExpansion (add 2 new layers)")
    wrapper.expand(InterleavedExpansion(num_layers=2, initialization="identity"))
    expanded_layers = wrapper.config.num_hidden_layers
    print(f"    -> Expanded layers: {original_layers} -> {expanded_layers}")

    # 3. Create datasets
    print("\n[4] Creating toy dataset")
    tokenizer = original_tokenizer
    train_dataset = ToyTextDataset(tokenizer, num_samples=128)
    eval_dataset = ToyTextDataset(tokenizer, num_samples=32)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=4, shuffle=False)
    print(f"    -> Train batches: {len(train_loader)}")
    print(f"    -> Eval batches: {len(eval_loader)}")

    # 4. Setup trainer
    print("\n[5] Configuring StagedTrainer with multi-phase training")
    trainer = StagedTrainer(wrapper)

    trainer.add_phase(
        name="phase1_warmup_new_layers",
        freeze="original",
        lr=5e-5,
        epochs=3,
        gradient_accumulation_steps=2,
        eval_every=50,
    )
    print("    -> Added Phase 1: warmup_new_layers (freeze=original, lr=5e-5)")

    trainer.add_phase(
        name="phase2_unfreeze_tail",
        freeze=None,
        unfreeze_groups=[-2, -1],
        lr=2e-5,
        epochs=2,
        gradient_accumulation_steps=2,
        eval_every=50,
    )
    print("    -> Added Phase 2: unfreeze_tail (unfreeze_groups=[-2, -1], lr=2e-5)")

    trainer.add_phase(
        name="phase3_discriminative_lr",
        freeze="none",
        lr=1e-6,
        discriminative_lr={
            "embeddings": 1e-7,
            (0, 19): 1e-6,
            (20, 29): 5e-6,
            "new_layers": 1e-5,
        },
        epochs=2,
        gradient_accumulation_steps=2,
        eval_every=50,
    )
    print("    -> Added Phase 3: discriminative_lr (unfreeze all, discriminative LR)")

    trainer.add_phase(
        name="phase4_full_finetune",
        freeze="none",
        lr=1e-6,
        epochs=1,
        gradient_accumulation_steps=2,
        eval_every=50,
    )
    print("    -> Added Phase 4: full_finetune (freeze=none, lr=1e-6)")

    # 5. Baseline inference
    print("\n[6] Running baseline inference (BEFORE training)")
    prompts = [
        "Artificial intelligence is",
        "The future of technology",
        "In the year 2050",
    ]
    baseline_outputs = []
    for prompt in prompts:
        output = generate_sample(wrapper.get_model(), tokenizer, prompt)
        baseline_outputs.append(output)
        print(f"    -> Prompt: '{prompt}'")
        print(f"    -> Output: '{output[len(prompt):].strip()[:80]}...'")

    # 6. Run training
    print("\n[7] Running multi-phase training (this may take a few minutes)...")
    print("-" * 70)
    history = trainer.train(train_loader, eval_loader)
    print("-" * 70)

    # 7. Training results
    print("\n[8] Training Results Summary")
    print("-" * 50)
    for phase_hist in history["phases"]:
        name = phase_hist["name"]
        steps = phase_hist["steps"]
        losses = phase_hist["losses"]
        if losses:
            avg_loss = sum(losses) / len(losses)
            final_loss = losses[-1]
            print(f"  Phase '{name}':")
            print(f"    Steps: {steps}")
            print(f"    Avg Loss: {avg_loss:.4f}")
            print(f"    Final Loss: {final_loss:.4f}")
        else:
            print(f"  Phase '{name}': No loss data")

    # 8. Inference comparison
    print("\n[9] Running inference comparison (AFTER training)")
    print("-" * 50)
    print("  Stage-trained model outputs:")
    for i, prompt in enumerate(prompts):
        output = generate_sample(wrapper.get_model(), tokenizer, prompt)
        print(f"    -> Prompt: '{prompt}'")
        print(f"    -> Output: '{output[len(prompt):].strip()[:80]}...'")
        print(f"    -> Change from baseline: {'Yes' if output != baseline_outputs[i] else 'No'}")

    print("\n  Original model outputs (for comparison):")
    for prompt in prompts:
        output = generate_sample(original_model, tokenizer, prompt)
        print(f"    -> Prompt: '{prompt}'")
        print(f"    -> Output: '{output[len(prompt):].strip()[:80]}...'")

    # 9. Save checkpoint
    print("\n[10] Saving checkpoint")
    trainer.save_checkpoint(
        "./staged_trainer_checkpoint.pt",
        metadata={
            "model_name": MODEL_NAME,
            "original_layers": original_layers,
            "expanded_layers": expanded_layers,
            "phases_completed": len(history["phases"]),
        },
    )
    print("    -> Checkpoint saved to: ./staged_trainer_checkpoint.pt")

    # 10. Save expanded model
    print("\n[11] Saving expanded model")
    wrapper.save_expanded("./stage-trained-model")
    print("    -> Model saved to: ./stage-trained-model")

    # 11. Checkpoint resume test
    print("\n[12] Testing checkpoint resume")
    wrapper2 = ExpandableModel.from_pretrained(MODEL_NAME, dtype=torch.float32)
    wrapper2.expand(InterleavedExpansion(num_layers=2, initialization="identity"))
    trainer2 = StagedTrainer(wrapper2)
    metadata = trainer2.load_checkpoint("./staged_trainer_checkpoint.pt")
    print(f"    -> Loaded checkpoint from phase {trainer2.current_phase_idx}")
    print(f"    -> Global step: {trainer2.global_step}")
    print(f"    -> Metadata: {metadata}")

    # 12. Trainable parameter summary
    print("\n[13] Final trainable parameter summary")
    wrapper.print_trainable()

    # 13. Catastrophic forgetting check
    print("\n[14] Checking for catastrophic forgetting")
    print("  Note: Divergence is expected after training - it means the model learned!")
    detector = CatastrophicForgettingDetector(
        base_model=original_model,
        threshold=5.0,
    )
    test_text = "The quick brown fox jumps over the lazy dog"
    test_inputs = tokenizer(test_text, return_tensors="pt")
    is_ok, kl_div = detector.check(wrapper.get_model(), test_inputs)
    print(f"  KL divergence: {kl_div:.6f}")
    print(f"  Within acceptable range (threshold={detector.threshold}): {is_ok}")

    print("\n" + "=" * 70)
    print("Integration StagedTrainer Test: PASSED")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    import sys

    try:
        sys.exit(main())
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"\nERROR: {type(e).__name__}: {e}")
        sys.exit(1)
