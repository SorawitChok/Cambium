"""
Integration test for cross-model grafting and full-model fine-tuning.

This script grafts a decoder block from a larger Llama model
(unsloth/Llama-3.2-1B-Instruct) into a smaller target model
(HuggingFaceTB/SmolLM2-135M), then fine-tunes the whole network on a toy
dataset and compares generation quality before and after training.

Note:
- The source checkpoint is a single safetensors file, so the full file is
  downloaded even though only one block's weights are kept in memory.
- Full fine-tuning of ~200M parameters on CPU is intentionally kept very
  short (16 steps) so this script remains runnable on a laptop.
"""
import io
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from cambium import ExpandableModel, GraftedBlockExpansion
from cambium.training.staged_trainer import StagedTrainer

TARGET_MODEL = "HuggingFaceTB/SmolLM2-135M"
SOURCE_MODEL = "unsloth/Llama-3.2-1B-Instruct"
SOURCE_BLOCK_IDX = 8
TARGET_POSITION = 3

results: list[tuple[str, bool, str]] = []


class ToyTextDataset(Dataset):
    """Simple text dataset for fine-tuning alignment."""

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


def generate_text(model, tokenizer, prompt, max_new_tokens=40):
    """Generate text from a prompt using greedy decoding."""
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def run_section(name: str, fn):
    """Run a section, capture its printed output, record pass/fail."""
    print()
    print("-" * 70)
    print(f"[SECTION] {name}")
    print("-" * 70)
    buf_out, buf_err = io.StringIO(), io.StringIO()
    try:
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            fn()
        captured = buf_out.getvalue()
        if captured:
            print(captured, end="" if captured.endswith("\n") else "\n")
        err = buf_err.getvalue()
        if err:
            print("STDERR:", err)
        results.append((name, True, ""))
        print(f">>> RESULT: PASS  ({name})")
    except Exception as e:
        captured = buf_out.getvalue()
        if captured:
            print(captured, end="" if captured.endswith("\n") else "\n")
        err = buf_err.getvalue()
        if err:
            print("STDERR:", err)
        traceback.print_exc()
        results.append((name, False, f"{type(e).__name__}: {e}"))
        print(f">>> RESULT: FAIL  ({name})")


def main():
    print("=" * 70)
    print("Integration Test: Cross-Model Grafted Block + Full Fine-Tuning")
    print("=" * 70)
    print(f"Target: {TARGET_MODEL}")
    print(f"Source: {SOURCE_MODEL} layer {SOURCE_BLOCK_IDX}")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # =======================================================================
    # Section A: Graft a block from the larger source model
    # =======================================================================
    def section_a():
        print(f"\n    Loading target model: {TARGET_MODEL}")
        wrapper = ExpandableModel.from_pretrained(TARGET_MODEL, dtype=torch.float32)
        print(
            f"    Target: {wrapper.config.num_hidden_layers} layers, "
            f"hidden_size={wrapper.config.hidden_size}"
        )

        print(f"    Grafting source block from: {SOURCE_MODEL}")
        wrapper.expand(
            GraftedBlockExpansion(
                source_model_id=SOURCE_MODEL,
                source_block_idx=SOURCE_BLOCK_IDX,
                positions=[TARGET_POSITION],
                projection=True,
            )
        )
        print(
            f"    Expanded target: {wrapper.config.num_hidden_layers} layers, "
            f"trainable params={sum(p.numel() for p in wrapper.get_model().parameters() if p.requires_grad):,}"
        )

    run_section("A. Cross-model graft with projection", section_a)

    # =======================================================================
    # Section B: Generate before fine-tuning
    # =======================================================================
    def section_b():
        print("\n    Loading fresh target model and grafting...")
        wrapper = ExpandableModel.from_pretrained(TARGET_MODEL, dtype=torch.float32)
        wrapper.expand(
            GraftedBlockExpansion(
                source_model_id=SOURCE_MODEL,
                source_block_idx=SOURCE_BLOCK_IDX,
                positions=[TARGET_POSITION],
                projection=True,
            )
        )
        model = wrapper.get_model()

        prompts = [
            "Artificial intelligence is",
            "The future of technology",
            "In the year 2050",
        ]

        print("\n    --- Grafted model BEFORE fine-tuning ---")
        before_outputs = []
        for prompt in prompts:
            text = generate_text(model, tokenizer, prompt)
            before_outputs.append(text)
            print(f"    Prompt: {prompt!r}")
            print(f"    Output: {text!r}")
            print()

        # Save for next section
        section_b.before_outputs = before_outputs
        section_b.wrapper = wrapper

    run_section("B. Generation before fine-tuning", section_b)

    # =======================================================================
    # Section C: Staged fine-tune (warmup graft + full fine-tune)
    # =======================================================================
    def section_c():
        wrapper = section_b.wrapper

        train_dataset = ToyTextDataset(tokenizer, num_samples=64)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

        print("\n    Fine-tuning with StagedTrainer")
        trainer = StagedTrainer(wrapper)

        # Phase 1: only adapt the grafted block and its projection layers.
        trainer.add_phase(
            name="warmup_graft",
            freeze="original",
            lr=5e-5,
            epochs=1,
            gradient_accumulation_steps=1,
            eval_every=1000,
        )

        # Phase 2: full fine-tune while keeping embeddings frozen.
        trainer.add_phase(
            name="full_finetune_frozen_embeddings",
            freeze="none",
            lr=1e-5,
            epochs=1,
            gradient_accumulation_steps=1,
            eval_every=1000,
        )

        wrapper.freezing_manager.freeze_embeddings()
        history = trainer.train(train_loader)

        for phase_hist in history["phases"]:
            losses = phase_hist["losses"]
            if losses:
                avg = sum(losses) / len(losses)
                final = losses[-1]
                print(
                    f"    Phase '{phase_hist['name']}': "
                    f"avg_loss={avg:.4f}, final_loss={final:.4f}"
                )

        prompts = [
            "Artificial intelligence is",
            "The future of technology",
            "In the year 2050",
        ]

        print("\n    --- Grafted model AFTER fine-tuning ---")
        for idx, prompt in enumerate(prompts):
            text = generate_text(wrapper.get_model(), tokenizer, prompt)
            before_text = section_b.before_outputs[idx]
            changed = text != before_text
            print(f"    Prompt: {prompt!r}")
            print(f"    Output: {text!r}")
            print(f"    Changed from before fine-tuning: {changed}")
            print()

    run_section("C. Staged fine-tune and compare generation", section_c)

    # =======================================================================
    # Section D: Inverse strategy - freeze graft, train original layers
    # =======================================================================
    def section_d():
        print("\n    Loading fresh target model for inverse fine-tuning strategy...")
        wrapper = ExpandableModel.from_pretrained(TARGET_MODEL, dtype=torch.float32)
        wrapper.expand(
            GraftedBlockExpansion(
                source_model_id=SOURCE_MODEL,
                source_block_idx=SOURCE_BLOCK_IDX,
                positions=[TARGET_POSITION],
                projection=True,
            )
        )

        prompts = [
            "Artificial intelligence is",
            "The future of technology",
            "In the year 2050",
        ]

        print("\n    --- Grafted model BEFORE inverse fine-tuning ---")
        before_outputs = []
        for prompt in prompts:
            text = generate_text(wrapper.get_model(), tokenizer, prompt)
            before_outputs.append(text)
            print(f"    Prompt: {prompt!r}")
            print(f"    Output: {text!r}")
            print()

        train_dataset = ToyTextDataset(tokenizer, num_samples=64)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

        wrapper.freezing_manager.freeze_embeddings()
        wrapper.freezing_manager.freeze_by_pattern(rf".*model\.layers\.{TARGET_POSITION}\.")

        trainable = sum(p.numel() for p in wrapper.get_model().parameters() if p.requires_grad)
        print(f"    Trainable parameters (excluding graft+embeddings): {trainable:,}")

        trainer = StagedTrainer(wrapper)
        trainer.add_phase(
            name="adapt_around_frozen_graft",
            freeze=None,
            lr=1e-5,
            epochs=2,
            gradient_accumulation_steps=1,
            eval_every=1000,
        )

        history = trainer.train(train_loader)
        for phase_hist in history["phases"]:
            losses = phase_hist["losses"]
            if losses:
                avg = sum(losses) / len(losses)
                final = losses[-1]
                print(
                    f"    Phase '{phase_hist['name']}': "
                    f"avg_loss={avg:.4f}, final_loss={final:.4f}"
                )

        print("\n    --- Grafted model AFTER inverse fine-tuning ---")
        for idx, prompt in enumerate(prompts):
            text = generate_text(wrapper.get_model(), tokenizer, prompt)
            changed = text != before_outputs[idx]
            print(f"    Prompt: {prompt!r}")
            print(f"    Output: {text!r}")
            print(f"    Changed from before fine-tuning: {changed}")
            print()

    run_section("D. Inverse: freeze graft, train original layers", section_d)

    # =======================================================================
    # Summary
    # =======================================================================
    print()
    print("=" * 70)
    print("Integration Cross-Model Grafting Test - Summary")
    print("=" * 70)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    total = len(results)
    for name, ok, err in results:
        status = "PASS" if ok else "FAIL"
        line = f"  [{status}] {name}"
        if not ok:
            line += f"  -- {err}"
        print(line)
    print()
    print(f"Total: {total} | Passed: {passed} | Failed: {failed}")
    print("=" * 70)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
