"""Integration test for Cambium grafted block expansion (examples/08_grafted_block.md).

This script runs every runnable code pattern from the grafted block example
against the small SmolLM2-135M model, including a short fine-tuning run to
show qualitative alignment.
"""
import io
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from cambium import ExpandableModel, GraftedBlockExpansion
from cambium.exceptions import GraftingError
from cambium.training.staged_trainer import StagedTrainer

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"

results: list[tuple[str, bool, str]] = []


class ToyTextDataset(Dataset):
    """Simple text dataset for testing fine-tuning alignment."""

    def __init__(self, tokenizer, num_samples=64, seq_length=64):
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
    print("Integration Test: Grafted Block Expansion (08_grafted_block.md)")
    print("=" * 70)

    print()
    print("=" * 70)
    print(f"Loading base model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = ExpandableModel.from_pretrained(MODEL_NAME, dtype=torch.float32)
    print(
        f"Loaded: {base.config.num_hidden_layers} layers, " f"hidden_size={base.config.hidden_size}"
    )

    # =======================================================================
    # Section A: Graft by index from the same model
    # =======================================================================
    def section_a():
        w = ExpandableModel.from_pretrained(MODEL_NAME, dtype=torch.float32)
        initial = w.config.num_hidden_layers
        print(f"    Original layers: {initial}")
        w.expand(
            GraftedBlockExpansion(
                source_model_id=MODEL_NAME,
                source_block_idx=5,
                positions=[3],
            )
        )
        print(f"    Expanded layers: {w.config.num_hidden_layers}")
        assert w.config.num_hidden_layers == initial + 1

        # Verify the grafted block is trainable by default.
        grafted = w.get_model().model.layers[3]
        trainable = any(p.requires_grad for p in grafted.parameters(recurse=True))
        assert trainable, "Grafted block should be trainable by default"
        print("    Section A: grafted block inserted and is trainable")

    run_section("A. Graft by index (same model)", section_a)

    # =======================================================================
    # Section B: Graft by layer name
    # =======================================================================
    def section_b():
        w = ExpandableModel.from_pretrained(MODEL_NAME, dtype=torch.float32)
        initial = w.config.num_hidden_layers
        w.expand(
            GraftedBlockExpansion(
                source_model_id=MODEL_NAME,
                source_block_name="model.layers.5",
                positions=[3],
            )
        )
        assert w.config.num_hidden_layers == initial + 1
        print(f"    {initial} -> {w.config.num_hidden_layers} layers using source_block_name")

    run_section("B. Graft by layer name", section_b)

    # =======================================================================
    # Section C: Freeze the grafted block
    # =======================================================================
    def section_c():
        w = ExpandableModel.from_pretrained(MODEL_NAME, dtype=torch.float32)
        initial = w.config.num_hidden_layers
        w.expand(
            GraftedBlockExpansion(
                source_model_id=MODEL_NAME,
                source_block_idx=5,
                positions=[3],
                freeze=True,
            )
        )
        assert w.config.num_hidden_layers == initial + 1

        grafted = w.get_model().model.layers[3]
        frozen = all(not p.requires_grad for p in grafted.parameters(recurse=True))
        assert frozen, "Grafted block should be frozen when freeze=True"
        print("    Grafted block is frozen")

    run_section("C. Freeze the grafted block", section_c)

    # =======================================================================
    # Section D: Cross-model projection path
    # =======================================================================
    def section_d():
        # Build a tiny source decoder layer (16 hidden dims) to force the
        # projection path, then wrap it for the 576-dim target model.
        from transformers.models.llama.configuration_llama import LlamaConfig

        from cambium.core.grafting import build_source_decoder_layer
        from cambium.strategies.grafted_expansion import _GraftedBlockWrapper

        small_config = LlamaConfig(
            hidden_size=16,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=64,
            max_position_embeddings=128,
            rms_norm_eps=1e-6,
        )
        small_config._attn_implementation = "eager"
        small_block = build_source_decoder_layer(small_config, layer_idx=0)

        w = ExpandableModel.from_pretrained(MODEL_NAME, dtype=torch.float32)
        wrapped = _GraftedBlockWrapper(
            small_block,
            source_hidden_size=16,
            target_hidden_size=w.config.hidden_size,
        )
        dummy = torch.randn(1, 1, w.config.hidden_size)

        # Provide RoPE position embeddings like a real forward pass would.
        head_dim = small_config.hidden_size // small_config.num_attention_heads
        cos = sin = torch.zeros(1, 1, head_dim)
        with torch.no_grad():
            out = wrapped(dummy, position_embeddings=(cos, sin))
        assert out.shape == dummy.shape
        print("    Cross-size wrapper produces correct output shape with projection")

    run_section("D. Cross-model projection path", section_d)

    # =======================================================================
    # Section E: Validation rejects incompatible block
    # =======================================================================
    def section_e():
        import torch.nn as nn

        w = ExpandableModel.from_pretrained(MODEL_NAME, dtype=torch.float32)

        class BadBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden_size = w.config.hidden_size
                self.proj = nn.Linear(self.hidden_size, self.hidden_size * 2)

            def forward(self, hidden_states, **kwargs):
                return self.proj(hidden_states)

        raised = False
        try:
            # Patch the loader to inject a bad block.
            expander = GraftedBlockExpansion(
                source_model_id=MODEL_NAME,
                source_block_idx=5,
                positions=[3],
                validate=True,
            )
            # Monkey-patch the block before expand()
            bad_block = BadBlock()
            from cambium.strategies import grafted_expansion

            original_loader = grafted_expansion.load_grafted_block
            grafted_expansion.load_grafted_block = lambda **kwargs: bad_block
            expander.expand(w.get_model(), w.engine)
        except Exception as e:
            raised = True
            print(f"    Incompatible block rejected: {type(e).__name__}: {str(e)[:120]}")
        finally:
            grafted_expansion.load_grafted_block = original_loader

        assert raised, "Bad block should have been rejected"

    run_section("E. Validation rejects incompatible block", section_e)

    # =======================================================================
    # Section F: Forward pass with a small prompt
    # =======================================================================
    def section_f():
        w = ExpandableModel.from_pretrained(MODEL_NAME, dtype=torch.float32)
        initial = w.config.num_hidden_layers
        w.expand(
            GraftedBlockExpansion(
                source_model_id=MODEL_NAME,
                source_block_idx=5,
                positions=[3],
            )
        )
        assert w.config.num_hidden_layers == initial + 1

        model = w.get_model()
        model.eval()
        inputs = tokenizer("Artificial intelligence is", return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        assert outputs.logits.shape[-1] == w.config.vocab_size
        print("    Forward pass on a real prompt succeeds")

    run_section("F. Forward pass with real input", section_f)

    # =======================================================================
    # Section G: Qualitative comparison of original vs grafted model outputs
    # =======================================================================
    def section_g():
        prompts = [
            "Artificial intelligence is",
            "The future of technology",
            "In the year 2050",
        ]

        print("\n    --- Baseline (original model) ---")
        baseline = ExpandableModel.from_pretrained(MODEL_NAME, dtype=torch.float32)
        baseline_model = baseline.get_model()
        baseline_model.eval()
        baseline_outputs = []
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                output_ids = baseline_model.generate(
                    **inputs,
                    max_new_tokens=40,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            baseline_outputs.append(text)
            print(f"    Prompt: {prompt!r}")
            print(f"    Output: {text!r}")
            print()

        print("    --- Grafted model (source layer 5 inserted at position 3) ---")
        grafted = ExpandableModel.from_pretrained(MODEL_NAME, dtype=torch.float32)
        grafted.expand(
            GraftedBlockExpansion(
                source_model_id=MODEL_NAME,
                source_block_idx=5,
                positions=[3],
            )
        )
        grafted_model = grafted.get_model()
        grafted_model.eval()
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                output_ids = grafted_model.generate(
                    **inputs,
                    max_new_tokens=40,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            baseline_text = baseline_outputs[prompts.index(prompt)]
            changed = text != baseline_text
            print(f"    Prompt: {prompt!r}")
            print(f"    Output: {text!r}")
            print(f"    Changed from baseline: {changed}")
            print()

    run_section("G. Qualitative output comparison", section_g)

    # =======================================================================
    # Section H: Fine-tune the grafted model and compare again
    # =======================================================================
    def section_h():
        prompts = [
            "Artificial intelligence is",
            "The future of technology",
            "In the year 2050",
        ]

        print("\n    --- Grafted model BEFORE fine-tuning ---")
        w = ExpandableModel.from_pretrained(MODEL_NAME, dtype=torch.float32)
        w.expand(
            GraftedBlockExpansion(
                source_model_id=MODEL_NAME,
                source_block_idx=5,
                positions=[3],
            )
        )

        before_outputs = []
        for prompt in prompts:
            text = generate_text(w.get_model(), tokenizer, prompt)
            before_outputs.append(text)
            print(f"    Prompt: {prompt!r}")
            print(f"    Output: {text!r}")
            print()

        train_dataset = ToyTextDataset(tokenizer, num_samples=64)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

        # Use StagedTrainer for alignment: warmup only the grafted block, then
        # a short full fine-tune with embeddings kept frozen.
        trainer = StagedTrainer(w)
        trainer.add_phase(
            name="warmup_graft",
            freeze="original",
            lr=5e-5,
            epochs=1,
            gradient_accumulation_steps=1,
            eval_every=1000,
        )
        trainer.add_phase(
            name="full_finetune_frozen_embeddings",
            freeze="none",
            lr=1e-5,
            epochs=1,
            gradient_accumulation_steps=1,
            eval_every=1000,
        )

        print("    Fine-tuning grafted model with StagedTrainer...")
        w.freezing_manager.freeze_embeddings()
        trainer.train(train_loader)

        print("\n    --- Grafted model AFTER fine-tuning ---")
        for idx, prompt in enumerate(prompts):
            text = generate_text(w.get_model(), tokenizer, prompt)
            changed = text != before_outputs[idx]
            print(f"    Prompt: {prompt!r}")
            print(f"    Output: {text!r}")
            print(f"    Changed from before fine-tuning: {changed}")
            print()

    run_section("H. Fine-tune grafted model and compare outputs", section_h)

    # =======================================================================
    # Summary
    # =======================================================================
    print()
    print("=" * 70)
    print("Integration Grafted Block Test - Summary")
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
    import sys

    sys.exit(main())
