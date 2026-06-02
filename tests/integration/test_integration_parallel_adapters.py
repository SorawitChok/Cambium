"""Integration test for Cambium parallel-adapter expansion.

Exercises the API surface from examples/06_parallel_adapters.md:
1. Standalone ParallelBottleneckAdapter / ParallelAttentionAdapter modules.
2. End-to-end ParallelAdapterExpansion (bottleneck) with frozen-base training.
3. Targeted-layer variant (last few layers only).
4. Attention adapter variant.
5. Side-by-side generation summary.
"""
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from cambium import ExpandableModel
from cambium.strategies import ParallelAdapterExpansion
from cambium.strategies.parallel_adapters import (
    ParallelAttentionAdapter,
    ParallelBottleneckAdapter,
)
from cambium.training.staged_trainer import StagedTrainer
from cambium.utils.validation import validate_model_output

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
PROMPT = "Artificial intelligence is"
GEN_KWARGS = {"max_new_tokens": 30, "do_sample": False}
BOTTLENECK_DIM = 64
ATTN_HEADS = 4


class ToyTextDataset(Dataset):
    """Tiny in-memory dataset for quick training demos."""

    TEXTS = [
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

    def __init__(self, tokenizer, num_samples=64, seq_length=64):
        self.samples = []
        for i in range(num_samples):
            text = self.TEXTS[i % len(self.TEXTS)] + tokenizer.eos_token
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


def main():
    print("=" * 70)
    print("Integration Test: Parallel Adapters")
    print("=" * 70)

    # ========================================================================
    # Part 0 – Standalone adapter modules
    # ========================================================================
    print("\n" + "-" * 70)
    print("Part 0: Standalone adapter modules")
    print("-" * 70)

    print("\n[0.1] ParallelBottleneckAdapter(64 -> 16 -> 64)")
    bottleneck = ParallelBottleneckAdapter(hidden_dim=64, bottleneck_dim=16)
    bottleneck.eval()
    test_input = torch.randn(2, 8, 64)
    with torch.no_grad():
        bottleneck_out = bottleneck(test_input)
    assert bottleneck_out.shape == test_input.shape
    assert not torch.isnan(bottleneck_out).any()
    assert not torch.isinf(bottleneck_out).any()
    with torch.no_grad():
        gate_value = torch.sigmoid(bottleneck.gate(test_input))
        assert torch.allclose(gate_value, torch.full_like(gate_value, 0.5), atol=1e-6)
        assert bottleneck_out.abs().max().item() < 1.0
    print(f"    -> output shape: {tuple(bottleneck_out.shape)}")
    print(f"    -> gate value at init: 0.5 (expected)")
    print(f"    -> max |output| at init: {bottleneck_out.abs().max().item():.6f}")

    print("\n[0.2] ParallelAttentionAdapter(64, num_heads=4)")
    attn = ParallelAttentionAdapter(hidden_dim=64, num_heads=ATTN_HEADS)
    attn.eval()
    with torch.no_grad():
        attn_out = attn(test_input)
    assert attn_out.shape == test_input.shape
    assert not torch.isnan(attn_out).any()
    assert not torch.isinf(attn_out).any()
    print(f"    -> output shape: {tuple(attn_out.shape)}")
    print(f"    -> head_dim: {attn.head_dim}")

    n_params = sum(p.numel() for p in bottleneck.parameters())
    n_params_attn = sum(p.numel() for p in attn.parameters())
    print(f"    -> bottleneck params: {n_params:,}")
    print(f"    -> attention params: {n_params_attn:,}")
    print("    -> Part 0 PASS")

    # ========================================================================
    # Setup: tokenizer + dataset
    # ========================================================================
    print("\n" + "-" * 70)
    print("Setup: tokenizer + toy dataset")
    print("-" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset = ToyTextDataset(tokenizer, num_samples=64)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    print(f"    -> model: {MODEL_NAME}")
    print(f"    -> train batches: {len(train_loader)}")

    # ========================================================================
    # Part A – Full bottleneck adapter expansion
    # ========================================================================
    print("\n" + "-" * 70)
    print("Part A: Full bottleneck adapter expansion")
    print("-" * 70)

    print("\n[A1] Baseline generation (original model)")
    orig_wrapper = ExpandableModel.from_pretrained(MODEL_NAME, dtype=torch.float32)
    orig_model = orig_wrapper.get_model()
    n_layers = orig_wrapper.config.num_hidden_layers
    hidden_size = orig_wrapper.config.hidden_size
    print(f"    -> hidden_size: {hidden_size}, num_hidden_layers: {n_layers}")
    baseline_text = generate_text(orig_model, tokenizer, PROMPT)
    print(f"    -> Baseline: '{baseline_text}'")

    print("\n[A2] Expanding with bottleneck adapters (every layer)")
    a_wrapper = ExpandableModel.from_pretrained(MODEL_NAME, dtype=torch.float32)
    a_model = a_wrapper.get_model()
    a_wrapper.expand(
        ParallelAdapterExpansion(
            adapter_type="bottleneck",
            bottleneck_dim=BOTTLENECK_DIM,
            initialization="zero",
        )
    )
    adapter_counts = [len(getattr(layer, "cambium_adapters", [])) for layer in a_model.model.layers]
    assert all(c == 1 for c in adapter_counts)
    assert all(
        isinstance(a_model.model.layers[i].cambium_adapters[0], ParallelBottleneckAdapter)
        for i in range(n_layers)
    )
    print(f"    -> attached {sum(adapter_counts)} bottleneck adapters ({n_layers} layers x 1)")

    print("\n[A4] Forward + validation")
    a_model.eval()
    with torch.no_grad():
        dummy = torch.randint(0, a_wrapper.config.vocab_size, (1, 10))
        out_logits = a_model(dummy).logits
    assert not torch.isnan(out_logits).any()
    assert not torch.isinf(out_logits).any()
    results = validate_model_output(a_model, dummy)
    print(f"    -> validate_model_output: success={results['success']}")

    a_text_before = generate_text(a_model, tokenizer, PROMPT)
    print(f"    -> Expanded (before train): '{a_text_before}'")

    print("\n[A6] Training (frozen base, adapters only)")
    a_wrapper.freeze_original()
    a_wrapper.freezing_manager.unfreeze_by_pattern(r"cambium_adapters")
    info = a_wrapper.freezing_manager.get_trainable_params()
    assert all("cambium_adapters" in n for n in info["trainable_names"])
    assert info["percent_trainable"] < 2.0
    a_wrapper.print_trainable()

    trainer_a = StagedTrainer(a_wrapper)
    trainer_a.add_phase(name="adapter_training", freeze=None, lr=1e-3, epochs=10)
    trainer_a.train(train_loader)

    a_text_after = generate_text(a_model, tokenizer, PROMPT)
    print(f"    -> Expanded (after train):  '{a_text_after}'")

    print("\n[A7] Save and reload")
    a_wrapper.save_expanded("test-parallel-adapter-a")
    reloaded = ExpandableModel.load_expanded("test-parallel-adapter-a")
    reloaded_model = reloaded.get_model()
    assert reloaded.is_expanded
    assert len(reloaded.expansions) == len(a_wrapper.expansions)
    assert all(
        hasattr(reloaded_model.model.layers[i], "cambium_adapters")
        and len(reloaded_model.model.layers[i].cambium_adapters) == 1
        for i in range(n_layers)
    )
    assert all(
        isinstance(reloaded_model.model.layers[i].cambium_adapters[0], ParallelBottleneckAdapter)
        for i in range(n_layers)
    )
    reloaded_text = generate_text(reloaded_model, tokenizer, PROMPT)
    print(f"    -> Reload OK ({n_layers} adapters re-attached)")
    print(f"    -> Reloaded generation: '{reloaded_text}'")

    with torch.no_grad():
        reloaded_logits = reloaded_model(dummy).logits
        in_mem_logits = a_model(dummy).logits
        max_reload_diff = (reloaded_logits - in_mem_logits).abs().max().item()
        print(f"    -> max |in-memory - reloaded| logits: {max_reload_diff:.6e}")
        assert max_reload_diff < 1e-5

    # ========================================================================
    # Part B – Targeted layers
    # ========================================================================
    print("\n" + "-" * 70)
    print("Part B: Targeted bottleneck adapters (last 2 layers only)")
    print("-" * 70)

    target_layers = [n_layers - 2, n_layers - 1]
    print(f"\n[B1] Expanding with target_layers={target_layers}")
    b_wrapper = ExpandableModel.from_pretrained(MODEL_NAME, dtype=torch.float32)
    b_model = b_wrapper.get_model()
    b_wrapper.expand(
        ParallelAdapterExpansion(
            adapter_type="bottleneck",
            bottleneck_dim=BOTTLENECK_DIM,
            initialization="zero",
            target_layers=target_layers,
        )
    )
    b_counts = [len(getattr(layer, "cambium_adapters", [])) for layer in b_model.model.layers]
    for i in range(n_layers):
        expected = 1 if i in target_layers else 0
        assert b_counts[i] == expected
    print(f"    -> adapter counts per layer: {b_counts}")
    print(f"    -> total adapters: {sum(b_counts)} (expected {len(target_layers)})")

    print("\n[B2] Training (frozen base, adapters only)")
    b_wrapper.freeze_original()
    b_wrapper.freezing_manager.unfreeze_by_pattern(r"cambium_adapters")
    info_b = b_wrapper.freezing_manager.get_trainable_params()
    assert all("cambium_adapters" in n for n in info_b["trainable_names"])

    trainer_b = StagedTrainer(b_wrapper)
    trainer_b.add_phase(name="targeted_adapter_training", freeze=None, lr=1e-3, epochs=10)
    trainer_b.train(train_loader)

    b_text_after = generate_text(b_model, tokenizer, PROMPT)
    print(f"    -> Targeted (after train):  '{b_text_after}'")

    # ========================================================================
    # Part C – Attention adapter
    # ========================================================================
    print("\n" + "-" * 70)
    print("Part C: Attention adapter (every layer)")
    print("-" * 70)

    assert hidden_size % ATTN_HEADS == 0
    print(f"\n[C1] Expanding with attention adapters (num_heads={ATTN_HEADS})")
    c_wrapper = ExpandableModel.from_pretrained(MODEL_NAME, dtype=torch.float32)
    c_model = c_wrapper.get_model()
    c_wrapper.expand(
        ParallelAdapterExpansion(
            adapter_type="attention",
            num_heads=ATTN_HEADS,
            initialization="zero",
        )
    )
    c_counts = [len(getattr(layer, "cambium_adapters", [])) for layer in c_model.model.layers]
    assert all(c == 1 for c in c_counts)
    assert all(
        isinstance(c_model.model.layers[i].cambium_adapters[0], ParallelAttentionAdapter)
        for i in range(n_layers)
    )
    print(f"    -> attached {sum(c_counts)} attention adapters ({n_layers} layers x 1)")

    print("\n[C2] Forward + validation")
    c_model.eval()
    with torch.no_grad():
        out_logits_c = c_model(dummy).logits
    assert not torch.isnan(out_logits_c).any()
    assert not torch.isinf(out_logits_c).any()
    print(f"    -> output logits shape: {tuple(out_logits_c.shape)}")
    print(
        f"    -> validate_model_output: success={validate_model_output(c_model, dummy)['success']}"
    )

    c_text_before = generate_text(c_model, tokenizer, PROMPT)
    print(f"    -> Attention (before train): '{c_text_before}'")

    print("\n[C3] Training (frozen base, attention adapters only)")
    c_wrapper.freeze_original()
    c_wrapper.freezing_manager.unfreeze_by_pattern(r"cambium_adapters")
    info_c = c_wrapper.freezing_manager.get_trainable_params()
    assert all("cambium_adapters" in n for n in info_c["trainable_names"])

    trainer_c = StagedTrainer(c_wrapper)
    trainer_c.add_phase(name="attention_adapter_training", freeze=None, lr=5e-4, epochs=10)
    trainer_c.train(train_loader)

    c_text_after = generate_text(c_model, tokenizer, PROMPT)
    print(f"    -> Attention (after train):  '{c_text_after}'")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("Generation Comparison")
    print("=" * 70)
    print(f"\nPrompt: '{PROMPT}'")
    print(f"\n[Original (baseline)]")
    print(f"  {baseline_text}")
    print(f"\n[Bottleneck adapters (all layers)]")
    print(f"  {a_text_after}")
    print(f"\n[Bottleneck adapters (targeted)]")
    print(f"  {b_text_after}")
    print(f"\n[Attention adapters (all layers)]")
    print(f"  {c_text_after}")

    print("\n" + "=" * 70)
    print("Integration Parallel Adapter Test: PASSED")
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
