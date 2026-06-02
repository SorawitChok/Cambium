"""Test script for cambium.utils.memory.estimate_memory_usage().

Validates that the estimator returns sensible values for a real model and
that each configuration (dtype, gradient checkpointing, batch size)
behaves as expected.
"""
import torch
from transformers import AutoModelForCausalLM

from cambium.utils.memory import estimate_memory_usage, get_memory_profile

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"


def test_estimate_smollm():
    """Test memory estimate on SmolLM2-135M (30 layers, hidden_size=576)."""
    print("Loading model for estimation...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)

    est = estimate_memory_usage(
        model,
        batch_size=4,
        sequence_length=512,
        dtype="fp16",
        gradient_checkpointing=False,
    )

    print("\nSmolLM2-135M estimate (bs=4, seq=512, fp16, no grad-checkpoint):")
    for k, v in est.items():
        print(f"  {k}: {v} GB")

    # Sanity checks
    total_params = sum(p.numel() for p in model.parameters())
    expected_model_weights = total_params * 2 / (1024 ** 3)
    assert est["model_weights_gb"] == round(expected_model_weights, 2), (
        f"model_weights mismatch: expected {round(expected_model_weights, 2)}, got {est['model_weights_gb']}"
    )
    assert est["total_gb"] > 0, "total_gb must be positive"
    assert est["recommended_gb"] == round(est["total_gb"] * 1.2, 2), "recommended_gb should be total * 1.2"

    # Activations: 4 * 512 * 576 * 30 * 4 / 1024^3 ≈ 0.13 GB
    assert est["activations_gb"] > 0, "activations must be positive"
    print("  -> SmolLM estimate OK")


def test_dtype_scaling():
    """FP32 uses 4 bytes per param, FP16/BF16 use 2."""
    print("\n--- dtype scaling ---")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)

    fp32 = estimate_memory_usage(model, dtype="fp32")
    fp16 = estimate_memory_usage(model, dtype="fp16")
    bf16 = estimate_memory_usage(model, dtype="bf16")
    unknown = estimate_memory_usage(model, dtype="int8")  # falls back to 2

    # fp32 model weights should be exactly 2x fp16
    assert fp32["model_weights_gb"] == fp16["model_weights_gb"] * 2, (
        f"FP32 weights {fp32['model_weights_gb']} != 2x FP16 {fp16['model_weights_gb']}"
    )
    assert fp16["model_weights_gb"] == bf16["model_weights_gb"], "FP16 and BF16 should be equal"
    assert unknown["model_weights_gb"] == bf16["model_weights_gb"], "Unknown dtype falls back to 2 bytes"
    print(f"  fp32: {fp32['model_weights_gb']} GB")
    print(f"  fp16: {fp16['model_weights_gb']} GB")
    print(f"  bf16: {bf16['model_weights_gb']} GB")
    print("  -> dtype scaling OK")


def test_gradient_checkpointing_saves_memory():
    """Gradient checkpointing should reduce activation memory."""
    print("\n--- gradient checkpointing ---")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)

    no_gc = estimate_memory_usage(model, gradient_checkpointing=False)
    with_gc = estimate_memory_usage(model, gradient_checkpointing=True)

    assert with_gc["activations_gb"] < no_gc["activations_gb"], (
        f"GC should reduce activations: {with_gc['activations_gb']} >= {no_gc['activations_gb']}"
    )
    assert with_gc["total_gb"] < no_gc["total_gb"], "GC should reduce total estimate"
    print(f"  no GC:  activations={no_gc['activations_gb']} GB, total={no_gc['total_gb']} GB")
    print(f"  with GC: activations={with_gc['activations_gb']} GB, total={with_gc['total_gb']} GB")
    print("  -> gradient checkpointing reduces memory OK")


def test_batch_size_scaling():
    """Larger batch size increases activation memory."""
    print("\n--- batch size scaling ---")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)

    bs1 = estimate_memory_usage(model, batch_size=1)
    bs8 = estimate_memory_usage(model, batch_size=8)

    # Activations scale linearly, but values are rounded to 2 decimals.
    assert abs(bs8["activations_gb"] - bs1["activations_gb"] * 8) <= 0.05, (
        f"Activations should scale linearly with batch size: {bs8['activations_gb']} !≈ 8 * {bs1['activations_gb']}"
    )
    # Model weights and optimizer states are independent of batch size
    assert bs8["model_weights_gb"] == bs1["model_weights_gb"], "Model weights should not depend on batch size"
    assert bs8["optimizer_states_gb"] == bs1["optimizer_states_gb"], "Optimizer states should not depend on batch size"
    print(f"  bs=1:  activations={bs1['activations_gb']} GB")
    print(f"  bs=8:  activations={bs8['activations_gb']} GB")
    print("  -> batch size scaling OK")


def test_no_config_fallback():
    """A plain nn.Module without config should use sensible defaults."""
    print("\n--- no-config fallback ---")
    dummy = torch.nn.Linear(100, 100)
    est = estimate_memory_usage(dummy, batch_size=1, sequence_length=128)

    # Should not crash; uses hidden_size=2048, num_layers=24 defaults
    assert est["total_gb"] > 0
    print(f"  plain nn.Linear estimate: {est['total_gb']} GB")
    print("  -> fallback OK")


def test_get_memory_profile():
    """GPU memory profile should return dict with cuda_available key."""
    print("\n--- get_memory_profile ---")
    profile = get_memory_profile()
    assert "cuda_available" in profile
    if not profile["cuda_available"]:
        print("  CUDA not available (expected on CPU-only machine)")
    else:
        assert "allocated_gb" in profile
        assert profile["allocated_gb"] >= 0
        print(f"  allocated: {profile['allocated_gb']} GB")
    print("  -> memory profile OK")


def main():
    print("=" * 60)
    print("Memory Estimation Test")
    print("=" * 60)

    test_estimate_smollm()
    test_dtype_scaling()
    test_gradient_checkpointing_saves_memory()
    test_batch_size_scaling()
    test_no_config_fallback()
    test_get_memory_profile()

    print("\n" + "=" * 60)
    print("All memory estimation tests PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
