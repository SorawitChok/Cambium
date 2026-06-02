"""Integration test for cambium.utils.memory.estimate_memory_usage().

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
    print("[1] Loading model for estimation...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)

    est = estimate_memory_usage(
        model,
        batch_size=4,
        sequence_length=512,
        dtype="fp16",
        gradient_checkpointing=False,
    )

    print("\n  SmolLM2-135M estimate (bs=4, seq=512, fp16, no grad-checkpoint):")
    for k, v in est.items():
        print(f"    {k}: {v} GB")

    total_params = sum(p.numel() for p in model.parameters())
    expected_model_weights = total_params * 2 / (1024 ** 3)
    assert est["model_weights_gb"] == round(expected_model_weights, 2), (
        f"model_weights mismatch: expected {round(expected_model_weights, 2)}, got {est['model_weights_gb']}"
    )
    assert est["total_gb"] > 0, "total_gb must be positive"
    assert est["recommended_gb"] == round(est["total_gb"] * 1.2, 2), "recommended_gb should be total * 1.2"
    assert est["activations_gb"] > 0, "activations must be positive"
    print("  -> SmolLM estimate PASS")


def test_dtype_scaling():
    """FP32 uses 4 bytes per param, FP16/BF16 use 2."""
    print("\n[2] dtype scaling")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)

    fp32 = estimate_memory_usage(model, dtype="fp32")
    fp16 = estimate_memory_usage(model, dtype="fp16")
    bf16 = estimate_memory_usage(model, dtype="bf16")
    unknown = estimate_memory_usage(model, dtype="int8")  # falls back to 2

    assert fp32["model_weights_gb"] == fp16["model_weights_gb"] * 2, (
        f"FP32 weights {fp32['model_weights_gb']} != 2x FP16 {fp16['model_weights_gb']}"
    )
    assert fp16["model_weights_gb"] == bf16["model_weights_gb"], "FP16 and BF16 should be equal"
    assert unknown["model_weights_gb"] == bf16["model_weights_gb"], "Unknown dtype falls back to 2 bytes"
    print(f"    fp32: {fp32['model_weights_gb']} GB")
    print(f"    fp16: {fp16['model_weights_gb']} GB")
    print(f"    bf16: {bf16['model_weights_gb']} GB")
    print("  -> dtype scaling PASS")


def test_gradient_checkpointing_saves_memory():
    """Gradient checkpointing should reduce activation memory."""
    print("\n[3] gradient checkpointing")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)

    no_gc = estimate_memory_usage(model, gradient_checkpointing=False)
    with_gc = estimate_memory_usage(model, gradient_checkpointing=True)

    assert with_gc["activations_gb"] < no_gc["activations_gb"], (
        f"GC should reduce activations: {with_gc['activations_gb']} >= {no_gc['activations_gb']}"
    )
    assert with_gc["total_gb"] < no_gc["total_gb"], "GC should reduce total estimate"
    print(f"    no GC:  activations={no_gc['activations_gb']} GB, total={no_gc['total_gb']} GB")
    print(f"    with GC: activations={with_gc['activations_gb']} GB, total={with_gc['total_gb']} GB")
    print("  -> gradient checkpointing PASS")


def test_batch_size_scaling():
    """Larger batch size increases activation memory."""
    print("\n[4] batch size scaling")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)

    bs1 = estimate_memory_usage(model, batch_size=1)
    bs8 = estimate_memory_usage(model, batch_size=8)

    assert abs(bs8["activations_gb"] - bs1["activations_gb"] * 8) <= 0.05, (
        f"Activations should scale linearly: {bs8['activations_gb']} !≈ 8 * {bs1['activations_gb']}"
    )
    assert bs8["model_weights_gb"] == bs1["model_weights_gb"], "Model weights should not depend on batch size"
    assert bs8["optimizer_states_gb"] == bs1["optimizer_states_gb"], "Optimizer states should not depend on batch size"
    print(f"    bs=1:  activations={bs1['activations_gb']} GB")
    print(f"    bs=8:  activations={bs8['activations_gb']} GB")
    print("  -> batch size scaling PASS")


def test_no_config_fallback():
    """A plain nn.Module without config should use sensible defaults."""
    print("\n[5] no-config fallback")
    dummy = torch.nn.Linear(100, 100)
    est = estimate_memory_usage(dummy, batch_size=1, sequence_length=128)
    assert est["total_gb"] > 0
    print(f"    plain nn.Linear estimate: {est['total_gb']} GB")
    print("  -> fallback PASS")


def test_get_memory_profile():
    """GPU memory profile should return dict with cuda_available key."""
    print("\n[6] get_memory_profile")
    profile = get_memory_profile()
    assert "cuda_available" in profile
    if not profile["cuda_available"]:
        print("    CUDA not available (expected on CPU-only machine)")
    else:
        assert "allocated_gb" in profile
        assert profile["allocated_gb"] >= 0
        print(f"    allocated: {profile['allocated_gb']} GB")
    print("  -> memory profile PASS")


def main():
    print("=" * 70)
    print("Integration Test: Memory Estimation")
    print("=" * 70)

    test_estimate_smollm()
    test_dtype_scaling()
    test_gradient_checkpointing_saves_memory()
    test_batch_size_scaling()
    test_no_config_fallback()
    test_get_memory_profile()

    print("\n" + "=" * 70)
    print("Integration Memory Estimation Test: PASSED")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    import sys
    try:
        sys.exit(main())
    except AssertionError as e:
        print(f"\nASSERTION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        sys.exit(1)
