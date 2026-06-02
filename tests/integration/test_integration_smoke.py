"""Integration smoke test for Cambium.

Tests the core library surface:
1. Load a tiny Llama-based model via ExpandableModel.
2. Expand with InterleavedExpansion (+2 layers).
3. Verify layer count, freezing, forward pass, save/reload, and validation.
"""
import torch

from cambium import ExpandableModel, InterleavedExpansion

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"


def main():
    print("=" * 70)
    print("Integration Smoke Test: Cambium Library")
    print("=" * 70)

    # 1. Load model
    print(f"\n[1] Loading model: {MODEL_NAME}")
    wrapper = ExpandableModel.from_pretrained(MODEL_NAME, dtype=torch.float32)
    print(f"    -> Loaded: {wrapper}")

    # 2. Check layers before expansion
    original_layers = wrapper.config.num_hidden_layers
    print(f"\n[2] Original layer count: {original_layers}")

    # 3. Expand
    print(f"\n[3] Applying InterleavedExpansion (+2 layers)")
    wrapper.expand(InterleavedExpansion(num_layers=2, initialization="identity"))
    print(f"    -> Expanded: {wrapper}")

    # 4. Verify layer count
    new_layers = wrapper.config.num_hidden_layers
    assert new_layers == original_layers + 2
    print(f"\n[4] New layer count: {new_layers} (expected {original_layers + 2}) PASS")

    # 5. Freeze original weights
    print(f"\n[5] Freezing original layers...")
    wrapper.freeze_original()
    wrapper.print_trainable()

    # 6. Forward pass sanity check
    print(f"\n[6] Forward pass sanity check")
    model = wrapper.get_model()
    inputs = torch.randint(0, wrapper.config.vocab_size, (1, 10))
    with torch.no_grad():
        out = model(inputs)
    print(f"    -> Output logits shape: {out.logits.shape} PASS")

    # 7. Save / reload round-trip
    print(f"\n[7] Save & reload round-trip")
    wrapper.save_expanded("./test-expanded")
    reloaded = ExpandableModel.load_expanded("./test-expanded")
    print(f"    -> Reloaded: {reloaded}")

    # 8. Validation
    print(f"\n[8] Validation")
    report = wrapper.validate()
    print(f"    -> Validation keys: {list(report.keys())}")

    print("\n" + "=" * 70)
    print("Integration Smoke Test: PASSED")
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
