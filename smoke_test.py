"""Manual smoke test for Cambium library."""
import torch
from transformers import AutoModelForCausalLM

from cambium import ExpandableModel, InterleavedExpansion

print("=" * 50)
print("Manual Smoke Test: Cambium Library")
print("=" * 50)

# 1. Load a tiny Llama-based model (supported by InterleavedExpansion)
model_name = "HuggingFaceTB/SmolLM2-135M"  # Llama-like, small, fast
print(f"\n[1] Loading model: {model_name}")
wrapper = ExpandableModel.from_pretrained(model_name, dtype=torch.float32)
print(f"    -> Loaded: {wrapper}")

# 2. Check layers before expansion
original_layers = wrapper.config.num_hidden_layers
print(f"\n[2] Original layer count: {original_layers}")

# 3. Expand with interleaved strategy
print(f"\n[3] Applying InterleavedExpansion (+2 layers)")
expander = InterleavedExpansion(
    num_layers=2,
    initialization="identity",
    # default layer_attribute="model.layers" works for Llama/SmolLM
)
wrapper.expand(expander)
print(f"    -> Expanded: {wrapper}")

# 4. Verify layer count grew
new_layers = wrapper.config.num_hidden_layers
assert new_layers == original_layers + 2
print(f"\n[4] New layer count: {new_layers} (expected {original_layers + 2}) ✓")

# 5. Freeze original and inspect trainable params
print(f"\n[5] Freezing original layers...")
wrapper.freeze_original()
wrapper.print_trainable()

# 6. Forward pass sanity check
print(f"\n[6] Forward pass sanity check")
model = wrapper.get_model()
inputs = torch.randint(0, wrapper.config.vocab_size, (1, 10))
with torch.no_grad():
    out = model(inputs)
print(f"    -> Output logits shape: {out.logits.shape} ✓")

# 7. Save / reload round-trip
print(f"\n[7] Save & reload round-trip")
wrapper.save_expanded("./test-expanded")
reloaded = ExpandableModel.load_expanded("./test-expanded")
print(f"    -> Reloaded: {reloaded}")

# 8. Validation
print(f"\n[8] Validation")
report = wrapper.validate()
print(f"    -> Validation keys: {list(report.keys())}")

print("\n" + "=" * 50)
print("All manual smoke tests passed!")
print("=" * 50)
