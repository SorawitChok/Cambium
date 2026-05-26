"""Manual inference test for Cambium expanded model.

Generates text with the expanded model before any training
so you can see that identity-initialized new layers act as
near pass-throughs.
"""
import torch
from transformers import AutoTokenizer

from cambium import ExpandableModel, InterleavedExpansion

print("=" * 50)
print("Manual Inference Test")
print("=" * 50)

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
PROMPT = "Artificial intelligence is"

# 1. Load model and expand
print(f"\n[1] Loading + expanding model: {MODEL_NAME}")
wrapper = ExpandableModel.from_pretrained(MODEL_NAME, dtype=torch.float32)
wrapper.expand(InterleavedExpansion(num_layers=2, initialization="identity"))
print(f"    -> {wrapper}")

# 2. Grab the underlying model
model = wrapper.get_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 3. Load tokenizer
print(f"\n[2] Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 4. Generate with expanded model (untrained new layers)
print(f"\n[3] Generating with EXPANDED model (new layers untrained)")
inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
with torch.no_grad():
    expanded_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,  # greedy for reproducibility
        pad_token_id=tokenizer.pad_token_id,
    )
expanded_text = tokenizer.decode(expanded_ids[0], skip_special_tokens=True)
print(f"    Prompt : {PROMPT}")
print(f"    Output : {expanded_text}")

# 5. For comparison, generate with original (unexpanded) model
print(f"\n[4] Generating with ORIGINAL model for comparison")
original_wrapper = ExpandableModel.from_pretrained(MODEL_NAME, dtype=torch.float32)
original_model = original_wrapper.get_model()
original_model.to(device)
original_model.eval()
with torch.no_grad():
    original_ids = original_model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
original_text = tokenizer.decode(original_ids[0], skip_special_tokens=True)
print(f"    Prompt : {PROMPT}")
print(f"    Output : {original_text}")

# 6. Quick sanity: the outputs should be very close because identity init
# makes new layers behave like near pass-throughs.
print("\n" + "=" * 50)
print("Inference test complete.")
print("=" * 50)
print(
    "Note: The expanded model's output should look very similar to\n"
    "the original because the new layers use identity initialization.\n"
    "After training the new layers, the expanded model will diverge."
)
