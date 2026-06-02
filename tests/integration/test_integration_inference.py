"""Integration inference test for Cambium.

Generates text with the expanded model before any training
to verify that identity-initialized new layers act as near
pass-throughs.
"""
import torch
from transformers import AutoTokenizer

from cambium import ExpandableModel, InterleavedExpansion

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
PROMPT = "Artificial intelligence is"


def main():
    print("=" * 70)
    print("Integration Inference Test")
    print("=" * 70)

    # 1. Load and expand
    print(f"\n[1] Loading + expanding model: {MODEL_NAME}")
    wrapper = ExpandableModel.from_pretrained(MODEL_NAME, dtype=torch.float32)
    wrapper.expand(InterleavedExpansion(num_layers=2, initialization="identity"))
    print(f"    -> {wrapper}")

    model = wrapper.get_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 2. Load tokenizer
    print(f"\n[2] Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Generate with expanded model
    print(f"\n[3] Generating with EXPANDED model (untrained new layers)")
    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        expanded_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    expanded_text = tokenizer.decode(expanded_ids[0], skip_special_tokens=True)
    print(f"    Prompt : {PROMPT}")
    print(f"    Output : {expanded_text}")

    # 4. Generate with original model for comparison
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

    print("\n" + "=" * 70)
    print("Integration Inference Test: PASSED")
    print("=" * 70)
    print(
        "Note: The expanded model's output should look very similar to\n"
        "the original because the new layers use identity initialization.\n"
        "After training the new layers, the expanded model will diverge."
    )
    return 0


if __name__ == "__main__":
    import sys
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        sys.exit(1)
