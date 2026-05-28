"""Manual test for Cambium width expansion with generation comparison.

This script:
1. Loads the smallest non-GQA model (JackFram/llama-160m).
2. Compares generation before / after width expansion / after training.
3. Uses StagedTrainer for training with library freezing utilities.
4. Freezes embeddings and LM head during training to preserve pretrained
   token representations, which also avoids an inf-gradient issue in
   embed_tokens on PyTorch 2.12.0+cpu.
5. Uses our ToyTextDataset from train_warmup.py.
"""
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from cambium import ExpandableModel
from cambium.strategies import WidthExpansion
from cambium.training.staged_trainer import StagedTrainer
from cambium.utils.validation import validate_model_output

MODEL_NAME = "JackFram/llama-160m"
MULTIPLIER = 1.25
PROMPT = "The future of artificial intelligence is"
GEN_KWARGS = {"max_new_tokens": 30, "do_sample": False}


class ToyTextDataset(Dataset):
    """Tiny in-memory dataset for quick training demos."""

    def __init__(self, tokenizer, num_samples=64, seq_length=64):
        self.samples = []
        texts = [
            "The proliferation of large language models has precipitated a paradigm shift in how we conceptualize intelligence, blurring the once-distinct boundary between statistical pattern matching and genuine cognitive reasoning.",
            "In the philosophy of mind, the hard problem of consciousness asks why subjective experience arises from physical processes, a question that remains stubbornly resistant to reductionist explanation despite centuries of inquiry.",
            "Contemporary geopolitical dynamics are increasingly shaped by the asymmetric distribution of computational resources, wherein nation-states and corporate entities that control advanced semiconductor fabrication exert disproportionate influence over global information ecosystems.",
            "The second law of thermodynamics, while often misconstrued as a principle of universal decay, more accurately describes the statistical tendency of isolated systems to evolve toward macrostates with the greatest number of corresponding microstates.",
            "Epistemologically, Bayesian inference offers a coherent framework for updating beliefs in light of new evidence, yet its practical application demands careful scrutiny of prior assumptions that may encode unrecognized biases.",
            "During the European Renaissance, the recovery of classical Greek and Arabic manuscripts catalyzed intellectual movements that fundamentally reconceptualized humanity's relationship to nature, authority, and the limits of knowledge.",
            "Climate feedback mechanisms, including albedo reduction from melting ice and methane release from thawing permafrost, introduce nonlinearities into atmospheric models that complicate precise long-term predictions.",
            "The architecture of transformer-based neural networks leverages self-attention mechanisms to compute contextualized representations, enabling the modeling of long-range dependencies that recurrent architectures struggle to capture efficiently.",
            "In constitutional democracies, the tension between majoritarian impulses and minority protections necessitates institutional safeguards, such as judicial review and supermajoritarian thresholds, that deliberately slow the pace of political change.",
            "Emergent phenomena in complex systems, from ant colonies to financial markets, demonstrate how localized interactions among simple agents can generate collective behaviors that are not obviously derivable from the properties of individual components.",
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


# ==============================================================================
# Main test
# ==============================================================================
print("=" * 60)
print("Manual Test: Width Expansion")
print("=" * 60)

# Shared tokenizer
print("\n[0] Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Shared dataset / dataloader
print("\n[0] Building toy dataset")
train_data = ToyTextDataset(tokenizer, num_samples=64)
train_loader = DataLoader(train_data, batch_size=4, shuffle=True)

# ==============================================================================
# Part A – Fully expanded model
# ==============================================================================
print("\n" + "-" * 60)
print("A) Full width expansion")
print("-" * 60)

# A1. Original model (baseline)
print("\n[A1] Loading original model")
orig_wrapper = ExpandableModel.from_pretrained(MODEL_NAME, dtype=torch.float32)
orig_model = orig_wrapper.get_model()
print(f"    -> hidden_size: {orig_wrapper.config.hidden_size}")
orig_text = generate_text(orig_model, tokenizer, PROMPT)
print(f"    -> Original: '{orig_text}'")

# A2. Expand
print(f"\n[A2] Expanding width (x{MULTIPLIER})")
exp_wrapper = ExpandableModel.from_pretrained(MODEL_NAME, dtype=torch.float32)
exp_model = exp_wrapper.get_model()
exp_wrapper.expand(WidthExpansion(hidden_dim_multiplier=MULTIPLIER, initialization="zero"))
print(f"    -> hidden_size: {exp_wrapper.config.hidden_size}")
exp_text_before = generate_text(exp_model, tokenizer, PROMPT)
print(f"    -> Expanded (before train): '{exp_text_before}'")

# A3. Validate forward pass
print("\n[A3] Forward / validation")
with torch.no_grad():
    dummy = torch.randint(0, exp_wrapper.config.vocab_size, (1, 10))
    out_logits = exp_model(dummy).logits
assert not torch.isnan(out_logits).any(), "NaN in output"
assert not torch.isinf(out_logits).any(), "Inf in output"
results = validate_model_output(exp_model, dummy)
print(f"    -> validate_model_output: success={results['success']}")

# A4. Train full model with StagedTrainer
print("\n[A4] Training fully expanded model")
# Freeze original weights and embeddings/LM head to preserve pretrained
# token representations while training only new expanded layer dims.
exp_wrapper.freeze_original()
exp_wrapper.freezing_manager.freeze_embeddings()
exp_wrapper.print_trainable()

trainer_full = StagedTrainer(exp_wrapper)
trainer_full.add_phase(
    name="full_expansion",
    freeze=None,  # Keep our manual freeze config
    lr=1e-5,
    epochs=10,
)
trainer_full.train(train_loader)
exp_text_after = generate_text(exp_model, tokenizer, PROMPT)
print(f"    -> Expanded (after train): '{exp_text_after}'")

# A5. Save / reload
print("\n[A5] Save and reload")
exp_wrapper.save_expanded("test-width-expanded")
reloaded = ExpandableModel.from_pretrained("test-width-expanded")
assert reloaded.get_model().config.hidden_size == exp_wrapper.config.hidden_size
print("    -> Reload OK")

# ==============================================================================
# Part B – Selective expansion
# ==============================================================================
print("\n" + "-" * 60)
print("B) Selective layer expansion (layers 4-7 MLP only)")
print("-" * 60)

# B1. Expand selectively
print("\n[B1] Expanding selectively")
sel_wrapper = ExpandableModel.from_pretrained(MODEL_NAME, dtype=torch.float32)
sel_model = sel_wrapper.get_model()
sel_wrapper.expand(
    WidthExpansion(
        hidden_dim_multiplier=MULTIPLIER,
        initialization="zero",
        layer_indices=list(range(4, 8)),
        expand_attention=False,
    )
)
for i in [0, 4, 7, 11]:
    layer = sel_model.model.layers[i]
    print(f"    layer {i}: up_proj.out={layer.mlp.up_proj.weight.shape[0]}")
sel_text_before = generate_text(sel_model, tokenizer, PROMPT)
print(f"    -> Selective (before train): '{sel_text_before}'")

# B2. Train selective model (only new params trainable)
print("\n[B2] Training selectively expanded model")
sel_wrapper.freeze_original()
sel_wrapper.freezing_manager.freeze_embeddings()
sel_wrapper.print_trainable()

trainer_sel = StagedTrainer(sel_wrapper)
trainer_sel.add_phase(
    name="selective_expansion",
    freeze=None,  # Keep our manual freeze config
    lr=1e-5,
    epochs=10,
)
trainer_sel.train(train_loader)
sel_text_after = generate_text(sel_model, tokenizer, PROMPT)
print(f"    -> Selective (after train): '{sel_text_after}'")

# ==============================================================================
# Summary
# ==============================================================================
print("\n" + "=" * 60)
print("Generation Comparison")
print("=" * 60)
print(f"\nPrompt: '{PROMPT}'")
print(f"\n[Original (baseline)]")
print(f"  {orig_text}")
print(f"\n[Full expansion – before training]")
print(f"  {exp_text_before}")
print(f"\n[Full expansion – after training]")
print(f"  {exp_text_after}")
print(f"\n[Selective expansion (layers 4-7) – before training]")
print(f"  {sel_text_before}")
print(f"\n[Selective expansion (layers 4-7) – after training]")
print(f"  {sel_text_after}")

print("\n" + "=" * 60)
print("Width Expansion Test: PASSED")
print("=" * 60)
