"""Comprehensive manual test for examples/07_custom_blocks.md.

This script walks through every code block in the custom blocks example
and verifies it actually runs against the current library. Each section
prints a clear PASS / FAIL line so we can see what needs fixing in the
example doc.

Sections covered (matching the example doc):
  A. Using Template Blocks
  B. Defining Custom Blocks - Simple Custom Block
  C. Block with Internal Residual
  D. Block Without CambiumBlock Base
  E. Three Ways to Provide Blocks (block_class / block_factory / block_instances)
  F. Initialization Strategies
  G. Validation (default + skip)
  H. Mixing with Other Strategies
"""

import io
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from cambium import (
    CambiumBlock,
    CustomBlockExpansion,
    ExpandableModel,
    GatedResidualBlock,
    InterleavedExpansion,
    SwiGLUBlock,
)
from cambium.exceptions import BlockValidationError

print("=" * 70)
print("Comprehensive Custom Blocks Example Test (07_custom_blocks.md)")
print("=" * 70)

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"

# Per-section result tracking
results: list[tuple[str, bool, str]] = []


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
        # Echo captured output to the real terminal
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
        tb = traceback.format_exc()
        print(tb)
        results.append((name, False, f"{type(e).__name__}: {e}"))
        print(f">>> RESULT: FAIL  ({name})")


# ============================================================================
# Load model + tokenizer (shared across sections)
# ============================================================================
print()
print("=" * 70)
print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# A separate "load and wrap" step used by sections A/B/H
wrapper = ExpandableModel.from_pretrained(MODEL_NAME, dtype=torch.float32)
print(f"Loaded: {wrapper.config.num_hidden_layers} layers, "
      f"hidden_size={wrapper.config.hidden_size}")


# ============================================================================
# Section A: Using Template Blocks
# ============================================================================
def section_a_template_blocks():
    """Insert SwiGLU MLP blocks (mirror of example doc section A)."""
    # Fresh wrapper so this section is independent
    w = ExpandableModel.from_pretrained(MODEL_NAME, dtype=torch.float32)
    initial = w.config.num_hidden_layers
    print(f"    Original layers: {initial}")

    # Insert SwiGLU MLP blocks (exactly as the example shows)
    w.expand(CustomBlockExpansion(
        block_class=SwiGLUBlock,
        num_layers=4,
        residual_connection=True,  # output = input + block(input)
        initialization="smart",
    ))
    print(f"    Expanded layers: {w.config.num_hidden_layers}")
    assert w.config.num_hidden_layers == initial + 4, w.config.num_hidden_layers
    print(f"    Section A: {initial + 4} layers (4 SwiGLU blocks inserted) OK")


run_section("A. Using Template Blocks (SwiGLUBlock x4)", section_a_template_blocks)


# ============================================================================
# Section B: Defining Custom Blocks - Simple Custom Block
# ============================================================================
def section_b_simple_custom_block():
    """Define MyBlock (CambiumBlock subclass) and insert it."""
    w = ExpandableModel.from_pretrained(MODEL_NAME, dtype=torch.float32)

    class MyBlock(CambiumBlock):
        """My custom architecture block."""

        # Declare what config keys your block needs
        required_config_keys = ["hidden_size"]

        def __init__(self, config, layer_idx=0):
            super().__init__()
            hidden = config.hidden_size
            self.proj = nn.Linear(hidden, hidden)
            self.norm = nn.LayerNorm(hidden)
            self.act = nn.GELU()

        def forward(self, hidden_states, **kwargs):
            # Return a delta - residual_connection=True will add input back
            x = self.proj(self.norm(hidden_states))
            return self.act(x)

    initial_layers = w.config.num_hidden_layers
    w.expand(CustomBlockExpansion(
        block_class=MyBlock,
        num_layers=2,
        positions=[8, 16],
        residual_connection=True,
        initialization="smart",
    ))
    print(f"    {initial_layers} -> {w.config.num_hidden_layers} layers")
    assert w.config.num_hidden_layers == initial_layers + 2


run_section("B. Simple Custom Block (CambiumBlock subclass)", section_b_simple_custom_block)


# ============================================================================
# Section C: Block with Internal Residual
# ============================================================================
def section_c_internal_residual():
    """ResidualMLP that already includes its own residual connection."""
    w = ExpandableModel.from_pretrained(MODEL_NAME, dtype=torch.float32)

    class ResidualMLP(CambiumBlock):
        required_config_keys = ["hidden_size"]

        def __init__(self, config, layer_idx=0):
            super().__init__()
            hidden = config.hidden_size
            self.up = nn.Linear(hidden, hidden * 4)
            self.down = nn.Linear(hidden * 4, hidden)
            self.act = nn.GELU()

        def forward(self, hidden_states, **kwargs):
            # This block already handles residual internally
            return hidden_states + self.down(self.act(self.up(hidden_states)))

    initial = w.config.num_hidden_layers
    w.expand(CustomBlockExpansion(
        block_class=ResidualMLP,
        num_layers=2,
        residual_connection=False,  # Don't double-wrap
    ))
    print(f"    {initial} -> {w.config.num_hidden_layers} layers")
    assert w.config.num_hidden_layers == initial + 2


run_section("C. Block with Internal Residual (residual_connection=False)",
           section_c_internal_residual)


# ============================================================================
# Section D: Block Without CambiumBlock Base
# ============================================================================
def section_d_plain_block():
    """Plain nn.Module subclass - no CambiumBlock parent."""
    w = ExpandableModel.from_pretrained(MODEL_NAME, dtype=torch.float32)

    # NOTE: the example doc shows __init__(self, hidden_size) which DOES
    # NOT work with block_class=... because Cambium calls block(config).
    # We use the signature the library actually requires here, and we
    # also assert the doc's PlainBlock fails (covered in section D-2).
    class PlainBlock(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.linear = nn.Linear(config.hidden_size, config.hidden_size)

        def forward(self, hidden_states, **kwargs):
            return self.linear(hidden_states)

    initial = w.config.num_hidden_layers
    w.expand(CustomBlockExpansion(
        block_class=PlainBlock,
        num_layers=2,
    ))
    print(f"    {initial} -> {w.config.num_hidden_layers} layers (config signature)")
    assert w.config.num_hidden_layers == initial + 2

    # Verify the doc's PlainBlock signature (__init__(self, hidden_size))
    # really does fail - this is the bug we are flagging in the example.
    class DocPlainBlock(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.linear = nn.Linear(hidden_size, hidden_size)

        def forward(self, hidden_states, **kwargs):
            return self.linear(hidden_states)

    raised = False
    try:
        w.expand(CustomBlockExpansion(
            block_class=DocPlainBlock,
            num_layers=1,
        ))
    except TypeError as e:
        raised = True
        print(f"    DocPlainBlock (hidden_size signature) correctly fails: {e}")
    assert raised, "DocPlainBlock should have failed but did not"


run_section("D. Block Without CambiumBlock Base (plain nn.Module)",
           section_d_plain_block)


# ============================================================================
# Section E: Three Ways to Provide Blocks
# ============================================================================
def section_e1_block_class():
    """Mode 1: block_class."""
    w = ExpandableModel.from_pretrained(MODEL_NAME, dtype=torch.float32)

    class MyBlock(CambiumBlock):
        required_config_keys = ["hidden_size"]

        def __init__(self, config, layer_idx=0):
            super().__init__()
            self.proj = nn.Linear(config.hidden_size, config.hidden_size)

        def forward(self, hidden_states, **kwargs):
            return self.proj(hidden_states)

    initial = w.config.num_hidden_layers
    w.expand(CustomBlockExpansion(
        block_class=MyBlock,
        num_layers=4,
        residual_connection=True,
    ))
    print(f"    block_class mode: {initial} -> {w.config.num_hidden_layers} layers")
    assert w.config.num_hidden_layers == initial + 4


def section_e2_block_factory():
    """Mode 2: block_factory - called as factory() with no args."""
    w = ExpandableModel.from_pretrained(MODEL_NAME, dtype=torch.float32)

    class MyBlock(CambiumBlock):
        required_config_keys = ["hidden_size"]

        def __init__(self, config, layer_idx=0):
            super().__init__()
            self.proj = nn.Linear(config.hidden_size, config.hidden_size)

        def forward(self, hidden_states, **kwargs):
            return self.proj(hidden_states)

    config = w.get_model().config

    def my_factory():
        """Create a block with custom logic. NOTE: factory takes no args -
        the library calls it as factory() and uses the model config we
        captured via closure."""
        block = MyBlock(config)
        # Custom setup here
        nn.init.xavier_uniform_(block.proj.weight)
        return block

    initial = w.config.num_hidden_layers
    w.expand(CustomBlockExpansion(
        block_factory=my_factory,
        num_layers=4,
        residual_connection=True,
    ))
    print(f"    block_factory mode: {initial} -> {w.config.num_hidden_layers} layers")
    assert w.config.num_hidden_layers == initial + 4


def section_e3_block_instances():
    """Mode 3: block_instances - pre-created."""
    w = ExpandableModel.from_pretrained(MODEL_NAME, dtype=torch.float32)

    class MyBlock(CambiumBlock):
        required_config_keys = ["hidden_size"]

        def __init__(self, config, layer_idx=0):
            super().__init__()
            self.proj = nn.Linear(config.hidden_size, config.hidden_size)

        def forward(self, hidden_states, **kwargs):
            return self.proj(hidden_states)

    config = w.get_model().config
    blocks = [MyBlock(config, layer_idx=i) for i in range(4)]

    initial = w.config.num_hidden_layers
    w.expand(CustomBlockExpansion(
        block_instances=blocks,
        positions=[4, 8, 12, 16],
        residual_connection=True,
    ))
    print(f"    block_instances mode: {initial} -> {w.config.num_hidden_layers} layers")
    assert w.config.num_hidden_layers == initial + 4


run_section("E1. block_class mode", section_e1_block_class)
run_section("E2. block_factory mode", section_e2_block_factory)
run_section("E3. block_instances mode", section_e3_block_instances)


# ============================================================================
# Section F: Initialization Strategies
# ============================================================================
def section_f_smart_init():
    """Smart init (default) - near-zero for output projections."""
    w = ExpandableModel.from_pretrained(MODEL_NAME, dtype=torch.float32)
    initial = w.config.num_hidden_layers
    w.expand(CustomBlockExpansion(
        block_class=SwiGLUBlock,
        num_layers=2,
        initialization="smart",
    ))
    print(f"    smart init: {initial} -> {w.config.num_hidden_layers} layers")
    assert w.config.num_hidden_layers == initial + 2


def section_f_custom_init():
    """Custom initialization function."""
    w = ExpandableModel.from_pretrained(MODEL_NAME, dtype=torch.float32)

    class MyBlock(CambiumBlock):
        required_config_keys = ["hidden_size"]

        def __init__(self, config, layer_idx=0):
            super().__init__()
            self.proj = nn.Linear(config.hidden_size, config.hidden_size)

        def forward(self, hidden_states, **kwargs):
            return self.proj(hidden_states)

    def my_init(block):
        nn.init.xavier_uniform_(block.proj.weight)
        nn.init.zeros_(block.proj.bias)

    initial = w.config.num_hidden_layers
    w.expand(CustomBlockExpansion(
        block_class=MyBlock,
        num_layers=2,
        initialization="custom",
        custom_init_fn=my_init,
    ))
    print(f"    custom init: {initial} -> {w.config.num_hidden_layers} layers")
    assert w.config.num_hidden_layers == initial + 2


run_section("F1. Initialization: smart (default)", section_f_smart_init)
run_section("F2. Initialization: custom (with custom_init_fn)", section_f_custom_init)


# ============================================================================
# Section G: Validation
# ============================================================================
def section_g_validation_enabled():
    """Validation enabled (default) - catch issues early."""
    w = ExpandableModel.from_pretrained(MODEL_NAME, dtype=torch.float32)

    class MyBlock(CambiumBlock):
        required_config_keys = ["hidden_size"]

        def __init__(self, config, layer_idx=0):
            super().__init__()
            self.proj = nn.Linear(config.hidden_size, config.hidden_size)

        def forward(self, hidden_states, **kwargs):
            return self.proj(hidden_states)

    initial = w.config.num_hidden_layers
    w.expand(CustomBlockExpansion(
        block_class=MyBlock,
        num_layers=2,
        validate=True,  # Catch issues early
    ))
    print(f"    validation enabled: {initial} -> {w.config.num_hidden_layers}")
    assert w.config.num_hidden_layers == initial + 2


def section_g_validation_catches_bad_shape():
    """Validation catches a block with the wrong output shape."""
    w = ExpandableModel.from_pretrained(MODEL_NAME, dtype=torch.float32)

    class BadShapeBlock(CambiumBlock):
        required_config_keys = ["hidden_size"]

        def __init__(self, config, layer_idx=0):
            super().__init__()
            self.proj = nn.Linear(config.hidden_size, config.hidden_size * 2)

        def forward(self, hidden_states, **kwargs):
            return self.proj(hidden_states)

    raised = False
    try:
        w.expand(CustomBlockExpansion(
            block_class=BadShapeBlock,
            num_layers=1,
            validate=True,
            residual_connection=True,
        ))
    except BlockValidationError as e:
        raised = True
        # Print first 200 chars of the actual error so we can compare to the
        # example doc's claimed error message.
        msg = str(e)
        print(f"    BlockValidationError raised (expected): {msg[:200]}...")
    assert raised, "BadShapeBlock should have raised BlockValidationError"


def section_g_validation_disabled():
    """Skip validation - if you know what you're doing."""
    w = ExpandableModel.from_pretrained(MODEL_NAME, dtype=torch.float32)

    class MyBlock(CambiumBlock):
        required_config_keys = ["hidden_size"]

        def __init__(self, config, layer_idx=0):
            super().__init__()
            self.proj = nn.Linear(config.hidden_size, config.hidden_size)

        def forward(self, hidden_states, **kwargs):
            return self.proj(hidden_states)

    initial = w.config.num_hidden_layers
    w.expand(CustomBlockExpansion(
        block_class=MyBlock,
        num_layers=2,
        validate=False,
    ))
    print(f"    validation disabled: {initial} -> {w.config.num_hidden_layers}")
    assert w.config.num_hidden_layers == initial + 2


run_section("G1. Validation enabled (default)", section_g_validation_enabled)
run_section("G2. Validation catches shape mismatch", section_g_validation_catches_bad_shape)
run_section("G3. Validation disabled", section_g_validation_disabled)


# ============================================================================
# Section H: Mixing with Other Strategies
# ============================================================================
def section_h_mixed_strategies():
    """Interleave standard blocks, then add custom SwiGLU + GatedResidual."""
    w = ExpandableModel.from_pretrained(MODEL_NAME, dtype=torch.float32)
    initial = w.config.num_hidden_layers
    print(f"    Starting layers: {initial}")

    # First: insert standard blocks at even positions
    w.expand(InterleavedExpansion(
        num_layers=2,
        positions=[6, 18],
    ))
    after_interleave = w.config.num_hidden_layers
    print(f"    After InterleavedExpansion: {after_interleave}")

    # Then: insert custom SwiGLU blocks at specific positions
    w.expand(CustomBlockExpansion(
        block_class=SwiGLUBlock,
        num_layers=2,
        positions=[12, 24],
        residual_connection=True,
    ))
    after_swiglu = w.config.num_hidden_layers
    print(f"    After SwiGLU custom: {after_swiglu}")

    # Then: add gated residual blocks
    w.expand(CustomBlockExpansion(
        block_class=GatedResidualBlock,
        num_layers=2,
        residual_connection=True,
    ))
    after_gated = w.config.num_hidden_layers
    print(f"    After GatedResidual custom: {after_gated}")
    assert after_gated == after_interleave + 4

    # expansion report should be printable
    report = w.get_expansion_report()
    assert "InterleavedExpansion" in report
    assert "CustomBlockExpansion" in report
    print(f"    Expansion report contains both strategies (length={len(report)})")


run_section("H. Mixing with Other Strategies (Interleaved + SwiGLU + GatedResidual)",
           section_h_mixed_strategies)


# ============================================================================
# Summary
# ============================================================================
print()
print("=" * 70)
print("Custom Blocks Example Test - Summary")
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

# Exit with non-zero code so the test can be wired into CI later
sys.exit(0 if failed == 0 else 1)
