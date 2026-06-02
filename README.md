# Cambium 🌱

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

> **Advanced LLM Architecture Augmentation Library**
>
> Expand your LLMs like nature intended - surgically, efficiently, and beautifully.

Cambium is an open-source Python library that enables surgical expansion of Large Language Models (LLMs). Named after the plant tissue responsible for secondary growth, Cambium allows developers and researchers to add new layers or architecture blocks to existing models using familiar PyTorch-like APIs.

## 🎯 The Core Idea

```
Traditional: Train new 7B model from scratch (expensive 💰)
Cambium:     Expand 2B → 4B, preserve weights, train only new (efficient 🌱)
```

**Key Benefits:**
- 🔄 **Preserve pretrained knowledge** - Keep all original weights intact
- 📈 **Modular expansion** - Add capacity where needed
- 🎛️ **Staged training** - Progressive unfreezing strategies
- ⚡ **Minimal compute** - Train only new parameters initially
- 🤝 **Framework compatible** - Works with HF Transformers, TRL, PEFT

## 📦 Installation

```bash
# Basic installation
pip install cambium

# With training dependencies (recommended)
pip install "cambium[train]"

# Development installation
pip install "cambium[dev]"
```

## 🚀 Quick Start

```python
from cambium import ExpandableModel, InterleavedExpansion
from cambium.training import StagedTrainer

# 1. Load base model
model = ExpandableModel.from_pretrained("google/gemma-2b")

# 2. Expand with 4 new transformer blocks
model.expand(InterleavedExpansion(num_layers=4, initialization="identity"))

# 3. Setup staged training
trainer = StagedTrainer(model)
trainer.add_phase(name="warmup", freeze="original", lr=1e-4, epochs=2)
trainer.add_phase(name="finetune", freeze="none", lr=1e-6, epochs=1)

# 4. Train
trainer.train(train_dataloader, eval_dataloader)

# 5. Save
model.save_expanded("./my-expanded-model")
```

## 📚 Expansion Strategies

### 1. Interleaved Block Expansion (LLaMA-Pro style)

Insert new transformer blocks between existing ones:

```python
from cambium import InterleavedExpansion

expander = InterleavedExpansion(
    num_layers=4,
    initialization="identity",  # Near-identity init
)
model.expand(expander)

# Original: [Block0] → [Block1] → [Block2]
# Expanded: [Block0] → [New0] → [Block1] → [New1] → [Block2]
```

### 2. Width Expansion

Increase hidden dimensions:

```python
from cambium.strategies import WidthExpansion

expander = WidthExpansion(
    hidden_dim_multiplier=1.5,  # 768 → 1152
    initialization="copy",
)
model.expand(expander)
```

### 3. Parallel Adapters

Add parallel pathways:

```python
from cambium.strategies import ParallelAdapterExpansion

expander = ParallelAdapterExpansion(
    adapter_type="bottleneck",
    bottleneck_dim=256,
    target_layers=[20, 21, 22, 23],  # Last 4 layers
)
model.expand(expander)
```

### 4. Custom Block Expansion

Define and insert your own architecture blocks:

```python
from cambium import CustomBlockExpansion
from cambium.blocks import CambiumBlock, SwiGLUBlock
import torch.nn as nn

# Use a template
model.expand(CustomBlockExpansion(
    block_class=SwiGLUBlock,
    num_layers=4,
    residual_connection=True,
))

# Or define your own
class MyBlock(CambiumBlock):
    required_config_keys = ["hidden_size"]

    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states, **kwargs):
        return self.proj(hidden_states)

model.expand(CustomBlockExpansion(
    block_class=MyBlock,
    num_layers=2,
    residual_connection=True,
))
```

## 🎓 Training Strategies

### Phase 1: Warmup New Layers
```python
trainer.add_phase(
    name="warmup",
    freeze="original",  # Freeze all original weights
    lr=1e-4,
    epochs=2,
)
```

### Phase 2: Progressive Unfreezing
```python
trainer.add_phase(
    name="unfreeze",
    unfreeze_groups=[-4, -3, -2, -1],  # Last 4 layer groups
    lr=5e-5,
    epochs=1,
)
```

### Phase 3: Full Fine-tuning
```python
trainer.add_phase(
    name="finetune",
    freeze="none",  # Unfreeze all
    lr=1e-6,
    discriminative_lr={
        "embeddings": 1e-8,
        "original_layers": 1e-6,
        "new_layers": 1e-5,
    },
    epochs=1,
)
```

## 🔧 Supported Models

| Model | Block Expansion | Width Expansion | Adapters |
|-------|-----------------|-----------------|----------|
| LLaMA 2/3 | ✅ | ✅ | ✅ |
| Gemma | ✅ | ✅ | ✅ |
| Mistral | ✅ | ✅ | ✅ |
| Qwen2 | ✅ | ✅ | ✅ |

*More models coming soon!*

## 📖 Documentation

- [01 - Quickstart Guide](examples/01_quickstart.md)
- [02 - Interleaved Expansion](examples/02_interleaved_expansion.md)
- [03 - Staged Training](examples/03_staged_training.md)
- [04 - Complete Workflow](examples/04_complete_workflow.md)
- [05 - Width Expansion](examples/05_width_expansion.md)
- [06 - Parallel Adapters](examples/06_parallel_adapters.md)
- [07 - Custom Blocks](examples/07_custom_blocks.md)

## 🏗️ Architecture

```
cambium/
├── core/              # Low-level surgical operations
│   ├── expansion.py   # Model surgery engine
│   ├── initialization.py  # Smart init strategies
│   └── freezing.py    # Advanced freezing logic
├── strategies/        # Expansion strategies
│   ├── block_expansion.py
│   ├── width_expansion.py
│   ├── parallel_adapters.py
│   └── custom_expansion.py  # Custom block insertion
├── blocks/            # Custom block definitions
│   ├── base.py        # CambiumBlock ABC, ResidualWrapper
│   └── templates.py   # Pre-built blocks (SwiGLU, etc.)
├── training/          # Training utilities
│   ├── staged_trainer.py
│   └── utilities.py
├── models/            # Model wrappers
│   └── expandable.py
├── utils/             # Helper utilities
│   ├── memory.py
│   └── validation.py
└── exceptions.py      # Error classes
```

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=cambium tests/

# Run specific test file
pytest tests/unit/test_expansion.py -v
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/SorawitChok/Cambium.git
cd cambium

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## 📊 Comparison with Alternatives

| Approach | Method | Pros | Cons |
|----------|--------|------|------|
| **Cambium** | Add new blocks | Full capacity increase | More memory |
| LoRA | Low-rank adapters | Minimal params | Limited capacity |
| IA³ | Learned scaling | Preserves structure | Architecture-specific |
| Prefix Tuning | Learned prefixes | No model changes | Limited expressiveness |
| Full Fine-tune | Train all params | Best performance | Expensive |

**When to use Cambium:**
- You need more model capacity than adapters can provide
- You want to add architectural components (new attention types, etc.)
- You have compute for training new layers but not full models
- You want progressive training strategies

## 📄 Citation

If you use Cambium in your research, please cite:

```bibtex
@software{cambium2024,
  title={Cambium: Advanced LLM Architecture Augmentation},
  author={Sorawit Chokphantavee, Sirawit Chokphantavee, and Cambium Team},
  year={2024},
  url={https://github.com/SorawitChok/Cambium}
}
```

## 📜 License

Cambium is released under the Apache License 2.0. See [LICENSE](LICENSE) for details.

### Commercial Use

Cambium is free for commercial use under Apache 2.0. For enterprises requiring:
- Professional support & SLAs
- Custom expansion strategies
- Training & consulting services

See [Enterprise](https://cambium.dev/enterprise) for commercial offerings.

## 🙏 Acknowledgments

Cambium builds on the excellent work of:
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/)
- [LLaMA-Pro](https://arxiv.org/abs/2401.02412) (inspiration for interleaved expansion)

## 💬 Community

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Q&A and general discussion
- **Discord**: [Join our community](https://discord.gg/cambium)

---

<p align="center">
  <i>Grow your models like nature intended 🌱</i>
</p>
