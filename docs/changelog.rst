Changelog
=========

Version 0.1.0 (2024-06-02)
--------------------------

Initial release of Cambium.

Expansion Strategies
~~~~~~~~~~~~~~~~~~~~

- :class:`~cambium.strategies.block_expansion.InterleavedExpansion` — insert native transformer blocks between existing layers (LLaMA-Pro style).
- :class:`~cambium.strategies.width_expansion.WidthExpansion` — increase hidden dimensions with copy/zero/noise initialization.
- :class:`~cambium.strategies.parallel_adapters.ParallelAdapterExpansion` — add bottleneck or attention adapter pathways alongside existing blocks.
- :class:`~cambium.strategies.custom_expansion.CustomBlockExpansion` — insert user-defined PyTorch modules with validation and residual wrapping.

Core Engine
~~~~~~~~~~~

- :class:`~cambium.core.expansion.ExpansionEngine` — surgical block insertion, dimension expansion, and history tracking.
- :class:`~cambium.core.initialization.Initializer` and :class:`~cambium.core.initialization.InitializationStrategy` — identity, small-random, noise, zero, Xavier, Kaiming, and knowledge-distillation initializers.
- :class:`~cambium.core.freezing.FreezingManager` — fine-grained parameter freezing by pattern, layer range, progressive groups, and semantic categories.

Blocks
~~~~~~

- :class:`~cambium.blocks.base.CambiumBlock` — abstract base class for custom blocks.
- :class:`~cambium.blocks.base.ResidualWrapper` — automatic residual wrapping.
- Pre-built templates: :class:`~cambium.blocks.templates.SwiGLUBlock`, :class:`~cambium.blocks.templates.MultiQueryAttentionBlock`, :class:`~cambium.blocks.templates.GatedResidualBlock`, :class:`~cambium.blocks.templates.CrossAttentionBlock`, :class:`~cambium.blocks.templates.RetentionBlock`.

Models
~~~~~~

- :class:`~cambium.models.expandable.ExpandableModel` — PyTorch-native wrapper with ``from_pretrained``, ``expand``, ``save_expanded``, and ``load_expanded`` (including orphan-weight reload for parallel adapters).

Training
~~~~~~~~

- :class:`~cambium.training.staged_trainer.StagedTrainer` — multi-phase training with progressive unfreezing.
- :class:`~cambium.training.staged_trainer.TrainingPhase` — per-phase configuration dataclass.
- :class:`~cambium.training.utilities.TrainingUtilities` — discriminative LR, memory optimizations, HF/TRL integration, and staged LR schedules.

Utilities
~~~~~~~~~

- :func:`~cambium.utils.memory.estimate_memory_usage` — estimate training memory (weights, activations, gradients, optimizer states).
- :func:`~cambium.utils.memory.get_memory_profile` / :func:`~cambium.utils.memory.print_memory_profile` — GPU memory profiling.
- :func:`~cambium.utils.validation.validate_model_output` — sanity-check model outputs.
- :class:`~cambium.utils.validation.CatastrophicForgettingDetector` — KL-divergence monitoring against a base model.

Exceptions
~~~~~~~~~~

- :class:`~cambium.exceptions.CambiumError` — base exception.
- :class:`~cambium.exceptions.BlockValidationError` — custom block validation failure.
- :class:`~cambium.exceptions.ShapeMismatchError` — block output shape mismatch.
- :class:`~cambium.exceptions.ConfigMismatchError` — missing config keys.
- :class:`~cambium.exceptions.ExpansionError` — general expansion failure.
