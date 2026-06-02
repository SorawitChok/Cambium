Training Guide
==============

After expanding a model, you typically train only the new parameters first, then progressively unfreeze original layers. Cambium provides :class:`~cambium.training.staged_trainer.StagedTrainer` and :class:`~cambium.training.utilities.TrainingUtilities` to automate this.

StagedTrainer Overview
----------------------

:class:`~cambium.training.staged_trainer.StagedTrainer` orchestrates multi-phase training. Each phase is a :class:`~cambium.training.staged_trainer.TrainingPhase` dataclass that defines epochs, learning rate, freezing behavior, and optional discriminative learning rates.

Basic recipe
~~~~~~~~~~~~

.. code-block:: python

   from cambium.training import StagedTrainer

   trainer = StagedTrainer(model)

   # Phase 1: train only new layers
   trainer.add_phase(name="warmup", freeze="original", lr=1e-4, epochs=2)

   # Phase 2: unfreeze the last few layer groups
   trainer.add_phase(name="progressive", unfreeze_groups=[-2, -1], lr=5e-5, epochs=1)

   # Phase 3: full fine-tune
   trainer.add_phase(name="finetune", freeze="none", lr=1e-6, epochs=1)

   trainer.train(train_dataloader, eval_dataloader)

Phase configuration
~~~~~~~~~~~~~~~~~~~

Each phase accepts these key arguments:

- ``name`` — human-readable phase name.
- ``freeze`` — ``"original"`` | ``"all"`` | ``"none"`` | ``None``.
- ``unfreeze_groups`` — list of group indices to unfreeze (see Progressive Unfreezing below).
- ``lr`` — base learning rate for the phase.
- ``discriminative_lr`` — per-group learning-rate multipliers (see below).
- ``epochs`` — training epochs for this phase.
- ``batch_size`` — override batch size.
- ``gradient_accumulation_steps`` — gradient accumulation.
- ``warmup_steps`` — linear warmup steps.
- ``gradient_clipping`` — max gradient norm.
- ``eval_every`` — evaluate every N steps.
- ``save_every`` — checkpoint every N steps.

Progressive Unfreezing
----------------------

Instead of unfreezing all layers at once, you can unfreeze them in groups. This reduces catastrophic forgetting and stabilizes training.

Freezing modes
~~~~~~~~~~~~~~

- ``freeze="original"`` — freeze every parameter that existed before expansion (new layers remain trainable).
- ``freeze="all"`` — freeze everything.
- ``freeze="none"`` or ``None`` — unfreeze everything.

Layer groups
~~~~~~~~~~~~

:meth:`~cambium.core.freezing.FreezingManager.unfreeze_group` divides the transformer layers into ``num_groups`` equal groups. With the default ``num_groups=4``::

   # Group -1 = last quarter of layers
   # Group -2 = second-to-last quarter
   # Group -3 = second quarter
   # Group -4 = first quarter

Valid indices for the default are ``[-4, -3, -2, -1]``.

Manual freezing
~~~~~~~~~~~~~~~

For finer control, use the freezing manager directly::

   fm = model.freezing_manager

   # By regex pattern
   fm.freeze_by_pattern(r"model\.layers\.[0-9]+\.")

   # By layer index range
   fm.unfreeze_layer_range(20, 29)

   # By group
   fm.unfreeze_group(-2, num_groups=4)

   # Print status
   model.print_trainable()

Discriminative Learning Rates
------------------------------

Different parameter groups can use different learning rates. Cambium supports three ways to specify groups:

1. **Semantic names** — ``"embeddings"``, ``"new_layers"``, ``"original_layers"``.
2. **Layer index tuples** — ``(start, end)`` ranges.
3. **Raw regex strings** — matched against parameter names.

Example with semantic names::

   trainer.add_phase(
       name="finetune",
       freeze="none",
       lr=1e-6,
       discriminative_lr={
           "embeddings": 1e-8,
           "original_layers": 1e-6,
           "new_layers": 1e-5,
       },
       epochs=1,
   )

Example with layer ranges::

   trainer.add_phase(
       name="finetune",
       freeze="none",
       lr=1e-6,
       discriminative_lr={
           (0, 10): 1e-7,
           (11, 20): 1e-6,
           (21, 29): 1e-5,
       },
       epochs=1,
   )

TrainingUtilities
-----------------

:class:`~cambium.training.utilities.TrainingUtilities` provides static helpers for common training tasks.

Memory optimizations
~~~~~~~~~~~~~~~~~~~~

Enable gradient checkpointing and mixed precision::

   from cambium.training import TrainingUtilities

   TrainingUtilities.enable_memory_optimizations(
       model.get_model(),
       gradient_checkpointing=True,
       mixed_precision="fp16",
   )

Discriminative LR optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create an optimizer with per-group learning rates::

   optimizer = TrainingUtilities.get_optimizer_with_discriminative_lr(
       model.get_model(),
       base_lr=1e-4,
       discriminative_lr={"new_layers": 1e-3, "original_layers": 1e-5},
   )

HF Trainer integration
~~~~~~~~~~~~~~~~~~~~~~

Use Cambium's optimizer inside Hugging Face ``Trainer``::

   from transformers import Trainer, TrainingArguments

   optimizer = TrainingUtilities.get_optimizer_with_discriminative_lr(...)

   trainer = Trainer(
       model=model.get_model(),
       args=TrainingArguments(...),
       train_dataset=train_dataset,
       optimizers=(optimizer, None),
   )

TRL SFTTrainer integration
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use with TRL's ``SFTTrainer`` (requires ``pip install trl``)::

   from trl import SFTTrainer

   trainer = SFTTrainer(
       model=model.get_model(),
       train_dataset=train_dataset,
       optimizers=(optimizer, None),
       ...
   )

Checkpointing
-------------

Save and resume training state::

   # Save
   trainer.save_checkpoint("./checkpoint_dir", epoch=current_epoch)

   # Resume
   trainer.load_checkpoint("./checkpoint_dir")

Catastrophic Forgetting Detection
---------------------------------

Monitor whether the expanded model is diverging from the original::

   from cambium.utils.validation import CatastrophicForgettingDetector

   detector = CatastrophicForgettingDetector(base_model=model.get_model())

   # During or after training
   detector.evaluate(expanded_model, eval_dataloader)
   if detector.is_forgetting():
       print("Warning: catastrophic forgetting detected")

You can also use the convenience function::

   from cambium.utils import check_for_catastrophic_forgetting

   check_for_catastrophic_forgetting(original_model, expanded_model, eval_dataset)

Recommended Hyperparameters
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Phase
     - LR
     - Freeze
     - Notes
   * - Warmup
     - 1e-4
     - original
     - Train only new layers. Use identity init so output stays close to original.
   * - Progressive
     - 5e-5
     - unfreeze tail
     - Unfreeze last 1–2 groups. Lower LR to prevent forgetting.
   * - Full fine-tune
     - 1e-6
     - none
     - Optional. Use discriminative LR: embeddings ≪ original ≪ new.

Tips
----

- Start with a small model (e.g., ``HuggingFaceTB/SmolLM2-135M``) to iterate quickly.
- Save checkpoints after every phase.
- Validate generation quality at each phase to catch forgetting early.
- Use ``model.print_trainable()`` after each ``add_phase`` to verify what is trainable.
