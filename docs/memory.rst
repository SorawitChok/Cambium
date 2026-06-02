Memory Estimation and Profiling
===============================

Training expanded models can consume significant memory. Cambium provides utilities to estimate requirements before training and profile actual GPU usage during training.

estimate_memory_usage
---------------------

:func:`~cambium.utils.memory.estimate_memory_usage` calculates a rough breakdown of memory consumption in gigabytes.

Basic usage
~~~~~~~~~~~

.. code-block:: python

   from cambium.utils.memory import estimate_memory_usage

   estimate = estimate_memory_usage(
       model,
       batch_size=4,
       sequence_length=512,
       dtype=torch.float32,
   )

   print(f"Total: {estimate['total_gb']:.2f} GB")
   print(f"Weights: {estimate['weights_gb']:.2f} GB")
   print(f"Activations: {estimate['activations_gb']:.2f} GB")
   print(f"Gradients: {estimate['gradients_gb']:.2f} GB")
   print(f"Optimizer: {estimate['optimizer_gb']:.2f} GB")
   print(f"Recommended: {estimate['recommended_gb']:.2f} GB")

Parameters
~~~~~~~~~~

- ``model`` — the PyTorch model (or ``ExpandableModel``).
- ``batch_size`` — training batch size.
- ``sequence_length`` — token sequence length.
- ``dtype`` — ``torch.float32`` or ``torch.float16``.
- ``gradient_checkpointing`` — whether gradient checkpointing is enabled (reduces activation memory).

The "recommended" value adds a 20% safety buffer to the total.

Scaling effects
~~~~~~~~~~~~~~~

- **Batch size** — activations scale linearly with batch size.
- **Sequence length** — activations scale linearly with sequence length.
- **Dtype** — float16 halves weight and optimizer memory compared to float32.
- **Gradient checkpointing** — roughly halves activation memory at the cost of a small compute overhead.

GPU Memory Profiling
--------------------

:func:`~cambium.utils.memory.get_memory_profile` returns current GPU memory statistics::

   from cambium.utils.memory import get_memory_profile

   profile = get_memory_profile()
   print(f"Allocated: {profile['allocated_gb']:.2f} GB")
   print(f"Reserved:  {profile['reserved_gb']:.2f} GB")
   print(f"Max allocated: {profile['max_allocated_gb']:.2f} GB")

There is also a convenience printer::

   from cambium.utils.memory import print_memory_profile
   print_memory_profile()

Model Output Validation
-----------------------

Before launching a long training run, sanity-check that the model produces valid outputs::

   from cambium.utils import validate_model_output

   validate_model_output(
       model.get_model(),
       tokenizer,
       test_prompt="Hello, world!",
   )

This verifies:

- No NaN or Inf in the output logits.
- The model runs without raising an exception.
- Optionally compares against expected outputs within a tolerance.

Integration with Training
-------------------------

A typical workflow::

   import torch
   from cambium.utils.memory import estimate_memory_usage
   from cambium.training import TrainingUtilities

   # 1. Estimate before training
   estimate = estimate_memory_usage(
       model.get_model(),
       batch_size=4,
       sequence_length=512,
       dtype=torch.float32,
   )
   print(f"Estimated need: {estimate['recommended_gb']:.2f} GB")

   # 2. Enable optimizations if tight on memory
   if estimate['recommended_gb'] > 12:
       TrainingUtilities.enable_memory_optimizations(
           model.get_model(),
           gradient_checkpointing=True,
           mixed_precision="fp16",
       )

   # 3. Train
   trainer.train(...)

   # 4. Profile actual usage
   from cambium.utils.memory import print_memory_profile
   print_memory_profile()
