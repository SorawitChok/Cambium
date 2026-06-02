Examples
========

The following example guides walk through real use cases with runnable code. Each example uses the small, ungated ``HuggingFaceTB/SmolLM2-135M`` model so you can run them on CPU or modest GPU hardware.

.. toctree::
   :maxdepth: 1

   examples/01_quickstart
   examples/02_interleaved_expansion
   examples/03_staged_training
   examples/04_complete_workflow
   examples/05_width_expansion
   examples/06_parallel_adapters
   examples/07_custom_blocks

Notes
-----

- All examples assume ``pip install cambium-llm`` and ``pip install transformers`` are already installed.
- Some examples reference optional integrations (``trl``, ``peft``, ``datasets``). Install them with::

     pip install trl peft datasets

- If you run on CPU, every ``from_pretrained`` call should include ``dtype=torch.float32`` to avoid BFloat16/Float mismatches.
- The example markdown files live in the repository ``examples/`` directory and are included here via ``myst-parser``.
