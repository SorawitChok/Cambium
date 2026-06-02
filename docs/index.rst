Cambium Documentation
=====================

**Cambium** is an open-source Python library for **surgical expansion of Large Language Models (LLMs)**.

Named after the plant tissue responsible for secondary growth, Cambium lets you add new layers, width, or custom architecture blocks to existing pretrained models while preserving original weights. This means you can grow a 2B model toward 4B capacity without training from scratch.

.. code-block:: python

   from cambium import ExpandableModel, InterleavedExpansion

   model = ExpandableModel.from_pretrained(
       "HuggingFaceTB/SmolLM2-135M",
       dtype=torch.float32,
   )
   model.expand(InterleavedExpansion(num_layers=2, initialization="identity"))

Key Benefits
------------

- **Preserve pretrained knowledge** — all original weights stay intact
- **Modular expansion** — add capacity exactly where you need it
- **Staged training** — progressive unfreezing with discriminative learning rates
- **Minimal compute** — train only new parameters initially
- **Framework compatible** — works with Hugging Face Transformers, TRL, and PEFT

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   concepts

.. toctree::
   :maxdepth: 2
   :caption: User Guides

   strategies
   training
   custom_blocks
   memory

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
