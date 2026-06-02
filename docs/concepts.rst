Core Concepts
=============

What Is Surgical Expansion?
-----------------------------

Traditional fine-tuning changes existing weights. Cambium **adds new parameters** — layers, width, or custom blocks — while keeping the original weights intact. This is analogous to how a tree adds new growth rings: the old wood stays, and new tissue expands the trunk.

.. code-block:: text

   Traditional fine-tune:  [Block0] → [Block1] → [Block2]
                           (all weights drift)

   Cambium block expansion: [Block0] → [New0] → [Block1] → [New1] → [Block2]
                           (original weights frozen, new blocks trained)

The Four Layers
---------------

Cambium is organized into four layers, each with a distinct responsibility:

Engine
~~~~~~

Located in ``cambium/core/``, the engine performs low-level surgical operations:

- **Insertion** — inserts new blocks into ``model.model.layers`` at precise indices.
- **Dimension expansion** — expands hidden dimensions of linear, embedding, and LayerNorm weights.
- **Validation** — checks that inserted blocks match expected shapes and signatures.
- **History tracking** — logs every mutation so expansions can be reported or replayed.

The engine is used directly only when writing custom strategies. Most users interact with it indirectly through :doc:`strategies`.

Strategies
~~~~~~~~~~

Located in ``cambium/strategies/``, strategies are user-facing configuration objects that decide **where** and **how** to expand. Each strategy implements ``expand(model, engine)``.

Available strategies:

- :class:`~cambium.strategies.block_expansion.InterleavedExpansion` — insert native transformer blocks between existing ones (LLaMA-Pro style).
- :class:`~cambium.strategies.width_expansion.WidthExpansion` — increase hidden dimensions.
- :class:`~cambium.strategies.parallel_adapters.ParallelAdapterExpansion` — add parallel adapter pathways.
- :class:`~cambium.strategies.custom_expansion.CustomBlockExpansion` — insert user-defined blocks.

Blocks
~~~~~~

Located in ``cambium/blocks/``, blocks are base classes and templates for custom architecture:

- :class:`~cambium.blocks.base.CambiumBlock` — abstract base class defining the contract.
- :class:`~cambium.blocks.base.ResidualWrapper` — wraps a block so ``output = input + block(input)``.
- :class:`~cambium.blocks.templates.SwiGLUBlock`, :class:`~cambium.blocks.templates.MultiQueryAttentionBlock`, etc. — pre-built templates.

Models
~~~~~~

Located in ``cambium/models/``:

- :class:`~cambium.models.expandable.ExpandableModel` — PyTorch-native wrapper that loads HF models, applies strategies, and handles save/load.

Interaction Flow
----------------

A typical expansion follows this flow:

1. **User** creates a strategy (e.g., ``InterleavedExpansion``).
2. **Strategy** builds blocks and calls ``engine.insert_blocks(model, positions, factory)``.
3. **Engine** mutates the model in-place (inserts into ``model.model.layers``).
4. **Strategy** initializes the new blocks **by instance reference**, never by re-reading from mutated positions.
5. **User** optionally freezes original weights and trains the model.

Critical Invariants
-------------------

Positions are stale after insertion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After ``engine.insert_blocks(...)`` mutates the layers list, the original ``positions`` indices are no longer valid. Strategies must capture the actual block instances before or during insertion and pass those to ``_apply_initialization``. See ``CustomBlockExpansion`` and ``InterleavedExpansion`` for the correct pattern.

The ``_cambium_new`` attribute
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The engine tags every newly inserted module with ``module._cambium_new = True``. The :class:`~cambium.core.freezing.FreezingManager` and training utilities rely on this attribute (not name patterns) to distinguish new parameters from original ones.

``layer_idx`` synchronization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After block insertion, the engine updates ``layer_idx`` on all attention modules inside the layers list. This ensures KV-cache indexing remains correct during text generation.

Save / Load Lifecycle
---------------------

:meth:`~cambium.models.expandable.ExpandableModel.save_expanded` uses Hugging Face's ``save_pretrained``, which correctly handles tied embeddings and ``safetensors`` shared tensors.

:meth:`~cambium.models.expandable.ExpandableModel.load_expanded` performs three steps:

1. Loads the base model and metadata.
2. Re-applies any registered strategies from the saved config. This is required for strategies like ``ParallelAdapterExpansion`` that attach side-modules as plain attributes and patch forward closures — neither of which survive a raw ``from_pretrained`` cycle.
3. Reloads "orphan" weights from the ``model.safetensors`` file. The base HF loader drops keys it does not recognize; Cambium copies them back into the live parameters after re-attaching the expansion modules.

Strategies that capture callables (e.g., ``CustomBlockExpansion`` with a custom ``block_class``) cannot be auto-reconstructed from JSON. In that case ``load_expanded`` skips re-application and leaves a placeholder record.
