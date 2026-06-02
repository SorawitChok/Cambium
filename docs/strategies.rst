Expansion Strategies
====================

Cambium provides four built-in expansion strategies. Each strategy is a dataclass-like object that configures **what** to add, **where** to add it, and **how** to initialize it.

All strategies follow the same usage pattern::

   from cambium import ExpandableModel

   model = ExpandableModel.from_pretrained("HuggingFaceTB/SmolLM2-135M", dtype=torch.float32)
   model.expand(SomeStrategy(...))

Interleaved Block Expansion
---------------------------

:class:`~cambium.strategies.block_expansion.InterleavedExpansion` inserts new native transformer blocks (e.g., ``LlamaDecoderLayer``) between existing ones. This is the LLaMA-Pro style approach.

Basic usage
~~~~~~~~~~~

.. code-block:: python

   from cambium import InterleavedExpansion

   model.expand(InterleavedExpansion(num_layers=2, initialization="identity"))

- ``num_layers`` — how many new blocks to insert.
- ``initialization`` — one of ``"identity"``, ``"small_random"``, ``"noise"``, ``"zero"``.

Cambium automatically distributes the new blocks evenly. If you want explicit control, pass ``positions``::

   model.expand(InterleavedExpansion(
       num_layers=2,
       positions=[2, 5],  # Insert after layer 2 and layer 5
       initialization="identity",
   ))

The engine updates ``layer_idx`` on every attention module so generation remains consistent.

Width Expansion
---------------

:class:`~cambium.strategies.width_expansion.WidthExpansion` increases hidden dimensions (and optionally intermediate or attention dimensions) rather than adding layers.

Basic usage
~~~~~~~~~~~

.. code-block:: python

   from cambium.strategies import WidthExpansion

   model.expand(WidthExpansion(
       hidden_dim_multiplier=1.25,
       initialization="zero",
   ))

- ``hidden_dim_multiplier`` — factor applied to the current hidden size (e.g., 576 → 720).
- ``initialization`` — how to initialize new weight rows/columns: ``"copy"``, ``"zero"``, or ``"noise"``.

Selective expansion
~~~~~~~~~~~~~~~~~~~

You can target only specific layers or only the MLP::

   model.expand(WidthExpansion(
       hidden_dim_multiplier=1.25,
       initialization="zero",
       layer_indices=list(range(4, 8)),  # Only layers 4-7
       expand_attention=False,             # MLP only
   ))

Width expansion requires more careful training than block expansion. See :doc:`training` for recommendations.

Parallel Adapter Expansion
--------------------------

:class:`~cambium.strategies.parallel_adapters.ParallelAdapterExpansion` adds small parallel pathways alongside existing transformer blocks without inserting new layers. This is similar in spirit to LoRA but adds full bottleneck or attention layers.

Bottleneck adapters
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cambium.strategies import ParallelAdapterExpansion

   model.expand(ParallelAdapterExpansion(
       adapter_type="bottleneck",
       bottleneck_dim=64,
       initialization="zero",
   ))

- ``bottleneck_dim`` — inner dimension of the down-project → up-project bottleneck. A good rule of thumb is 1/8 to 1/4 of the model's hidden size.
- ``initialization="zero"`` makes the adapter start as a no-op (output ≈ 0), so the model behaves like the original before training.

Targeting specific layers
~~~~~~~~~~~~~~~~~~~~~~~~~~

Adapters are most effective in later layers. Target the last 8 layers dynamically::

   n_layers = len(model.get_model().model.layers)
   model.expand(ParallelAdapterExpansion(
       adapter_type="bottleneck",
       bottleneck_dim=64,
       target_layers=list(range(n_layers - 8, n_layers)),
   ))

Attention adapters
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   model.expand(ParallelAdapterExpansion(
       adapter_type="attention",
       num_heads=4,
   ))

Training adapters
~~~~~~~~~~~~~~~~~

Freeze the base model and train only the adapter parameters::

   model.freeze_original()
   # Unfreeze only adapter parameters
   for name, param in model.get_model().named_parameters():
       if "cambium_adapter" in name:
           param.requires_grad = True

Adapters add roughly 0.5–1% of the base model's parameters, making them ideal for limited compute or quick domain-adaptation experiments.

Custom Block Expansion
----------------------

:class:`~cambium.strategies.custom_expansion.CustomBlockExpansion` inserts user-defined PyTorch modules into the model. This is the most flexible strategy.

Using a built-in template
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cambium import CustomBlockExpansion
   from cambium.blocks import SwiGLUBlock

   model.expand(CustomBlockExpansion(
       block_class=SwiGLUBlock,
       num_layers=2,
       residual_connection=True,
   ))

Using a custom block
~~~~~~~~~~~~~~~~~~~~

Define a class that follows the :class:`~cambium.blocks.base.CambiumBlock` contract::

   import torch.nn as nn
   from cambium.blocks import CambiumBlock

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

Three provider modes
~~~~~~~~~~~~~~~~~~~~

``CustomBlockExpansion`` accepts blocks in three ways:

1. ``block_class`` — a class constructor; Cambium instantiates one per insertion position.
2. ``block_factory`` — a callable ``(config, layer_idx) -> nn.Module``.
3. ``block_instances`` — a list of pre-instantiated modules.

Validation
~~~~~~~~~~

Set ``validate=True`` to run pre-insertion checks (shape, signature, config keys, NaN)::

   model.expand(CustomBlockExpansion(
       block_class=MyBlock,
       num_layers=2,
       validate=True,
   ))

If validation fails, a :class:`~cambium.exceptions.BlockValidationError` is raised with a clear message.

Comparison
----------

.. list-table::
   :header-rows: 1
   :widths: 20 30 25 25

   * - Strategy
     - What it adds
     - Best for
     - Param increase
   * - InterleavedExpansion
     - Native transformer blocks
     - More capacity, new reasoning steps
     - ~20–30%
   * - WidthExpansion
     - Wider hidden dims
     - More FLOPs per layer, memory-constrained
     - ~50–100%
   * - ParallelAdapterExpansion
     - Bottleneck / attention adapters
     - Domain adaptation, quick experiments
     - ~0.5–1%
   * - CustomBlockExpansion
     - User-defined blocks
     - Novel architectures, research
     - User-defined

Append Expansion
----------------

:class:`~cambium.strategies.block_expansion.AppendExpansion` (public but not exported in the top-level ``__init__``) appends new native transformer blocks at the **end** of the model rather than interleaving them. Import it directly if needed::

   from cambium.strategies.block_expansion import AppendExpansion

Mixing Strategies
-----------------

You can apply multiple strategies to the same model::

   model.expand(InterleavedExpansion(num_layers=2))
   model.expand(ParallelAdapterExpansion(adapter_type="bottleneck", bottleneck_dim=64))

Each call updates the model in-place and records its config in the expansion history.
