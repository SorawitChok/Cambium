Custom Blocks
=============

When built-in strategies are not enough, Cambium lets you define and insert your own PyTorch modules. This guide covers the block contract, built-in templates, validation, and mixing strategies.

The CambiumBlock Contract
-------------------------

All custom blocks must subclass :class:`~cambium.blocks.base.CambiumBlock` (which itself inherits ``nn.Module`` and ``ABC``). The contract is simple:

1. Accept ``hidden_states`` as the first positional argument in ``forward``.
2. Accept ``**kwargs`` (Hugging Face passes ``attention_mask``, ``position_ids``, etc.).
3. Return a tensor of the **same shape** as ``hidden_states``.

If your block returns a **delta** (residual) rather than full hidden states, set ``residual_connection=True`` when inserting. Cambium then wraps the block in :class:`~cambium.blocks.base.ResidualWrapper` so ``output = input + block(input)``.

Minimal example
~~~~~~~~~~~~~~~

.. code-block:: python

   import torch.nn as nn
   from cambium.blocks import CambiumBlock

   class MyBlock(CambiumBlock):
       required_config_keys = ["hidden_size"]

       def __init__(self, config, layer_idx=0):
           super().__init__()
           self.hidden_size = config.hidden_size
           self.proj = nn.Linear(self.hidden_size, self.hidden_size)

       def forward(self, hidden_states, **kwargs):
           return self.proj(hidden_states)

Insert it::

   from cambium import CustomBlockExpansion

   model.expand(CustomBlockExpansion(
       block_class=MyBlock,
       num_layers=2,
       residual_connection=True,
   ))

``required_config_keys``
~~~~~~~~~~~~~~~~~~~~~~~~

List the config attributes your block needs. Cambium validates that the model config contains them before insertion. If a key is missing, :class:`~cambium.exceptions.ConfigMismatchError` is raised with the missing and available keys.

Residual Connection
~~~~~~~~~~~~~~~~~~~

- ``residual_connection=True`` (default) — Cambium wraps the block in :class:`~cambium.blocks.base.ResidualWrapper`. Use this when ``forward`` returns a delta.
- ``residual_connection=False`` — Cambium wraps the block in :class:`~cambium.blocks.base.BlockOutputWrapper`. Use this when ``forward`` already adds the residual internally.

Plain ``nn.Module`` Blocks
~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also pass a plain ``nn.Module`` class. It must still accept ``hidden_states`` and ``**kwargs``, but it does not need to inherit ``CambiumBlock``::

   class PlainBlock(nn.Module):
       def __init__(self, config, layer_idx=0):
           super().__init__()
           self.proj = nn.Linear(config.hidden_size, config.hidden_size)

       def forward(self, hidden_states, **kwargs):
           return self.proj(hidden_states)

Built-in Templates
------------------

Cambium ships with several pre-built blocks in ``cambium.blocks.templates``:

SwiGLUBlock
~~~~~~~~~~~

A SwiGLU MLP block (``gate_proj * up_proj -> down_proj``). Returns a delta::

   from cambium.blocks import SwiGLUBlock

   model.expand(CustomBlockExpansion(
       block_class=SwiGLUBlock,
       num_layers=2,
       residual_connection=True,
   ))

Requires ``hidden_size`` in the model config.

MultiQueryAttentionBlock
~~~~~~~~~~~~~~~~~~~~~~~~

Multi-query attention (single KV head, multiple Q heads). Returns a delta::

   from cambium.blocks import MultiQueryAttentionBlock

   model.expand(CustomBlockExpansion(
       block_class=MultiQueryAttentionBlock,
       num_layers=2,
       residual_connection=True,
   ))

Requires ``hidden_size`` and ``num_attention_heads``.

GatedResidualBlock
~~~~~~~~~~~~~~~~~~

Gated residual using ``SiLU(x) * sigmoid(gate)``::

   from cambium.blocks import GatedResidualBlock

CrossAttentionBlock
~~~~~~~~~~~~~~~~~~~

Self-attention block with learned output gating, initialized near zero for identity-like start::

   from cambium.blocks import CrossAttentionBlock

RetentionBlock
~~~~~~~~~~~~~~

Linear-complexity retention mechanism with causal decay matrix::

   from cambium.blocks import RetentionBlock

Three Provider Modes
--------------------

``CustomBlockExpansion`` accepts blocks in three ways:

1. ``block_class`` — a class constructor. Cambium instantiates one instance per insertion position, passing ``(config, layer_idx)``.
2. ``block_factory`` — a callable ``(config, layer_idx) -> nn.Module``. Use this when you need custom logic per position.
3. ``block_instances`` — a list of pre-instantiated modules. Use this when you have already created the blocks outside Cambium.

Example with a factory::

   def make_block(config, layer_idx):
       block = MyBlock(config, layer_idx)
       # custom per-layer logic
       return block

   model.expand(CustomBlockExpansion(
       block_factory=make_block,
       num_layers=2,
   ))

Custom Initialization
---------------------

Pass a callable to ``custom_init_fn`` to run your own initialization after the block is created but before insertion::

   def my_init(block):
       nn.init.xavier_uniform_(block.proj.weight)

   model.expand(CustomBlockExpansion(
       block_class=MyBlock,
       num_layers=2,
       custom_init_fn=my_init,
   ))

Validation
----------

Set ``validate=True`` to run a full pre-insertion check::

   model.expand(CustomBlockExpansion(
       block_class=MyBlock,
       num_layers=2,
       validate=True,
   ))

Validation checks:

- **Signature** — ``forward`` accepts ``hidden_states`` and ``**kwargs``.
- **Shape** — output shape matches input shape.
- **Config keys** — all ``required_config_keys`` are present in the model config.
- **NaN** — no NaN parameters after initialization.

If any check fails, :class:`~cambium.exceptions.BlockValidationError` is raised with a clear message including the block index and reason.

Mixing Strategies
-----------------

You can combine ``CustomBlockExpansion`` with other strategies::

   # First, add native blocks
   model.expand(InterleavedExpansion(num_layers=2))

   # Then, insert custom blocks at specific positions
   model.expand(CustomBlockExpansion(
       block_class=MyBlock,
       positions=[3, 7],
       residual_connection=True,
   ))

Each expansion updates the model in-place and records its metadata.
