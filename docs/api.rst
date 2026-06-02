API Reference
=============

This page provides a curated reference of Cambium's public API. For full module-level documentation (including private helpers), see the generated module index.

Core Components
---------------

The engine layer performs low-level model surgery and state management.

.. automodule:: cambium.core.expansion
   :members:
   :show-inheritance:

.. automodule:: cambium.core.freezing
   :members:
   :show-inheritance:

.. automodule:: cambium.core.initialization
   :members:
   :show-inheritance:

Expansion Strategies
--------------------

Strategies decide where and how to expand a model.

.. automodule:: cambium.strategies.block_expansion
   :members:
   :show-inheritance:

.. automodule:: cambium.strategies.width_expansion
   :members:
   :show-inheritance:

.. automodule:: cambium.strategies.parallel_adapters
   :members:
   :show-inheritance:

.. automodule:: cambium.strategies.custom_expansion
   :members:
   :show-inheritance:

Custom Blocks
-------------

Base classes and templates for defining custom architecture blocks.

.. automodule:: cambium.blocks.base
   :members:
   :show-inheritance:

.. automodule:: cambium.blocks.templates
   :members:
   :show-inheritance:

Models
------

.. automodule:: cambium.models.expandable
   :members:
   :show-inheritance:

Training
--------

.. automodule:: cambium.training.staged_trainer
   :members:
   :show-inheritance:

.. automodule:: cambium.training.utilities
   :members:
   :show-inheritance:

Utilities
---------

.. automodule:: cambium.utils.memory
   :members:
   :show-inheritance:

.. automodule:: cambium.utils.validation
   :members:
   :show-inheritance:

Exceptions
----------

.. automodule:: cambium.exceptions
   :members:
   :show-inheritance:
