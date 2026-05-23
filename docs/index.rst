Cambium Documentation
=====================

Cambium is an open-source Python library for **surgical expansion of Large Language Models (LLMs)**.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   api

Quick Start
-----------

Install Cambium::

   pip install cambium

Expand a model in 3 lines::

   from cambium import ExpandableModel, InterleavedExpansion

   model = ExpandableModel.from_pretrained("google/gemma-2b")
   model.expand(InterleavedExpansion(num_layers=4))

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
