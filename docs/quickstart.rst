Quick Start Guide
=================

Installation
------------

Install Cambium from PyPI::

   pip install cambium

For training dependencies::

   pip install "cambium[train]"

For development::

   pip install "cambium[dev]"

Basic Usage
-----------

Load and Expand a Model
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cambium import ExpandableModel, InterleavedExpansion

   # 1. Load a pretrained model
   model = ExpandableModel.from_pretrained("google/gemma-2b")

   # 2. Expand with 4 new transformer blocks
   model.expand(InterleavedExpansion(num_layers=4, initialization="identity"))

   # 3. Get the expanded model for training
   expanded = model.get_model()

Staged Training
~~~~~~~~~~~~~~~

.. code-block:: python

   from cambium.training import StagedTrainer

   trainer = StagedTrainer(model)
   trainer.add_phase(name="warmup", freeze="original", lr=1e-4, epochs=2)
   trainer.add_phase(name="finetune", freeze="none", lr=1e-6, epochs=1)
   trainer.train(train_dataloader, eval_dataloader)

Save and Load
~~~~~~~~~~~~~

.. code-block:: python

   # Save
   model.save_expanded("./my-expanded-model")

   # Load later
   from cambium import ExpandableModel
   model = ExpandableModel.load_expanded("./my-expanded-model")
