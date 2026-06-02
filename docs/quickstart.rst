Quick Start Guide
=================

This guide walks through a complete expansion workflow in a few minutes.

Load a Model
------------

Cambium wraps Hugging Face models with the :class:`~cambium.models.expandable.ExpandableModel` class.

.. code-block:: python

   import torch
   from cambium import ExpandableModel

   model = ExpandableModel.from_pretrained(
       "HuggingFaceTB/SmolLM2-135M",
       dtype=torch.float32,
   )

We use ``HuggingFaceTB/SmolLM2-135M`` because it is small, ungated, and works on CPU. The ``dtype=torch.float32`` argument is passed through to ``AutoModelForCausalLM.from_pretrained``; on CPU this avoids BFloat16/Float mismatches that can cause runtime errors.

Expand with New Layers
----------------------

Insert two new transformer blocks with near-identity initialization. This preserves the original model's behavior before training::

   from cambium import InterleavedExpansion

   model.expand(InterleavedExpansion(num_layers=2, initialization="identity"))

Cambium automatically:

1. Chooses insertion positions so new blocks are spread evenly.
2. Creates blocks matching the base architecture (e.g., ``LlamaDecoderLayer``).
3. Initializes new weights so the block acts like a near pass-through.
4. Updates ``layer_idx`` and model config to keep generation consistent.

Freeze Original Weights
-----------------------

Freeze all pretrained parameters so only the new layers are trainable::

   model.freeze_original()

Run a Forward Pass
------------------

The expanded model is a standard PyTorch module. You can run inference immediately (before training the new layers behave almost like the original)::

   from transformers import AutoTokenizer

   tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
   if tokenizer.pad_token is None:
       tokenizer.pad_token = tokenizer.eos_token

   inputs = tokenizer("Hello, world!", return_tensors="pt")
   outputs = model.get_model()(**inputs)

Save and Reload
---------------

Save the expanded weights and metadata, then load them later::

   # Save
   model.save_expanded("./my-expanded-model")

   # Load
   from cambium import ExpandableModel
   reloaded = ExpandableModel.load_expanded("./my-expanded-model")

``save_expanded`` uses Hugging Face's native ``save_pretrained``, so tied embeddings and ``safetensors`` shared tensors are handled correctly. ``load_expanded`` re-applies any registered expansions and reloads orphan weights for strategies such as parallel adapters.

Validate the Expansion
----------------------

Run a structural sanity check::

   report = model.validate()
   print(model.get_expansion_report())

Full Script
-----------

Here is the complete script::

   import torch
   from cambium import ExpandableModel, InterleavedExpansion
   from transformers import AutoTokenizer

   # 1. Load
   model = ExpandableModel.from_pretrained(
       "HuggingFaceTB/SmolLM2-135M",
       dtype=torch.float32,
   )

   # 2. Expand
   model.expand(InterleavedExpansion(num_layers=2, initialization="identity"))

   # 3. Freeze original weights
   model.freeze_original()

   # 4. Quick inference check
   tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
   if tokenizer.pad_token is None:
       tokenizer.pad_token = tokenizer.eos_token

   inputs = tokenizer("The future of AI is", return_tensors="pt")
   with torch.no_grad():
       outputs = model.get_model().generate(
           **inputs,
           max_new_tokens=20,
           do_sample=False,
       )
   print(tokenizer.decode(outputs[0], skip_special_tokens=True))

   # 5. Save
   model.save_expanded("./my-expanded-model")

Next Steps
----------

- :doc:`concepts` — understand the four architectural layers
- :doc:`strategies` — explore width expansion, parallel adapters, and custom blocks
- :doc:`training` — learn staged training and progressive unfreezing
- :doc:`examples` — run the full example walkthroughs
