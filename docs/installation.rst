Installation
============

Requirements
------------

- Python 3.8 or newer
- PyTorch 2.0 or newer
- Hugging Face ``transformers`` 4.35+

Basic Installation
------------------

Install Cambium from PyPI::

   pip install cambium

This installs the core library with the minimal runtime dependencies:

- ``torch`` — deep-learning framework
- ``transformers`` — Hugging Face model loading and saving
- ``accelerate`` — device placement and mixed-precision helpers
- ``safetensors`` — fast, safe checkpoint serialization
- ``numpy`` — numerical utilities

Optional Dependencies
---------------------

Cambium provides optional extras for training, development, and advanced use cases.

Training extras
~~~~~~~~~~~~~~~

Install if you plan to use ``StagedTrainer``, ``TrainingUtilities``, or integrate with TRL::

   pip install "cambium[train]"

Extra packages installed:

- ``datasets`` — loading and preprocessing training data
- ``trl`` — TRL ``SFTTrainer`` and other RLHF trainers
- ``wandb`` — experiment tracking
- ``bitsandbytes`` — 8-bit / 4-bit quantization support

Development extras
~~~~~~~~~~~~~~~~~~

Install if you are contributing to Cambium or building the documentation::

   pip install "cambium[dev]"

Extra packages installed:

- ``pytest`` / ``pytest-cov`` — testing
- ``black`` / ``isort`` / ``mypy`` — linting and formatting
- ``pre-commit`` — git hooks
- ``sphinx`` / ``sphinx-rtd-theme`` / ``myst-parser`` — documentation build

All extras
~~~~~~~~~~

Install everything at once::

   pip install "cambium[all]"

This includes ``[dev]``, ``[train]``, and the optional ``unsloth`` integration.

Editable Installation (for Contributors)
----------------------------------------

Clone the repository and install in editable mode::

   git clone https://github.com/SorawitChok/Cambium.git
   cd cambium
   pip install -e ".[dev]"

If you use ``pre-commit``::

   pre-commit install

Verifying the Installation
--------------------------

After installation, verify that Cambium loads correctly::

   python -c "import cambium; print(cambium.__version__)"

Troubleshooting
---------------

ImportError: transformers library required
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you see::

   ImportError: transformers library required

Install the core dependencies::

   pip install transformers accelerate safetensors

CUDA / torch version mismatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cambium does not pin a specific CUDA-enabled PyTorch wheel. If you encounter CUDA errors, reinstall PyTorch matching your CUDA version from `pytorch.org <https://pytorch.org/>`_.

Out-of-memory during model loading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Small models such as ``HuggingFaceTB/SmolLM2-135M`` (135M parameters) load comfortably on most CPUs and GPUs. If you expand larger models, use ``dtype=torch.float16`` or enable 8-bit loading via ``transformers``::

   model = ExpandableModel.from_pretrained(
       "meta-llama/Llama-2-7b-hf",
       dtype=torch.float16,
       device_map="auto",
   )

Gated models require authentication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some models (e.g., ``meta-llama/Llama-2-7b-hf``, ``google/gemma-2b``) require Hugging Face authentication. Log in with::

   huggingface-cli login

For documentation and quick experiments we recommend the ungated ``HuggingFaceTB/SmolLM2-135M``.

Optional integration not found
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If ``trl``, ``peft``, or ``datasets`` is referenced in an example but not installed, you will see a ``ModuleNotFoundError``. Install the missing library::

   pip install trl peft datasets
