# Configuration file for the Sphinx documentation builder.
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import shutil
import sys

# Add the project root to the path so Sphinx can import cambium
sys.path.insert(0, os.path.abspath(".."))

# Copy example markdown files into the docs source tree so Sphinx can include them.
_examples_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples"))
_examples_dst = os.path.abspath(os.path.join(os.path.dirname(__file__), "examples"))
if os.path.exists(_examples_src):
    if os.path.exists(_examples_dst):
        shutil.rmtree(_examples_dst)
    shutil.copytree(_examples_src, _examples_dst)


# -- Project information -----------------------------------------------------
project = "Cambium"
copyright = "2026, Sorawit Chokphantavee, Sirawit Chokphantavee, and Cambium Team"
author = "Sorawit Chokphantavee, Sirawit Chokphantavee, and Cambium Team"
release = "0.1.0"
version = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "myst_parser",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = "Cambium Documentation"
html_short_title = "Cambium"

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# MyST-Parser settings for Markdown example files
myst_enable_extensions = ["colon_fence"]

# HTML theme options
html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# Suppress duplicate object warnings when autosummary and automodule overlap
suppress_warnings = ["autodoc.duplicate_object"]
