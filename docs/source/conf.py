# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Metagenomes Simulation"
copyright = "2026, Carlo Mattia Lovecchio"
author = "Carlo Mattia Lovecchio"
release = "0.0.2"
master_doc = "index"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_rtd_theme",
    #'rst2pdf.pdfbuilder',
    "nbsphinx",
    "IPython.sphinxext.ipython_console_highlighting",
]

templates_path = []
exclude_patterns = ["_build", "**.ipynb_checkpoints"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = []

# -- Options for PDF output --------------------------------------------------

# Grouping the document tree into LaTeX files. List of tuples# (source start file, target name, title, author, documentclass [howto/manual]).
latex_engine = "xelatex"
latex_documents = [
    (
        "index",
        "metagenomes_simulation.tex",
        "Metagenomes Simulation - Analysis of biases in metagenomes correlation networks",
        "Carlo Mattia Lovecchio",
        "manual",
    ),
]
latex_show_pagerefs = True
latex_domain_indices = False

pdf_documents = [
    (
        "index",
        "metagenomes_simulation",
        "Metagenomes Simulation - Analysis of biases in metagenomes correlation networks",
        "Carlo Mattia Lovecchio",
    ),
]

nbsphinx_input_prompt = "In [%s]:"
nbsphinx_kernel_name = "python3"
nbsphinx_output_prompt = "Out[%s]:"
