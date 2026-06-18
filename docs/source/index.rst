Metagenomes Simulation
======================

``metagenomes_simulation`` is a Python project for studying how data
processing choices can bias microbial correlation networks inferred from
metagenomic abundance data.

The project focuses on a central problem in metagenomic network analysis:
relative abundance data are compositional. Each sample lies in a simplex,
because all abundances are constrained to sum to one. This constraint can
create apparent associations between taxa even when no direct biological
interaction is present.

Project Goal
------------

The main goal is to identify whether, and how, the construction of the
analysis dataset introduces bias in microbial association networks.

In particular, the project compares the effect of:

* abundance normalization;
* compositional constraints;
* noise added to real data;
* increased sparsity;
* correlation method selection;
* threshold selection during network construction.

Scientific Context
------------------

Microbial abundance tables are usually represented as samples by taxa. When
absolute counts are converted to relative abundances, every sample satisfies:

.. math::

   \sum_{i=1}^{N} x_i = 1

This means that the data do not freely occupy :math:`R^N`. Instead, they live
in an :math:`N-1` dimensional simplex.

Because of this, classical correlation methods such as Pearson, Spearman and
Kendall can be affected by the compositional constraint. One way to reduce
this issue is to use log-ratio transformations, such as the centered log-ratio
transformation (CLR):

.. math::

   clr(x_i) = \log \left( \frac{x_i}{g(x)} \right)

where :math:`g(x)` is the geometric mean of the sample.

The CLR transformation maps compositional data into a real-valued space with
zero-sum coordinates. This makes the data more suitable for methods based on
Euclidean geometry, while still requiring careful handling of zeros.

Analysis Workflow
-----------------

The intended workflow is:

1. Load real metagenomic abundance data.
2. Select a subject or subset of samples.
3. Generate controlled variants of the real data:

   * original relative abundances;
   * noisy data;
   * sparse data;
   * noisy and sparse data.

4. Compare raw relative and CLR-transformed abundances.
5. Compute correlation matrices and build microbial networks with different threshold.
7. Measure network properties across thresholds.
8. Identify which metrics are stable and which are sensitive to dataset
   construction choices.

Repository Structure
--------------------

The repository is organized as follows:

.. code-block:: text

   data/           Input and processed datasets
   notebooks/      Exploratory and reproducible analyses
   src/            Python package source code
   tests/          Unit tests
   docs/           Sphinx documentation

References
----------

* Fuschi, A.; Merlotti, A.; Tran, T.D.B.; Nguyen, H.; Weinstock, G.M.;
  Remondini, D. Correlation Measures in Metagenomic Data: The Blessing of
  Dimensionality. *Applied Sciences*, 2025.
* Zhou, W., Sailani, M.R., Contrepois, K. et al. Longitudinal multi-omics of
  host-microbe dynamics in prediabetes. *Nature*, 2019.

Contents
--------

.. toctree::
   :maxdepth: 2

   api
