# Metagenomes Simulation

This project investigates how microbial correlation networks inferred from metagenomic sequencing data are affected by different data processing choices and analysis pipelines.

The main objective is to study biases introduced during the construction of microbial networks, considering preprocessing steps, normalization methods, and threshold selection.

## Installation

Clone the repository:

```bash
git clone https://github.com/Sanniks/Metagenomes_Simulation.git
cd Metagenomes_Simulation
```

Install the package:

```bash
python -m pip install -e .
```

Python 3.10 or later is required.

## Project Goals

- Simulate realistic correlation matrices, considered as ground truth, from which we simulate the abundances.
- Simulate metagenomic sequencing with a Poisson-distributed sequencing depth.
- Construct correlation networks from abundance data.
- Study how the generation of the microbial abundances changes the metrics ( blocks, community edge density, sequencing depth, correlation intensity).
- Study how the creation of the correlation network from the simulated sequencing changes the metrics (Different types of normalization, different correlation coefficient calculation, different thresholds, different community detection algorithm).

## Metrics and Variables

### Data generation

- Community blocks: Describe how many communities and how many members (e.g. [15,10] represents two communities of 15 and 10 OTUs)

- Within Density: density of correlations inside a community block (e.g. 0.95, corresponds to a target density of 95% of all possible within-community correlations)

- Correlation range: represents the range from which correlation coefficient are initially sampled. Correlation coefficients are rescaled to ensure that the correlation matrix is positive semidefinite.

- Number of samples: How many samples of abundances are taken.

- Mean abundance: Mean parameter of the simulated abundance
  distribution.

- Abundance variance: Variance parameter of the simulated
  abundance distribution.

- Sequencing depth: Expected total number of sequencing reads
  per sample.

### Data Analysis

- Abundance normalization:
  - Absolute (raw sequencing counts)
  - Relative abundance (L1 normalization)
  - Centered log-ratio transformation (CLR)

- Correlation coefficient calculation method:
  - Pearson
  - Spearman
  - Kendall
  - SparCC (To be implemented)

- Correlation threshold

- Community detection algorithm:
  - Louvain
  - Leiden
  - Infomap
  - Walktrap

### Metrics

Metrics are used to quantify how much the reconstructed networks are different from the starting correlation network.

- Correlation difference with ground truth:
  - Bias
  - Mean Absolute Error (MAE)
  - Root Mean Square Error (RMSE)

- Network reconstruction:
  - F1 Score
  - Jaccard Similarity
  - Precision
  - Recall

- Community reconstruction:
  - Adjusted Rand Index (ARI)
  - Normalized Mutual Information (NMI)
  - Variation of Information (VI)

## Dataset

The main analyses are based on synthetic microbial abundance
data generated from a known ground-truth correlation matrix.

The repository also includes exploratory analysis of publicly
available real-world data from the study:

> **Longitudinal multi-omics of host–microbe dynamics in prediabetes**

## Technologies

The project is developed primarily in **Python** and makes use of scientific computing libraries such as:

- NumPy
- pandas
- SciPy
- Matplotlib
- igraph
- leidenalg
- pathos

## Repository Structure

```text
data/           Input and processed datasets
notebooks/      Exploratory and reproducible analyses
src/            Python package source code
tests/          Unit and integration tests
docs/           Sphinx documentation
.github/        Continuous integration workflows
```

## Testing

Install the testing dependencies:

```bash
python -m pip install -e ".[test]"
```

Run the tests:

```bash
pytest
```

Measure test coverage:

```bash
pytest --cov=metagenomes_simulation --cov-report=term-missing
```

Continuous integration is configured through GitHub Actions,
with automated testing on Python 3.10, 3.11, and 3.12.

## License

This project is licensed under the MIT License.
See [LICENCE.txt](LICENCE.txt) for details.

## References

- Fuschi, A.; Merlotti, A.; Tran, T.D.B.; Nguyen, H.; Weinstock, G.M.; Remondini, D. Correlation Measures in Metagenomic Data: The Blessing of Dimensionality. Appl. Sci. 2025, 15, 8602. <https://doi.org/10.3390/app15158602>

- Zhou, W., Sailani, M.R., Contrepois, K. et al. Longitudinal multi-omics of host–microbe dynamics in prediabetes. Nature 569, 663–671 (2019). <https://doi.org/10.1038/s41586-019-1236-x>

- Becsei Á, Fuschi A, Otani S, Kant R, Weinstein I, Alba P, Stéger J, Visontai D, Brinch C, de Graaf M, Schapendonk CME, Battisti A, De Cesare A, Oliveri C, Troja F, Sironen T, Vapalahti O, Pasquali F, Bányai K, Makó M, Pollner P, Merlotti A, Koopmans M, Csabai I, Remondini D, Aarestrup FM, Munk P. Time-series sewage metagenomics distinguishes seasonal, human-derived and environmental microbial communities potentially allowing source-attributed surveillance. Nat Commun. 2024 Aug 30;15(1):7551. doi: 10.1038/s41467-024-51957-8. Erratum in: Nat Commun. 2024 Oct 17;15(1):8953. doi: 10.1038/s41467-024-53282-6. PMID: 39215001; PMCID: PMC11364805.
