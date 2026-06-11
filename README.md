# Metagenomic Network Analysis

This project investigates how microbial interaction networks inferred from metagenomic sequencing data are affected by different data processing choices and analysis pipelines.

The main objective is to study biases introduced during the construction of microbial networks, considering preprocessing steps, normalization methods, and threshold selection strategies.

## Project Goals

- Analyze real metagenomic datasets.
- Construct microbial association networks from abundance data.
- Study how different preprocessing approaches influence inferred network topology.
- Evaluate the impact of normalization techniques.
- Investigate how threshold selection affects network structure and biological interpretation.
- Identify potential sources of bias introduced during data analysis.

## Dataset

The analyses are based on publicly available real-world data from the study:

> **Longitudinal multi-omics of host–microbe dynamics in prediabetes**

## Planned Analyses

Some of the analyses performed in this project include:

- Data quality assessment and preprocessing
- Comparison of normalization methods (e.g., relative abundances, L1 and CLR transformation)
- Evaluation of different correlation thresholds
- Network visualization and topological analysis
- Sensitivity analysis of network properties with respect to preprocessing choices

## Technologies

The project is developed primarily in **Python** and makes use of scientific computing libraries such as:

- NumPy
- pandas
- SciPy
- Matplotlib
- scikit-learn
- NetworkX

## Repository Structure

```bash
data/           # Input datasets
notebooks/      # Exploratory analyses
src/            # Source code
tests/          # Unit tests
docs/           # Documentation
```

## License

This project is intended for academic and research purposes.

## References

-Fuschi, A.; Merlotti, A.; Tran, T.D.B.; Nguyen, H.; Weinstock, G.M.; Remondini, D. Correlation Measures in Metagenomic Data: The Blessing of Dimensionality. Appl. Sci. 2025, 15, 8602. <https://doi.org/10.3390/app15158602>

-Zhou, W., Sailani, M.R., Contrepois, K. et al. Longitudinal multi-omics of host–microbe dynamics in prediabetes. Nature 569, 663–671 (2019). <https://doi.org/10.1038/s41586-019-1236-x>

-Becsei Á, Fuschi A, Otani S, Kant R, Weinstein I, Alba P, Stéger J, Visontai D, Brinch C, de Graaf M, Schapendonk CME, Battisti A, De Cesare A, Oliveri C, Troja F, Sironen T, Vapalahti O, Pasquali F, Bányai K, Makó M, Pollner P, Merlotti A, Koopmans M, Csabai I, Remondini D, Aarestrup FM, Munk P. Time-series sewage metagenomics distinguishes seasonal, human-derived and environmental microbial communities potentially allowing source-attributed surveillance. Nat Commun. 2024 Aug 30;15(1):7551. doi: 10.1038/s41467-024-51957-8. Erratum in: Nat Commun. 2024 Oct 17;15(1):8953. doi: 10.1038/s41467-024-53282-6. PMID: 39215001; PMCID: PMC11364805.
