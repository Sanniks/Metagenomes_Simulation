import numpy as np
import pandas as pd
from pynetcor.cor import corrcoef

from metagenomes_simulation.normalization import remove_constant_otus

# --------------------------------------------------------------
# Correlations matrix
# --------------------------------------------------------------


def create_correlation_matrix(otu_df, correlation_method="Pearson"):
    correlation_method = correlation_method.casefold()

    if correlation_method not in ("pearson", "spearman", "kendall"):
        raise ValueError("Invalid correlation method.")

    # Remove constant OTUs
    otu_df_processed = remove_constant_otus(otu_df)
    otu_names = otu_df_processed.columns

    # Handle fewer than two OTUs
    if len(otu_names) < 2:
        return pd.DataFrame(
            np.eye(len(otu_names)),
            index=otu_names,
            columns=otu_names,
        )

    # Compute correlation matrix
    correlation_matrix = corrcoef(
        otu_df_processed.to_numpy(dtype=np.float64).T,
        method=correlation_method,
    )
    np.fill_diagonal(correlation_matrix, 1.0)
    return pd.DataFrame(
        correlation_matrix,
        index=otu_names,
        columns=otu_names,
    )


def create_correlation_matrix_all_methods(otu_df):
    correlation_methods = ("Pearson", "Spearman", "Kendall")

    correlation_matrix_dict = {}

    for correlation_method in correlation_methods:

        correlation_matrix_dict[correlation_method] = (
            create_correlation_matrix(
                otu_df,
                correlation_method=correlation_method,
            )
        )

    return correlation_matrix_dict
