import numpy as np
import pandas as pd
import networkx as nx
from itertools import combinations

from metagenomes_simulation.correlation import correlation_coefficient
from metagenomes_simulation.data_processing import clean_otus_df

# Parallel calculation
from pathos.multiprocessing import ProcessingPool as Pool


# Correlations matrix
def create_correlation_matrix(otu_df, correlation_method="Pearson"):
    """Create a matrix cointaining the correlation coefficients between different OTUs.
    Correlation method allow to choose between Pearson, Spearman and Kendall coefficients.

    Args:
        otu_df (DataFrame): DataFrame containing the abundances of every sample.
        correlation_method (str, optional): Correlation method: "Pearson", "Spearman", or "Kendall". Defaults to "Pearson".

    Returns:
        DataFrame: Correlation matrix between OTUs.
    """
    otu_df_processed = clean_otus_df(otu_df)
    otu_names = otu_df_processed.columns

    data = otu_df_processed.to_numpy()

    correlation_matrix = pd.DataFrame(
        np.eye(len(otu_names)),
        index=otu_names,
        columns=otu_names,
    )

    # Measure correlation corefficient between every pair of otu
    for i in range(len(otu_names)):
        for j in range(i + 1, len(otu_names)):
            otu_1 = otu_names[i]
            otu_2 = otu_names[j]

            corr_coeff = correlation_coefficient(
                data[:, i],
                data[:, j],
                correlation_method,
            )

            correlation_matrix.loc[otu_1, otu_2] = corr_coeff
            correlation_matrix.loc[otu_2, otu_1] = corr_coeff

    return correlation_matrix


# Parallel version of correlation matrix


def _compute_correlation_parallel(args):
    """Compute correlation coefficients for a block of OTU index pairs.

    Args:
        args (tuple): A tuple containing:
            otu_data (ndarray): 2D array of OTU abundances.
            block (list[tuple[int, int]]): List of index pairs for correlations.
            correlation_method (str): Correlation method to use.

    Returns:
        list[tuple[int, int, float]]: Correlation results for each index pair.
    """
    otu_data, block, correlation_method = args

    results = []
    for i, j in block:
        # compute the correlation coefficient for the pair of OTUs
        corr = correlation_coefficient(
            otu_data[:, i],
            otu_data[:, j],
            correlation_method,
        )
        results.append((i, j, corr))
    return results


def create_correlation_matrix_parallel(otu_df, correlation_method="Pearson", num_processes=4):
    """Create a matrix cointaining the correlation coefficients between different OTUs.
    Correlation method allow to choose between Pearson, Spearman and Kendall coefficients.

    It allows for faster calculation time, by performing some calculations in parallel
    Args:
        otu_df (DataFrame): DataFrame containing the abundances of every sample.
        correlation_method (str, optional): Correlation method: "Pearson", "Spearman", or "Kendall". Defaults to "Pearson".
        num_processes (int, optional): Number of parallel processes. Defaults to 4.

    Returns:
        DataFrame: Correlation matrix between OTUs.
    """
    if num_processes < 1:
        raise ValueError("num_processes must be at least 1.")

    otu_df_processed = clean_otus_df(otu_df)

    otu_names = otu_df_processed.columns
    otu_data = otu_df_processed.to_numpy()

    correlation_matrix = np.eye(len(otu_names))

    # Array containing all combination of OTUs for the
    # calculation of every correlation coefficients
    otu_combinations = combinations(range(len(otu_names)), 2)
    tasks = list(otu_combinations)

    #
    n = len(tasks)
    block_size = (n + num_processes - 1) // num_processes
    blocks = [tasks[i : i + block_size] for i in range(0, n, block_size)]

    arguments_per_block = [(otu_data, block, correlation_method) for block in blocks]
    with Pool(nodes=num_processes) as pool:
        results = pool.map(_compute_correlation_parallel, arguments_per_block)

    for result in results:
        for i, j, corr_coef in result:
            correlation_matrix[i, j] = corr_coef
            correlation_matrix[j, i] = corr_coef

    correlation_matrix_df = pd.DataFrame(
        correlation_matrix,
        index=otu_names,
        columns=otu_names,
    )

    return correlation_matrix_df


# Correlations network
def create_network_from_correlation_matrix(
    correlation_matrix,
    threshold=0.1,
):
    """Create a NetworkX graph from a correlation matrix.

    It removes all edges of correlation coeffitients lower than the threshold.

    Args:
        correlation_matrix (np.array): Matrix containing the
        threshold (float, optional): Threshold to cut the smaller edges. Defaults to 0.1.

    Returns:
        nx.Graph: Network built from the correlation matrix.
    """

    G = nx.Graph()

    otu_names = list(correlation_matrix.columns)
    G.add_nodes_from(otu_names)

    for i, otu1 in enumerate(otu_names):
        for otu2 in otu_names[i + 1 :]:
            corr_coef = correlation_matrix.loc[otu1, otu2]

            if abs(corr_coef) >= threshold:
                G.add_edge(otu1, otu2, weight=corr_coef)

    # Remove nodes with no edges
    G.remove_nodes_from(list(nx.isolates(G)))
    return G
