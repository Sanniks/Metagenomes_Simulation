import numpy as np
import pandas as pd
import igraph as ig
from itertools import combinations

from metagenomes_simulation.correlation import measure_correlation_coefficient
from metagenomes_simulation.data_processing import clean_otus_df

# Parallel calculation
from pathos.multiprocessing import ProcessingPool as Pool

# --------------------------------------------------------------
# Correlations matrix
# --------------------------------------------------------------


def create_correlation_matrix(otu_df, correlation_method="Pearson"):
    """Create a matrix cointaining the correlation coefficients between
    different OTUs using abundances data.

    Correlation_method allow to choose between Pearson, Spearman and Kendall
    coefficients.

    If the data processing return a DataFrame with less then 2 otus,
    correlation is impossible and the function returns an empty df.

    Args:
        otu_df (DataFrame): DataFrame containing the abundances of every
            sample.
        correlation_method (str, optional): Correlation method: "Pearson",
            "Spearman", or "Kendall". Defaults to "Pearson".

    Returns:
        DataFrame: Correlation matrix between OTUs.
    """

    # Remove constant OTUs
    otu_df_processed = clean_otus_df(otu_df)
    # Take the name of each OTU
    otu_names = otu_df_processed.columns
    # Convert from pandas dataframe to numpy matrix
    otu_data = otu_df_processed.to_numpy()

    # Return empty dataframe for less than 2
    # non constant OTUs
    if len(otu_names) < 2:
        empty_dataframe = pd.DataFrame(
            np.eye(len(otu_names)),
            index=otu_names,
            columns=otu_names,
        )
        return empty_dataframe

    # Create empty dataframe
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

            corr_coeff = measure_correlation_coefficient(
                otu_data[:, i],
                otu_data[:, j],
                correlation_method,
            )

            correlation_matrix.loc[otu_1, otu_2] = corr_coeff
            correlation_matrix.loc[otu_2, otu_1] = corr_coeff

    return correlation_matrix


# --------------------------------------------------------------
# Parallel version of correlation matrix creation
# --------------------------------------------------------------


def _compute_correlation_parallel(args):
    """Compute correlation coefficients for a block of OTU index pairs.

    Args:
        args (tuple): A tuple containing:
        otu_data (ndarray): 2D array of OTU abundances.
        block (list[tuple[int, int]]): List of index pairs for correlations.
        correlation_method (str): Correlation method to use.

    Returns:
        list[tuple[int, int, float]]: Correlation results for each index
            pair.
    """
    otu_data, block, correlation_method = args

    results = []
    for i, j in block:
        # compute the correlation coefficient for the pair of OTUs
        corr = measure_correlation_coefficient(
            otu_data[:, i],
            otu_data[:, j],
            correlation_method,
        )
        results.append((i, j, corr))
    return results


def create_correlation_matrix_parallel(
    otu_df, correlation_method="Pearson", num_processes=4
):
    """Create a matrix cointaining the correlation coefficients between
    different OTUs using abundances data.

    Correlation_method allow to choose between Pearson, Spearman and Kendall
    coefficients.

    It allows for faster calculation time, by performing some calculations
    in parallel.

    If the data processing return a DataFrame with less then 2 otus,
    correlation is impossible and the function returns an empty df.

    Args:
        otu_df (DataFrame): DataFrame containing the abundances of every
            sample.
        correlation_method (str, optional): Correlation method: "Pearson",
            "Spearman", or "Kendall". Defaults to "Pearson".
        num_processes (int, optional): Number of parallel processes.
            Defaults to 4.

    Returns:
        DataFrame: Correlation matrix between OTUs.
    """
    if num_processes < 1:
        raise ValueError("num_processes must be at least 1.")

    # Remove constant OTUs
    otu_df_processed = clean_otus_df(otu_df)
    # Take the name of each OTU
    otu_names = otu_df_processed.columns

    # Convert from pandas dataframe to numpy matrix
    otu_data = otu_df_processed.to_numpy()

    # Return empty dataframe for less than 2
    # non constant OTUs
    if len(otu_names) < 2:
        empty_dataframe = pd.DataFrame(
            np.eye(len(otu_names)),
            index=otu_names,
            columns=otu_names,
        )
        return empty_dataframe

    # Create empty numpy matrix
    correlation_matrix = np.eye(len(otu_names))

    # Array containing all combination of OTUs for the
    # calculation of every correlation coefficients
    otu_combinations = combinations(range(len(otu_names)), 2)
    tasks = list(otu_combinations)

    # Divide the calculation in independet blocks
    n = len(tasks)  # Total number of calculations
    block_size = (n + num_processes - 1) // num_processes
    blocks = [tasks[i : i + block_size] for i in range(0, n, block_size)]

    # Arguments to pass to the function used in the parallel calculation
    arguments_per_block = [
        (otu_data, block, correlation_method) for block in blocks
    ]
    # Parallelization:
    with Pool(nodes=num_processes) as pool:
        results = pool.map(_compute_correlation_parallel, arguments_per_block)

    for result in results:
        for i, j, corr_coef in result:
            correlation_matrix[i, j] = corr_coef
            correlation_matrix[j, i] = corr_coef

    # Convert numpy matrix in pandas dataframe
    correlation_matrix_df = pd.DataFrame(
        correlation_matrix,
        index=otu_names,
        columns=otu_names,
    )

    return correlation_matrix_df


# --------------------------------------------------------------
# Correlations network from correlation matrix using igraph
# --------------------------------------------------------------


def create_network_from_correlation_matrix(
    correlation_matrix,
    threshold=0.1,
    remove_isolated=False,
    binary_network=False,
):
    """Create an igraph network from a correlation matrix.

    It also removes all edges of weigth lower than the threshold.

    Args:
        correlation_matrix (pd.DataFrame): Matrix containing the correlation
            coefficients between OTUs.
        threshold (float, optional): Threshold to cut the smaller edges.
            Defaults to 0.1.
        remove_isolated (bool, optional): Allows to remove any isolated
            nodes. Defaults to False.
        binary_network (bool, optional): All edges above the threshold have
            weigth 1 instead of the correlation coefficient. Defaults to
            False.

    Returns:
        ig.Graph: igraph network built from the correlation matrix.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    # Copy the input matrix
    adjacency = correlation_matrix.copy()

    # Removes self-loops
    np.fill_diagonal(adjacency.values, 0)

    # If an edge is lower to the threshold, it removes it
    adjacency[adjacency.abs() < threshold] = 0

    if binary_network:
        # If an edge is higher than the threshold, it makes it 1
        adjacency[adjacency != 0] = 1
    if remove_isolated:
        # Keeps only nodes with edges
        nodes_accepted = adjacency.abs().max(axis=1) > 0
        filtered_correlation_matrix = adjacency.loc[
            nodes_accepted, nodes_accepted
        ]
    else:
        filtered_correlation_matrix = adjacency

    # If every node is removed
    if filtered_correlation_matrix.shape[0] == 0:
        return ig.Graph(n=0)

    # Create a igraph network
    graph_ig = ig.Graph.Weighted_Adjacency(
        matrix=filtered_correlation_matrix.values,
        mode="undirected",
        attr="weight",
        loops=False,
    )

    # Assign a name to each OTU node
    graph_ig.vs["name"] = [
        str(col) for col in filtered_correlation_matrix.columns
    ]
    return graph_ig
