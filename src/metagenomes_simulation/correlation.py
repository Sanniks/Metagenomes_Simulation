import scipy.stats as ss
import numpy as np
import pandas as pd

from itertools import combinations
from metagenomes_simulation.normalization import remove_constant_otus

# Parallel calculation
from pathos.multiprocessing import ProcessingPool as Pool


def measure_correlation_coefficient(x, y, correlation_method):
    """Calculate the correlation coefficient using the selected method.
    -Pearson correlation: r = cov(x,y)/(std(x)std(y)) -Spearman's
    correlation coefficient: rho = pearson_corralation(rank(x),rank(y))
    where rank(x) transforms the values in the x array into their
    corresponding ranks in the sorted order. -Kendall's correlation
    coefficient: tau = (C - D) / sqrt((C + D + T_x)(C + D + T_y)) where: C =
    number of concordant pairs (pairs where x and y change in the same
    direction) D = number of discordant pairs (pairs where x and y change in
    opposite directions) T_x = number of ties only in x T_y = number of ties
    only in y

    Args:
        x (array): First set of data.
        y (array): Second set of data.
        correlation_method (str): Correlation method: "Pearson", "Spearman",
            or "Kendall".

    Returns:
        float: Correlation coefficient between x and y. Between -1 and 1.

    Raises:
        `ValueError`: If the correlation method is not valid.
    """
    correlation_method = correlation_method.casefold()
    if correlation_method == "pearson":
        r, _ = ss.pearsonr(x, y)
        return r
    elif correlation_method == "spearman":
        rho, _ = ss.spearmanr(x, y)
        return rho
    elif correlation_method == "kendall":
        tau, _ = ss.kendalltau(x, y)
        return tau
    else:
        raise ValueError(
            "Incorrect value in correlation_method."
            " Only Pearson, Spearman or Kendall are allowed"
            " in correlation_method. "
        )


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
    otu_df_processed = remove_constant_otus(otu_df)
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
    otu_df_processed = remove_constant_otus(otu_df)
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


def measure_correlation_all_methods(otu_df, num_processes=4):

    correlation_methods = ("Pearson", "Spearman", "Kendall")
    if num_processes < 1:
        raise ValueError("num_processes must be at least 1.")

    # Remove constant OTUs
    otu_df_processed = remove_constant_otus(otu_df)
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

    # Array containing all combination of OTUs for the
    # calculation of every correlation coefficients
    otu_combinations = combinations(range(len(otu_names)), 2)
    tasks = list(otu_combinations)

    # Divide the calculation in independet blocks
    n = len(tasks)  # Total number of calculations
    block_size = (n + num_processes - 1) // num_processes
    blocks = [tasks[i : i + block_size] for i in range(0, n, block_size)]

    correlation_matrix_dict = {}

    for correlation_method in correlation_methods:
        # Create empty numpy matrix
        correlation_matrix = np.eye(len(otu_names))

        # Arguments to pass to the function used in the parallel calculation
        arguments_per_block = [
            (otu_data, block, correlation_method) for block in blocks
        ]
        # Parallelization:
        with Pool(nodes=num_processes) as pool:
            results = pool.map(
                _compute_correlation_parallel, arguments_per_block
            )

        for result in results:
            for i, j, corr_coef in result:
                correlation_matrix[i, j] = corr_coef
                correlation_matrix[j, i] = corr_coef

        correlation_matrix_dict[correlation_method] = pd.DataFrame(
            correlation_matrix,
            index=otu_names,
            columns=otu_names,
        )

    return correlation_matrix_dict
