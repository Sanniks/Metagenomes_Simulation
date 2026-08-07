import numpy as np
import pandas as pd

from metagenomes_simulation.data_processing import (
    absolute_to_relative_abundance,
)


def create_block_correlation(
    block_sizes=[15, 20, 15],
    within_correlation=0.5,
    between_correlation=0.1,
):
    """Create a correlation DataFrame used to simulate metagenomic abundances.

    It creates a block matrix made from the argument blocks.

    On the diagonal, every element is 1. In the blocks, the value of every
    element is "within_correlation" Outside the blocks, the value of every
    element is "between_correlation"

    Args:
        block_sizes (list[int]): Number of taxa in each community. Default
            [15, 20, 15].
        within_correlation (float): Correlation between taxa in the same
            community. Default 0.5.
        between_correlation (float): Correlation between taxa in different
            communities. Default 0.1

    Raises:
        `ValueError`: Raises error if a correlation is over 1 or under 0.
        `ValueError`: Raises error if block_sizes is empty or contains
                non-positive integers.

    Returns:
        pd.DataFrame: Block correlation matrix with OTUs in rows and
            columns.
    """
    # Invalid inputs
    if not 0 <= between_correlation <= within_correlation <= 1:
        raise ValueError(
            "correlations must be between 0 and 1 and 'between_correlation'"
            " must be lower than 'within_correlation'"
        )

    if not isinstance(block_sizes, (list, tuple)) or len(block_sizes) == 0:
        raise ValueError("block_sizes must be a non-empty list.")
    if any(not isinstance(size, int) or size <= 0 for size in block_sizes):
        raise ValueError("block_sizes must contain positive integers.")

    # Create matrix of correlations between communities
    total_taxa = sum(block_sizes)
    correlation_matrix = np.full((total_taxa, total_taxa), between_correlation)

    # Create correlation blocks within communities
    start = 0
    for block_size in block_sizes:
        end = start + block_size
        correlation_matrix[start:end, start:end] = within_correlation
        start = end

    # Insert 1 to every diagonal element
    np.fill_diagonal(correlation_matrix, 1)

    n_otus = np.sum(block_sizes)
    otu_names = [f"OTU_{i + 1}" for i in range(n_otus)]

    correlation_matrix_df = pd.DataFrame(
        correlation_matrix,
        index=otu_names,
        columns=otu_names,
    )

    return correlation_matrix_df


def create_sparse_correlation(
    n_taxa=100,
    density=0.05,
    correlation_range=(0.2, 0.6),
    negative_fraction=0.2,
    random_state=None,
):
    """Create a sparse positive semidefinite correlation matrix as a pandas
    dataframe.

    Args:
        n_taxa (int): Number of taxa. Default 100.
        density (float): Fraction of possible edges to create. Default 0.05.
        correlation_range (tuple[float, float]): Minimum and maximum edge
            weights. Default (0.2, 0.6).
        negative_fraction (float): Fraction of edges with a negative weight.
            Default 0.2.
        random_state (int, optional): Seed used for reproducible random
            generation. If None, fresh entropy is used. Default None.

    Raises:
        `ValueError`: Raises error if n_taxa is not a positive integer.
        `ValueError`: Raises error if density or negative_fraction is not
                between 0 and 1.
        `ValueError`: Raises error if correlation_range is invalid.
        `ValueError`: Raises error if random_state is not an integer or
                None.

    Returns:
        pd.DataFrame: Sparse positive semidefinite correlation matrix with
            OTUs in rows and columns.
    """
    # n_taxa check
    if not isinstance(n_taxa, int) or n_taxa <= 0:
        raise ValueError("n_taxa must be a positive integer.")

    # density check
    if not isinstance(density, (int, float)) or not 0 <= density <= 1:
        raise ValueError("density must be between 0 and 1.")

    # correlation_range check
    if (
        not isinstance(correlation_range, (list, tuple))
        or len(correlation_range) != 2
        or not isinstance(correlation_range[0], (int, float))
        or not isinstance(correlation_range[1], (int, float))
        or not 0 <= correlation_range[0] <= correlation_range[1] < 1
    ):
        raise ValueError(
            " correlation_range must contain two ordered values"
            " between 0 inclusive and 1 exclusive. Correlation = 1"
            " means that two taxa are perfectly linearly dependent"
            " and it reduces the dimension of the correlation matrix"
        )

    # negative_fraction check
    if (
        not isinstance(negative_fraction, (int, float))
        or not 0 <= negative_fraction <= 1
    ):
        raise ValueError("negative_fraction must be between 0 and 1.")

    # random_state check
    if random_state is not None and not isinstance(random_state, int):
        raise ValueError("random_state must be an integer or None.")

    rng = np.random.default_rng(random_state)
    correlation_matrix = np.eye(n_taxa)

    # Choose random edges
    rows, columns = np.triu_indices(n_taxa, k=1)
    n_edges = round(density * len(rows))
    selected = rng.choice(len(rows), size=n_edges, replace=False)

    # Assign positive and negative weights
    weights = rng.uniform(
        correlation_range[0], correlation_range[1], size=n_edges
    )
    n_negative = round(negative_fraction * n_edges)
    if n_negative != 0:
        negative_edges = rng.choice(n_edges, size=n_negative, replace=False)
        weights[negative_edges] *= -1

    correlation_matrix[rows[selected], columns[selected]] = weights
    correlation_matrix[columns[selected], rows[selected]] = weights

    # Scale off-diagonal values to guarantee positive
    # semidefiniteness (Normalization)
    row_sum = np.sum(np.abs(correlation_matrix), axis=1)
    row_sum_no_diagonal = row_sum - 1
    maximum_row_sum = np.max(row_sum_no_diagonal)
    if maximum_row_sum > 1:
        off_diagonal = correlation_matrix - np.eye(n_taxa)
        correlation_matrix = np.eye(n_taxa) + off_diagonal / maximum_row_sum

    otu_names = [f"OTU{i + 1}" for i in range(n_taxa)]

    correlation_matrix_df = pd.DataFrame(
        correlation_matrix,
        index=otu_names,
        columns=otu_names,
    )

    return correlation_matrix_df


def simulate_absolute_abundances_ground_truth(
    correlation_matrix,
    n_sample=10,
    mean_abundance=1e3,
    var_abundance=1,
    taxa_var_abundance=0.5,
    random_state=None,
):
    """Simulate absolute abundances from a multivariate log-normal
    distribution.

    The function first generates a multivariate normal vector for each
    sample:

    M_i ~ LogNormal(log(mean_abundance), taxa_var_abundance)

    Where M_i is the baseline abundance of each taxon. taxa_var_abundance
    controls how different is the baseline abundance between different
    taxon.

    Z ~ N(log(M), var_abundance * correlation_matrix)

    Z is the vector of log-abundances. It then transforms every value with
    the exponential function:

    X = exp(Z)

    The first log-normal distribution assigns each taxon a fixed median
    abundance M_i, so some taxa remain systematically more abundant than
    others across all samples, to follow real data behaviour. Setting
    taxa_var_abundance to zero assigns mean_abundance as the median of every
    taxon.

    X follows a multivariate log-normal distribution and contains continuous
    positive abundances. The input correlation dataframe describes the
    dependence between taxa on the normal scale, while var_abundance
    controls the variation of each taxon between samples around its fixed
    median.

    Args:
        correlation_matrix (pd.DataFrame): Correlation matrix with the same
            OTU names in its index and columns.
        n_sample (int): Number of samples to generate.
        mean_abundance (float): Median abundance around which taxon medians
            are generated.
        var_abundance (float): Variance on the latent normal scale.
        random_state (int, optional): Seed used for reproducible random
            generation. If None, fresh entropy is used. Default None.
        taxa_var_abundance (float): Variance on the log scale used to
            generate stable differences in median abundance between taxa.
            Default 0.5.

    Raises:
        `ValueError`: Raises error if correlation_matrix is not a symmetric
                square DataFrame with each element on the diagonal equal to
                one.
        `ValueError`: Raises error if n_sample is not a positive integer.
        `ValueError`: Raises error if mean_abundance is not positive.
        `ValueError`: Raises error if var_abundance is negative.
        `ValueError`: Raises error if taxa_var_abundance is negative.
        `ValueError`: Raises error if random_state is not an integer or
                None.

    Returns:
        pd.DataFrame: Float abundance matrix with samples in rows and OTUs
            in columns.
    """
    # correlation_matrix check
    if not isinstance(correlation_matrix, pd.DataFrame):
        raise ValueError("correlation_matrix must be a pandas DataFrame.")
    if (
        correlation_matrix.empty
        or correlation_matrix.shape[0] != correlation_matrix.shape[1]
    ):
        raise ValueError("correlation_matrix must be a square matrix.")
    if not np.allclose(
        correlation_matrix.to_numpy(), correlation_matrix.to_numpy().T
    ):
        raise ValueError("correlation_matrix must be symmetric.")
    if not np.allclose(np.diag(correlation_matrix.to_numpy()), 1):
        raise ValueError("correlation_matrix must have a unit diagonal.")
    eigenvalues = np.linalg.eigvalsh(correlation_matrix.to_numpy())
    if not np.all(eigenvalues >= -1e-12):
        raise ValueError("correlation_matrix must be positive semidefinite.")

    # n_sample check
    if not isinstance(n_sample, int) or n_sample <= 0:
        raise ValueError("n_sample must be a positive integer.")

    # mean_abundance check
    if not isinstance(mean_abundance, (int, float)) or mean_abundance <= 0:
        raise ValueError("mean_abundance must be positive.")

    # var_abundance check
    if not isinstance(var_abundance, (int, float)) or var_abundance < 0:
        raise ValueError("var_abundance must be non-negative.")

    # taxa_var_abundance check
    if (
        not isinstance(taxa_var_abundance, (int, float))
        or taxa_var_abundance < 0
    ):
        raise ValueError("taxa_var_abundance must be non-negative.")

    # random_state check
    if random_state is not None and not isinstance(random_state, int):
        raise ValueError("random_state must be an integer or None.")

    rng = np.random.default_rng(random_state)

    n_taxa = correlation_matrix.shape[0]
    correlation_values = correlation_matrix.to_numpy()

    # Using the correlation coefficient we find
    # the covariance between taxa
    covariance_matrix = correlation_values * var_abundance

    # Create baseline abundances
    if taxa_var_abundance == 0:
        taxa_medians = np.full(n_taxa, mean_abundance)
    else:
        taxa_medians = rng.lognormal(
            mean=np.log(mean_abundance),
            sigma=np.sqrt(taxa_var_abundance),
            size=n_taxa,
        )
    baseline_abundances = np.log(taxa_medians)

    # Create taxa abundances
    if var_abundance == 0:
        latent_abundances = np.tile(baseline_abundances, (n_sample, 1))
    else:
        latent_abundances = rng.multivariate_normal(
            baseline_abundances, covariance_matrix, size=n_sample
        )
    sample_names = [f"Sample{i}" for i in range(n_sample)]

    correlation_matrix_df = pd.DataFrame(
        np.exp(latent_abundances),
        index=sample_names,
        columns=correlation_matrix.columns,
    )
    return correlation_matrix_df


def simulate_sequencing(
    abundance_matrix, sequencing_depth=1000, random_state=None
):
    """Simulate sequencing counts from an absolute abundance matrix.

    The absolute abundance matrix is normalized to a relative abundance
    matrix. For each sample, the relative abundances are used as
    probabilities in a multinomial distribution. The resulting row contains
    exactly 'sequencing_depth' integer counts distributed among the OTUs.

    Args:
        abundance_matrix (pd.DataFrame): Abundances with samples in rows and
            taxa in columns. Every row must sum to one.
        sequencing_depth (int): Total number of sequencing counts in each
            sample.
        random_state (int, optional): Seed used for reproducible random
            generation. If None, fresh entropy is used. Default None.

    Raises:
        `ValueError`: Raises error if the input is not a non-empty
                DataFrame.
        `ValueError`: Raises error if relative abundances are invalid or do
                not sum to one.
        `ValueError`: Raises error if sequencing_depth is not a positive
                integer.
        `ValueError`: Raises error if random_state is not an integer or
                None.

    Returns:
        pd.DataFrame: Integer sequencing counts with samples in rows and
            OTUs in columns.
    """
    # abundance_matrix check
    if not isinstance(abundance_matrix, pd.DataFrame):
        raise ValueError("abundance_matrix must be a pandas DataFrame.")
    if abundance_matrix.empty:
        raise ValueError("abundance_matrix must be a non-empty DataFrame.")
    if np.any(abundance_matrix < 0):
        raise ValueError("abundance_matrix cannot contain negative values.")

    # sequencing_depth check
    if (
        not isinstance(sequencing_depth, (int, float))
        or sequencing_depth <= 0
        or not float(sequencing_depth).is_integer()
    ):
        raise ValueError("sequencing_depth must be a positive integer.")

    # random_state check
    if random_state is not None and not isinstance(random_state, int):
        raise ValueError("random_state must be an integer or None.")

    sequencing_depth = int(sequencing_depth)

    relative_abundance_matrix = absolute_to_relative_abundance(
        abundance_matrix
    )

    relative_values = relative_abundance_matrix.to_numpy()

    rng = np.random.default_rng(random_state)
    sequencing_counts = np.array(
        [
            rng.multinomial(sequencing_depth, sample_probabilities)
            for sample_probabilities in relative_values
        ]
    )

    sequencing_df = pd.DataFrame(
        sequencing_counts,
        index=relative_abundance_matrix.index,
        columns=relative_abundance_matrix.columns,
    )
    return sequencing_df
