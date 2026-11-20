import pandas as pd
import numpy as np


def substitute_zero(otu_df, delta=None):
    """Replace zero abundances using multiplicative replacement.

    Constant OTUs are first removed with :func:`remove_constant_otus`. Each
    sample is then converted to proportions. For a sample containing 'z'
    zeros, every zero is replaced by 'delta' and each positive proportion is
    multiplied by '1 - z * delta'. This preserves the ratios between the
    positive components and keeps each row summing to one.

    Args:
        otu_df (pd.DataFrame): Samples-by-OTUs dataframe containing
            non-negative abundances.
        delta (float, optional): Proportion assigned to every zero. It must
            satisfy 'z * delta < 1' for every sample. If 'None', it is set
            to '1 / D**2', where 'D' is the number of retained OTUs.
            Defaults to 'None'.

    Returns:
        pd.DataFrame: Sample proportions after multiplicative zero
            replacement.

    Raises:
        `ValueError`: If any abundance is negative or a sample has total
                abundance equal to zero after constant OTUs are removed.
    """

    otu_cleaned = remove_constant_otus(otu_df)
    values = otu_cleaned.to_numpy(dtype=float, copy=True)
    if np.any(values < 0):
        raise ValueError("Abundances cannot be negative.")

    sample_total_abundance = values.sum(axis=1, keepdims=True)
    if np.any(sample_total_abundance == 0):
        raise ValueError("Samples cannot have total abundance zero.")

    proportions = values / sample_total_abundance
    n_zeros = np.count_nonzero(proportions == 0, axis=1, keepdims=True)

    n_components = proportions.shape[1]
    if delta is None:
        delta = 1 / n_components**2
    if delta < 0:
        raise ValueError("'delta' cannot be negative.")
    scaling_factor = 1 - n_zeros * delta
    otu_replaced = proportions * scaling_factor
    otu_replaced[otu_replaced == 0] = delta
    otu_replaced_df = pd.DataFrame(
        otu_replaced,
        index=otu_cleaned.index,
        columns=otu_cleaned.columns,
    )
    return otu_replaced_df


def remove_constant_otus(otu_df):
    """Remove OTUs with constant values.

    Args:
        otu_df (pd.DataFrame): OTU abundance table.

    Returns:
        pd.DataFrame: OTU table without constant columns.
    """

    # Count how many different values each OTU column has
    return otu_df.loc[:, (otu_df != otu_df.iloc[0]).any()]


def absolute_to_relative_abundance(otu_df):
    """Transform a dataframe with absolute abundances into relative abundances.

    If a sample has total abundance equal to zero, its total abundance is
    replaced with 1 before division. In this way, zero-sum samples are kept
    as rows of zeros.

    Args:
        otu_df (pd.DataFrame): Dataframe with absolute abundances.

    Returns:
        pd.DataFrame: Dataframe with relative abundances.
    """
    # Sum all the row abundances
    sample_total_abundance = otu_df.sum(axis=1)

    # Replace zeros with 1 to avoid division by zero
    sample_total_abundance = sample_total_abundance.replace(0, 1)

    # Divide sample abundances by the sum of the sample abundances
    otu_rel = otu_df.div(sample_total_abundance, axis=0)
    return otu_rel


def absolute_to_clr_abundance(otu_df, delta=None):
    """Transform a dataframe with absolute abundances into centered log-ratio
    abundances.

    If a OTU has sample abundance equal to zero, it substitute it with 0.65
    to make it negligible after log (value used in the Correlation-Biases-
    on-Metagenomics-Data repo). This could be a parameter to study for the
    bias

    Args:
        otu_df (pd.DataFrame): Input Dataframe.
        delta (float, optional): Proportion assigned to every zero. It must
            satisfy 'z * delta < 1' for every sample. If 'None', it is set
            to '1 / D**2', where 'D' is the number of retained OTUs.
            Defaults to 'None'

    Returns:
        pd.DataFrame: Dataframe with centered log-ratio abundances.
    """
    otu_no_zeros = substitute_zero(otu_df, delta=delta)
    log_otu = np.log(otu_no_zeros)
    otu_clr = log_otu.sub(log_otu.mean(axis=1), axis=0)

    return otu_clr


def normalize_abundance_all_methods(otu_df):
    """Normalize absolute OTU abundances using relative abundance and CLR.

    Args:
        otu_df (pd.DataFrame): Absolute OTU abundance matrix.

    Returns:
        dict[str, pd.DataFrame]: Relative and CLR-transformed abundances.
    """
    return {
        "relative": absolute_to_relative_abundance(otu_df),
        "clr": absolute_to_clr_abundance(otu_df),
    }
