import numpy as np
import pandas as pd


def group_otu_by_subject(otu_df, meta_df):
    """Split the raw OTU table in one table for each subject.

    Args:
        otu_df (pd.DataFrame): DataFrame containing the abundances of every
            sample.
        meta_df (pd.DataFrame): DataFrame containing metadata on every
            sample. Column "SubjectID" is necessary to divide the sample for
            every subject.

    Returns:
        pd.DataFrame: Array of DataFrames containing OTUs divided by
            subject.
    """
    if otu_df.shape[0] != meta_df.shape[0]:
        raise ValueError(
            "OTUs dataframe and meta dataframe must"
            " have the same number of rows"
        )

    if not otu_df.index.equals(meta_df.index):
        raise ValueError(
            "OTUs dataframe and meta dataframe must"
            " have the same samples ID list"
        )

    if "SubjectID" not in meta_df.columns:
        raise ValueError(
            "SubjectID doesn't exist. Please check the input dataframe"
        )

    otu_by_subject = {}

    subject_ids = meta_df["SubjectID"].unique()

    for subject_id in subject_ids:
        # Find all samples belonging to the current subject
        sample_ids = meta_df.loc[meta_df["SubjectID"] == subject_id].index

        # Select only those samples from the OTU table
        otu_subject = otu_df.loc[sample_ids]

        # Store the result
        otu_by_subject[subject_id] = otu_subject

    return otu_by_subject


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


def absolute_to_clr_abundance(otu_df, substitute_zero=0.65):
    """Transform a dataframe with absolute abundances into centered log-ratio
    abundances.

    If a OTU has sample abundance equal to zero, it substitute it with 0.65
    to make it negligible after log (value used in the Correlation-Biases-
    on-Metagenomics-Data repo). This could be a parameter to study for the
    bias

    Args:
        otu_df (pd.DataFrame): Input Dataframe.
        substitute_zero (float, optional): Value used to substitute the
            zeros. Defaults to 0.65.

    Returns:
        pd.DataFrame: Dataframe with centered log-ratio abundances.
    """
    otu_abs = otu_df.to_numpy(dtype=float, copy=True)

    if np.any(otu_abs < 0):
        raise ValueError("Abundances cannot be negative.")

    if np.any(otu_abs == 0):
        if substitute_zero is None or substitute_zero <= 0:
            raise ValueError("'substitute_zero' must be a positive number")

        otu_abs[otu_abs == 0] = substitute_zero

    log_otu = np.log(otu_abs)
    otu_clr = log_otu - log_otu.mean(axis=1, keepdims=True)

    otu_clr_df = pd.DataFrame(
        otu_clr,
        index=otu_df.index,
        columns=otu_df.columns,
    )

    return otu_clr_df


def select_most_abundant_otus(otu_df, number_of_otus):
    """Keeps only the most abundant OTUs of a Dataframe.

    Args:
        otu_df (pd.DataFrame): Dataframe with abundances.
        number_of_otus (int, string): Number of OTUs to keep. Use "all" to
            keep every OTU.

    Returns:
        pd.DataFrame: Dataframe with only the most abundant OTUs.
    """
    if isinstance(number_of_otus, str):
        if number_of_otus.lower() == "all":
            return otu_df

        # For CLI commands
        try:
            number_of_otus = int(number_of_otus)
        except ValueError:
            raise ValueError(
                "'number_of_otus' must be a positive integer or 'all'"
            ) from None

    if (
        number_of_otus is None
        or number_of_otus <= 0
        or not isinstance(number_of_otus, int)
    ):
        raise ValueError(
            "'number_of_otus' must be a positive integer number or 'all'"
        )

    # Sum all the abundances of every OTU and sort them.
    otu_abundance = otu_df.sum(axis=0).sort_values(ascending=False)

    # Select only a number of the most abundant OTUs.
    selected_otus = otu_abundance.head(number_of_otus).index

    return otu_df.loc[:, selected_otus]


def generate_gaussian_noise(otu_df, mean=0, std=1, seed=None, relative=False):
    """Add Gaussian noise to an OTU dataframe.

    Negative values are clipped to zero. If relative is True, each row is
    normalized so that its total abundance is equal to 1.

    Args:
        otu_df (pd.DataFrame): OTU abundance dataframe.
        mean (float, optional): Mean of the Gaussian noise. Defaults to 0.
        std (float, optional): Standard deviation of the Gaussian noise.
            Defaults to 1.
        seed (int, optional): Seed for reproducible random noise. Defaults
            to None.
        relative (bool, optional): If True, convert noisy abundances to
            relative abundances. Defaults to False.

    Returns:
        pd.DataFrame: OTU dataframe with added Gaussian noise.
    """
    rng = np.random.default_rng(seed)

    noise = rng.normal(mean, std, otu_df.shape)
    otu_noise = otu_df + noise
    otu_noise = otu_noise.clip(lower=0)

    if relative:
        otu_noise = absolute_to_relative_abundance(otu_noise)
    return otu_noise


def clean_otus_df(otu_df):
    """Remove OTUs with constant values.

    Args:
        otu_df (pd.DataFrame): OTU abundance table.

    Returns:
        pd.DataFrame: OTU table without constant columns.
    """

    # Count how many different values each OTU column has
    return otu_df.loc[:, (otu_df != otu_df.iloc[0]).any()]
