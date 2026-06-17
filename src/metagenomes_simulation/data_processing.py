import numpy as np


def group_otu_by_subject(otu_df, meta_df):
    """Split the raw OTU table in one table for each subject

    Args:
        otu_df (DataFrame): DataFrame containing the abundances of every sample.
        meta_df (DataFrame): DataFrame containing metadata on every sample.
                            Column "SubjectID" is necessary to divide the sample for every
                            subject.

    Returns:
        DataFrame: Array of DataFrames containing OTUs divided by subject.
    """
    if otu_df.shape[0] != meta_df.shape[0]:
        raise ValueError("OTUs dataframe and meta dataframe must have the same number of rows")

    if not otu_df.index.equals(meta_df.index):
        raise ValueError("OTUs dataframe and meta dataframe must have the same samples ID list")

    if "SubjectID" not in meta_df.columns:
        raise ValueError("SubjectID doesn't exist. Please check the input dataframe")

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
        otu_df (DataFrame): Dataframe with absolute abundances.

    Returns:
        DataFrame: Dataframe with relative abundances.
    """

    # Sum all the row abundances
    sample_total_abundance = otu_df.sum(axis=1)

    # Replace zeros with 1 to avoid division by zero
    sample_total_abundance = sample_total_abundance.replace(0, 1)

    # Divide sample abundances by the sum of the sample abundances
    otu_rel = otu_df.div(sample_total_abundance, axis=0)
    return otu_rel


def generate_gaussian_noise(otu_df, mean=0, std=1, seed=None, relative=False):
    """Add Gaussian noise to an OTU dataframe.

    Negative values are clipped to zero. If relative is True, each row is
    normalized so that its total abundance is equal to 1.

    Args:
        otu_df (pd.DataFrame): OTU abundance dataframe.
        mean (float, optional): Mean of the Gaussian noise. Defaults to 0.
        std (float, optional): Standard deviation of the Gaussian noise. Defaults to 1.
        seed (int, optional): Seed for reproducible random noise. Defaults to None.
        relative (bool, optional): If True, convert noisy abundances to relative abundances.
            Defaults to False.

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
        otu_df (DataFrame): OTU abundance table.

    Returns:
        DataFrame: OTU table without constant columns.
    """

    # Count how many different values each OTU column has
    return otu_df.loc[:, (otu_df != otu_df.iloc[0]).any()]
