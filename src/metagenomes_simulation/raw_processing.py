import numpy as np
import pandas as pd


def group_otu_by_subject(otu_df, meta_df):
    """Split the raw OTUs table in different tables per subject.

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
    # Check validity of raw data
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
