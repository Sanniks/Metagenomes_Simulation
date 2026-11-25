import pytest
import pandas as pd
import pandas.testing as pdt

from metagenomes_simulation.raw_processing import (
    group_otu_by_subject,
    select_most_abundant_otus,
)

# --------------------------------------------------------------------------------------------
# TEST group_otu_by_subject
# --------------------------------------------------------------------------------------------


def test_group_otu_by_subject_correct_groups():
    """This tests that the separation by subject function works correctly.

    GIVEN: inputs meta_df and otu_df recreated
    with the same structure as the raw data files
    WHEN: I apply to it the separation function
    THEN: the function separates the otu_df dataframe
    in two dataframes with the same structure
    """
    # Input dataframes
    meta_df = pd.DataFrame(
        {"SubjectID": ["69-001", "69-001", "69-074"]},
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )

    otu_df = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU2": [0, 5, 3],
            "OTU3": [2, 1, 4],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )
    otu_divided = group_otu_by_subject(otu_df, meta_df)

    # Expected output dataframes
    otu_divided_0_expected = pd.DataFrame(
        {
            "OTU1": [10, 20],
            "OTU2": [0, 5],
            "OTU3": [2, 1],
        },
        index=["69-001-1010", "69-001-1011"],
    )

    otu_divided_1_expected = pd.DataFrame(
        {
            "OTU1": [0],
            "OTU2": [3],
            "OTU3": [4],
        },
        index=["69-074-6012"],
    )

    pdt.assert_frame_equal(otu_divided["69-001"], otu_divided_0_expected)
    pdt.assert_frame_equal(otu_divided["69-074"], otu_divided_1_expected)


def test_group_otu_by_subject_single_subject():
    """This tests that the separation by subject function return the same
    dataframe if it detects a single subject.

    GIVEN: inputs meta_df and otu_df with single subject,
    recreated with the same structure as the raw data files
    WHEN: I apply to it the separation function
    THEN: the function returns the same otu_df dataframe as the input
    """
    meta_df = pd.DataFrame(
        {"SubjectID": ["69-001", "69-001", "69-001"]},
        index=["69-001-1010", "69-001-1011", "69-001-1012"],
    )

    otu_df = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU2": [0, 5, 3],
            "OTU3": [2, 1, 4],
        },
        index=["69-001-1010", "69-001-1011", "69-001-1012"],
    )
    otu_divided = group_otu_by_subject(otu_df, meta_df)

    pdt.assert_frame_equal(otu_divided["69-001"], otu_df)


def test_group_otu_by_subject_error_same_indexings():
    """This tests that the separation by subject function raises an error if
    the indeces of the two input matrices don't correspond.

    GIVEN: inputs meta_df and otu_df with different indices,
    recreated with the same structure as the raw data files
    WHEN: I apply to it the separation function
    THEN: the function raises an error
    """
    meta_df = pd.DataFrame(
        {"SubjectID": ["69-001", "69-001", "69-074"]},
        index=["69-001-0", "69-001-1", "69-074-2"],
    )

    otu_df = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU2": [0, 5, 3],
            "OTU3": [2, 1, 4],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )
    with pytest.raises(ValueError):
        group_otu_by_subject(otu_df, meta_df)


def test_group_otu_by_subject_error_missing_SubjectID_in_meta():
    """This tests that the separation by subject function raises an error if
    the label for the subjects in meta_df is different from "SubjectID".

    GIVEN: inputs meta_df without the label "SubjectID and otu_df,
    recreated with the same structure as the raw data files
    WHEN: I apply to it the separation function
    THEN: the function raises an error
    """
    meta_df = pd.DataFrame(
        {"WrongLabel_ID": ["69-001", "69-001", "69-074"]},
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )

    otu_df = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU2": [0, 5, 3],
            "OTU3": [2, 1, 4],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )

    with pytest.raises(ValueError):
        group_otu_by_subject(otu_df, meta_df)


def test_group_otu_by_subject_error_different_dimensions():
    """This tests dataframes with different numbers of rows.

    GIVEN: OTU and metadata dataframes with different row counts
    WHEN: I group the OTUs by subject
    THEN: the function raises an error
    """
    meta_df = pd.DataFrame(
        {"SubjectID": ["69-001", "69-001", "69-074", "69-074"]},
        index=["69-001-1010", "69-001-1011", "69-074-6012", "69-074-6013"],
    )

    otu_df = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU2": [0, 5, 3],
            "OTU3": [2, 1, 4],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )
    with pytest.raises(ValueError):
        group_otu_by_subject(otu_df, meta_df)


# --------------------------------------------------------------------------------------------
# TEST select_most_abundant_otus
# --------------------------------------------------------------------------------------------


def test_select_most_abundant_otus_keeps_all_otus():
    """This tests selection when every OTU is requested.

    GIVEN: a dataframe with three OTUs
    WHEN: I request three OTUs
    THEN: every OTU is returned
    """
    otu_df = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU2": [0, 5, 3],
            "OTU3": [2, 1, 4],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )

    otu_selected = select_most_abundant_otus(otu_df, 3)

    pdt.assert_frame_equal(otu_selected, otu_df)


def test_select_most_abundant_otus_when_more_are_requested():
    """This tests selection when too many OTUs are requested.

    GIVEN: a dataframe with three OTUs
    WHEN: I request four OTUs
    THEN: every OTU is returned
    """
    otu_df = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU2": [0, 5, 3],
            "OTU3": [2, 1, 4],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )

    otu_selected = select_most_abundant_otus(otu_df, 4)

    pdt.assert_frame_equal(otu_selected, otu_df)


def test_select_most_abundant_otus_all_is_case_insensitive():
    """This tests the special value "all".

    GIVEN: a dataframe with three OTUs
    WHEN: I request "ALL"
    THEN: every OTU is returned
    """
    otu_df = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU2": [0, 5, 3],
            "OTU3": [2, 1, 4],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )

    otu_selected = select_most_abundant_otus(otu_df, "ALL")

    pdt.assert_frame_equal(otu_selected, otu_df)


def test_select_most_abundant_otus_accepts_numeric_string():
    """This tests a numeric string as the number of OTUs.

    GIVEN: a dataframe with three OTUs
    WHEN: I request the two most abundant OTUs using "2"
    THEN: the two OTUs with the largest sums are returned
    """
    otu_df = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU2": [0, 5, 3],
            "OTU3": [2, 1, 4],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )
    otu_expected = otu_df.loc[:, ["OTU1", "OTU2"]]

    otu_selected = select_most_abundant_otus(otu_df, "2")

    pdt.assert_frame_equal(otu_selected, otu_expected)


def test_select_most_abundant_otus_rejects_non_numeric_string():
    """This tests an invalid string as the number of OTUs.

    GIVEN: a dataframe with three OTUs
    WHEN: I request OTUs using a non-numeric string
    THEN: the function raises an error
    """
    otu_df = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU2": [0, 5, 3],
            "OTU3": [2, 1, 4],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )

    with pytest.raises(ValueError):
        select_most_abundant_otus(otu_df, "two")


def test_select_most_abundant_otus_rejects_none():
    """This tests a missing number of OTUs.

    GIVEN: a dataframe with three OTUs
    WHEN: I use None as the number of OTUs
    THEN: the function raises an error
    """
    otu_df = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU2": [0, 5, 3],
            "OTU3": [2, 1, 4],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )

    with pytest.raises(ValueError):
        select_most_abundant_otus(otu_df, None)


def test_select_most_abundant_otus_rejects_negative_number():
    """This tests a negative number of OTUs.

    GIVEN: a dataframe with three OTUs
    WHEN: I request a negative number of OTUs
    THEN: the function raises an error
    """
    otu_df = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU2": [0, 5, 3],
            "OTU3": [2, 1, 4],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )

    with pytest.raises(ValueError):
        select_most_abundant_otus(otu_df, -1)


def test_select_most_abundant_otus_rejects_float():
    """This tests a non-integer number of OTUs.

    GIVEN: a dataframe with three OTUs
    WHEN: I request a float number of OTUs
    THEN: the function raises an error
    """
    otu_df = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU2": [0, 5, 3],
            "OTU3": [2, 1, 4],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )

    with pytest.raises(ValueError):
        select_most_abundant_otus(otu_df, 1.5)


def test_select_most_abundant_otus_expected():
    """This tests selection of the most abundant OTUs.

    GIVEN: a dataframe with three OTUs
    WHEN: I request the two most abundant OTUs
    THEN: the two OTUs with the largest sums are returned
    """
    otu_df = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU2": [0, 5, 3],
            "OTU3": [2, 1, 4],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )
    otu_expected = otu_df.loc[:, ["OTU1", "OTU2"]]

    otu_selected = select_most_abundant_otus(otu_df, 2)

    pdt.assert_frame_equal(otu_selected, otu_expected)


def test_select_most_abundant_otus_keep_index_and_column_preserved():
    """This tests preservation of labels.

    GIVEN: an abundance dataframe with row and column labels
    WHEN: I convert it to relative abundances
    THEN: its labels are preserved
    """
    otu_df_raw = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU2": [0, 5, 0],
            "OTU3": [2, 1, 3],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )

    otu_selected = select_most_abundant_otus(otu_df_raw, "all")

    pdt.assert_index_equal(otu_selected.index, otu_df_raw.index)
    pdt.assert_index_equal(otu_selected.columns, otu_df_raw.columns)
