import pytest

import pandas as pd
import pandas._testing as pdt

import numpy as np

from metagenomes_simulation.data_processing import (
    group_otu_by_subject,
    absolute_to_relative_abundance,
    absolute_to_clr_abundance,
    select_most_abundant_otus,
    generate_gaussian_noise,
    clean_otus_df,
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
# TEST absolute_to_relative_abundance
# --------------------------------------------------------------------------------------------


def test_absolute_to_relative_abundance_expected():
    """This tests the relative-abundance calculation.

    GIVEN: a dataframe containing absolute abundances
    WHEN: I convert it to relative abundances
    THEN: the expected proportions are returned
    """
    otu_df_raw = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU2": [0, 5, 0],
            "OTU3": [2, 1, 3],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )

    otu_expected = pd.DataFrame(
        {
            "OTU1": [10 / 12, 20 / 26, 0 / 3],
            "OTU2": [0 / 12, 5 / 26, 0 / 3],
            "OTU3": [2 / 12, 1 / 26, 3 / 3],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )
    otu_df_rel = absolute_to_relative_abundance(otu_df_raw)

    pdt.assert_frame_equal(otu_expected, otu_df_rel)


def test_absolute_to_relative_abundance_sum_of_every_row_equal_to_one():
    """This tests the sum of relative abundances.

    GIVEN: samples with positive total abundances
    WHEN: I convert them to relative abundances
    THEN: every row sums to one
    """
    otu_df_raw = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU2": [0, 5, 3],
            "OTU3": [2, 1, 4],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )
    otu_df_rel = absolute_to_relative_abundance(otu_df_raw)
    sample_sum_rel_abundance = otu_df_rel.sum(axis=1)
    assert np.allclose(sample_sum_rel_abundance, [1, 1, 1])


def test_absolute_to_relative_abundance_empty_sample():
    """This tests a sample containing only zeros.

    GIVEN: a sample with zero total abundance
    WHEN: I convert it to relative abundances
    THEN: the sample remains a row of zeros
    """
    otu_df_raw = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU2": [0, 5, 0],
            "OTU3": [2, 1, 0],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )
    otu_df_rel = absolute_to_relative_abundance(otu_df_raw)
    sample_sum_rel_abundance = otu_df_rel.sum(axis=1)
    assert np.allclose(sample_sum_rel_abundance, [1, 1, 0])


def test_absolute_to_relative_abundance_keep_index_and_column_preserved():
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
    otu_df_rel = absolute_to_relative_abundance(otu_df_raw)

    pdt.assert_index_equal(otu_df_rel.index, otu_df_raw.index)
    pdt.assert_index_equal(otu_df_rel.columns, otu_df_raw.columns)


# --------------------------------------------------------------------------------------------
# TEST absolute_to_clr_abundance
# --------------------------------------------------------------------------------------------


def test_absolute_to_clr_abundance_expected():
    """This tests the centered log-ratio calculation.

    GIVEN: a dataframe containing positive abundances
    WHEN: I apply the CLR transformation
    THEN: the expected centered logarithms are returned
    """
    otu_df_abs = pd.DataFrame(
        {
            "OTU1": [10, 20, 1],
            "OTU2": [1, 5, 1],
            "OTU3": [2, 1, 3],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )

    otu_expected = pd.DataFrame(
        {
            "OTU1": [np.log(10), np.log(20), np.log(1)],
            "OTU2": [np.log(1), np.log(5), np.log(1)],
            "OTU3": [np.log(2), np.log(1), np.log(3)],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )
    otu_expected = otu_expected.sub(otu_expected.mean(axis=1), axis=0)
    otu_df_clr = absolute_to_clr_abundance(otu_df_abs)

    pdt.assert_frame_equal(otu_expected, otu_df_clr)


def test_absolute_to_clr_abundance_zero_behaviour():
    """This tests zero replacement during CLR transformation.

    GIVEN: an abundance dataframe containing a zero
    WHEN: I apply the CLR transformation
    THEN: the zero is replaced before taking the logarithm
    """
    otu_df_abs = pd.DataFrame(
        {
            "OTU1": [10, 20, 1],
            "OTU2": [0, 5, 1],
            "OTU3": [2, 1, 3],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )

    otu_expected = pd.DataFrame(
        {
            "OTU1": [np.log(10), np.log(20), np.log(1)],
            "OTU2": [np.log(0.65), np.log(5), np.log(1)],
            "OTU3": [np.log(2), np.log(1), np.log(3)],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )
    otu_expected = otu_expected.sub(otu_expected.mean(axis=1), axis=0)
    otu_df_clr = absolute_to_clr_abundance(otu_df_abs)

    pdt.assert_frame_equal(otu_expected, otu_df_clr)


def test_absolute_to_clr_abundance_keep_index_and_column_preserved():
    """This tests preservation of labels after CLR transformation.

    GIVEN: an abundance dataframe with row and column labels
    WHEN: I apply the CLR transformation
    THEN: its labels are preserved
    """
    otu_df_abs = pd.DataFrame(
        {
            "OTU1": [10, 20, 1],
            "OTU2": [1, 5, 1],
            "OTU3": [2, 1, 3],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )

    otu_df_clr = absolute_to_clr_abundance(otu_df_abs)

    pdt.assert_index_equal(otu_df_abs.index, otu_df_clr.index)
    pdt.assert_index_equal(otu_df_abs.columns, otu_df_clr.columns)


def test_absolute_to_clr_abundance_rejects_negative_substitute():
    """This tests a negative zero-replacement value.

    GIVEN: an abundance dataframe containing zeros
    WHEN: I apply CLR with a negative substitute value
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
        absolute_to_clr_abundance(otu_df, substitute_zero=-1)


def test_absolute_to_clr_abundance_rejects_negative_abundances():
    """This tests negative abundance values.

    GIVEN: an abundance dataframe containing a negative value
    WHEN: I apply the CLR transformation
    THEN: the function raises an error
    """
    otu_df = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU2": [0, 5, 3],
            "OTU3": [2, -1, 4],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )

    with pytest.raises(ValueError):
        absolute_to_clr_abundance(otu_df)


def test_absolute_to_clr_abundance_rejects_none_substitute():
    """This tests a missing zero-replacement value.

    GIVEN: an abundance dataframe containing zeros
    WHEN: I apply CLR with None as the substitute value
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
        absolute_to_clr_abundance(otu_df, None)


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


# --------------------------------------------------------------------------------------------
# TEST generate_gaussian_noise
# --------------------------------------------------------------------------------------------


def test_generate_gaussian_noise_output_has_same_shape():
    """This tests the shape of the noisy dataframe.

    GIVEN: an OTU dataframe
    WHEN: I add Gaussian noise
    THEN: the output has the same shape as the input
    """
    otu_df = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU2": [0, 5, 0],
            "OTU3": [2, 1, 3],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )
    otu_noise = generate_gaussian_noise(otu_df, seed=12)

    assert otu_noise.shape == otu_df.shape


def test_generate_gaussian_noise_is_reproducible_with_same_seed():
    """This tests reproducible Gaussian noise.

    GIVEN: an OTU dataframe and a fixed seed
    WHEN: I add Gaussian noise twice
    THEN: both results are equal
    """
    otu_df = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU2": [0, 5, 0],
            "OTU3": [2, 1, 3],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )

    otu_noise_1 = generate_gaussian_noise(otu_df, seed=12)
    otu_noise_2 = generate_gaussian_noise(otu_df, seed=12)

    pd.testing.assert_frame_equal(otu_noise_1, otu_noise_2)


def test_generate_gaussian_noise_all_positive_values():
    """This tests clipping of negative noisy values.

    GIVEN: an OTU dataframe
    WHEN: I add Gaussian noise
    THEN: every resulting abundance is non-negative
    """
    otu_df = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU2": [0, 5, 0],
            "OTU3": [2, 1, 3],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )
    otu_noise = generate_gaussian_noise(otu_df, seed=12)

    assert (otu_noise >= 0).all().all()


def test_generate_gaussian_noise_relative_true_normalizes_rows():
    """This tests relative normalization after adding noise.

    GIVEN: an OTU dataframe and relative mode enabled
    WHEN: I add Gaussian noise
    THEN: every resulting row sums to one
    """
    otu_df = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU2": [0, 5, 0],
            "OTU3": [2, 1, 3],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )

    otu_noise = generate_gaussian_noise(
        otu_df, mean=1, std=0, seed=42, relative=True
    )

    sample_sum_rel_abundance = otu_noise.sum(axis=1)

    assert np.allclose(sample_sum_rel_abundance, 1)


def test_generate_gaussian_noise_keep_index_and_column_preserved():
    """This tests preservation of labels after adding noise.

    GIVEN: an OTU dataframe with row and column labels
    WHEN: I add Gaussian noise
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
    otu_noise = generate_gaussian_noise(otu_df_raw)

    pdt.assert_index_equal(otu_noise.index, otu_df_raw.index)
    pdt.assert_index_equal(otu_noise.columns, otu_df_raw.columns)


# --------------------------------------------------------------------------------------------
# TEST clean_otus_df
# --------------------------------------------------------------------------------------------


def test_clean_otus_df_remove_empty_columns():
    """This tests removal of an all-zero OTU.

    GIVEN: an OTU dataframe with an all-zero column
    WHEN: I clean the dataframe
    THEN: the all-zero column is removed
    """
    otu_df_raw = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU2": [0, 0, 0],
            "OTU3": [2, 1, 3],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )
    otu_expected = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU3": [2, 1, 3],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )
    otu_df_new = clean_otus_df(otu_df_raw)

    pdt.assert_frame_equal(otu_expected, otu_df_new)


def test_clean_otus_df_remove_constant_columns():
    """This tests removal of a constant OTU.

    GIVEN: an OTU dataframe with a constant column
    WHEN: I clean the dataframe
    THEN: the constant column is removed
    """
    otu_df_raw = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU2": [4, 4, 4],
            "OTU3": [2, 1, 3],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )
    otu_expected = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU3": [2, 1, 3],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )
    otu_df_new = clean_otus_df(otu_df_raw)

    pdt.assert_frame_equal(otu_expected, otu_df_new)


def test_clean_otus_df_all_constant_columns():
    """This tests a dataframe containing only constant OTUs.

    GIVEN: an OTU dataframe with only constant columns
    WHEN: I clean the dataframe
    THEN: the result contains no columns
    """
    otu_df_raw = pd.DataFrame(
        {
            "OTU1": [4, 4, 4],
            "OTU2": [4, 4, 4],
            "OTU3": [4, 4, 4],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )
    otu_expected = otu_df_raw.iloc[:, 0:0]
    otu_df_new = clean_otus_df(otu_df_raw)

    pdt.assert_frame_equal(otu_expected, otu_df_new)


def test_clean_otus_df_empty_df():
    """This tests an empty OTU dataframe.

    GIVEN: an OTU dataframe without columns
    WHEN: I clean the dataframe
    THEN: the dataframe remains empty
    """
    otu_df_raw = pd.DataFrame(
        {},
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )
    otu_expected = pd.DataFrame(
        {},
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )
    otu_df_new = clean_otus_df(otu_df_raw)

    pdt.assert_frame_equal(otu_expected, otu_df_new)
