import pytest

import pandas as pd
import pandas.testing as pdt

import numpy as np

from metagenomes_simulation.normalization import (
    absolute_to_relative_abundance,
    absolute_to_clr_abundance,
    remove_constant_otus,
    normalize_and_substitute_zeros,
    normalize_abundance_all_methods,
)

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
# TEST normalize_and_substitute_zeros
# --------------------------------------------------------------------------------------------


def test_normalize_and_substitute_zeros_default_delta_expected():
    """This tests zero replacement with the default delta.

    GIVEN: an abundance dataframe containing different numbers of zeros
        in each sample
    WHEN: I replace zeros using the default delta
    THEN: the expected positive proportions are returned and every
        sample sums to one
    """
    otu_df = pd.DataFrame(
        [[2, 2, 0, 0], [0, 1, 3, 4]],
        index=["Sample1", "Sample2"],
        columns=["OTU1", "OTU2", "OTU3", "OTU4"],
    )
    expected = np.array(
        [
            [0.4375, 0.4375, 0.0625, 0.0625],
            [0.0625, 0.1171875, 0.3515625, 0.46875],
        ]
    )

    result = normalize_and_substitute_zeros(otu_df)

    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(result.sum(axis=1), np.ones(2))
    assert np.all(result > 0)


def test_normalize_and_substitute_zeros_custom_delta_expected():
    """This tests zero replacement with a custom delta.

    GIVEN: an abundance dataframe containing zeros and a custom delta
    WHEN: I apply normalization zero replacement
    THEN: the expected positive proportions are returned and every
        sample sums to one
    """
    otu_df = pd.DataFrame(
        [[10, 5, 0], [0, 3, 9]],
        columns=["OTU1", "OTU2", "OTU3"],
    )
    result = normalize_and_substitute_zeros(otu_df, delta=0.01)
    
    expected = np.array([[0.66, 0.33, 0.01], [0.01, 0.2475, 0.7425]])

    np.testing.assert_allclose(result, expected)


def test_normalize_and_substitute_zeros_without_zeros_returns_proportions():
    """This tests normalization without zeros.

    GIVEN: an abundance dataframe containing only positive values
    WHEN: I apply normalization zero replacement
    THEN: the expected positive proportions are returned and every
        sample sums to one
    """
    otu_df = pd.DataFrame(
        [[2, 3, 5], [1, 4, 2]],
        columns=["OTU1", "OTU2", "OTU3"],
    )
    expected = np.array([[0.2, 0.3, 0.5], [1 / 7, 4 / 7, 2 / 7]])

    result = normalize_and_substitute_zeros(otu_df)

    np.testing.assert_allclose(result, expected)


def test_normalize_and_substitute_zeros_removes_constant_otus():
    """This tests the removal of constant OTUs before zero replacement.

    GIVEN: an abundance dataframe containing a constant OTU
    WHEN: I apply normalization zero replacement
    THEN: the constant OTU is removed and the expected proportions
        are computed using only the remaining OTUs
    """
    otu_df = pd.DataFrame(
        [[5, 2, 0], [5, 3, 4]],
        columns=["constant", "OTU1", "OTU2"],
    )
    expected = np.array([[0.75, 0.25], [3 / 7, 4 / 7]])

    result = normalize_and_substitute_zeros(otu_df)

    np.testing.assert_allclose(result, expected)
    assert result.shape == (2, 2)


def test_normalize_and_substitute_zeros_does_not_modify_input():
    """This tests that zero replacement preserves the original dataframe.

    GIVEN: an abundance dataframe containing zeros
    WHEN: I apply normalization zero replacement
    THEN: the original dataframe remains unchanged
    """
    otu_df = pd.DataFrame(
        [[2, 0, 1], [1, 3, 0]],
        columns=["OTU1", "OTU2", "OTU3"],
    )
    original = otu_df.copy(deep=True)

    normalize_and_substitute_zeros(otu_df)

    pdt.assert_frame_equal(otu_df, original)


def test_normalize_and_substitute_zeros_rejects_negative_abundances():
    """This tests that negative abundances raise an error.

    GIVEN: an abundance dataframe containing a negative value
    WHEN: I apply normalization zero replacement
    THEN: the function raises a ValueError
    """
    otu_df = pd.DataFrame(
        [[1, -1], [2, 3]],
        columns=["OTU1", "OTU2"],
    )

    with pytest.raises(ValueError):
        normalize_and_substitute_zeros(otu_df)


def test_normalize_and_substitute_zeros_rejects_zero_total_sample():
    """This tests that samples with zero total abundance raise an error.

    GIVEN: an abundance dataframe containing a sample with only zeros
    WHEN: I apply normalization zero replacement
    THEN: the function raises a ValueError
    """
    otu_df = pd.DataFrame(
        [[0, 0], [1, 2]],
        columns=["OTU1", "OTU2"],
    )

    with pytest.raises(
        ValueError
    ):
        normalize_and_substitute_zeros(otu_df)

@pytest.mark.parametrize("delta", [0.5, 0.6])
def test_normalize_and_substitute_zeros_rejects_large_delta(delta):
    """This tests that excessively large delta values raise an error.

    GIVEN: an abundance dataframe containing two zeros in a sample
    WHEN: I replace zeros with a delta such that n_zeros * delta >= 1
    THEN: the function raises an error
    """
    otu_df = pd.DataFrame(
        [[10, 0, 0], [5, 3, 4]],
        columns=["OTU1", "OTU2", "OTU3"],
    )

    with pytest.raises(ValueError):
        normalize_and_substitute_zeros(otu_df, delta=delta)
        
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
    """This tests multiplicative zero replacement during CLR transformation.

    GIVEN: an abundance dataframe containing a zero
    WHEN: I apply the CLR transformation
    THEN: the zero is multiplicatively replaced before taking the logarithm
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
            "OTU1": [20 / 27, 20 / 26, 1 / 5],
            "OTU2": [1 / 9, 5 / 26, 1 / 5],
            "OTU3": [4 / 27, 1 / 26, 3 / 5],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )
    otu_expected = np.log(otu_expected)
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


def test_absolute_to_clr_abundance_rejects_negative_delta():
    """This tests a negative multiplicative-replacement value.

    GIVEN: an abundance dataframe containing zeros
    WHEN: I apply CLR with a negative delta value
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
        absolute_to_clr_abundance(otu_df, delta=-1)


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


def test_absolute_to_clr_abundance_uses_default_delta_when_none():
    """This tests the default multiplicative-replacement value.

    GIVEN: an abundance dataframe containing zeros
    WHEN: I apply CLR with None as delta
    THEN: the default value 1 / D**2 is used
    """
    otu_df = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU2": [0, 5, 3],
            "OTU3": [2, 1, 4],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )

    n_components = otu_df.shape[1]
    expected = absolute_to_clr_abundance(
        otu_df,
        delta=1 / n_components**2,
    )

    result = absolute_to_clr_abundance(otu_df, delta=None)

    pdt.assert_frame_equal(result, expected)


# --------------------------------------------------------------------------------------------
# TEST remove_constant_otus
# --------------------------------------------------------------------------------------------


def test_remove_constant_otus_remove_empty_columns():
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
    otu_df_new = remove_constant_otus(otu_df_raw)

    pdt.assert_frame_equal(otu_expected, otu_df_new)


def test_remove_constant_otus_remove_constant_columns():
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
    otu_df_new = remove_constant_otus(otu_df_raw)

    pdt.assert_frame_equal(otu_expected, otu_df_new)


def test_remove_constant_otus_all_constant_columns():
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
    otu_df_new = remove_constant_otus(otu_df_raw)

    pdt.assert_frame_equal(otu_expected, otu_df_new)


def test_remove_constant_otus_empty_df():
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
    otu_df_new = remove_constant_otus(otu_df_raw)

    pdt.assert_frame_equal(otu_expected, otu_df_new)


# --------------------------------------------------------------------------------------------
# TEST normalize_abundance_all_methods
# --------------------------------------------------------------------------------------------


def test_normalize_abundance_all_methods_returns_all_methods():
    """This tests that both normalization methods are returned.

    GIVEN: an OTU abundance dataframe
    WHEN: I apply all normalization methods
    THEN: the result contains relative and CLR abundances
    """

    otu_df = pd.DataFrame(
        {
            "OTU1": [10, 20, 30],
            "OTU2": [20, 30, 40],
            "OTU3": [30, 40, 50],
        }
    )

    results = normalize_abundance_all_methods(otu_df)

    assert set(results.keys()) == {"relative", "clr"}


def test_normalize_abundance_all_methods_correct_relative():
    """This tests that relative abundances are correctly returned.

    GIVEN: an OTU abundance dataframe
    WHEN: I apply all normalization methods
    THEN: relative abundances match the individual normalization
    """

    otu_df = pd.DataFrame(
        {
            "OTU1": [10, 20, 30],
            "OTU2": [20, 30, 40],
            "OTU3": [30, 40, 50],
        }
    )

    results = normalize_abundance_all_methods(otu_df)
    expected = absolute_to_relative_abundance(otu_df)

    pdt.assert_frame_equal(results["relative"], expected)


def test_normalize_abundance_all_methods_correct_clr():
    """This tests that CLR abundances are correctly returned.

    GIVEN: an OTU abundance dataframe
    WHEN: I apply all normalization methods
    THEN: CLR abundances match the individual normalization
    """

    otu_df = pd.DataFrame(
        {
            "OTU1": [10, 20, 30],
            "OTU2": [20, 30, 40],
            "OTU3": [30, 40, 50],
        }
    )

    results = normalize_abundance_all_methods(otu_df)
    expected = absolute_to_clr_abundance(otu_df)

    pdt.assert_frame_equal(results["clr"], expected)
