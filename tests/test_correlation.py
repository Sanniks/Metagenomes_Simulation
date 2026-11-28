from hypothesis import given, assume
from hypothesis import strategies as st
import pytest
import numpy as np
import pandas as pd
import pandas.testing as pdt

from metagenomes_simulation.correlation import (
    create_correlation_matrix,
    create_correlation_matrix_all_methods,
)

# --------------------------------------------------------------------------------------------
# TEST create_correlation_matrix
# --------------------------------------------------------------------------------------------


@given(
    st.floats(min_value=0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0, allow_nan=False, allow_infinity=False),
)
def test_create_correlation_matrix_symmetrical_matrix(x, y, z):
    """This tests that the correlation matrix is symmetric.

    GIVEN: an OTU dataframe containing 3 taxa and 3 samples
    WHEN: I create its correlation matrix
    THEN: the matrix is equal to its transpose
    """
    otu_df = pd.DataFrame(
        {
            "OTU1": [10, x, 0],
            "OTU2": [0, y, 3],
            "OTU3": [2, z, 4],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )

    corr_matrix = create_correlation_matrix(otu_df)
    pdt.assert_frame_equal(corr_matrix, corr_matrix.T)


@given(
    st.floats(min_value=0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0, allow_nan=False, allow_infinity=False),
)
def test_create_correlation_matrix_diagonal_is_one(x, y, z):
    """This tests that the correlation matrix has a unit diagonal.

    GIVEN: an OTU dataframe containing 3 taxa and 3 samples
    WHEN: I create its correlation matrix
    THEN: every value on the main diagonal is equal to one
    """
    otu_df = pd.DataFrame(
        {
            "OTU1": [10, x, 0],
            "OTU2": [0, y, 3],
            "OTU3": [2, z, 4],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )

    corr_matrix = create_correlation_matrix(otu_df)
    diagonal_values = np.diag(corr_matrix)
    assert np.allclose(diagonal_values, 1)


def test_create_correlation_matrix_correct_values_for_known_pairs():
    """This tests that known taxon relationships produce the expected
    correlations.

    GIVEN: two perfectly positively correlated taxa
    and one perfectly negatively correlated taxon
    WHEN: I create their Pearson correlation matrix
    THEN: the corresponding matrix values are one and minus one
    """
    otu_df = pd.DataFrame(
        {
            "OTU1": [1, 2, 3, 4],
            "OTU2": [2, 4, 6, 8],
            "OTU3": [4, 3, 2, 1],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012", "69-074-6013"],
    )

    corr_matrix = create_correlation_matrix(
        otu_df, correlation_method="Pearson"
    )

    expected_matrix = pd.DataFrame(
        {
            "OTU1": [1.0, 1.0, -1.0],
            "OTU2": [1.0, 1.0, -1.0],
            "OTU3": [-1.0, -1.0, 1.0],
        },
        index=["OTU1", "OTU2", "OTU3"],
    )

    pdt.assert_frame_equal(
        corr_matrix, expected_matrix, check_exact=False, atol=1e-12
    )


def test_create_correlation_matrix_remove_constant_columns():
    """This tests that constant taxa are removed from the correlation matrix.

    GIVEN: an OTU dataframe containing three variable and two constant taxa
    WHEN: I create its Pearson correlation matrix
    THEN: the result contains only the three variable
    taxa and their expected correlations
    """
    otu_df = pd.DataFrame(
        {
            "OTU1": [1, 2, 3, 4],
            "OTU2": [2, 4, 6, 8],
            "OTU3": [4, 3, 2, 1],
            "OTU4": [1, 1, 1, 1],
            "OTU5": [0, 0, 0, 0],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012", "69-074-6013"],
    )

    corr_matrix = create_correlation_matrix(
        otu_df, correlation_method="Pearson"
    )

    expected_matrix = pd.DataFrame(
        {
            "OTU1": [1.0, 1.0, -1.0],
            "OTU2": [1.0, 1.0, -1.0],
            "OTU3": [-1.0, -1.0, 1.0],
        },
        index=["OTU1", "OTU2", "OTU3"],
    )

    pdt.assert_frame_equal(
        corr_matrix, expected_matrix, check_exact=False, atol=1e-12
    )


def test_create_correlation_matrix_all_constant_columns():
    """This tests correlation matrix creation when every taxon is constant.

    GIVEN: an OTU dataframe containing only constant taxa
    WHEN: I create its Pearson correlation matrix
    THEN: the result is an empty matrix
    """
    otu_df = pd.DataFrame(
        {
            "OTU4": [1, 1, 1, 1],
            "OTU5": [0, 0, 0, 0],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012", "69-074-6013"],
    )

    corr_matrix = create_correlation_matrix(
        otu_df, correlation_method="Pearson"
    )

    assert corr_matrix.shape == (0, 0)
    assert corr_matrix.empty


def test_create_correlation_matrix_with_one_otu():
    """This tests the parallel function with one variable OTU.

    GIVEN: a dataframe containing one variable OTU
    WHEN: I create its correlation matrix in parallel
    THEN: the result is a one-by-one matrix containing one
    """
    otu_df = pd.DataFrame(
        {
            "OTU1": [1, 2, 3],
        }
    )

    corr_matrix = create_correlation_matrix(otu_df)

    expected_matrix = pd.DataFrame(
        [[1.0]],
        index=["OTU1"],
        columns=["OTU1"],
    )
    pdt.assert_frame_equal(corr_matrix, expected_matrix)


# --------------------------------------------------------------------------------------------
# TEST create_correlation_matrix_all_methods
# --------------------------------------------------------------------------------------------


def test_create_correlation_matrix_all_methods_returns_all_methods():
    """This tests that all correlation methods are returned.

    GIVEN: an OTU dataframe containing three variable taxa
    WHEN: I compute all correlation matrices
    THEN: the result contains Pearson, Spearman, and Kendall
    """

    otu_df = pd.DataFrame(
        {
            "OTU1": [1, 2, 3],
            "OTU2": [2, 4, 6],
            "OTU3": [3, 2, 1],
        }
    )

    results = create_correlation_matrix_all_methods(otu_df)

    assert set(results.keys()) == {"Pearson", "Spearman", "Kendall"}


def test_create_correlation_matrix_all_methods_same_results_as_classic():
    """This tests that all correlation matrices match the sequential ones.

    GIVEN: an OTU dataframe containing three variable taxa
    WHEN: I compute the correlation matrices using both functions
    THEN: the matrices are equal for each correlation method
    """

    otu_df = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU2": [0, 5, 3],
            "OTU3": [2, 1, 4],
        }
    )

    results = create_correlation_matrix_all_methods(otu_df)

    for method in ("Pearson", "Spearman", "Kendall"):
        expected_matrix = create_correlation_matrix(
            otu_df, correlation_method=method
        )

        pdt.assert_frame_equal(results[method], expected_matrix)


def test_create_correlation_matrix_all_methods_remove_constant_columns():
    """This tests that constant OTUs are removed.

    GIVEN: an OTU dataframe containing two variable and one constant OTU
    WHEN: I compute all correlation matrices
    THEN: every matrix contains only the variable OTUs
    """

    otu_df = pd.DataFrame(
        {
            "OTU1": [1, 2, 3],
            "OTU2": [2, 4, 6],
            "OTU3": [1, 1, 1],
        }
    )

    results = create_correlation_matrix_all_methods(otu_df)

    for matrix in results.values():
        assert matrix.shape == (2, 2)
        assert "OTU3" not in matrix.columns
        assert "OTU3" not in matrix.index
