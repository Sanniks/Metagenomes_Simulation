from metagenomes_simulation.network import (
    create_correlation_matrix,
    create_correlation_matrix_parallel,
    create_network_from_correlation_matrix,
    _compute_correlation_parallel,
)

import pandas as pd
import pandas.testing as pdt
import numpy as np

from hypothesis import given
from hypothesis import strategies as st

import pytest

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


def test_create_correlation_matrix_parallel_with_no_otus():
    """This tests the parallel function with no variable OTUs.

    GIVEN: a dataframe containing only constant OTUs
    WHEN: I create its correlation matrix in parallel
    THEN: the result is an empty dataframe
    """
    otu_df = pd.DataFrame(
        {
            "OTU1": [1, 1, 1],
            "OTU2": [0, 0, 0],
        }
    )

    corr_matrix = create_correlation_matrix_parallel(otu_df)

    assert corr_matrix.empty
    assert corr_matrix.shape == (0, 0)


# --------------------------------------------------------------------------------------------
# TEST _compute_correlation_parallel
# --------------------------------------------------------------------------------------------


def test_compute_correlation_parallel_expected():
    """This tests that a parallel correlation block returns the expected
    values.

    GIVEN: OTU data and a block containing
    two pairs of taxa with known relationships
    WHEN: I compute the block correlations with the parallel worker
    THEN: each result contains the expected taxon
    indices and Pearson correlation
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

    otu_data = otu_df.to_numpy()
    block = [(0, 1), (0, 2)]
    args = (otu_data, block, "Pearson")

    results = _compute_correlation_parallel(args)

    assert results[0][0] == 0
    assert results[0][1] == 1
    assert results[0][2] == pytest.approx(1.0)

    assert results[1][0] == 0
    assert results[1][1] == 2
    assert results[1][2] == pytest.approx(-1.0)


# --------------------------------------------------------------------------------------------
# TEST create_correlation_matrix_parallel
# --------------------------------------------------------------------------------------------


def test_create_correlation_matrix_parallel_same_result_as_classic():
    """This tests that parallel and sequential correlation matrices are
    equivalent.

    GIVEN: the same OTU dataframe and Pearson correlation method
    WHEN: I create the correlation matrix sequentially and in parallel
    THEN: both matrices are equal
    """
    otu_df = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU2": [0, 5, 3],
            "OTU3": [2, 1, 4],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )

    corr_matrix = create_correlation_matrix(
        otu_df, correlation_method="Pearson"
    )
    corr_matrix_parallel = create_correlation_matrix_parallel(
        otu_df,
        correlation_method="Pearson",
        num_processes=2,
    )

    pdt.assert_frame_equal(
        corr_matrix,
        corr_matrix_parallel,
    )


def test_create_correlation_matrix_parallel_with_different_num_processes():
    """This tests that the parallel result does not depend on the number of
    processes.

    GIVEN: an OTU dataframe and three valid process counts
    WHEN: I create a parallel Pearson correlation matrix
    with each process count
    THEN: all the resulting matrices are equal
    """
    otu_df = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU2": [0, 5, 3],
            "OTU3": [2, 1, 4],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )

    corr_matrix_1 = create_correlation_matrix_parallel(
        otu_df,
        correlation_method="Pearson",
        num_processes=1,
    )

    corr_matrix_2 = create_correlation_matrix_parallel(
        otu_df,
        correlation_method="Pearson",
        num_processes=2,
    )

    corr_matrix_4 = create_correlation_matrix_parallel(
        otu_df,
        correlation_method="Pearson",
        num_processes=4,
    )

    pdt.assert_frame_equal(corr_matrix_1, corr_matrix_2)
    pdt.assert_frame_equal(corr_matrix_1, corr_matrix_4)


def test_create_correlation_matrix_parallel_error_number_of_processes():
    """This tests that a non-positive number of processes raises an error.

    GIVEN: an OTU dataframe and a process count equal to zero
    WHEN: I create its parallel Pearson correlation matrix
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
        create_correlation_matrix_parallel(
            otu_df,
            correlation_method="Pearson",
            num_processes=0,
        )


def test_create_correlation_matrix_parallel_with_one_otu():
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

    corr_matrix = create_correlation_matrix_parallel(otu_df)

    expected_matrix = pd.DataFrame(
        [[1.0]],
        index=["OTU1"],
        columns=["OTU1"],
    )
    pdt.assert_frame_equal(corr_matrix, expected_matrix)


# --------------------------------------------------------------------------------------------
# TEST create_network_from_correlation_matrix
# --------------------------------------------------------------------------------------------


def test_create_network_from_correlation_matrix_keeps_isolated_nodes_default():
    """This tests that isolated taxa are kept in the network by default.

    GIVEN: a correlation matrix with one correlated pair and one isolated taxon
    WHEN: I create a network with a threshold equal to 0.5
    THEN: the network contains all three taxa and only one edge
    """
    correlation_matrix = pd.DataFrame(
        {
            "OTU1": [1.0, 0.8, 0.0],
            "OTU2": [0.8, 1.0, 0.0],
            "OTU3": [0.0, 0.0, 1.0],
        },
        index=["OTU1", "OTU2", "OTU3"],
    )

    G = create_network_from_correlation_matrix(
        correlation_matrix,
    )

    assert set(G.vs["name"]) == {"OTU1", "OTU2", "OTU3"}
    assert G.ecount() == 1


def test_create_network_from_correlation_matrix_remove_isolat_node_requested():
    """This tests that isolated taxa can still be removed explicitly.

    GIVEN: a correlation matrix with one correlated pair and one isolated taxon
    WHEN: I create a network with removal of isolated nodes enabled
    THEN: the network contains only the two taxa connected by an edge
    """
    correlation_matrix = pd.DataFrame(
        {
            "OTU1": [1.0, 0.8, 0.0],
            "OTU2": [0.8, 1.0, 0.0],
            "OTU3": [0.0, 0.0, 1.0],
        },
        index=["OTU1", "OTU2", "OTU3"],
    )

    G = create_network_from_correlation_matrix(
        correlation_matrix,
        remove_isolated=True,
    )

    assert set(G.vs["name"]) == {"OTU1", "OTU2"}
    assert G.ecount() == 1


def test_create_network_from_correlation_matrix_keeps_nodes_without_edges():
    """This tests that a network without edges still contains all taxa.

    GIVEN: a correlation matrix without off-diagonal values above the threshold
    WHEN: I create a network without removing isolated nodes
    THEN: the network contains every taxon and no edges
    """
    correlation_matrix = pd.DataFrame(
        np.eye(3),
        index=["OTU1", "OTU2", "OTU3"],
        columns=["OTU1", "OTU2", "OTU3"],
    )

    G = create_network_from_correlation_matrix(
        correlation_matrix,
    )

    assert set(G.vs["name"]) == {"OTU1", "OTU2", "OTU3"}
    assert G.ecount() == 0


def test_create_network_from_correlation_matrix_thresholds_absolute_weights():
    """This tests that retained edges preserve signed correlation weights.

    GIVEN: a matrix containing positive, negative,
    and below-threshold correlations
    WHEN: I create a network with a threshold equal to 0.5
    THEN: only absolute correlations above the threshold
    become signed weighted edges
    """
    correlation_matrix = pd.DataFrame(
        {
            "OTU1": [1.0, 0.8, 0.3],
            "OTU2": [0.8, 1.0, -0.6],
            "OTU3": [0.3, -0.6, 1.0],
        },
        index=["OTU1", "OTU2", "OTU3"],
    )

    G = create_network_from_correlation_matrix(
        correlation_matrix,
        threshold=0.5,
    )

    positive_edge = G.es[G.get_eid("OTU1", "OTU2")]["weight"]
    negative_edge = G.es[G.get_eid("OTU2", "OTU3")]["weight"]

    assert G.ecount() == 2
    assert np.isclose(positive_edge, 0.8)
    assert np.isclose(negative_edge, -0.6)


def test_create_network_from_correlation_matrix_binary_weights():
    """This tests the creation of a binary network.

    GIVEN: a matrix with positive and negative correlations
    WHEN: I create a binary network
    THEN: every retained edge has weight one
    """
    correlation_matrix = pd.DataFrame(
        {
            "OTU1": [1.0, 0.8, 0.0],
            "OTU2": [0.8, 1.0, -0.6],
            "OTU3": [0.0, -0.6, 1.0],
        },
        index=["OTU1", "OTU2", "OTU3"],
    )

    G = create_network_from_correlation_matrix(
        correlation_matrix,
        threshold=0.5,
        binary_network=True,
    )

    assert G.ecount() == 2
    assert G.es[G.get_eid("OTU1", "OTU2")]["weight"] == 1
    assert G.es[G.get_eid("OTU2", "OTU3")]["weight"] == 1


def test_create_network_binary_with_zero_threshold_keeps_one():
    """This tests a binary network with a one threshold.
    Edge case for threshold.

    GIVEN: a matrix with zero correlations
    WHEN: I create a binary network with threshold one
    THEN: edges are not created from nothing
    """
    correlation_matrix = pd.DataFrame(
        np.eye(3),
        index=["OTU1", "OTU2", "OTU3"],
        columns=["OTU1", "OTU2", "OTU3"],
    )

    G = create_network_from_correlation_matrix(
        correlation_matrix,
        threshold=1,
        binary_network=True,
    )

    assert G.ecount() == 0


def test_create_network_rejects_threshold_outside_valid_range():
    """This tests invalid threshold values.

    GIVEN: a correlation matrix
    WHEN: I use a threshold outside the interval from zero to one
    THEN: the function raises an error
    """
    correlation_matrix = pd.DataFrame(
        np.eye(2),
        index=["OTU1", "OTU2"],
        columns=["OTU1", "OTU2"],
    )

    with pytest.raises(ValueError):
        create_network_from_correlation_matrix(
            correlation_matrix,
            threshold=-0.1,
        )

    with pytest.raises(ValueError):
        create_network_from_correlation_matrix(
            correlation_matrix,
            threshold=1.1,
        )


def test_create_network_keeps_edge_equal_to_threshold():
    """This tests a correlation equal to the threshold.

    GIVEN: a correlation equal to the threshold
    WHEN: I create the network
    THEN: the corresponding edge is retained
    """
    correlation_matrix = pd.DataFrame(
        {
            "OTU1": [1.0, 0.5],
            "OTU2": [0.5, 1.0],
        },
        index=["OTU1", "OTU2"],
    )

    G = create_network_from_correlation_matrix(
        correlation_matrix,
        threshold=0.5,
    )

    assert G.ecount() == 1
    assert G.es[0]["weight"] == 0.5


def test_create_network_removes_all_isolated_nodes():
    """This tests removal when every node is isolated.

    GIVEN: a matrix without correlations between different OTUs
    WHEN: I request the removal of isolated nodes
    THEN: the result is an empty graph
    """
    correlation_matrix = pd.DataFrame(
        np.eye(3),
        index=["OTU1", "OTU2", "OTU3"],
        columns=["OTU1", "OTU2", "OTU3"],
    )

    G = create_network_from_correlation_matrix(
        correlation_matrix,
        remove_isolated=True,
    )

    assert G.vcount() == 0
    assert G.ecount() == 0
