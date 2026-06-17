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


# TEST create_correlation_matrix
@given(
    st.floats(min_value=0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0, allow_nan=False, allow_infinity=False),
)
def test_create_correlation_matrix_symmetrical_matrix(x, y, z):
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
    otu_df = pd.DataFrame(
        {
            "OTU1": [1, 2, 3, 4],
            "OTU2": [2, 4, 6, 8],
            "OTU3": [4, 3, 2, 1],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012", "69-074-6013"],
    )

    corr_matrix = create_correlation_matrix(otu_df, correlation_method="Pearson")

    expected_matrix = pd.DataFrame(
        {
            "OTU1": [1.0, 1.0, -1.0],
            "OTU2": [1.0, 1.0, -1.0],
            "OTU3": [-1.0, -1.0, 1.0],
        },
        index=["OTU1", "OTU2", "OTU3"],
    )

    pdt.assert_frame_equal(corr_matrix, expected_matrix, check_exact=False, atol=1e-12)


def test_create_correlation_matrix_remove_constant_columns():
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

    corr_matrix = create_correlation_matrix(otu_df, correlation_method="Pearson")

    expected_matrix = pd.DataFrame(
        {
            "OTU1": [1.0, 1.0, -1.0],
            "OTU2": [1.0, 1.0, -1.0],
            "OTU3": [-1.0, -1.0, 1.0],
        },
        index=["OTU1", "OTU2", "OTU3"],
    )

    pdt.assert_frame_equal(corr_matrix, expected_matrix, check_exact=False, atol=1e-12)


def test_create_correlation_matrix_all_constant_columns():
    otu_df = pd.DataFrame(
        {
            "OTU4": [1, 1, 1, 1],
            "OTU5": [0, 0, 0, 0],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012", "69-074-6013"],
    )

    corr_matrix = create_correlation_matrix(otu_df, correlation_method="Pearson")

    assert corr_matrix.shape == (0, 0)
    assert corr_matrix.empty


# TEST _compute_correlation_parallel


def test_compute_correlation_parallel_expected():
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


# TEST create_correlation_matrix_parallel


def test_create_correlation_matrix_parallel_same_result_as_classic():
    otu_df = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU2": [0, 5, 3],
            "OTU3": [2, 1, 4],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )

    corr_matrix = create_correlation_matrix(otu_df, correlation_method="Pearson")
    corr_matrix_parallel = create_correlation_matrix_parallel(
        otu_df,
        correlation_method="Pearson",
        num_processes=2,
    )

    pdt.assert_frame_equal(
        corr_matrix,
        corr_matrix_parallel,
    )


def test_create_correlation_matrix_parallel_works_with_different_num_processes():
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


# TEST create_network_from_correlation_matrix


def test_create_network_from_correlation_matrix_creates_only_nodes_with_edges():
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
        threshold=0.5,
    )

    assert set(G.nodes) == {"OTU1", "OTU2"}


def test_create_network_from_correlation_matrix_creates_only_nodes_with_edges_higher_than_treshold():
    correlation_matrix = pd.DataFrame(
        {
            "OTU1": [1.0, 0.8, 0.3],
            "OTU2": [0.8, 1.0, 0.3],
            "OTU3": [0.3, 0.3, 1.0],
        },
        index=["OTU1", "OTU2", "OTU3"],
    )

    G = create_network_from_correlation_matrix(
        correlation_matrix,
        threshold=0.5,
    )

    assert set(G.nodes) == {"OTU1", "OTU2"}


def test_create_network_from_correlation_matrix_weights_edges():
    correlation_matrix = pd.DataFrame(
        {
            "OTU1": [1.0, 0.8, 0.3],
            "OTU2": [0.8, 1.0, 0.3],
            "OTU3": [0.3, 0.3, 1.0],
        },
        index=["OTU1", "OTU2", "OTU3"],
    )

    G = create_network_from_correlation_matrix(
        correlation_matrix,
        threshold=0.5,
    )

    edge_weight = G["OTU1"]["OTU2"]["weight"]

    assert np.isclose(edge_weight, 0.8)
