import numpy as np
import pandas as pd
import pandas.testing as pdt

import pytest

from metagenomes_simulation.network import (
    create_network_from_correlation_matrix,
    network_density,
)

from metagenomes_simulation.generators import (
    create_block_correlation,
    create_sparse_correlation,
    create_sparse_block_correlation,
    simulate_absolute_abundances_ground_truth,
    simulate_sequencing,
    generate_gaussian_noise,
)

# --------------------------------------------------------------------------------------------
# TEST create_block_correlation
# --------------------------------------------------------------------------------------------


def test_create_block_correlation_work_correctly():
    """This tests that the block correlation matrix is created correctly.

    GIVEN: three communities containing three taxa each and
        two correlation coefficients
    WHEN: I create the block correlation matrix
    THEN: the function returns three blocks joined
        by the between-community correlation
    """
    otu_names = [f"OTU_{i + 1}" for i in range(9)]
    expected_corr_matrix = pd.DataFrame(
        np.array(
            [
                [1.0, 0.4, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.4, 1.0, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.4, 0.4, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 1.0, 0.4, 0.4, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.4, 1.0, 0.4, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.4, 0.4, 1.0, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.4, 0.4],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4, 1.0, 0.4],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4, 0.4, 1.0],
            ]
        ),
        index=otu_names,
        columns=otu_names,
    )

    corr_matrix = create_block_correlation(
        block_sizes=[3, 3, 3],
        within_correlation=0.4,
        between_correlation=0.1,
    )

    pd.testing.assert_frame_equal(corr_matrix, expected_corr_matrix)


def test_create_block_correlation_semipositive():
    """This tests that the block correlation matrix is positive semidefinite.

    GIVEN: valid block sizes and correlation coefficients
    WHEN: I create the block correlation matrix and compute its eigenvalues
    THEN: every eigenvalue is non-negative within numerical tolerance
    """
    corr_matrix = create_block_correlation([3, 3, 3], 0.4, 0.1)
    eigenvalues = np.linalg.eigvalsh(corr_matrix.to_numpy())
    assert np.all(eigenvalues >= -1e-12)


def test_create_block_correlation_minimum_dimensions():
    """This tests the smallest valid block correlation matrix.

    GIVEN: one taxon in one community
    WHEN: I create the block correlation matrix
    THEN: the function returns a one by one matrix containing one
    """
    corr_matrix = create_block_correlation([1], 0.4, 0.1)
    expected = pd.DataFrame([[1.0]], index=["OTU_1"], columns=["OTU_1"])
    pd.testing.assert_frame_equal(corr_matrix, expected)


def test_create_block_correlation_zero_correlations():
    """This tests a block correlation matrix with zero correlations.

    GIVEN: valid communities with both correlation coefficients equal to zero
    WHEN: I create the block correlation matrix
    THEN: the function returns an identity matrix
    """
    corr_matrix = create_block_correlation([3, 3], 0.0, 0.0)
    otu_names = [f"OTU_{i + 1}" for i in range(6)]
    expected = pd.DataFrame(np.eye(6), index=otu_names, columns=otu_names)
    pd.testing.assert_frame_equal(corr_matrix, expected)


def test_create_block_correlation_correct_shape():
    """This tests that the block correlation matrix has the correct shape.

    GIVEN: communities containing different numbers of taxa
    WHEN: I create the block correlation matrix
    THEN: each dimension equals the sum of the block sizes
    """
    corr_matrix = create_block_correlation([4, 3, 2], 0.4, 0.1)
    assert isinstance(corr_matrix, pd.DataFrame)
    assert corr_matrix.shape == (9, 9)
    assert corr_matrix.index.equals(corr_matrix.columns)


def test_create_block_correlation_error_negative_within_correlation():
    """This tests that a negative within correlation raises an error.

    GIVEN: a within correlation lower than zero
    WHEN: I create the block correlation matrix
    THEN: the function raises an error
    """
    with pytest.raises(ValueError):
        create_block_correlation([3, 3], -0.1, 0.1)


def test_create_block_correlation_error_within_correlation_equal_to_one():
    """This tests that a within correlation over one raises an error.

    GIVEN: a within correlation over one
    WHEN: I create the block correlation matrix
    THEN: the function raises an error
    """
    with pytest.raises(ValueError):
        create_block_correlation([3, 3], 1.1, 0.1)


def test_create_block_correlation_error_negative_between_correlation():
    """This tests that a negative between correlation raises an error.

    GIVEN: a between correlation lower than zero
    WHEN: I create the block correlation matrix
    THEN: the function raises an error
    """
    with pytest.raises(ValueError):
        create_block_correlation([3, 3], 0.4, -0.1)


def test_create_block_correlation_error_between_bigger_than_within():
    """This tests that a between correlation bigger than
    within correlation raises an error.

    GIVEN: a between correlation over within correlation
    WHEN: I create the block correlation matrix
    THEN: the function raises an error
    """
    with pytest.raises(ValueError):
        create_block_correlation([3, 3], 0.4, 0.5)


def test_create_block_correlation_error_non_positive_block_size():
    """This tests that a non-positive block size raises an error.

    GIVEN: a block containing a non-positive number of taxa
    WHEN: I create the block correlation matrix
    THEN: the function raises an error
    """
    with pytest.raises(ValueError):
        create_block_correlation([3, 0], 0.4, 0.1)


def test_create_block_correlation_error_empty_block_size():
    """This tests that an empty block size raises an error.

    GIVEN: a block containing an empty list
    WHEN: I create the block correlation matrix
    THEN: the function raises an error
    """
    with pytest.raises(ValueError):
        create_block_correlation([], 0.4, 0.1)


def test_create_block_correlation_error_non_list_block_size():
    """This tests that a non-list block size raises an error.

    GIVEN: a block containing a integer number of taxa
    WHEN: I create the block correlation matrix
    THEN: the function raises an error
    """
    with pytest.raises(ValueError):
        create_block_correlation(1, 0.4, 0.1)


def test_create_block_correlation_error_non_integer_block_size():
    """This tests that a non-integer block size raises an error.

    GIVEN: a block containing a non-integer number of taxa
    WHEN: I create the block correlation matrix
    THEN: the function raises an error
    """
    with pytest.raises(ValueError):
        create_block_correlation([3, 2.5], 0.4, 0.1)


# --------------------------------------------------------------------------------------------
# TEST create_sparse_correlation
# --------------------------------------------------------------------------------------------


def test_create_sparse_correlation_returns_dataframe_with_expected_shape():
    """Test that the correlation matrix has the expected type and dimensions.

    GIVEN: a valid number of taxa, density, and random seed
    WHEN: a sparse correlation matrix is created
    THEN: the result is a square DataFrame with one row and column per taxon
    """
    corr_matrix = create_sparse_correlation(
        n_taxa=10, density=0.2, random_state=12
    )

    assert isinstance(corr_matrix, pd.DataFrame)
    assert corr_matrix.shape == (10, 10)


def test_create_sparse_correlation_has_matching_axis_labels():
    """Test that rows and columns represent the same taxa in the same order.

    GIVEN: valid sparse-correlation parameters
    WHEN: a sparse correlation matrix is created
    THEN: its row and column labels are identical and equally ordered
    """
    corr_matrix = create_sparse_correlation(
        n_taxa=10, density=0.2, random_state=12
    )

    assert corr_matrix.index.equals(corr_matrix.columns)


def test_create_sparse_correlation_is_symmetric():
    """Test that the generated correlation matrix is symmetric.

    GIVEN: valid sparse-correlation parameters
    WHEN: a sparse correlation matrix is created
    THEN: the matrix equals its transpose within numerical tolerance
    """
    corr_matrix = create_sparse_correlation(
        n_taxa=10, density=0.2, random_state=12
    )

    np.testing.assert_allclose(
        corr_matrix.to_numpy(),
        corr_matrix.to_numpy().T,
    )


def test_create_sparse_correlation_has_unit_diagonal():
    """Test that every taxon is perfectly correlated with itself.

    GIVEN: valid sparse-correlation parameters
    WHEN: a sparse correlation matrix is created
    THEN: every value on its main diagonal is equal to one
    """
    corr_matrix = create_sparse_correlation(
        n_taxa=10, density=0.2, random_state=12
    )

    np.testing.assert_allclose(
        np.diag(corr_matrix.to_numpy()),
        np.ones(corr_matrix.shape[0]),
    )


def test_create_sparse_correlation_correct_density():
    """This tests that the requested number of sparse edges is created.

    GIVEN: ten taxa and density equal to 0.2
    WHEN: I create the sparse correlation matrix
    THEN: graph density is equal to the expected one
    """
    density_test = 0.2
    corr_matrix = create_sparse_correlation(
        n_taxa=10, density=density_test, random_state=12
    )
    graph = create_network_from_correlation_matrix(corr_matrix)

    assert pytest.approx(network_density(graph)) == density_test


def test_create_sparse_correlation_correct_negative_fraction():
    """This tests that the requested fraction of edges is negative.

    GIVEN: a sparse matrix with half of its edges requested as negative
    WHEN: I create the sparse correlation matrix
    THEN: half of the non-zero upper-triangle values are negative
    """
    corr_matrix = create_sparse_correlation(
        n_taxa=10,
        density=0.4,
        negative_fraction=0.5,
        random_state=12,
    )
    edges = corr_matrix.to_numpy()[np.triu_indices(10, k=1)]
    edges = edges[edges != 0]

    assert np.count_nonzero(edges < 0) == round(0.5 * len(edges))


def test_create_sparse_correlation_semipositive():
    """This tests that the sparse correlation matrix is positive semidefinite.

    GIVEN: valid sparse correlation parameters
    WHEN: I create the matrix and compute its eigenvalues
    THEN: every eigenvalue is non-negative within numerical tolerance
    """
    corr_matrix = create_sparse_correlation(
        n_taxa=10, density=0.4, random_state=12
    )
    eigenvalues = np.linalg.eigvalsh(corr_matrix.to_numpy())

    assert np.all(eigenvalues >= -1e-12)


def test_create_sparse_correlation_reproducible():
    """This tests that sparse matrix generation is reproducible.

    GIVEN: the same parameters and random seed
    WHEN: I create two sparse correlation matrices
    THEN: both matrices are equal
    """
    corr_matrix_1 = create_sparse_correlation(n_taxa=10, random_state=12)
    corr_matrix_2 = create_sparse_correlation(n_taxa=10, random_state=12)

    pd.testing.assert_frame_equal(corr_matrix_1, corr_matrix_2)


def test_create_sparse_correlation_error_non_positive_taxa():
    """This tests that a non-positive number of taxa raises an error.

    GIVEN: a non-positive number of taxa
    WHEN: I create the sparse correlation matrix
    THEN: the function raises an error
    """
    with pytest.raises(ValueError):
        create_sparse_correlation(n_taxa=0)


def test_create_sparse_correlation_error_invalid_density():
    """This tests that an invalid density raises an error.

    GIVEN: a density greater than one
    WHEN: I create the sparse correlation matrix
    THEN: the function raises an error
    """
    with pytest.raises(ValueError):
        create_sparse_correlation(density=1.1)


def test_create_sparse_correlation_error_invalid_correlation_range():
    """This tests that an invalid correlation range raises an error.

    GIVEN: a correlation range in decreasing order
    WHEN: I create the sparse correlation matrix
    THEN: the function raises an error
    """
    with pytest.raises(ValueError):
        create_sparse_correlation(correlation_range=(0.6, 0.2))


def test_create_sparse_correlation_error_invalid_negative_fraction():
    """This tests that an invalid negative fraction raises an error.

    GIVEN: a negative fraction lower than zero
    WHEN: I create the sparse correlation matrix
    THEN: the function raises an error
    """
    with pytest.raises(ValueError):
        create_sparse_correlation(negative_fraction=-0.1)


def test_create_sparse_correlation_error_invalid_random_state():
    """This tests that an invalid random state raises an error.

    GIVEN: a non-integer random state
    WHEN: I create the sparse correlation matrix
    THEN: the function raises an error
    """
    with pytest.raises(ValueError):
        create_sparse_correlation(random_state=12.5)


# --------------------------------------------------------------------------------------------
# TEST create_sparse_block_correlation
# --------------------------------------------------------------------------------------------


def test_create_sparse_block_correlation_returns_expected_shape():
    """Test that the sparse block matrix has the expected dimensions.

    GIVEN: communities containing different numbers of taxa
    WHEN: I create the sparse block correlation matrix
    THEN: each dimension equals the sum of the block sizes
    """
    corr_matrix, membership = create_sparse_block_correlation(
        block_sizes=[4, 3, 2],
        within_density=0.5,
        between_density=0.1,
        random_state=12,
    )

    assert isinstance(corr_matrix, pd.DataFrame)
    assert corr_matrix.shape == (9, 9)
    assert membership.shape[0] == 9


def test_create_sparse_block_correlation_has_matching_axis_labels():
    """Test that rows and columns represent the same taxa in the same order.

    GIVEN: valid sparse block correlation parameters
    WHEN: I create the sparse block correlation matrix
    THEN: its row and column labels are identical and equally ordered
    """
    corr_matrix, _ = create_sparse_block_correlation(
        block_sizes=[3, 3],
        within_density=0.5,
        between_density=0.1,
        random_state=12,
    )

    assert corr_matrix.index.equals(corr_matrix.columns)


def test_create_sparse_block_correlation_is_symmetric():
    """Test that the sparse block correlation matrix is symmetric.

    GIVEN: valid sparse block correlation parameters
    WHEN: I create the sparse block correlation matrix
    THEN: the matrix equals its transpose within numerical tolerance
    """
    corr_matrix, _ = create_sparse_block_correlation(
        block_sizes=[3, 4, 3],
        within_density=0.5,
        between_density=0.1,
        random_state=12,
    )

    np.testing.assert_allclose(
        corr_matrix.to_numpy(),
        corr_matrix.to_numpy().T,
    )


def test_create_sparse_block_correlation_has_unit_diagonal():
    """Test that every taxon is perfectly correlated with itself.

    GIVEN: valid sparse block correlation parameters
    WHEN: I create the sparse block correlation matrix
    THEN: every value on its main diagonal is equal to one
    """
    corr_matrix, _ = create_sparse_block_correlation(
        block_sizes=[3, 4, 3],
        within_density=0.5,
        between_density=0.1,
        random_state=12,
    )

    np.testing.assert_allclose(
        np.diag(corr_matrix.to_numpy()),
        np.ones(corr_matrix.shape[0]),
    )


def test_create_sparse_block_correlation_respects_communities():
    """Test that extreme densities create only within-community edges.

    GIVEN: within density equal to one and between density equal to zero
    WHEN: I create the sparse block correlation matrix
    THEN: every within-community edge is present and every between-community
        edge is absent
    """
    block_sizes = [3, 2]
    corr_matrix, membership = create_sparse_block_correlation(
        block_sizes=block_sizes,
        within_density=1.0,
        between_density=0.0,
        random_state=12,
    )

    values = corr_matrix.to_numpy()

    same_community = membership[:, None] == membership[None, :]
    off_diagonal = ~np.eye(len(membership), dtype=bool)

    within_edges = same_community & off_diagonal
    between_edges = ~same_community

    assert np.all(values[within_edges] > 0)
    np.testing.assert_allclose(values[between_edges], 0.0)


def test_create_sparse_block_correlation_semipositive():
    """Test that the sparse block matrix is positive semidefinite.

    GIVEN: valid sparse block correlation parameters
    WHEN: I create the matrix and compute its eigenvalues
    THEN: every eigenvalue is non-negative within numerical tolerance
    """
    corr_matrix, _ = create_sparse_block_correlation(
        block_sizes=[3, 4, 3],
        within_density=0.5,
        between_density=0.1,
        random_state=12,
    )
    eigenvalues = np.linalg.eigvalsh(corr_matrix.to_numpy())

    assert np.all(eigenvalues >= -1e-12)


def test_create_sparse_block_correlation_reproducible():
    """Test that sparse block matrix generation is reproducible.

    GIVEN: the same parameters and random seed
    WHEN: I create two sparse block correlation matrices
    THEN: both matrices are equal
    """
    corr_matrix_1, membership_1 = create_sparse_block_correlation(
        block_sizes=[3, 4, 3],
        within_density=0.5,
        between_density=0.1,
        random_state=12,
    )
    corr_matrix_2, membership_2 = create_sparse_block_correlation(
        block_sizes=[3, 4, 3],
        within_density=0.5,
        between_density=0.1,
        random_state=12,
    )

    pd.testing.assert_frame_equal(corr_matrix_1, corr_matrix_2)
    np.testing.assert_array_equal(membership_1, membership_2)


def test_create_sparse_block_correlation_returns_expected_memberships():
    """This tests that each taxon receives its block membership.

    GIVEN: communities containing different numbers of taxa
    WHEN: I create the sparse block correlation matrix
    THEN: the returned memberships identify the block of every taxon
    """
    _, memberships = create_sparse_block_correlation(
        block_sizes=[2, 3, 1],
        random_state=12,
    )

    expected = np.array([0, 0, 1, 1, 1, 2])
    np.testing.assert_array_equal(memberships, expected)


def test_create_sparse_block_correlation_requires_scaling():
    """This tests that the correlation matrix is rescaled when needed.

    GIVEN: high correlation values and a sparse block structure
    WHEN: I create the sparse block correlation matrix
    THEN: the matrix is positive definite and its minimum
        eigenvalue is equal to epsilon within numerical tolerance
    """
    corr_matrix, _ = create_sparse_block_correlation(
        block_sizes=[20, 20],
        within_density=0.5,
        between_density=0.2,
        correlation_range=(0.8, 0.99),
        random_state=12,
    )

    eigenvalues = np.linalg.eigvalsh(corr_matrix.to_numpy())

    assert np.all(eigenvalues > 0)

    assert np.isclose(
        eigenvalues.min(),
        1e-6,
        atol=1e-10,
        rtol=0,
    )


@pytest.mark.parametrize(
    "invalid_arguments",
    [
        {"within_density": 1.1},
        {"between_density": -0.1},
        {"within_density": 0.1, "between_density": 0.2},
        {"block_sizes": []},
        {"block_sizes": [3, 0]},
        {"block_sizes": [3, 2.5]},
        {"block_sizes": 3},
        {"random_state": 12.5},
        {"correlation_range": (-0.1, 0.1)},
        {"correlation_range": (0.6, 0.2)},
        {"correlation_range": (0.4, 1)},
        {"correlation_range": 1.2},
    ],
)
def test_create_sparse_block_correlation_error_invalid_inputs(
    invalid_arguments,
):
    """This tests that invalid input parameters raise an error.

    GIVEN: invalid densities, block sizes, correlation ranges,
        or random state
    WHEN: I create the sparse block correlation matrix
    THEN: the function raises an error
    """
    with pytest.raises(ValueError):
        create_sparse_block_correlation(**invalid_arguments)


# --------------------------------------------------------------------------------------------
# TEST simulate_absolute_abundances_ground_truth
# --------------------------------------------------------------------------------------------


def test_simulate_absolute_abundances_ground_truth_work_correctly():
    """This tests that absolute abundances are simulated correctly.

    GIVEN: a valid correlation matrix and simulation parameters
    WHEN: I simulate the absolute abundances
    THEN: the result has one row per sample,
        one column per taxon and positive values
    """
    correlation_matrix = create_block_correlation([2, 3], 0.4, 0.1)
    abundance_matrix = simulate_absolute_abundances_ground_truth(
        correlation_matrix, random_state=12
    )

    assert isinstance(abundance_matrix, pd.DataFrame)
    assert abundance_matrix.shape == (10, 5)
    assert list(abundance_matrix.index) == [f"Sample{i}" for i in range(10)]
    assert abundance_matrix.columns.equals(correlation_matrix.columns)
    assert (abundance_matrix > 0).all().all()
    assert all(
        np.issubdtype(dtype, np.floating) for dtype in abundance_matrix.dtypes
    )


def test_simulate_absolute_abundances_ground_truth_reproducible():
    """This tests that absolute abundance simulation is reproducible.

    GIVEN: the same correlation matrix, parameters and random seed
    WHEN: I simulate two absolute abundance matrices
    THEN: both matrices are equal
    """
    correlation_matrix = create_block_correlation([2, 2], 0.4, 0.1)
    abundance_matrix_1 = simulate_absolute_abundances_ground_truth(
        correlation_matrix, random_state=12
    )
    abundance_matrix_2 = simulate_absolute_abundances_ground_truth(
        correlation_matrix, random_state=12
    )

    pd.testing.assert_frame_equal(abundance_matrix_1, abundance_matrix_2)


def test_simulate_absolute_abundances_ground_truth_zero_variance():
    """This tests absolute abundance simulation with both variances equal to
    zero.

    GIVEN: a valid correlation matrix and
        both abundance variances equal to zero
    WHEN: I simulate the absolute abundances
    THEN: every abundance is equal to the requested mean abundance
    """
    correlation_matrix = create_block_correlation([2, 2], 0.4, 0.1)
    abundance_matrix = simulate_absolute_abundances_ground_truth(
        correlation_matrix,
        n_sample=3,
        mean_abundance=100,
        var_abundance=0,
        taxa_var_abundance=0,
        random_state=12,
    )

    expected = pd.DataFrame(
        np.full((3, 4), 100.0),
        index=["Sample0", "Sample1", "Sample2"],
        columns=correlation_matrix.columns,
    )
    pd.testing.assert_frame_equal(abundance_matrix, expected)


def test_simulate_absolute_abundances_ground_truth_var_between_taxa():
    """This tests abundance between taxa is stable for zero variance between
    sample and positive variance between taxa.

    GIVEN: positive between-taxa variance and zero variation between samples
    WHEN: I simulate absolute abundances
    THEN: taxa have different abundances that remain fixed across samples
    """
    correlation_matrix = create_block_correlation([2, 2], 0.4, 0.1)
    abundance_matrix = simulate_absolute_abundances_ground_truth(
        correlation_matrix,
        n_sample=3,
        mean_abundance=100,
        var_abundance=0,
        taxa_var_abundance=0.5,
        random_state=12,
    )

    np.testing.assert_allclose(
        abundance_matrix.iloc[0], abundance_matrix.iloc[1]
    )
    np.testing.assert_allclose(
        abundance_matrix.iloc[0], abundance_matrix.iloc[2]
    )
    assert not np.allclose(
        abundance_matrix.iloc[:, 0], abundance_matrix.iloc[:, 1]
    )


def test_simulate_absolute_abundances_with_zero_variances_is_constant():
    """Test that zero variances produce identical abundances.

    GIVEN: four taxa with no between-taxon or between-sample variation
    WHEN: their absolute abundances are simulated
    THEN: every taxon has `mean_abundance` in every sample
    """
    correlation_matrix = create_block_correlation([2, 2], 0.4, 0.1)

    abundances = simulate_absolute_abundances_ground_truth(
        correlation_matrix=correlation_matrix,
        n_sample=4,
        mean_abundance=500,
        var_abundance=0,
        taxa_var_abundance=0,
        random_state=12,
    )

    np.testing.assert_allclose(abundances.to_numpy(), np.full((4, 4), 500.0))


def test_simulate_absolute_abundances_ground_truth_error_non_square_matrix():
    """This tests that a non-square correlation matrix raises an error.

    GIVEN: a non-square correlation matrix
    WHEN: I simulate the absolute abundances
    THEN: the function raises an error
    """
    with pytest.raises(ValueError):
        correlation_matrix = pd.DataFrame(
            np.ones((2, 3)),
            index=["OTU1", "OTU2"],
            columns=["OTU1", "OTU2", "OTU3"],
        )
        simulate_absolute_abundances_ground_truth(correlation_matrix)


def test_simulate_absolute_abundances_ground_truth_error_non_dataframe():
    """This tests that a non-dataframe correlation matrix raises an error.

    GIVEN: a correlation matrix represented by a NumPy array
    WHEN: I simulate the absolute abundances
    THEN: the function raises an error
    """
    with pytest.raises(ValueError):
        simulate_absolute_abundances_ground_truth(np.eye(2))


def test_simulate_absolute_abundances_ground_truth_non_semipositive_matrix():
    """This tests that a non-positive-semidefinite matrix raises an error.

    GIVEN: a symmetric matrix with a unit diagonal but a negative eigenvalue
    WHEN: I simulate the absolute abundances
    THEN: the function raises an error
    """
    correlation_matrix = pd.DataFrame(
        [
            [1.0, 0.9, 0.9],
            [0.9, 1.0, -0.9],
            [0.9, -0.9, 1.0],
        ],
        index=["OTU1", "OTU2", "OTU3"],
        columns=["OTU1", "OTU2", "OTU3"],
    )
    with pytest.raises(ValueError):
        simulate_absolute_abundances_ground_truth(correlation_matrix)


def test_simulate_absolute_abundances_ground_truth_non_symmetric_matrix():
    """This tests that a non-symmetric correlation matrix raises an error.

    GIVEN: a square but non-symmetric correlation matrix
    WHEN: I simulate the absolute abundances
    THEN: the function raises an error
    """
    correlation_matrix = pd.DataFrame(
        [[1.0, 0.4], [0.1, 1.0]],
        index=["OTU1", "OTU2"],
        columns=["OTU1", "OTU2"],
    )
    with pytest.raises(ValueError):
        simulate_absolute_abundances_ground_truth(correlation_matrix)


def test_simulate_absolute_abundances_ground_truth_error_non_unit_diagonal():
    """This tests that a matrix without a unit diagonal raises an error.

    GIVEN: a symmetric matrix whose diagonal does not contain only ones
    WHEN: I simulate the absolute abundances
    THEN: the function raises an error
    """
    correlation_matrix = pd.DataFrame(
        [[1.0, 0.4], [0.4, 0.5]],
        index=["OTU1", "OTU2"],
        columns=["OTU1", "OTU2"],
    )
    with pytest.raises(ValueError):
        simulate_absolute_abundances_ground_truth(correlation_matrix)


def test_simulate_absolute_abundances_ground_truth_non_positive_samples():
    """This tests that a non-positive number of samples raises an error.

    GIVEN: a non-positive number of samples
    WHEN: I simulate the absolute abundances
    THEN: the function raises an error
    """
    correlation_matrix = create_block_correlation([2], 0.4, 0.1)
    with pytest.raises(ValueError):
        simulate_absolute_abundances_ground_truth(
            correlation_matrix, n_sample=0
        )


def test_simulate_absolute_abundances_ground_truth_error_non_positive_mean():
    """This tests that a non-positive mean abundance raises an error.

    GIVEN: a non-positive mean abundance
    WHEN: I simulate the absolute abundances
    THEN: the function raises an error
    """
    correlation_matrix = create_block_correlation([2], 0.4, 0.1)
    with pytest.raises(ValueError):
        simulate_absolute_abundances_ground_truth(
            correlation_matrix, mean_abundance=0
        )


def test_simulate_absolute_abundances_ground_truth_error_negative_variance():
    """This tests that a negative variance raises an error.

    GIVEN: a negative abundance variance
    WHEN: I simulate the absolute abundances
    THEN: the function raises an error
    """
    correlation_matrix = create_block_correlation([2], 0.4, 0.1)
    with pytest.raises(ValueError):
        simulate_absolute_abundances_ground_truth(
            correlation_matrix, var_abundance=-0.5
        )


def test_simulate_absolute_abundances_ground_truth_negative_taxa_variance():
    """This tests that a negative between-taxa variance raises an error.

    GIVEN: a negative variance for stable abundance differences between taxa
    WHEN: I simulate the absolute abundances
    THEN: the function raises an error
    """
    correlation_matrix = create_block_correlation([2], 0.4, 0.1)
    with pytest.raises(ValueError):
        simulate_absolute_abundances_ground_truth(
            correlation_matrix, taxa_var_abundance=-0.5
        )


def test_simulate_absolute_abundances_ground_truth_invalid_random_state():
    """This tests that an invalid random state raises an error.

    GIVEN: a non-integer random state
    WHEN: I simulate the absolute abundances
    THEN: the function raises an error
    """
    correlation_matrix = create_block_correlation([2], 0.4, 0.1)
    with pytest.raises(ValueError):
        simulate_absolute_abundances_ground_truth(
            correlation_matrix, random_state=1.23
        )


# --------------------------------------------------------------------------------------------
# TEST simulate_sequencing
# --------------------------------------------------------------------------------------------


def test_simulate_sequencing_work_correctly():
    """This tests that sequencing sampling works correctly.

    GIVEN: absolute abundances for two samples and three OTUs
    WHEN: I simulate sequencing with one thousand counts per sample
    THEN: the function returns an integer dataframe with the expected
    shape and columns
    """
    absolute_abundance_matrix = pd.DataFrame(
        [[0.5, 0.3, 0.2], [0.1, 0.2, 0.7]],
        index=["SampleA", "SampleB"],
        columns=["OTU1", "OTU2", "OTU3"],
    )
    sequencing_df = simulate_sequencing(
        absolute_abundance_matrix, 1000, random_state=12
    )

    assert isinstance(sequencing_df, pd.DataFrame)
    assert sequencing_df.shape == (2, 3)
    assert list(sequencing_df.columns) == ["OTU1", "OTU2", "OTU3"]
    assert list(sequencing_df.index) == ["SampleA", "SampleB"]
    assert all(
        np.issubdtype(dtype, np.integer) for dtype in sequencing_df.dtypes
    )


def test_simulate_sequencing_poisson_depth():
    """This tests the statistical moments of the sequencing depth.

    GIVEN: 10000 samples with identical absolute abundances
    WHEN: I simulate sequencing with a mean depth of 1000
    THEN: both the mean and variance of the sequencing depth
        are close to 1000
    """
    sequencing_depth = 1000

    abundance_matrix = pd.DataFrame(
        np.ones((10000, 3)),
        columns=["OTU1", "OTU2", "OTU3"],
    )

    sequencing_df = simulate_sequencing(
        abundance_matrix,
        sequencing_depth=sequencing_depth,
        random_state=12,
    )

    depths = sequencing_df.sum(axis=1)

    assert np.isclose(depths.mean(), sequencing_depth, rtol=0.01)
    assert np.isclose(depths.var(), sequencing_depth, rtol=0.05)


def test_simulate_sequencing_rows_sum_to_sequencing_depth():
    """This tests that every sequencing sample fluctuate around the
    requested number of counts.

    GIVEN: absolute abundances for 1000 samples
    WHEN: I simulate sequencing sequencing with a mean depth of 1000
    THEN: sequencing depths have a mean close to 1000
    """
    correlation_matrix = create_block_correlation([1], 0.4, 0.1)

    absolute_abundance_matrix = simulate_absolute_abundances_ground_truth(
        correlation_matrix=correlation_matrix,
        n_sample=1000,
        random_state=12,
    )

    sequencing_df = simulate_sequencing(
        absolute_abundance_matrix, sequencing_depth=1000, random_state=12
    )

    depths = sequencing_df.sum(axis=1)
    assert np.isclose(depths.mean(), 1000, atol=5)


def test_simulate_sequencing_accepts_integer_scientific_notation():
    """This tests that an integer written in scientific notation is accepted.

    GIVEN: a sequencing depth equal to 1e3
    WHEN: I simulate sequencing
    THEN: every sample contains one thousand counts
    """
    correlation_matrix = create_block_correlation([1], 0.4, 0.1)

    absolute_abundance_matrix = simulate_absolute_abundances_ground_truth(
        correlation_matrix=correlation_matrix,
        n_sample=2,
        random_state=12,
    )
    sequencing_df = simulate_sequencing(
        absolute_abundance_matrix, 1e3, random_state=12
    )

    assert sequencing_df.shape == absolute_abundance_matrix.shape


def test_simulate_sequencing_reproducible():
    """This tests that sequencing sampling is reproducible.

    GIVEN: the same absolute abundances, sequencing depth and random seed
    WHEN: I simulate sequencing twice
    THEN: both dataframes are equal
    """
    correlation_matrix = create_block_correlation([1], 0.4, 0.1)

    absolute_abundance_matrix = simulate_absolute_abundances_ground_truth(
        correlation_matrix=correlation_matrix,
        n_sample=2,
        random_state=12,
    )
    sequencing_df_1 = simulate_sequencing(
        absolute_abundance_matrix, 1000, random_state=12
    )
    sequencing_df_2 = simulate_sequencing(
        absolute_abundance_matrix, 1000, random_state=12
    )

    pd.testing.assert_frame_equal(sequencing_df_1, sequencing_df_2)


def test_simulate_sequencing_error_non_matrix_input():
    """This tests that a non-matrix input raises an error.

    GIVEN: a one-dimensional absolute abundance array
    WHEN: I simulate sequencing
    THEN: the function raises an error
    """
    with pytest.raises(ValueError):
        simulate_sequencing(np.array([10, 10]), 1000)


def test_simulate_sequencing_error_empty_abundance():
    """This tests that an empty absolute abundance raises an error.

    GIVEN: an empty dataframe as absolute abundance
    WHEN: I simulate sequencing
    THEN: the function raises an error
    """
    abundance = pd.DataFrame([], columns=["OTU1", "OTU2", "OTU3"])
    with pytest.raises(ValueError):
        simulate_sequencing(abundance)


def test_simulate_sequencing_error_non_positive_sequencing_depth():
    """This tests that a non-positive sequencing depth raises an error.

    GIVEN: a non-positive number of sequencing depth per sample
    WHEN: I simulate sequencing
    THEN: the function raises an error
    """
    correlation_matrix = create_block_correlation([1], 0.4, 0.1)

    absolute_abundance_matrix = simulate_absolute_abundances_ground_truth(
        correlation_matrix=correlation_matrix,
        n_sample=2,
        random_state=12,
    )
    with pytest.raises(ValueError):
        simulate_sequencing(absolute_abundance_matrix, sequencing_depth=0)


def test_simulate_sequencing_error_non_integer_sequencing_depth():
    """This tests that a non integer sequencing_depth reaise an error.

    GIVEN: a sequencing depth equal to 1.3
    WHEN: I simulate sequencing
    THEN: the function raises an error
    """
    correlation_matrix = create_block_correlation([1], 0.4, 0.1)

    absolute_abundance_matrix = simulate_absolute_abundances_ground_truth(
        correlation_matrix=correlation_matrix,
        n_sample=2,
        random_state=12,
    )
    with pytest.raises(ValueError):
        simulate_sequencing(absolute_abundance_matrix, 1.3, random_state=12)


def test_simulate_sequencing_error_invalid_random_state():
    """This tests that an invalid random state raises an error.

    GIVEN: a non-integer random state
    WHEN: I simulate sequencing
    THEN: the function raises an error
    """
    correlation_matrix = create_block_correlation([1], 0.4, 0.1)

    absolute_abundance_matrix = simulate_absolute_abundances_ground_truth(
        correlation_matrix=correlation_matrix,
        n_sample=2,
        random_state=12,
    )
    with pytest.raises(ValueError):
        simulate_sequencing(absolute_abundance_matrix, 1000, random_state=12.5)


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
def test_simulate_sequencing_error_non_finite_abundance(value):
    """This tests that non-finite abundances raise an error.

    GIVEN: an abundance matrix containing a non-finite value
    WHEN: I simulate sequencing
    THEN: the function raises an error
    """
    absolute_abundance_matrix = pd.DataFrame({"OTU_1": [value]})

    with pytest.raises(ValueError):
        simulate_sequencing(absolute_abundance_matrix)


def test_simulate_sequencing_error_negative_abundance():
    """This tests that negative abundances raise an error.

    GIVEN: an abundance matrix containing a negative value
    WHEN: I simulate sequencing
    THEN: the function raises an error
    """
    absolute_abundance_matrix = pd.DataFrame({"OTU_1": [-1]})

    with pytest.raises(ValueError):
        simulate_sequencing(absolute_abundance_matrix)


def test_simulate_sequencing_error_zero_total_abundance():
    """This tests that samples with zero total abundance raise an error.

    GIVEN: an abundance matrix containing a sample with zero total abundance
    WHEN: I simulate sequencing
    THEN: the function raises an error
    """
    absolute_abundance_matrix = pd.DataFrame(
        [[10, 20, 30], [0, 0, 0]],
        columns=["OTU_1", "OTU_2", "OTU_3"],
    )

    with pytest.raises(ValueError):
        simulate_sequencing(absolute_abundance_matrix)


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
