from hypothesis import given, assume
from hypothesis import strategies as st
import pytest
import numpy as np
import pandas as pd
import pandas.testing as pdt

from metagenomes_simulation.correlation import (
    measure_correlation_coefficient,
    create_correlation_matrix,
    create_correlation_matrix_parallel,
    _compute_correlation_parallel,
)

# --------------------------------------------------------------------------------------------
# TEST pearson_correlation
# --------------------------------------------------------------------------------------------


def test_pearson_raise_error_for_different_array_dimentions():
    """This tests that the pearson correlation function raises an error for
    arguments of different dimensions.

    GIVEN: inputs x and y with dimension 2 and 3 respectively
    WHEN: I apply to it the correlation function
    THEN: the function raises an error
    """
    y = (0, 3)
    x = (0, 1, 2)

    with pytest.raises(ValueError):
        measure_correlation_coefficient(x, y, "pearson")


@given(
    st.lists(
        st.floats(allow_nan=False, max_value=1e6, min_value=-1e6), min_size=2
    )
)
def test_pearson_correlation_coefficient_perfect_positive(x):
    """This tests that the pearson correlation function returns 1 for any
    combination of values that is perfectly positively correlated.

    GIVEN: x is a random array of dim 2 given by the library hypothesis,
           y is defined positively proportional to x
    WHEN: I apply to it the correlation function
    THEN: the function returns 1
    """
    assume(not np.allclose(x, x[0]))  # Remove cases where x is constant
    x = np.array(x)
    y = 3 * x
    assert measure_correlation_coefficient(x, y, "pearson") == pytest.approx(1)


@given(
    st.lists(
        st.floats(allow_nan=False, max_value=1e6, min_value=-1e6), min_size=2
    )
)
def test_pearson_correlation_coefficient_perfect_negative(x):
    """This tests that the pearson correlation function returns -1 for any
    combination of values that is perfectly negatively correlated.

    GIVEN: x is a random array of dim 2 given by the library hypothesis,
           y is defined negatively proportional to x
    WHEN: I apply to it the correlation function
    THEN: the function returns -1
    """
    assume(not np.allclose(x, x[0]))  # Remove cases where x is constant
    x = np.array(x)
    y = -3 * x
    assert measure_correlation_coefficient(x, y, "pearson") == pytest.approx(
        -1
    )


@given(
    st.lists(
        st.floats(allow_nan=False, max_value=1e6, min_value=-1e6),
        min_size=10,
        max_size=10,
    ),
    st.lists(
        st.floats(allow_nan=False, max_value=1e6, min_value=-1e6),
        min_size=10,
        max_size=10,
    ),
)
def test_pearson_is_between_minus_one_and_one(x, y):
    """This tests the bounds of the pearson correlation function.

    GIVEN: x is a random array of dim 10 given by the library hypothesis,
           y is a random array of dim 10 given by the library hypothesis
    WHEN: I apply to it the correlation function
    THEN: the function returns a value between 1 and -1
    """
    assume(not np.allclose(x, x[0]))  # Remove cases where x is constant
    assume(not np.allclose(y, y[0]))  # Remove cases where y is constant

    r = measure_correlation_coefficient(x, y, "pearson")

    tol = 1e-8
    assert -1 - tol <= r <= 1 + tol


# --------------------------------------------------------------------------------------------
# TEST spearman_correlation
# --------------------------------------------------------------------------------------------


def test_spearman_raise_error_for_different_array_dimentions():
    """This tests that the spearman correlation function raises an error for
    arguments of different dimensions.

    GIVEN: inputs x and y with dimension 2 and 3 respectively
    WHEN: I apply to it the correlation function
    THEN: the function raises an error
    """
    y = (0, 3)
    x = (0, 1, 2)

    with pytest.raises(ValueError):
        measure_correlation_coefficient(x, y, "spearman")


@given(
    st.lists(
        st.floats(allow_nan=False, max_value=1e6, min_value=-1e6), min_size=2
    )
)
def test_spearman_correlation_coefficient_perfect_positive(x):
    """This tests that the spearman correlation function returns 1 for any
    combination of values that is perfectly positively correlated.

    GIVEN: x is a random array of dim 2 given by the library hypothesis,
           y is defined positively proportional to x
    WHEN: I apply to it the correlation function
    THEN: the function returns 1
    """
    assume(not np.allclose(x, x[0]))  # Remove cases where x is constant
    x = np.array(x)
    y = 3 * x
    assert measure_correlation_coefficient(x, y, "spearman") == pytest.approx(
        1
    )


@given(
    st.lists(
        st.floats(allow_nan=False, max_value=1e6, min_value=-1e6), min_size=2
    )
)
def test_spearman_correlation_coefficient_perfect_negative(x):
    """This tests that the spearman correlation function returns -1 for any
    combination of values that is perfectly negatively correlated.

    GIVEN: x is a random array of dim 2 given by the library hypothesis,
           y is defined negatively proportional to x
    WHEN: I apply to it the correlation function
    THEN: the function returns -1
    """
    assume(not np.allclose(x, x[0]))  # Remove cases where x is constant
    x = np.array(x)
    y = -3 * x
    assert measure_correlation_coefficient(x, y, "spearman") == pytest.approx(
        -1
    )


@given(
    st.lists(
        st.floats(allow_nan=False, max_value=1e6, min_value=-1e6),
        min_size=10,
        max_size=10,
    ),
    st.lists(
        st.floats(allow_nan=False, max_value=1e6, min_value=-1e6),
        min_size=10,
        max_size=10,
    ),
)
def test_spearman_is_between_minus_one_and_one(x, y):
    """This tests the bounds of the spearman correlation function.

    GIVEN: x is a random array of dim 10 given by the library hypothesis,
           y is a random array of dim 10 given by the library hypothesis
    WHEN: I apply to it the correlation function
    THEN: the function returns a value between 1 and -1
    """
    assume(not np.allclose(x, x[0]))
    assume(not np.allclose(y, y[0]))

    r = measure_correlation_coefficient(x, y, "spearman")

    tol = 1e-8
    assert -1 - tol <= r <= 1 + tol


# --------------------------------------------------------------------------------------------
# TEST kendall_correlation
# --------------------------------------------------------------------------------------------


def test_kendall_raise_error_for_different_array_dimentions():
    """This tests that the kendall correlation function raises an error for
    arguments of different dimensions.

    GIVEN: inputs x and y with dimension 2 and 3 respectively
    WHEN: I apply to it the correlation function
    THEN: the function raises an error
    """
    y = (0, 3)
    x = (0, 1, 2)

    with pytest.raises(ValueError):
        measure_correlation_coefficient(x, y, "kendall")


@given(
    st.lists(
        st.floats(allow_nan=False, max_value=1e6, min_value=-1e6), min_size=2
    )
)
def test_kendall_correlation_coefficient_perfect_positive(x):
    """This tests that the kendall correlation function returns 1 for any
    combination of values that is perfectly positively correlated.

    GIVEN: x is a random array of dim 2 given by the library hypothesis,
           y is defined positively proportional to x
    WHEN: I apply to it the correlation function
    THEN: the function returns 1
    """
    assume(not np.allclose(x, x[0]))  # Remove cases where x is constant
    x = np.array(x)
    y = 3 * x
    assert measure_correlation_coefficient(x, y, "kendall") == pytest.approx(1)


@given(
    st.lists(
        st.floats(allow_nan=False, max_value=1e6, min_value=-1e6), min_size=2
    )
)
def test_kendall_correlation_coefficient_perfect_negative(x):
    """This tests that the kendall correlation function returns -1 for any
    combination of values that is perfectly negatively correlated.

    GIVEN: x is a random array of dim 2 given by the library hypothesis,
           y is defined negatively proportional to x
    WHEN: I apply to it the correlation function
    THEN: the function returns -1
    """
    assume(not np.allclose(x, x[0]))  # Remove cases where x is constant
    x = np.array(x)
    y = -3 * x
    assert measure_correlation_coefficient(x, y, "kendall") == pytest.approx(
        -1
    )


@given(
    st.lists(
        st.floats(allow_nan=False, max_value=1e6, min_value=-1e6),
        min_size=10,
        max_size=10,
    ),
    st.lists(
        st.floats(allow_nan=False, max_value=1e6, min_value=-1e6),
        min_size=10,
        max_size=10,
    ),
)
def test_kendall_is_between_minus_one_and_one(x, y):
    """This tests the bounds of the kendall correlation function.

    GIVEN: x is a random array of dim 10 given by the library hypothesis,
           y is a random array of dim 10 given by the library hypothesis
    WHEN: I apply to it the correlation function
    THEN: the function returns a value between 1 and -1
    """
    assume(not np.allclose(x, x[0]))
    assume(not np.allclose(y, y[0]))

    r = measure_correlation_coefficient(x, y, "kendall")

    tol = 1e-8
    assert -1 - tol <= r <= 1 + tol


# --------------------------------------------------------------------------------------------
# TEST correlation_coefficient
# --------------------------------------------------------------------------------------------


def test_correlation_coefficient_is_case_insensitive():
    """This tests if the wrap correlation function gives the same result if the
    argument of the correlation method change case on some characters.

    GIVEN: x and y are two arrays of dimension 5
    with perfectly positive correlation
    WHEN: I apply the correlation_coefficient function
    with "Pearson" written with different case.
    THEN: the correlation coefficient of the functions is the same
    """
    x = (1, 2, 3, 4, 5)
    y = (2, 4, 6, 8, 10)

    r_1 = measure_correlation_coefficient(x, y, "Pearson")
    r_2 = measure_correlation_coefficient(x, y, "pearson")
    r_3 = measure_correlation_coefficient(x, y, "PEARSON")

    assert r_1 == pytest.approx(r_2)
    assert r_1 == pytest.approx(r_3)


def test_correlation_coefficient_raise_error_for_wrong_method():
    """This tests if the wrap correlation function returns an error if the
    method selected is not found.

    GIVEN: x and y are two arrays of dimension 5
    with perfectly positive correlation
    WHEN: I apply the correlation_coefficient function with "WrongMethod" as
    the method argumetn
    THEN: the correlation coefficient raises an error
    """
    x = (1, 2, 3, 4, 5)
    y = (2, 4, 6, 8, 10)

    with pytest.raises(ValueError):
        measure_correlation_coefficient(x, y, "WrongMethod")


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
