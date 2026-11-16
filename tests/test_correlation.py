from hypothesis import given, assume
from hypothesis import strategies as st
import pytest
import numpy as np
from metagenomes_simulation.correlation import (
    measure_correlation_coefficient,
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
