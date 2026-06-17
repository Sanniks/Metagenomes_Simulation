from hypothesis import given, assume
from hypothesis import strategies as st
import pytest
import numpy as np
from metagenomes_simulation.correlation import (
    pearson_correlation,
    spearman_correlation,
    kendall_correlation,
    correlation_coefficient,
)


# TEST pearson_correlation
def test_pearson_raise_error_for_different_array_dimentions():
    y = (0, 3)
    x = (0, 1, 2)

    with pytest.raises(ValueError):
        pearson_correlation(x, y)


@given(st.lists(st.floats(allow_nan=False, max_value=1e6, min_value=-1e6), min_size=2))
def test_pearson_correlation_coefficient_perfect_positive(x):
    assume(not np.allclose(x, x[0]))  # Remove cases where x is constant
    x = np.array(x)
    y = 3 * x
    assert pearson_correlation(x, y) == pytest.approx(1)


@given(st.lists(st.floats(allow_nan=False, max_value=1e6, min_value=-1e6), min_size=2))
def test_pearson_correlation_coefficient_perfect_negative(x):
    assume(not np.allclose(x, x[0]))  # Remove cases where x is constant
    x = np.array(x)
    y = -3 * x
    assert pearson_correlation(x, y) == pytest.approx(-1)


@given(
    st.lists(st.floats(allow_nan=False, max_value=1e6, min_value=-1e6), min_size=10, max_size=10),
    st.lists(st.floats(allow_nan=False, max_value=1e6, min_value=-1e6), min_size=10, max_size=10),
)
def test_pearson_is_between_minus_one_and_one(x, y):
    assume(not np.allclose(x, x[0]))
    assume(not np.allclose(y, y[0]))

    r = pearson_correlation(x, y)

    tol = 1e-8
    assert -1 - tol <= r <= 1 + tol


# TEST spearman_correlation
def test_spearman_raise_error_for_different_array_dimentions():
    y = (0, 3)
    x = (0, 1, 2)

    with pytest.raises(ValueError):
        spearman_correlation(x, y)


@given(st.lists(st.floats(allow_nan=False, max_value=1e6, min_value=-1e6), min_size=2))
def test_spearman_correlation_coefficient_perfect_positive(x):
    assume(not np.allclose(x, x[0]))  # Remove cases where x is constant
    x = np.array(x)
    y = 3 * x
    assert spearman_correlation(x, y) == pytest.approx(1)


@given(st.lists(st.floats(allow_nan=False, max_value=1e6, min_value=-1e6), min_size=2))
def test_spearman_correlation_coefficient_perfect_negative(x):
    assume(not np.allclose(x, x[0]))  # Remove cases where x is constant
    x = np.array(x)
    y = -3 * x
    assert spearman_correlation(x, y) == pytest.approx(-1)


@given(
    st.lists(st.floats(allow_nan=False, max_value=1e6, min_value=-1e6), min_size=10, max_size=10),
    st.lists(st.floats(allow_nan=False, max_value=1e6, min_value=-1e6), min_size=10, max_size=10),
)
def test_spearman_is_between_minus_one_and_one(x, y):
    assume(not np.allclose(x, x[0]))
    assume(not np.allclose(y, y[0]))

    r = spearman_correlation(x, y)

    tol = 1e-8
    assert -1 - tol <= r <= 1 + tol


# TEST kendall_correlation
def test_kendall_raise_error_for_different_array_dimentions():
    y = (0, 3)
    x = (0, 1, 2)

    with pytest.raises(ValueError):
        kendall_correlation(x, y)


@given(st.lists(st.floats(allow_nan=False, max_value=1e6, min_value=-1e6), min_size=2))
def test_kendall_correlation_coefficient_perfect_positive(x):
    assume(not np.allclose(x, x[0]))  # Remove cases where x is constant
    x = np.array(x)
    y = 3 * x
    assert kendall_correlation(x, y) == pytest.approx(1)


@given(st.lists(st.floats(allow_nan=False, max_value=1e6, min_value=-1e6), min_size=2))
def test_kendall_correlation_coefficient_perfect_negative(x):
    assume(not np.allclose(x, x[0]))  # Remove cases where x is constant
    x = np.array(x)
    y = -3 * x
    assert kendall_correlation(x, y) == pytest.approx(-1)


@given(
    st.lists(st.floats(allow_nan=False, max_value=1e6, min_value=-1e6), min_size=10, max_size=10),
    st.lists(st.floats(allow_nan=False, max_value=1e6, min_value=-1e6), min_size=10, max_size=10),
)
def test_kendall_is_between_minus_one_and_one(x, y):
    assume(not np.allclose(x, x[0]))
    assume(not np.allclose(y, y[0]))

    r = kendall_correlation(x, y)

    tol = 1e-8
    assert -1 - tol <= r <= 1 + tol


# TEST correlation_coefficient


def test_correlation_coefficient_returns_same_value_as_pearson():
    x = (1, 2, 3, 4, 5)
    y = (2, 4, 6, 8, 10)

    r_expected = pearson_correlation(x, y)
    r = correlation_coefficient(x, y, "Pearson")

    assert r == pytest.approx(r_expected)


def test_correlation_coefficient_returns_same_value_as_spearman():
    x = (1, 2, 3, 4, 5)
    y = (5, 6, 7, 8, 9)

    rho_expected = spearman_correlation(x, y)
    rho = correlation_coefficient(x, y, "Spearman")

    assert rho == pytest.approx(rho_expected)


def test_correlation_coefficient_returns_same_value_as_kendall():
    x = (1, 2, 3, 4, 5)
    y = (5, 4, 3, 2, 1)

    tau_expected = kendall_correlation(x, y)
    tau = correlation_coefficient(x, y, "Kendall")

    assert tau == pytest.approx(tau_expected)


def test_correlation_coefficient_is_case_insensitive():
    x = (1, 2, 3, 4, 5)
    y = (2, 4, 6, 8, 10)

    r_1 = correlation_coefficient(x, y, "Pearson")
    r_2 = correlation_coefficient(x, y, "pearson")
    r_3 = correlation_coefficient(x, y, "PEARSON")

    assert r_1 == pytest.approx(r_2)
    assert r_1 == pytest.approx(r_3)


def test_correlation_coefficient_raise_error_for_wrong_method():
    x = (1, 2, 3, 4, 5)
    y = (2, 4, 6, 8, 10)

    with pytest.raises(ValueError):
        correlation_coefficient(x, y, "WrongMethod")
