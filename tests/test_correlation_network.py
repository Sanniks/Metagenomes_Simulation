from hypothesis import given, assume
from hypothesis import strategies as st
import pytest
import numpy as np
from metagenomes_simulation.correlation_network import (
    pearson_correlation,
    spearman_correlation,
    kendall_correlation,
)


# -----------------------------PEARSON_CORRELATION-----------------------------
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


# -----------------------------SPEARMAN_CORRELATION-----------------------------
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


# -----------------------------KENDALL_CORRELATION-----------------------------
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
