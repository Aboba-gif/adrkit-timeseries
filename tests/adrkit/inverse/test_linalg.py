"""Аналитические проверки отбеливания, единиц и симметрии SPD-метрик."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from adrkit.inverse.linalg import gram_root, whiten


def _hand_problem(kind):
    white_matrix = np.array([[1.0, 2.0, -1.0], [3.0, 4.0, 2.0]])
    white_vector = np.array([5.0, 6.0])
    if kind == "diagonal":
        covariance = np.array([4.0, 9.0])
        matrix = np.array([[2.0, 4.0, -2.0], [9.0, 12.0, 6.0]])
        vector = np.array([10.0, 18.0])
        root = np.array([[2.0, 0.0], [0.0, 3.0]])
    else:
        covariance = np.array([[4.0, 2.0], [2.0, 10.0]])
        matrix = np.array([[2.0, 4.0, -2.0], [10.0, 14.0, 5.0]])
        vector = np.array([10.0, 23.0])
        root = np.array([[2.0, 1.0], [0.0, 3.0]])
    return covariance, matrix, vector, white_matrix, white_vector, root


def test_dense_hand_factor_and_whitening():
    covariance, matrix, vector, expected_matrix, expected_vector, expected_root = _hand_problem(
        "dense"
    )
    actual_matrix, actual_vector = whiten(matrix, vector, covariance)
    root = gram_root(covariance, 2)
    assert_allclose(actual_matrix, expected_matrix, rtol=2e-15, atol=0)
    assert_allclose(actual_vector, expected_vector, rtol=2e-15, atol=0)
    assert_array_equal(root, expected_root)
    assert_array_equal(root.T @ root, covariance)
    assert actual_matrix.shape == (2, 3)
    assert actual_vector.shape == (2,)


@pytest.mark.parametrize("dense_diagonal", [False, True])
def test_diagonal_hand_problem_matches_both_metric_representations(dense_diagonal):
    covariance, matrix, vector, expected_matrix, expected_vector, expected_root = _hand_problem(
        "diagonal"
    )
    if dense_diagonal:
        covariance = np.diag(covariance)
    actual_matrix, actual_vector = whiten(matrix, vector, covariance)
    assert_allclose(actual_matrix, expected_matrix, rtol=2e-15, atol=0)
    assert_allclose(actual_vector, expected_vector, rtol=2e-15, atol=0)
    assert_array_equal(gram_root(covariance, 2), expected_root)


@pytest.mark.parametrize("kind", ["diagonal", "dense"])
@pytest.mark.parametrize("unit", [1e-8, 1.0, 1e8])
def test_whitening_is_invariant_to_observation_unit_rescaling(kind, unit):
    covariance, matrix, vector, expected_matrix, expected_vector, _ = _hand_problem(kind)
    actual_matrix, actual_vector = whiten(unit * matrix, unit * vector, unit**2 * covariance)
    assert_allclose(actual_matrix, expected_matrix, rtol=2e-14, atol=0)
    assert_allclose(actual_vector, expected_vector, rtol=2e-14, atol=0)


@pytest.mark.parametrize("scale", [1e-16, 1.0, 1e16])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("operation", ["whiten", "gram_root"])
def test_gross_asymmetry_is_rejected_for_all_units_and_triangles(scale, transposed, operation):
    covariance = scale * np.array([[1.0, 0.0], [0.9, 1.0]])
    if transposed:
        covariance = covariance.T
    with pytest.raises(ValueError, match="symmetric"):
        if operation == "whiten":
            whiten(np.eye(2), np.ones(2), covariance)
        else:
            gram_root(covariance, 2)


@pytest.mark.parametrize("scale", [1e-16, 1.0, 1e16])
def test_benign_asymmetry_uses_one_midpoint_for_both_triangles(scale):
    covariance, matrix, vector, expected_matrix, expected_vector, expected_root = _hand_problem(
        "dense"
    )
    delta = 2.0**-40
    supplied = scale * np.array([[4.0, 2.0 + delta], [2.0 - delta, 10.0]])
    original = supplied.copy()
    root = gram_root(supplied, 2)
    transposed_root = gram_root(supplied.T, 2)
    assert_array_equal(root, transposed_root)
    factor = root.T
    assert_allclose(factor @ factor.T, scale * covariance, rtol=4e-15, atol=0)
    assert_allclose(root, np.sqrt(scale) * expected_root, rtol=4e-15, atol=0)
    unit = np.sqrt(scale)
    actual_matrix, actual_vector = whiten(unit * matrix, unit * vector, supplied)
    transpose_matrix, transpose_vector = whiten(unit * matrix, unit * vector, supplied.T)
    assert_array_equal(actual_matrix, transpose_matrix)
    assert_array_equal(actual_vector, transpose_vector)
    assert_allclose(actual_matrix, expected_matrix, rtol=2e-14, atol=0)
    assert_allclose(actual_vector, expected_vector, rtol=2e-14, atol=0)
    assert_array_equal(supplied, original)


@pytest.mark.parametrize("variance", [1e-300, 5e-324])
@pytest.mark.parametrize("dense", [False, True])
def test_exact_tiny_and_subnormal_metrics_keep_the_scalar_factor(variance, dense):
    covariance = np.array([[variance]]) if dense else np.array([variance])
    standard_deviation = np.sqrt(variance)
    root = gram_root(covariance, 1)
    assert root[0, 0] > 0
    assert_array_equal(root, [[standard_deviation]])
    assert_allclose(root.T @ root, [[variance]], rtol=4e-15, atol=0)
    matrix, vector = whiten(
        np.array([[standard_deviation, -2 * standard_deviation]]),
        np.array([3 * standard_deviation]),
        covariance,
    )
    assert_allclose(matrix, [[1.0, -2.0]], rtol=2e-15, atol=0)
    assert_allclose(vector, [3.0], rtol=2e-15, atol=0)


@pytest.mark.parametrize(
    "matrix, vector",
    [
        ([1.0, 2.0], [1.0, 2.0]),
        (1.0, [1.0, 2.0]),
        ([[[1.0]], [[2.0]]], [1.0, 2.0]),
        ([[1.0, 0.0], [0.0, 1.0]], [[1.0], [2.0]]),
        ([[1.0, 0.0], [0.0, 1.0]], [[1.0, 2.0], [3.0, 4.0]]),
        ([[1.0, 0.0], [0.0, 1.0]], [1.0, 2.0, 3.0]),
    ],
)
def test_whiten_rejects_incompatible_shapes_instead_of_broadcasting(matrix, vector):
    with pytest.raises(ValueError):
        whiten(np.asarray(matrix), np.asarray(vector), np.array([1.0, 4.0]))


@pytest.mark.parametrize("covariance", [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 2.0], [2.0, 1.0]]])
@pytest.mark.parametrize("operation", ["whiten", "gram_root"])
def test_singular_or_indefinite_metrics_are_not_repaired(covariance, operation):
    covariance = np.asarray(covariance)
    with pytest.raises(ValueError, match="positive definite"):
        if operation == "whiten":
            whiten(np.eye(2), np.ones(2), covariance)
        else:
            gram_root(covariance, 2)
