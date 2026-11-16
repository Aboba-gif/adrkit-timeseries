"""KKT-обратное действие на нижней, верхней и фиксированной гранях коробки."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator

from adrkit.inverse.constraints import stationary_box_inverse


def _options():
    return {"kkt_tolerance": 1e-10, "active_margin": 1e-8, "solve_rtol": 1e-11, "maxiter": 100}


def test_coupled_quadratic_inverse_restricts_both_bounds_and_fixed_coordinates():
    # Around point, the objective is g.T*delta + delta.T*H*delta/2.
    # H is strictly diagonally dominant and positive definite. Free H is
    # [[4,1],[1,3]], so its inverse is [[3,-1],[-1,4]]/11.
    hessian = np.array(
        [
            [6, 1, 1, 0, 1],
            [1, 7, 0, 1, -1],
            [1, 0, 4, 1, 0.5],
            [0, 1, 1, 3, -0.5],
            [1, -1, 0.5, -0.5, 8],
        ]
    )
    point = [0.0, 1.0, 0.5, -0.5, 2.0]
    result = stationary_box_inverse(
        point,
        [1.0, -2.0, 0.0, 0.0, 123.0],
        hessian,
        [0.0, -1.0, -2.0, -2.0, 2.0],
        [3.0, 1.0, 2.0, 2.0, 2.0],
        **_options(),
    )
    expected = np.zeros((5, 5))
    expected[2:4, 2:4] = np.array([[3.0, -1.0], [-1.0, 4.0]]) / 11
    np.testing.assert_allclose(result.operator @ np.eye(5), expected, rtol=2e-11, atol=2e-13)
    force = np.array([3.0, -1.0, 0.4, -0.7, 2.0])
    np.testing.assert_allclose(
        result.operator @ force,
        [0.0, 0.0, 1.9 / 11, -3.2 / 11, 0.0],
        rtol=2e-11,
        atol=2e-13,
    )
    covector = np.array([-0.2, 0.3, 0.7, -0.4, 1.0])
    np.testing.assert_allclose(
        covector @ (result.operator @ force),
        force @ result.operator.rmatvec(covector),
        rtol=2e-11,
        atol=2e-13,
    )
    assert result.metadata["free_indices"] == (2, 3)
    assert result.metadata["lower_active_indices"] == (0,)
    assert result.metadata["upper_active_indices"] == (1,)
    assert result.metadata["fixed_indices"] == (4,)
    assert result.metadata["lower_gradient_min"] == 1.0
    assert result.metadata["upper_negative_gradient_min"] == 2.0
    assert result.metadata["projected_gradient_inf"] == 0.0


def test_dense_and_self_adjoint_operator_give_the_same_unconstrained_inverse():
    hessian = np.array([[2.0, 1.0], [1.0, 3.0]])
    action = LinearOperator((2, 2), matvec=lambda vector: hessian @ vector, dtype=float)
    results = [
        stationary_box_inverse([1.0, 2.0], [0.0, 0.0], matrix, -np.inf, np.inf, **_options())
        for matrix in (hessian, action)
    ]
    expected = np.array([[3.0, -1.0], [-1.0, 2.0]]) / 5
    for result in results:
        np.testing.assert_allclose(result.operator @ np.eye(2), expected, rtol=2e-11, atol=2e-13)
        assert result.metadata["positive_definiteness"] == "not_checked"


def test_indefinite_stationary_hessian_is_not_reported_as_a_minimum():
    result = stationary_box_inverse(
        [0.0, 0.0], [0.0, 0.0], np.diag([-2.0, 3.0]), -np.inf, np.inf, **_options()
    )
    np.testing.assert_allclose(result.operator @ [1.0, 1.0], [-0.5, 1 / 3], rtol=2e-11, atol=2e-13)
    assert result.metadata["positive_definiteness"] == "not_checked"


def test_fixed_box_coordinates_have_no_sign_condition_and_need_no_hessian_action():
    def forbidden(vector):
        raise AssertionError("An all-fixed box has no reduced Hessian")

    hessian = LinearOperator((2, 2), matvec=forbidden, dtype=float)
    result = stationary_box_inverse(
        [1.0, 2.0], [-1e308, 1e308], hessian, [1.0, 2.0], [1.0, 2.0], **_options()
    )
    np.testing.assert_array_equal(result.operator @ [5.0, -6.0], [0.0, 0.0])
    np.testing.assert_array_equal(result.operator.rmatvec([3.0, 4.0]), [0.0, 0.0])
    assert result.metadata["fixed_indices"] == (0, 1)
    assert result.metadata["free_indices"] == ()
    assert result.metadata["lower_gradient_min"] is None
    assert result.metadata["upper_negative_gradient_min"] is None


@pytest.mark.parametrize("point,gradient", [(1e-12, 1.0), (1 - 1e-12, -1.0)])
def test_small_projected_residual_near_bounds_does_not_hide_a_large_free_gradient(point, gradient):
    with pytest.raises(ValueError, match="free stationarity"):
        stationary_box_inverse([point], [gradient], [[1.0]], 0.0, 1.0, **_options())


@pytest.mark.parametrize("point,gradient", [(0.0, -1e-12), (1.0, 1e-12), (0.0, 0.0), (1.0, 0.0)])
def test_weak_or_wrongly_signed_active_faces_are_not_smooth_branch_certificates(point, gradient):
    with pytest.raises(ValueError, match="active box face"):
        stationary_box_inverse([point], [gradient], [[1.0]], 0.0, 1.0, **_options())


def test_scalar_bounds_allow_a_strict_lower_face_and_an_unbounded_upper_side():
    result = stationary_box_inverse(
        [0.0, 1.0], [1.0, 0.0], np.diag([2.0, 3.0]), 0.0, np.inf, **_options()
    )
    np.testing.assert_allclose(result.operator @ [1.0, 1.0], [0.0, 1 / 3], rtol=2e-11)
    assert result.metadata["lower"] == (0.0, 0.0)
    assert result.metadata["upper"] == (np.inf, np.inf)


def test_dense_frame_and_metadata_own_their_inputs_without_hidden_symmetrization():
    point, gradient = np.array([1.0, 2.0]), np.zeros(2)
    hessian = np.array([[2.0, 1.0], [1.0, 3.0]])
    lower, upper = np.array([-3.0, -4.0]), np.array([4.0, 5.0])
    result = stationary_box_inverse(point, gradient, hessian, lower, upper, **_options())
    direction = np.array([0.3, -0.5])
    before = result.operator @ direction
    for array in (point, gradient, hessian, lower, upper):
        array[:] = 90.0
    np.testing.assert_array_equal(result.operator @ direction, before)
    assert result.metadata["point"] == (1.0, 2.0)
    with pytest.raises(TypeError):
        result.metadata["point"] = (90.0, 90.0)
    with pytest.raises(FrozenInstanceError):
        result.operator = None
    with pytest.raises(ValueError, match="exactly symmetric"):
        stationary_box_inverse(
            [0.0, 0.0],
            [0.0, 0.0],
            [[2.0, 1.0], [1.0 + 1e-15, 2.0]],
            -np.inf,
            np.inf,
            **_options(),
        )


def test_singular_free_hessian_has_no_pseudoinverse_or_positive_residual_allowance():
    result = stationary_box_inverse([0.0], [0.0], [[0.0]], -1.0, 1.0, **_options())
    with pytest.raises(RuntimeError, match="true relative linear residual"):
        result.operator @ [1.0]
