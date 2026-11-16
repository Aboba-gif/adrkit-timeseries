"""Производная стационарной ветви ограниченного вариационного анализа."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from adrkit.inverse import constraints
from adrkit.inverse.constraints import nonnegative_analysis_linear_operator
from adrkit.maps import map_from_functions
from adrkit.spaces import ArraySpace


def _space(size, name):
    return ArraySpace(
        (size,), axes=(name,), coordinates=(tuple(range(size)),), units="dimensionless"
    )


def _affine(matrix):
    matrix = np.array(matrix, dtype=float)
    return map_from_functions(
        domain=_space(matrix.shape[1], "state"),
        codomain=_space(matrix.shape[0], "observation"),
        predict=lambda point: matrix @ point,
        vjp=lambda point, cotangent: matrix.T @ cotangent,
    )


def _nonlinear():
    return map_from_functions(
        domain=_space(1, "state"),
        codomain=_space(1, "observation"),
        predict=lambda point: point / (1.0 + point),
        vjp=lambda point, cotangent: cotangent / (1.0 + point) ** 2,
    )


def _options():
    return {"kkt_tolerance": 1e-10, "active_margin": 1e-8, "solve_rtol": 1e-10, "maxiter": 100}


def test_nonlinear_residual_curvature_changes_the_implicit_derivative():
    point = np.array([1.0])
    fixed_point = float(point[0])

    def curvature(multiplier, direction):
        return -2.0 * multiplier * direction / (1.0 + fixed_point) ** 3

    exact = nonnegative_analysis_linear_operator(
        [7 / 8], _nonlinear(), [1.0], [1.0], [1.0], point, observation_hvp=curvature, **_options()
    )

    def unused(*args):
        raise AssertionError("Gauss-Newton must not evaluate residual curvature")

    gauss_newton = nonnegative_analysis_linear_operator(
        [7 / 8],
        _nonlinear(),
        [1.0],
        [1.0],
        [1.0],
        point,
        observation_hvp=unused,
        curvature="gauss-newton",
        **_options(),
    )
    # At x=1: A'=1/4, A''=-1/4, zeta=-1/2, hence H=19/16, H_GN=17/16.
    np.testing.assert_allclose(
        exact.operator @ np.eye(2), [[16 / 19, 4 / 19]], rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(
        gauss_newton.operator @ np.eye(2), [[16 / 17, 4 / 17]], rtol=1e-12, atol=1e-14
    )
    point[:] = 20
    np.testing.assert_allclose(exact.operator @ [1.0, 0.0], [16 / 19], rtol=1e-12, atol=1e-14)
    assert exact.metadata["point"] == (1.0,)
    assert exact.metadata["curvature"] == "exact"
    assert gauss_newton.metadata["curvature"] == "gauss-newton"


def test_affine_correlated_covariance_and_active_face_match_the_reduced_normal_equation():
    matrix = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    covariance = np.array([[2.0, 0.5], [0.5, 1.5]])
    mass = np.array([2.0, 3.0, 4.0])
    point = np.array([0.0, 1.0, 2.0])
    multiplier = np.array([1.0, -0.5])
    gradient = np.array([0.75, 0.0, 0.0])
    reference = point + (matrix.T @ multiplier - gradient) / mass
    target = matrix @ point - covariance @ multiplier
    linearization = nonnegative_analysis_linear_operator(
        reference,
        _affine(matrix),
        target,
        mass,
        covariance,
        point,
        observation_hvp=lambda zeta, direction: np.zeros(3),
        **_options(),
    )
    free = [1, 2]
    precision = np.linalg.solve(covariance, np.eye(2))
    hessian = np.diag(mass) + matrix.T @ precision @ matrix
    expected = np.zeros((3, 5))
    right = np.column_stack((np.diag(mass), matrix.T @ precision))
    expected[free] = np.linalg.solve(hessian[np.ix_(free, free)], right[free])
    np.testing.assert_allclose(
        linearization.operator @ np.eye(5), expected, rtol=1e-11, atol=1e-13
    )
    direction = np.linspace(-0.3, 0.7, 5)
    cotangent = np.array([0.2, -0.4, 0.6])
    np.testing.assert_allclose(
        cotangent @ (linearization.operator @ direction),
        direction @ linearization.operator.rmatvec(cotangent),
        rtol=1e-11,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        linearization.operator.rmatvec(cotangent), expected.T @ cotangent, rtol=1e-11, atol=1e-13
    )
    assert linearization.metadata["free_indices"] == (1, 2)
    assert linearization.metadata["active_indices"] == (0,)
    assert linearization.metadata["active_gradient_min"] > 0


def test_fixed_linearization_owns_its_numeric_inputs_and_metadata():
    reference, point, target = np.array([1.0]), np.array([1.0]), np.array([1.0])
    mass, covariance = np.array([2.0]), np.array([3.0])
    linearization = nonnegative_analysis_linear_operator(
        reference,
        _affine([[1.0]]),
        target,
        mass,
        covariance,
        point,
        curvature="gauss-newton",
        **_options(),
    )
    for value in (reference, point, target, mass, covariance):
        value[:] = 90
    np.testing.assert_allclose(linearization.operator @ [1.0, 1.0], [1.0], rtol=1e-12, atol=1e-14)
    with pytest.raises(TypeError):
        linearization.metadata["point"] = (90.0,)
    with pytest.raises(FrozenInstanceError):
        linearization.operator = None


def test_empty_free_face_returns_zero_without_evaluating_the_hessian():
    def forbidden(*args):
        raise AssertionError("An all-active face has no reduced Hessian solve")

    linearization = nonnegative_analysis_linear_operator(
        [-1.0, -2.0],
        _affine(np.eye(2)),
        [0.0, 0.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
        observation_hvp=forbidden,
        **_options(),
    )
    np.testing.assert_array_equal(linearization.operator @ [1.0, 2.0, 3.0, 4.0], [0.0, 0.0])
    np.testing.assert_array_equal(linearization.operator.rmatvec([1.0, 2.0]), np.zeros(4))
    assert linearization.metadata["free_indices"] == ()
    assert linearization.metadata["free_min"] is None


@pytest.mark.parametrize(
    "reference,point,target,message",
    [
        ([0.0], [1.0], [0.0], "stationarity"),
        ([0.0], [0.0], [0.0], "weak"),
        ([-1.0], [1e-12], [0.0], "stationarity"),
    ],
)
def test_stationarity_and_strict_active_separation_are_required(reference, point, target, message):
    options = {**_options(), "kkt_tolerance": 1e-8}
    with pytest.raises(ValueError, match=message):
        nonnegative_analysis_linear_operator(
            reference,
            _affine([[1.0]]),
            target,
            [1.0],
            [1.0],
            point,
            curvature="gauss-newton",
            **options,
        )


def test_an_invertible_negative_hessian_is_not_replaced_by_gauss_newton():
    linearization = nonnegative_analysis_linear_operator(
        [5.0],
        _nonlinear(),
        [-0.5],
        [1 / 16],
        [1.0],
        [1.0],
        observation_hvp=lambda zeta, direction: -0.25 * zeta * direction,
        **_options(),
    )
    # Stationarity holds at x=1, while H=1/16+1/16-1/4=-1/8.
    np.testing.assert_allclose(linearization.operator @ [2.0, 3.0], [-7.0], rtol=1e-12, atol=1e-14)


def test_exact_mode_requires_a_callback_and_zero_rhs_skips_minres(monkeypatch):
    with pytest.raises(ValueError, match="observation_hvp"):
        nonnegative_analysis_linear_operator(
            [1.0], _affine([[1.0]]), [1.0], [1.0], [1.0], [1.0], **_options()
        )
    linearization = nonnegative_analysis_linear_operator(
        [1.0], _affine([[1.0]]), [1.0], [1.0], [1.0], [1.0], curvature="gauss-newton", **_options()
    )
    monkeypatch.setattr(
        constraints, "minres", lambda *a, **k: pytest.fail("Zero RHS requires no solve")
    )
    np.testing.assert_array_equal(linearization.operator @ [0.0, 0.0], [0.0])
    np.testing.assert_array_equal(linearization.operator.rmatvec([0.0]), [0.0, 0.0])


def test_minres_success_flag_does_not_replace_the_true_residual_check(monkeypatch):
    linearization = nonnegative_analysis_linear_operator(
        [1.0], _affine([[1.0]]), [1.0], [1.0], [1.0], [1.0], curvature="gauss-newton", **_options()
    )
    monkeypatch.setattr(
        constraints, "minres", lambda operator, right, **kwargs: (np.zeros_like(right), 0)
    )
    with pytest.raises(RuntimeError, match="true relative linear residual"):
        linearization.operator @ [1.0, 0.0]


@pytest.mark.parametrize("extra", [np.array([np.nan]), np.array([0.0, 0.0])])
def test_exact_hessian_callback_must_return_a_finite_state_vector(extra):
    linearization = nonnegative_analysis_linear_operator(
        [1.0],
        _affine([[1.0]]),
        [1.0],
        [1.0],
        [1.0],
        [1.0],
        observation_hvp=lambda zeta, direction: extra.copy(),
        **_options(),
    )
    with pytest.raises(ValueError, match="Hessian product"):
        linearization.operator @ [1.0, 0.0]
