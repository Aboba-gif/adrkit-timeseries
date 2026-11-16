"""Nonnegative nonlinear analysis against known variational problems."""

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from adrkit.errors import CapabilityError
from adrkit.inverse import constraints
from adrkit.maps import map_from_functions
from adrkit.spaces import ArraySpace


def _space(size, name):
    return ArraySpace((size,), axes=(name,), coordinates=(tuple(range(size)),), units="1")


def _linear(matrix, *, vjp_calls=None):
    matrix = np.asarray(matrix, dtype=float)

    def vjp(point, cotangent):
        if vjp_calls is not None:
            vjp_calls.append(cotangent.copy())
        return matrix.T @ cotangent

    def forbidden_jacobian(point):
        raise AssertionError("The nonlinear analysis must use only declared rowwise VJPs")

    return map_from_functions(
        domain=_space(matrix.shape[1], "state"),
        codomain=_space(matrix.shape[0], "data"),
        predict=lambda point: matrix @ point,
        vjp=vjp,
        jacobian=forbidden_jacobian,
    )


def _scalar(predict, derivative):
    return map_from_functions(
        domain=_space(1, "state"),
        codomain=_space(1, "data"),
        predict=lambda point: np.array([predict(point[0])]),
        vjp=lambda point, cotangent: np.array([derivative(point[0]) * cotangent[0]]),
    )


def test_linear_limit_matches_native_analysis_with_correlated_errors_and_thin_vjp(monkeypatch):
    matrix = np.array([[1, 0, 0.5, -0.2, 0.3], [0.2, 1, 0, 0.6, -0.1]])
    reference = np.array([0.2, -0.1, 0.5, 0.0, 0.3])
    mass = np.array([1, 2, 3, 0.5, 1.5])
    covariance = np.array([[0.4, 0.1], [0.1, 0.8]])
    target = np.array([0.8, -0.2])
    expected = constraints.analyze_nonnegative_observations(
        reference, matrix, target, mass, covariance
    )
    calls = []
    mapping = _linear(matrix, vjp_calls=calls)
    original_eye = np.eye

    def no_state_identity(size, *args, **kwargs):
        assert size != len(reference), "No dense state-size prior or identity is needed"
        return original_eye(size, *args, **kwargs)

    monkeypatch.setattr(constraints.np, "eye", no_state_identity)
    result = constraints.analyze_nonnegative_map(
        reference, mapping, target, mass, covariance, x0=np.ones(5)
    )
    assert result.success
    assert result.nit == 1
    np.testing.assert_allclose(result.x, expected.x, atol=2e-9)
    np.testing.assert_allclose(result.fun, expected.fun, atol=1e-11)
    assert result.nvjp == 2 * result.njev == len(calls)
    for cotangent in calls:
        assert np.count_nonzero(cotangent) == 1 and np.max(cotangent) == 1
    assert np.all(result.x >= 0)
    assert result.diagnostics["projected_gradient_inf"] <= 1e-8


def test_signed_target_and_boundary_are_solved_without_clipping_a_free_solution():
    result = constraints.analyze_nonnegative_map([0.3], _linear([[1]]), [-2], [1], [1], x0=[2])
    assert result.success
    np.testing.assert_array_equal(result.x, [0])
    assert result.jac[0] > 0
    assert result.diagnostics["projected_gradient_inf"] == 0


def test_nonlinear_saturating_map_recovers_known_unique_minimum_with_signed_adjusted_target(
    monkeypatch,
):
    # For A=x/(1+x), r=.2 and this d, Phi'(.3)=0. On x>=0,
    # Phi''>=1-27/128>0, so this stationary point is the unique minimum.
    reference, minimizer = 0.2, 0.3
    predict = lambda x: x / (1 + x)
    derivative = lambda x: 1 / (1 + x) ** 2
    target = predict(minimizer) + (minimizer - reference) / derivative(minimizer)
    assert 0 <= target <= 1
    adjusted = []
    native = constraints.analyze_nonnegative_observations

    def record_adjusted(reference, matrix, data, mass, covariance, **options):
        adjusted.append(data.copy())
        return native(reference, matrix, data, mass, covariance, **options)

    monkeypatch.setattr(constraints, "analyze_nonnegative_observations", record_adjusted)
    result = constraints.analyze_nonnegative_map(
        [reference], _scalar(predict, derivative), [target], [1], [1], x0=[5]
    )
    assert result.success
    assert adjusted[0][0] < 0
    np.testing.assert_allclose(result.x, [minimizer], atol=2e-8)
    gradient = result.x[0] - reference + (predict(result.x[0]) - target) * derivative(result.x[0])
    np.testing.assert_allclose(result.jac, [gradient], atol=1e-14)
    assert abs(gradient) <= 1e-8
    assert np.all(np.diff(result.objective_history) < 0)


def test_armijo_rejects_an_increasing_full_gn_step_on_the_original_objective():
    result = constraints.analyze_nonnegative_map(
        [0],
        _scalar(lambda x: x * x, lambda x: 2 * x),
        [1],
        [0.1],
        [1],
        x0=[0.1],
        maxiter=1,
    )
    assert not result.success and result.status == 1
    assert 0 < result.diagnostics["last_step_size"] < 1
    assert result.nfev >= 3
    assert len(result.objective_history) == 2
    assert result.objective_history[1] < result.objective_history[0]


def test_stationarity_at_a_nonglobal_point_is_not_reported_as_global_optimality():
    mapping = _scalar(lambda x: x * x, lambda x: 2 * x)
    result = constraints.analyze_nonnegative_map([0], mapping, [1], [1], [1], x0=[0])
    assert result.success and result.nit == 0
    assert result.diagnostics["projected_gradient_inf"] == 0
    better = 1 / np.sqrt(2)
    better_value = 0.5 * better**2 + 0.5 * (better**2 - 1) ** 2
    assert better_value < result.fun
    assert "stationarity" in result.message and "not a global minimum" in result.message


def test_inner_analysis_failure_is_not_original_stationarity(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("Inner tolerance not reached")

    monkeypatch.setattr(constraints, "analyze_nonnegative_observations", fail)
    result = constraints.analyze_nonnegative_map([0], _linear([[1]]), [1], [1], [1], x0=[1])
    assert not result.success and result.status == 2
    assert result.diagnostics["projected_gradient_inf"] > 1e-8


def test_negative_first_order_derivative_alone_does_not_pass_the_model_change_check(monkeypatch):
    # Phi=(x-3)^2 at x=5: endpoint=0 has g*p=-20 but actual quadratic change=+5.
    monkeypatch.setattr(
        constraints,
        "analyze_nonnegative_observations",
        lambda *args, **kwargs: OptimizeResult(x=np.array([0.0]), success=True, nit=0),
    )
    result = constraints.analyze_nonnegative_map([3], _linear([[1]]), [3], [1], [1], x0=[5])
    assert not result.success and result.status == 3
    assert result.diagnostics["last_directional_derivative"] < 0
    assert result.diagnostics["last_model_change"] > result.diagnostics["last_model_allowance"]
    np.testing.assert_array_equal(result.x, [5])


def test_non_descent_endpoint_cannot_trigger_a_fallback_direction(monkeypatch):
    monkeypatch.setattr(
        constraints,
        "analyze_nonnegative_observations",
        lambda *args, **kwargs: OptimizeResult(x=np.array([2.0]), success=True, nit=0),
    )
    result = constraints.analyze_nonnegative_map([0], _linear([[1]]), [0], [1], [1], x0=[1])
    assert not result.success and result.status == 3
    np.testing.assert_array_equal(result.x, [1])
    assert len(result.objective_history) == 1


def test_projected_gradient_does_not_cancel_when_state_is_much_larger_than_gradient():
    result = constraints.analyze_nonnegative_map(
        [1e10],
        _linear([[1e-10]]),
        [0],
        [1],
        [1],
        x0=[1e10],
        kkt_tolerance=1e-12,
        maxiter=1,
    )
    assert not result.success
    assert result.diagnostics["projected_gradient_inf"] > 1e-12


def test_missing_declared_vjp_and_nonvector_spaces_are_rejected():
    mapping = map_from_functions(
        domain=_space(1, "state"), codomain=_space(1, "data"), predict=lambda x: x
    )
    with pytest.raises(CapabilityError, match="VJP"):
        constraints.analyze_nonnegative_map([0], mapping, [1], [1], [1])
    grid = ArraySpace((1, 1), axes=("x", "y"), coordinates=((0,), (0,)), units="1")
    mapping = map_from_functions(
        domain=grid,
        codomain=_space(1, "data"),
        predict=lambda x: x.ravel(),
        vjp=lambda x, cotangent: cotangent.reshape(1, 1),
    )
    with pytest.raises(ValueError, match="vector"):
        constraints.analyze_nonnegative_map([[0]], mapping, [1], [[1]], [1])


@pytest.mark.parametrize(
    "options",
    [
        {"mass": [0]},
        {"covariance": [-1]},
        {"x0": [-1]},
        {"model_change_rtol": -1},
        {"model_change_rtol": 1},
        {"linear_options": {"other": 1}},
        {"linear_options": {"kkt_tolerance": 0}},
    ],
)
def test_invalid_metrics_and_solver_options_are_not_repaired(options):
    arguments = dict(
        reference=[0], observation=_linear([[1]]), target=[1], mass=[1], covariance=[1]
    )
    arguments.update(options)
    with pytest.raises(ValueError):
        constraints.analyze_nonnegative_map(**arguments)
