"""Спуск, ограничения и локальная сходимость нелинейных наименьших квадратов."""

import numpy as np
import pytest
from scipy.optimize import LinearConstraint, OptimizeResult
from scipy.sparse import csr_array

from adrkit.contracts import MapCapabilities
from adrkit.errors import CapabilityError
from adrkit.inverse import optimizers
from adrkit.inverse.functionals import scalar_from_functions
from adrkit.maps import map_from_functions
from adrkit.spaces import ArraySpace


def space(size, axis):
    return ArraySpace((size,), axes=(axis,), coordinates=(tuple(range(size)),), units="1")


def linear_residual(matrix, target):
    return map_from_functions(
        domain=space(matrix.shape[1], "parameter"),
        codomain=space(matrix.shape[0], "residual"),
        predict=lambda x: matrix @ x - target,
        jvp=lambda x, p: matrix @ p,
    )


def test_coupled_box_step_is_not_a_clipped_unconstrained_step():
    matrix = np.array([[1.0, 1.0], [0.0, 1.0]])
    target = np.array([2.0, 0.0])
    residual = linear_residual(matrix, target)
    trajectory = []
    result = optimizers.solve_damped_gauss_newton(
        residual,
        [0.0, 0.0],
        bounds=[(0.0, 0.2), (0.0, None)],
        options={"gtol": 0.0, "maxiter": 1},
        callback=trajectory.append,
    )
    np.testing.assert_allclose(result.x, [0.2, 0.45], rtol=0, atol=3e-16)
    unconstrained = np.linalg.solve(matrix.T @ matrix + 2 * np.eye(2), matrix.T @ target)
    assert abs(result.x[1] - unconstrained[1]) > 0.08
    assert result.nit == 1 and result.status == 0 and not result.success
    assert result.step_size == 1.0
    assert result.fun < trajectory[0].fun
    assert result.directional_derivative <= -result.step_curvature + 1e-14
    np.testing.assert_allclose(
        result.jac, matrix.T @ (matrix @ result.x - target), rtol=0, atol=1e-15
    )


def test_descent_to_known_nonzero_residual_stationary_point():
    residual = linear_residual(np.ones((2, 1)), np.array([1.0, -1.0]))
    trajectory = []
    result = optimizers.solve_damped_gauss_newton(
        residual, [2.0], options={"gtol": 1e-10}, callback=trajectory.append
    )
    assert result.success and result.status == 2
    assert result.fun > 0.99
    np.testing.assert_allclose(result.x, 0.0, atol=5e-11, rtol=0)
    assert all(b.fun <= a.fun for a, b in zip(trajectory, trajectory[1:]))
    for point in trajectory[1:]:
        assert point.directional_derivative < 0.0
        assert point.directional_derivative + point.step_curvature <= point.qp_error_bound


def test_zero_residual_stops_without_evaluating_a_derivative():
    class Operation:
        def predict(self, x):
            return x - 2.0

        def jvp(self, x, p):
            raise AssertionError("A zero residual needs no Jacobian")

    layout = space(1, "x")
    operation = Operation()
    result = optimizers.solve_damped_gauss_newton(
        MapCapabilities(operation, layout, layout, operation), [2.0]
    )
    assert result.status == 1 and result.success
    assert result.nit == result.njev == result.njvp == 0
    np.testing.assert_array_equal(result.jac, [0.0])


def test_stationarity_of_a_constant_nonzero_residual_is_not_a_root():
    residual = map_from_functions(
        domain=space(1, "x"),
        codomain=space(1, "r"),
        predict=lambda x: np.ones(1),
        jvp=lambda x, p: np.zeros(1),
    )
    result = optimizers.solve_damped_gauss_newton(residual, [0.0], options={"gtol": 0.0})
    assert result.status == 2 and result.success and result.fun == 0.5
    assert "nonzero" in result.message
    assert result.nit == 0


def test_large_coordinate_origin_does_not_change_qp_optimality():
    origin = 1e8
    residual = map_from_functions(
        domain=space(1, "x"),
        codomain=space(1, "r"),
        predict=lambda x: x - (origin - 1.0),
        jvp=lambda x, p: p,
    )
    result = optimizers.solve_damped_gauss_newton(
        residual, [origin], options={"maxiter": 1, "gtol": 0.0}
    )
    assert result.status == 0 and result.nit == 1
    np.testing.assert_allclose(result.x, [origin - 2.0 / 3.0], rtol=0, atol=np.spacing(origin))
    assert result.fun < 0.056 and result.qp_optimality <= result.qp_error_bound


def test_an_unrepresentable_step_is_not_projected_stationarity():
    origin = 1e20
    residual = map_from_functions(
        domain=space(1, "x"),
        codomain=space(1, "r"),
        predict=lambda x: 1.0 + (x - origin),
        jvp=lambda x, p: p,
    )
    result = optimizers.solve_damped_gauss_newton(residual, [origin], options={"gtol": 0.0})
    assert result.status == -3 and not result.success
    assert result.optimality == 1.0 and result.fun == 0.5
    np.testing.assert_array_equal(result.x, [origin])


def test_known_nonlinear_root_has_eventual_full_quadratic_steps():
    residual = map_from_functions(
        domain=space(1, "x"),
        codomain=space(1, "r"),
        predict=lambda x: np.exp(x) - 2.0,
        jvp=lambda x, p: np.exp(x) * p,
    )
    trajectory = []
    result = optimizers.solve_damped_gauss_newton(
        residual, [0.0], options={"gtol": 1e-13, "maxiter": 12}, callback=trajectory.append
    )
    assert result.success
    np.testing.assert_allclose(result.x, np.log(2.0), atol=1e-13, rtol=0)
    errors = [abs(point.x[0] - np.log(2.0)) for point in trajectory]
    ratios = [b / a**2 for a, b in zip(errors, errors[1:]) if 1e-6 < a < 0.03]
    assert len(ratios) >= 2
    assert all(0.45 < value < 0.55 for value in ratios)
    assert all(point.step_size == 1.0 for point in trajectory[1:])


def test_armijo_reduces_an_overshooting_step_on_the_true_objective():
    residual = map_from_functions(
        domain=space(1, "x"),
        codomain=space(1, "r"),
        predict=lambda x: np.exp(5 * x) - 2.0,
        jvp=lambda x, p: 5 * np.exp(5 * x) * p,
    )
    trajectory = []
    result = optimizers.solve_damped_gauss_newton(
        residual, [-0.2], options={"maxiter": 1, "gtol": 0.0}, callback=trajectory.append
    )
    assert result.step_size < 1.0
    first, last = trajectory
    assert last.fun <= first.fun + 1e-4 * last.step_size * last.directional_derivative


def test_replicated_mean_square_residual_has_the_same_step_and_normalization():
    matrix = np.array([[1.0, 1.0], [0.0, 1.0], [2.0, -1.0]])
    target = np.array([2.0, 0.0, -1.0])
    scale = np.sqrt(2.0 / len(target))
    normalized = linear_residual(scale * matrix, scale * target)
    repeated = linear_residual(
        np.tile(scale * matrix / 2.0, (4, 1)), np.tile(scale * target / 2.0, 4)
    )
    settings = {"gtol": 0.0, "maxiter": 1}
    left = optimizers.solve_damped_gauss_newton(normalized, [0.0, 0.0], options=settings)
    right = optimizers.solve_damped_gauss_newton(repeated, [0.0, 0.0], options=settings)
    np.testing.assert_allclose(left.x, right.x, rtol=0, atol=5e-16)
    np.testing.assert_allclose(left.fun, np.mean((matrix @ left.x - target) ** 2), rtol=1e-15)
    np.testing.assert_allclose(
        left.jac, 2 * matrix.T @ (matrix @ left.x - target) / len(target), rtol=2e-14
    )
    np.testing.assert_allclose(left.fun, right.fun, rtol=1e-15)


def test_fixed_coordinates_and_multidimensional_domain():
    domain = ArraySpace((2, 1), axes=("row", "column"), coordinates=((0, 1), (0,)), units="1")
    residual = map_from_functions(
        domain=domain,
        codomain=space(2, "r"),
        predict=lambda x: x.ravel() - [2.0, 3.0],
        jvp=lambda x, p: p.ravel(),
    )
    result = optimizers.solve_damped_gauss_newton(
        residual, [[1.0], [0.0]], bounds=[(1.0, 1.0), (None, None)]
    )
    assert result.success and result.status == 2
    np.testing.assert_allclose(result.x, [1.0, 3.0], atol=1e-8, rtol=0)
    assert result.jac[0] == -1.0


def test_callback_receives_owned_results_and_can_stop():
    initial = np.array([0.0])
    residual = linear_residual(np.ones((1, 1)), np.ones(1))
    seen = []

    def callback(point):
        seen.append(point.fun)
        point.x[:] = point.jac[:] = point.residual[:] = 100.0
        if point.nit == 1:
            raise StopIteration

    result = optimizers.solve_damped_gauss_newton(residual, initial, callback=callback)
    assert result.status == 99 and not result.success and result.nit == 1
    np.testing.assert_allclose(result.x, [2.0 / 3.0], rtol=1e-15)
    np.testing.assert_array_equal(initial, [0.0])
    assert seen[1] < seen[0]


def test_success_flag_alone_cannot_accept_an_inaccurate_qp(monkeypatch):
    residual = linear_residual(np.eye(2), np.ones(2))
    monkeypatch.setattr(
        optimizers,
        "lsq_linear",
        lambda *a, **k: OptimizeResult(x=np.array([0.1, 0.2]), success=True),
    )
    result = optimizers.solve_damped_gauss_newton(residual, [0.0, 0.0], options={"gtol": 0.0})
    assert result.status == -1 and not result.success
    np.testing.assert_array_equal(result.x, [0.0, 0.0])
    assert result.qp_optimality > result.qp_error_bound


def test_exhausted_armijo_returns_the_last_accepted_point():
    residual = map_from_functions(
        domain=space(1, "x"),
        codomain=space(1, "r"),
        predict=lambda x: np.exp(5 * x) - 2.0,
        jvp=lambda x, p: 5 * np.exp(5 * x) * p,
    )
    result = optimizers.solve_damped_gauss_newton(residual, [-0.2], options={"max_backtracks": 0})
    assert result.status == -2 and not result.success and result.nit == 0
    np.testing.assert_array_equal(result.x, [-0.2])
    np.testing.assert_allclose(result.fun, 0.5 * (np.exp(-1) - 2) ** 2, rtol=1e-15)


def test_lbfgsb_passes_the_callback_to_scipy():
    objective = scalar_from_functions(
        domain=space(1, "x"),
        value=lambda x: 0.5 * float((x[0] - 1.0) ** 2),
        cotangent=lambda x: x - 1.0,
    )
    accepted = []

    def callback(intermediate_result):
        accepted.append(intermediate_result.x.copy())

    result = optimizers.solve_lbfgsb(objective, [3.0], callback=callback)
    assert result.success and len(accepted) >= 1
    np.testing.assert_allclose(accepted[-1], [1.0], rtol=0, atol=1e-14)


def test_invalid_initial_point_and_missing_jvp_fail_before_prediction():
    class Operation:
        def predict(self, x):
            raise AssertionError("Invalid inputs must fail before evaluation")

        def jvp(self, x, p):
            return p

    layout = space(1, "x")
    operation = Operation()
    with pytest.raises(ValueError, match="bounds"):
        optimizers.solve_damped_gauss_newton(
            MapCapabilities(operation, layout, layout, operation), [2.0], bounds=[(0.0, 1.0)]
        )
    with pytest.raises(CapabilityError, match="JVP"):
        optimizers.solve_damped_gauss_newton(MapCapabilities(operation, layout, layout), [0.0])


def test_default_damping_preserves_the_explicit_objective_rule():
    residual = linear_residual(np.array([[1.0, 2.0], [2.0, -1.0]]), np.ones(2))
    settings = {"gtol": 0.0, "maxiter": 3}
    default = optimizers.solve_damped_gauss_newton(residual, [0.0, 0.0], options=settings)
    explicit = optimizers.solve_damped_gauss_newton(
        residual, [0.0, 0.0], options={**settings, "damping_rule": "objective"}
    )
    np.testing.assert_array_equal(default.x, explicit.x)
    assert default.fun == explicit.fun and default.nfev == explicit.nfev
    assert default.damping == explicit.damping
    assert default.damping_rule == explicit.damping_rule == "objective"


def test_projected_gradient_damping_resolves_weak_curvature_with_nonzero_noise_floor():
    curvature, floor_residual = 2e-8, 0.02
    residual = linear_residual(
        np.array([[np.sqrt(curvature)], [0.0]]),
        np.array([np.sqrt(curvature), -floor_residual]),
    )
    settings = {"gtol": 1e-12, "maxiter": 20}
    fixed = optimizers.solve_damped_gauss_newton(residual, [0.0], options=settings)
    trajectory = []
    adaptive = optimizers.solve_damped_gauss_newton(
        residual,
        [0.0],
        options={**settings, "damping_rule": "projected-gradient"},
        callback=trajectory.append,
    )
    assert not fixed.success and fixed.x[0] < 0.01
    assert adaptive.success and adaptive.status == 2
    np.testing.assert_allclose(adaptive.x, [1.0], rtol=0, atol=1e-4)
    assert adaptive.fun == pytest.approx(0.5 * floor_residual**2, rel=1e-12)
    assert adaptive.fun == 0.5 * (adaptive.residual @ adaptive.residual)
    for previous, point in zip(trajectory, trajectory[1:]):
        assert point.damping == min(previous.fun, previous.optimality) > 0
        assert point.fun <= previous.fun + 1e-4 * point.step_size * point.directional_derivative
        assert point.directional_derivative < 0
        assert point.directional_derivative + point.step_curvature <= point.qp_error_bound


@pytest.mark.parametrize("rule", [None, True, 1, "unknown", []])
def test_invalid_damping_rule_fails_before_prediction(rule):
    def unexpected(point):
        raise AssertionError("The invalid option must fail before any model evaluation")

    residual = map_from_functions(
        domain=space(1, "x"),
        codomain=space(1, "r"),
        predict=unexpected,
        jvp=lambda x, p: p,
    )
    with pytest.raises(ValueError, match="damping_rule"):
        optimizers.solve_damped_gauss_newton(residual, [0.0], options={"damping_rule": rule})


def test_trust_constr_solves_coupled_inequality_without_infeasible_evaluations():
    def value(point):
        assert np.all(point >= 0.0) and np.all(point <= 1.0)
        assert point.sum() <= 1.0
        return 0.5 * np.sum((point - 2.0) ** 2)

    objective = scalar_from_functions(
        domain=space(2, "x"), value=value, cotangent=lambda x: x - 2.0
    )
    accepted = []

    def callback(intermediate_result):
        accepted.append(intermediate_result.x.copy())

    constraint = LinearConstraint([[1.0, 1.0]], -np.inf, 1.0, keep_feasible=True)
    result = optimizers.solve_trust_constr(
        objective,
        [0.2, 0.2],
        bounds=[(0.0, 1.0)] * 2,
        constraints=constraint,
        options={
            "gtol": 1e-10,
            "initial_barrier_parameter": 1e-8,
            "initial_barrier_tolerance": 1e-8,
        },
        callback=callback,
    )
    assert result.success and result.constr_violation == 0.0
    np.testing.assert_allclose(result.x, [0.5, 0.5], rtol=0, atol=1e-5)
    assert result.fun == pytest.approx(2.25, abs=2e-5)
    assert accepted and all(point.sum() <= 1.0 for point in accepted)
    np.testing.assert_array_equal(constraint.keep_feasible, [True])


def test_trust_constr_preserves_multidimensional_coordinates_and_combined_operation():
    domain = ArraySpace((2, 2), axes=("row", "column"), coordinates=((0, 1), (0, 1)), units="1")
    target = np.array([[2.0, 3.0], [4.0, 5.0]])

    def unexpected(point):
        raise AssertionError("The combined operation must be used")

    def combined(point):
        assert point.shape == domain.shape
        return 0.5 * np.sum((point - target) ** 2), point - target

    objective = scalar_from_functions(
        domain=domain,
        value=unexpected,
        cotangent=unexpected,
        value_and_cotangent=combined,
    )
    constraint = LinearConstraint(csr_array([[1.0, 2.0, 0.0, 0.0]]), 1.0, 1.0)
    initial = np.array([[0.2, 0.4], [0.0, 0.0]])
    result = optimizers.solve_trust_constr(
        objective, initial, constraints=[constraint], options={"gtol": 1e-10}
    )
    assert result.success
    np.testing.assert_allclose(result.x, [0.6, 0.2, 4.0, 5.0], atol=1e-9, rtol=0)
    np.testing.assert_allclose(result.grad, [-1.4, -2.8, 0.0, 0.0], atol=1e-9, rtol=0)
    np.testing.assert_array_equal(initial, [[0.2, 0.4], [0.0, 0.0]])


@pytest.mark.parametrize(
    "initial,bounds,constraints,exception,message",
    [
        ([0.0], None, object(), TypeError, "LinearConstraint"),
        ([0.0], None, [object()], TypeError, "LinearConstraint"),
        ([0.0], None, LinearConstraint([[1.0, 1.0]], 0.0, 1.0), ValueError, "columns"),
        ([0.0], [(1.0, 2.0)], (), ValueError, "Initial point violates bounds"),
        (
            [0.0],
            None,
            LinearConstraint([[1.0]], 1.0, np.inf, keep_feasible=True),
            ValueError,
            "keep_feasible",
        ),
    ],
)
def test_trust_constr_rejects_invalid_initial_setup_before_objective(
    initial, bounds, constraints, exception, message
):
    def unexpected(point):
        raise AssertionError("Invalid initial setup must fail before objective evaluation")

    objective = scalar_from_functions(domain=space(1, "x"), value=unexpected, cotangent=unexpected)
    with pytest.raises(exception, match=message):
        optimizers.solve_trust_constr(objective, initial, bounds=bounds, constraints=constraints)


def test_trust_constr_allows_an_infeasible_start_for_an_unprotected_constraint():
    objective = scalar_from_functions(
        domain=space(1, "x"),
        value=lambda x: 0.5 * (x[0] - 2.0) ** 2,
        cotangent=lambda x: x - 2.0,
    )
    constraint = LinearConstraint([[1.0]], 1.0, np.inf, keep_feasible=False)
    result = optimizers.solve_trust_constr(
        objective, [0.0], constraints=constraint, options={"gtol": 1e-10}
    )
    assert result.success
    np.testing.assert_allclose(result.x, [2.0], rtol=0, atol=1e-5)
    np.testing.assert_array_equal(constraint.keep_feasible, [False])
