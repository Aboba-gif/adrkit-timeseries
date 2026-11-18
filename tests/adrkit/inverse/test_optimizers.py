"""Спуск, ограничения и локальная сходимость нелинейных наименьших квадратов."""

from fractions import Fraction

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


def _rounded_constant_channel(*, declared_jacobian=True, ulps=1):
    """Known quadratic objective with controlled rounding in a constant channel.

    The exact residual is (scale*x, 1), with minimizer x=0. Only the computed
    constant channel changes by the requested ULP count after the first point;
    the derivative is that of the smooth exact residual, not artificial data noise.
    """
    origin, scale = 2.0**-12, 2.0**-16
    rounded = 1.0
    for _ in range(ulps):
        rounded = np.nextafter(rounded, np.inf)
    jacobian = np.array([[scale], [0.0]])
    residual = map_from_functions(
        domain=space(1, "parameter"),
        codomain=space(2, "residual"),
        predict=lambda x: np.array([scale * x[0], 1.0 if x[0] == origin else rounded]),
        jvp=lambda x, direction: jacobian @ direction,
        jacobian=(lambda x: jacobian) if declared_jacobian else None,
    )
    return residual, origin, scale


@pytest.mark.parametrize("declared_jacobian", [False, True])
def test_original_stationarity_at_rounded_trial_is_terminal_not_armijo(declared_jacobian):
    residual, origin, scale = _rounded_constant_channel(declared_jacobian=declared_jacobian)
    trajectory = []
    # With mu=scale**2*origin, the exact GN candidate is origin**2/(1+origin).
    tolerance = 1e-16
    result = optimizers.solve_damped_gauss_newton(
        residual,
        [origin],
        bounds=[(0.0, 1.0)],
        options={
            "gtol": tolerance,
            "qp_tolerance": 1e-24,
            "damping_rule": "projected-gradient",
            "max_backtracks": 0,
        },
        callback=trajectory.append,
    )
    assert result.success and result.status == 2 and result.optimality <= tolerance
    assert abs(result.x[0]) <= tolerance / scale**2
    assert abs(result.jac[0]) <= tolerance
    assert result.fun > trajectory[0].fun
    assert result.step_acceptance == "stationary_trial"
    assert "Armijo not certified" in result.step_acceptance_reason
    assert "Trial satisfies original projected/free" in result.message
    assert result.nit == 1 and result.nfev == 2 and result.njev == 2
    assert result.njvp == (0 if declared_jacobian else 2)
    assert [point.nit for point in trajectory] == [0, 1]
    assert trajectory[0].step_acceptance is None
    assert trajectory[1].step_acceptance == "stationary_trial"
    np.testing.assert_array_equal(trajectory[-1].x, result.x)
    np.testing.assert_array_equal(trajectory[-1].jac, result.jac)


def test_roundoff_screen_does_not_accept_a_nonstationary_trial():
    residual, origin, _ = _rounded_constant_channel()
    trajectory = []
    tolerance = 1e-25
    result = optimizers.solve_damped_gauss_newton(
        residual,
        [origin],
        bounds=[(0.0, 1.0)],
        options={
            "gtol": tolerance,
            "qp_tolerance": 1e-24,
            "damping_rule": "projected-gradient",
            "max_backtracks": 0,
        },
        callback=trajectory.append,
    )
    assert not result.success and result.status == -2
    assert result.optimality > tolerance and result.step_acceptance is None
    assert result.nit == 0 and result.njev == 2 and len(trajectory) == 1
    np.testing.assert_array_equal(result.x, [origin])


def test_stationary_trial_outside_the_dot_roundoff_envelope_is_not_accepted():
    residual, origin, _ = _rounded_constant_channel(ulps=16)
    result = optimizers.solve_damped_gauss_newton(
        residual,
        [origin],
        bounds=[(0.0, 1.0)],
        options={
            "gtol": 1e-16,
            "qp_tolerance": 1e-24,
            "damping_rule": "projected-gradient",
            "max_backtracks": 0,
        },
    )
    assert not result.success and result.status == -2
    assert result.step_acceptance is None and result.nit == 0
    # A trial derivative is not evaluated outside the declared arithmetic screen.
    assert result.njev == 1
    np.testing.assert_array_equal(result.x, [origin])


def test_dot_roundoff_bound_covers_exact_rational_half_squared_norm():
    values = np.array([1.0, 2.0**-27, 2.0**-26])
    computed = float(0.5 * (values @ values))
    exact = sum((Fraction.from_float(float(value)) ** 2 for value in values), Fraction()) / 2
    bound = optimizers._squared_norm_roundoff(values, computed)
    assert bound is not None and bound > 0
    assert abs(Fraction.from_float(computed) - exact) <= Fraction.from_float(bound)


def test_armijo_rejects_an_increase_hidden_by_a_constant_residual():
    def predict(point):
        x = point[0]
        return np.array([1.0 + x + 4.0 * x**2, 2.0**30])

    def jacobian(point):
        return np.array([[1.0 + 8.0 * point[0]], [0.0]])

    residual = map_from_functions(
        domain=space(1, "parameter"),
        codomain=space(2, "residual"),
        predict=predict,
        jacobian=jacobian,
    )
    # The projected-gradient damping is 1, so the full GN direction is -1/2.
    # At that point the variable residual grows from 1 to 3/2, while the
    # constant channel makes both computed scalar costs round to the same value.
    result = optimizers.solve_damped_gauss_newton(
        residual,
        [0.0],
        bounds=[(-1.0, 1.0)],
        options={"damping_rule": "projected-gradient", "max_backtracks": 0},
    )
    assert result.status == -2 and result.nit == 0
    assert result.step_acceptance is None
    np.testing.assert_array_equal(result.x, [0.0])


def test_armijo_resolves_cancellation_between_signed_residual_changes():
    unit = 2.0**-52
    values = np.array([1 + 2 * unit, 1 + 2 * unit, 1 + 3 * unit, unit])
    trial_values = np.array([1 + 3 * unit, 1 + 3 * unit, 1 + unit, 0.0])

    # Rounding the two sums 2 + 5*unit can reverse the sign of the factored dot.
    assert not optimizers._armijo_decrease(values, trial_values, 0.25, 1.0, -(unit**2))
    assert optimizers._armijo_decrease(trial_values, values, 0.25, 1.0, -(unit**2))


def test_armijo_requires_decrease_even_if_the_negative_threshold_underflows():
    values = np.ones(1)
    slope = -float(np.nextafter(0.0, 1.0))
    assert not optimizers._armijo_decrease(values, values, 0.25, 0.5, slope)


def test_roundoff_trial_near_a_bound_still_requires_the_interior_gradient():
    origin, floor = 2.0**-20, 2.0**28
    offset = 1.0 - 2.0**-8
    rounded_floor = np.nextafter(floor, np.inf)
    derivative_points = []

    def predict(point):
        value = point[0]
        return np.array(
            [
                offset + value + (value / origin) ** 2,
                floor if value == origin else rounded_floor,
            ]
        )

    def jacobian(point):
        derivative_points.append(float(point[0]))
        return np.array([[1.0 + 2 * point[0] / origin**2], [0.0]])

    residual = map_from_functions(
        domain=space(1, "parameter"),
        codomain=space(2, "residual"),
        predict=predict,
        jacobian=jacobian,
    )
    tolerance = 1e-8
    # At origin, a=r'=1+2/origin and r=2-2**-8+origin. The undamped GN
    # endpoint is q=origin*2**-8/(2+origin)>0. Since 0<=mu<=g=r*a,
    # damping puts the endpoint in [q, q+origin**2], wholly inside (0, gtol).
    # Nevertheless r(x)*r'(x)>=offset>gtol there: only the projected test passes.
    result = optimizers.solve_damped_gauss_newton(
        residual,
        [origin],
        bounds=[(0.0, 1.0)],
        options={
            "gtol": tolerance,
            "damping_rule": "projected-gradient",
            "max_backtracks": 0,
        },
    )
    assert len(derivative_points) == 2
    candidate = derivative_points[-1]
    candidate_gradient = (offset + candidate + (candidate / origin) ** 2) * (
        1.0 + 2 * candidate / origin**2
    )
    assert 0 < candidate < tolerance
    assert min(candidate_gradient, candidate) <= tolerance < candidate_gradient
    assert not result.success and result.status == -2 and result.nit == 0
    assert result.step_acceptance is None
    np.testing.assert_array_equal(result.x, [origin])


def test_roundoff_terminal_check_rejects_wrong_active_sign_in_a_narrow_box():
    width, slope, start_y = 2.0**-40, 2.0**26, 2.0**-52
    start = np.array([0.0, start_y])
    trial_points = []

    def predict(point):
        x, y = point
        constant = 1.0 if np.array_equal(point, start) else np.nextafter(1.0, np.inf)
        return np.array([1.0 - x + x**2 / width, slope * y, constant])

    def jacobian(point):
        trial_points.append(point.copy())
        return np.array([[-1.0 + 2 * point[0] / width, 0.0], [0.0, slope], [0.0, 0.0]])

    residual = map_from_functions(
        domain=space(2, "parameter"),
        codomain=space(3, "residual"),
        predict=predict,
        jacobian=jacobian,
    )
    tolerance = 1e-8
    result = optimizers.solve_damped_gauss_newton(
        residual,
        start,
        bounds=[(0.0, width), (None, None)],
        options={"gtol": tolerance, "damping_rule": "projected-gradient", "max_backtracks": 0},
    )
    assert len(trial_points) == 2
    trial = trial_points[-1]
    assert trial[0] == width < tolerance
    # The residual is 1 and its x derivative is +1 at the upper endpoint.
    # PG is capped by the narrow width; the original upper KKT sign is wrong.
    assert not result.success and result.status == -2 and result.nit == 0
    assert result.step_acceptance is None
    np.testing.assert_array_equal(result.x, start)


@pytest.mark.parametrize("subnormal_product", [True, False])
def test_relative_dot_screen_is_disabled_when_its_normal_arithmetic_model_fails(
    subnormal_product,
):
    root_tiny = np.sqrt(np.finfo(float).tiny)
    values = np.array([1.0, root_tiny / 2]) if subnormal_product else np.array([root_tiny])
    computed = float(0.5 * (values @ values))
    assert optimizers._squared_norm_roundoff(values, computed) is None


def test_declared_full_jacobian_avoids_scalar_jvp_and_keeps_the_same_problem():
    matrix, target = np.array([[1.0, 1.0], [0.0, 1.0]]), np.array([2.0, 0.0])
    fallback = linear_residual(matrix, target)

    def forbidden_jvp(point, direction):
        raise AssertionError("The declared Jacobian must replace scalar JVP calls")

    accelerated = map_from_functions(
        domain=fallback.domain,
        codomain=fallback.codomain,
        predict=lambda x: matrix @ x - target,
        jvp=forbidden_jvp,
        jacobian=lambda x: matrix,
    )
    options = {"gtol": 1e-8, "maxiter": 100}
    direct = optimizers.solve_damped_gauss_newton(
        accelerated, [0.2, 0.2], bounds=[(0, 1), (0, 1)], options=options
    )
    scalar = optimizers.solve_damped_gauss_newton(
        fallback, [0.2, 0.2], bounds=[(0, 1), (0, 1)], options=options
    )
    np.testing.assert_allclose(direct.x, scalar.x, rtol=0, atol=1e-13)
    np.testing.assert_allclose(direct.fun, scalar.fun, rtol=3e-14)
    assert direct.success and direct.njvp == 0 and direct.njev > 0
    assert scalar.njvp > 0


def test_bad_declared_jacobian_does_not_silently_fall_back_to_jvp():
    domain = space(2, "parameter")
    residual = map_from_functions(
        domain=domain,
        codomain=domain,
        predict=lambda x: x - 1,
        jvp=lambda x, h: h,
        jacobian=lambda x: np.ones((1, 2)),
    )
    with pytest.raises(ValueError, match="Jacobian"):
        optimizers.solve_damped_gauss_newton(residual, [0.0, 0.0])


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

    def squared_norm(point):
        values = residual.predict.predict(point.x)
        return sum(Fraction(float(value)) ** 2 for value in values)

    for current, trial in zip(trajectory, trajectory[1:]):
        if trial.step_acceptance == "stationary_trial":
            assert trial is trajectory[-1] and trial.optimality <= 1e-10
        else:
            assert squared_norm(trial) < squared_norm(current)
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
    # Replication changes floating-point summation and the LAPACK factorization.
    np.testing.assert_allclose(left.x, right.x, rtol=0, atol=8 * np.finfo(float).eps)
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


@pytest.mark.parametrize("side", ["lower", "upper"])
@pytest.mark.parametrize("offset_sign", [-1, 1])
def test_revalidated_bvls_active_bound_removes_signed_roundoff(monkeypatch, side, offset_sign):
    target = np.array([-1.0 if side == "lower" else 1.0, 1.0])
    residual = linear_residual(np.eye(2), target)
    bound = (0.0, None) if side == "lower" else (None, 0.0)
    active_code = -1 if side == "lower" else 1
    original = optimizers.lsq_linear
    returned = []

    def rounded_bound(*args, **kwargs):
        qp = original(*args, **kwargs)
        assert qp.success and qp.active_mask[0] == active_code
        # At x=0, mu=1: the exact QP solution is (0, 1/2).
        # A signed nonzero first coordinate either violates its bound or
        # incorrectly appears free with a QP gradient of magnitude near one.
        qp.x[0] = offset_sign * 2.0**-56
        returned.append(qp.x.copy())
        return qp

    monkeypatch.setattr(optimizers, "lsq_linear", rounded_bound)
    result = optimizers.solve_damped_gauss_newton(
        residual,
        [0.0, 0.0],
        bounds=[bound, (0.0, None)],
        options={"maxiter": 1, "gtol": 0.0},
    )
    assert len(returned) == 1 and returned[0][0] != 0.0
    assert result.status == 0 and result.nit == 1
    assert result.x[0] == 0.0
    np.testing.assert_allclose(result.x[1], 0.5, rtol=0, atol=2e-15)
    assert result.fun < 1.0 and result.step_size == 1.0
    assert result.qp_optimality <= result.qp_error_bound
    assert result.directional_derivative < 0


def test_original_valid_bvls_steps_ignore_misleading_masks_bitwise(monkeypatch):
    residual = linear_residual(np.eye(2), np.array([-1.0, 1.0]))
    bounds = [(0.0, None)] * 2
    options = {"maxiter": 8, "gtol": 1e-10, "damping_rule": "projected-gradient"}
    expected, actual = [], []
    reference = optimizers.solve_damped_gauss_newton(
        residual, [0.0, 0.0], bounds=bounds, options=options, callback=expected.append
    )
    original = optimizers.lsq_linear

    def misleading_mask(*args, **kwargs):
        qp = original(*args, **kwargs)
        # The second coordinate is free. This mask must never alter a step
        # which already satisfies the original feasibility/KKT/descent checks.
        qp.active_mask = np.full(qp.x.shape, -1.0)
        return qp

    monkeypatch.setattr(optimizers, "lsq_linear", misleading_mask)
    result = optimizers.solve_damped_gauss_newton(
        residual, [0.0, 0.0], bounds=bounds, options=options, callback=actual.append
    )
    assert result.success and reference.success
    assert len(expected) == len(actual)
    for before, after in zip(expected, actual):
        for name in ("x", "jac", "residual"):
            np.testing.assert_array_equal(before[name], after[name])
        for name in ("fun", "damping", "qp_optimality", "qp_error_bound", "step_size"):
            assert before[name] == after[name]
    assert (result.nit, result.nfev, result.njev, result.njvp) == (
        reference.nit,
        reference.nfev,
        reference.njev,
        reference.njvp,
    )


def test_bvls_mask_proposal_with_wrong_kkt_sign_is_rejected(monkeypatch):
    residual = linear_residual(np.eye(2), np.array([-1.0, 1.0]))
    original = optimizers.lsq_linear

    def wrong_face(*args, **kwargs):
        qp = original(*args, **kwargs)
        qp.x[0] = 2.0**-56
        qp.active_mask[0] = 1
        return qp

    monkeypatch.setattr(optimizers, "lsq_linear", wrong_face)
    result = optimizers.solve_damped_gauss_newton(
        residual, [0.0, 0.0], bounds=[(0.0, 2.0), (0.0, None)], options={"gtol": 0.0}
    )
    # The proposal p0=2 is feasible, but its QP gradient is +5 at an upper
    # bound, where KKT requires a nonpositive gradient. The mask is not proof.
    assert result.status == -1 and not result.success and result.nfev == 1
    assert "optimality or descent" in result.message
    assert result.qp_optimality > result.qp_error_bound
    np.testing.assert_array_equal(result.x, [0.0, 0.0])


def test_bvls_free_coordinate_outside_bounds_is_not_projected(monkeypatch):
    residual = linear_residual(np.eye(2), np.array([-1.0, 1.0]))
    original = optimizers.lsq_linear

    def infeasible_free_coordinate(*args, **kwargs):
        qp = original(*args, **kwargs)
        qp.x[1] = -(2.0**-56)
        qp.active_mask[1] = 0
        return qp

    monkeypatch.setattr(optimizers, "lsq_linear", infeasible_free_coordinate)
    result = optimizers.solve_damped_gauss_newton(
        residual, [0.0, 0.0], bounds=[(0.0, None)] * 2, options={"gtol": 0.0}
    )
    assert result.status == -1 and not result.success and result.nfev == 1
    assert "feasible converged step" in result.message
    assert result.qp_optimality is None
    np.testing.assert_array_equal(result.x, [0.0, 0.0])


@pytest.mark.parametrize(
    "mask",
    [
        [-1.0],
        [-1.0, 0.25],
        [-1.0, np.nan],
        [-1.0, np.inf],
        [True, False],
        ["-1", "0"],
        [1.0, 0.0],
    ],
)
def test_invalid_bvls_mask_or_infinite_active_bound_cannot_repair_step(monkeypatch, mask):
    residual = linear_residual(np.eye(2), np.array([-1.0, 1.0]))
    original = optimizers.lsq_linear

    def invalid_mask(*args, **kwargs):
        qp = original(*args, **kwargs)
        qp.x[0] = 2.0**-56
        qp.active_mask = np.asarray(mask)
        return qp

    monkeypatch.setattr(optimizers, "lsq_linear", invalid_mask)
    result = optimizers.solve_damped_gauss_newton(
        residual, [0.0, 0.0], bounds=[(0.0, None)] * 2, options={"gtol": 0.0}
    )
    assert result.status == -1 and not result.success and result.nfev == 1
    assert result.qp_optimality > result.qp_error_bound
    np.testing.assert_array_equal(result.x, [0.0, 0.0])


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


def test_cyclic_gauss_newton_reopens_source_after_coefficients_change():
    residual = linear_residual(np.array([[1.0, -1.0], [0.0, 1.0]]), np.array([-0.1, 1.0]))
    trajectory = []
    result = optimizers.solve_damped_gauss_newton(
        residual,
        [0.0, 0.0],
        bounds=((0.0, None), (0.0, 2.0)),
        blocks=((0,), (1,)),
        callback=trajectory.append,
    )
    assert trajectory[0].block_optimality[0] == 0
    assert trajectory[1].x[0] == 0 and trajectory[1].x[1] > 0
    assert trajectory[1].block_optimality[0] > 0
    assert trajectory[2].x[0] > 0 and trajectory[2].x[1] == trajectory[1].x[1]
    assert result.success
    np.testing.assert_allclose(result.x, [0.9, 1.0], rtol=0.0, atol=1e-7)
    assert max(result.block_optimality) == result.optimality <= 1e-8
    for previous, current in zip(trajectory, trajectory[1:]):
        assert np.count_nonzero(current.x != previous.x) == 1
        assert current.fun <= previous.fun


def test_cyclic_gauss_newton_reports_early_limit_on_the_current_pair():
    residual = linear_residual(np.array([[1.0, -1.0], [0.0, 1.0]]), np.array([-0.1, 1.0]))
    result = optimizers.solve_damped_gauss_newton(
        residual,
        [0.0, 0.0],
        bounds=((0.0, None), (0.0, 2.0)),
        blocks=((0,), (1,)),
        options={"maxiter": 1},
    )
    assert result.status == 0 and not result.success
    assert result.x[0] == 0 and result.block_optimality[0] > 0


@pytest.mark.parametrize(
    "blocks", [((0,),), ((0,), (0, 1)), ((0,), (2,)), ((0, False), (1,)), ((), (0, 1))]
)
def test_cyclic_gauss_newton_rejects_invalid_partitions(blocks):
    residual = linear_residual(np.eye(2), np.ones(2))
    with pytest.raises(ValueError, match="[Cc]oordinate blocks"):
        optimizers.solve_damped_gauss_newton(residual, [0.0, 0.0], blocks=blocks)


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
