"""Положительность, порядок и производные конечномерного расщепления."""

from dataclasses import FrozenInstanceError
from math import factorial

import numpy as np
import pytest
from scipy import sparse
from scipy.integrate import quad, solve_ivp
from scipy.linalg import expm
from scipy.special import gammaln, logsumexp, xlogy
from scipy.stats import poisson

from adrkit.backends.bounded import BoundedResidual
from adrkit.time_integration import (
    PositiveAffineAction,
    poisson_degree,
    quadratic_decay,
    quadratic_decay_partials,
    solve_strang,
)


def generator():
    return sparse.csr_matrix([[-2.0, 2.0, 0.0], [1.0, -4.0, 3.0], [0.0, 2.0, -2.0]])


@pytest.mark.parametrize("distributed", [False, True])
def test_many_tangents_match_finite_differences_scalar_actions_and_pullback(distributed):
    rng = np.random.default_rng(501)
    count, intervals = 4, 5
    q = generator()
    partials = {"scale": q, "edge": sparse.csr_matrix([[-1, 1, 0], [0, 0, 0], [0, 0, 0]])}
    coefficients = rng.normal(scale=0.1, size=(len(partials), count))
    initial = np.array([0.3, 0.1, 0.4])
    forcing = np.full((intervals, 3) if distributed else (intervals,), 0.3)
    load = None if distributed else np.array([0.4, 0.0, 0.2])
    di = rng.normal(scale=0.1, size=(3, count))
    df = rng.normal(scale=0.1, size=(*forcing.shape, count))
    da, db = rng.normal(scale=0.1, size=count), rng.normal(scale=0.1, size=count)
    action = PositiveAffineAction(q, -0.2, step=0.1, rate=5, degree=14)
    trajectory = solve_strang(action, initial, forcing, beta=0.3, load=load)
    batch = trajectory.jvp_many(
        di, df, operator_partials=partials, operator_directions=coefficients, dalpha=da, dbeta=db
    )
    streamed = list(
        trajectory.iter_jvp_many(
            di,
            df,
            operator_partials=partials,
            operator_directions=coefficients,
            dalpha=da,
            dbeta=db,
        )
    )
    assert all(not value.flags.writeable for value in streamed)
    np.testing.assert_array_equal(np.stack(streamed), batch)
    cotangent = rng.normal(size=trajectory.states.shape)
    pulled = trajectory.vjp(cotangent, partials)
    step = 1e-5
    for j in range(count):
        dq = sum(coefficients[i, j] * matrix for i, matrix in enumerate(partials.values()))
        scalar = trajectory.jvp(di[:, j], df[..., j], dQ=dq, dalpha=da[j], dbeta=db[j])
        np.testing.assert_allclose(batch[..., j], scalar, rtol=3e-14, atol=3e-15)
        varied = [
            solve_strang(
                PositiveAffineAction(
                    q + sign * step * dq, -0.2 + sign * step * da[j], step=0.1, rate=5, degree=14
                ),
                initial + sign * step * di[:, j],
                forcing + sign * step * df[..., j],
                beta=0.3 + sign * step * db[j],
                load=load,
            ).states
            for sign in (1, -1)
        ]
        np.testing.assert_allclose(
            batch[..., j], (varied[0] - varied[1]) / (2 * step), rtol=3e-8, atol=3e-11
        )
        right = (
            di[:, j] @ pulled["initial"]
            + np.sum(df[..., j] * pulled["forcing"])
            + da[j] * pulled["alpha"]
            + db[j] * pulled["beta"]
            + sum(coefficients[i, j] * pulled["operator"][name] for i, name in enumerate(partials))
        )
        np.testing.assert_allclose(
            np.sum(cotangent * batch[..., j]), right, rtol=3e-13, atol=3e-14
        )
    np.testing.assert_array_equal(action.transport_weights, trajectory.affine.transport_weights)


def test_affine_many_tangent_shares_the_same_fixed_polynomial_action():
    q = generator()
    action = PositiveAffineAction(q, 0.2, step=0.1, rate=5, degree=12)
    u, f = np.array([0.3, 0.1, 0.4]), np.array([0.1, 0.2, 0.1])
    du, df = np.arange(9).reshape(3, 3) / 20, np.eye(3) / 10
    coefficients = np.array([[0.1, -0.2, 0.0]])
    da = np.array([0.2, -0.1, 0.0])
    actual = action.jvp_many(
        u, f, du, df, operator_partials={"scale": q}, operator_directions=coefficients, dalpha=da
    )
    expected = np.column_stack(
        [
            action.jvp(u, f, du[:, j], df[:, j], dQ=coefficients[0, j] * q, dalpha=da[j])
            for j in range(3)
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=3e-15)


def test_many_tangent_requires_declared_operator_direction_dimensions():
    action = PositiveAffineAction(generator(), 0.0, step=0.1, rate=5, degree=12)
    trajectory = solve_strang(action, [0.3, 0.1, 0.4], [0.3, 0.3], load=[1, 0, 0])
    with pytest.raises(ValueError, match="operator_directions"):
        list(trajectory.iter_jvp_many(np.zeros((3, 2)), operator_partials={"one": generator()}))
    with pytest.raises(ValueError, match="shape"):
        list(trajectory.iter_jvp_many(np.zeros((3, 2)), np.zeros((2, 3))))


@pytest.mark.parametrize("alpha", [-10.0, -0.7, 0.0, 0.8, 5.0, 6.0])
def test_integral_weights_and_alpha_derivatives_match_independent_quadrature(alpha):
    step, rate, degree = 0.2, 5.0, 8
    action = PositiveAffineAction(generator(), alpha, step=step, rate=rate, degree=degree)
    w = np.array(
        [
            quad(
                lambda s: np.exp((alpha - rate) * s) * (rate * s) ** k / factorial(k),
                0.0,
                step,
                epsabs=1e-15,
            )[0]
            for k in range(degree + 1)
        ]
    )
    dw = np.array(
        [
            quad(
                lambda s: s * np.exp((alpha - rate) * s) * (rate * s) ** k / factorial(k),
                0.0,
                step,
                epsabs=1e-15,
            )[0]
            for k in range(degree + 1)
        ]
    )
    ddw = np.array(
        [
            quad(
                lambda s: s**2 * np.exp((alpha - rate) * s) * (rate * s) ** k / factorial(k),
                0.0,
                step,
                epsabs=1e-15,
            )[0]
            for k in range(degree + 1)
        ]
    )
    total = quad(lambda s: np.exp(alpha * s), 0.0, step, epsabs=1e-15)[0]
    total_derivative = quad(lambda s: s * np.exp(alpha * s), 0.0, step, epsabs=1e-15)[0]
    total_second = quad(lambda s: s**2 * np.exp(alpha * s), 0.0, step, epsabs=1e-15)[0]
    expected = total * w / w.sum()
    derivative = (
        total_derivative * w / w.sum() + total * dw / w.sum() - total * w * dw.sum() / w.sum() ** 2
    )
    second = (
        (total_second * w + 2 * total_derivative * dw + total * ddw) / w.sum()
        - 2 * (total_derivative * w + total * dw) * dw.sum() / w.sum() ** 2
        - total * w * ddw.sum() / w.sum() ** 2
        + 2 * total * w * dw.sum() ** 2 / w.sum() ** 3
    )
    np.testing.assert_allclose(action.source_weights, expected, rtol=3e-14, atol=1e-17)
    np.testing.assert_allclose(action.source_alpha_derivative, derivative, rtol=5e-14, atol=1e-17)
    np.testing.assert_allclose(
        action.source_alpha_second_derivative, second, rtol=2e-13, atol=1e-17
    )
    assert np.all(action.source_weights >= 0) and np.all(action.source_alpha_derivative > 0)
    np.testing.assert_allclose(action.transport_weights.sum(), 1.0, rtol=0, atol=3e-16)
    np.testing.assert_allclose(action.source_weights.sum(), total, rtol=3e-15)
    np.testing.assert_allclose(action.source_alpha_derivative.sum(), total_derivative, rtol=3e-15)
    np.testing.assert_allclose(
        action.source_alpha_second_derivative.sum(), total_second, rtol=5e-15
    )


@pytest.mark.parametrize("alpha", [-10.0, 0.0, 5.0, 6.0])
def test_source_second_alpha_derivative_matches_centered_difference_of_first(alpha):
    options = {"step": 0.2, "rate": 5.0, "degree": 8}
    action = PositiveAffineAction(generator(), alpha, **options)
    displacement = 1e-4
    first_upper = PositiveAffineAction(
        generator(), alpha + displacement, **options
    ).source_alpha_derivative
    first_lower = PositiveAffineAction(
        generator(), alpha - displacement, **options
    ).source_alpha_derivative
    np.testing.assert_allclose(
        action.source_alpha_second_derivative,
        (first_upper - first_lower) / (2 * displacement),
        rtol=2e-8,
        atol=1e-13,
    )


@pytest.mark.parametrize("mean", [0.0, 0.2, 8.0, 1000.0])
def test_poisson_degree_uses_a_direct_tail_and_is_minimal_above_two(mean):
    tolerance = 1e-10
    degree = poisson_degree(mean, tolerance)
    assert degree >= 2 and 2 * poisson.sf(degree, mean) <= tolerance
    if degree > 2:
        assert 2 * poisson.sf(degree - 1, mean) > tolerance


def test_large_poisson_mean_keeps_positive_normalized_weights():
    rate, step = 1000.0, 1.0
    action = PositiveAffineAction(
        generator(), 0.1, step=step, rate=rate, degree=poisson_degree(rate * step, 1e-12)
    )
    assert np.isfinite(action.source_weights).all() and np.all(action.source_weights >= 0)
    np.testing.assert_allclose(action.transport_weights.sum(), 1.0, atol=2e-14, rtol=0)
    np.testing.assert_allclose(
        action.apply(np.ones(3), np.ones(3)), np.exp(0.1) + np.expm1(0.1) / 0.1, rtol=3e-13
    )
    second_total = quad(lambda s: s**2 * np.exp(0.1 * s), 0.0, step, epsabs=1e-14)[0]
    np.testing.assert_allclose(
        action.source_alpha_second_derivative.sum(), second_total, rtol=5e-13
    )
    displacement = 1e-4
    varied = [
        PositiveAffineAction(
            generator(), 0.1 + sign * displacement, step=step, rate=rate, degree=action.degree
        ).source_alpha_derivative
        for sign in (1, -1)
    ]
    np.testing.assert_allclose(
        action.source_alpha_second_derivative,
        (varied[0] - varied[1]) / (2 * displacement),
        rtol=3e-6,
        atol=2e-10,
    )


def test_large_mean_individual_weights_and_two_derivatives_match_log_integral_quadrature():
    rate, step, alpha = 1000.0, 1.0, 0.1
    degree = poisson_degree(rate * step, 1e-12)
    action = PositiveAffineAction(generator(), alpha, step=step, rate=rate, degree=degree)
    indices = np.array([0, degree // 3, degree // 2, int(rate * step), degree])
    powers = np.arange(degree + 1)
    log_factorials = gammaln(powers + 1)

    def moment(order, source_index=None):
        def integrand(time):
            if source_index is None:
                log_polynomial = logsumexp(xlogy(powers, rate * time) - log_factorials)
            else:
                log_polynomial = xlogy(source_index, rate * time) - gammaln(source_index + 1)
            return np.exp((alpha - rate) * time + log_polynomial + xlogy(order, time))

        points = None
        if source_index is not None:
            center = (source_index + order + 1) / (rate - alpha)
            if 0 < center < step:
                points = [center]
        return quad(integrand, 0.0, step, points=points, epsabs=1e-18, epsrel=3e-12, limit=250)[0]

    total = np.array(
        [
            quad(lambda time: time**order * np.exp(alpha * time), 0.0, step, epsabs=1e-14)[0]
            for order in range(3)
        ]
    )
    normalizer = np.array([moment(order) for order in range(3)])
    individual = np.array([[moment(order, int(k)) for k in indices] for order in range(3)])
    source = total[0] * individual[0] / normalizer[0]
    derivative = (total[1] * individual[0] + total[0] * individual[1]) / normalizer[0] - total[
        0
    ] * individual[0] * normalizer[1] / normalizer[0] ** 2
    second = (
        (total[2] * individual[0] + 2 * total[1] * individual[1] + total[0] * individual[2])
        / normalizer[0]
        - 2
        * (total[1] * individual[0] + total[0] * individual[1])
        * normalizer[1]
        / normalizer[0] ** 2
        - total[0] * individual[0] * normalizer[2] / normalizer[0] ** 2
        + 2 * total[0] * individual[0] * normalizer[1] ** 2 / normalizer[0] ** 3
    )
    np.testing.assert_allclose(action.source_weights[indices], source, rtol=5e-10, atol=5e-15)
    np.testing.assert_allclose(
        action.source_alpha_derivative[indices], derivative, rtol=5e-10, atol=5e-15
    )
    np.testing.assert_allclose(
        action.source_alpha_second_derivative[indices], second, rtol=5e-10, atol=5e-15
    )


def test_long_decay_interval_has_finite_source_integral_and_alpha_derivative():
    action = PositiveAffineAction(
        sparse.csr_matrix((1, 1)), -1.0, step=1e200, rate=1e-200, degree=2
    )
    # integral(exp(-s)) -> 1; integral(s*exp(-s)) -> 1 при step -> infinity.
    np.testing.assert_allclose(action.source_weight_sum, 1.0, rtol=1e-14)
    np.testing.assert_allclose(action.source_alpha_derivative.sum(), 1.0, rtol=1e-14)
    np.testing.assert_allclose(action.source_alpha_second_derivative.sum(), 2.0, rtol=1e-14)
    np.testing.assert_allclose(action.apply([2.0], [3.0]), [3.0], rtol=1e-14)
    np.testing.assert_allclose(action.jvp([2.0], [3.0], dalpha=1.0), [3.0], rtol=1e-14)


@pytest.mark.parametrize("alpha_step", [-1.0, 0.0, 1.0])
def test_zero_mean_branch_has_only_the_total_second_moment(alpha_step):
    step, rate = 1e-100, 1e-300
    action = PositiveAffineAction(
        sparse.csr_matrix((1, 1)), alpha_step / step, step=step, rate=rate, degree=2
    )
    expected = step**3 * quad(lambda t: t**2 * np.exp(alpha_step * t), 0.0, 1.0)[0]
    actual = action.source_alpha_second_derivative
    np.testing.assert_allclose(actual[0], expected, rtol=2e-13, atol=0)
    np.testing.assert_array_equal(actual[1:], [0.0, 0.0])
    assert not actual.flags.writeable


def test_unrepresentable_second_moment_does_not_change_first_order_range():
    step, rate = 1e110, 1e-110
    action = PositiveAffineAction(sparse.csr_matrix((1, 1)), 0.0, step=step, rate=rate, degree=2)
    np.testing.assert_allclose(action.source_weight_sum, step, rtol=1e-14)
    np.testing.assert_allclose(action.source_alpha_derivative.sum(), step**2 / 2, rtol=1e-13)
    np.testing.assert_allclose(action.apply([1.0], [1e-110]), [2.0], rtol=1e-13)
    np.testing.assert_allclose(action.jvp([1.0], [1e-110], dalpha=1e-110), [1.5], rtol=1e-13)
    with pytest.raises(FloatingPointError, match="second derivative.*representable"):
        action.source_alpha_second_derivative
    np.testing.assert_allclose(action.apply([1.0], [1e-110]), [2.0], rtol=1e-13)


def test_subnormal_gamma_probabilities_preserve_representable_source_weights():
    step, rate = 1e140, 1e-300
    mean = rate * step
    action = PositiveAffineAction(sparse.csr_matrix((1, 1)), 0.0, step=step, rate=rate, degree=2)
    powers = np.arange(3)
    # For mean << eps, exp(-rate*s)=1 to FP64 relative accuracy and the
    # normalized-polynomial tail is O(mean**3). The individual integrals
    # therefore have the leading values below. Logs avoid losing mean**2
    # before multiplication by the large, representable step.
    expected = np.exp(np.log(step) + powers * np.log(mean) - gammaln(powers + 2))
    expected_derivative = np.exp(
        2 * np.log(step) + powers * np.log(mean) - gammaln(powers + 1) - np.log(powers + 2)
    )
    np.testing.assert_allclose(action.source_weights, expected, rtol=5e-13, atol=0)
    np.testing.assert_allclose(
        action.source_alpha_derivative, expected_derivative, rtol=5e-13, atol=0
    )


@pytest.mark.parametrize("alpha", [-0.4, 0.0, 0.3])
def test_affine_action_matches_a_tiny_augmented_matrix_exponential(alpha):
    Q = generator()
    step, rate = 0.17, 5.0
    action = PositiveAffineAction(
        Q, alpha, step=step, rate=rate, degree=poisson_degree(rate * step, 1e-14)
    )
    initial, forcing = np.array([0.2, 0.4, 0.1]), np.array([0.3, 0.0, 0.2])
    augmented = np.zeros((4, 4))
    augmented[:3, :3] = Q.toarray() + alpha * np.eye(3)
    augmented[:3, 3] = forcing
    expected = expm(step * augmented) @ np.r_[initial, 1.0]
    np.testing.assert_allclose(
        action.apply(initial, forcing), expected[:3], rtol=3e-14, atol=1e-15
    )
    np.testing.assert_allclose(
        action.apply(np.ones(3), np.zeros(3)), np.exp(alpha * step), rtol=1e-15
    )


@pytest.mark.parametrize("degree", [2, 4, 8])
def test_normalized_operator_errors_obey_the_two_tail_bounds(degree):
    Q = generator()
    step, rate = 0.3, 5.0
    action = PositiveAffineAction(Q, 0.0, step=step, rate=rate, degree=degree)
    eye, zero = np.eye(3), np.zeros(3)
    transport = np.column_stack([action.apply(column, zero) for column in eye])
    source = np.column_stack([action.apply(zero, column) for column in eye])
    block = np.block([[Q.toarray(), eye], [np.zeros((3, 6))]])
    exact = expm(step * block)
    rounding = 32 * np.finfo(float).eps
    assert (
        np.linalg.norm(transport - exact[:3, :3], np.inf) <= 2 * action.transport_tail + rounding
    )
    assert (
        np.linalg.norm(source - exact[:3, 3:], np.inf)
        <= 2 * step * action.transport_tail + rounding
    )
    assert np.min(transport) >= 0 and np.min(source) >= 0


def test_quadratic_decay_partials_include_the_zero_beta_direction():
    initial = np.array([0.0, 0.4, 2.0])
    value = quadratic_decay(initial, 0.0, 0.3)
    du, db = quadratic_decay_partials(initial, 0.0, 0.3)
    np.testing.assert_array_equal(value, initial)
    np.testing.assert_array_equal(du, np.ones(3))
    np.testing.assert_allclose(db, -0.3 * initial**2, rtol=1e-15)
    h = 1e-5
    difference = (
        -3 * value + 4 * quadratic_decay(initial, h, 0.3) - quadratic_decay(initial, 2 * h, 0.3)
    ) / (2 * h)
    np.testing.assert_allclose(db, difference, rtol=3e-10, atol=2e-11)
    assert quadratic_decay(np.array([1e308]), 2.0, 2.0)[0] == pytest.approx(0.25, rel=1e-13)
    assert quadratic_decay(0.0, 0.4, 0.1) == 0.0
    assert quadratic_decay(2.0, 0.5, 1.0) == 1.0
    du, db = quadratic_decay_partials(2.0, 0.5, 1.0)
    assert du == 0.25 and db == -1.0


def test_full_composition_jvp_finite_difference_and_dual_identity():
    Q = generator()
    p1 = Q * 0.2
    p2 = sparse.csr_matrix([[-1.0, 1.0, 0.0], [0.0, -0.5, 0.5], [0.0, 0.0, 0.0]])
    direction_Q = 0.13 * p1 - 0.07 * p2
    alpha, beta, step, rate, degree = 0.3, 0.4, 0.12, 8.0, 12
    initial = np.array([0.2, 0.4, 0.1])
    load, controls = np.array([0.4, 0.0, 0.2]), np.array([0.3, 0.6, 0.2, 0.7])
    di, dc, da, db = np.array([0.1, -0.2, 0.1]), np.array([-0.1, 0.1, 0.2, -0.2]), 0.11, -0.08
    action = PositiveAffineAction(Q, alpha, step=step, rate=rate, degree=degree)
    trajectory = solve_strang(action, initial, controls, beta=beta, load=load)
    tangent = trajectory.jvp(di, dc, direction_Q, da, db)
    h = 1e-5
    varied = []
    for sign in (1.0, -1.0):
        operator = PositiveAffineAction(
            Q + sign * h * direction_Q, alpha + sign * h * da, step=step, rate=rate, degree=degree
        )
        varied.append(
            solve_strang(
                operator,
                initial + sign * h * di,
                controls + sign * h * dc,
                beta=beta + sign * h * db,
                load=load,
            ).states
        )
    np.testing.assert_allclose(tangent, (varied[0] - varied[1]) / (2 * h), rtol=2e-8, atol=3e-11)
    cotangent = np.random.default_rng(719).normal(size=trajectory.states.shape)
    pulled = trajectory.vjp(cotangent, {"one": p1, "two": p2})
    right = (
        di @ pulled["initial"]
        + dc @ pulled["forcing"]
        + da * pulled["alpha"]
        + db * pulled["beta"]
        + 0.13 * pulled["operator"]["one"]
        - 0.07 * pulled["operator"]["two"]
    )
    np.testing.assert_allclose(np.sum(cotangent * tangent), right, rtol=2e-14, atol=2e-14)


def test_distributed_forcing_and_zero_beta_pullback():
    action = PositiveAffineAction(generator(), 0.0, step=0.1, rate=5.0, degree=12)
    initial = np.array([0.3, 0.1, 0.4])
    forcing = np.array([[0.2, 0.1, 0.0], [0.3, 0.0, 0.1], [0.1, 0.4, 0.2]])
    trajectory = solve_strang(action, initial, forcing, beta=0.0)
    tangent = trajectory.jvp(dbeta=1.0)
    h = 1e-5
    finite_difference = (
        -3 * trajectory.states
        + 4 * solve_strang(action, initial, forcing, beta=h).states
        - solve_strang(action, initial, forcing, beta=2 * h).states
    ) / (2 * h)
    np.testing.assert_allclose(tangent, finite_difference, rtol=2e-8, atol=3e-11)
    cotangent = np.random.default_rng(41).normal(size=trajectory.states.shape)
    np.testing.assert_allclose(
        np.sum(cotangent * tangent), trajectory.vjp(cotangent)["beta"], rtol=2e-14
    )
    directions = np.full_like(forcing, -0.1)
    tangent = trajectory.jvp(d_forcing=directions)
    np.testing.assert_allclose(
        np.sum(cotangent * tangent),
        np.sum(directions * trajectory.vjp(cotangent)["forcing"]),
        rtol=2e-14,
    )


@pytest.mark.parametrize("beta", [0.0, 0.4])
def test_fixed_generator_second_order_with_exact_interval_averages(beta):
    Q = generator()
    initial, load, alpha = np.array([0.2, 0.4, 0.1]), np.array([0.4, 0.0, 0.2]), 0.2
    exact = solve_ivp(
        lambda t, u: Q @ u + alpha * u - beta * u * u + load * (1.0 + 0.3 * np.sin(t)),
        (0.0, 1.0),
        initial,
        method="DOP853",
        rtol=2e-13,
        atol=2e-14,
    ).y[:, -1]
    errors = []
    for intervals in (10, 20, 40):
        step = 1.0 / intervals
        times = np.linspace(0.0, 1.0, intervals + 1)
        means = 1.0 + 0.3 * (np.cos(times[:-1]) - np.cos(times[1:])) / step
        action = PositiveAffineAction(Q, alpha, step=step, rate=5.0, degree=20)
        trajectory = solve_strang(action, initial, means, beta=beta, load=load)
        errors.append(np.max(np.abs(trajectory.states[-1] - exact)))
    assert 3.8 < errors[0] / errors[1] < 4.2
    assert 3.8 < errors[1] / errors[2] < 4.2


def test_accumulated_polynomial_bound_applies_to_polynomial_error_only():
    initial, forcing = np.array([0.2, 0.4, 0.1]), np.full((12, 3), 0.2)
    low = solve_strang(
        PositiveAffineAction(generator(), 0.2, step=0.1, rate=5.0, degree=2),
        initial,
        forcing,
        beta=0.3,
    )
    high = solve_strang(
        PositiveAffineAction(generator(), 0.2, step=0.1, rate=5.0, degree=30),
        initial,
        forcing,
        beta=0.3,
    )
    assert np.max(np.abs(low.states[-1] - high.states[-1])) <= low.polynomial_error_bound
    assert 0 < low.polynomial_error_bound and high.polynomial_error_bound < 1e-25


def test_mass_requires_a_left_null_vector_and_not_just_zero_row_sums():
    Q = generator()
    weights = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(weights @ Q, np.zeros(3))
    action = PositiveAffineAction(Q, 0.0, step=0.2, rate=5.0, degree=12)
    initial = np.array([1.0, 0.0, 0.0])
    trajectory = solve_strang(action, initial, np.zeros((5, 3)))
    np.testing.assert_allclose(trajectory.states @ weights, weights @ initial, rtol=2e-15)
    assert abs(trajectory.states[-1].sum() - initial.sum()) > 0.1


@pytest.mark.parametrize("velocity", [(0.0, 0.0), (0.3, 0.0)])
def test_neumann_quadrature_conserves_diffusion_mass_but_not_normal_advection(velocity):
    grid = BoundedResidual(
        dict(
            diffusion=0.2,
            velocity=velocity,
            linear_loss=0.0,
            reaction_gamma=0.0,
            reaction_c_star=1.0,
            horizon=1.0,
            boundary_condition="neumann",
            source_position=None,
        ),
        interior_points=3,
        time_steps=3,
        domain=(0.0, 1.0, 0.0, 1.0),
    )
    Q = -grid.operator_at(0)
    weights = grid.spatial_quadrature_weights
    initial = 2.0 + np.cos(np.pi * grid.xy[:, 0])
    action = PositiveAffineAction(
        Q, 0.0, step=0.04, rate=np.nextafter(float(np.max(-Q.diagonal())), np.inf), degree=14
    )
    trajectory = solve_strang(action, initial, np.zeros((3, grid.size)))
    mass = trajectory.states @ weights
    if velocity[0] == 0:
        np.testing.assert_allclose(weights @ Q, 0.0, rtol=0, atol=1e-15)
        np.testing.assert_allclose(mass, mass[0], rtol=3e-15)
    else:
        assert np.max(np.abs(weights @ Q)) > 0.01
        assert abs(mass[-1] - mass[0]) > 0.01


def test_trajectory_owns_its_operator_and_forcing_point():
    Q = generator()
    action = PositiveAffineAction(Q, 0.2, step=0.1, rate=5.0, degree=12)
    initial, controls, load = np.ones(3), np.ones(3), np.array([0.3, 0.0, 0.1])
    trajectory = solve_strang(action, initial, controls, beta=0.2, load=load)
    expected = trajectory.jvp(dalpha=1.0)
    Q.data[:] = 0.0
    initial[:] = controls[:] = load[:] = 100.0
    np.testing.assert_array_equal(trajectory.jvp(dalpha=1.0), expected)
    with pytest.raises(FrozenInstanceError):
        action.alpha = 0.8
    with pytest.raises(ValueError):
        trajectory.states[0, 0] = -1.0
