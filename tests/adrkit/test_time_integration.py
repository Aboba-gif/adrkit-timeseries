"""Положительность, порядок и производные конечномерного расщепления."""

from dataclasses import FrozenInstanceError
from math import factorial

import numpy as np
import pytest
from scipy import sparse
from scipy.integrate import quad, solve_ivp
from scipy.linalg import expm
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


@pytest.mark.parametrize("alpha", [-0.7, 0.0, 0.8, 5.0, 6.0])
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
    total = quad(lambda s: np.exp(alpha * s), 0.0, step, epsabs=1e-15)[0]
    total_derivative = quad(lambda s: s * np.exp(alpha * s), 0.0, step, epsabs=1e-15)[0]
    expected = total * w / w.sum()
    derivative = (
        total_derivative * w / w.sum() + total * dw / w.sum() - total * w * dw.sum() / w.sum() ** 2
    )
    np.testing.assert_allclose(action.source_weights, expected, rtol=3e-14, atol=1e-17)
    np.testing.assert_allclose(action.source_alpha_derivative, derivative, rtol=5e-14, atol=1e-17)
    assert np.all(action.source_weights >= 0) and np.all(action.source_alpha_derivative > 0)
    np.testing.assert_allclose(action.transport_weights.sum(), 1.0, rtol=0, atol=3e-16)
    np.testing.assert_allclose(action.source_weights.sum(), total, rtol=3e-15)
    np.testing.assert_allclose(action.source_alpha_derivative.sum(), total_derivative, rtol=3e-15)


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


def test_long_decay_interval_has_finite_source_integral_and_alpha_derivative():
    action = PositiveAffineAction(
        sparse.csr_matrix((1, 1)), -1.0, step=1e200, rate=1e-200, degree=2
    )
    # integral(exp(-s)) -> 1; integral(s*exp(-s)) -> 1 при step -> infinity.
    np.testing.assert_allclose(action.source_weight_sum, 1.0, rtol=1e-14)
    np.testing.assert_allclose(action.source_alpha_derivative.sum(), 1.0, rtol=1e-14)
    np.testing.assert_allclose(action.apply([2.0], [3.0]), [3.0], rtol=1e-14)
    np.testing.assert_allclose(action.jvp([2.0], [3.0], dalpha=1.0), [3.0], rtol=1e-14)


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
