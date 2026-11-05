"""Начальные данные, условия Неймана и производные разностной ADR."""

import numpy as np
import pytest
from scipy import sparse

from adrkit.backends.bounded import BoundedSolver
from adrkit.backends.cached import CachedSolver
from adrkit.coefficients import ADRUnits, ReactionLaw, ScalarField
from adrkit.errors import CapabilityError
from adrkit.pde import ADRProblem


def model(**changes):
    values = dict(
        horizon=0.2,
        diffusion=0.08,
        velocity=(0.2, -0.1),
        linear_loss=0.0,
        reaction_gamma=0.0,
        reaction_c_star=1.0,
        source_position=None,
        boundary_condition="neumann",
    )
    values.update(changes)
    return values


@pytest.mark.parametrize("factory", [BoundedSolver, CachedSolver])
def test_neumann_constant_state_and_spatial_weights(factory):
    solver = factory(
        model(initial_state=3.0, source_position=(0.5, 0.5)),
        interior_points=(3, 7),
        time_steps=4,
        domain=(0.0, 1.0, 0.0, 1.0),
    )
    assert (solver.nx, solver.ny) == (5, 9)
    np.testing.assert_allclose((solver.hx, solver.hy), (0.25, 0.125), rtol=0, atol=0)
    weights = solver.spatial_quadrature_weights
    np.testing.assert_allclose(weights.sum(), 1.0, rtol=0, atol=0)
    np.testing.assert_allclose(weights[0], solver.cell_area / 4, rtol=0, atol=0)
    np.testing.assert_allclose(weights[1], solver.cell_area / 2, rtol=0, atol=0)
    np.testing.assert_allclose(weights @ solver.load, 1.0, rtol=0, atol=0)
    forward = solver.solve_controls(np.zeros(solver.nt + 1))
    np.testing.assert_allclose(forward.states, 3.0, rtol=0, atol=5e-15)
    np.testing.assert_allclose(solver.operator @ np.ones(solver.size), 0.0, rtol=0, atol=4e-15)


def test_neumann_source_must_map_to_a_strictly_interior_node():
    with pytest.raises(ValueError, match="strictly interior"):
        BoundedSolver(
            model(source_position=(5e-11, 0.5)),
            interior_points=3,
            time_steps=1,
            domain=(0.0, 1.0, 0.0, 1.0),
        )


def test_neumann_cosine_mode_matches_discrete_backward_euler():
    diffusion = 0.12
    solver = BoundedSolver(
        model(
            diffusion=diffusion,
            velocity=(0.0, 0.0),
            initial_state=lambda xy, t: 2.0
            + 0.3 * np.cos(np.pi * xy[:, 0]) * np.cos(np.pi * xy[:, 1]),
        ),
        interior_points=(7, 3),
        time_steps=12,
        domain=(0.0, 1.0, 0.0, 1.0),
    )
    eigenvalue = (
        4 * np.sin(np.pi * solver.hx / 2) ** 2 / solver.hx**2
        + 4 * np.sin(np.pi * solver.hy / 2) ** 2 / solver.hy**2
    )
    amplitude = (1 + solver.dt * diffusion * eigenvalue) ** (-solver.nt)
    expected = 2.0 + (solver.initial_state - 2.0) * amplitude
    forward = solver.solve_forcing(np.zeros((solver.nt + 1, solver.size)))
    np.testing.assert_allclose(forward.states[-1], expected, rtol=0, atol=7e-15)


def test_neumann_diffusion_converges_to_continuous_cosine_mode():
    errors = []
    for interior, steps in ((3, 16), (7, 64), (15, 256)):
        solver = BoundedSolver(
            model(
                velocity=(0.0, 0.0),
                initial_state=lambda xy, t: np.cos(np.pi * xy[:, 0]) * np.cos(np.pi * xy[:, 1]),
            ),
            interior_points=interior,
            time_steps=steps,
            domain=(0.0, 1.0, 0.0, 1.0),
        )
        forward = solver.solve_forcing(np.zeros((solver.nt + 1, solver.size)))
        exact = solver.initial_state * np.exp(-2 * 0.08 * np.pi**2 * 0.2)
        errors.append(np.max(np.abs(forward.states[-1] - exact)))
    assert 3.7 < errors[0] / errors[1] < 4.1
    assert 3.9 < errors[1] / errors[2] < 4.1


def test_neumann_diffusion_mass_balance_with_point_source():
    solver = BoundedSolver(
        model(
            velocity=(0.0, 0.0),
            source_position=(0.5, 0.5),
            initial_state=lambda xy, t: 2.0 + np.cos(np.pi * xy[:, 0]),
        ),
        interior_points=7,
        time_steps=8,
        domain=(0.0, 1.0, 0.0, 1.0),
    )
    weights = solver.spatial_quadrature_weights
    np.testing.assert_allclose(weights @ solver.operator, 0.0, rtol=0, atol=3e-16)
    controls = np.linspace(0.5, 1.5, solver.nt + 1)
    mass = solver.solve_controls(controls).states @ weights
    expected = mass[0] + solver.dt * np.r_[0.0, np.cumsum(controls[1:])]
    np.testing.assert_allclose(mass, expected, rtol=0, atol=5e-15)


def test_neumann_normal_derivative_does_not_remove_advective_mass_flux():
    solver = BoundedSolver(
        model(velocity=(0.3, 0.0), initial_state=lambda xy, t: 2.0 + np.cos(np.pi * xy[:, 0])),
        interior_points=7,
        time_steps=1,
        domain=(0.0, 1.0, 0.0, 1.0),
    )
    weights = solver.spatial_quadrature_weights
    states = solver.solve_forcing(np.zeros((2, solver.size))).states
    change = weights @ (states[1] - states[0])
    np.testing.assert_allclose(
        change, -solver.dt * (weights @ (solver.operator @ states[1])), rtol=0, atol=2e-15
    )
    assert change > 0.05


@pytest.mark.parametrize("boundary", ["dirichlet", "neumann"])
@pytest.mark.parametrize("velocity", [(0.2, 0.1), (0.2, -0.1), (-0.2, 0.1), (-0.2, -0.1)])
def test_constant_transport_derivatives_in_each_velocity_sector(boundary, velocity):
    data = model(boundary_condition=boundary, velocity=velocity, linear_loss=0.03)
    options = dict(interior_points=(3, 5), time_steps=2, domain=(-1.0, 1.0, -1.0, 1.0))
    solver = BoundedSolver(data, **options)
    derivatives = solver.transport_derivatives()
    epsilon = 1e-6
    for parameter, derivative in derivatives.items():
        plus, minus = dict(data), dict(data)
        if parameter == "diffusion":
            plus["diffusion"] += epsilon
            minus["diffusion"] -= epsilon
        else:
            axis = 0 if parameter == "velocity_x" else 1
            direction = np.eye(2)[axis] * epsilon
            plus["velocity"] = np.array(velocity) + direction
            minus["velocity"] = np.array(velocity) - direction
        difference = (
            BoundedSolver(plus, **options).operator_at(0)
            - BoundedSolver(minus, **options).operator_at(0)
        ) / (2 * epsilon)
        np.testing.assert_allclose(
            derivative.toarray(), difference.toarray(), rtol=2e-10, atol=5e-10
        )


@pytest.mark.parametrize("boundary", ["dirichlet", "neumann"])
@pytest.mark.parametrize("sign", [-1, 1])
def test_zero_velocity_has_explicit_one_sided_derivatives(boundary, sign):
    data = model(boundary_condition=boundary, velocity=(0.0, 0.0))
    options = dict(interior_points=3, time_steps=2, domain=(0.0, 1.0, 0.0, 1.0))
    solver = BoundedSolver(data, **options)
    with pytest.raises(CapabilityError, match="one-sided"):
        solver.transport_derivatives()
    derivatives = solver.transport_derivatives(zero_velocity_signs=(sign, -sign))
    epsilon = 1e-6
    for axis, name, branch in ((0, "velocity_x", sign), (1, "velocity_y", -sign)):
        perturbed = dict(data, velocity=branch * epsilon * np.eye(2)[axis])
        difference = (
            BoundedSolver(perturbed, **options).operator_at(0) - solver.operator_at(0)
        ) / (branch * epsilon)
        np.testing.assert_allclose(
            derivatives[name].toarray(), difference.toarray(), rtol=0, atol=6e-10
        )


@pytest.mark.parametrize("boundary", ["dirichlet", "neumann"])
def test_source_and_forcing_derivatives_with_fixed_nonzero_initial_state(boundary):
    law = ReactionLaw(
        lambda u, xy, t: 0.02 * u * u - 0.03 * u, lambda u, xy, t: 0.04 * u - 0.03, pure=True
    )
    solver = BoundedSolver(
        model(
            boundary_condition=boundary,
            reaction=law,
            initial_state=lambda xy, t: 1.0 + 0.2 * np.cos(np.pi * xy[:, 0]),
            source_position=(0.5, 0.5),
        ),
        interior_points=3,
        time_steps=4,
        domain=(0.0, 1.0, 0.0, 1.0),
        residual_tolerance=1e-13,
    )
    controls = np.linspace(0.1, 0.2, solver.nt + 1)
    direction = np.linspace(-0.2, 0.1, solver.nt + 1)
    forward = solver.solve_controls(controls)
    tangent = solver.tangent_controls(forward, direction)
    epsilon = 1e-4
    difference = (
        solver.solve_controls(controls + epsilon * direction).states
        - solver.solve_controls(controls - epsilon * direction).states
    ) / (2 * epsilon)
    np.testing.assert_allclose(tangent, difference, rtol=1e-7, atol=2e-10)
    np.testing.assert_array_equal(tangent[0], np.zeros(solver.size))
    rng = np.random.default_rng(418)
    cotangent = rng.normal(size=forward.states.shape)
    np.testing.assert_allclose(
        np.sum(tangent * cotangent),
        direction @ solver.adjoint_controls(forward, cotangent),
        rtol=3e-14,
        atol=3e-14,
    )
    forcing = controls[:, None] * solver.load
    forcing_direction = rng.normal(size=forward.states.shape)
    forcing_tangent = solver.tangent_forcing(forward, forcing_direction)
    forcing_difference = (
        solver.solve_forcing(forcing + epsilon * forcing_direction).states
        - solver.solve_forcing(forcing - epsilon * forcing_direction).states
    ) / (2 * epsilon)
    np.testing.assert_allclose(forcing_tangent, forcing_difference, rtol=2e-7, atol=2e-10)
    np.testing.assert_allclose(
        np.sum(forcing_tangent * cotangent),
        np.sum(forcing_direction * solver.adjoint_forcing(forward, cotangent)),
        rtol=3e-14,
        atol=3e-14,
    )


def test_initial_field_is_sampled_once_at_physical_origin_and_owned_by_solver():
    calls = []
    values = np.arange(25.0) / 25

    def initial(points, time):
        calls.append((points.copy(), time))
        return values

    problem = ADRProblem.from_model(model(initial_state=ScalarField(initial), time_origin=7.0))
    solver = BoundedSolver(problem, interior_points=3, time_steps=2, domain=(0.0, 1.0, 0.0, 1.0))
    expected = values.copy()
    values[:] = -10.0
    np.testing.assert_array_equal(solver.initial_state, expected)
    snapshot = solver.snapshot()
    forcing = np.zeros((solver.nt + 1, solver.size))
    np.testing.assert_array_equal(
        snapshot.solve_forcing(forcing).states, solver.solve_forcing(forcing).states
    )
    assert len(calls) == 1
    assert calls[0][1] == 7.0
    np.testing.assert_array_equal(calls[0][0], solver.xy)
    exposed = solver.initial_state
    exposed.flags.writeable = True
    exposed[:] = 0.0
    np.testing.assert_array_equal(solver.initial_state, expected)


def test_initial_array_uses_xy_order_and_concentration_units():
    initial = np.arange(25.0)
    problem = ADRProblem.from_model(model(initial_state=initial))
    initial[:] = -1.0
    solver = BoundedSolver(problem, interior_points=3, time_steps=1, domain=(0.0, 1.0, 0.0, 1.0))
    np.testing.assert_array_equal(solver.initial_state, np.arange(25.0))
    with pytest.raises(ValueError, match="unit"):
        ADRProblem.from_model(
            model(
                units=ADRUnits("km", "h", "ug/m^3"),
                initial_state=ScalarField(lambda xy, t: 1.0, unit="ppm"),
            )
        )


@pytest.mark.parametrize(
    "name,value",
    [
        ("diffusion", ScalarField(lambda xy, t: 0.1)),
        ("velocity", lambda xy, t: (0.1, 0.2)),
        ("linear_loss", lambda xy, t: 0.01),
    ],
)
def test_neumann_variable_transport_is_explicitly_unsupported(name, value):
    with pytest.raises(CapabilityError, match="constant numeric"):
        BoundedSolver(model(**{name: value}), interior_points=3, time_steps=1)


def test_dirichlet_default_operator_retains_original_stencil():
    data = model(boundary_condition="dirichlet", velocity=(-0.3, 0.2), linear_loss=0.04)
    solver = BoundedSolver(data, interior_points=(3, 5), time_steps=3)

    def axis(v, n, h):
        d = data["diffusion"] / h**2
        return sparse.diags(
            [
                np.full(n - 1, -d - max(v, 0) / h),
                np.full(n, 2 * d + abs(v) / h),
                np.full(n - 1, -d + min(v, 0) / h),
            ],
            [-1, 0, 1],
            format="csc",
        )

    expected = sparse.kron(sparse.eye(solver.ny), axis(-0.3, solver.nx, solver.hx)) + sparse.kron(
        axis(0.2, solver.ny, solver.hy), sparse.eye(solver.nx)
    )
    expected += 0.04 * sparse.eye(solver.size)
    np.testing.assert_array_equal(solver.operator.toarray(), expected.toarray())
    without_setting = dict(data)
    without_setting.pop("boundary_condition")
    old_default = BoundedSolver(without_setting, interior_points=(3, 5), time_steps=3)
    np.testing.assert_array_equal(
        solver.solve_forcing(np.ones((4, solver.size))).states,
        old_default.solve_forcing(np.ones((4, solver.size))).states,
    )


def test_newton_initial_guess_is_independent_of_internal_arrays():
    def initial_guess(rhs, previous, step):
        result = np.full_like(rhs, np.max(np.abs(rhs)) + 1.0)
        rhs[:] = previous[:] = -100.0
        return result

    alpha = 0.03
    law = ReactionLaw(
        lambda u, xy, t: 0.02 * u * u - alpha * u, lambda u, xy, t: 0.04 * u - alpha, pure=True
    )
    data = model(initial_state=2.0, reaction=law)
    options = dict(interior_points=3, time_steps=3, domain=(0.0, 1.0, 0.0, 1.0))
    solver = BoundedSolver(data, initial_guess=initial_guess, residual_tolerance=1e-13, **options)
    forcing = np.ones((solver.nt + 1, solver.size))
    reference = BoundedSolver(data, residual_tolerance=1e-13, **options).solve_forcing(forcing)
    result = solver.solve_forcing(forcing)
    assert max(result.max_scaled_residual, reference.max_scaled_residual) <= 1e-13
    assert np.min(result.states) >= 0 and np.min(reference.states) >= 0
    error_bound = 0.0
    b0 = 1.0 - alpha * solver.dt
    for n in range(solver.nt):
        scale_sum = (
            2.0
            + np.linalg.norm(result.states[n] + solver.dt * forcing[n + 1], np.inf)
            + np.linalg.norm(reference.states[n] + solver.dt * forcing[n + 1], np.inf)
        )
        # Для X,Y >= 0 обратная норма A+dt*beta*diag(X+Y) не больше 1/b0.
        error_bound = (error_bound + solver.residual_tolerance * scale_sum) / b0
        roundoff = 32 * np.finfo(float).eps * np.max(np.abs(reference.states[n + 1]))
        np.testing.assert_allclose(
            result.states[n + 1], reference.states[n + 1], rtol=0, atol=error_bound + roundoff
        )
    np.testing.assert_array_equal(solver.snapshot().solve_forcing(forcing).states, result.states)
