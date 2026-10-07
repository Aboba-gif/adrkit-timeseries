"""Внешние решатели состояния и совместимость адаптеров решателя в ограниченной области."""
from dataclasses import dataclass
import threading

import numpy as np
import pytest

from adrkit.backends.bounded import BoundedSolver
from adrkit.coefficients import ADRUnits
from adrkit.errors import CapabilityError
from adrkit.pde import ADRProblem
from adrkit.spaces import ArraySpace
from adrkit.state_maps import forcing_state_map, state_map, state_map_from_solver


def spaces():
    domain = ArraySpace((2,), axes=("parameter",), coordinates=(("a", "b"),), units="input")
    codomain = ArraySpace((3,), axes=("state",), coordinates=(("u", "v", "w"),), units="state")
    return domain, codomain


def external_solver():
    r"""Создать нелинейный внешний решатель с векторным состоянием.

    Returns
    -------
    mapping : MapCapabilities
        Отображение из двух параметров в три значения состояния.
    returned : list of dict
        Траектории, переданные адаптеру для проверки владения массивами.
    calls : list of ndarray
        Копии аргументов прямого решателя.

    Notes
    -----
    Для фиксированной матрицы :math:`A` решатель возвращает
    :math:`T(x)=\sin(Ax)` и сохраняет :math:`u=Ax` в траектории.
    Действия производных равны

    .. math::

        J(x)h=\cos(u)\odot Ah,\qquad
        J(x)^\mathsf{T}z=A^\mathsf{T}(\cos(u)\odot z).
    """
    matrix = np.array([[1., 2.], [-.5, .3], [2., -.7]])
    returned, calls = [], []

    def solve(x):
        calls.append(x.copy())
        u = matrix @ x
        trajectory = {"u": u, "prediction": np.sin(u)}
        returned.append(trajectory)
        return trajectory

    def tangent(trajectory, h):
        return np.cos(trajectory["u"])*(matrix @ h)

    def pullback(trajectory, z):
        return matrix.T @ (np.cos(trajectory["u"])*z)

    domain, codomain = spaces()
    mapping = state_map_from_solver(domain=domain, codomain=codomain,
        solve=solve, jvp=tangent, vjp=pullback, states=lambda t: t["prediction"])
    return mapping, returned, calls


def test_vector_state_fd_duality_and_fixed_linear_operator():
    mapping, _, calls = external_solver()
    assert not calls
    at, h = np.array([.2, -.1]), np.array([.7, .3])
    z = np.array([.1, .2, -.4])
    expected = mapping.predict.predict(at)
    jh, gz = mapping.jvp.jvp(at, h), mapping.vjp.vjp(at, z)
    assert len(calls) == 1
    eps = 1e-6
    numerical = (mapping.predict.predict(at+eps*h)-mapping.predict.predict(at-eps*h))/(2*eps)
    np.testing.assert_allclose(jh, numerical, rtol=2e-9, atol=2e-10)
    np.testing.assert_allclose(jh @ z, h @ gz, rtol=1e-13, atol=1e-14)
    linear = mapping.predict.linear_operator(at)
    mapping.predict.predict(at*3)
    np.testing.assert_array_equal(linear.matvec(h), jh)
    np.testing.assert_array_equal(linear.rmatvec(z), gz)
    assert expected.shape == (3,)


def test_exact_at_cache_and_trajectory_prediction_ownership():
    mapping, returned, calls = external_solver()
    at, h = np.array([.2, -.1]), np.array([.7, .3])
    prediction = mapping.predict.predict(at)
    expected = prediction.copy()
    tangent = mapping.jvp.jvp(at, h)
    linear = mapping.predict.linear_operator(at)
    returned[0]["u"][:] = 300.
    returned[0]["prediction"][:] = 400.
    prediction[:] = -100.
    np.testing.assert_array_equal(mapping.predict.predict(at), expected)
    np.testing.assert_array_equal(mapping.jvp.jvp(at, h), tangent)
    assert len(calls) == 1
    mapping.predict.predict(np.nextafter(at, np.inf))
    assert len(calls) == 2
    at[:] = 99.
    mapping.predict.predict(at)
    np.testing.assert_array_equal(linear.matvec(h), tangent)


@pytest.mark.parametrize("name", ["solve", "states", "copy_trajectory", "jvp", "vjp"])
def test_noncallable_callback_rejected_without_evaluation(name):
    domain, codomain = spaces()

    def forbidden(*args):
        raise AssertionError("construction must not execute a callback")

    kwargs = dict(domain=domain, codomain=codomain, solve=forbidden,
                  jvp=forbidden, vjp=forbidden, states=forbidden, copy_trajectory=forbidden)
    kwargs[name] = "not executable"
    with pytest.raises(TypeError, match=name):
        state_map_from_solver(**kwargs)


@pytest.mark.parametrize("which", ["domain", "codomain"])
def test_spaces_must_be_declared_before_evaluation(which):
    domain, codomain = spaces()
    kwargs = dict(domain=domain, codomain=codomain, solve=lambda x: x)
    kwargs[which] = None
    with pytest.raises(TypeError, match="ArraySpace"):
        state_map_from_solver(**kwargs)


def test_missing_derivatives_fail_before_state_evaluation():
    domain, codomain = spaces()

    def forbidden(x):
        raise AssertionError("missing capabilities must fail before solve")

    mapping = state_map_from_solver(domain=domain, codomain=codomain, solve=forbidden)
    assert mapping.jvp is None and mapping.vjp is None
    with pytest.raises(CapabilityError, match="JVP"):
        mapping.predict.jvp(np.ones(2), np.ones(2))
    with pytest.raises(CapabilityError, match="VJP"):
        mapping.predict.vjp(np.ones(2), np.ones(3))
    with pytest.raises(CapabilityError, match="JVP"):
        mapping.predict.linear_operator(np.ones(2))


@pytest.mark.parametrize("jvp_present,vjp_present", [(True, False), (False, True)])
def test_derivative_capabilities_are_independent(jvp_present, vjp_present):
    domain, _ = spaces()
    calls = []

    def solve(x):
        calls.append(True)
        return 2*x

    tangent = (lambda trajectory, h: 2*h) if jvp_present else None
    pullback = (lambda trajectory, z: 2*z) if vjp_present else None
    mapping = state_map_from_solver(domain=domain, codomain=domain, solve=solve,
        jvp=tangent, vjp=pullback, states=lambda trajectory: trajectory)
    assert (mapping.jvp is not None) == jvp_present
    assert (mapping.vjp is not None) == vjp_present
    with pytest.raises(CapabilityError):
        mapping.predict.linear_operator(np.ones(2))
    assert not calls
    at, direction = np.array([-1., 2.]), np.array([.3, -.4])
    if jvp_present:
        result = mapping.jvp.jvp(at, direction)
    else:
        result = mapping.vjp.vjp(at, direction)
    np.testing.assert_array_equal(result, 2*direction)
    assert calls == [True]


@pytest.mark.parametrize("bad", [np.ones(2), np.array([1., np.nan, 0.]), np.ones(3)*1j])
def test_invalid_state_extraction_is_rejected(bad):
    domain, codomain = spaces()
    mapping = state_map_from_solver(domain=domain, codomain=codomain,
                                   solve=lambda x: bad, states=lambda trajectory: trajectory)
    with pytest.raises(ValueError):
        mapping.predict.predict(np.ones(2))


@pytest.mark.parametrize("operation,bad", [
    ("jvp", np.ones(2)), ("jvp", np.array([1., np.nan, 0.])), ("jvp", np.ones(3)*1j),
    ("vjp", np.ones(3)), ("vjp", np.array([1., np.inf])), ("vjp", np.ones(2)*1j),
])
def test_invalid_derivative_arrays_are_rejected(operation, bad):
    domain, codomain = spaces()
    mapping = state_map_from_solver(domain=domain, codomain=codomain,
        solve=lambda x: np.ones(3), states=lambda t: t,
        jvp=lambda trajectory, h: bad if operation == "jvp" else np.ones(3),
        vjp=lambda trajectory, z: bad if operation == "vjp" else np.ones(2))
    with pytest.raises(ValueError):
        if operation == "jvp":
            mapping.jvp.jvp(np.ones(2), np.ones(2))
        else:
            mapping.vjp.vjp(np.ones(2), np.ones(3))
    linear = mapping.predict.linear_operator(np.ones(2))
    with pytest.raises(ValueError):
        (linear.matvec if operation == "jvp" else linear.rmatvec)(
            np.ones(2 if operation == "jvp" else 3))


def test_array_arguments_and_derivative_results_are_copied():
    domain, _ = spaces()
    derivative_result = np.ones(2)

    def solve(x):
        x[:] = 5.
        return x

    def derivative(trajectory, direction):
        direction[:] = -8.
        return derivative_result

    mapping = state_map_from_solver(domain=domain, codomain=domain, solve=solve,
        states=lambda t: t, jvp=derivative, vjp=derivative)
    at, h, z = np.ones(2), np.array([2., 3.]), np.array([4., 5.])
    np.testing.assert_array_equal(mapping.predict.predict(at), [5., 5.])
    tangent, pullback = mapping.jvp.jvp(at, h), mapping.vjp.vjp(at, z)
    np.testing.assert_array_equal(at, np.ones(2))
    np.testing.assert_array_equal(h, [2., 3.])
    np.testing.assert_array_equal(z, [4., 5.])
    tangent[:] = 9.
    pullback[:] = 10.
    np.testing.assert_array_equal(derivative_result, np.ones(2))


def test_explicit_copy_supports_resource_owning_trajectory_without_freezing_functions():
    @dataclass
    class Trajectory:
        point: np.ndarray
        states: np.ndarray
        lock: object

    domain, _ = spaces()
    copies = []
    configuration = {"scale": 2.}

    def solve(x):
        return Trajectory(x.copy(), configuration["scale"]*x, threading.Lock())

    def copy_data(trajectory):
        copies.append(True)
        return Trajectory(trajectory.point.copy(), trajectory.states.copy(), trajectory.lock)

    mapping = state_map_from_solver(domain=domain, codomain=domain, solve=solve,
        jvp=lambda t, h: configuration["scale"]*h,
        vjp=lambda t, z: configuration["scale"]*z, copy_trajectory=copy_data)
    assert not copies
    np.testing.assert_array_equal(mapping.predict.predict(np.ones(2)), [2., 2.])
    linear = mapping.predict.linear_operator(np.ones(2))
    assert len(copies) == 2
    # Объявленные функции остаются у вызывающей стороны; данные траектории копируются.
    configuration["scale"] = 3.
    np.testing.assert_array_equal(linear.matvec(np.ones(2)), [3., 3.])


def settings(**changes):
    values = dict(horizon=.2, diffusion=.13, velocity=(.15, -.1),
                  linear_loss=.04, reaction_gamma=.3, reaction_c_star=.7,
                  source_position=(0., 0.))
    values.update(changes)
    return values


def bounded_spaces(solver, *, forcing=False):
    coordinates = (solver.times, tuple(map(tuple, solver.xy)))
    state = ArraySpace((solver.nt+1, solver.size), axes=("time", "node"),
                       coordinates=coordinates, units="state")
    domain = state if forcing else ArraySpace((solver.nt+1,), axes=("time",),
                                              coordinates=(solver.times,), units="input")
    return domain, state


@pytest.mark.parametrize("forcing", [False, True])
def test_bounded_structural_backend_does_not_require_solver_inheritance(forcing):
    class StructuralBackend:
        def __init__(self, solver):
            self.solver = solver

        def __getattr__(self, name):
            return getattr(self.solver, name)

        def snapshot(self):
            return StructuralBackend(self.solver.snapshot())

    solver = BoundedSolver(settings(), interior_points=5, time_steps=4)
    backend = StructuralBackend(solver)
    assert not isinstance(backend, BoundedSolver)
    domain, codomain = bounded_spaces(backend, forcing=forcing)
    factory = forcing_state_map if forcing else state_map
    mapping = factory(backend, domain=domain, codomain=codomain)
    controls = np.ones(domain.shape)
    expected = (solver.solve_forcing if forcing else solver.solve_controls)(controls).states
    solver.load[:] = 0.
    solver.model["diffusion"] = 99.
    np.testing.assert_array_equal(mapping.predict.predict(controls), expected)


def test_bounded_solver_accepts_explicit_problem_without_changing_arithmetic():
    data = settings()
    typed = BoundedSolver(ADRProblem.from_model(data), interior_points=5, time_steps=4)
    reference_solver = BoundedSolver(data, interior_points=5, time_steps=4)
    np.testing.assert_array_equal(typed.solve_controls(np.ones(5)).states,
                                  reference_solver.solve_controls(np.ones(5)).states)


def test_structural_bounded_metadata_and_methods_fail_before_snapshot_or_solve():
    class MissingMethods:
        def snapshot(self):
            raise AssertionError("methods must be checked before snapshot")

    domain, codomain = spaces()
    with pytest.raises(TypeError, match="callable methods"):
        state_map(MissingMethods(), domain=domain, codomain=codomain)

    class MissingMetadata:
        solve_controls = tangent_controls = adjoint_controls = lambda *args: None

        def snapshot(self):
            raise AssertionError("metadata must be checked before snapshot")

    with pytest.raises(TypeError, match="metadata"):
        state_map(MissingMetadata(), domain=domain, codomain=codomain)


def test_bounded_snapshot_must_retain_the_validated_geometry():
    class BadSnapshot:
        def __init__(self, solver):
            self.solver = solver

        def __getattr__(self, name):
            return getattr(self.solver, name)

        def snapshot(self):
            bad = self.solver.snapshot()
            bad.xy = bad.xy[::-1]
            return bad

    solver = BoundedSolver(settings(), interior_points=5, time_steps=4)
    domain, codomain = bounded_spaces(solver)
    with pytest.raises(ValueError, match="geometry"):
        state_map(BadSnapshot(solver), domain=domain, codomain=codomain)


def test_bounded_declared_units_are_preserved_in_structural_boundary():
    units = ADRUnits("km", "h", "ug/m^3")
    solver = BoundedSolver(settings(units=units), interior_points=5, time_steps=4)
    domain = ArraySpace((5,), axes=("time",), coordinates=(solver.times,), units=units.source)
    codomain = ArraySpace((5, 25), axes=("time", "node"),
        coordinates=(solver.times, tuple(map(tuple, solver.xy))), units=units.concentration)
    mapping = state_map(solver, domain=domain, codomain=codomain)
    assert mapping.domain.units == units.source
    assert mapping.codomain.units == units.concentration


def _snapshot_spaces(solver, *, forcing):
    coordinates = (solver.times, tuple(range(solver.size)))
    units = solver.units
    states = ArraySpace((solver.nt+1, solver.size), axes=("time", "node"),
        coordinates=coordinates, units="state" if units is None else units.concentration)
    if forcing:
        domain = ArraySpace(states.shape, axes=states.axes, coordinates=coordinates,
                            units="input" if units is None else units.reaction)
    else:
        domain = ArraySpace((solver.nt+1,), axes=("time",), coordinates=(solver.times,),
                            units="input" if units is None else units.source)
    return domain, states


def _solver_with_snapshot(solver, make_snapshot):
    class SolverView:
        def __getattr__(self, name):
            return getattr(solver, name)

        def snapshot(self):
            return make_snapshot()

    return SolverView()


@pytest.mark.parametrize("forcing", [False, True])
def test_compatible_solver_private_backend_is_not_read(forcing):
    solver = BoundedSolver(settings(), interior_points=5, time_steps=4)

    class CompatibleSolver:
        def __getattr__(self, name):
            return getattr(solver, name)

        @property
        def _backend(self):
            raise AssertionError("Private solver state must not be inspected")

        def snapshot(self):
            return solver.snapshot()

    domain, codomain = bounded_spaces(solver, forcing=forcing)
    factory = forcing_state_map if forcing else state_map
    mapping = factory(model=CompatibleSolver(), domain=domain, codomain=codomain)
    at = np.linspace(.1, .5, domain.size).reshape(domain.shape)
    direction = np.cos(np.arange(domain.size)).reshape(domain.shape)
    cotangent = np.sin(np.arange(codomain.size)).reshape(codomain.shape)
    solve = solver.solve_forcing if forcing else solver.solve_controls
    tangent = solver.tangent_forcing if forcing else solver.tangent_controls
    pullback = solver.adjoint_forcing if forcing else solver.adjoint_controls
    trajectory = solve(at)
    np.testing.assert_array_equal(mapping.predict.predict(at), trajectory.states)
    np.testing.assert_array_equal(mapping.jvp.jvp(at, direction), tangent(trajectory, direction))
    np.testing.assert_array_equal(mapping.vjp.vjp(at, cotangent), pullback(trajectory, cotangent))


@pytest.mark.parametrize("forcing", [False, True])
def test_private_backend_without_public_methods_is_rejected_before_execution(forcing, monkeypatch):
    solver = BoundedSolver(settings(), interior_points=5, time_steps=4)
    calls = []

    def forbidden(*args):
        calls.append(True)
        raise AssertionError("Missing public methods must fail before snapshot or solve")

    monkeypatch.setattr(solver, "snapshot", forbidden)
    monkeypatch.setattr(solver, "solve_controls", forbidden)
    monkeypatch.setattr(solver, "solve_forcing", forbidden)

    class BackendHolder:
        _backend = solver

    domain, codomain = bounded_spaces(solver, forcing=forcing)
    factory = forcing_state_map if forcing else state_map
    with pytest.raises(TypeError, match="callable methods"):
        factory(model=BackendHolder(), domain=domain, codomain=codomain)
    assert not calls


@pytest.mark.parametrize("forcing", [False, True])
@pytest.mark.parametrize("nested_lists", [False, True])
@pytest.mark.parametrize("change_original", [False, True])
def test_snapshot_geometry_is_preserved_for_indexed_nodes(forcing, nested_lists, change_original):
    solver = BoundedSolver(settings(), interior_points=5, time_steps=4)
    if nested_lists:
        solver.xy = solver.xy.tolist()
    domain, codomain = _snapshot_spaces(solver, forcing=forcing)

    def changed_snapshot():
        snapshot = solver.snapshot()
        if nested_lists:
            snapshot.xy[0][0] += 1.
            if change_original:
                solver.xy[0][0] = snapshot.xy[0][0]
        else:
            snapshot.xy[:] = snapshot.xy[::-1]
            if change_original:
                solver.xy[:] = snapshot.xy
        return snapshot

    factory = forcing_state_map if forcing else state_map
    with pytest.raises(ValueError, match="snapshot.*geometry"):
        factory(model=_solver_with_snapshot(solver, changed_snapshot),
                domain=domain, codomain=codomain)


@pytest.mark.parametrize("forcing", [False, True])
@pytest.mark.parametrize("change", ["drop", "length", "declare"])
def test_snapshot_preserves_the_original_unit_declaration(forcing, change):
    units = ADRUnits("km", "h", "ug/m^3")
    solver = BoundedSolver(settings(units=None if change == "declare" else units),
                           interior_points=5, time_steps=4)
    if change == "declare":
        declared = BoundedSolver(settings(units=units), interior_points=5, time_steps=4)
        domain, codomain = _snapshot_spaces(declared, forcing=forcing)
    else:
        domain, codomain = _snapshot_spaces(solver, forcing=forcing)

    def changed_snapshot():
        snapshot = solver.snapshot()
        if change == "drop":
            snapshot.units = None
        elif change == "length":
            snapshot.units = ADRUnits("m", units.time, units.concentration)
        else:
            snapshot.units = units
        return snapshot

    factory = forcing_state_map if forcing else state_map
    with pytest.raises(ValueError, match="unit"):
        factory(model=_solver_with_snapshot(solver, changed_snapshot),
                domain=domain, codomain=codomain)


@pytest.mark.parametrize("forcing", [False, True])
def test_snapshot_time_nodes_must_match_the_declared_spaces(forcing):
    solver = BoundedSolver(settings(), interior_points=5, time_steps=4)
    domain, codomain = _snapshot_spaces(solver, forcing=forcing)

    def changed_snapshot():
        snapshot = solver.snapshot()
        snapshot.times += 1.
        return snapshot

    factory = forcing_state_map if forcing else state_map
    with pytest.raises(ValueError, match="time nodes"):
        factory(model=_solver_with_snapshot(solver, changed_snapshot),
                domain=domain, codomain=codomain)


@pytest.mark.parametrize("forcing", [False, True])
def test_indexed_snapshot_accepts_equal_independent_unit_records(forcing):
    units = ADRUnits("km", "h", "ug/m^3")
    solver = BoundedSolver(settings(units=units), interior_points=5, time_steps=4)
    domain, codomain = _snapshot_spaces(solver, forcing=forcing)

    def independent_snapshot():
        snapshot = solver.snapshot()
        snapshot.units = ADRUnits(units.length, units.time, units.concentration)
        assert snapshot.units == units and snapshot.units is not units
        return snapshot

    factory = forcing_state_map if forcing else state_map
    mapping = factory(model=_solver_with_snapshot(solver, independent_snapshot),
                      domain=domain, codomain=codomain)
    values = np.linspace(.1, .5, domain.size).reshape(domain.shape)
    expected = (solver.solve_forcing if forcing else solver.solve_controls)(values).states
    np.testing.assert_array_equal(mapping.predict.predict(values), expected)
