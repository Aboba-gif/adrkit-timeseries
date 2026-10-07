"""Проверки владения компонентами PDE и отказов предварительной проверки."""
from copy import copy
from types import SimpleNamespace

import numpy as np
import pytest

from adrkit.backends.bounded import BoundedSolver
from adrkit.backends.whole_space import Transport, gaussian_kernel
from adrkit.coefficients import ADRUnits
from adrkit.contracts import MapCapabilities
from adrkit.loads import DisclosedHistory, IntervalAverageLoad, disclose_history
from adrkit.pde import ADRProblem
from adrkit.inverse.objectives import compose_maps
from adrkit.predictions import GridPrediction, SelectedPrediction, grid_observation, solver_state_space
from adrkit.sources import P1Basis
from adrkit.spaces import ArraySpace, matrix_map
from adrkit.state_maps import state_map


@pytest.fixture
def solver_calls(monkeypatch):
    calls = {}
    for name in ("solve_controls", "tangent_controls", "adjoint_controls"):
        original = getattr(BoundedSolver, name)

        def record(solver, *args, _name=name, _method=original):
            counts = calls.setdefault(id(solver),
                dict(solve_controls=0, tangent_controls=0, adjoint_controls=0))
            counts[_name] += 1
            return _method(solver, *args)

        monkeypatch.setattr(BoundedSolver, name, record)
    return calls


def components():
    units = ADRUnits("km", "h", "C")
    solver = ADRProblem(diffusion=.1, velocity=(.1, -.1), linear_loss=.01,
                        horizon=1., units=units).make_solver(
        BoundedSolver, interior_points=3, time_steps=2)
    load = IntervalAverageLoad(P1Basis([0., 1.], time_unit="h"), solver.times,
        origin=0., q_reference=100., history_integral=lambda a, b: 0.,
        history_initial=0., source_unit=units.source)
    states = solver_state_space(solver)
    values = ArraySpace((1,), axes=("observation",), coordinates=((0,),),
                        units=states.units, dtype=states.dtype)
    observation = grid_observation(np.ones((1, solver.size)),
        np.ones((1, solver.nt+1)), state_space=states, codomain=values)
    return solver, load, observation


@pytest.mark.parametrize("method,bad", [
    ("jvp", np.zeros(3)), ("jvp", np.array([0., np.inf])),
    ("vjp", np.zeros(2)), ("vjp", np.array([np.nan])),
])
def test_bad_derivative_argument_does_not_solve(method, bad, solver_calls):
    prediction = GridPrediction(*components())
    with pytest.raises(ValueError):
        getattr(prediction, method)(np.zeros(2), bad)
    assert not solver_calls


@pytest.mark.parametrize("change", ["times", "points", "units", "method"])
def test_owned_solver_is_rechecked_after_snapshot(change):
    solver, load, observation = components()
    snapshot = solver.snapshot()
    if change == "times":
        snapshot.times *= 2
    elif change == "points":
        snapshot.xy[:, 0] += 1
    elif change == "units":
        snapshot.units = ADRUnits("m", "s", "K")
    else:
        snapshot.tangent_controls = None
    snapshot.solve_controls = lambda x: pytest.fail("Invalid snapshot must not solve")
    solver.snapshot = lambda: snapshot
    with pytest.raises((ValueError, TypeError)):
        GridPrediction(solver, load, observation)


def test_owned_load_cannot_relabel_parameter_space_after_deepcopy():
    solver, load, observation = components()

    class ChangedLoad:
        domain, codomain = load.domain, load.codomain
        predict, jvp, vjp = load.predict, load.jvp, load.vjp

        def __deepcopy__(self, memo):
            result = copy(self)
            result.domain = ArraySpace((2,), axes=("coefficient",),
                coordinates=(load.domain.coordinates[0],), units="changed")
            return result

    with pytest.raises(ValueError, match="Incompatible"):
        GridPrediction(solver, ChangedLoad(), observation)


@pytest.mark.parametrize("method", ["predict", "jvp", "vjp"])
def test_interval_load_rejects_nonfinite_result_from_finite_input(method):
    _, load, _ = components()
    huge = np.full(load.codomain.shape if method == "vjp" else load.domain.shape, 1e308)
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="valid values"):
            if method == "predict":
                load.predict(huge)
            else:
                getattr(load, method)(np.zeros(load.domain.shape), huge)


def test_interval_load_preserves_existing_arithmetic():
    _, load, _ = components()
    at, direction, cotangent = np.array([.15, .25]), np.array([-.2, .4]), np.array([.7, -.3, .8])
    np.testing.assert_array_equal(load.predict(at), load.history + load.q_reference*(load.averages @ at))
    np.testing.assert_array_equal(load.jvp(at, direction), load.q_reference*(load.averages @ direction))
    np.testing.assert_array_equal(load.vjp(at, cotangent), load.q_reference*(load.averages.T @ cotangent))


@pytest.mark.parametrize("separate_providers", [False, True])
def test_grid_prediction_accepts_load_capabilities(separate_providers, solver_calls):
    solver, load, observation = components()
    if separate_providers:
        capabilities = MapCapabilities(
            SimpleNamespace(predict=load.predict), load.domain, load.codomain,
            SimpleNamespace(jvp=load.jvp), SimpleNamespace(vjp=load.vjp))
    else:
        capabilities = load.capabilities()
    direct = GridPrediction(solver, load, observation)
    declared = GridPrediction(solver, capabilities, observation)
    at, direction, cotangent = [.15, .25], [-.2, .4], [.7]
    np.testing.assert_array_equal(declared.predict(at), direct.predict(at))
    np.testing.assert_array_equal(declared.jvp(at, direction), direct.jvp(at, direction))
    np.testing.assert_array_equal(declared.vjp(at, cotangent), direct.vjp(at, cotangent))
    assert direct._load.q_reference == load.q_reference
    assert list(solver_calls.values()) == [
        dict(solve_controls=1, tangent_controls=1, adjoint_controls=1)]*2


def test_grid_prediction_accepts_public_matrix_load():
    solver, load, observation = components()
    capabilities = matrix_map(load.q_reference*load.averages,
                              domain=load.domain, codomain=load.codomain)
    direct = GridPrediction(solver, capabilities.predict, observation)
    matrix = GridPrediction(solver, capabilities, observation)
    at, direction, cotangent = [.15, .25], [-.2, .4], [.7]
    np.testing.assert_array_equal(matrix.predict(at), direct.predict(at))
    np.testing.assert_array_equal(matrix.jvp(at, direction), direct.jvp(at, direction))
    np.testing.assert_array_equal(matrix.vjp(at, cotangent), direct.vjp(at, cotangent))


@pytest.mark.parametrize("missing", ["jvp", "vjp"])
def test_grid_prediction_does_not_infer_undeclared_derivatives(missing):
    solver, load, observation = components()
    providers = dict(jvp=load, vjp=load)
    providers[missing] = None
    capabilities = MapCapabilities(load, load.domain, load.codomain, **providers)
    solver.snapshot = lambda: pytest.fail("Missing derivative must fail before snapshot")
    with pytest.raises(TypeError, match=missing):
        GridPrediction(solver, capabilities, observation)


def test_general_state_functional_composition_and_projection(solver_calls):
    solver, load, fit_observation = components()
    states_space = fit_observation.domain
    values = ArraySpace((2,), axes=("observation",),
        coordinates=(("integral", "contrast"),), units="C*km^2*h", binding="grid-functional")
    spatial = np.vstack((np.ones(solver.size), np.arange(solver.size)-4.))
    temporal = np.array([[.25, .5, .25]])
    observation = grid_observation(spatial, temporal,
        state_space=states_space, codomain=values)
    state = state_map(solver, domain=load.codomain, codomain=states_space)
    composed = compose_maps(load.capabilities(), state, observation.capabilities())
    at, direction, cotangent = np.array([.15, .25]), np.array([-.2, .4]), np.array([.7, -.3])
    trajectory = solver.solve_controls(load.predict(at))
    tangent = solver.tangent_controls(trajectory, load.jvp(at, direction))

    def manual(states):
        return np.array([sum(temporal[0, t]*spatial[r, n]*states[t, n]
            for t in range(solver.nt+1) for n in range(solver.size)) for r in range(2)])

    np.testing.assert_allclose(composed.predict.predict(at), manual(trajectory.states), rtol=1e-14, atol=1e-14)
    jvp = composed.jvp.jvp(at, direction)
    np.testing.assert_allclose(jvp, manual(tangent), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(jvp @ cotangent,
        direction @ composed.vjp.vjp(at, cotangent), rtol=1e-13, atol=1e-14)
    eps = 1e-6
    difference = (composed.predict.predict(at+eps*direction)-composed.predict.predict(at-eps*direction))/(2*eps)
    np.testing.assert_allclose(jvp, difference, rtol=2e-9, atol=2e-10)
    solver_calls.clear()
    prediction = GridPrediction(solver, load, fit_observation)
    np.testing.assert_allclose(prediction.project(at, observation), manual(trajectory.states), rtol=1e-14, atol=1e-14)
    matrix, = prediction.projected_jacobians(at, [observation])
    np.testing.assert_allclose(matrix @ direction, jvp, rtol=1e-14, atol=1e-14)
    assert composed.codomain == values
    assert prediction.codomain == fit_observation.codomain
    assert list(solver_calls.values()) == [
        dict(solve_controls=1, tangent_controls=2, adjoint_controls=0)]


def test_concentration_prediction_rejects_integral_units_before_snapshot():
    solver, load, observation = components()
    integral = ArraySpace(observation.codomain.shape, axes=observation.codomain.axes,
        coordinates=observation.codomain.coordinates, units="C*km^2*h")
    observation = grid_observation(observation.space_weights, observation.time_weights,
                                   state_space=observation.domain, codomain=integral)
    solver.snapshot = lambda: pytest.fail("Incompatible concentration units must fail before snapshot")
    with pytest.raises(ValueError, match="concentration units"):
        GridPrediction(solver, load, observation)


def test_transport_owns_mutable_numerical_inputs():
    velocity = np.array([.2, -.1])
    diffusion = np.array(.15)
    transport = Transport(diffusion, velocity, .12, .7)
    before = gaussian_kernel(np.array([1.]), [1., 0.], [0., 0.], transport)
    velocity[0] = 1
    diffusion[...] = 2
    np.testing.assert_array_equal(gaussian_kernel(np.array([1.]), [1., 0.], [0., 0.], transport), before)
    assert transport.velocity == (.2, -.1)


@pytest.mark.parametrize("lag,expected", [
    (0., 1/(2*np.pi)),
    ([0., 1.], [1/(2*np.pi), 1/(4*np.pi)]),
    ([[0., 1.], [3., 2.]], [[1/(2*np.pi), 1/(4*np.pi)], [1/(8*np.pi), 1/(6*np.pi)]]),
])
def test_gaussian_kernel_normalization_preserves_scalar_and_array_lag_shapes(lag, expected):
    # При совпадении источника и приёмника, нулевых скорости и потерях
    # и D=1/2, w=1 ядро равно 1/(2*pi*(1+lag)).
    model = Transport(.5, (0., 0.), 0., 1.)
    value = gaussian_kernel(lag, [2., -1.], [2., -1.], model)
    assert np.shape(value) == np.shape(lag)
    np.testing.assert_allclose(value, expected, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("field", ["receiver", "source"])
@pytest.mark.parametrize("point", [[1.], [1., 0., 100.], [[1., 0.]],
                                    [np.nan, 0.], [np.inf, 0.], np.array([1.+1.j, 0.])])
def test_gaussian_kernel_rejects_invalid_coordinate_before_kernel_evaluation(monkeypatch, field, point):
    model = Transport(.15, (.2, -.1), .12, .7)
    coordinates = dict(receiver=[1., 0.], source=[0., 0.])
    coordinates[field] = point
    monkeypatch.setattr(np, "exp", lambda _: pytest.fail("invalid coordinate reached exponential"))
    with pytest.raises(ValueError, match=field):
        gaussian_kernel([0., 1.], model=model, **coordinates)


@pytest.mark.parametrize("lag", [[-1.], [np.inf], np.array([1.+1.j])])
def test_gaussian_kernel_rejects_invalid_lag_before_kernel_evaluation(monkeypatch, lag):
    model = Transport(.15, (.2, -.1), .12, .7)
    monkeypatch.setattr(np, "exp", lambda _: pytest.fail("invalid lag reached exponential"))
    with pytest.raises(ValueError, match="Lag"):
        gaussian_kernel(lag, [1., 0.], [0., 0.], model)


@pytest.mark.parametrize("velocity", [[True, .2], [.1, np.inf], [.2], [.2, .3j]])
def test_transport_rejects_invalid_velocity(velocity):
    with pytest.raises(ValueError, match="velocity"):
        Transport(.15, velocity, .12, .7)


@pytest.mark.parametrize("method", ["predict", "jvp", "vjp"])
def test_selected_prediction_checks_point_before_external_base(method):
    layout = ArraySpace((2,), axes=("entry",), coordinates=((0, 1),), units="1")
    def unexpected(*args):
        pytest.fail("Invalid point must not reach external prediction")
    base = SimpleNamespace(domain=layout, codomain=layout, predict=unexpected,
                           jvp=unexpected, vjp=unexpected)
    selected = SelectedPrediction(base, [0])
    with pytest.raises(ValueError):
        if method == "predict":
            selected.predict(np.zeros(3))
        else:
            getattr(selected, method)(np.zeros(3), np.zeros(2 if method == "jvp" else 1))


def test_solver_state_space_uses_declared_unit_and_explicit_override():
    solver, _, _ = components()
    assert solver_state_space(solver).units == "C"
    assert solver_state_space(solver, state_unit="other").units == "other"


def test_disclosed_history_owns_intervals_integrals_and_initial_scalar():
    intervals, integrals, initial = [[-.5, 0.]], [2.], np.array(3.)
    history = DisclosedHistory(intervals, integrals, initial)
    intervals[0][0] = -9
    integrals[0] = 99
    initial[...] = 100
    assert history.intervals == ((-.5, 0.),)
    assert history.integrals == (2.,)
    assert history.initial == 3.
    assert history.integral(-.5, 0.) == 2.


@pytest.mark.parametrize("integrals", [[2., 99.], [2., 2.]])
def test_disclosed_history_rejects_ambiguous_duplicate_interval_ids(integrals):
    with pytest.raises(ValueError, match="unique endpoint pairs"):
        DisclosedHistory([[-.5, 0.], [-.5, 0.]], integrals, 0.)


def test_disclosed_history_preserves_overlapping_intervals_and_supplied_order():
    history = DisclosedHistory([[-.4, -.1], [-.5, -.2]], [2., 3.], 0.)
    assert history.intervals == ((-.4, -.1), (-.5, -.2))
    assert history.integral(-.4, -.1) == 2.
    assert history.integral(-.5, -.2) == 3.


@pytest.mark.parametrize("bad", [np.nan, np.inf, np.complex128(1+2j), True, "1", [1.]])
@pytest.mark.parametrize("operation", ["integral", "value"])
def test_disclose_history_rejects_invalid_sample_without_complex_cast(operation, bad):
    source = SimpleNamespace(integral=lambda a, b: 2., value=lambda t: 3.)
    if operation == "integral":
        source.integral = lambda a, b: bad
    else:
        source.value = lambda t: bad
    with pytest.raises(ValueError):
        disclose_history(source, [0., .5, 1.], -.5)


@pytest.mark.parametrize("times,origin", [
    ([0., .5, np.nan], -.5), ([0., .5, .25], -.5), ([.1, .5], -.5),
    ([0., 1.], np.inf), ([0., 1.], np.complex128(1+2j)), ([0.], -.5),
])
def test_disclosed_history_clock_rejected_before_source_sampling(times, origin):
    def forbidden(*args):
        pytest.fail("Invalid clock must fail before sampling the history source")
    with pytest.raises(ValueError):
        disclose_history(SimpleNamespace(integral=forbidden, value=forbidden), times, origin)


@pytest.mark.parametrize("missing", ["integral", "value"])
def test_disclosed_history_requires_source_operations_before_sampling(missing):
    def forbidden(*args):
        pytest.fail("Malformed source must fail before sampling")
    source = SimpleNamespace(integral=forbidden, value=forbidden)
    setattr(source, missing, None)
    with pytest.raises(TypeError, match=missing):
        disclose_history(source, [0., .5, 1.], -.5)


def test_disclosed_history_preserves_interval_sampling_order_and_arithmetic():
    events = []
    def integral(left, right):
        events.append((left, right))
        return .5*(right**2-left**2)
    def value(time):
        events.append(time)
        return time
    times, origin = np.array([0., .25, .5, .75, 1.]), -.5
    history = disclose_history(SimpleNamespace(integral=integral, value=value), times, origin)
    assert events == [(-.5, -.25), (-.25, 0.), -.5]
    assert history.intervals == ((-.5, -.25), (-.25, 0.))
    assert history.integrals == (-.09375, -.03125)
    assert history.initial == -.5


def test_prediction_and_selected_view_do_not_store_operation_counters():
    prediction = GridPrediction(*components())
    selected = SelectedPrediction(prediction, [0])
    for model in (prediction, selected):
        for name in ("forward_calls", "tangent_calls", "adjoint_calls"):
            assert not hasattr(model, name)
