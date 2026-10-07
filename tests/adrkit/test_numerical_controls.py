"""Сдвиги времени, физические единицы, внешние метрики и параметры нелинейного решателя."""
from types import SimpleNamespace

import numpy as np
import pytest

from adrkit.backends.bounded import BoundedSolver
from adrkit.backends.cached import CachedSolver
from adrkit.backends.whole_space import Transport, observation_matrix
from adrkit.contracts import MapCapabilities
from adrkit.errors import CapabilityError
from adrkit.loads import IntervalAverageLoad, disclose_history
from adrkit.observations.analytic import Quadrature, SensorSpec, SpatialSource, p1_response
from adrkit.predictions import solver_state_space
from adrkit.sources import P1Basis
from adrkit.spaces import ArraySpace
from adrkit.inverse.sensitivity import SensitivityLimits, sensitivity_spectrum
from adrkit.inverse.projected import fit
from adrkit.inverse.regularization import QuadraticPenalty
from adrkit.inverse import tikhonov


def vector_space(coordinates, *, units="1", axis="parameter"):
    return ArraySpace((len(coordinates),), axes=(axis,),
                      coordinates=(tuple(coordinates),), units=units)


@pytest.mark.parametrize("shift", [-17., 0., 23.])
def test_known_history_and_unknown_load_share_arbitrary_physical_origin(shift):
    origin, end = 3.+shift, 5.+shift
    times = np.array([0., 1., 2.5, 4.])
    basis = P1Basis(np.array([5., 6., 7.])+shift, time_unit="day")
    source = SimpleNamespace(value=lambda t: 2*(t-shift)+3,
        integral=lambda a, b: (b-shift)**2-(a-shift)**2+3*(b-a))
    history = disclose_history(source, times, origin, end=end)
    load = IntervalAverageLoad(basis, times, origin=origin, q_reference=2.,
        history_integral=history.integral, history_initial=history.initial,
        source_unit="kg/day")
    assert history.end == end
    assert history.intervals == ((3.+shift, 4.+shift), (4.+shift, 5.+shift))
    # Шаг состояния включает сутки заданной предыстории источника и половину суток неизвестного
    # источника. Известная часть делится на весь шаг длиной 1.5 суток.
    np.testing.assert_allclose(load.history, [9., 10., 8., 0.], atol=1e-14)
    np.testing.assert_allclose(load.averages[2], [.25, 1/12, 0.], atol=1e-14)
    with pytest.raises(ValueError, match="future"):
        history.integral(end, end+1.)


def test_history_rejects_unknown_origin_before_sampling():
    source = SimpleNamespace(value=lambda t: pytest.fail("unknown value sampled"),
                             integral=lambda a, b: pytest.fail("unknown integral sampled"))
    with pytest.raises(ValueError, match="origin"):
        disclose_history(source, [0., 1.], 6., end=5.)


@pytest.mark.parametrize("windows", [None, [.6, .4, 1.5]])
def test_analytic_source_and_observations_depend_on_time_differences(windows):
    times, knots = np.array([1.7, .2, 2.3]), np.array([0., .3, 1.1, 2.])
    receivers = np.array([[.2, -.1], [0., .3], [-.4, .2]])
    model = Transport(.15, (.2, -.1), .12, .7)
    matrices = []
    for shift in (0., -7., 12.):
        basis = P1Basis(knots+shift, time_unit="day")
        mapping = p1_response(basis,
            SensorSpec(times+shift, receivers, windows=windows, time_unit="day"),
            model, SpatialSource((0., 0.)), quadrature=Quadrature(24),
            coefficient_scale=2., domain=vector_space(basis.knots),
            codomain=vector_space(range(3), axis="measurement"))
        matrices.append(mapping.predict.matrix)
        p0 = observation_matrix(times+shift, receivers, [0., 0.], knots+shift, model)
        np.testing.assert_allclose(p0,
            observation_matrix(times, receivers, [0., 0.], knots, model), rtol=2e-14, atol=2e-14)
    for matrix in matrices[1:]:
        np.testing.assert_allclose(matrix, matrices[0], rtol=2e-14, atol=2e-14)


def test_unlabelled_state_does_not_assume_pollutant_units():
    solver = BoundedSolver(dict(diffusion=.1, velocity=(.1, -.1),
        linear_loss=.02, horizon=.5), interior_points=3, time_steps=4)
    assert solver_state_space(solver).units == "unknown"
    assert solver_state_space(solver, state_unit="kelvin").units == "kelvin"


def nonlinear_model():
    return dict(diffusion=.1, velocity=(.1, -.1), linear_loss=.02,
                horizon=.5, reaction_gamma=2., reaction_c_star=.1)


def test_explicit_newton_defaults_preserve_states_and_snapshot_options():
    default = BoundedSolver(nonlinear_model(), interior_points=3, time_steps=4)
    configured = BoundedSolver(nonlinear_model(), interior_points=3, time_steps=4,
        residual_tolerance=1e-11, max_newton_iterations=20,
        max_backtracks=20, backtrack_factor=.5)
    controls = np.linspace(.1, 5., 5)
    first, second = default.solve_controls(controls), configured.solve_controls(controls)
    np.testing.assert_array_equal(first.states, second.states)
    assert (first.max_scaled_residual, first.nonlinear_iterations) == (
        second.max_scaled_residual, second.nonlinear_iterations)
    snapshot = configured.snapshot()
    assert snapshot.residual_tolerance == configured.residual_tolerance
    assert snapshot.max_newton_iterations == configured.max_newton_iterations
    assert snapshot.max_backtracks == configured.max_backtracks
    assert snapshot.backtrack_factor == configured.backtrack_factor


def test_newton_limit_is_an_actual_configurable_stop():
    solver = BoundedSolver(nonlinear_model(), interior_points=3, time_steps=4,
                           max_newton_iterations=1)
    with pytest.raises(RuntimeError, match="Newton failed"):
        solver.solve_controls(np.full(5, 5.))


@pytest.mark.parametrize("option,value", [
    ("residual_tolerance", 0.), ("residual_tolerance", np.nan),
    ("residual_tolerance", True), ("max_newton_iterations", 1.5),
    ("max_newton_iterations", True), ("max_backtracks", 0),
    ("backtrack_factor", 1.), ("backtrack_factor", np.inf),
])
def test_invalid_solver_controls_fail_before_field_sampling(option, value):
    model = nonlinear_model()
    model["diffusion"] = lambda xy, t: pytest.fail("invalid options sampled fields")
    with pytest.raises(ValueError, match=option):
        BoundedSolver(model, interior_points=3, time_steps=4, **{option: value})


def test_external_whitening_metrics_match_independent_scaled_jacobian():
    domain, codomain = vector_space(["a", "b"]), vector_space(["u", "v", "w"])
    at = np.array([.3, -.2])
    matrix = np.array([[1., .4], [2., -.3], [.2, 1.5]])
    root, weights = np.array([[2., .3], [0., 1.5]]), np.array([2., 3., .5])
    class Forward:
        def predict(self, x):
            return matrix @ (x*x)

        def jvp(self, x, h):
            return matrix @ (2*x*h)
    class ExternalCoefficientMetric:
        def __init__(self):
            self.domain, self.root = domain, root
    class ExternalObservationMetric:
        layout = codomain

        def whiten_matrix(self, value):
            return weights[:, None]*value
    forward = Forward()
    result = sensitivity_spectrum(MapCapabilities(forward, domain, codomain, forward), at,
        coefficient_metric=ExternalCoefficientMetric(),
        observation_metric=ExternalObservationMetric(), limits=SensitivityLimits(2, 6, 2))
    expected = (weights[:, None]*matrix) @ np.diag(2*at) @ np.linalg.inv(root)
    np.testing.assert_allclose(result.singular_values, np.linalg.svd(expected, compute_uv=False),
                               rtol=2e-14, atol=2e-14)


def test_invalid_external_coefficient_factor_rejected_before_jvp():
    domain = vector_space(["a", "b"])
    class Forbidden:
        def predict(self, x):
            pytest.fail("invalid root predicted")

        def jvp(self, x, h):
            pytest.fail("invalid root differentiated")
    operation = Forbidden()
    observation = SimpleNamespace(layout=domain, whiten_matrix=lambda matrix: matrix)
    coefficient = SimpleNamespace(domain=domain, root=[[1., 0.], [.3, 1.]])
    with pytest.raises(ValueError, match="upper-triangular"):
        sensitivity_spectrum(MapCapabilities(operation, domain, domain, operation), [1., 2.],
            coefficient_metric=coefficient, observation_metric=observation,
            limits=SensitivityLimits(2, 4, 2))


class LinearFit:
    def predict(self, x):
        return x.copy()

    def vjp(self, x, z):
        return z.copy()


class IdentityMetric:
    def whiten(self, x):
        return x.copy()

    def whiten_matrix(self, matrix):
        return matrix.copy()

    def apply_precision(self, x):
        return x.copy()


def fit_linear(observed, **options):
    observed = np.asarray(observed, dtype=float)
    return fit(LinearFit(), observed, IdentityMetric(),
        np.eye(observed.size), 1., np.eye(observed.size), np.zeros(observed.size), **options)


def test_projected_fit_requires_only_prediction_and_vjp():
    class ForeignProperties(LinearFit):
        @property
        def forward_calls(self):
            pytest.fail("fit read a foreign forward counter")

        @property
        def adjoint_calls(self):
            pytest.fail("fit read a foreign adjoint counter")

        @property
        def trajectory(self):
            pytest.fail("fit read a foreign trajectory")

    result = fit(ForeignProperties(), np.array([2., 4.]), IdentityMetric(),
        np.eye(2), 1., np.eye(2), np.zeros(2))
    np.testing.assert_allclose(result["coefficients"], [1., 2.], atol=1e-14)
    assert result["status"] == "accepted" and result["accepted"]
    assert not {"seconds", "forward_calls", "adjoint_calls", "forward_residual"} & result.keys()


def test_projected_iteration_budget_stops_a_real_descent():
    stopped = fit_linear([2., 4.], max_iterations=0, initial_point=np.zeros(2))
    solved = fit_linear([2., 4.], max_iterations=1, initial_point=np.zeros(2))
    assert stopped["status"] == "iteration_limit" and stopped["iterations"] == 0
    assert solved["status"] == "accepted" and solved["iterations"] == 1
    np.testing.assert_allclose(solved["coefficients"], [1., 2.], atol=1e-14)


def test_projected_boundary_tolerance_changes_the_reported_stationarity():
    boundary = fit_linear([-1.], tolerance=1e-12, max_iterations=0,
                          initial_point=np.array([1e-15]))
    interior = fit_linear([-1.], tolerance=1e-12, max_iterations=0,
                          initial_point=np.array([1e-15]), active_tolerance=0.)
    assert boundary["accepted"] and boundary["stationarity"] == 0.
    assert not interior["accepted"] and interior["stationarity"] > .9


@pytest.mark.parametrize("option,value", [
    ("active_tolerance", True),
    ("nnls_max_iterations", 0), ("max_iterations", -1),
    ("max_backtracks", 1.5), ("armijo_coefficient", 1.),
    ("backtrack_factor", 0.), ("tolerance", np.nan),
])
def test_projected_invalid_controls_fail_before_whitening_or_prediction(option, value):
    metric = SimpleNamespace(whiten_matrix=lambda _: pytest.fail("invalid controls whitened"))
    prediction = SimpleNamespace(predict=lambda _: pytest.fail("invalid controls predicted"))
    with pytest.raises(ValueError):
        fit(prediction, np.ones(1), metric, np.eye(1), 1., np.eye(1), np.zeros(1),
            **{option: value})


@pytest.mark.parametrize("alpha", [-.1, True, "0.1", np.nan, np.inf, .1+0j, [.1]])
def test_projected_invalid_penalty_weight_fails_before_callbacks(alpha):
    metric = SimpleNamespace(whiten_matrix=lambda _: pytest.fail("invalid alpha whitened"))
    prediction = SimpleNamespace(predict=lambda _: pytest.fail("invalid alpha predicted"))
    with pytest.raises(ValueError, match="alpha"):
        fit(prediction, [1.], metric, [[1.]], alpha, [[1.]], [0.])


@pytest.mark.parametrize("gram,message", [
    (np.diag([-1., 0.]), "positive semidefinite"),
    (np.array([[1., .5], [-.5, 1.]]), "symmetric"),
])
def test_projected_rejects_invalid_gram_even_when_affine_hessian_is_positive(gram, message):
    # Обе матрицы прежде давали положительно определённую верхнюю часть H;
    # одного успешного разложения Холецкого для проверки штрафа недостаточно.
    metric = SimpleNamespace(whiten_matrix=lambda _: pytest.fail("invalid Gram whitened"))
    prediction = SimpleNamespace(predict=lambda _: pytest.fail("invalid Gram predicted"))
    with pytest.raises(ValueError, match=message):
        fit(prediction, np.ones(2), metric, gram, .1, np.eye(2), np.zeros(2))


@pytest.mark.parametrize("field,value", [
    ("observations", [[1., 2.]]), ("observations", [1.]),
    ("observations", [1., np.nan]), ("observations", [True, False]),
    ("jacobian", [1., 2.]), ("jacobian", np.ones((1, 2))),
    ("jacobian", np.zeros((2, 0))), ("jacobian", [[np.inf, 0.], [0., 1.]]),
    ("gram", [1., 1.]), ("gram", np.eye(3)),
    ("gram", [[1.+0j, 0.], [0., 1.]]), ("gram", [[1., 0.], [0., np.nan]]),
    ("offset", [0.]), ("offset", [[0., 0.]]),
    ("offset", [0., np.inf]), ("offset", ["0", "0"]),
    ("initial_point", [1.]), ("initial_point", [[1., 1.]]),
    ("initial_point", [1., -1.]), ("initial_point", [1., np.nan]),
])
def test_projected_numeric_inputs_fail_before_external_callbacks(field, value):
    arguments = dict(observations=np.ones(2), gram=np.eye(2), alpha=.1,
                     jacobian=np.eye(2), offset=np.zeros(2), initial_point=np.ones(2))
    arguments[field] = value
    metric = SimpleNamespace(whiten_matrix=lambda _: pytest.fail("invalid input whitened"))
    prediction = SimpleNamespace(predict=lambda _: pytest.fail("invalid input predicted"))
    with pytest.raises(ValueError, match=field):
        fit(prediction, metric=metric, **arguments)


@pytest.mark.parametrize("alpha", [0., .3])
def test_projected_weighted_affine_fit_matches_independent_normal_equations(alpha):
    matrix = np.array([[1., .2], [.3, 1.], [.5, .4]])
    offset, observed = np.array([.3, -.2, .1]), np.array([1.4, 1.8, 1.])
    weights, gram = np.array([2., 3., .5]), np.array([[2., .4], [.4, 1.]])

    class AffinePrediction:
        def predict(self, point):
            return matrix @ point+offset

        def vjp(self, point, cotangent):
            return matrix.T @ cotangent

    class WeightedMetric:
        def whiten(self, value):
            return np.sqrt(weights)*value

        def whiten_matrix(self, value):
            return np.sqrt(weights)[:, None]*value

        def apply_precision(self, value):
            return weights*value

    expected = np.linalg.solve(matrix.T @ (weights[:, None]*matrix)+alpha*gram,
                               matrix.T @ (weights*(observed-offset)))
    assert np.all(expected > 0)
    result = fit(AffinePrediction(), observed, WeightedMetric(), gram, alpha, matrix, offset)
    assert result["accepted"] and result["iterations"] == 0
    np.testing.assert_allclose(result["coefficients"], expected, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(result["prediction"], matrix @ expected+offset,
                               rtol=2e-14, atol=2e-14)
    expected_residual = np.linalg.norm(np.sqrt(weights)*(matrix @ expected+offset-observed))
    np.testing.assert_allclose(result["residual_norm"], expected_residual, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(result["penalty_norm"], np.sqrt(expected @ gram @ expected),
                               rtol=2e-14, atol=2e-14)


def test_projected_semidefinite_gram_preserves_unpenalized_coordinate():
    result = fit(LinearFit(), [1., 2.], IdentityMetric(), np.diag([1., 0.]),
                 .1, np.eye(2), np.zeros(2))
    assert result["accepted"]
    np.testing.assert_allclose(result["coefficients"], [10/11, 2.], rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(result["penalty_norm"], 10/11, rtol=2e-14, atol=2e-14)


def test_projected_roundoff_negative_penalty_has_finite_norm_without_changing_gram():
    gram = np.array([[-np.finfo(float).eps]])
    penalty = QuadraticPenalty([0.], gram, .1)
    np.testing.assert_array_equal(penalty.gram, gram)
    # Допуск допускает малую отрицательную часть; матрица и функционал
    # сохраняются, а норма штрафа принимает предельное значение ноль.
    assert penalty.value([1.]) < 0
    result = fit(LinearFit(), [1.], IdentityMetric(), gram, .1, [[1.]], [0.])
    assert result["accepted"] and result["penalty_norm"] == 0.
    assert all(np.isfinite(result[key]) for key in
               ("residual_norm", "penalty_norm", "stationarity", "complementarity"))
    np.testing.assert_array_equal(gram, [[-np.finfo(float).eps]])


def test_nearly_symmetric_indefinite_gram_fails_before_projected_callbacks():
    delta = 2.**-40
    gram = np.array([[1., -1.-delta], [-1., 1.]])
    # Верхний и нижний треугольники различаются в допуске симметрии,
    # но квадратичная форма на векторе (1, 1) отрицательна вне допуска PSD.
    direction = np.ones(2)
    assert direction @ gram @ direction == -delta
    with pytest.raises(ValueError, match="positive semidefinite"):
        QuadraticPenalty(np.zeros(2), gram, 0.)
    metric = SimpleNamespace(whiten_matrix=lambda _: pytest.fail("indefinite Gram whitened"))
    prediction = SimpleNamespace(predict=lambda _: pytest.fail("indefinite Gram predicted"))
    with pytest.raises(ValueError, match="positive semidefinite"):
        fit(prediction, direction, metric, gram, 0., np.eye(2), np.zeros(2))


def test_nearly_symmetric_positive_gram_is_accepted_without_replacement():
    delta = 2.**-40
    gram = np.array([[2., 1.+delta], [1., 2.]])
    original = gram.copy()
    penalty = QuadraticPenalty(np.zeros(2), gram, .1)
    np.testing.assert_array_equal(penalty.gram, original)
    result = fit(LinearFit(), [1., 1.], IdentityMetric(), gram, 0., np.eye(2), np.zeros(2))
    assert result["accepted"]
    np.testing.assert_allclose(result["coefficients"], [1., 1.], rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(result["penalty_norm"], np.sqrt(6.+delta), rtol=2e-14, atol=2e-14)
    np.testing.assert_array_equal(gram, original)


def test_tikhonov_feasibility_tolerance_changes_acceptance(monkeypatch):
    # Заданная неточность решателя делает различие проверяемым: дефект допустимости равен 1e-9, а
    # независимый допуск KKT задан менее строгим.
    def inaccurate(matrix, rhs, **options):
        return SimpleNamespace(x=np.array([-1e-9]), nit=1, status=1, success=True)
    monkeypatch.setattr(tikhonov, "lsq_linear", inaccurate)
    strict = tikhonov.PositiveTikhonov([[1.]], [[1.]], [[1.]], 1., [0.],
                                     kkt_tolerance=1., feasibility_tolerance=1e-10)
    relaxed = tikhonov.PositiveTikhonov([[1.]], [[1.]], [[1.]], 1., [0.],
                                      kkt_tolerance=1., feasibility_tolerance=1e-8)
    assert not strict.solve([-1.]).success and relaxed.solve([-1.]).success


def test_tikhonov_passes_explicit_limits_to_the_numerical_solver(monkeypatch):
    calls = []
    numerical = tikhonov.lsq_linear
    def record(matrix, rhs, **options):
        calls.append(options)
        return numerical(matrix, rhs, **options)
    monkeypatch.setattr(tikhonov, "lsq_linear", record)
    prepared = tikhonov.PositiveTikhonov(np.eye(2), np.eye(2), np.eye(2), 1., [0., 0.],
        solve_tolerance=1e-8, feasibility_tolerance=1e-9, max_iterations=7)
    result = prepared.solve([-1., 4.])
    assert result.success
    np.testing.assert_allclose(result.theta, [0., 2.], atol=1e-13)
    assert calls[0]["max_iter"] == 7 and calls[0]["tol"] == 1e-8


@pytest.mark.parametrize("option,value", [
    ("interior_points", 3.9), ("interior_points", (3, 4.5)),
    ("interior_points", True), ("interior_points", (3, 4, 5)),
    ("time_steps", 2.9), ("time_steps", True), ("time_steps", "3"),
])
def test_grid_counts_reject_silent_rounding_before_coefficient_sampling(option, value):
    from adrkit.coefficients import ScalarField
    model = dict(diffusion=ScalarField(lambda xy, t: pytest.fail("invalid grid sampled")),
                 velocity=(0., 0.), linear_loss=0., horizon=.5, source_position=None)
    settings = dict(interior_points=3, time_steps=3)
    settings[option] = value
    with pytest.raises(ValueError):
        BoundedSolver(model, **settings)


@pytest.mark.parametrize("option,value", [
    ("max_fixed_point_iterations", 2.9), ("max_fixed_point_iterations", True),
    ("max_fixed_point_iterations", 0), ("residual_tolerance", True),
    ("residual_tolerance", np.nan),
])
def test_cached_controls_fail_before_spatial_operator_construction(monkeypatch, option, value):
    monkeypatch.setattr(BoundedSolver, "__init__", lambda *a, **kw: pytest.fail("invalid controls constructed grid"))
    with pytest.raises(ValueError):
        CachedSolver({}, **{option: value})


def test_cached_inherited_controls_dispatch_to_its_own_time_step_method():
    solver = CachedSolver.__new__(CachedSolver)
    solver.times, solver.dt = np.array([0., .25, .5]), .25
    solver.load = np.array([2., 4.])
    contributions = []
    trajectory = object()

    def solve_steps(contribution):
        contributions.extend(contribution(n) for n in (1, 2))
        return trajectory

    solver._solve_steps = solve_steps
    assert solver.solve_controls([999., 1., 3.]) is trajectory
    np.testing.assert_array_equal(contributions, [[.5, 1.], [1.5, 3.]])
    assert CachedSolver.solve_controls is BoundedSolver.solve_controls


@pytest.mark.parametrize("controls", [[1., 2.], [1., np.nan, 2.], [1., 2.+0j, 3.]])
def test_cached_inherited_controls_reject_invalid_history_before_time_steps(controls):
    solver = CachedSolver.__new__(CachedSolver)
    solver.times, solver.load = np.array([0., .25, .5]), np.ones(2)
    solver._solve_steps = lambda _: pytest.fail("invalid control history reached time steps")
    with pytest.raises(ValueError):
        solver.solve_controls(controls)


def test_cached_inherited_controls_require_a_configured_point_source():
    solver = CachedSolver.__new__(CachedSolver)
    solver.load = None
    solver._solve_steps = lambda _: pytest.fail("missing source reached time steps")
    with pytest.raises(CapabilityError, match="No point source"):
        solver.solve_controls([1.])


@pytest.mark.parametrize("backend", [BoundedSolver, CachedSolver])
def test_numpy_integer_counts_preserve_rectangular_forward_solution(backend):
    model = dict(diffusion=.1, velocity=(.2, -.1), linear_loss=.03,
                 horizon=.5, source_position=None)
    ordinary = backend(model, interior_points=(3, 5), time_steps=3)
    declared = backend(model, interior_points=np.array([3, 5]), time_steps=np.int64(3))
    forcing = np.linspace(.1, .8, (ordinary.nt+1)*ordinary.size).reshape(ordinary.nt+1, ordinary.size)
    first, second = ordinary.solve_forcing(forcing), declared.solve_forcing(forcing)
    np.testing.assert_array_equal(first.states, second.states)
    assert first.max_scaled_residual == second.max_scaled_residual
    assert first.nonlinear_iterations == second.nonlinear_iterations


def test_projected_fit_owns_returned_arrays_independently_of_model_and_input():
    class SharedOutput(LinearFit):
        def __init__(self):
            self.value = np.zeros(2)

        def predict(self, point):
            self.value[:] = point
            return self.value

    model = SharedOutput()
    initial = np.array([8., 5.])
    original = initial.copy()
    result = fit(model, np.array([2., 4.]), IdentityMetric(), np.eye(2), 1.,
                 np.eye(2), np.zeros(2), initial_point=initial)
    assert result["accepted"]
    for field in ("coefficients", "prediction"):
        assert isinstance(result[field], np.ndarray)
        assert not np.shares_memory(result[field], model.value)
        assert not np.shares_memory(result[field], initial)
    assert not np.shares_memory(result["coefficients"], result["prediction"])
    prediction = result["prediction"].copy()
    model.value[:] = 999.
    result["coefficients"][:] = -1.
    np.testing.assert_array_equal(result["prediction"], prediction)
    np.testing.assert_array_equal(initial, original)


def test_failed_backtracking_restores_prediction_at_returned_point():
    class ExponentialPrediction:
        def __init__(self):
            self.points = []
            self.current_point = None

        def predict(self, point):
            self.points.append(point.copy())
            self.current_point = point.copy()
            return np.exp(point)

        def vjp(self, point, cotangent):
            return np.exp(point)*cotangent

    model = ExponentialPrediction()
    initial = np.zeros(1)
    result = fit(model, np.exp(np.ones(1)), IdentityMetric(), np.eye(1), .01,
        np.eye(1), np.ones(1), initial_point=initial, max_backtracks=1)
    assert result["status"] == "line_search_failed" and not result["accepted"]
    assert len(model.points) == 3
    assert model.points[-2][0] > 1.
    np.testing.assert_array_equal(model.points[0], initial)
    np.testing.assert_array_equal(model.points[-1], result["coefficients"])
    np.testing.assert_array_equal(model.current_point, result["coefficients"])
    np.testing.assert_array_equal(result["prediction"], np.exp(initial))
