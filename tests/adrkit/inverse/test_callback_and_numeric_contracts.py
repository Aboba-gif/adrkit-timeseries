"""Изоляция внешних операций и явные условия совместного вычисления скалярного функционала."""
from types import SimpleNamespace

import numpy as np
import pytest

from adrkit.contracts import MapCapabilities, ScalarCapabilities
from adrkit.inverse import optimizers
from adrkit.inverse.functionals import scalar_from_functions, scale_functional, sum_functionals, pullback_functional
from adrkit.inverse.misfit import covariance_metric, DiagonalWorkingMetric, SupportWhitening, QuadraticMisfit, on_support
from adrkit.inverse.regularization import QuadraticPenalty
from adrkit.inverse.tikhonov import PositiveTikhonov
from adrkit.inverse.objectives import compose_maps, make_objective
from adrkit.inverse.sensitivity import matrix_spectrum
from adrkit.spaces import ArraySpace, matrix_map


def space(dtype=np.float64):
    return ArraySpace((2,), axes=("parameter",), coordinates=(("a", "b"),),
                      units="1", dtype=dtype)


def one_evaluation(monkeypatch, *, expected_value, expected_gradient):
    """Подставить одно вычисление функционала вместо оптимизации."""
    def minimize(function, initial, **options):
        original = initial.copy()
        value, gradient = function(initial)
        np.testing.assert_array_equal(initial, original)
        assert value == expected_value
        np.testing.assert_array_equal(gradient, expected_gradient)
        return SimpleNamespace()
    monkeypatch.setattr(optimizers, "minimize", minimize)


def test_separate_external_callbacks_cannot_mutate_optimizer_or_each_other(monkeypatch):
    class Operation:
        def value(self, point):
            np.testing.assert_array_equal(point, [1.0, 2.0])
            point[:] = 99
            return 2.5

        def cotangent(self, point):
            np.testing.assert_array_equal(point, [1.0, 2.0])
            result = point.copy()
            point[:] = -99
            return result

    one_evaluation(monkeypatch, expected_value=2.5, expected_gradient=[1.0, 2.0])
    operation = Operation()
    initial = np.array([1.0, 2.0])
    optimizers.solve_lbfgsb(ScalarCapabilities(operation, space(), operation), initial)
    np.testing.assert_array_equal(initial, [1.0, 2.0])


def test_joint_external_callback_is_explicit_and_owns_input(monkeypatch):
    class Operation:
        def value(self, point):
            raise AssertionError("declared joint operation must be used")

        def cotangent(self, point):
            raise AssertionError("declared joint operation must be used")

        def value_and_cotangent(self, point):
            value, covector = .5 * float(point @ point), point.copy()
            point[:] = 99
            return value, covector

    one_evaluation(monkeypatch, expected_value=2.5, expected_gradient=[1.0, 2.0])
    operation = Operation()
    objective = ScalarCapabilities(operation, space(), operation,
                                   value_and_cotangent=operation)
    optimizers.solve_lbfgsb(objective, [1.0, 2.0])


def test_joint_method_is_never_inferred(monkeypatch):
    class Operation:
        def value(self, point):
            return .5 * float(point @ point)

        def cotangent(self, point):
            return point

        def value_and_cotangent(self, point):
            raise AssertionError("undeclared joint operation must not be discovered")

    one_evaluation(monkeypatch, expected_value=2.5, expected_gradient=[1.0, 2.0])
    operation = Operation()
    optimizers.solve_lbfgsb(ScalarCapabilities(operation, space(), operation), [1.0, 2.0])


@pytest.mark.parametrize("malformed", ["value", "cotangent", "value_and_cotangent"])
def test_malformed_operation_rejected_before_optimizer_and_value(monkeypatch, malformed):
    class Forbidden:
        def value(self, point):
            raise AssertionError("must reject before value")

        def cotangent(self, point):
            raise AssertionError("must reject before derivative")

        def value_and_cotangent(self, point):
            raise AssertionError("must reject before joint evaluation")

    def forbidden_minimize(*args, **kwargs):
        raise AssertionError("must reject before optimizer")
    monkeypatch.setattr(optimizers, "minimize", forbidden_minimize)
    operation = Forbidden()
    kwargs = {"value": operation, "domain": space(), "cotangent": operation,
              "value_and_cotangent": operation}
    kwargs[malformed] = object()
    with pytest.raises(TypeError, match="callable"):
        optimizers.solve_lbfgsb(ScalarCapabilities(**kwargs), [1.0, 2.0])


@pytest.mark.parametrize("bad_value", [[1.0], complex(1, 2), True, "1", np.inf, np.nan])
def test_optimizer_rejects_nonreal_or_nonscalar_values(monkeypatch, bad_value):
    class Operation:
        def value(self, point):
            return bad_value

        def cotangent(self, point):
            return point.copy()

    def evaluate(function, initial, **options):
        return function(initial)
    monkeypatch.setattr(optimizers, "minimize", evaluate)
    operation = Operation()
    with pytest.raises(ValueError, match="objective"):
        optimizers.solve_lbfgsb(ScalarCapabilities(operation, space(), operation), [1.0, 2.0])


@pytest.mark.parametrize("kind", ["covariance", "working"])
@pytest.mark.parametrize("bad_matrix", [
    np.array([[1 + 2j], [3 + 4j]]), np.array([["1"], ["3"]]),
    np.array([[True], [False]]), np.array([[np.nan], [3.0]]), np.array([[np.inf], [3.0]]),
])
def test_whitening_rejects_nonreal_or_nonfinite_matrices(kind, bad_matrix):
    domain = space()
    metric = (covariance_metric(np.eye(2), layout=domain) if kind == "covariance" else
              DiagonalWorkingMetric([1.0, 2.0], layout=domain))
    with pytest.raises(ValueError, match="finite real"):
        metric.whiten_matrix(bad_matrix)


@pytest.mark.parametrize("kind", ["covariance", "working"])
def test_whitening_rejects_overflow_in_layout_dtype(kind):
    domain = space(np.float32)
    metric = (covariance_metric(np.eye(2), layout=domain) if kind == "covariance" else
              DiagonalWorkingMetric([1.0, 2.0], layout=domain))
    with pytest.raises(ValueError, match="remain finite"):
        metric.whiten_matrix(np.array([[1e100], [3.0]], dtype=np.float64))


def test_joint_functional_algebra_and_optimizer_match_independent_quadratic():
    domain = space()
    calls = []
    def joint(point):
        calls.append(point.copy())
        return .5 * float(point @ point), point.copy()
    functional = scalar_from_functions(domain=domain, value=lambda x: .5 * float(x @ x),
                                       cotangent=lambda x: x.copy(), value_and_cotangent=joint)
    matrix = np.array([[1.0, .3], [-.2, 2.0]])
    mapping = matrix_map(matrix, domain=domain, codomain=domain)
    assembled = pullback_functional(scale_functional(sum_functionals(functional, functional), 3), mapping)
    point = np.array([1.0, 2.0])
    value, covector = assembled.value_and_cotangent.value_and_cotangent(point)
    np.testing.assert_allclose(value, 3 * np.linalg.norm(matrix @ point)**2)
    np.testing.assert_allclose(covector, 6 * matrix.T @ matrix @ point)
    np.testing.assert_array_equal(point, [1.0, 2.0])
    calls.clear()
    solved = optimizers.solve_lbfgsb(assembled, point, options={"gtol": 1e-12, "ftol": 1e-15})
    assert solved.success
    np.testing.assert_allclose(solved.x, [0.0, 0.0], atol=1e-9)
    assert len(calls) == 2 * solved.nfev


def test_joint_evaluation_cannot_create_missing_cotangent():
    with pytest.raises(ValueError, match="explicitly declared cotangent"):
        scalar_from_functions(domain=space(), value=lambda x: 0.0,
                              value_and_cotangent=lambda x: (0.0, x))


class MutatingIdentity:
    def predict(self, point):
        result = point.copy()
        point[:] = 99
        return result

    def jvp(self, point, direction):
        np.testing.assert_array_equal(point, [1.0, 2.0])
        result = direction.copy()
        point[:] = direction[:] = 99
        return result

    def vjp(self, point, covector):
        np.testing.assert_array_equal(point, [1.0, 2.0])
        result = covector.copy()
        point[:] = covector[:] = 99
        return result


def test_composed_maps_isolate_trace_and_derivative_arguments():
    domain = space()
    operation = MutatingIdentity()
    mapping = MapCapabilities(operation, domain, domain, operation, operation)
    composed = compose_maps(mapping, mapping)
    point, direction, covector = (np.array([1.0, 2.0]), np.array([.3, -.2]), np.array([.7, .1]))
    np.testing.assert_array_equal(composed.predict.predict(point), point)
    np.testing.assert_array_equal(composed.jvp.jvp(point, direction), direction)
    np.testing.assert_array_equal(composed.vjp.vjp(point, covector), covector)
    np.testing.assert_array_equal(point, [1.0, 2.0])
    np.testing.assert_array_equal(direction, [.3, -.2])
    np.testing.assert_array_equal(covector, [.7, .1])


@pytest.mark.parametrize("joint", [False, True])
def test_objective_isolates_prediction_loss_penalty_and_derivative_arguments(joint):
    domain = space()
    shared_scalar = np.zeros(())
    class Scalar:
        def __init__(self, weight):
            self.weight = weight

        def value(self, point):
            np.testing.assert_array_equal(point, [1.0, 2.0])
            shared_scalar[()] = .5 * self.weight * float(point @ point)
            point[:] = 99
            return shared_scalar

        def cotangent(self, point):
            np.testing.assert_array_equal(point, [1.0, 2.0])
            result = self.weight * point
            point[:] = 99
            return result

        def value_and_cotangent(self, point):
            return self.value(point.copy()), self.cotangent(point.copy())

    prediction_operation = MutatingIdentity()
    prediction = MapCapabilities(prediction_operation, domain, domain,
                                 prediction_operation, prediction_operation)
    loss, penalty = Scalar(1), Scalar(2)
    loss_caps = ScalarCapabilities(loss, domain, loss, value_and_cotangent=loss if joint else None)
    penalty_caps = ScalarCapabilities(penalty, domain, penalty,
                                     value_and_cotangent=penalty if joint else None)
    objective = make_objective(prediction, None, loss_caps, penalty_caps)
    point = np.array([1.0, 2.0])
    value, covector = objective.value_and_cotangent.value_and_cotangent(point)
    assert value == 7.5
    np.testing.assert_array_equal(covector, [3.0, 6.0])
    assert objective.value.value(point) == 7.5
    np.testing.assert_array_equal(objective.cotangent.cotangent(point), [3.0, 6.0])
    np.testing.assert_array_equal(point, [1.0, 2.0])


def test_functional_sum_snapshots_shared_joint_result_buffers():
    domain = space()
    shared_covector, shared_scalar = np.zeros(2), np.zeros(())
    class Scalar:
        def __init__(self, weight):
            self.weight = weight

        def value(self, point):
            return .5 * self.weight * float(point @ point)

        def cotangent(self, point):
            return self.weight * point

        def value_and_cotangent(self, point):
            shared_covector[:] = self.weight * point
            shared_scalar[()] = self.value(point)
            return shared_scalar, shared_covector

    first, second = Scalar(1), Scalar(2)
    summed = sum_functionals(
        ScalarCapabilities(first, domain, first, value_and_cotangent=first),
        ScalarCapabilities(second, domain, second, value_and_cotangent=second),
    )
    value, covector = summed.value_and_cotangent.value_and_cotangent([1.0, 2.0])
    assert value == 7.5
    np.testing.assert_array_equal(covector, [3.0, 6.0])


@pytest.mark.parametrize("matrix", [
    np.array([[1 + 2j, 0], [0, 3 + 4j]]), np.array([[True, False], [False, True]]),
    np.array([["1", "0"], ["0", "3"]]), np.array([[np.nan, 0], [0, 3.0]]),
    np.array([[np.inf, 0], [0, 3.0]]), np.array([1.0, 3.0]),
])
def test_matrix_spectrum_rejects_invalid_real_contract(matrix):
    with pytest.raises(ValueError, match="finite real"):
        matrix_spectrum(matrix, np.eye(2), np.eye(2))


@pytest.mark.parametrize("operation", ["jvp", "vjp"])
def test_composition_rejects_invalid_derivative_arguments_before_prediction(operation):
    class ForbiddenIdentity:
        def predict(self, point):
            raise AssertionError("must reject invalid derivative before prediction")

        def jvp(self, point, direction):
            raise AssertionError("must reject invalid direction before derivative")

        def vjp(self, point, covector):
            raise AssertionError("must reject invalid covector before derivative")
    domain = space()
    identity = ForbiddenIdentity()
    mapping = MapCapabilities(identity, domain, domain, identity, identity)
    composed = compose_maps(mapping, mapping)
    with pytest.raises(ValueError, match="shape"):
        if operation == "jvp":
            composed.jvp.jvp([1.0, 2.0], [1.0, 2.0, 3.0])
        else:
            composed.vjp.vjp([1.0, 2.0], [1.0, 2.0, 3.0])


def test_optimizer_snapshots_scalar_before_derivative_reuses_buffer(monkeypatch):
    class SharedScalar:
        scalar = np.zeros(())
        def value(self, point):
            self.scalar[()] = .5 * float(point @ point)
            return self.scalar

        def cotangent(self, point):
            self.scalar[()] = 99
            return point.copy()
    one_evaluation(monkeypatch, expected_value=2.5, expected_gradient=[1.0, 2.0])
    operation = SharedScalar()
    optimizers.solve_lbfgsb(ScalarCapabilities(operation, space(), operation), [1.0, 2.0])


@pytest.mark.parametrize("field", ["A", "Sigma", "M", "theta0", "alpha", "kkt_tolerance"])
@pytest.mark.parametrize("kind", ["complex", "string", "boolean", "object"])
def test_prepared_tikhonov_rejects_unsupported_values_before_cast(field, kind):
    options = dict(A=np.eye(2), Sigma=np.eye(2), M=np.eye(2), alpha=1.0,
                   theta0=np.zeros(2), kkt_tolerance=1e-7)
    dtypes = {"complex": complex, "string": str, "boolean": bool, "object": object}
    options[field] = np.asarray(options[field], dtype=dtypes[kind])
    if kind == "complex":
        options[field] += 1j
    with pytest.raises(ValueError, match="real"):
        PositiveTikhonov(**options)


@pytest.mark.parametrize("field", ["lift", "variance", "values"])
@pytest.mark.parametrize("kind", ["complex", "string", "boolean", "object"])
def test_support_whitening_rejects_unsupported_values_before_cast(field, kind):
    lift, variance, values = np.eye(2), np.ones(2), np.array([1.0, 2.0])
    invalid = np.asarray({"lift": lift, "variance": variance, "values": values}[field],
                         dtype={"complex": complex, "string": str, "boolean": bool, "object": object}[kind])
    if kind == "complex":
        invalid += 1j
    with pytest.raises(ValueError, match="real"):
        if field == "values":
            SupportWhitening(lift, variance).whiten(invalid)
        else:
            SupportWhitening(invalid if field == "lift" else lift,
                             invalid if field == "variance" else variance)


@pytest.mark.parametrize("field", ["penalty_alpha", "normalization"])
@pytest.mark.parametrize("bad", [1j, "1", True, np.array(1.0, dtype=object), np.inf, np.nan, [1.0]])
def test_current_quadratic_scalar_parameters_are_real_finite_scalars(field, bad):
    with pytest.raises(ValueError, match="finite real scalar"):
        if field == "penalty_alpha":
            QuadraticPenalty(np.zeros(2), np.eye(2), bad)
        else:
            QuadraticMisfit(np.zeros(2), covariance_metric(np.eye(2), layout=space()), normalization=bad)


def test_prepared_tikhonov_and_support_whitening_match_independent_algebra():
    matrix, prior = np.eye(2), np.zeros(2)
    prepared = PositiveTikhonov(matrix, np.eye(2), np.eye(2), 1.0, prior)
    matrix[:] = prior[:] = 99
    solved = prepared.solve(np.array([2.0, 4.0]))
    assert solved.success
    np.testing.assert_allclose(solved.theta, [1.0, 2.0], rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(solved.objective, 5.0, rtol=1e-14, atol=1e-14)
    lift = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    variance = np.array([2.0, 3.0])
    whitener = SupportWhitening(lift, variance)
    transformed = whitener.whiten_matrix(lift @ np.diag(np.sqrt(variance)))
    np.testing.assert_allclose(transformed @ transformed.T, np.eye(2), atol=1e-14)


@pytest.mark.parametrize("tolerance", [np.nan, np.inf, -np.inf, -0.1, True, 1j, "1e-10", [1e-10], None])
def test_on_support_requires_a_finite_nonnegative_scalar_tolerance(tolerance):
    with pytest.raises(ValueError):
        on_support([1., 0.], [[1.], [0.]], tolerance)


@pytest.mark.parametrize("values", [
    np.array([0., 1e200]),
    np.array([[0., 0.], [1e200, -1e200]]),
])
def test_on_support_rejects_large_residuals_outside_the_subspace(values):
    with np.errstate(over="raise", invalid="raise"):
        with pytest.raises(ValueError, match="outside"):
            on_support(values, [[1.], [0.]])


@pytest.mark.parametrize("values", [
    np.array([1e200, 0.]),
    np.array([[1e200, -1e200], [0., 0.]]),
])
def test_on_support_retains_large_values_in_the_subspace(values):
    original = values.copy()
    with np.errstate(over="raise", invalid="raise"):
        result = on_support(values, [[1.], [0.]])
    assert result.shape == original.shape and result.dtype == original.dtype
    assert result.tobytes() == original.tobytes()
    assert values.tobytes() == original.tobytes()


@pytest.mark.parametrize("values", [np.array([1., 1e-200]), np.array([[0.], [1e-200]])])
def test_on_support_distinguishes_a_small_residual_from_the_requested_tolerance(values):
    with pytest.raises(ValueError, match="outside"):
        on_support(values, [[1.], [0.]], tolerance=1e-210)
    np.testing.assert_array_equal(on_support(values, [[1.], [0.]], tolerance=1e-190), values)


@pytest.mark.parametrize("scale", [1., 1e150])
def test_on_support_applies_the_same_relative_tolerance_at_different_scales(scale):
    values = scale * np.array([2., 1e-7])
    with pytest.raises(ValueError, match="outside"):
        on_support(values, [[1.], [0.]], tolerance=1e-8)
    np.testing.assert_array_equal(on_support(values, [[1.], [0.]], tolerance=1e-6), values)


@pytest.mark.parametrize("values", [
    np.array([1e200, 1e-200]),
    np.array([[1e200, -1e200], [1e-200, 0.]]),
])
def test_on_support_zero_tolerance_rejects_a_nonzero_residual_with_mixed_scales(values):
    with pytest.raises(ValueError, match="outside"):
        on_support(values, [[1.], [0.]], tolerance=0.)


@pytest.mark.parametrize("values", [np.array([1., 0.]), np.array([[2., -3.], [0., 0.]])])
def test_on_support_zero_tolerance_retains_an_exactly_represented_projection(values):
    np.testing.assert_array_equal(on_support(values, [[1.], [0.]], tolerance=0.), values)


def test_on_support_accepts_an_empty_matrix_of_columns():
    values = np.empty((2, 0))
    assert on_support(values, [[1.], [0.]]).shape == (2, 0)
