"""Сборка параметрической задачи, аналитические решения и проверка параметров до расчёта."""

import numpy as np
import pytest

from adrkit.contracts import MapCapabilities, ScalarCapabilities
from adrkit.errors import CapabilityError
from adrkit.inverse import InverseProblem
from adrkit.inverse.constraints import (
    analyze_nonnegative_observations,
    project_nonnegative_equalities,
)
from adrkit.selection import Selection
from adrkit.spaces import ArraySpace


def space(shape=(2, 2), *, units="1", binding="parameters", coordinates=None):
    return ArraySpace(
        shape,
        axes=tuple(f"axis-{i}" for i in range(len(shape))),
        coordinates=coordinates or tuple(tuple(range(n)) for n in shape),
        units=units,
        binding=binding,
    )


class MatrixForward:
    r"""Матричное отображение массивов с сопряжённой производной.

    Parameters
    ----------
    matrix : array_like of shape (codomain.size, domain.size)
        Матрица отображения; значения копируются.
    domain : ArraySpace
        Пространство входного массива.
    codomain : ArraySpace
        Пространство выходного массива.

    Notes
    -----
    Массивы разворачиваются в порядке C. Для матрицы :math:`A`
    прямое действие и сопряжённая производная равны

    .. math::

        y=A\,\operatorname{vec}_C(x),\qquad
        J^\mathsf{T}z=A^\mathsf{T}\operatorname{vec}_C(z).

    Результаты приводятся к форме соответствующего пространства.
    """

    def __init__(self, matrix, domain, codomain):
        self.matrix = np.array(matrix, dtype=float, copy=True)
        self.domain, self.codomain = domain, codomain
        self.calls = 0

    def predict(self, x):
        self.calls += 1
        return (self.matrix @ x.ravel()).reshape(self.codomain.shape)

    def vjp(self, at, cotangent):
        return (self.matrix.T @ cotangent.ravel()).reshape(self.domain.shape)

    def capabilities(self):
        # Гладкому оптимизатору нужен VJP; наличие JVP не требуется.
        return MapCapabilities(self, self.domain, self.codomain, vjp=self)


class SquaredDistance:
    r"""Квадратичная потеря с заданной симметричной матрицей весов.

    Parameters
    ----------
    target : array_like
        Целевой массив в форме ``domain.shape``.
    weight : array_like of shape (domain.size, domain.size)
        Симметричная матрица весов; значения копируются.
    domain : ArraySpace
        Пространство аргумента потери.

    Notes
    -----
    При :math:`d=\operatorname{vec}_C(x-y)` значение и координатный
    ковектор равны

    .. math::

        L(x)=\frac12d^\mathsf{T}Wd,\qquad dL(x)=Wd.

    Ковектор возвращается в форме ``domain.shape``.
    """

    def __init__(self, target, weight, domain):
        self.target = domain.array(target, copy=True)
        self.weight = np.array(weight, dtype=float, copy=True)
        self.domain = domain

    def value(self, x):
        d = (x - self.target).ravel()
        return 0.5 * d @ self.weight @ d

    def cotangent(self, x):
        return (self.weight @ (x - self.target).ravel()).reshape(self.domain.shape)

    def capabilities(self):
        return ScalarCapabilities(self, self.domain, self)


class Forbidden:
    def predict(self, x):
        raise AssertionError("construction/preflight must not evaluate forward")

    def vjp(self, at, cotangent):
        raise AssertionError("preflight must not evaluate VJP")

    def value(self, x):
        raise AssertionError("construction/preflight must not evaluate scalar")

    def cotangent(self, x):
        raise AssertionError("preflight must not evaluate cotangent")


def quadratic_case():
    parameters, data = space(), space((5,), binding="data")
    matrix = np.array(
        [
            [1.0, 2.0, 0.0, -1.0],
            [0.0, 1.0, 3.0, 0.0],
            [2.0, 0.0, 1.0, 1.0],
            [-1.0, 0.0, 0.0, 2.0],
            [1.0, -1.0, 1.0, 0.0],
        ]
    )
    observed = np.array([2.0, -1.0, 3.0, 0.5, -2.0])
    weights = np.diag([1.0, 2.0, 3.0, 1.5, 0.5])
    prior = np.array([[0.2, -0.3], [0.1, 0.4]])
    gram = np.array(
        [[2.0, 0.3, 0.0, 0.0], [0.3, 1.0, 0.0, 0.0], [0.0, 0.0, 3.0, -0.2], [0.0, 0.0, -0.2, 2.0]]
    )
    penalty_weight = 0.7 * gram
    forward = MatrixForward(matrix, parameters, data)
    loss = SquaredDistance(observed, weights, data)
    penalty = SquaredDistance(prior, penalty_weight, parameters)
    # Нормальные уравнения получены независимо от производной реализации.
    exact = np.linalg.solve(
        matrix.T @ weights @ matrix + penalty_weight,
        matrix.T @ weights @ observed + penalty_weight @ prior.ravel(),
    )
    return parameters, forward, loss, penalty, exact


def test_tensor_parameters_match_independent_normal_equations():
    parameters, forward, loss, penalty, exact = quadratic_case()
    problem = InverseProblem(
        forward.capabilities(), loss.capabilities(), penalty=penalty.capabilities()
    )
    assert forward.calls == 0
    assert problem.parameter_space == parameters
    assert problem.objective.domain == parameters
    assert problem.forward.vjp is forward
    result = problem.solve(np.zeros(parameters.shape), options={"gtol": 1e-10, "ftol": 1e-14})
    assert result.success, result.message
    np.testing.assert_allclose(result.x, exact, atol=2e-8, rtol=2e-8)
    assert result.x.shape == (4,)
    assert np.any(result.x < 0)  # Неотрицательность не вводится как неявное ограничение.
    np.testing.assert_allclose(
        problem.cotangent(result.x.reshape(parameters.shape)), 0.0, atol=2e-7
    )
    np.testing.assert_allclose(
        problem.predict(exact.reshape(parameters.shape)), forward.matrix @ exact
    )


def test_explicit_box_bounds_have_independent_clipped_solution_and_are_owned():
    domain = space()
    forward = MatrixForward(np.diag([1.0, 2.0, 3.0, 4.0]), domain, domain)
    target = np.array([[-3.0, 6.0], [9.0, -8.0]])
    loss = SquaredDistance(target, np.eye(4), domain)
    bounds = [[-1.0, 2.0], [None, 1.0], [-2.0, 2.5], [-1.5, None]]
    problem = InverseProblem(forward.capabilities(), loss.capabilities(), bounds=bounds)
    bounds[0][0] = 100.0  # Численные границы копируются при сборке задачи.
    assert problem.bounds[0] == (-1.0, 2.0)
    result = problem.solve(np.zeros(domain.shape), options={"gtol": 1e-10})
    assert result.success, result.message
    np.testing.assert_allclose(result.x, [-1.0, 1.0, 2.5, -1.5], atol=1e-8)


def test_composed_user_parameterization_matches_transformed_normal_equations():
    parameters, forward, loss, penalty, _ = quadratic_case()
    latent = space(binding="latent")
    transform = np.array(
        [[1.0, 0.5, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 1.0, -0.2], [0.0, 0.0, 0.0, 0.5]]
    )
    parameterization = MatrixForward(transform, latent, parameters)
    prior = SquaredDistance(np.zeros(latent.shape), 0.2 * np.eye(4), latent)
    problem = InverseProblem.from_maps(
        parameterization.capabilities(),
        forward.capabilities(),
        loss=loss.capabilities(),
        penalty=prior.capabilities(),
    )
    assert parameterization.calls == forward.calls == 0
    matrix = forward.matrix @ transform
    exact = np.linalg.solve(
        matrix.T @ loss.weight @ matrix + 0.2 * np.eye(4), matrix.T @ loss.weight @ loss.target
    )
    result = problem.solve(np.zeros(latent.shape), options={"gtol": 1e-10, "ftol": 1e-14})
    assert result.success, result.message
    np.testing.assert_allclose(result.x, exact, atol=3e-7, rtol=3e-7)


@pytest.mark.parametrize("missing", ["VJP", "loss", "penalty"])
@pytest.mark.parametrize("method", ["solve", "cotangent"])
def test_missing_derivative_rejected_before_any_computation(missing, method):
    domain, operation = space(), Forbidden()
    forward = MapCapabilities(
        operation, domain, domain, vjp=None if missing == "VJP" else operation
    )
    loss = ScalarCapabilities(operation, domain, None if missing == "loss" else operation)
    penalty = ScalarCapabilities(operation, domain, None if missing == "penalty" else operation)
    problem = InverseProblem(forward, loss, penalty=penalty)
    with pytest.raises(CapabilityError, match=missing):
        getattr(problem, method)(np.zeros(domain.shape))


def test_value_only_problem_is_allowed_but_does_not_acquire_a_derivative():
    domain = space()
    operation = MatrixForward(np.eye(4), domain, domain)
    loss = SquaredDistance(np.ones(domain.shape), np.eye(4), domain)
    problem = InverseProblem(
        MapCapabilities(operation, domain, domain), ScalarCapabilities(loss, domain)
    )
    assert problem.value(np.zeros(domain.shape)) == 2.0
    assert problem.objective.cotangent is None
    assert operation.calls == 1


@pytest.mark.parametrize("difference", ["coordinates", "units", "binding"])
@pytest.mark.parametrize("component", ["loss", "penalty", "composition"])
def test_equal_shape_but_different_layout_is_rejected_without_prediction(difference, component):
    domain, operation = space(), Forbidden()
    options = (
        {"coordinates": ((1, 0), (0, 1))} if difference == "coordinates" else {difference: "other"}
    )
    incompatible = space(**options)
    forward = MapCapabilities(operation, domain, domain, vjp=operation)
    loss = ScalarCapabilities(
        operation, incompatible if component == "loss" else domain, operation
    )
    penalty = ScalarCapabilities(
        operation, incompatible if component == "penalty" else domain, operation
    )
    with pytest.raises(ValueError, match="Incompatible"):
        if component == "composition":
            other = MapCapabilities(operation, incompatible, domain, vjp=operation)
            InverseProblem.from_maps(forward, other, loss=loss)
        else:
            InverseProblem(forward, loss, penalty=penalty)


def test_selection_uses_declared_row_order_and_full_prediction_is_available():
    parameters, data = space(), space((4,), binding="observations")
    forward = MatrixForward(np.eye(4), parameters, data)
    selection = Selection([3, 0], data)
    loss = SquaredDistance(np.array([10.0, 20.0]), np.diag([2.0, 3.0]), selection.selected_layout)
    problem = InverseProblem(forward.capabilities(), loss.capabilities(), selection=selection)
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert problem.value(x) == 0.5 * (2.0 * (4.0 - 10.0) ** 2 + 3.0 * (1.0 - 20.0) ** 2)
    np.testing.assert_array_equal(problem.cotangent(x), [[-57.0, 0.0], [0.0, -12.0]])
    np.testing.assert_array_equal(problem.predict(x), [1.0, 2.0, 3.0, 4.0])


@pytest.mark.parametrize("bounds", [[(0.0, 1.0)], [(2.0, 1.0)] * 4, [(np.nan, 1.0)] * 4])
def test_invalid_bounds_fail_during_inert_construction(bounds):
    domain, operation = space(), Forbidden()
    with pytest.raises(ValueError, match="bound"):
        InverseProblem(
            MapCapabilities(operation, domain, domain, vjp=operation),
            ScalarCapabilities(operation, domain, operation),
            bounds=bounds,
        )


@pytest.mark.parametrize("method", ["predict", "value", "cotangent", "solve"])
def test_wrong_parameter_shape_fails_before_prediction(method):
    domain, operation = space(), Forbidden()
    problem = InverseProblem(
        MapCapabilities(operation, domain, domain, vjp=operation),
        ScalarCapabilities(operation, domain, operation),
    )
    with pytest.raises(ValueError, match="shape"):
        getattr(problem, method)(np.zeros(4))


def test_runtime_output_layout_is_checked():
    domain = space()

    class WrongOutput:
        def predict(self, x):
            return np.zeros(4)

    scalar = SquaredDistance(np.zeros(domain.shape), np.eye(4), domain)
    problem = InverseProblem(MapCapabilities(WrongOutput(), domain, domain), scalar.capabilities())
    with pytest.raises(ValueError, match="shape"):
        problem.predict(np.zeros(domain.shape))
    with pytest.raises(ValueError, match="shape"):
        problem.value(np.zeros(domain.shape))


def test_nonlinear_user_forward_and_nonquadratic_loss_chain_rule():
    domain = space()

    class Exponential:
        def predict(self, x):
            return np.exp(x)

        def vjp(self, at, cotangent):
            return np.exp(at) * cotangent

    class FourthPowerLoss:
        def value(self, y):
            return 0.25 * np.sum((y - 2.0) ** 4)

        def cotangent(self, y):
            return (y - 2.0) ** 3

    forward, loss = Exponential(), FourthPowerLoss()
    penalty = SquaredDistance(np.zeros(domain.shape), 0.3 * np.eye(4), domain)
    problem = InverseProblem(
        MapCapabilities(forward, domain, domain, vjp=forward),
        ScalarCapabilities(loss, domain, loss),
        penalty=penalty.capabilities(),
    )
    x = np.array([[0.2, -0.1], [0.3, 0.5]])
    h = np.array([[0.7, 0.4], [-0.3, 0.1]])
    exact = np.sum((np.exp(x) * (np.exp(x) - 2.0) ** 3 + 0.3 * x) * h)
    np.testing.assert_allclose(np.sum(problem.cotangent(x) * h), exact, rtol=1e-14)
    epsilon = 1e-5
    numerical = (problem.value(x + epsilon * h) - problem.value(x - epsilon * h)) / (2 * epsilon)
    np.testing.assert_allclose(numerical, exact, rtol=1e-9, atol=1e-10)


def test_builtin_capability_adapters_use_the_same_neutral_assembly():
    r"""Проверить сборку встроенных потерь и штрафа по аналитическому решению.

    Notes
    -----
    Для диагональных дисперсий :math:`v_i` и веса штрафа 0.5
    условие стационарности по каждой координате имеет вид

    .. math::

        \frac{x_i-y_i}{v_i}+0.5x_i=0,\qquad
        x_i=\frac{y_i}{1+0.5v_i}.

    При :math:`y=(4,-2)` и :math:`v=(2,1)` решение равно
    :math:`x=(2,-4/3)`.
    """
    from adrkit.inverse import QuadraticMisfit, QuadraticPenalty, covariance_metric

    domain = space((2,))
    forward = MatrixForward(np.eye(2), domain, domain)
    loss = QuadraticMisfit(
        np.array([4.0, -2.0]), covariance_metric(np.array([2.0, 1.0]), layout=domain)
    )
    penalty = QuadraticPenalty(np.zeros(2), np.eye(2), 0.5)
    problem = InverseProblem(
        forward.capabilities(), loss.capabilities(), penalty=penalty.capabilities(domain)
    )
    result = problem.solve(np.zeros(2), options={"gtol": 1e-10, "ftol": 1e-14})
    assert result.success, result.message
    np.testing.assert_allclose(result.x, [2.0, -4.0 / 3.0], atol=2e-8)


def test_zero_dimensional_parameter_layout_is_preserved_at_component_boundary():
    domain = space(())
    forward = MatrixForward([[2.0]], domain, domain)
    loss = SquaredDistance(np.asarray(6.0), np.eye(1), domain)
    problem = InverseProblem(forward.capabilities(), loss.capabilities())
    result = problem.solve(np.asarray(0.0))
    assert result.success, result.message
    np.testing.assert_allclose(result.x, [3.0], atol=1e-8)
    assert problem.predict(np.asarray(3.0)).shape == ()


@pytest.mark.parametrize(
    "component,method",
    [
        ("forward", "predict"),
        ("forward", "jvp"),
        ("forward", "vjp"),
        ("loss", "value"),
        ("loss", "cotangent"),
        ("loss", "prox"),
        ("penalty", "value"),
        ("penalty", "cotangent"),
        ("penalty", "prox"),
    ],
)
@pytest.mark.parametrize("malformed_kind", ["missing_method", "noncallable_method"])
def test_malformed_declared_operation_rejected_during_construction(
    component, method, malformed_kind
):
    from types import SimpleNamespace

    domain, operation = space(), Forbidden()
    malformed = object() if malformed_kind == "missing_method" else SimpleNamespace(**{method: 3})
    forward_options = dict(predict=operation, domain=domain, codomain=domain, vjp=operation)
    scalar_options = {
        name: dict(value=operation, domain=domain, cotangent=operation)
        for name in ("loss", "penalty")
    }
    if component == "forward":
        forward_options[method] = malformed
    else:
        scalar_options[component][method] = malformed
        if method == "prox":
            scalar_options[component]["prox_pairing"] = "euclidean"
    with pytest.raises(TypeError, match=f"{component}.*callable {method}"):
        InverseProblem(
            MapCapabilities(**forward_options),
            ScalarCapabilities(**scalar_options["loss"]),
            penalty=ScalarCapabilities(**scalar_options["penalty"]),
        )


def test_composed_map_checks_each_declared_operation_before_building():
    domain, operation = space(), Forbidden()
    stages = (
        MapCapabilities(operation, domain, domain, vjp=operation),
        MapCapabilities(operation, domain, domain, vjp=object()),
    )
    loss = ScalarCapabilities(operation, domain, operation)
    with pytest.raises(TypeError, match="forward VJP.*callable vjp"):
        InverseProblem.from_maps(*stages, loss=loss)


@pytest.mark.parametrize("malformed_part", ["take", "scatter", "full_layout", "selected_layout"])
def test_malformed_selection_rejected_before_forward(malformed_part):
    from types import SimpleNamespace

    domain, operation = space(), Forbidden()
    values = dict(
        full_layout=domain, selected_layout=domain, take=lambda x: x, scatter=lambda x: x
    )
    values[malformed_part] = None
    with pytest.raises(TypeError, match="selection"):
        InverseProblem(
            MapCapabilities(operation, domain, domain, vjp=operation),
            ScalarCapabilities(operation, domain, operation),
            selection=SimpleNamespace(**values),
        )


def test_weighted_nonnegative_projection_matches_unconstrained_analytic_solution():
    reference = np.ones(3)
    h, mass = np.ones((1, 3)), np.array([1.0, 2.0, 4.0])
    result = project_nonnegative_equalities(reference, h, [6.0], mass)
    np.testing.assert_allclose(result.x, np.array([19.0, 13.0, 10.0]) / 7, atol=1e-12)
    np.testing.assert_allclose(h @ result.x, [6.0], atol=1e-12)
    assert result.success and result.diagnostics["active_count"] == 0
    np.testing.assert_array_equal(reference, np.ones(3))


def test_nonnegative_projection_resolves_active_bound_without_clipping_equalities():
    result = project_nonnegative_equalities([2.0, 0.0], [[1.0, 1.0]], [1.0], [1.0, 3.0])
    # Unconstrained projection is (1.25,-0.25); clipping would violate H U = 1.
    np.testing.assert_allclose(result.x, [1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(result.equality_multipliers, [1.0], atol=1e-12)
    np.testing.assert_allclose(result.lower_multipliers, [0.0, 1.0], atol=1e-12)
    assert result.fun == pytest.approx(0.5)
    assert result.diagnostics["equality_residual_inf"] < 1e-12
    assert result.diagnostics["kkt_residual_inf"] < 1e-12


def test_nonnegative_projection_accepts_signed_redundant_observations():
    result = project_nonnegative_equalities(
        [0.0, 0.0], [[1.0, -1.0], [2.0, -2.0]], [1.0, 2.0], [2.0, 1.0]
    )
    np.testing.assert_allclose(result.x, [1.0, 0.0], atol=1e-12)
    assert result.diagnostics["kkt_residual_inf"] < 1e-12


def test_nonnegative_projection_recovers_independently_constructed_kkt_point():
    h = np.array([[1, -2, 3, 1, 2], [0.5, 1, -1, 2, 3]], dtype=float)
    mass = np.array([1.0, 2.0, 0.5, 3.0, 4.0])
    solution = np.array([1.0, 0.0, 0.3, 0.0, 2.0])
    mu, lower = np.array([0.2, -0.3]), np.array([0.0, 0.2, 0.0, 1.0, 0.0])
    reference = solution + (h.T @ mu - lower) / mass
    result = project_nonnegative_equalities(reference, h, h @ solution, mass)
    np.testing.assert_allclose(result.x, solution, atol=1e-10)
    assert result.diagnostics["equality_residual_inf"] <= 1e-10
    assert result.diagnostics["kkt_residual_inf"] <= 1e-10


def test_nonnegative_projection_distinguishes_infeasibility_from_iteration_limit():
    with pytest.raises(ValueError, match="infeasible.*separation"):
        project_nonnegative_equalities([1.0, 1.0], [[1.0, 1.0]], [-1.0], [1.0, 1.0])
    with pytest.raises(RuntimeError, match="uncertified"):
        project_nonnegative_equalities(
            [3.0, 2.0, 0.0], [[1.0, 1.0, 1.0]], [0.5], np.ones(3), maxiter=1
        )


def test_feasible_ill_scaled_nonnegative_projection_is_never_labelled_infeasible():
    # U=(1e12,1) is feasible. A Gram least-squares solve or LP tolerance may
    # discard the small first row, but a negative H.T@w is no Farkas vector.
    try:
        result = project_nonnegative_equalities(
            [0.0, 0.0], np.diag([1e-12, 1.0]), [1.0, 1.0], [1.0, 1.0], maxiter=3
        )
    except RuntimeError as error:
        assert "uncertified" in str(error)
    else:
        np.testing.assert_allclose(np.diag([1e-12, 1.0]) @ result.x, [1.0, 1.0], atol=1e-10)


@pytest.mark.parametrize(
    "reference,observation,mass", [([-2.0], [[0.0]], [1e308]), ([1e100], [[1e100]], [1e300])]
)
def test_nonnegative_projection_rejects_nonfinite_certificate_from_finite_inputs(
    reference, observation, mass
):
    with pytest.raises(RuntimeError, match="uncertified.*nonfinite KKT"):
        project_nonnegative_equalities(reference, observation, [0.0], mass)


@pytest.mark.parametrize("mass", [[0.0, 1.0], [-1.0, 1.0], [[1.0, 0.0], [0.0, 1.0]]])
def test_nonnegative_projection_requires_strictly_positive_diagonal_metric(mass):
    with pytest.raises(ValueError, match="positive diagonal mass"):
        project_nonnegative_equalities([1.0, 1.0], [[1.0, 1.0]], [1.0], mass)


def test_nonnegative_observation_analysis_matches_gaussian_linear_update():
    reference = np.array([1.0, 2.0, 3.0])
    h = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 1.0]])
    target, mass = np.array([2.3, 5.4]), np.array([2.0, 3.0, 4.0])
    covariance = np.array([[0.4, 0.1], [0.1, 0.8]])
    expected = reference + (h.T / mass[:, None]) @ np.linalg.solve(
        (h / mass) @ h.T + covariance, target - h @ reference
    )
    result = analyze_nonnegative_observations(reference, h, target, mass, covariance)
    np.testing.assert_allclose(result.x, expected, atol=1e-12)
    assert result.success and result.diagnostics["active_count"] == 0
    assert result.diagnostics["kkt_residual_inf"] < 1e-11
    np.testing.assert_array_equal(reference, [1.0, 2.0, 3.0])


def test_nonnegative_observation_analysis_solves_active_bound_without_post_clipping():
    result = analyze_nonnegative_observations([2.0, 0.0], [[1.0, 1.0]], [0.1], [1.0, 3.0], [0.2])
    np.testing.assert_allclose(result.x, [5 / 12, 0.0], atol=1e-12)
    np.testing.assert_allclose(result.observation_multipliers, [19 / 12], atol=1e-12)
    np.testing.assert_allclose(result.lower_multipliers, [0.0, 19 / 12], atol=1e-12)
    assert result.diagnostics["active_count"] == 1
    assert result.diagnostics["kkt_residual_inf"] < 1e-11


def test_nonnegative_observation_analysis_recovers_independent_full_covariance_kkt_point():
    h = np.array([[1, -2, 3, 1, 2], [0.5, 1, -1, 2, 3]], dtype=float)
    mass = np.array([1.0, 2.0, 0.5, 3.0, 4.0])
    covariance = np.array([[0.3, -0.1], [-0.1, 0.2]])
    solution = np.array([1.0, 0.0, 0.3, 0.0, 2.0])
    mu = np.array([0.2, -0.3])
    lower = np.array([0.0, 0.2, 0.0, 1.0, 0.0])
    reference = solution + (h.T @ mu - lower) / mass
    target = h @ solution - covariance @ mu
    result = analyze_nonnegative_observations(reference, h, target, mass, covariance)
    np.testing.assert_allclose(result.x, solution, atol=1e-10)
    np.testing.assert_allclose(result.observation_multipliers, mu, atol=1e-10)
    np.testing.assert_allclose(result.lower_multipliers, lower, atol=1e-10)
    assert result.diagnostics["kkt_residual_inf"] <= 1e-10


@pytest.mark.parametrize("covariance", [[], np.empty((0, 0))])
def test_nonnegative_observation_analysis_without_observations(covariance):
    result = analyze_nonnegative_observations(
        [-2.0, 3.0, 0.0], np.empty((0, 3)), [], [2.0, 4.0, 1.0], covariance
    )
    np.testing.assert_array_equal(result.x, [0.0, 3.0, 0.0])
    np.testing.assert_array_equal(result.lower_multipliers, [4.0, 0.0, 0.0])
    assert result.fun == pytest.approx(4.0)
    assert result.nit == 0 and result.success
    assert result.observation_multipliers.shape == (0,)
    assert result.diagnostics["kkt_residual_inf"] == 0


def test_nonnegative_observation_analysis_accepts_negative_noisy_observation():
    result = analyze_nonnegative_observations([0.0], [[1.0]], [-1.0], [1.0], [0.5])
    np.testing.assert_array_equal(result.x, [0.0])
    assert result.lower_multipliers[0] == pytest.approx(2.0)
    assert result.fun == pytest.approx(1.0)


@pytest.mark.parametrize(
    "covariance",
    [[0.0], [-1.0], [True], [complex(1.0)], [[1.0, 0.2], [0.0, 1.0]], [[1.0, 2.0], [2.0, 1.0]]],
)
def test_nonnegative_observation_analysis_requires_positive_covariance(covariance):
    size = 2 if np.ndim(covariance) == 2 else 1
    with pytest.raises(ValueError, match="covariance"):
        analyze_nonnegative_observations(
            np.ones(size), np.eye(size), np.ones(size), np.ones(size), covariance
        )


@pytest.mark.parametrize("mass", [[0.0], [-1.0], [[1.0]], [True]])
def test_nonnegative_observation_analysis_requires_positive_diagonal_precision(mass):
    with pytest.raises(ValueError, match="mass"):
        analyze_nonnegative_observations([1.0], [[1.0]], [1.0], mass, [1.0])


@pytest.mark.parametrize(
    "reference,observation,mass,exception",
    [
        ([1.0], [[1e308]], [1.0], ValueError),
        ([-2.0], [[0.0]], [1e308], RuntimeError),
    ],
)
def test_nonnegative_observation_analysis_rejects_nonfinite_arithmetic(
    reference, observation, mass, exception
):
    with pytest.raises(exception, match="overflows|uncertified.*nonfinite"):
        analyze_nonnegative_observations(reference, observation, [0.0], mass, [1.0])


def test_nonnegative_observation_analysis_reports_iteration_limit_as_uncertified():
    with pytest.raises(RuntimeError, match="uncertified"):
        analyze_nonnegative_observations(
            [3.0, 2.0, 0.0],
            [[1.0, 1.0, 1.0]],
            [0.5],
            [1.0, 1.0, 1.0],
            [0.01],
            maxiter=1,
        )
