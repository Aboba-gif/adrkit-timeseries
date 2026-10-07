"""Статусы оптимизации, состояния Калмана и параметры численных ядер."""
from types import SimpleNamespace

import numpy as np
import pytest

from timeseries import numerics
from timeseries.numerics import HGBFiller, KalmanDLM, SoftImpute, fill_linear, fill_pchip


@pytest.mark.parametrize("previous", [None, np.array([-2., -3., -4., .2, -1.])])
def test_unsuccessful_fit_keeps_previous_parameters_and_does_not_retry(monkeypatch, previous):
    calls = []

    def minimize(objective, x0, *, method, options):
        calls.append((x0.copy(), method, options.copy()))
        return SimpleNamespace(success=False, status=1, message="iteration limit",
                               x=np.ones(5), fun=1.)

    monkeypatch.setattr(numerics.optimize, "minimize", minimize)
    model = KalmanDLM(period=8, harmonics=1, theta=previous)
    x0 = np.array([-1., -2., -3., .4, -.5])
    before = None if previous is None else previous.copy()
    with pytest.raises(RuntimeError, match=r"status 1.*iteration limit"):
        model.fit(np.array([1., 2., np.nan, 4.]), x0=x0, maxiter=7)
    assert model.theta is previous
    if previous is not None:
        np.testing.assert_array_equal(previous, before)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], x0)
    assert calls[0][1:] == ("L-BFGS-B", {"maxiter": 7})


def test_successful_fit_owns_optimizer_coefficients_and_returns_model(monkeypatch):
    coefficients = np.array([-1., -2., -3., .4, -.5])
    expected = coefficients.copy()
    calls = []

    def minimize(objective, x0, *, method, options):
        calls.append((method, options.copy()))
        return SimpleNamespace(success=True, status=0, message="converged",
                               x=coefficients, fun=2.)

    monkeypatch.setattr(numerics.optimize, "minimize", minimize)
    model = KalmanDLM(period=8, harmonics=1)
    returned = model.fit(np.array([1., 2., np.nan, 4.]), x0=expected.copy(), maxiter=9)
    assert returned is model
    assert calls == [("L-BFGS-B", {"maxiter": 9})]
    coefficients[:] = 123.
    np.testing.assert_array_equal(model.theta, expected)


@pytest.mark.parametrize("coefficients, objective", [
    (np.zeros(4), 1.),
    (np.zeros((5, 1)), 1.),
    (np.array([0., 0., np.nan, 0., 0.]), 1.),
    (np.array([0., 0., np.inf, 0., 0.]), 1.),
    (np.zeros(5, dtype=complex), 1.),
    (np.zeros(5), np.nan),
    (np.zeros(5), np.inf),
    (np.zeros(5), np.array([1.])),
])
def test_unusable_successful_result_cannot_replace_previous_theta(monkeypatch, coefficients, objective):
    calls = []

    def minimize(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(success=True, status=0, message="converged",
                               x=coefficients, fun=objective)

    monkeypatch.setattr(numerics.optimize, "minimize", minimize)
    previous = np.array([-1., -2., -3., .4, -.5])
    before = previous.copy()
    model = KalmanDLM(period=8, harmonics=1, theta=previous)
    with pytest.raises(RuntimeError, match="usable solution"):
        model.fit(np.array([1., 2., np.nan, 4.]), x0=before.copy())
    assert len(calls) == 1
    assert model.theta is previous
    np.testing.assert_array_equal(previous, before)


def test_likelihood_without_storage_matches_and_preserves_existing_histories():
    y = np.array([2., np.nan, -1., 4., np.nan, 3.])
    theta = np.array([-1., -2., -3., .4, -.5])
    model = KalmanDLM(period=8, harmonics=1)
    names = ("_at", "_Pt", "_af", "_Pf")
    without_storage = model._filter(y, theta, store=False)
    assert all(not hasattr(model, name) for name in names)
    with_storage = model._filter(y, theta, store=True)
    assert without_storage == with_storage
    histories = {name: getattr(model, name) for name in names}
    saved = {name: values.copy() for name, values in histories.items()}
    model._filter(np.array([10., 9., np.nan]), theta, store=False)
    for name, values in histories.items():
        assert getattr(model, name) is values
        np.testing.assert_array_equal(values, saved[name])


def test_default_initial_variance_matches_explicit_ten_exactly():
    y = np.array([2., np.nan, -1., 4.])
    theta = np.array([-1., -2., -3., .4, -.5])
    default = KalmanDLM(period=8, harmonics=1)
    explicit = KalmanDLM(period=8, harmonics=1, initial_variance=10.0)
    assert default._filter(y, theta) == explicit._filter(y, theta)
    for name in ("_at", "_Pt", "_af", "_Pf"):
        np.testing.assert_array_equal(getattr(default, name), getattr(explicit, name))


@pytest.mark.parametrize("variance", [.5, 2.])
def test_selected_initial_variance_changes_first_prediction_covariance(variance):
    model = KalmanDLM(period=4, harmonics=1, initial_variance=variance)
    model._filter(np.array([2., np.nan, -1.]), np.zeros(5))
    # Уровень и сезонные вращения сохраняют начальную дисперсию.
    # Нулевой коэффициент AR-компоненты устраняет её начальный вклад;
    # дисперсия каждого возмущения процесса здесь равна единице.
    expected = np.diag([variance + 1., variance + 1., variance + 1., 1.])
    np.testing.assert_array_equal(model._Pt[0], expected)


@pytest.mark.parametrize("variance", [0., -.5, np.nan, np.inf, True, 1j, "10"])
def test_invalid_initial_variance_is_rejected_at_construction(variance):
    with pytest.raises(ValueError, match="initial_variance"):
        KalmanDLM(period=8, harmonics=1, initial_variance=variance)


@pytest.mark.parametrize("function, arguments", [
    (fill_linear, (np.array([1., np.nan, 3.]),)),
    (fill_pchip, (np.array([1., np.nan, 3.]),)),
    (KalmanDLM(period=8, harmonics=1).smooth, (np.array([1., np.nan, 3.]),)),
    (KalmanDLM(period=8, harmonics=1).filter_only, (np.array([1., np.nan, 3.]),)),
    (SoftImpute(cycle_length=3).fill, (np.array([1., np.nan, 3.]),)),
    (HGBFiller([], [], [], []).fill, (np.array([1., np.nan, 3.]), None)),
], ids=["linear", "pchip", "smooth", "filter_only", "soft_impute", "hgb"])
def test_numerical_functions_reject_unknown_keywords(function, arguments):
    with pytest.raises(TypeError, match="unknown_option"):
        function(*arguments, unknown_option=True)


@pytest.mark.parametrize("mode", ["unknown", "interior", None])
def test_unknown_filter_mode_is_rejected_before_filter(monkeypatch, mode):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid mode must not execute the filter")

    monkeypatch.setattr(KalmanDLM, "_filter", forbidden)
    model = KalmanDLM(period=8, harmonics=1, theta=np.zeros(5))
    with pytest.raises(ValueError, match="head.*tail"):
        model.filter_only(np.array([1., np.nan, 3.]), mode=mode)


def test_arbitrary_cycle_length_preserves_order_and_pads_incomplete_cycle():
    y = np.arange(12, dtype=float)
    y[2] = np.nan
    original = y.copy()
    model = SoftImpute(cycle_length=5, lam=100., rank_max=2, max_iter=2, tol=0.)
    matrix = model._to_matrix(y)
    np.testing.assert_array_equal(matrix, np.array([
        [0., 1., np.nan, 3., 4.],
        [5., 6., 7., 8., 9.],
        [10., 11., np.nan, np.nan, np.nan],
    ]))
    matrix[0, 0] = 999.
    np.testing.assert_array_equal(y, original)
    filled, uncertainty = model.fill(y)
    expected = original.copy()
    expected[2] = 0.  # Порог lam превышает все сингулярные числа этой матрицы.
    np.testing.assert_array_equal(filled, expected)
    np.testing.assert_array_equal(y, original)
    assert uncertainty is None
