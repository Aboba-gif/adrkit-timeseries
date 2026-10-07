"""Проверки адаптеров Калмана и SoftImpute: численные ядра и привязки данных."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from timeseries.contracts import DataContractError, NetworkPanel, PanelSpec, conceal
import timeseries.imputation as imputation
from timeseries.imputation import (
    KalmanImputer, KalmanParameters, KalmanState, SoftImputeParameters,
    build_imputer, input_layout, validate_imputer_spec,
)
from timeseries.numerics import KalmanDLM, SoftImpute


def inputs(*, kind, hidden=False):
    spec = PanelSpec(dict(start="2020-01-01", stop="2020-01-01 08:00",
        frequency="1h", origin="2020-01-01", clock="naive", timezone=None,
        calendar="proleptic_gregorian", sites=["lab"], channels=["temperature"],
        units={"temperature": "degC"}))
    values = (np.arange(8, dtype=float) - 10)[:, None, None]
    observed = np.ones_like(values, dtype=bool)
    panel = NetworkPanel(spec, tuple(pd.date_range("2020-01-01", periods=8, freq="1h")),
        values, observed, observed, ~observed)
    hide = np.zeros_like(observed)
    if hidden:
        hide[3, 0, 0] = True
    return conceal(panel, artificial_mask=hide, mit_mask=np.zeros_like(hide),
        protocol_id="adapter-fixture", population_id=kind, population_kind=kind)


@pytest.mark.parametrize("mode", ["interior", "tail", "head"])
def test_kalman_kernel_prediction_unchanged_and_state_is_not_mutated(mode):
    train, _ = inputs(kind="train")
    test, truth = inputs(kind="test", hidden=True)
    imputer = build_imputer(dict(kind="kalman_dlm",
        parameters=dict(period=8, harmonics=1, maxiter=2, mode=mode)))
    state = KalmanState(imputer.parameters, train.protocol_id, train.population_id,
                        input_layout(train), np.array([[[-4., -5., -2., .3, -1.]]]))
    coefficients = state.theta.copy()
    result = imputer.predict(test, fitted=state)
    result.require_binding(truth)
    kernel = KalmanDLM(period=8, harmonics=1, theta=coefficients[0, 0].copy())
    expected, sd = (kernel.smooth(test.values[:, 0, 0]) if mode == "interior" else
                    kernel.filter_only(test.values[:, 0, 0], mode=mode))
    expected[test.visible[:, 0, 0]] = test.values[:, 0, 0][test.visible[:, 0, 0]]
    np.testing.assert_array_equal(result.prediction[:, 0, 0], expected)
    np.testing.assert_array_equal(result.uncertainty[:, 0, 0], sd)
    np.testing.assert_array_equal(state.theta, coefficients)
    repeated = imputer.predict(test, fitted=state)
    np.testing.assert_array_equal(result.prediction, repeated.prediction)
    with pytest.raises(DataContractError, match="binding"):
        imputer.predict(replace(test, protocol_id="foreign"), fitted=state)


def test_soft_impute_kernel_prediction_unchanged_and_signed_scale_retained():
    test, truth = inputs(kind="test", hidden=True)
    parameters = dict(lam=.1, rank_max=2, max_iter=3, tol=0., cycle_length=4)
    imputer = build_imputer(dict(kind="soft_impute", parameters=parameters))
    result = imputer.predict(test, fitted=imputer.no_fit_state)
    result.require_binding(truth)
    expected, _ = SoftImpute(**parameters).fill(test.values[:, 0, 0])
    np.testing.assert_array_equal(result.prediction[:, 0, 0], expected)
    assert (result.prediction < 0).all()


def test_unobserved_soft_impute_slot_is_unavailable_not_fabricated():
    test, _ = inputs(kind="test", hidden=True)
    values = test.values.copy()
    hidden = test.artificial_mask.copy()
    hidden[7, 0, 0] = True
    values[7, 0, 0] = np.nan
    test = replace(test, values=values, artificial_mask=hidden)
    imputer = build_imputer(dict(kind="soft_impute",
        parameters=dict(lam=.1, rank_max=2, max_iter=3, tol=0., cycle_length=4)))
    result = imputer.predict(test, fitted=imputer.no_fit_state)
    assert not result.availability[3, 0, 0] and not result.availability[7, 0, 0]
    assert result.status[3, 0, 0] == "unavailable"
    np.testing.assert_array_equal(result.prediction[test.visible], test.values[test.visible])


def kalman_parameters():
    return dict(period=8, harmonics=1, maxiter=2, mode="interior")


def test_kalman_default_parameters_are_normalized_and_serialized():
    requested = kalman_parameters()
    expected = dict(requested, initial_variance=10.0, x0=None)
    parameters = KalmanParameters.from_dict(requested)
    assert parameters.to_dict() == expected
    assert validate_imputer_spec(dict(kind="kalman_dlm", parameters=requested)) == {
        "kind": "kalman_dlm", "parameters": expected}
    assert build_imputer(dict(kind="kalman_dlm", parameters=requested)).parameters == parameters
    assert requested == kalman_parameters()
    with pytest.raises(TypeError):
        KalmanParameters(8, 1, 2, "interior", 10.0)
    with pytest.raises(TypeError):
        KalmanParameters(8, 1, 2, "interior", 10.0, None)


@pytest.mark.parametrize("variance", [0., -1., np.nan, np.inf, True, np.bool_(False),
                                      "10", [10.], 10+0j])
def test_kalman_initial_variance_rejects_invalid_values(variance):
    with pytest.raises(DataContractError, match="initial_variance"):
        KalmanParameters(8, 1, 2, "interior", initial_variance=variance)


@pytest.mark.parametrize("point", [
    [], [1., 2., 3., 4.], [1., 2., 3., 4., 5., 6.], [[1., 2., 3., 4., 5.]],
    ["1", 2., 3., 4., 5.], [True, 2., 3., 4., 5.],
    [1., np.nan, 3., 4., 5.], [1., np.inf, 3., 4., 5.], [1j, 2., 3., 4., 5.],
])
def test_kalman_initial_point_requires_five_finite_real_values(point):
    with pytest.raises(DataContractError, match="x0"):
        KalmanParameters(8, 1, 2, "interior", x0=point)


@pytest.mark.parametrize("factory", [list, tuple, np.array])
def test_kalman_parameters_own_initial_point_and_serialized_values(factory):
    point = factory([-4., -5., -2., .3, -1.])
    parameters = KalmanParameters(8, 1, 2, "interior",
                                  initial_variance=np.float64(3.5), x0=point)
    expected = (-4., -5., -2., .3, -1.)
    assert parameters.initial_variance == 3.5
    assert type(parameters.initial_variance) is float
    assert parameters.x0 == expected and type(parameters.x0) is tuple
    assert all(type(value) is float for value in parameters.x0)
    if not isinstance(point, tuple):
        point[0] = 999.
    assert parameters.x0 == expected
    serialized = parameters.to_dict()
    assert serialized["x0"] == list(expected)
    serialized["x0"][0] = 999.
    assert parameters.x0 == expected
    assert KalmanParameters.from_dict(parameters.to_dict()) == parameters


@pytest.mark.parametrize("missing", ["period", "harmonics", "maxiter", "mode"])
def test_kalman_original_parameters_remain_required(missing):
    requested = kalman_parameters()
    del requested[missing]
    with pytest.raises(DataContractError, match="missing"):
        KalmanParameters.from_dict(requested)


@pytest.mark.parametrize("mode", ["interior", "tail", "head"])
def test_kalman_adapter_passes_initial_parameters_only_to_required_kernel_operations(monkeypatch, mode):
    kernels, operations = [], []
    theta = np.array([-4., -5., -2., .3, -1.])

    class RecordingKernel:
        def __init__(self, period, harmonics, theta=None, *, initial_variance):
            assert period == 8 and harmonics == 1 and initial_variance == 3.5
            self.theta = theta
            kernels.append(self)

        def fit(self, values, *, x0, maxiter):
            assert maxiter == 2 and x0 == (-6., -7., -3., .2, -2.)
            operations.append(("fit", values.copy()))
            self.theta = theta.copy()
            return self

        def smooth(self, values):
            operations.append(("smooth", None))
            return np.nan_to_num(values, nan=-5.), np.ones_like(values)

        def filter_only(self, values, *, mode):
            operations.append(("filter_only", mode))
            return np.nan_to_num(values, nan=-5.), np.ones_like(values)

    monkeypatch.setattr(imputation, "KalmanDLM", RecordingKernel)
    point = [-6., -7., -3., .2, -2.]
    requested = dict(period=8, harmonics=1, maxiter=2, mode=mode,
                     initial_variance=3.5, x0=point)
    imputer = build_imputer(dict(kind="kalman_dlm", parameters=requested))
    point[0] = 999.
    train, _ = inputs(kind="train")
    test, truth = inputs(kind="test", hidden=True)
    rng = np.random.default_rng(3)
    before = rng.bit_generator.state
    state = imputer.fit(train, rng=rng)
    assert rng.bit_generator.state == before
    assert len(kernels) == 1 and kernels[0].theta is not state.theta
    np.testing.assert_array_equal(operations[0][1], train.values[:, 0, 0])
    kernels[0].theta[:] = 999.
    np.testing.assert_array_equal(state.theta[0, 0], theta)
    result = imputer.predict(test, fitted=state)
    result.require_binding(truth)
    assert len(kernels) == 2 and not np.shares_memory(kernels[1].theta, state.theta)
    np.testing.assert_array_equal(kernels[1].theta, theta)
    kernels[1].theta[:] = 999.
    np.testing.assert_array_equal(state.theta[0, 0], theta)
    assert state.parameters.to_dict() == dict(requested, x0=[-6., -7., -3., .2, -2.])
    assert operations[1] == (("smooth", None) if mode == "interior" else ("filter_only", mode))
    np.testing.assert_array_equal(result.prediction[test.visible], test.values[test.visible])


@pytest.mark.parametrize("changes", [dict(initial_variance=11.), dict(x0=[-4., -5., -2., .3, -1.])])
def test_kalman_state_binding_includes_initial_variance_and_point(monkeypatch, changes):
    train, _ = inputs(kind="train")
    test, _ = inputs(kind="test", hidden=True)
    parameters = KalmanParameters.from_dict(kalman_parameters())
    state = KalmanState(parameters, train.protocol_id, train.population_id,
                        input_layout(train), np.array([[[-4., -5., -2., .3, -1.]]]))
    imputer = KalmanImputer(replace(parameters, **changes))
    def forbidden(*args, **kwargs):
        pytest.fail("Incompatible Kalman state reached kernel")
    monkeypatch.setattr(imputation, "KalmanDLM", forbidden)
    with pytest.raises(DataContractError, match="binding"):
        imputer.predict(test, fitted=state)


def test_kalman_adapter_propagates_fit_failure_without_retry_or_fitted_state(monkeypatch):
    train, _ = inputs(kind="train")
    error = RuntimeError("unsuccessful numerical fit")
    calls = []
    class FailingKernel:
        def __init__(self, **parameters):
            calls.append(("construct", parameters))
        def fit(self, values, **parameters):
            calls.append(("fit", parameters))
            raise error
    monkeypatch.setattr(imputation, "KalmanDLM", FailingKernel)
    imputer = build_imputer(dict(kind="kalman_dlm", parameters=kalman_parameters()))
    with pytest.raises(RuntimeError) as caught:
        imputer.fit(train, rng=np.random.default_rng(3))
    assert caught.value is error
    assert calls == [
        ("construct", dict(period=8, harmonics=1, initial_variance=10.0)),
        ("fit", dict(x0=None, maxiter=2)),
    ]


@pytest.mark.parametrize("include_canonical", [False, True])
def test_soft_impute_rejects_old_cycle_key_without_alias(include_canonical):
    parameters = dict(lam=.1, rank_max=2, max_iter=3, tol=0., slots_per_day=4)
    if include_canonical:
        parameters["cycle_length"] = 4
    with pytest.raises(DataContractError, match="slots_per_day"):
        build_imputer(dict(kind="soft_impute", parameters=parameters))


def test_soft_impute_parameters_serialize_only_canonical_cycle_length():
    parameters = dict(lam=.1, rank_max=2, max_iter=3, tol=0., cycle_length=4)
    result = SoftImputeParameters.from_dict(parameters)
    assert result.to_dict() == parameters
    assert validate_imputer_spec(dict(kind="soft_impute", parameters=parameters)) == {
        "kind": "soft_impute", "parameters": parameters}
