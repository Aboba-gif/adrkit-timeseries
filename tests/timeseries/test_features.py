"""Признаки строятся после скрытия контрольных значений и сохраняют объявленный порядок численных столбцов."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from timeseries.contracts import DataContractError, NetworkPanel, PanelSpec, conceal
from timeseries.features import FeatureValues, availability_features, calendar_features, channel_features, combine_features, feature_from_function, lag_features, lead_features
from timeseries.imputation import HGBEstimatorParameters, HGBFiller, HGBImputer, HGBParameters, build_imputer, hgb_family


def inputs(*, kind="export", hidden_value=-6., hide=True, start="2020-01-01",
           stop="2020-01-01 08:00", timezone=None):
    spec = PanelSpec(dict(start=start, stop=stop, frequency="1h", origin=start,
        clock="naive" if timezone is None else "localized", timezone=timezone,
        calendar="proleptic_gregorian", sites=["custom site", "another site"],
        channels=["temperature", "barometric pressure"], units={"temperature":"degC", "barometric pressure":"hPa"}))
    from timeseries.contracts import clock_timestamp
    grid = tuple(pd.date_range(clock_timestamp(start, spec.to_dict()),
                              clock_timestamp(stop, spec.to_dict()), freq="1h", inclusive="left"))
    values = np.empty((len(grid), 2, 2))
    values[:, :, 0] = -10. + np.arange(len(grid))[:, None]
    values[:, :, 1] = 990. + np.arange(len(grid))[:, None]
    if len(grid) > 2:
        values[2, 0, 0] = hidden_value
    observed = np.ones(values.shape, dtype=bool)
    panel = NetworkPanel(spec, grid, values, observed, observed, ~observed)
    hidden = np.zeros(values.shape, dtype=bool)
    if hide and len(grid) > 2:
        hidden[2, 0, 0] = True
    masked, truth = conceal(panel, artificial_mask=hidden, mit_mask=np.zeros_like(hidden),
        protocol_id="features", population_id=kind, population_kind=kind)
    return panel, masked, truth


def test_composed_features_retain_order_signed_scale_units_and_mask():
    _, masked, _ = inputs()
    features = combine_features(channel_features(["temperature"]),
        availability_features(["temperature"]), lag_features(["temperature"], [1, 2]),
        calendar_features(["hour", "hour_sin"]))
    output = features.transform(masked, context="causal")
    assert output.names == ("temperature:value", "temperature:available", "temperature:lag1",
                            "temperature:lag2", "calendar:hour", "calendar:hour_sin")
    assert output.units == ("degC", "1", "degC", "degC", "h", "1")
    assert output.sites == ("custom site", "another site")
    assert output.times == masked.times and output.panel_spec == masked.spec
    assert output.values[0, 0, 0] == -10.
    assert np.isnan(output.values[2, 0, 0]) and output.values[2, 0, 1] == 0.
    assert np.isnan(output.values[3, 0, 2])  # Лаг скрытого значения остаётся пропуском.
    np.testing.assert_array_equal(output.values[1:, 1, 2], masked.values[:-1, 1, 0])
    with pytest.raises(ValueError):
        output.values.flags.writeable = True


def test_poisoned_hidden_truth_cannot_change_features():
    _, a, _ = inputs(hidden_value=-6.)
    _, b, _ = inputs(hidden_value=1e30)
    features = combine_features(lag_features(["temperature"], [1]),
        lead_features(["temperature"], [1]), channel_features(["barometric pressure"]),
        availability_features(["temperature"]))
    np.testing.assert_array_equal(features.transform(a).values, features.transform(b).values)


def test_leads_and_unknown_callbacks_fail_causal_preflight():
    _, masked, _ = inputs()
    calls = []
    callback = feature_from_function(names=["custom"], units=["1"],
        transform=lambda x: calls.append(x) or np.zeros((*x.values.shape[:2], 1)))
    assert calls == []  # Конструктор не выполняет вычислений.
    with pytest.raises(DataContractError, match="no-future"):
        callback.transform(masked, context="causal")
    with pytest.raises(DataContractError, match="no-future"):
        combine_features(callback, lead_features(["temperature"], [1])).transform(masked, context="causal")
    assert calls == []
    assert lead_features(["temperature"], [1]).transform(masked).values[0, 0, 0] == -9.


def test_causal_builtins_do_not_depend_on_later_observations():
    _, a, _ = inputs(hide=False)
    values = a.values.copy()
    values[5:] = 1e6
    b = replace(a, values=values)
    features = combine_features(lag_features(["temperature"], [1, 3]),
                                channel_features(["barometric pressure"]), calendar_features(["month"]))
    np.testing.assert_array_equal(features.transform(a, context="causal").values[:5],
                                  features.transform(b, context="causal").values[:5])


def test_callback_sees_only_masked_owned_record_and_result_is_owned():
    panel, masked, truth = inputs()
    saved = np.zeros((*masked.values.shape[:2], 1))

    def custom(x):
        assert x is masked and not hasattr(x, "truth")
        assert np.isnan(x.values[2, 0, 0])
        with pytest.raises(ValueError):
            x.values.flags.writeable = True
        saved[:] = x.values[:, :, :1] ** 2
        return saved

    features = feature_from_function(names=["temperature_squared"], units=["degC^2"],
                                     transform=custom, uses_future=False)
    output = features.transform(masked, context="causal")
    saved[:] = 777.
    assert output.values[0, 0, 0] == 100.
    for wrong in (panel, truth):
        with pytest.raises(DataContractError, match="MaskedInputs"):
            features.transform(wrong)


@pytest.mark.parametrize("result", [np.zeros((8, 1)), np.full((8, 2, 1), np.inf),
                                    np.zeros((8, 2, 1), dtype=complex), np.full((8, 2, 1), "text")])
def test_callback_output_checks_real_shape_and_infinities(result):
    _, masked, _ = inputs()
    features = feature_from_function(names=["invalid"], units=["1"], transform=lambda _: result)
    with pytest.raises(DataContractError):
        features.transform(masked)


def test_missing_channels_fail_before_any_callback_and_names_are_unique():
    _, masked, _ = inputs()
    calls = []
    custom = feature_from_function(names=["x"], units=["1"],
        transform=lambda x: calls.append(x) or np.zeros((*x.values.shape[:2], 1)))
    with pytest.raises(DataContractError, match="channel"):
        combine_features(custom, channel_features(["absent"])).transform(masked)
    assert calls == []
    with pytest.raises(DataContractError, match="duplicate"):
        combine_features(custom, custom)
    with pytest.raises(DataContractError):
        feature_from_function(names=["x"], units=[], transform=lambda _: None)
    with pytest.raises(DataContractError):
        feature_from_function(names=["x"], units=["1"], transform=None)
    with pytest.raises(DataContractError):
        lag_features(["temperature"], [True])


@pytest.mark.parametrize("start,stop,count", [("2020-10-25", "2020-10-25 04:00", 5),
                                               ("2020-03-29", "2020-03-29 04:00", 3)])
def test_calendar_features_follow_declared_dst_grid(start, stop, count):
    _, masked, _ = inputs(start=start, stop=stop, timezone="Europe/Berlin")
    result = calendar_features(["hour"]).transform(masked, context="causal")
    assert len(result.times) == count
    np.testing.assert_array_equal(result.values[:, 0, 0], pd.DatetimeIndex(masked.times).hour)


def estimator_parameters():
    return dict(target="temperature", max_iter=2, learning_rate=.1, max_leaf_nodes=3,
                l2_regularization=0., min_samples_leaf=2, early_stopping=False)


def test_custom_hgb_features_fit_once_on_signed_visible_targets():
    _, train, _ = inputs(kind="train", hidden_value=1e30)
    _, heldout, _ = inputs(kind="test", hidden_value=1e30)
    features = combine_features(channel_features(["barometric pressure"]),
                                calendar_features(["hour"]))
    family = hgb_family(features, context="causal")
    spec = dict(kind="temperature_hgb", parameters=estimator_parameters())
    imputer = build_imputer(spec, families={"temperature_hgb": family})
    with threadpool_limits(limits=1):
        state = imputer.fit(train, rng=np.random.default_rng(17))
        result = imputer.predict(heldout, fitted=state)
        _, poisoned, _ = inputs(kind="test", hidden_value=-1e30)
        repeated = imputer.predict(poisoned, fitted=state)
    assert state.feature_units == ("hPa", "h") and state.features is features
    assert result.prediction[2, 0, 0] < 0.  # Отрицательная температура остаётся допустимым результатом.
    np.testing.assert_array_equal(result.prediction, repeated.prediction)
    np.testing.assert_array_equal(result.prediction[heldout.visible], heldout.values[heldout.visible])
    assert state.feature_names == features.names
    with pytest.raises(DataContractError, match="binding"):
        HGBImputer(state.parameters, channel_features(["barometric pressure"])).predict(heldout, fitted=state)
    with pytest.raises(DataContractError, match="unknown"):
        build_imputer({**spec, "parameters": {**estimator_parameters(), "lags": [1]}},
                      families={"temperature_hgb": family})


def test_custom_hgb_rejects_false_causal_configuration_without_fit():
    with pytest.raises(DataContractError, match="no-future"):
        hgb_family(lead_features(["temperature"], [1]), context="causal")
    with pytest.raises(DataContractError, match="FeatureSet"):
        HGBImputer(HGBEstimatorParameters(**estimator_parameters()), features=lambda _: None)


def test_builtin_hgb_features_keep_same_estimator_path():
    _, train, _ = inputs(kind="train")
    _, heldout, _ = inputs(kind="test")
    parameters = HGBParameters(**estimator_parameters(), lags=(1,), leads=(1,),
                              covs=("barometric pressure",), calendar=(), use_leads=True)
    imputer = HGBImputer(parameters)
    with threadpool_limits(limits=1):
        fitted = imputer.fit(train, rng=np.random.default_rng(8))
        actual = imputer.predict(heldout, fitted=fitted)
        for site, seed in enumerate(fitted.seeds):
            reference = HGBFiller([1], [1], ["barometric pressure"], [], parameters.estimator_parameters(seed))
            train_x = pd.DataFrame({"barometric pressure": train.values[:, site, 1]})
            reference.fit(train.values[:, site, 0], train_x, use_leads=True)
            expected = heldout.values[:, site, 0].copy()
            if np.isnan(expected).any():
                expected, _ = reference.fill(expected,
                    pd.DataFrame({"barometric pressure": heldout.values[:, site, 1]}))
            np.testing.assert_array_equal(actual.prediction[:, site, 0], expected)
    assert fitted.features is None and fitted.feature_names == ()
