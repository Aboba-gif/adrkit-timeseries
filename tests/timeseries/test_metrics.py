"""Физические каналы, недоступные контрольные значения и внешние метрики заданы явно."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from timeseries.contracts import DataContractError, NetworkPanel, PanelSpec, closed, conceal, result_from_inputs
from timeseries.metrics import MetricFamily, MetricSet, build_metrics, evaluate_metrics, metric_from_function
from timeseries.pipeline import Population, build_pipeline, evaluate_prediction


def population(channels=("temperature", "pressure"), units=None):
    units = units or {"temperature": "degC", "pressure": "hPa"}
    spec = PanelSpec(dict(start="2020-01-01 00:00", stop="2020-01-01 04:00",
        frequency="1h", origin="2020-01-01 00:00", clock="naive", timezone=None,
        calendar="proleptic_gregorian", sites=["lab-west", "lab-east"],
        channels=list(channels), units={name: units[name] for name in channels}))
    values = np.zeros((4, 2, len(channels)))
    for j, name in enumerate(channels):
        values[:, :, j] = (np.arange(4)[:, None] - 8 if name == "temperature" else 1000)
    observed = np.ones_like(values, dtype=bool)
    panel = NetworkPanel(spec, tuple(pd.date_range("2020-01-01", periods=4, freq="1h")),
        values, observed, observed, ~observed)
    hide = np.zeros_like(observed)
    hide[1:3] = True
    inputs, truth = conceal(panel, artificial_mask=hide, mit_mask=np.zeros_like(hide),
        protocol_id="generic", population_id="heldout", population_kind="test")
    prediction = values.copy()
    prediction[hide] += np.broadcast_to(np.arange(1, len(channels)+1), values.shape)[hide]
    result = result_from_inputs(inputs, prediction)
    return Population(inputs, truth), result


def test_errors_have_channel_site_units_without_mixed_rmse():
    pop, result = population()
    report = evaluate_prediction(pop, result)
    assert report["rmse"] is None and report["unit"] is None
    assert report["requested"] == report["scored"] == 8
    rows = {row["channel"]: row for row in report["by_channel"]}
    assert rows["temperature"]["rmse"] == 1 and rows["temperature"]["unit"] == "degC"
    assert rows["pressure"]["rmse"] == 2 and rows["pressure"]["metric_units"]["bias"] == "hPa"
    assert [(r["channel"], r["site"]) for r in report["by_site_channel"]] == [
        ("temperature", "lab-west"), ("temperature", "lab-east"),
        ("pressure", "lab-west"), ("pressure", "lab-east")]


def test_equal_unit_labels_do_not_pool_distinct_channels():
    pop, result = population(units={"temperature": "unknown", "pressure": "unknown"})
    assert evaluate_prediction(pop, result)["rmse"] is None


def test_single_channel_retains_exact_previous_flat_arithmetic():
    pop, result = population(("temperature",))
    scored = pop.truth.target_mask & np.isfinite(pop.truth.values) & result.availability
    error = result.prediction[scored] - pop.truth.values[scored]
    report = evaluate_prediction(pop, result)
    assert report["rmse"] == float(np.sqrt(np.mean(error**2)))
    assert report["mae"] == float(np.mean(np.abs(error)))
    assert report["bias"] == float(np.mean(error))


def test_missing_truth_and_abstentions_are_counted_separately():
    pop, result = population(("temperature",))
    values = pop.truth.values.copy()
    values[1, 0, 0] = np.nan
    truth = replace(pop.truth, values=values)
    prediction = result.prediction.copy()
    prediction[2, 1, 0] = np.nan
    result = result_from_inputs(pop.inputs, prediction)
    report = evaluate_prediction(replace(pop, truth=truth), result)
    assert (report["requested"], report["known_truth"], report["available"], report["scored"]) == (4, 3, 3, 2)


def test_empty_scores_do_not_call_custom_metric():
    pop, result = population(("temperature",))
    truth = replace(pop.truth, values=np.full_like(pop.truth.values, np.nan))
    metric = metric_from_function("never", lambda *_: pytest.fail("empty callback"), unit="1")
    report = evaluate_metrics(truth, result, metrics=MetricSet((metric,)))
    assert report["never"] is None and report["scored"] == 0


def test_callback_arrays_cannot_contaminate_later_metrics_or_originals():
    pop, result = population(("temperature",))
    def mutate(y, p):
        y.setflags(write=True)
        y[:] = 12345
        return 0
    metrics = MetricSet((metric_from_function("mutator", mutate, unit="1"), *build_metrics().metrics))
    report = evaluate_prediction(pop, result, metrics=metrics)
    assert report["rmse"] == 1 and pop.truth.values[1, 0, 0] == -7


@pytest.mark.parametrize("value", [np.inf, np.nan, 1j, [1], True])
def test_invalid_callback_scalar_rejected(value):
    metric = metric_from_function("invalid", lambda *_: value, unit="1")
    with pytest.raises(DataContractError, match="finite real scalar"):
        metric.score([1.], [2.])


def test_custom_metric_registry_in_pipeline_with_signed_values(tmp_path):
    pop, _ = population(("temperature",))
    family = MetricFamily(lambda p: dict(closed(p, set())),
        lambda p: metric_from_function("max_error", lambda y, prediction: np.max(np.abs(prediction-y)), unit="channel"))
    spec = dict(protocol_id="generic", seed=4,
        imputer={"kind": "linear", "parameters": {"edge_policy": "abstain"}},
        metrics=[{"kind": "max_error", "parameters": {}}])
    pipeline = build_pipeline(spec, metric_families={"max_error": family})
    spec["metrics"][0]["kind"] = "rmse"
    output = pipeline.run(train=None, calibration=None, test=pop, output_root=tmp_path / "new-output")
    assert output.metrics["test"]["max_error"] == 0
    assert (output.test.prediction < 0).all()
    import json
    metadata = json.loads((output.output_root / "pipeline.json").read_text())
    assert metadata["metric_configuration"][0]["kind"] == "max_error"
    assert metadata["metric_definitions"] == [{"name": "max_error", "unit": "channel"}]


def test_squared_metric_units_and_closed_configuration():
    pop, result = population(("temperature",))
    report = evaluate_prediction(pop, result, metrics=build_metrics([{"kind": "mse", "parameters": {}}]))
    assert report["mse"] == 1 and report["metric_units"]["mse"] == "(degC)^2"
    with pytest.raises(DataContractError):
        build_metrics([{"kind": "rmse", "parameters": {"unused": 1}}])
    with pytest.raises(DataContractError):
        build_metrics([{"kind": "rmse", "parameters": {}}] * 2)
    with pytest.raises(DataContractError):
        build_metrics(families={"rmse": MetricFamily(lambda p: p, lambda p: None)})


def test_float_overflow_is_explicit_and_does_not_write_a_fake_score():
    with pytest.raises(DataContractError, match="finite real scalar"):
        build_metrics().metrics[0].score([1e308], [-1e308])


@pytest.mark.parametrize("parameters", [[], {"scale": np.nan}])
def test_external_validator_must_return_a_finite_parameters_object(parameters):
    family = MetricFamily(lambda p: parameters,
        lambda p: metric_from_function("never", lambda *_: 0., unit="1"))
    with pytest.raises(DataContractError, match="parameters"):
        build_metrics([{"kind": "external", "parameters": {}}], families={"external": family})
