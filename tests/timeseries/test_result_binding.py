"""Равенство формы массива не заменяет тождество каналов, времени и выборки."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from timeseries.contracts import DataContractError, ImputationResult, NetworkPanel, PanelSpec, conceal, result_from_inputs
from timeseries.imputation import ImputerFamily, build_imputer
from timeseries.metrics import MetricSet, evaluate_metrics, metric_from_function
from timeseries.pipeline import Population, build_pipeline, evaluate_prediction


def population(*, changes=None, protocol="binding", population_id="heldout", kind="test",
               hide_all=True):
    layout = dict(start="2020-01-01", stop="2020-01-01 04:00", frequency="1h",
        origin="2020-01-01", clock="naive", timezone=None,
        calendar="proleptic_gregorian", sites=["west", "east"],
        channels=["temperature", "pressure"], units={"temperature": "degC", "pressure": "hPa"})
    layout.update(changes or {})
    spec = PanelSpec(layout)
    from timeseries.contracts import clock_timestamp
    times = tuple(pd.date_range(clock_timestamp(layout["start"], layout),
        clock_timestamp(layout["stop"], layout), freq=layout["frequency"], inclusive="left"))
    values = np.broadcast_to(np.array([-8., 1000.]), (len(times), 2, 2)).copy()
    observed = np.ones(values.shape, dtype=bool)
    panel = NetworkPanel(spec, times, values, observed, observed, ~observed)
    hide = observed if hide_all else np.zeros_like(observed)
    inputs, truth = conceal(panel, artificial_mask=hide, mit_mask=np.zeros_like(hide),
        protocol_id=protocol, population_id=population_id, population_kind=kind)
    return panel, Population(inputs, truth)


@pytest.mark.parametrize("changes", [
    {"sites": ["east", "west"]},
    {"channels": ["pressure", "temperature"]},
    {"units": {"temperature": "K", "pressure": "hPa"}},
    {"clock": "localized", "timezone": "UTC"},
    {"start": "2020-01-02", "stop": "2020-01-02 04:00"},
    {"frequency": "30min", "stop": "2020-01-01 02:00"},
])
def test_foreign_axes_units_clock_or_time_rejected_before_metric(changes):
    panel, original = population()
    _, foreign = population(changes=changes)
    result = result_from_inputs(original.inputs, panel.values)
    assert result.prediction.shape == foreign.truth.values.shape
    calls = []
    metrics = MetricSet((metric_from_function("called", lambda *_: calls.append(1) or 0., unit="1"),))
    with pytest.raises(DataContractError, match="binding"):
        evaluate_metrics(foreign.truth, result, metrics=metrics)
    with pytest.raises(DataContractError, match="binding"):
        evaluate_prediction(foreign, result, metrics=metrics)
    assert calls == []


@pytest.mark.parametrize("metadata", [
    {"protocol": "another-protocol"}, {"population_id": "another-population"}, {"kind": "calibration"},
])
def test_foreign_population_rejected_even_with_equal_panel(metadata):
    panel, original = population()
    _, foreign = population(**metadata)
    result = result_from_inputs(original.inputs, panel.values)
    assert result.spec == foreign.inputs.spec
    with pytest.raises(DataContractError, match="binding"):
        evaluate_metrics(foreign.truth, result)


def test_result_constructor_requires_explicit_metadata_and_owns_arrays():
    panel, pop = population()
    with pytest.raises(TypeError):
        ImputationResult(panel.values, panel.original_observed, np.full(panel.values.shape, "imputed"))
    predictions = panel.values.copy()
    result = result_from_inputs(pop.inputs, predictions)
    predictions[:] = 777.
    assert result.prediction[0, 0, 0] == -8.
    assert result.spec is pop.inputs.spec and result.times == pop.inputs.times
    assert result.population_id == "heldout" and not hasattr(result, "truth")
    with pytest.raises(ValueError):
        result.prediction.flags.writeable = True
    assert evaluate_prediction(pop, result)["by_channel"][0]["rmse"] == 0.


def test_visible_changes_and_false_observed_status_are_not_repaired():
    panel, pop = population(hide_all=False)
    predictions = panel.values.copy()
    predictions[0, 0, 0] += 1.
    with pytest.raises(DataContractError, match="visible"):
        result_from_inputs(pop.inputs, predictions)
    with pytest.raises(DataContractError, match="status"):
        result_from_inputs(pop.inputs, panel.values, status=np.full(panel.values.shape, "imputed"))


@pytest.mark.parametrize("record", ["panel", "inputs", "truth", "result", "uncertainty"])
def test_complex_values_are_rejected_before_float_cast(record):
    panel, pop = population()
    complex_values = panel.values.astype(complex) + 3j
    with pytest.raises(DataContractError, match="real numeric"):
        if record == "panel":
            replace(panel, values=complex_values)
        elif record == "inputs":
            replace(pop.inputs, values=complex_values)
        elif record == "truth":
            replace(pop.truth, values=complex_values)
        elif record == "result":
            result_from_inputs(pop.inputs, complex_values)
        else:
            result_from_inputs(pop.inputs, panel.values, uncertainty=complex_values)


@pytest.mark.parametrize("value", [np.ones((4, 2)), np.full((4, 2, 2), "12")])
def test_result_factory_rejects_invalid_shape_and_nonnumeric_values(value):
    _, pop = population()
    with pytest.raises(DataContractError, match="real numeric"):
        result_from_inputs(pop.inputs, value)


def test_foreign_external_imputer_result_never_reaches_csv_write(tmp_path):
    panel, original = population()
    _, foreign = population(population_id="foreign")
    foreign_result = result_from_inputs(foreign.inputs, panel.values)

    class WrongPopulation:
        no_fit_state = None

        def predict(self, inputs, *, fitted):
            return foreign_result

    family = ImputerFamily(lambda parameters: dict(parameters), lambda _: WrongPopulation())
    pipeline = build_pipeline(dict(protocol_id="binding", seed=1,
        imputer=dict(kind="foreign", parameters={})), families={"foreign": family})
    output = tmp_path / "new-output"
    with pytest.raises(DataContractError, match="binding"):
        pipeline.run(train=None, calibration=None, test=original, output_root=output)
    assert not output.exists()


@pytest.mark.parametrize("phase", ["fit", "predict", "metric"])
def test_failed_computation_does_not_create_output_directory(tmp_path, phase):
    panel, train = population(kind="train", population_id="training")
    _, test = population(changes={"start": "2020-01-02", "stop": "2020-01-02 04:00"})

    class FailingImputer:
        def fit(self, inputs, *, rng):
            if phase == "fit":
                raise RuntimeError("fit failed")
            return None

        def predict(self, inputs, *, fitted):
            if phase == "predict":
                raise RuntimeError("predict failed")
            return result_from_inputs(inputs, panel.values)

    def failing_metric(*_):
        raise RuntimeError("metric failed")

    family = ImputerFamily(lambda parameters: dict(parameters), lambda _: FailingImputer())
    pipeline = build_pipeline(dict(protocol_id="binding", seed=1,
        imputer=dict(kind="failing", parameters={})), families={"failing": family})
    if phase == "metric":
        pipeline = replace(pipeline, metrics=MetricSet((
            metric_from_function("failing", failing_metric, unit="channel"),)))
    output = tmp_path / "new-parent" / "results"
    with pytest.raises(RuntimeError, match=f"{phase} failed"):
        pipeline.run(train=train, calibration=None, test=test, output_root=output)
    assert not output.parent.exists()


@pytest.mark.parametrize("writer", ["csv", "json"])
def test_partial_write_is_removed_and_same_destination_can_be_retried(tmp_path, monkeypatch, writer):
    _, test = population(hide_all=False)
    request = dict(protocol_id="binding", seed=1,
        imputer=dict(kind="linear", parameters={"edge_policy": "abstain"}))
    pipeline = build_pipeline(request)
    output = tmp_path / "results"

    def failed_csv(self, stream, **kwargs):
        stream.write("partial CSV")
        raise OSError("CSV write failed")

    def failed_json(metadata, stream, **kwargs):
        assert (output / "test.csv").exists()
        stream.write("partial JSON")
        raise OSError("JSON write failed")

    with monkeypatch.context() as patch:
        if writer == "csv":
            patch.setattr(pd.DataFrame, "to_csv", failed_csv)
        else:
            patch.setattr(json, "dump", failed_json)
        with pytest.raises(OSError, match="write failed"):
            pipeline.run(train=None, calibration=None, test=test, output_root=output)
    assert not output.exists()
    result = pipeline.run(train=None, calibration=None, test=test, output_root=output)
    assert result.output_root == output
    assert {path.name for path in output.iterdir()} == {"test.csv", "pipeline.json"}
    metadata = json.loads((output / "pipeline.json").read_text(encoding="utf-8"))
    assert metadata["imputer"] == request["imputer"]
    rows = pd.read_csv(output / "test.csv")
    assert len(rows) == test.inputs.values.size
    assert (rows["prediction"] == rows["visible_value"]).all()


def test_existing_output_is_preserved_and_rejected_before_prediction(tmp_path):
    _, test = population(hide_all=False)
    calls = []

    class Imputer:
        no_fit_state = None

        def predict(self, inputs, *, fitted):
            calls.append(1)
            return result_from_inputs(inputs, inputs.values)

    family = ImputerFamily(lambda parameters: dict(parameters), lambda _: Imputer())
    pipeline = build_pipeline(dict(protocol_id="binding", seed=1,
        imputer=dict(kind="custom", parameters={})), families={"custom": family})
    output = tmp_path / "results"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_bytes(b"existing content")
    with pytest.raises(DataContractError, match="must be new"):
        pipeline.run(train=None, calibration=None, test=test, output_root=output)
    assert calls == [] and sentinel.read_bytes() == b"existing content"


def test_failed_write_preserves_foreign_file_and_reports_incomplete_cleanup(tmp_path, monkeypatch):
    _, test = population(hide_all=False)
    pipeline = build_pipeline(dict(protocol_id="binding", seed=1,
        imputer=dict(kind="linear", parameters={"edge_policy": "abstain"})))
    output = tmp_path / "results"

    def failed_write(self, stream, **kwargs):
        (output / "foreign.txt").write_bytes(b"another writer")
        stream.write("partial CSV")
        raise OSError("CSV write failed")

    monkeypatch.setattr(pd.DataFrame, "to_csv", failed_write)
    with pytest.raises(OSError, match="CSV write failed") as error:
        pipeline.run(train=None, calibration=None, test=test, output_root=output)
    assert [path.name for path in output.iterdir()] == ["foreign.txt"]
    assert (output / "foreign.txt").read_bytes() == b"another writer"
    assert any("could not remove" in note for note in error.value.__notes__)


@pytest.mark.parametrize("kind", ["linear", "pchip"])
def test_builtin_results_are_bound_and_preserve_signed_visible_values(kind):
    panel, pop = population(hide_all=False)
    imputer = build_imputer(dict(kind=kind, parameters={"edge_policy": "abstain"}))
    result = imputer.predict(pop.inputs, fitted=imputer.no_fit_state)
    result.require_binding(pop.truth)
    np.testing.assert_array_equal(result.prediction, panel.values)
    assert (result.prediction[:, :, 0] < 0).all()



@pytest.mark.parametrize("sd_case", ["supplied", "none", "all_nan"])
def test_csv_export_preserves_external_standard_deviation(tmp_path, sd_case):
    panel, _ = population(changes={"stop": "2020-01-01 02:00"}, hide_all=False,
                          population_id="external-sd", kind="export")
    hidden = np.zeros_like(panel.original_observed)
    hidden[0, 1, 0] = True
    hidden[1, 0, 1] = True
    inputs, truth = conceal(panel, artificial_mask=hidden, mit_mask=np.zeros_like(hidden),
        protocol_id="binding", population_id="external-sd", population_kind="export")
    pop = Population(inputs, truth)
    supplied_prediction = np.array([
        [[-8., 1000.], [-7.5, 1000.]],
        [[-8., 1001.25], [-8., 1000.]],
    ])
    supplied_sd = (None if sd_case == "none" else np.full((2, 2, 2), np.nan)
                   if sd_case == "all_nan" else np.array([
                       [[.25, 3.], [.5, 4.]], [[.75, 5.], [1., 6.]],
                   ]))
    no_fit = object()
    calls = []

    class ExternalStandardDeviation:
        no_fit_state = no_fit

        def predict(self, inputs, *, fitted):
            assert fitted is no_fit
            calls.append(inputs.population_id)
            return result_from_inputs(inputs, supplied_prediction, uncertainty=supplied_sd)

    family = ImputerFamily(lambda parameters: dict(parameters),
                           lambda _: ExternalStandardDeviation())
    request = dict(protocol_id="binding", seed=1,
                   imputer=dict(kind="external_sd", parameters={}))
    pipeline = build_pipeline(request, families={"external_sd": family})
    output = tmp_path / "generic-export"
    run = pipeline.run(train=None, calibration=None, test=None, export=pop, output_root=output)
    assert calls == ["external-sd"] and run.output_root == output
    assert {path.name for path in output.iterdir()} == {"export.csv", "pipeline.json"}
    np.testing.assert_array_equal(run.export.prediction, supplied_prediction)
    if supplied_sd is None:
        assert run.export.uncertainty is None
    else:
        np.testing.assert_array_equal(run.export.uncertainty, supplied_sd)

    rows = pd.read_csv(output / "export.csv")
    assert list(rows[["time", "site", "channel"]].itertuples(index=False, name=None)) == [
        ("2020-01-01 00:00:00", "west", "temperature"),
        ("2020-01-01 00:00:00", "west", "pressure"),
        ("2020-01-01 00:00:00", "east", "temperature"),
        ("2020-01-01 00:00:00", "east", "pressure"),
        ("2020-01-01 01:00:00", "west", "temperature"),
        ("2020-01-01 01:00:00", "west", "pressure"),
        ("2020-01-01 01:00:00", "east", "temperature"),
        ("2020-01-01 01:00:00", "east", "pressure"),
    ]
    np.testing.assert_array_equal(rows["prediction"],
                                  [-8., 1000., -7.5, 1000., -8., 1001.25, -8., 1000.])
    np.testing.assert_array_equal(rows["visible_value"],
                                  [-8., 1000., np.nan, 1000., -8., np.nan, -8., 1000.])
    assert rows["status"].tolist() == ["observed", "observed", "imputed", "observed",
                                     "observed", "imputed", "observed", "observed"]
    assert rows["artificial_mask"].tolist() == [False, False, True, False, False, True, False, False]
    assert rows["is_imputed"].tolist() == [False, False, True, False, False, True, False, False]
    assert rows["row_present"].tolist() == [True] * 8
    assert rows["original_observed"].tolist() == [True] * 8
    assert rows["availability"].tolist() == [True] * 8
    assert rows["mit_mask"].tolist() == [False] * 8
    metadata = json.loads((output / "pipeline.json").read_text(encoding="utf-8"))
    assert metadata["imputer"] == request["imputer"]
    assert metadata["populations"]["export"]["panel"]["units"] == {
        "temperature": "degC", "pressure": "hPa"}
    expected_columns = ["time", "site", "channel", "visible_value", "row_present",
        "original_observed", "artificial_mask", "mit_mask", "prediction", "availability",
        "status", "is_imputed"]
    if sd_case == "none":
        assert rows.columns.tolist() == expected_columns
    else:
        assert rows.columns.tolist() == expected_columns + ["standard_deviation"]
        expected_sd = ([np.nan] * 8 if sd_case == "all_nan" else
                       [.25, 3., .5, 4., .75, 5., 1., 6.])
        np.testing.assert_array_equal(rows["standard_deviation"], expected_sd)
