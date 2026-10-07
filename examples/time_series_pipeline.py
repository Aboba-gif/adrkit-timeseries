"""Пример HGB с собственными признаками и метрикой на синтетической панели.

Запуск из каталога проекта: python -B examples/time_series_pipeline.py
"""
import numpy as np
import pandas as pd

from timeseries.contracts import NetworkPanel, PanelSpec, conceal, closed
from timeseries.features import calendar_features, channel_features, combine_features, feature_from_function, lag_features
from timeseries.imputation import hgb_family
from timeseries.metrics import MetricFamily, metric_from_function
from timeseries.pipeline import Population, build_pipeline


def population(start, kind):
    """Создать синтетическую часовую выборку температуры и давления.

    Parameters
    ----------
    start : str
        Начало восьмичасового диапазона для ``pandas.date_range``; часы без
        часового пояса.
    kind : {"train", "calibration", "selection", "test", "export"}
        Назначение выборки; в ``test`` искусственно скрывается температура
        на четвёртом шаге.

    Returns
    -------
    sample : Population
        Один пост, восемь часовых шагов и два канала: температура в degC
        и давление в hPa. Скрытая тестовая температура находится только
        в ``truth``.
    """
    spec = PanelSpec(dict(start=start, stop=str(pd.Timestamp(start)+pd.Timedelta(hours=8)),
        frequency="1h", origin="2020-01-01", clock="naive", timezone=None,
        calendar="proleptic_gregorian", sites=["example site"],
        channels=["air_temperature", "pressure"],
        units={"air_temperature": "degC", "pressure": "hPa"}))
    values = np.column_stack([np.arange(8)-10., 990.+np.arange(8)])[:, None, :]
    observed = np.ones(values.shape, dtype=bool)
    panel = NetworkPanel(spec, tuple(pd.date_range(start, periods=8, freq="1h")),
        values, observed, observed, ~observed)
    hidden = np.zeros_like(observed)
    if kind == "test":
        hidden[3, 0, 0] = True
    inputs, truth = conceal(panel, artificial_mask=hidden, mit_mask=np.zeros_like(hidden),
        protocol_id="builder_example", population_id=kind, population_kind=kind)
    return Population(inputs, truth)


def main():
    """Выполнить синтетический пример с внешним признаком и метрикой."""
    def pressure_squared(inputs):
        """Вычислить квадрат видимого давления для внешнего признака.

        Parameters
        ----------
        inputs : MaskedInputs
            Панель с каналом ``pressure`` в hPa. Скрытые значения видны только
            как ``NaN``.

        Returns
        -------
        values : ndarray of float, shape (n_times, n_sites, 1)
            Квадрат давления в hPa^2; пропуски остаются ``NaN``.
        """
        channel = inputs.spec.to_dict()["channels"].index("pressure")
        return inputs.values[:, :, channel:channel+1] ** 2

    features = combine_features(
        channel_features(["pressure"]), lag_features(["air_temperature"], [1]),
        calendar_features(["hour_sin", "hour_cos"]),
        feature_from_function(names=["pressure_squared"], units=["hPa^2"],
                              transform=pressure_squared, uses_future=False))
    max_error = MetricFamily(lambda p: dict(closed(p, set())),
        lambda p: metric_from_function("max_error", lambda y, prediction: np.max(np.abs(prediction-y)),
                                       unit="channel"))
    pipeline = build_pipeline(dict(protocol_id="builder_example", seed=7,
        imputer=dict(kind="custom_hgb", parameters=dict(
            target="air_temperature", max_iter=2, learning_rate=.1,
            max_leaf_nodes=3, l2_regularization=0., min_samples_leaf=2, early_stopping=False)),
        metrics=[dict(kind="rmse", parameters={}), dict(kind="max_error", parameters={})]),
        families={"custom_hgb": hgb_family(features, context="causal")},
        metric_families={"max_error": max_error})
    train, test = population("2020-01-01", "train"), population("2020-01-02", "test")
    # Признаки используют давление, прошлую температуру и календарное время.
    output = pipeline.run(train=train, calibration=None, test=test)
    assert output.test.prediction[3, 0, 0] < 0.
    assert np.array_equal(output.test.prediction[test.inputs.visible], test.inputs.values[test.inputs.visible])
    assert output.metrics["test"]["rmse"] is None  # Ошибки выводятся отдельно для температуры и давления.
    print(output.metrics["test"]["by_channel"])


if __name__ == "__main__":
    main()
