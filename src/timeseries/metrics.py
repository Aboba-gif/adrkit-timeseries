"""Ошибки заполнения по контрольным значениям каждого канала."""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Callable

import numpy as np

from .contracts import DataContractError, EvaluationTruth, ImputationResult, closed, text


_RESERVED = {"requested", "known_truth", "available", "scored", "unit",
             "metric_units", "channel", "site", "by_channel", "by_site_channel"}


@dataclass(frozen=True)
class Metric:
    """Вычислять именованную скалярную метрику в заданной единице.

    Parameters
    ----------
    name : str
        Имя результата, отличное от служебных имён отчёта.
    calculate : callable
        Функция ``calculate(target, prediction)``, возвращающая конечный
        вещественный скаляр. Получает отдельные массивы без права
        записи; формулу и веса определяет сама функция.
    unit : str
        Обозначение единицы: ``"channel"`` для единицы канала,
        ``"channel_squared"`` для её квадрата либо фиксированная строка.
        Числового преобразования единиц нет.
    """
    name: str
    calculate: Callable
    unit: str

    def __post_init__(self):
        text(self.name, "metric.name")
        text(self.unit, "metric.unit")
        if self.name in _RESERVED or not callable(self.calculate):
            raise DataContractError("metric requires an unreserved name and a callable")

    def score(self, target, prediction):
        """Вычислить метрику по конечным контрольным значениям и прогнозам.

        Parameters
        ----------
        target : array_like of real, shape (n,)
            Конечные контрольные значения в единице канала.
        prediction : array_like of real, shape (n,)
            Конечные прогнозы в той же единице и порядке.

        Returns
        -------
        value : float or None
            Конечный результат ``calculate`` в единице метрики. При n = 0
            возвращается ``None`` без вызова ``calculate``.

        Notes
        -----
        Перед вызовом ``calculate`` оба вектора копируются и закрываются для записи.
        Исходные массивы не передаются пользовательской функции.
        """
        vectors = []
        for name, value in (("target", target), ("prediction", prediction)):
            raw = np.asarray(value)
            if raw.ndim != 1 or raw.dtype.kind not in "fiu":
                raise DataContractError(f"metric {name} must be a real vector")
            vector = np.array(raw, dtype=float, copy=True)
            if not np.isfinite(vector).all():
                raise DataContractError(f"metric {name} must be finite")
            vector.setflags(write=False)
            vectors.append(vector)
        if vectors[0].shape != vectors[1].shape:
            raise DataContractError("metric target/prediction shape mismatch")
        if not vectors[0].size:
            return None
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            value = np.asarray(self.calculate(*vectors))
        if value.ndim != 0 or value.dtype.kind not in "fiu" or not np.isfinite(value):
            raise DataContractError(f"metric {self.name} must return a finite real scalar")
        result = float(value)
        if not np.isfinite(result):
            raise DataContractError(f"metric {self.name} exceeds float64 range")
        return result

    def result_unit(self, channel_unit):
        """Получить обозначение единицы метрики для данного канала.

        Parameters
        ----------
        channel_unit : str
            Единица канала, например ``"degC"``.

        Returns
        -------
        unit : str
            Единица канала, её квадрат либо фиксированное обозначение
            ``Metric.unit``. Значения данных не преобразуются.
        """
        return (channel_unit if self.unit == "channel" else
                f"({channel_unit})^2" if self.unit == "channel_squared" else self.unit)


def metric_from_function(name, calculate, *, unit):
    """Создать метрику из пользовательской функции и единицы.

    Parameters
    ----------
    name : str
        Уникальное имя метрики, отличное от служебных имён отчёта.
    calculate : callable
        Функция двух конечных векторов одинаковой длины, возвращающая
        конечный вещественный скаляр. Формулу и веса определяет функция.
    unit : str
        ``"channel"``, ``"channel_squared"`` либо фиксированная единица
        результата.

    Returns
    -------
    metric : Metric
        Метрика с проверкой входов и изолированными копиями массивов при
        вычислении.
    """
    return Metric(name, calculate, unit)


@dataclass(frozen=True)
class MetricSet:
    """Вычислять упорядоченный набор метрик с различными именами.

    Parameters
    ----------
    metrics : tuple of Metric
        Непустой набор метрик без повторяющихся имён. Порядок
        сохраняется в результатах.
    """
    metrics: tuple[Metric, ...]

    def __post_init__(self):
        metrics = tuple(self.metrics)
        if (not metrics or any(not isinstance(metric, Metric) for metric in metrics) or
                len({metric.name for metric in metrics}) != len(metrics)):
            raise DataContractError("metrics require a nonempty sequence of uniquely named Metric objects")
        object.__setattr__(self, "metrics", metrics)

    def score(self, target, prediction):
        """Вычислить все метрики по одной паре векторов.

        Parameters
        ----------
        target : array_like of real, shape (n,)
            Конечные контрольные значения в единице канала.
        prediction : array_like of real, shape (n,)
            Конечные прогнозы в той же единице и порядке.

        Returns
        -------
        scores : dict of str to float or None
            Результат для каждого имени в порядке ``metrics``. Для пустой пары
            каждый результат равен ``None``.
        """
        return {metric.name: metric.score(target, prediction) for metric in self.metrics}

    def units(self, channel_unit):
        """Получить обозначения единиц всех метрик для канала.

        Parameters
        ----------
        channel_unit : str
            Единица оцениваемого канала.

        Returns
        -------
        units : dict of str to str
            Имена метрик и соответствующие единицы без преобразования
            данных.
        """
        return {metric.name: metric.result_unit(channel_unit) for metric in self.metrics}


@dataclass(frozen=True)
class MetricFamily:
    """Создавать метрики из проверенных параметров семейства.

    Parameters
    ----------
    validate : callable
        Функция ``validate(parameters)``, проверяющая словарь спецификации и
        возвращающая словарь конечных JSON-значений.
    constructor : callable
        Функция ``constructor(parameters)``, создающая ``Metric`` из
        проверенного словаря.
    """
    validate: Callable
    constructor: Callable

    def __post_init__(self):
        if not callable(self.validate) or not callable(self.constructor):
            raise DataContractError("metric family requires validator and constructor")


def _no_parameters(parameters):
    """Проверить, что встроенной метрике передан пустой словарь параметров."""
    return dict(closed(parameters, set(), where="metric.parameters"))


def _builtin(name, calculate, unit="channel"):
    """Связать встроенную формулу метрики с проверкой пустых параметров."""
    return MetricFamily(_no_parameters, lambda parameters: Metric(name, calculate, unit))


def build_metrics(specs=None, *, families=None):
    r"""Создать метрики по спецификациям встроенных и внешних семейств.

    Parameters
    ----------
    specs : list or tuple of dict, optional
        Непустая последовательность словарей {``"kind"``: имя, ``"parameters"``:
        словарь}. По умолчанию выбираются ``rmse``, ``mae`` и ``bias`` с пустыми
        параметрами.
    families : dict of str to MetricFamily, optional
        Дополнительные семейства; имена ``rmse``, ``mae``, ``bias`` и ``mse`` заменять
        нельзя. Функции внешних метрик задают собственные формулы и
        веса.

    Returns
    -------
    metrics : MetricSet
        Упорядоченный набор метрик с уникальными именами.

    Notes
    -----
    Для встроенных метрик все n переданных точек имеют одинаковый вес.
    Положительный ``bias`` означает завышенный прогноз p относительно ответа y.

    .. math::

        \begin{aligned}
        \operatorname{RMSE} &= \sqrt{\frac{1}{n}\sum_{i=1}^n(p_i-y_i)^2}, \\
        \operatorname{MAE} &= \frac{1}{n}\sum_{i=1}^n|p_i-y_i|, \\
        \operatorname{bias} &= \frac{1}{n}\sum_{i=1}^n(p_i-y_i), \\
        \operatorname{MSE} &= \frac{1}{n}\sum_{i=1}^n(p_i-y_i)^2.
        \end{aligned}

    Здесь n — число сравниваемых пар; i — индекс пары от 1 до n;
    p_i — конечный прогноз, y_i — конечное контрольное значение той же точки
    в той же единице. RMSE — корень из средней квадратичной ошибки;
    MAE — средняя абсолютная ошибка; ``bias`` — средняя знаковая ошибка;
    MSE — средняя квадратичная ошибка.
    RMSE, MAE и ``bias`` имеют единицу канала, MSE — её квадрат.
    Для n = 0 возвращается ``None``, а формула не вычисляется.
    """
    if specs is None:
        specs = [{"kind": kind, "parameters": {}} for kind in ("rmse", "mae", "bias")]
    if type(specs) not in (list, tuple) or not specs:
        raise DataContractError("metric specs require a nonempty ordered sequence")
    try:
        owned = json.loads(json.dumps(specs, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise DataContractError("metric specs must be finite JSON") from exc
    registry = {
        "rmse": _builtin("rmse", lambda y, p: np.sqrt(np.mean((p-y)**2))),
        "mae": _builtin("mae", lambda y, p: np.mean(np.abs(p-y))),
        "bias": _builtin("bias", lambda y, p: np.mean(p-y)),
        "mse": _builtin("mse", lambda y, p: np.mean((p-y)**2), "channel_squared"),
    }
    if families is not None:
        if (type(families) is not dict or set(families) & set(registry) or
                any(not isinstance(family, MetricFamily) for family in families.values())):
            raise DataContractError("custom metric families must be explicit and cannot replace built-ins")
        for name in families:
            text(name, "metric family name")
        registry.update(families)
    metrics = []
    for spec in owned:
        closed(spec, {"kind", "parameters"}, where="metric spec")
        kind = text(spec["kind"], "metric.kind")
        if kind not in registry:
            raise DataContractError(f"unknown metric family: {kind}")
        if type(spec["parameters"]) is not dict:
            raise DataContractError("metric parameters must be an object")
        family = registry[kind]
        try:
            parameters = family.validate(spec["parameters"])
            if type(parameters) is not dict:
                raise DataContractError("metric validator must return a parameters object")
            parameters = json.loads(json.dumps(parameters, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise DataContractError(f"invalid {kind} parameters: {exc}") from exc
        metrics.append(family.constructor(parameters))
    return MetricSet(tuple(metrics))


def evaluate_metrics(truth, result, *, metrics=None):
    """Оценить прогнозы на доступных контрольных точках каждого канала.

    Parameters
    ----------
    truth : EvaluationTruth
        Ответы, целевая маска и привязка панели. Скрытые ответы
        используются только для оценки.
    result : ImputationResult
        Прогнозы с той же привязкой панели и маской доступности.
    metrics : MetricSet, optional
        Метрики; по умолчанию ``rmse``, ``mae`` и ``bias``. Пользовательские функции
        могут задавать собственные формулы и веса.

    Returns
    -------
    report : dict
        Числа ``requested``, ``known_truth``, ``available`` и ``scored``; результаты и
        единицы метрик по каналам и парам пост/канал. Общие ошибки и
        ``unit`` определены только для одного канала.

    Notes
    -----
    В расчёт ошибок входят одновременно целевые точки, конечные ответы и
    доступные прогнозы. Встроенные метрики дают им равные веса. Различные
    каналы не объединяются в одну ошибку даже при одинаковых единицах.
    Число ``available`` учитывает доступные прогнозы на целевых точках, включая
    точки без известного ответа; ``scored`` считает точки с обоими значениями.
    """
    if (not isinstance(truth, EvaluationTruth) or not isinstance(result, ImputationResult) or
            truth.values.shape != result.prediction.shape):
        raise DataContractError("evaluation requires bound truth and prediction layouts")
    result.require_binding(truth)
    metrics = build_metrics() if metrics is None else metrics
    if not isinstance(metrics, MetricSet):
        raise DataContractError("evaluation metrics must be a MetricSet")
    layout = truth.spec.to_dict()
    requested = truth.target_mask
    known = requested & np.isfinite(truth.values)
    scored = known & result.availability

    def group(selection, unit):
        """Подсчитать контрольные точки и метрики выбранной группы в её единице."""
        mask = scored[selection]
        return dict(requested=int(requested[selection].sum()),
                    known_truth=int(known[selection].sum()),
                    available=int((requested & result.availability)[selection].sum()),
                    scored=int(mask.sum()), unit=unit, metric_units=metrics.units(unit),
                    **metrics.score(truth.values[selection][mask], result.prediction[selection][mask]))

    by_channel, by_site_channel = [], []
    for channel_index, channel in enumerate(layout["channels"]):
        unit = layout["units"][channel]
        by_channel.append(dict(channel=channel, **group((slice(None), slice(None), channel_index), unit)))
        for site_index, site in enumerate(layout["sites"]):
            by_site_channel.append(dict(site=site, channel=channel,
                **group((slice(None), site_index, channel_index), unit)))
    if len(by_channel) == 1:
        flat = {key: value for key, value in by_channel[0].items() if key != "channel"}
    else:
        flat = dict(requested=int(requested.sum()), known_truth=int(known.sum()),
                    available=int((requested & result.availability).sum()), scored=int(scored.sum()),
                    unit=None, metric_units={metric.name: None for metric in metrics.metrics},
                    **{metric.name: None for metric in metrics.metrics})
    return dict(**flat, by_channel=by_channel, by_site_channel=by_site_channel)
