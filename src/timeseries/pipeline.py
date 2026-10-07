"""Обучение, заполнение, оценка и запись результатов временных рядов."""
from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
import pandas as pd

from timeseries.contracts import (
    AcceptedValue, DataContractError, EvaluationTruth, ImputationResult, MaskedInputs,
    QuarantinedValue, SourceProvenance, closed, conceal, text,
)
from timeseries.imputation import build_imputer, input_layout
from timeseries.metrics import MetricSet, build_metrics, evaluate_metrics


@dataclass(frozen=True)
class Population:
    """Связать видимые входы выборки с контрольными ответами и происхождением.

    Parameters
    ----------
    inputs : MaskedInputs
        Входы с наблюдениями, масками и привязкой протокола/выборки.
    truth : EvaluationTruth
        Контрольные ответы с той же привязкой панели. Целевые точки
        совпадают с объединением ``artificial_mask`` и ``mit_mask`` входов.
    provenance : tuple of SourceProvenance, optional
        Источники и сведения загрузки; по умолчанию пустой кортеж.
    quarantine : tuple of QuarantinedValue, optional
        Отброшенные значения с причинами и координатами; по умолчанию
        пустой кортеж.
    row_origins : tuple of AcceptedValue, optional
        Происхождение принятых значений; по умолчанию пустой кортеж.

    Notes
    -----
    Метод заполнения получает только ``inputs``. Контрольные ответы используются
    отдельно при оценке; искусственно скрытые и MIT-значения не становятся
    наблюдениями для ``fit`` или ``predict``.
    """
    inputs: MaskedInputs
    truth: EvaluationTruth
    provenance: tuple = ()
    quarantine: tuple = ()
    row_origins: tuple = ()

    def __post_init__(self):
        x, y = self.inputs, self.truth
        if not isinstance(x, MaskedInputs) or not isinstance(y, EvaluationTruth):
            raise DataContractError("population requires separate inputs and truth")
        if (x.spec != y.spec or x.times != y.times or
                any(getattr(x, k) != getattr(y, k) for k in
                    ("protocol_id", "population_id", "population_kind")) or
                not np.array_equal(x.artificial_mask | x.mit_mask, y.target_mask)):
            raise DataContractError("truth/population binding mismatch")
        sources = tuple(self.provenance)
        if any(not isinstance(p, SourceProvenance) for p in sources):
            raise DataContractError("population provenance requires SourceProvenance")
        object.__setattr__(self, "provenance", sources)
        for name, record_type in (("quarantine", QuarantinedValue), ("row_origins", AcceptedValue)):
            records = tuple(getattr(self, name))
            if any(not isinstance(row, record_type) for row in records):
                raise DataContractError(f"{name} requires {record_type.__name__} records")
            object.__setattr__(self, name, records)


def load_population(data, *, artificial_mask, mit_mask, protocol_id,
                    population_id, population_kind):
    """Загрузить панель и разделить входы заполнения и контрольные ответы.

    Parameters
    ----------
    data : DataSource
        Загрузчик панели с происхождением принятых и отброшенных
        значений.
    artificial_mask : array_like of bool, shape (n_times, n_sites, n_channels)
        Искусственно скрываемые конечные наблюдения; передаются только в
        ``truth``.
    mit_mask : array_like of bool, shape (n_times, n_sites, n_channels)
        Скрываемые исходные наблюдения для ``train``; не пересекаются с
        ``artificial_mask``. Для остальных назначений маска должна быть
        пустой.
    protocol_id : str
        Идентификатор протокола для привязки входов и ответов.
    population_id : str
        Идентификатор выборки.
    population_kind : {"train", "calibration", "selection", "test", "export"}
        Назначение выборки.

    Returns
    -------
    population : Population
        Видимые входы и скрытые ответы одной панели, вместе с
        ``provenance``, карантином и происхождением строк.
    """
    from timeseries.data import DataSource
    if not isinstance(data, DataSource):
        raise DataContractError("load_population requires build_data's validated source")
    loaded = data.load()
    inputs, truth = conceal(loaded.panel, artificial_mask=artificial_mask, mit_mask=mit_mask,
                            protocol_id=protocol_id, population_id=population_id,
                            population_kind=population_kind)
    return Population(inputs, truth, loaded.provenance, loaded.quarantine, loaded.panel.row_origins)


def evaluate_prediction(population, result, *, metrics=None):
    """Оценить заполнение на контрольных ответах данной выборки.

    Parameters
    ----------
    population : Population
        Выборка с входами, целевой маской и отдельными ответами.
    result : ImputationResult
        Результат с той же привязкой панели. Наблюдаемые значения и их
        статусы должны быть сохранены.
    metrics : MetricSet, optional
        Набор метрик; по умолчанию ``rmse``, ``mae`` и ``bias``.

    Returns
    -------
    report : dict
        Числа доступных точек и метрики по каналам и парам пост/канал в
        их единицах; структура определена ``evaluate_metrics``.
    """
    _check_prediction(population.inputs, result)
    return evaluate_metrics(population.truth, result, metrics=metrics)


def _check_prediction(inputs, result):
    """Проверить привязку результата и сохранение исходных наблюдений и статусов."""
    if not isinstance(result, ImputationResult):
        raise DataContractError("predict must return an ImputationResult")
    result.require_binding(inputs)
    if (result.prediction.shape != inputs.values.shape or
            not np.array_equal(result.prediction[inputs.visible], inputs.values[inputs.visible]) or
            not np.array_equal(result.status == "observed", inputs.visible)):
        raise DataContractError("predict must retain visible values/status and the declared layout")


@dataclass(frozen=True)
class PipelineResult:
    """Хранить состояние метода, заполнения и оценки выбранных стадий.

    Parameters
    ----------
    fitted : object
        Состояние, возвращённое ``fit``, либо ``no_fit_state`` метода без
        обучения.
    calibration : ImputationResult or None
        Заполнение калибровочной выборки; ``None``, если стадия не выбрана.
    test : ImputationResult or None
        Заполнение тестовой выборки; ``None``, если стадия не выбрана.
    export : ImputationResult or None
        Заполнение экспортной выборки; ``None``, если стадия не выбрана.
    metrics : dict
        Отчёты ``evaluate_prediction`` по выбранным стадиям, включая ``export``.
        Неизвестные ответы не участвуют в ошибках.
    output_root : pathlib.Path or None
        Каталог записанных результатов; ``None`` при выполнении без
        сохранения.
    """
    fitted: object
    calibration: ImputationResult | None
    test: ImputationResult | None
    export: ImputationResult | None
    metrics: dict
    output_root: Path | None


@dataclass(frozen=True)
class FamilyPipeline:
    """Выполнять заполнение и оценку выборок одного протокола.

    Parameters
    ----------
    protocol_id : str
        Общий идентификатор протокола выбранных стадий.
    seed : int
        Начальное состояние генератора, передаваемого ``fit``;
        ``build_pipeline`` требует неотрицательное целое.
    method_json : str
        JSON спецификации метода, записываемый в метаданные результата.
    imputer : object
        Метод с ``predict(inputs, fitted=...)``. Обучаемый метод
        предоставляет ``fit(inputs, rng=...)``, метод без обучения —
        ``no_fit_state``. Реализация может быть внешней.
    metrics : MetricSet, optional
        Набор метрик; по умолчанию ``rmse``, ``mae`` и ``bias``. Внешние формулы и
        веса допускаются через ``Metric``.
    metric_specs_json : str, optional
        JSON спецификаций метрик для метаданных; по умолчанию ``"null"``.

    Notes
    -----
    Для сборки по спецификации предназначен ``build_pipeline``.
    Один ``fitted`` используется на всех выбранных стадиях. Метод получает
    ``MaskedInputs``; контрольные ответы передаются только функции оценки.
    """
    protocol_id: str
    seed: int
    method_json: str
    imputer: object
    metrics: MetricSet = field(default_factory=build_metrics)
    metric_specs_json: str = "null"

    def run(self, *, train, calibration, test, export=None, output_root=None):
        """Обучить метод один раз и заполнить выбранные выборки.

        Parameters
        ----------
        train : Population or None
            Обучающая выборка для метода с ``fit``. Для метода без обучения
            требуется ``None``.
        calibration : Population or None
            Калибровочная выборка либо ``None``, если стадия не используется.
        test : Population or None
            Тестовая выборка либо ``None``, если стадия не используется.
        export : Population or None, optional
            Экспортная выборка; по умолчанию ``None``. Может пересекаться по
            времени с другими стадиями.
        output_root : path-like or None, optional
            Новый абсолютный каталог результатов вне каталогов источников.
            ``None`` отключает запись файлов.

        Returns
        -------
        result : PipelineResult
            Общее ``fitted``, заполнения выбранных стадий, их оценки и каталог
            сохранения.

        Notes
        -----
        Нужна хотя бы одна из стадий ``calibration``, ``test`` или ``export``. Все выборки
        имеют один ``protocol_id`` и одинаковую структуру панели, но разные
        ``population_id``. Временные диапазоны ``train``, ``calibration`` и ``test`` не
        пересекаются; их хронологический порядок не задаётся.

        Сначала проверяются стадии и каталог, затем вызываются ``fit``, ``predict`` и
        оценка. Файлы записываются после успешного вычисления всех результатов.
        CSV содержит значения, маски и статусы; ``standard_deviation`` добавляется
        только при наличии SD в результате метода, в единице канала. SD не
        пересчитывается. Обученное состояние в JSON не сериализуется.
        """
        stages = {k: p for k, p in (("train", train), ("calibration", calibration),
                                   ("test", test), ("export", export)) if p is not None}
        if not any(k in stages for k in ("calibration", "test", "export")):
            raise DataContractError("at least one prediction population is required")
        layouts, ids = set(), set()
        for kind, population in stages.items():
            if not isinstance(population, Population):
                raise DataContractError("stages require Population records")
            inputs = population.inputs
            if inputs.protocol_id != self.protocol_id or inputs.population_kind != kind:
                raise DataContractError("pipeline protocol/population kind mismatch")
            if inputs.population_id in ids:
                raise DataContractError("population IDs must be distinct")
            ids.add(inputs.population_id)
            layouts.add(input_layout(inputs))
        if len(layouts) != 1:
            raise DataContractError("population axis/clock/units mismatch")
        temporal = [p.inputs for k, p in stages.items() if k != "export"]
        for i, a in enumerate(temporal):
            for b in temporal[i+1:]:
                if a.times[0] <= b.times[-1] and b.times[0] <= a.times[-1]:
                    raise DataContractError("train/calibration/test context ranges overlap")
        fit = getattr(self.imputer, "fit", None)
        if callable(fit):
            if train is None:
                raise DataContractError("train is required by this fitted family")
        elif train is not None or not hasattr(self.imputer, "no_fit_state"):
            raise DataContractError("no-fit family requires train=None and an explicit no-fit state")
        root = _validate_output(output_root, stages) if output_root is not None else None
        fitted = fit(train.inputs, rng=np.random.default_rng(self.seed)) if callable(fit) else self.imputer.no_fit_state
        predictions, metrics = {}, {}
        for kind, population in stages.items():
            if kind == "train":
                continue
            result = self.imputer.predict(population.inputs, fitted=fitted)
            _check_prediction(population.inputs, result)
            predictions[kind] = result
            metrics[kind] = evaluate_prediction(population, result, metrics=self.metrics)
        if root is not None:
            _write_results(root, self, stages, predictions, metrics)
        return PipelineResult(fitted, predictions.get("calibration"), predictions.get("test"),
                              predictions.get("export"), metrics, root)


def build_pipeline(spec, *, families=None, metric_families=None):
    """Собрать метод и метрики по спецификации протокола.

    Parameters
    ----------
    spec : dict
        Спецификация с ``protocol_id``, неотрицательным целым ``seed``, ``imputer``
        и необязательным ``metrics``. Содержит только конечные JSON-
        значения.
    families : dict of str to ImputerFamily, optional
        Дополнительные семейства заполнения для ``build_imputer``.
    metric_families : dict of str to MetricFamily, optional
        Дополнительные семейства метрик для ``build_metrics``.

    Returns
    -------
    pipeline : FamilyPipeline
        Метод и метрики с сохранёнными JSON-спецификациями. Данные ещё
        не прочитаны, ``fit`` и ``predict`` не вызваны.
    """
    closed(spec, {"protocol_id", "seed", "imputer", "metrics"}, optional={"metrics"}, where="pipeline")
    protocol = text(spec["protocol_id"], "protocol_id")
    if type(spec["seed"]) is not int or spec["seed"] < 0:
        raise DataContractError("seed must be a nonnegative integer")
    try:
        method_json = json.dumps(spec["imputer"], sort_keys=True, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise DataContractError("imputer must be finite JSON") from exc
    
    imputer = build_imputer(json.loads(method_json), families=families)
    metric_specs = spec.get("metrics")
    metrics = build_metrics(metric_specs, families=metric_families)
    metric_specs_json = json.dumps(metric_specs, sort_keys=True, allow_nan=False)
    return FamilyPipeline(protocol, spec["seed"], method_json, imputer, metrics, metric_specs_json)


def _validate_output(output_root, stages):
    """Проверить новый абсолютный каталог результатов вне каталогов источников."""
    root = Path(output_root)
    if not root.is_absolute():
        raise DataContractError("output_root must be absolute")
    if root.is_symlink() or root.exists():
        raise DataContractError("output_root must be new; existing results are never overwritten")
    root = root.resolve()
    for population in stages.values():
        for source in population.provenance:
            path = Path(source.resolved_path).resolve()
            if root.is_relative_to(path.parent) or path.is_relative_to(root):
                raise DataContractError("output_root overlaps an input directory")
    return root


def _write_results(root, pipeline, stages, predictions, metrics):
    """Записать метаданные и CSV выбранных стадий, сохраняя SD метода без пересчёта."""
    from dataclasses import asdict
    root = _validate_output(root, stages)
    metadata = dict(protocol_id=pipeline.protocol_id, imputer=json.loads(pipeline.method_json),
                    metric_configuration=json.loads(pipeline.metric_specs_json),
                    metric_definitions=[dict(name=metric.name, unit=metric.unit)
                                        for metric in pipeline.metrics.metrics],
                    seed=pipeline.seed, fit_population=stages["train"].inputs.population_id if "train" in stages else None,
                    metrics=metrics, populations={})
    root.mkdir(parents=True, exist_ok=False)
    created = []
    try:
        for kind, population in stages.items():
            inputs = population.inputs
            layout = inputs.spec.to_dict()
            metadata["populations"][kind] = dict(
                id=inputs.population_id, panel=layout,
                sources=[dict(spec=p.spec.to_dict(), resolved_path=p.resolved_path, byte_count=p.byte_count)
                         for p in population.provenance],
                quarantine=[asdict(row) for row in population.quarantine],
                row_origins=[asdict(row) for row in population.row_origins],
            )
            if kind not in predictions:
                continue
            result = predictions[kind]
            axes = pd.MultiIndex.from_product([inputs.times, layout["sites"], layout["channels"]],
                                               names=["time", "site", "channel"])
            frame = axes.to_frame(index=False)
            columns = {"visible_value": inputs.values, "row_present": inputs.row_present,
                       "original_observed": inputs.original_observed, "artificial_mask": inputs.artificial_mask,
                       "mit_mask": inputs.mit_mask, "prediction": result.prediction,
                       "availability": result.availability, "status": result.status,
                       "is_imputed": ~inputs.visible & result.availability}
            if result.uncertainty is not None:
                columns["standard_deviation"] = result.uncertainty
            for name, values in columns.items():
                frame[name] = values.reshape(-1)
            path = root / f"{kind}.csv"
            with path.open("x", encoding="utf-8", newline="") as stream:
                created.append(path)
                frame.to_csv(stream, index=False)
        path = root / "pipeline.json"
        with path.open("x", encoding="utf-8") as stream:
            created.append(path)
            json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    except Exception as exc:
        for path in reversed(created):
            try:
                path.unlink()
            except OSError as cleanup_error:
                exc.add_note(f"could not remove {path}: {cleanup_error}")
        try:
            root.rmdir()
        except OSError as cleanup_error:
            exc.add_note(f"could not remove {root}: {cleanup_error}")
        raise
