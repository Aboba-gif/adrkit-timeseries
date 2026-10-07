"""Сборка составной обратной задачи по JSON-конфигурации."""
from dataclasses import dataclass

import numpy as np

from adrkit.contracts import MapCapabilities, require_capability
from adrkit.errors import ConfigError
from adrkit.inverse.constraints import box_bounds
from adrkit.inverse.misfit import QuadraticMisfit, covariance_metric
from adrkit.inverse.objectives import compose_maps, make_objective
from adrkit.inverse.regularization import CoefficientMetric, QuadraticPenalty, build_penalty
from adrkit.inverse.sensitivity import SensitivityLimits
from adrkit.selection import Selection
from adrkit.sources import P1Basis, sampled_source_map
from adrkit.spaces import ArraySpace

from .loader import ValidationContext, compose_config, read_configs
from .schema import (
    _empty, _h1_parameters, _l2_parameters, _p1_parameters,
    _quadratic_parameters, _selection_parameters, composition_schema,
)
from .validation import string

__all__ = ["Family", "families", "Composition", "build_source", "build_identity",
           "build_observation", "load", "assemble"]


@dataclass(frozen=True)
class Family:
    """Проверка параметров и конструктор семейства компонентов.

    Parameters
    ----------
    validate : callable
        Принимает параметры и именованный аргумент partial;
        возвращает нормализованный словарь.
    build : callable
        Строит компонент из проверенных параметров; дополнительные
        аргументы определяет роль.
    """
    validate: object  # Проверка семейства возвращает нормализованные параметры.
    build: object


def build_source(p):
    """Создать P1-базис и отображение источника по параметрам.

    Parameters
    ----------
    p : dict
        Проверенные knots, times, time_unit, coefficient_unit,
        value_unit и положительный scale.

    Returns
    -------
    basis : P1Basis
        Базис заданных временных узлов.
    source : MapCapabilities
        Отображение коэффициентов в точечные значения на times.
    """
    basis = P1Basis(p["knots"], time_unit=p["time_unit"])
    domain = ArraySpace((basis.size,), axes=("coefficient",), coordinates=(p["knots"],),
                        units=p["coefficient_unit"])
    codomain = ArraySpace((len(p["times"]),), axes=("time",), coordinates=(p["times"],),
                          units=p["value_unit"])
    return basis, sampled_source_map(basis, p["times"], scale=p["scale"], domain=domain, codomain=codomain)


class _Identity:
    """Внутренний тождественный прогноз с такими же производными."""
    def predict(self, x):
        return np.array(x, copy=True)

    def jvp(self, at, direction):
        return np.array(direction, copy=True)

    def vjp(self, at, cotangent):
        return np.array(cotangent, copy=True)


def build_identity(p, domain):
    """Создать тождественное отображение пространства.

    Parameters
    ----------
    p : dict
        Проверенные параметры семейства; ожидается пустой словарь.
    domain : ArraySpace
        Общее пространство аргумента и результата.

    Returns
    -------
    mapping : MapCapabilities
        Независимые копии аргумента и действий JVP/VJP.
    """
    operation = _Identity()
    return MapCapabilities(operation, domain, domain, operation, operation)


class _SelectionMap:
    """Внутренний выбор строк и дополнение ковектора нулями."""
    def __init__(self, selection):
        self.selection = selection

    def predict(self, x):
        return self.selection.take(x)

    def jvp(self, at, direction):
        return self.selection.take(direction)

    def vjp(self, at, cotangent):
        return self.selection.scatter(cotangent)


def build_observation(p, domain):
    """Создать оператор выбора строк по параметрам.

    Parameters
    ----------
    p : dict
        Проверенное поле indices с уникальными индексами строк.
    domain : ArraySpace
        Полное векторное пространство наблюдений.

    Returns
    -------
    mapping : MapCapabilities
        Выбор в порядке indices; VJP дополняет ковектор нулями.
    """
    selection = Selection(p["indices"], domain)
    operation = _SelectionMap(selection)
    return MapCapabilities(operation, domain, selection.selected_layout, operation, operation)


def families(inputs=None):
    """Получить конструкторы встроенных и пользовательских семейств.

    Parameters
    ----------
    inputs : Mapping or None, optional
        Дополнительные Family по роли и имени; встроенные имена не
        заменяются.

    Returns
    -------
    families : dict
        Новая таблица source, surrogate, observation и regularizer
        для одной сборки.
    """
    result = {
        "source": {"p1": Family(_p1_parameters, build_source)},
        "surrogate": {"identity": Family(_empty, build_identity)},
        "observation": {"identity": Family(_empty, build_identity),
                        "selection": Family(_selection_parameters, build_observation)},
        "regularizer": {
            "l2": Family(_l2_parameters, lambda p, b, d: build_penalty("l2", b, p["center"], p["alpha"], tau=p["tau"], domain=d)),
            "h1": Family(_h1_parameters, lambda p, b, d: build_penalty("h1", b, p["center"], p["alpha"], tau=p["tau"], domain=d)),
            "quadratic": Family(_quadratic_parameters, lambda p, b, d: QuadraticPenalty(p["center"], p["gram"], p["alpha"]).capabilities(d)),
        },
    }
    for role, additional in (inputs or {}).items():
        if role not in result or not isinstance(additional, dict):
            raise ConfigError(f"unknown family role: {role}")
        for kind, family in additional.items():
            string(kind)
            if kind in result[role] or not isinstance(family, Family) or not callable(family.validate) or not callable(family.build):
                raise ConfigError(f"invalid/duplicate family: {role}/{kind}")
            result[role][kind] = family
    return result


@dataclass(frozen=True)
class Composition:
    """Компоненты и настройки составной обратной задачи.

    Parameters
    ----------
    prediction : MapCapabilities
        Составной прогноз.
    objective : ScalarCapabilities
        Потери и штраф на пространстве параметров.
    coefficient_metric : CoefficientMetric
        Метрика пространства параметров.
    observation_metric : CovarianceMetric
        Метрика наблюдений.
    initial : ndarray
        Начальная точка формы prediction.domain.shape.
    science : dict
        Нормализованные научные параметры, включая действие и
        настройки решения.
    """
    prediction: object
    objective: object
    coefficient_metric: object
    observation_metric: object
    initial: np.ndarray
    science: dict


def _assemble_composition(science, inputs):
    available = families(inputs)

    def selected(role):
        spec = science[role]
        return available[role][spec["kind"]].build, spec["parameters"]

    builder, params = selected("source")
    basis, source = builder(params)
    builder, params = selected("surrogate")
    surrogate = builder(params, source.codomain)
    builder, params = selected("observation")
    observation = builder(params, surrogate.codomain)
    prediction = compose_maps(source, surrogate, observation)
    builder, params = selected("regularizer")
    penalty = builder(params, basis, prediction.domain)
    metric = covariance_metric(science["covariance"], layout=prediction.codomain)
    misfit = QuadraticMisfit(science["target"], metric)
    objective = make_objective(prediction, None, misfit, penalty)
    coefficient_metric = CoefficientMetric(science["coefficient_metric"], domain=prediction.domain)
    initial = prediction.domain.array(science["initial"], copy=True)
    box_bounds(science["bounds"], initial.size)
    if science["action"] == "solve":
        require_capability(objective.cotangent, operation="smooth cotangent/VJP", consumer="solve")
    if science["action"] == "sensitivity":
        require_capability(prediction.jvp, operation="JVP", consumer="sensitivity")
        p = science["sensitivity"]
        limits = SensitivityLimits(**{k: p[k] for k in ("max_columns", "max_entries", "max_jvp_calls")})
        n, m = prediction.domain.size, prediction.codomain.size
        if n > limits.max_columns or n > limits.max_jvp_calls or n*m > limits.max_entries:
            raise ConfigError("dense sensitivity exceeds configured limits")
    return Composition(prediction, objective, coefficient_metric, metric, initial, science)


def load(files, *, inputs=None, overrides=(), config_root=None, output_root=None):
    """Прочитать и объединить конфигурации составной задачи.

    Parameters
    ----------
    files : sequence of str or pathlib.Path
        Непустой упорядоченный набор JSON-файлов.
    inputs : Mapping or None, optional
        Пользовательские семейства для схемы.
    overrides : sequence of str, optional
        Замены в формате /pointer=JSON.
    config_root : str, pathlib.Path or None, optional
        Абсолютный корень относительных имён и замен.
    output_root : str, pathlib.Path or None, optional
        Переопределённый каталог результата.

    Returns
    -------
    config : EffectiveConfig
        Объединённая конфигурация схемы составной задачи.
    """
    documents = read_configs(files, config_root=config_root)
    schema = composition_schema(families=families(inputs))
    return compose_config(documents, schema=schema, overrides=overrides,
                          config_root=config_root, output_root=output_root)


def assemble(config, *, inputs=None):
    """Создать компоненты составной обратной задачи.

    Parameters
    ----------
    config : EffectiveConfig
        Конфигурация adrkit.composition/1.
    inputs : Mapping or None, optional
        Пользовательские семейства для создания компонентов.

    Returns
    -------
    composition : Composition
        Прогноз, функционал, метрики и начальная точка; расчёт ещё не
        запускается.

    Notes
    -----
    Решение требует VJP, анализ чувствительности — JVP.
    Применяются пределы плотной чувствительности из конфигурации.
    """
    tree = config.to_dict()
    schema = composition_schema(families=families(inputs))
    if (tree["pipeline"] != schema.pipeline
            or tree["science_schema_version"] != schema.version):
        raise ConfigError("configuration must use the ADR composition schema")
    science = schema.normalize(tree["science"], ValidationContext(partial=False))
    return _assemble_composition(science, inputs)
