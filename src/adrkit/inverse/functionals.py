"""Скалярные функционалы и операции их композиции."""
from dataclasses import dataclass

import numpy as np

from adrkit.contracts import MapCapabilities, ScalarCapabilities, require_capability
from adrkit.spaces import ArraySpace
from .objectives import make_objective
from ._numeric import real_scalar

__all__ = ["scalar_from_functions", "sum_functionals", "scale_functional",
           "pullback_functional"]


def _check_operation(operation, method, *, name):
    if not callable(getattr(operation, method, None)):
        raise TypeError(f"{name} must provide callable {method}()")


def _check_scalar(functional):
    if not isinstance(functional, ScalarCapabilities):
        raise TypeError("functional must declare ScalarCapabilities")
    if not isinstance(functional.domain, ArraySpace):
        raise TypeError("functional domain must be an ArraySpace record")
    _check_operation(functional.value, "value", name="functional value")
    for method in ("cotangent", "prox", "value_and_cotangent"):
        operation = getattr(functional, method)
        if operation is not None:
            _check_operation(operation, method, name=f"functional {method}")


def _check_map(mapping):
    if not isinstance(mapping, MapCapabilities):
        raise TypeError("mapping must declare MapCapabilities")
    if not isinstance(mapping.domain, ArraySpace) or not isinstance(mapping.codomain, ArraySpace):
        raise TypeError("mapping domain and codomain must be ArraySpace records")
    _check_operation(mapping.predict, "predict", name="mapping prediction")
    for method in ("jvp", "vjp"):
        operation = getattr(mapping, method)
        if operation is not None:
            _check_operation(operation, method, name=f"mapping {method}")


@dataclass(frozen=True)
class _FunctionScalar:
    """Адаптер пользовательских скалярных операций с проверкой пространства."""
    domain: ArraySpace
    evaluation: object
    derivative: object | None
    proximal: object | None
    combined: object | None

    def value(self, x):
        x = self.domain.array(x, name="input", copy=True)
        return real_scalar(self.evaluation(x), name="functional value")

    def cotangent(self, x):
        action = require_capability(self.derivative, operation="cotangent", consumer="function scalar")
        x = self.domain.array(x, name="input", copy=True)
        return self.domain.array(action(x), name="functional cotangent", copy=True)

    def prox(self, x, step):
        action = require_capability(self.proximal, operation="prox", consumer="function scalar")
        step = real_scalar(step, name="prox step")
        if step <= 0:
            raise ValueError("prox step must be positive")
        x = self.domain.array(x, name="input", copy=True)
        return self.domain.array(action(x, step), name="prox output", copy=True)


    def value_and_cotangent(self, x):
        action = require_capability(self.combined, operation="value_and_cotangent", consumer="function scalar")
        x = self.domain.array(x, name="input", copy=True)
        value, covector = action(x)
        return (real_scalar(value, name="functional value"),
                self.domain.array(covector, name="functional cotangent", copy=True))


def scalar_from_functions(*, domain, value, cotangent=None, prox=None, prox_pairing=None,
                          value_and_cotangent=None):
    """Задать скалярный функционал пользовательскими функциями.

    Parameters
    ----------
    domain : ArraySpace
        Пространство аргумента функционала.
    value : callable
        Принимает точку формы domain.shape и возвращает конечный
        вещественный скаляр.
    cotangent : callable or None, optional
        Возвращает конечный координатный ковектор формы domain.shape.
    prox : callable or None, optional
        Принимает точку и положительный скаляр шага; возвращает точку
        формы domain.shape.
    prox_pairing : str or None, optional
        Описание скалярного произведения для переданной prox.
    value_and_cotangent : callable or None, optional
        Возвращает значение и ковектор одним вызовом; требует
        отдельный cotangent.

    Returns
    -------
    functional : ScalarCapabilities
        Только явно заданные операции; входы и массивы результатов
        копируются.
    """
    if not isinstance(domain, ArraySpace):
        raise TypeError("domain must be an ArraySpace record")
    if not callable(value):
        raise TypeError("value must be callable")
    for name, action in (("cotangent", cotangent), ("prox", prox),
                         ("value_and_cotangent", value_and_cotangent)):
        if action is not None and not callable(action):
            raise TypeError(f"{name} must be callable or None")
    operation = _FunctionScalar(domain, value, cotangent, prox, value_and_cotangent)
    return ScalarCapabilities(operation, domain,
                              operation if cotangent is not None else None,
                              operation if prox is not None else None,
                              prox_pairing,
                              operation if value_and_cotangent is not None else None)


def sum_functionals(*terms):
    """Сложить функционалы одного пространства.

    Parameters
    ----------
    *terms
        Непустой набор ScalarCapabilities с совместимыми domain.

    Returns
    -------
    functional : ScalarCapabilities
        Сумма значений. Ковектор доступен, если есть у всех
        слагаемых; prox не строится.
    """
    if not terms:
        raise ValueError("sum_functionals requires at least one functional")
    for term in terms:
        _check_scalar(term)
    domain = terms[0].domain
    for term in terms[1:]:
        domain.require_compatible(term.domain)

    def value(x):
        return sum(real_scalar(term.value.value(domain.array(x, copy=True)),
                                name="functional value") for term in terms)

    def cotangent(x):
        result = np.zeros(domain.shape, dtype=domain.dtype)
        for term in terms:
            action = require_capability(term.cotangent, operation="cotangent", consumer="functional sum")
            result += domain.array(action.cotangent(domain.array(x, copy=True)),
                                   name="summand cotangent")
        return result

    smooth = all(term.cotangent is not None for term in terms)
    combined = all(term.value_and_cotangent is not None for term in terms)

    def value_and_cotangent(x):
        pairs = []
        for term in terms:
            value, derivative = term.value_and_cotangent.value_and_cotangent(domain.array(x, copy=True))
            pairs.append((real_scalar(value, name="functional value"),
                          domain.array(derivative, name="summand cotangent", copy=True)))
        value = sum(item[0] for item in pairs)
        covector = np.zeros(domain.shape, dtype=domain.dtype)
        for _, derivative in pairs:
            covector += domain.array(derivative, name="summand cotangent")
        return value, covector

    return scalar_from_functions(domain=domain, value=value,
                                 cotangent=cotangent if smooth else None,
                                 value_and_cotangent=value_and_cotangent if combined else None)


def scale_functional(functional, weight):
    """Умножить функционал на вещественный вес.

    Parameters
    ----------
    functional : ScalarCapabilities
        Исходный функционал.
    weight : float
        Конечный множитель любого знака.

    Returns
    -------
    scaled : ScalarCapabilities
        Масштабированные значение и имеющиеся производные; prox не
        строится.

    Notes
    -----
    Нулевой вес не создаёт отсутствующую производную.
    """
    _check_scalar(functional)
    weight = real_scalar(weight, name="functional weight")
    domain = functional.domain

    def value(x):
        return weight * real_scalar(functional.value.value(x), name="functional value")

    def cotangent(x):
        action = require_capability(functional.cotangent, operation="cotangent", consumer="scaled functional")
        return weight * domain.array(action.cotangent(x), name="functional cotangent")

    def value_and_cotangent(x):
        value, covector = functional.value_and_cotangent.value_and_cotangent(x)
        return (weight * real_scalar(value, name="functional value"),
                weight * domain.array(covector, name="functional cotangent"))

    return scalar_from_functions(domain=domain, value=value,
                                 cotangent=cotangent if functional.cotangent is not None else None,
                                 value_and_cotangent=(value_and_cotangent
                                     if functional.value_and_cotangent is not None else None))


def pullback_functional(functional, mapping):
    r"""Составить функционал с отображением.

    Parameters
    ----------
    functional : ScalarCapabilities
        Функционал на пространстве результатов mapping.
    mapping : MapCapabilities
        Отображение с codomain, совместимым с domain функционала.

    Returns
    -------
    composed : ScalarCapabilities
        Функционал на mapping.domain; ковектор требует cotangent и
        VJP.

    Notes
    -----
    Пусть :math:`J` — `functional`, :math:`F` — `mapping`,
    а :math:`x` — точка области определения :math:`F`.
    Для дифференцируемых компонентов:

    .. math::

        d(J\circ F)(x)=DF(x)^\mathsf{T}\,dJ(F(x)).

    Формула относится к координатным ковекторам, без преобразования Рисса.
    """
    _check_scalar(functional)
    _check_map(mapping)
    mapping.codomain.require_compatible(functional.domain)
    objective = make_objective(mapping, None, functional)
    return scalar_from_functions(
        domain=objective.domain, value=objective.value.value,
        cotangent=objective.cotangent.cotangent if objective.cotangent is not None else None,
        value_and_cotangent=(objective.value_and_cotangent.value_and_cotangent
                            if objective.value_and_cotangent is not None else None),
    )
