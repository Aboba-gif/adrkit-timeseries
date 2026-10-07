"""Композиция прогнозов и скалярных функционалов."""
import numpy as np

from adrkit.contracts import MapCapabilities, ScalarCapabilities, require_capability
from .misfit import QuadraticMisfit
from .regularization import QuadraticPenalty
from ._numeric import real_scalar


class _Composition:
    """Внутренняя цепочка отображений с координатными производными."""
    def __init__(self, stages):
        self.stages = stages

    def _trace(self, x):
        values = [self.stages[0].domain.array(x, copy=True)]
        for stage in self.stages:
            values.append(stage.codomain.array(stage.predict.predict(values[-1].copy()), copy=True))
        return values

    def predict(self, x):
        return self._trace(x)[-1]

    def jvp(self, at, direction):
        h = self.stages[0].domain.array(direction, copy=True)
        values = self._trace(at)
        for stage, x in zip(self.stages, values):
            operation = require_capability(stage.jvp, operation="JVP", consumer="composition")
            h = stage.codomain.array(operation.jvp(x.copy(), h.copy()), copy=True)
        return h

    def vjp(self, at, cotangent):
        g = self.stages[-1].codomain.array(cotangent, copy=True)
        values = self._trace(at)
        for stage, x in reversed(list(zip(self.stages, values))):
            operation = require_capability(stage.vjp, operation="VJP", consumer="composition")
            g = stage.domain.array(operation.vjp(x.copy(), g.copy()), copy=True)
        return g


def compose_maps(*stages):
    r"""Составить отображения в порядке применения.

    Parameters
    ----------
    *stages
        Непустая последовательность MapCapabilities; соседние
        пространства codomain и domain совместимы.

    Returns
    -------
    mapping : MapCapabilities
        Композиция от domain первого этапа к codomain последнего; JVP
        или VJP доступны при наличии на всех этапах.

    Notes
    -----
    Для двух этапов :math:`F_1,F_2` результат равен

    .. math::

        F(x)=F_2(F_1(x)).

    JVP применяется по этапам от первого к последнему.
    VJP переносит ковектор в обратном порядке по правилу цепочки.
    """
    if not stages or any(not isinstance(s, MapCapabilities) for s in stages):
        raise TypeError("compose_maps requires explicit MapCapabilities")
    for left, right in zip(stages, stages[1:]):
        left.codomain.require_compatible(right.domain)
    operation = _Composition(stages)
    return MapCapabilities(operation, stages[0].domain, stages[-1].codomain,
                           operation if all(s.jvp is not None for s in stages) else None,
                           operation if all(s.vjp is not None for s in stages) else None)


class _Objective:
    """Внутренняя композиция прогноза, невязки и штрафа."""
    def __init__(self, prediction, selection, misfit, penalty):
        self.prediction, self.selection = prediction, selection
        self.misfit, self.penalty = misfit, penalty

    def _prediction(self, x):
        x = self.prediction.domain.array(x, copy=True)
        y = self.prediction.codomain.array(self.prediction.predict.predict(x.copy()), copy=True)
        return x, y if self.selection is None else self.selection.take(y)

    def _value(self, x, selected):
        value = _scalar_value(self.misfit.value.value(selected.copy()))
        if self.penalty is not None:
            value += _scalar_value(self.penalty.value.value(x.copy()))  # alpha уже включена в штраф.
        return float(value)

    def _cotangent(self, x, selected):
        operation = require_capability(self.misfit.cotangent, operation="misfit cotangent", consumer="objective")
        pullback = require_capability(self.prediction.vjp, operation="VJP", consumer="objective")
        g = self.misfit.domain.array(operation.cotangent(selected.copy()), copy=True)
        if self.selection is not None:
            g = self.selection.scatter(g)
        g = self.prediction.domain.array(pullback.vjp(x.copy(), g.copy()), copy=True)
        if self.penalty is not None:
            derivative = require_capability(self.penalty.cotangent, operation="penalty cotangent", consumer="objective")
            g += self.penalty.domain.array(derivative.cotangent(x.copy()))
        return g

    def value(self, x):
        return self._value(*self._prediction(x))

    def cotangent(self, x):
        return self._cotangent(*self._prediction(x))

    def value_and_cotangent(self, x):
        x, selected = self._prediction(x)
        if (self.misfit.value_and_cotangent is not None or
                (self.penalty is not None and self.penalty.value_and_cotangent is not None)):
            value, data_covector = _scalar_pair(self.misfit, selected)
            pullback = require_capability(self.prediction.vjp, operation="VJP", consumer="objective")
            g = self.misfit.domain.array(data_covector, copy=True)
            if self.selection is not None:
                g = self.selection.scatter(g)
            g = self.prediction.domain.array(pullback.vjp(x.copy(), g.copy()), copy=True)
            if self.penalty is not None:
                penalty_value, penalty_covector = _scalar_pair(self.penalty, x)
                value += penalty_value
                g += self.penalty.domain.array(penalty_covector)
            return float(value), g
        return self._value(x, selected), self._cotangent(x, selected)


def _scalar_value(value):
    """Вернуть значение функционала; бесконечности допустимы, NaN отклоняется."""
    scalar = real_scalar(value, name="Scalar functional", finite=False)
    if np.isnan(scalar):
        raise ValueError("Scalar functional must return a real scalar other than NaN")
    return scalar


def _scalar_pair(functional, point):
    """Вычислить значение и ковектор совместно или двумя вызовами."""
    if functional.value_and_cotangent is not None:
        value, covector = functional.value_and_cotangent.value_and_cotangent(
            functional.domain.array(point, copy=True))
        return _scalar_value(value), functional.domain.array(covector, copy=True)
    derivative = require_capability(functional.cotangent, operation="cotangent", consumer="objective")
    value = _scalar_value(functional.value.value(functional.domain.array(point, copy=True)))
    covector = derivative.cotangent(functional.domain.array(point, copy=True))
    return value, functional.domain.array(covector, copy=True)


def make_objective(prediction, selection, misfit, penalty=None):
    r"""Составить невязку прогноза и необязательный штраф.

    Parameters
    ----------
    prediction : MapCapabilities
        Прогноз из пространства параметров в пространство наблюдений.
    selection : Selection or None
        Выбор строк до вычисления невязки.
    misfit : ScalarCapabilities or QuadraticMisfit
        Функционал на пространстве выбранных наблюдений.
    penalty : ScalarCapabilities, QuadraticPenalty or None, optional
        Штраф на пространстве параметров; его вес уже включён.

    Returns
    -------
    objective : ScalarCapabilities
        Функционал на prediction.domain; ковектор требует VJP и
        производные всех скалярных компонентов.

    Notes
    -----
    Функционал имеет вид

    .. math::

        J(a)=\Phi(SF(a))+P(a),

    Здесь :math:`a` — параметры в `prediction.domain`,
    :math:`F` — `prediction`, :math:`S` — `selection`,
    :math:`\Phi` — `misfit`, :math:`P` — `penalty`.
    При `selection=None` оператор :math:`S` тождественный;
    при `penalty=None` функционал :math:`P` равен нулю.
    """
    if not isinstance(prediction, MapCapabilities):
        raise TypeError("prediction must declare MapCapabilities")
    if isinstance(misfit, QuadraticMisfit):
        misfit = misfit.capabilities()
    if isinstance(penalty, QuadraticPenalty):
        penalty = penalty.capabilities(prediction.domain)
    if not isinstance(misfit, ScalarCapabilities) or (penalty is not None and
                                                      not isinstance(penalty, ScalarCapabilities)):
        raise TypeError("scalar components must declare ScalarCapabilities")
    layout = prediction.codomain
    if selection is not None:
        layout.require_compatible(selection.full_layout)
        layout = selection.selected_layout
    layout.require_compatible(misfit.domain)
    if penalty is not None:
        prediction.domain.require_compatible(penalty.domain)
    operation = _Objective(prediction, selection, misfit, penalty)
    smooth = (prediction.vjp is not None and misfit.cotangent is not None
              and (penalty is None or penalty.cotangent is not None))
    return ScalarCapabilities(operation, prediction.domain, operation if smooth else None,
                              value_and_cotangent=operation if smooth else None)


def evaluate_score(objective, x):
    """Вычислить значение заданного функционала.

    Parameters
    ----------
    objective : ScalarCapabilities
        Функционал с объявленным пространством аргумента.
    x : array_like
        Точка формы objective.domain.shape.

    Returns
    -------
    value : float
        Скалярное значение; бесконечности допустимы, NaN отклоняется.
    """
    if not isinstance(objective, ScalarCapabilities):
        raise TypeError("objective must declare ScalarCapabilities")
    result = float(objective.value.value(objective.domain.array(x, copy=True)))
    if np.isnan(result):
        raise ValueError("Scalar score returned NaN")
    return result
