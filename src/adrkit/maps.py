"""Отображения из пользовательских функций с проверкой форм массивов."""
from dataclasses import dataclass

from adrkit.contracts import MapCapabilities, require_capability
from adrkit.spaces import ArraySpace

__all__ = ["map_from_functions"]


@dataclass(frozen=True)
class _FunctionMap:
    """Внутренний адаптер пользовательских функций к пространствам массивов."""
    domain: ArraySpace
    codomain: ArraySpace
    prediction: object
    tangent: object | None
    pullback: object | None

    def predict(self, x):
        x = self.domain.array(x, name="input", copy=True)
        return self.codomain.array(self.prediction(x), name="prediction", copy=True)

    def jvp(self, at, direction):
        action = require_capability(self.tangent, operation="JVP", consumer="function map")
        at = self.domain.array(at, name="at", copy=True)
        direction = self.domain.array(direction, name="direction", copy=True)
        return self.codomain.array(action(at, direction), name="JVP output", copy=True)

    def vjp(self, at, cotangent):
        action = require_capability(self.pullback, operation="VJP", consumer="function map")
        at = self.domain.array(at, name="at", copy=True)
        cotangent = self.codomain.array(cotangent, name="cotangent", copy=True)
        return self.domain.array(action(at, cotangent), name="VJP output", copy=True)


def map_from_functions(*, domain, codomain, predict, jvp=None, vjp=None):
    """Задать прогноз и его производные пользовательскими функциями.

    Parameters
    ----------
    domain, codomain : ArraySpace
        Пространства аргумента и результата.
    predict : callable
        Функция аргумента формы domain.shape с результатом формы
        codomain.shape.
    jvp : callable or None, optional
        Действие производной: принимает точку и направление формы
        domain.shape; возвращает массив формы codomain.shape.
    vjp : callable or None, optional
        Перенос ковектора: принимает точку формы domain.shape и
        ковектор формы codomain.shape; возвращает ковектор формы
        domain.shape.

    Returns
    -------
    mapping : MapCapabilities
        Прогноз и только явно переданные производные. Аргументы и
        результаты копируются и проверяются по пространствам.
    """
    if not isinstance(domain, ArraySpace) or not isinstance(codomain, ArraySpace):
        raise TypeError("domain and codomain must be ArraySpace records")
    if not callable(predict):
        raise TypeError("predict must be callable")
    for name, action in (("jvp", jvp), ("vjp", vjp)):
        if action is not None and not callable(action):
            raise TypeError(f"{name} must be callable or None")
    operation = _FunctionMap(domain, codomain, predict, jvp, vjp)
    return MapCapabilities(operation, domain, codomain,
                           operation if jvp is not None else None,
                           operation if vjp is not None else None)
