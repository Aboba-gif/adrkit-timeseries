"""Гладкая оптимизация явно заданного скалярного функционала."""
from scipy.optimize import minimize

from adrkit.contracts import ScalarCapabilities, require_capability
from adrkit.spaces import ArraySpace
from .constraints import box_bounds
from ._numeric import real_scalar


def solve_lbfgsb(objective, initial, *, bounds=None, options=None):
    """Минимизировать гладкий функционал методом L-BFGS-B.

    Parameters
    ----------
    objective : ScalarCapabilities
        Функционал с координатным cotangent; совместная операция
        используется при наличии.
    initial : array_like
        Конечная начальная точка формы objective.domain.shape.
    bounds : sequence of pairs or None, optional
        Покомпонентные границы в порядке C; None означает отсутствие
        границ.
    options : Mapping or None, optional
        Параметры scipy.optimize.minimize для L-BFGS-B.

    Returns
    -------
    result : scipy.optimize.OptimizeResult
        Оценка x и координатный градиент jac представлены плоскими
        массивами в порядке C.

    Notes
    -----
    Ковектор используется как евклидов градиент координат.
    Единицы аргумента и функционала задаются objective.
    """
    if not isinstance(objective, ScalarCapabilities):
        raise TypeError("objective must declare ScalarCapabilities")
    if not isinstance(objective.domain, ArraySpace):
        raise TypeError("objective domain must be an ArraySpace record")
    derivative = require_capability(objective.cotangent, operation="smooth cotangent", consumer="L-BFGS-B")
    for operation, method in ((objective.value, "value"), (derivative, "cotangent"),
                              (objective.value_and_cotangent, "value_and_cotangent")):
        if operation is not None and not callable(getattr(operation, method, None)):
            raise TypeError(f"L-BFGS-B {method} operation must provide callable {method}()")
    x = objective.domain.array(initial, copy=True)
    checked_bounds = box_bounds(bounds, x.size)

    def function(flat):
        point = objective.domain.array(flat.reshape(objective.domain.shape), copy=True)
        if objective.value_and_cotangent is not None:
            value, gradient = objective.value_and_cotangent.value_and_cotangent(point.copy())
            value = real_scalar(value, name="Smooth optimizer objective")
        else:
            value = real_scalar(objective.value.value(point.copy()), name="Smooth optimizer objective")
            gradient = derivative.cotangent(point.copy())
        return value, objective.domain.array(gradient, name="covector", copy=True).ravel()

    return minimize(function, x.ravel(), method="L-BFGS-B", jac=True,
                    bounds=checked_bounds, options=dict(options or {}))
