"""Семейство tanh с явно заданными производными."""
import numpy as np

from adrkit.config import number
from adrkit.contracts import MapCapabilities
from adrkit.config.composition import Family
from adrkit.config.schema import fields


class TanhSurrogate:
    r"""Покомпонентный гиперболический тангенс с заданным масштабом.

    Parameters
    ----------
    gain : float
        Масштаб аргумента гиперболического тангенса.

    Notes
    -----
    Значение отображения и диагональный якобиан равны

    .. math::

        T_i(x) = \tanh(g x_i),\qquad
        J_{ii}(x) = g\bigl(1-\tanh^2(g x_i)\bigr),

    где :math:`g` — ``gain``. Функция ``build`` требует безразмерный вход.
    """
    def __init__(self, gain):
        self.gain = gain

    def predict(self, x):
        """Вычислить покомпонентный гиперболический тангенс."""
        return np.tanh(self.gain*x)

    def jvp(self, at, direction):
        """Умножить направление на диагональный якобиан."""
        return self.gain * (1-self.predict(at)**2) * direction

    def vjp(self, at, cotangent):
        """Умножить ковектор на транспонированный диагональный якобиан."""
        return self.gain * (1-self.predict(at)**2) * cotangent


def validate(p, *, partial):
    """Проверить параметры семейства с учётом допустимости неполной конфигурации."""
    fields(p, {"gain", "capabilities"}, partial=partial)
    p = dict(p)
    if "gain" in p:
        p["gain"] = number(p["gain"], positive=True)
    if "capabilities" in p and p["capabilities"] not in ("predict", "jvp", "jvp_vjp"):
        raise ValueError("unknown declared capabilities")
    return p


def build(p, domain):
    """Собрать отображение tanh с явно выбранными производными."""
    if domain.units != "1":
        raise ValueError("this tanh diagnostic requires dimensionless inputs")
    operation = TanhSurrogate(p["gain"])
    return MapCapabilities(operation, domain, domain,
        operation if p["capabilities"] in ("jvp", "jvp_vjp") else None,
        operation if p["capabilities"] == "jvp_vjp" else None)


FAMILIES = {"surrogate": {"tanh": Family(validate, build)}}

