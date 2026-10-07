"""Семейство четвертичного штрафа."""
import numpy as np

from adrkit.config import number
from adrkit.contracts import ScalarCapabilities
from adrkit.config.composition import Family
from adrkit.config.schema import fields, vector


class QuarticPenalty:
    r"""Четвертичный штраф с евклидовым координатным ковектором.

    Parameters
    ----------
    center : array_like of shape (n,)
        Центр штрафа; значения копируются.
    alpha : float
        Вес штрафа.

    Notes
    -----
    Для центра :math:`c` значение и координатный ковектор равны

    .. math::

        R(x) = \frac{\alpha}{4}\sum_{i=0}^{n-1}(x_i-c_i)^4,
        \qquad (dR(x))_i = \alpha(x_i-c_i)^3.
    """
    def __init__(self, center, alpha):
        self.center = np.array(center, copy=True)
        self.alpha = alpha

    def value(self, x):
        """Вычислить значение четвертичного штрафа."""
        return self.alpha * float(np.sum((x-self.center)**4)) / 4

    def cotangent(self, x):
        """Вычислить координатный ковектор четвертичного штрафа."""
        return self.alpha * (x-self.center)**3


def validate(p, *, partial):
    """Проверить параметры семейства с учётом допустимости неполной конфигурации."""
    fields(p, {"center", "alpha"}, partial=partial)
    p = dict(p)
    if "center" in p:
        p["center"] = vector(p["center"])
    if "alpha" in p:
        p["alpha"] = number(p["alpha"], minimum=0)
    return p


def build(p, basis, domain):
    """Собрать четвертичный штраф в объявленном пространстве коэффициентов."""
    operation = QuarticPenalty(domain.array(p["center"]), p["alpha"])
    return ScalarCapabilities(operation, domain, operation)


FAMILIES = {"regularizer": {"quartic": Family(validate, build)}}

