"""Семейство оператора парного усреднения."""
import numpy as np

from adrkit.config import integer
from adrkit.contracts import MapCapabilities
from adrkit.config.composition import Family
from adrkit.config.schema import fields
from adrkit.spaces import ArraySpace


class PairAverage:
    r"""Усреднение выбранных пар координат вектора.

    Parameters
    ----------
    pairs : array_like of int, shape (n_pairs, 2)
        Индексы двух координат каждой пары; значения копируются.
    size : int
        Длина исходного вектора.

    Notes
    -----
    Для пары с индексами :math:`p,q` в строке :math:`j` массива `pairs`

    .. math::

        (Tx)_j = \frac{x_p + x_q}{2}.

    В сопряжённой операции вклады повторяющихся индексов суммируются.
    """
    def __init__(self, pairs, size):
        self.pairs = np.array(pairs, dtype=int, copy=True)
        self.size = size

    def predict(self, x):
        """Вычислить среднее каждой выбранной пары."""
        return np.mean(x[self.pairs], axis=1)

    def jvp(self, at, direction):
        """Применить линейное усреднение к направлению."""
        return self.predict(direction)

    def vjp(self, at, cotangent):
        """Суммировать половины ковектора по индексам каждой пары."""
        result = np.zeros(self.size)
        # add.at суммирует вклады повторяющихся входных индексов.
        np.add.at(result, self.pairs.ravel(), np.repeat(cotangent/2, 2))
        return result


def validate(p, *, partial):
    """Проверить параметры семейства с учётом допустимости неполной конфигурации."""
    fields(p, {"pairs"}, partial=partial)
    if "pairs" in p:
        if type(p["pairs"]) is not list or not p["pairs"]:
            raise ValueError("pairs must be a nonempty list")
        for pair in p["pairs"]:
            if type(pair) is not list or len(pair) != 2:
                raise ValueError("each pair needs two indices")
            for index in pair:
                integer(index, minimum=0)
    return dict(p)


def build(p, domain):
    """Собрать отображение усреднения пар в объявленном пространстве входа."""
    if len(domain.shape) != 1 or max(max(pair) for pair in p["pairs"]) >= domain.size:
        raise ValueError("pair indices exceed the vector layout")
    operation = PairAverage(p["pairs"], domain.size)
    codomain = ArraySpace((len(p["pairs"]),), axes=("pair_row",),
                          coordinates=(list(range(len(p["pairs"]))),), units=domain.units)
    return MapCapabilities(operation, domain, codomain, operation, operation)


FAMILIES = {"observation": {"pair_average": Family(validate, build)}}

