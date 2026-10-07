"""Семейство источника в базисе Бернштейна."""
from math import comb

import numpy as np

from adrkit.config import integer, number, string
from adrkit.config.composition import Family
from adrkit.config.schema import fields, vector
from adrkit.sources import sampled_source_map
from adrkit.spaces import ArraySpace


class BernsteinBasis:
    r"""Базис Бернштейна на заданном временном интервале.

    Parameters
    ----------
    degree : int
        Неотрицательная степень полиномов.
    interval : array_like, shape (2,)
        Действительные границы интервала в возрастающем порядке,
        выраженные в единицах `time_unit`.
    time_unit : str
        Единица времени.

    Notes
    -----
    Для степени :math:`n`, заданной параметром `degree`, базис имеет вид

    .. math::

        b_j(t) = \binom{n}{j}s^j(1-s)^{n-j},\qquad
        s = \frac{t-t_0}{t_1-t_0},\quad j=0,\ldots,n.
    """
    def __init__(self, degree, interval, time_unit):
        self.size = degree + 1
        self.interval = tuple(interval)
        self.time_unit = time_unit

    def values(self, times):
        """Вычислить значения всех базисных функций в заданные моменты.

        Parameters
        ----------
        times : array_like, shape (n_times,)
            Действительные моменты времени в единицах `time_unit`.

        Returns
        -------
        ndarray, shape (n_times, degree + 1)
            Безразмерные значения базиса; столбцы идут по возрастанию индекса.

        Notes
        -----
        Вне заданного интервала вычисляется полиномиальное продолжение.
        """
        t = (np.asarray(times) - self.interval[0]) / (self.interval[1] - self.interval[0])
        n = self.size - 1
        return np.column_stack([comb(n, j) * t**j * (1-t)**(n-j) for j in range(n+1)])


class UserTruth:
    r"""Заданный квадратичный профиль источника.

    Notes
    -----
    Профиль не зависит от коэффициентов восстанавливаемого источника:

    .. math::

        q(t) = 1 + t - \frac{t^2}{2}.
    """
    def value(self, times):
        """Вычислить заданный профиль в переданные моменты времени."""
        t = np.asarray(times)
        return 1 + t - t*t/2


def validate(p, *, partial):
    """Проверить параметры семейства с учётом допустимости неполной конфигурации."""
    fields(p, {"degree", "interval", "times", "time_unit", "coefficient_unit", "value_unit", "scale"}, partial=partial)
    p = dict(p)
    if "degree" in p:
        integer(p["degree"], minimum=1)
    for key in ("interval", "times"):
        if key in p:
            p[key] = vector(p[key])
    if "interval" in p and (len(p["interval"]) != 2 or p["interval"][0] >= p["interval"][1]):
        raise ValueError("interval must have two increasing endpoints")
    if "times" in p:
        if any(a >= b for a, b in zip(p["times"], p["times"][1:])):
            raise ValueError("times must be strictly increasing")
        if "interval" in p and (p["times"][0] < p["interval"][0] or p["times"][-1] > p["interval"][1]):
            raise ValueError("times outside Bernstein interval")
    for key in ("time_unit", "coefficient_unit", "value_unit"):
        if key in p:
            string(p[key])
    if "scale" in p:
        p["scale"] = number(p["scale"], positive=True)
    return p


def build(p):
    """Собрать базис Бернштейна и отображение его значений в заданные моменты."""
    basis = BernsteinBasis(p["degree"], p["interval"], p["time_unit"])
    domain = ArraySpace((basis.size,), axes=("bernstein_coefficient",),
                        coordinates=(tuple(range(basis.size)),), units=p["coefficient_unit"])
    codomain = ArraySpace((len(p["times"]),), axes=("time",), coordinates=(p["times"],), units=p["value_unit"])
    return basis, sampled_source_map(basis, p["times"], scale=p["scale"], domain=domain, codomain=codomain)


FAMILIES = {"source": {"bernstein": Family(validate, build)}}

