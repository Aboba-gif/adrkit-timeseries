"""Базис Бернштейна для временного профиля источника."""
from math import comb

import numpy as np

from adrkit.sources import sampled_source_map
from adrkit.spaces import ArraySpace


class BernsteinBasis:
    """Полиномиальный базис Бернштейна на временном интервале.

    Parameters
    ----------
    degree : int
        Неотрицательная степень полиномов.
    interval : sequence of float, length 2
        Возрастающие концы временного интервала.
    time_unit : str
        Обозначение единицы времени.

    Notes
    -----
    Базис вычисляется непосредственно как полином;
    нулевое продолжение вне интервала не задаётся.
    """
    def __init__(self, degree, interval, time_unit):
        self.size = degree + 1
        self.interval = tuple(interval)
        self.time_unit = time_unit

    def values(self, times):
        r"""Вычислить полиномы Бернштейна в заданные моменты.

        Parameters
        ----------
        times : array_like, shape (m,)
            Моменты в единицах time_unit.

        Returns
        -------
        values : ndarray, shape (m, size)
            Безразмерные значения; столбцы следуют индексу полинома.

        Notes
        -----
        Для нормированного времени :math:`z=(t-a)/(b-a)`:

        .. math::

            B_j^n(z)=\binom{n}{j}z^j(1-z)^{n-j},\quad j=0,\ldots,n.

        Здесь :math:`[a,b]` — interval, :math:`n` — degree.
        """
        t = (np.asarray(times) - self.interval[0]) / (self.interval[1] - self.interval[0])
        n = self.size - 1
        return np.column_stack([comb(n, j) * t**j * (1-t)**(n-j) for j in range(n+1)])


def main():
    """Показать линейное отображение источника в базисе Бернштейна."""
    times = np.array([0., .5, 1., 1.5, 2.])
    basis = BernsteinBasis(2, (0., 2.), "s")
    coefficients = ArraySpace((3,), axes=("bernstein_coefficient",), coordinates=((0, 1, 2),), units="1")
    values = ArraySpace((5,), axes=("time",), coordinates=(times,), units="1")
    mapping = sampled_source_map(basis, times, scale=1., domain=coefficients, codomain=values)
    point = np.array([.4, .6, .8])
    t = times/2
    expected = point[0]*(1-t)**2 + 2*point[1]*t*(1-t) + point[2]*t**2
    np.testing.assert_allclose(mapping.predict.predict(point), expected, rtol=1e-14)
    print(mapping.predict.predict(point))


if __name__ == "__main__":
    main()
