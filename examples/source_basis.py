"""Базис Бернштейна для временного профиля источника."""

from math import comb
from operator import index

import numpy as np
from scipy.special import beta, betainc

from adrkit.sources import register_source_basis, sampled_source_map, source_basis_from_record
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
    Значения и интегралы вычисляются на замкнутом interval.
    Функции неотрицательны на этом интервале и имеют единичную сумму.
    """

    def __init__(self, degree, interval, time_unit):
        if isinstance(degree, (bool, np.bool_)) or index(degree) < 0:
            raise ValueError("degree must be a nonnegative integer")
        endpoints = np.asarray(interval, dtype=float)
        if (
            endpoints.shape != (2,)
            or not np.isfinite(endpoints).all()
            or endpoints[0] >= endpoints[1]
        ):
            raise ValueError("interval must contain two finite increasing endpoints")
        if not isinstance(time_unit, str) or not time_unit.strip():
            raise ValueError("time_unit must be a nonempty string")
        self.degree = index(degree)
        self.size = self.degree + 1
        self.interval = tuple(endpoints.tolist())
        self.time_unit = time_unit

    def _normalized(self, times):
        times = np.asarray(times, dtype=float)
        left, right = self.interval
        if not np.isfinite(times).all() or np.any(times < left) or np.any(times > right):
            raise ValueError("times must lie in the closed basis interval")
        return (times - left) / (right - left)

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
        t = self._normalized(times)
        if t.ndim != 1:
            raise ValueError("times must be a vector")
        n = self.degree
        return np.column_stack([comb(n, j) * t**j * (1 - t) ** (n - j) for j in range(n + 1)])

    def integrals(self, left, right):
        r"""Интегрировать базис по совместимым временным интервалам.

        Parameters
        ----------
        left, right : array_like
            Упорядоченные концы интервалов в time_unit внутри interval.

        Returns
        -------
        integrals : ndarray, shape (..., size)
            Интегралы функций базиса в единицах time_unit.

        Notes
        -----
        Для :math:`L=b-a` и нормированных концов :math:`u,v`:

        .. math::

            \int_l^r B_j^n(t)\,dt = \frac{L}{n+1}
            \left[I_v(j+1,n-j+1)-I_u(j+1,n-j+1)\right].

        Здесь :math:`[a,b]` — interval, :math:`u=(l-a)/L`,
        :math:`v=(r-a)/L`, :math:`n` — degree,
        :math:`I` — регуляризованная неполная бета-функция.
        """
        left, right = np.broadcast_arrays(self._normalized(left), self._normalized(right))
        if np.any(left > right):
            raise ValueError("integration endpoints must be ordered")
        j = np.arange(self.size)
        length = self.interval[1] - self.interval[0]
        return (
            length
            / self.size
            * (
                betainc(j + 1, self.degree - j + 1, right[..., None])
                - betainc(j + 1, self.degree - j + 1, left[..., None])
            )
        )

    def gram(self, left, right):
        r"""Интегрировать попарные произведения базисных функций.

        Parameters
        ----------
        left, right : float
            Упорядоченные концы интервала в time_unit внутри interval.

        Returns
        -------
        gram : ndarray, shape (size, size)
            Матрица Грама в единицах time_unit.

        Notes
        -----
        Для :math:`L=b-a`, нормированных концов :math:`u,v` и
        :math:`p=i+j+1`, :math:`q=2n-i-j+1`:

        .. math::

            G_{ij}=L\binom{n}{i}\binom{n}{j} B(p,q)
            \left[I_v(p,q)-I_u(p,q)\right].

        Здесь :math:`[a,b]` — interval, :math:`u=(l-a)/L`,
        :math:`v=(r-a)/L`, :math:`n` — degree, :math:`B` — бета-функция,
        :math:`I` — регуляризованная неполная бета-функция.
        """
        left, right = self._normalized(left), self._normalized(right)
        if left.ndim or right.ndim or left > right:
            raise ValueError("Gram endpoints must be ordered scalars")
        n = self.degree
        i, j = np.arange(self.size)[:, None], np.arange(self.size)[None, :]
        p, q = i + j + 1, 2 * n - i - j + 1
        factors = np.array([comb(n, k) for k in range(self.size)], dtype=float)
        length = self.interval[1] - self.interval[0]
        return (
            length
            * factors[:, None]
            * factors[None, :]
            * beta(p, q)
            * (betainc(p, q, right) - betainc(p, q, left))
        )

    def to_record(self):
        """Вернуть настройки зарегистрированного базиса.

        Returns
        -------
        record : dict
            Имя типа и аргументы конструктора для source_basis_from_record.
        """
        return {
            "type": "bernstein",
            "degree": self.degree,
            "interval": list(self.interval),
            "time_unit": self.time_unit,
        }


def main():
    """Показать линейное отображение источника в базисе Бернштейна."""
    times = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    register_source_basis("bernstein", BernsteinBasis)
    basis = source_basis_from_record(
        {"type": "bernstein", "degree": 2, "interval": [0.0, 2.0], "time_unit": "s"}
    )
    coefficients = ArraySpace(
        (3,), axes=("bernstein_coefficient",), coordinates=((0, 1, 2),), units="1"
    )
    values = ArraySpace((5,), axes=("time",), coordinates=(times,), units="1")
    mapping = sampled_source_map(basis, times, scale=1.0, domain=coefficients, codomain=values)
    point = np.array([0.4, 0.6, 0.8])
    t = times / 2
    expected = point[0] * (1 - t) ** 2 + 2 * point[1] * t * (1 - t) + point[2] * t**2
    np.testing.assert_allclose(mapping.predict.predict(point), expected, rtol=1e-14)
    np.testing.assert_allclose(basis.integrals(0.0, 2.0), np.full(basis.size, 2 / 3))
    restored = source_basis_from_record(basis.to_record())
    np.testing.assert_allclose(restored.gram(0.0, 2.0), basis.gram(0.0, 2.0))
    print(mapping.predict.predict(point))


if __name__ == "__main__":
    main()
