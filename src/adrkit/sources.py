"""Базис источника P1 и линейные отображения коэффициентов."""
from dataclasses import dataclass

import numpy as np

from adrkit.coefficients import is_scalar_constant
from adrkit.spaces import matrix_map


def hat_matrix(times, knots):
    """Вычислить значения узловых функций кусочно-линейного базиса.

    Parameters
    ----------
    times : array_like, shape (m,)
        Моменты вычисления в единицах knots.
    knots : array_like, shape (n,)
        Узлы в возрастающем порядке.

    Returns
    -------
    values : ndarray, shape (m, n)
        Строки соответствуют times, столбцы — knots.

    Notes
    -----
    За крайними узлами продолжаются постоянные крайние значения.
    Это поведение отличается от ограниченного интервала P1Basis.
    """
    eye = np.eye(len(knots))
    return np.column_stack([np.interp(times, knots, column) for column in eye])


def _p1_knots(knots):
    """Проверить и преобразовать конечные возрастающие узлы P1."""
    if np.iscomplexobj(knots):
        raise ValueError("P1 knots must be real")
    knots = np.asarray(knots, dtype=float)
    if (knots.ndim != 1 or len(knots) < 2 or not np.isfinite(knots).all()
            or np.any(np.diff(knots) <= 0)):
        raise ValueError("P1 requires at least two finite increasing knots")
    return knots


def h1_gram(knots, time_scale):
    r"""Собрать грамову матрицу нормы кусочно-линейного профиля.

    Parameters
    ----------
    knots : array_like, shape (n,)
        Не менее двух конечных строго возрастающих временных узлов.
    time_scale : float
        Конечный вещественный скалярный масштаб производной
        в единицах времени knots.

    Returns
    -------
    gram : ndarray, shape (n, n)
        Матрица со свободными крайними коэффициентами.

    Notes
    -----
    Для матриц массы :math:`M` и жёсткости :math:`K`:

    .. math::

        G = M + \tau^2 K,

    Здесь :math:`G` — возвращаемая матрица, :math:`\tau` —
    `time_scale`, :math:`M,K` — матрицы интегралов произведений
    функций P1 и их временных производных. При нулевом `time_scale`
    остаётся L²-норма профиля.
    Знак масштаба не влияет на матрицу, поскольку используется его квадрат.
    """
    knots = _p1_knots(knots)
    if not is_scalar_constant(time_scale) or not np.isfinite(time_scale):
        raise ValueError("time_scale must be a finite real scalar")
    result = np.zeros((len(knots), len(knots)))
    for j, length in enumerate(np.diff(knots)):
        local = length / 6 * np.array([[2., 1.], [1., 2.]])
        local += time_scale**2 / length * np.array([[1., -1.], [-1., 1.]])
        result[j:j + 2, j:j + 2] += local
    return result


@dataclass(frozen=True, init=False)
class P1Basis:
    """Непрерывный кусочно-линейный временной базис.

    Parameters
    ----------
    knots : array_like, shape (n,)
        Не менее двух конечных строго возрастающих узлов.
    time_unit : str
        Непустое обозначение единицы времени.
    endpoint : {'free'}, optional
        Крайние коэффициенты свободны.

    Notes
    -----
    Значения определены на замкнутом интервале узлов.
    Во внутреннем узле используется правая производная,
    в первом узле — правая, в последнем — левая.
    """
    _knots: tuple
    time_unit: str
    endpoint: str

    def __init__(self, knots, *, time_unit, endpoint="free"):
        knots = _p1_knots(knots)
        if not isinstance(time_unit, str) or not time_unit.strip():
            raise ValueError("P1 requires an explicit time unit")
        if endpoint != "free":
            raise ValueError("Only free P1 endpoints are implemented")
        object.__setattr__(self, "_knots", tuple(knots.tolist()))
        object.__setattr__(self, "time_unit", time_unit)
        object.__setattr__(self, "endpoint", endpoint)

    @property
    def knots(self):
        """Независимая копия временных узлов базиса."""
        return np.array(self._knots)

    @property
    def size(self):
        """Число функций базиса, равное числу узлов."""
        return len(self._knots)

    def _intervals(self, times):
        if np.iscomplexobj(times):
            raise ValueError("Sample times must be real")
        times = np.asarray(times, dtype=float)
        if (times.ndim != 1 or not np.isfinite(times).all()
                or np.any(times < self._knots[0]) or np.any(times > self._knots[-1])):
            raise ValueError("Sample times must lie in the closed P1 interval")
        knots = self.knots
        indices = np.minimum(np.searchsorted(knots, times, side="right")-1,
                             self.size-2)
        return times, knots, indices

    def values(self, times):
        """Вычислить функции базиса в заданные моменты.

        Parameters
        ----------
        times : array_like, shape (m,)
            Конечные моменты внутри замкнутого интервала узлов в
            time_unit.

        Returns
        -------
        values : ndarray, shape (m, size)
            Безразмерные значения; столбцы следуют узлам базиса.
        """
        times, knots, indices = self._intervals(times)
        fraction = (times-knots[indices])/(knots[indices+1]-knots[indices])
        values = np.zeros((len(times), self.size))
        rows = np.arange(len(times))
        values[rows, indices] = 1-fraction
        values[rows, indices+1] = fraction
        return values

    def derivatives(self, times):
        """Вычислить производные функций базиса по времени.

        Parameters
        ----------
        times : array_like, shape (m,)
            Конечные моменты внутри замкнутого интервала узлов.

        Returns
        -------
        derivatives : ndarray, shape (m, size)
            Производные в обратных единицах time_unit; столбцы следуют
            узлам.

        Notes
        -----
        Во внутреннем узле используется правая производная,
        в последнем узле — левая.
        """
        times, knots, indices = self._intervals(times)
        slope = 1/(knots[indices+1]-knots[indices])
        values = np.zeros((len(times), self.size))
        rows = np.arange(len(times))
        values[rows, indices] = -slope
        values[rows, indices+1] = slope
        return values


def matrix_source_map(matrix, *, scale, domain, codomain):
    r"""Создать линейное отображение коэффициентов в источник.

    Parameters
    ----------
    matrix : array_like, shape (codomain.size, domain.size)
        Вещественная матрица значений; сохраняется собственная копия.
    scale : float
        Конечный положительный множитель источника.
    domain, codomain : ArraySpace
        Векторные пространства коэффициентов и источника с одинаковым
        dtype.

    Returns
    -------
    mapping : MapCapabilities
        Прогноз источника и его координатные производные.

    Notes
    -----
    Для матрицы :math:`C`, масштаба :math:`s` и коэффициентов :math:`a`:

    .. math::

        q = s C a.

    Здесь :math:`C` — `matrix`, :math:`s` — `scale`,
    :math:`a` — входные коэффициенты, :math:`q` — значения источника.
    Согласованность численных единиц матрицы и масштаба задаёт вызывающий код.
    """
    if isinstance(scale, (bool, np.bool_)) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Source scale must be finite and positive")
    if len(domain.shape) != 1 or len(codomain.shape) != 1:
        raise ValueError("Source maps require coefficient and control vectors")
    if np.iscomplexobj(matrix):
        raise ValueError("Source matrix must be real")
    return matrix_map(float(scale)*np.asarray(matrix, dtype=domain.dtype),
                      domain=domain, codomain=codomain)


def sampled_source_map(basis, times, *, scale, domain, codomain):
    """Создать отображение коэффициентов в точечные значения источника.

    Parameters
    ----------
    basis : object
        Базис с методом values(times), возвращающим матрицу значений.
    times : array_like, shape (m,)
        Конечные моменты вычисления в единицах времени базиса.
    scale : float
        Конечный положительный масштаб источника.
    domain, codomain : ArraySpace
        Векторные пространства коэффициентов и значений; координаты
        codomain совпадают с times.

    Returns
    -------
    mapping : MapCapabilities
        Линейное отображение со значениями в единицах codomain.

    See Also
    --------
    adrkit.loads.IntervalAverageLoad : Средняя нагрузка по временным шагам.
    """
    times = np.array(times, dtype=float, copy=True)
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError("Sampling times must be a finite vector")
    if codomain.coordinates != (tuple(times.tolist()),):
        raise ValueError("Control coordinates must match the supplied sampling times")
    return matrix_source_map(basis.values(times), scale=scale,
                             domain=domain, codomain=codomain)
