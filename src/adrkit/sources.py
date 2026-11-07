"""Временные базисы источника и линейные отображения коэффициентов."""

from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import entry_points

import numpy as np
from scipy.special import erf

from adrkit.coefficients import is_scalar_constant
from adrkit.spaces import matrix_map


def _integration_endpoints(left, right):
    if np.iscomplexobj(left) or np.iscomplexobj(right):
        raise ValueError("Integration endpoints must be real")
    left, right = np.broadcast_arrays(
        np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    )
    if not np.isfinite(left).all() or not np.isfinite(right).all() or np.any(left > right):
        raise ValueError("Integration endpoints must be finite and ordered")
    return left, right


@dataclass(frozen=True, init=False)
class GaussianBasis:
    r"""Гауссовы функции с общей шириной и необязательной постоянной частью.

    Parameters
    ----------
    centers : array_like, shape (n,)
        Конечные центры базисных функций в time_unit.
    width : float
        Конечная положительная ширина в time_unit.
    time_unit : str
        Непустое обозначение единицы времени.
    include_constant : bool, optional
        Добавить первым столбцом функцию, равную единице.

    Notes
    -----
    Для центра mu и ширины w базисная функция равна
    exp(-(t-mu)^2/(2w^2)). Функции определены на всей временной оси.
    Интегралы и грамова матрица вычисляются аналитически.
    """

    _centers: tuple[float, ...]
    width: float
    time_unit: str
    include_constant: bool

    def __init__(self, centers, width, *, time_unit, include_constant=True):
        if np.iscomplexobj(centers):
            raise ValueError("Gaussian centers must be real")
        centers = np.asarray(centers, dtype=float)
        if centers.ndim != 1 or not np.isfinite(centers).all():
            raise ValueError("Gaussian centers must be a finite vector")
        if not is_scalar_constant(width) or not np.isfinite(width) or width <= 0:
            raise ValueError("Gaussian width must be finite and positive")
        if not isinstance(time_unit, str) or not time_unit.strip():
            raise ValueError("Gaussian basis requires an explicit time unit")
        if not isinstance(include_constant, bool):
            raise ValueError("include_constant must be a bool")
        object.__setattr__(self, "_centers", tuple(centers.tolist()))
        object.__setattr__(self, "width", float(width))
        object.__setattr__(self, "time_unit", time_unit)
        object.__setattr__(self, "include_constant", include_constant)

    @property
    def centers(self):
        """Независимая копия центров в единицах времени базиса."""
        return np.asarray(self._centers)

    @property
    def size(self):
        """Число функций, включая постоянную часть при её наличии."""
        return len(self._centers) + int(self.include_constant)

    def values(self, times):
        """Вычислить значения функций базиса.

        Parameters
        ----------
        times : array_like, shape (m,)
            Конечные вещественные моменты в time_unit.

        Returns
        -------
        values : ndarray, shape (m, size)
            Безразмерные значения в порядке центров.
        """
        if np.iscomplexobj(times):
            raise ValueError("Sample times must be real")
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or not np.isfinite(times).all():
            raise ValueError("Sample times must be a finite vector")
        values = np.exp(-0.5 * ((times[:, None] - self.centers) / self.width) ** 2)
        if self.include_constant:
            values = np.column_stack((np.ones(len(times)), values))
        return values

    def integrals(self, left, right):
        """Интегрировать функции базиса по совместимым интервалам.

        Parameters
        ----------
        left, right : array_like
            Конечные упорядоченные концы интервалов в time_unit.

        Returns
        -------
        integrals : ndarray, shape (..., size)
            Интегралы в time_unit; последняя ось следует функциям базиса.
        """
        left, right = _integration_endpoints(left, right)
        scale = np.sqrt(2) * self.width
        values = (
            self.width
            * np.sqrt(np.pi / 2)
            * (
                erf((right[..., None] - self.centers) / scale)
                - erf((left[..., None] - self.centers) / scale)
            )
        )
        if self.include_constant:
            values = np.concatenate(((right - left)[..., None], values), axis=-1)
        return values

    def gram(self, left, right):
        r"""Интегрировать попарные произведения функций на одном интервале.

        Parameters
        ----------
        left, right : float
            Конечные упорядоченные концы интервала в time_unit.

        Returns
        -------
        gram : ndarray, shape (size, size)
            Симметричная матрица интегралов в time_unit.

        Notes
        -----
        Произведение гауссиан с центрами mu_i и mu_j имеет центр
        (mu_i+mu_j)/2, ширину w/sqrt(2) и множитель
        exp(-(mu_i-mu_j)^2/(4w^2)).
        """
        left, right = _integration_endpoints(left, right)
        if left.ndim or right.ndim:
            raise ValueError("Gram endpoints must be scalars")
        centers = self.centers
        difference = centers[:, None] - centers[None, :]
        midpoints = centers[:, None] / 2 + centers[None, :] / 2
        products = GaussianBasis(
            midpoints.ravel(),
            self.width / np.sqrt(2),
            time_unit=self.time_unit,
            include_constant=False,
        )
        result = np.empty((self.size, self.size))
        offset = int(self.include_constant)
        result[offset:, offset:] = np.exp(
            -(difference**2) / (4 * self.width**2)
        ) * products.integrals(left, right).reshape(midpoints.shape)
        if self.include_constant:
            result[:, 0] = result[0, :] = self.integrals(left, right)
        return result

    def to_record(self):
        """Вернуть численные настройки базиса и единицу времени."""
        return {
            "type": "gaussian",
            "centers": list(self._centers),
            "width": self.width,
            "time_unit": self.time_unit,
            "include_constant": self.include_constant,
        }


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
    if (
        knots.ndim != 1
        or len(knots) < 2
        or not np.isfinite(knots).all()
        or np.any(np.diff(knots) <= 0)
    ):
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
        local = length / 6 * np.array([[2.0, 1.0], [1.0, 2.0]])
        local += time_scale**2 / length * np.array([[1.0, -1.0], [-1.0, 1.0]])
        result[j : j + 2, j : j + 2] += local
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
        if (
            times.ndim != 1
            or not np.isfinite(times).all()
            or np.any(times < self._knots[0])
            or np.any(times > self._knots[-1])
        ):
            raise ValueError("Sample times must lie in the closed P1 interval")
        knots = self.knots
        indices = np.minimum(np.searchsorted(knots, times, side="right") - 1, self.size - 2)
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
        fraction = (times - knots[indices]) / (knots[indices + 1] - knots[indices])
        values = np.zeros((len(times), self.size))
        rows = np.arange(len(times))
        values[rows, indices] = 1 - fraction
        values[rows, indices + 1] = fraction
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
        slope = 1 / (knots[indices + 1] - knots[indices])
        values = np.zeros((len(times), self.size))
        rows = np.arange(len(times))
        values[rows, indices] = -slope
        values[rows, indices + 1] = slope
        return values

    def integrals(self, left, right):
        """Интегрировать P1-функции с нулевым продолжением за узлами.

        Parameters
        ----------
        left, right : array_like
            Конечные упорядоченные концы интервалов в time_unit.

        Returns
        -------
        integrals : ndarray, shape (..., size)
            Интегралы в time_unit; последняя ось следует узлам.
        """
        left, right = _integration_endpoints(left, right)
        result = np.zeros((*left.shape, self.size))
        knots = self.knots
        for j, (start, end) in enumerate(zip(knots[:-1], knots[1:])):
            a, b = np.clip(left, start, end), np.clip(right, start, end)
            width = b - a
            fraction = ((a - start) + (b - start)) / (2 * (end - start))
            result[..., j] += width * (1 - fraction)
            result[..., j + 1] += width * fraction
        return result

    def gram(self, left, right):
        """Интегрировать попарные произведения P1-функций.

        Parameters
        ----------
        left, right : float
            Конечные упорядоченные концы интервала в time_unit.

        Returns
        -------
        gram : ndarray, shape (size, size)
            Грамова матрица в time_unit с нулевым продолжением за узлами.
        """
        left, right = _integration_endpoints(left, right)
        if left.ndim or right.ndim:
            raise ValueError("Gram endpoints must be scalars")
        result = np.zeros((self.size, self.size))
        for j, (start, end) in enumerate(zip(self._knots[:-1], self._knots[1:])):
            a, b = max(float(left), start), min(float(right), end)
            if a < b:
                x, y = (a - start) / (end - start), (b - start) / (end - start)
                first = y - x
                second = (y**2 - x**2) / 2
                third = (y**3 - x**3) / 3
                result[j, j] += (end - start) * (first - 2 * second + third)
                result[j + 1, j + 1] += (end - start) * third
                mixed = (end - start) * (second - third)
                result[j, j + 1] += mixed
                result[j + 1, j] += mixed
        return result

    def to_record(self):
        """Вернуть временные узлы, единицу времени и условия на концах."""
        return {
            "type": "p1",
            "knots": list(self._knots),
            "time_unit": self.time_unit,
            "endpoint": self.endpoint,
        }


_SOURCE_BASIS_FACTORIES = {"gaussian": GaussianBasis, "p1": P1Basis}
_SOURCE_BASIS_ENTRY_POINT_GROUP = "adrkit_timeseries.source_bases"


def _source_basis_name(name):
    if not isinstance(name, str) or not name or name != name.strip():
        raise ValueError("Source basis name must be a nonempty string without surrounding spaces")


def register_source_basis(name, factory):
    """Зарегистрировать фабрику временного базиса в текущем процессе.

    Parameters
    ----------
    name : str
        Непустое имя, совпадающее с полем type сохранённого базиса.
    factory : callable
        Конструктор или функция, принимающая остальные поля записи
        как именованные аргументы и возвращающая объект базиса.

    Returns
    -------
    None
        Фабрика доступна последующим вызовам source_basis_from_record.

    Raises
    ------
    ValueError
        Имя пусто, содержит пробелы по краям либо занято другой
        локальной или установленной фабрикой.
    TypeError
        factory не является вызываемым объектом.

    Notes
    -----
    Повторная регистрация той же фабрики допустима. Другая фабрика
    под занятым именем отклоняется. Для независимых процессов фабрику
    публикуют точкой входа группы adrkit_timeseries.source_bases
    установленного Python-пакета; локальная регистрация не наследуется
    процессами, запущенными методом spawn.
    """
    _source_basis_name(name)
    if not callable(factory):
        raise TypeError("Source basis factory must be callable")
    installed = entry_points(group=_SOURCE_BASIS_ENTRY_POINT_GROUP, name=name)
    if installed:
        raise ValueError(f"Source basis name is already registered: {name!r}")
    current = _SOURCE_BASIS_FACTORIES.setdefault(name, factory)
    if current is not factory:
        raise ValueError(f"Source basis name is already registered: {name!r}")


def source_basis_from_record(record):
    """Создать временной базис по зарегистрированному имени и параметрам.

    Parameters
    ----------
    record : Mapping
        Поле type выбирает фабрику; остальные поля передаются ей
        без изменения как именованные аргументы.

    Returns
    -------
    object
        Объект выбранного базиса. Для расчётов источника он предоставляет
        size, time_unit и values(times); интегральная нагрузка и штраф
        дополнительно используют integrals(left, right) и gram(left, right).

    Raises
    ------
    KeyError
        В записи отсутствует поле type.
    ValueError
        Имя некорректно, неизвестно либо имеет нескольких поставщиков.
    TypeError
        Запись не является Mapping или фабрика не является вызываемой.

    Notes
    -----
    Встроенные и локально зарегистрированные фабрики дополняются точками
    входа группы adrkit_timeseries.source_bases. Загружается только
    запрошенная фабрика. Несколько поставщиков одного имени отклоняются;
    ошибки загрузки и конструктора передаются вызывающему коду.
    """
    if not isinstance(record, Mapping):
        raise TypeError("Source basis record must be a mapping")
    parameters = dict(record)
    name = parameters.pop("type")
    _source_basis_name(name)
    factory = _SOURCE_BASIS_FACTORIES.get(name)
    installed = tuple(entry_points(group=_SOURCE_BASIS_ENTRY_POINT_GROUP, name=name))
    if len(installed) + int(factory is not None) > 1:
        raise ValueError(f"Multiple source basis factories are registered: {name!r}")
    if installed:
        factory = installed[0].load()
    if factory is None:
        raise ValueError(f"Unknown source basis: {name!r}")
    if not callable(factory):
        raise TypeError(f"Source basis factory must be callable: {name!r}")
    return factory(**parameters)


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
    return matrix_map(
        float(scale) * np.asarray(matrix, dtype=domain.dtype), domain=domain, codomain=codomain
    )


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
    return matrix_source_map(basis.values(times), scale=scale, domain=domain, codomain=codomain)
