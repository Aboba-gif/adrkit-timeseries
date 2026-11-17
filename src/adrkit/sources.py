"""Временные базисы источника и линейные отображения коэффициентов."""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from importlib.metadata import entry_points
import json
from operator import index

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


@dataclass(frozen=True, init=False)
class ShiftedBasis:
    r"""Сдвинуть временную координату и явно продолжить исходный базис.

    Parameters
    ----------
    basis : object or Mapping
        Объект с size, time_unit, values, integrals, gram и to_record либо
        запись зарегистрированного базиса. Сохраняется собственная копия.
        В записи отсутствующая time_unit наследуется от адаптера; явно
        указанная единица должна с ней совпадать.
    origin : float
        Конечное начало локального времени в time_unit: psi(s)=E phi(s-origin).
    time_unit : str
        Явная единица времени, совпадающая с единицей исходного базиса.
    extension : {'native', 'zero', 'constant'}, optional
        native делегирует исходному базису без изменения его области
        определения. zero и constant задают соответственно нулевые и
        постоянные по крайним значениям продолжения за interval.
    interval : pair of float or None, optional
        Конечные строго возрастающие границы локального времени исходного
        базиса для zero или constant. Для native интервал не задаётся.

    Notes
    -----
    Продолжение применяется к каждой функции базиса и её интегралам,
    сохраняя число и порядок амплитуд. Для constant внешние части Gram
    равны длине участка, умноженной на внешнее произведение крайних
    значений. Интервалы пересечения интегрируются исходным методом;
    временная квадратура не добавляется. Необязательные производные
    исходного базиса адаптер не объявляет.
    """

    _basis: object
    _size: int
    origin: float
    time_unit: str
    extension: str
    interval: tuple[float, float] | None

    def __init__(self, basis, origin, *, time_unit, extension="native", interval=None):
        if not isinstance(time_unit, str) or not time_unit.strip():
            raise ValueError("A shifted basis requires an explicit time unit")
        if (
            np.ma.isMaskedArray(origin)
            or not is_scalar_constant(origin)
            or not np.isfinite(origin)
        ):
            raise ValueError("The time origin must be a finite real scalar")
        if not isinstance(extension, str) or extension not in ("native", "zero", "constant"):
            raise ValueError("extension must be 'native', 'zero', or 'constant'")
        if extension == "native":
            if interval is not None:
                raise ValueError("The native extension does not accept an interval")
            limits = None
        else:
            if np.ma.isMaskedArray(interval) or np.iscomplexobj(interval):
                raise ValueError("An extension requires a finite increasing interval")
            limits = np.asarray(interval, dtype=float)
            if limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] >= limits[1]:
                raise ValueError("An extension requires a finite increasing interval")
            limits = tuple(limits.tolist())
        if isinstance(basis, Mapping):
            record = self._child_record(basis, time_unit)
            child = deepcopy(source_basis_from_record(record))
        else:
            child = deepcopy(basis)
        if getattr(child, "time_unit", None) != time_unit:
            raise ValueError("The shifted basis and its child must use the same time unit")
        try:
            size = index(child.size)
        except (TypeError, AttributeError) as error:
            raise ValueError("The child basis size must be a positive integer") from error
        if isinstance(child.size, (bool, np.bool_)) or size < 1:
            raise ValueError("The child basis size must be a positive integer")
        for method in ("values", "integrals", "gram", "to_record"):
            if not callable(getattr(child, method, None)):
                raise TypeError(f"The child basis must provide {method}")
        object.__setattr__(self, "_basis", child)
        object.__setattr__(self, "_size", size)
        object.__setattr__(self, "origin", float(origin))
        object.__setattr__(self, "time_unit", time_unit)
        object.__setattr__(self, "extension", extension)
        object.__setattr__(self, "interval", limits)

    @staticmethod
    def _child_record(basis, time_unit):
        if not isinstance(basis, Mapping):
            raise TypeError("A serialized shifted basis requires a child mapping")
        if not isinstance(time_unit, str) or not time_unit.strip():
            raise ValueError("A shifted basis requires an explicit time unit")
        record = deepcopy(dict(basis))
        if record.get("time_unit", time_unit) != time_unit:
            raise ValueError("The shifted basis and its child must use the same time unit")
        record.setdefault("time_unit", time_unit)
        return record

    @staticmethod
    def source_basis_dependencies(parameters):
        """Объявить единственный вложенный базис без вызова его конструктора."""
        return (ShiftedBasis._child_record(parameters["basis"], parameters["time_unit"]),)

    @property
    def basis(self):
        """Независимая копия исходного базиса без доступа к сохранённому объекту."""
        return deepcopy(self._basis)

    @property
    def size(self):
        """Неизменённое число функций исходного базиса."""
        return self._size

    def _local(self, values):
        if np.ma.isMaskedArray(values) or np.iscomplexobj(values):
            raise ValueError("Shifted sample times must be finite real values")
        values = np.array(values, dtype=float, copy=True)
        if not np.isfinite(values).all():
            raise ValueError("Shifted sample times must be finite real values")
        with np.errstate(over="raise", invalid="raise"):
            return values - self.origin

    def _evaluate(self, method, *arguments, shape):
        result = getattr(self._basis, method)(*arguments)
        if np.ma.isMaskedArray(result) or np.iscomplexobj(result):
            raise ValueError(f"The child basis {method} must return finite real values")
        result = np.array(result, dtype=float, copy=True)
        if result.shape != shape or not np.isfinite(result).all():
            raise ValueError(f"The child basis {method} must have finite values of shape {shape}")
        return result

    def _endpoints(self, left, right):
        if np.ma.isMaskedArray(left) or np.ma.isMaskedArray(right):
            raise ValueError("Integration endpoints must be explicit finite values")
        left, right = _integration_endpoints(left, right)
        return self._local(left), self._local(right)

    def values(self, times):
        """Вычислить psi(s)=E phi(s-origin), вернув массив формы (m,size)."""
        local = self._local(times)
        if local.ndim != 1:
            raise ValueError("Shifted sample times must be a finite vector")
        if self.extension == "native":
            return self._evaluate("values", local, shape=(len(local), self.size))
        lo, hi = self.interval
        if self.extension == "constant":
            return self._evaluate("values", np.clip(local, lo, hi), shape=(len(local), self.size))
        result = np.zeros((len(local), self.size))
        inside = (local >= lo) & (local <= hi)
        if np.any(inside):
            result[inside] = self._evaluate(
                "values", local[inside], shape=(int(inside.sum()), self.size)
            )
        return result

    def integrals(self, left, right):
        """Интегрировать продолженный базис по совместимым интервалам времени."""
        left, right = self._endpoints(left, right)
        shape = (*left.shape, self.size)
        if self.extension == "native":
            return self._evaluate("integrals", left, right, shape=shape)
        lo, hi = self.interval
        result = self._evaluate(
            "integrals", np.clip(left, lo, hi), np.clip(right, lo, hi), shape=shape
        )
        if self.extension == "constant":
            endpoints = self._evaluate("values", np.array([lo, hi]), shape=(2, self.size))
            with np.errstate(over="raise", invalid="raise"):
                left_length = np.minimum(right, lo) - np.minimum(left, lo)
                right_length = np.maximum(right, hi) - np.maximum(left, hi)
                result += left_length[..., None] * endpoints[0]
                result += right_length[..., None] * endpoints[1]
        return result

    def gram(self, left, right):
        """Интегрировать попарные произведения psi на одном интервале времени."""
        left, right = self._endpoints(left, right)
        if left.ndim or right.ndim:
            raise ValueError("Gram endpoints must be scalars")
        shape = (self.size, self.size)
        if self.extension == "native":
            return self._evaluate("gram", left.item(), right.item(), shape=shape)
        lo, hi = self.interval
        result = self._evaluate(
            "gram", float(np.clip(left, lo, hi)), float(np.clip(right, lo, hi)), shape=shape
        )
        if self.extension == "constant":
            endpoints = self._evaluate("values", np.array([lo, hi]), shape=(2, self.size))
            with np.errstate(over="raise", invalid="raise"):
                left_length = np.minimum(right, lo) - np.minimum(left, lo)
                right_length = np.maximum(right, hi) - np.maximum(left, hi)
                for length, endpoint in zip((left_length, right_length), endpoints):
                    if length > 0:
                        result += length * np.outer(endpoint, endpoint)
        return result

    def to_record(self):
        """Сохранить вложенный базис, начало координат и правило продолжения."""
        record = self._basis.to_record()
        if not isinstance(record, Mapping):
            raise TypeError("The child basis to_record must return a mapping")
        if record.get("time_unit", self.time_unit) != self.time_unit:
            raise ValueError("The child basis record must preserve its time unit")
        return {
            "type": "shifted",
            "basis": deepcopy(dict(record)),
            "origin": self.origin,
            "time_unit": self.time_unit,
            "extension": self.extension,
            "interval": None if self.interval is None else list(self.interval),
        }


_SOURCE_BASIS_FACTORIES = {"gaussian": GaussianBasis, "p1": P1Basis, "shifted": ShiftedBasis}
_SOURCE_BASIS_ENTRY_POINT_GROUP = "adrkit_timeseries.source_bases"


def _source_basis_name(name):
    if not isinstance(name, str) or not name or name != name.strip():
        raise ValueError("Source basis name must be a nonempty string without surrounding spaces")


def _source_basis_factory(name):
    """Разрешить одну явно запрошенную фабрику и её установленного поставщика."""
    _source_basis_name(name)
    factory = _SOURCE_BASIS_FACTORIES.get(name)
    installed = tuple(entry_points(group=_SOURCE_BASIS_ENTRY_POINT_GROUP, name=name))
    if len(installed) + int(factory is not None) > 1:
        raise ValueError(f"Multiple source basis factories are registered: {name!r}")
    provider = installed[0] if installed else None
    if provider is not None:
        factory = provider.load()
    if factory is None:
        raise ValueError(f"Unknown source basis: {name!r}")
    if not callable(factory):
        raise TypeError(f"Source basis factory must be callable: {name!r}")
    return factory, provider


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
    factory, _ = _source_basis_factory(name)
    return factory(**parameters)


def _source_basis_signature(record):
    if not isinstance(record, Mapping):
        raise TypeError("Source basis dependencies must be mappings")
    try:
        return json.dumps(dict(record), sort_keys=True, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError(
            "Source provenance requires finite acyclic serialized JSON records"
        ) from error


def source_basis_providers(record, *additional_records):
    """Собрать установленных поставщиков явно объявленного дерева базисов.

    Parameters
    ----------
    record : Mapping
        Конечная ациклическая JSON-сериализуемая запись базиса. В отличие
        от source_basis_from_record, массивы NumPy и объекты базисов здесь
        не принимаются: происхождение относится к сохраняемой конфигурации.
        Пользовательский метод получает собственную JSON-декодированную запись.
    *additional_records : Mapping
        Дополнительные явно заданные корни с тем же контрактом. Например,
        вызывающий код передаёт и запрошенный базис, и итоговую запись объекта:
        фабрика может вернуть объект другого зарегистрированного семейства.

    Returns
    -------
    list of dict
        Уникальные установленные поставщики с полями name, factory,
        distribution, version. Порядок — сначала родитель, затем дети
        в порядке объявления; корни обходятся в порядке аргументов.
        Встроенные и локальные фабрики не получают
        вымышленной версии пакета; список для таких листьев пуст.

    Notes
    -----
    Фабрика составного базиса явно предоставляет необязательный метод
    source_basis_dependencies(parameters), где parameters — поля записи
    без type. Он возвращает конечную последовательность записей детей,
    не вызывает их конструкторы и не изменяет исходные настройки. Отсутствие
    метода означает лист. Ключи type внутри произвольных параметров сами
    по себе зависимостями не становятся.

    Разрешаются те же конфликты поставщиков, что при создании базиса.
    Загружаются только явно запрошенные фабрики, но их конструкторы не
    вызываются. Цикл определяется повторением канонической JSON-записи
    на активном пути; повторный тип с другими настройками и общий ребёнок
    разных ветвей допустимы. Объявления зависимостей должны задавать конечный
    детерминированный граф. Произвольное незавершающееся выполнение метода
    или бесконечная последовательность различных записей не ограничиваются
    искусственным порогом. Локальная регистрация по-прежнему не заменяет
    установленную точку входа для независимых процессов spawn.
    """
    providers, recorded, complete, active = [], set(), set(), set()

    def visit(current):
        signature = _source_basis_signature(current)
        if signature in active:
            raise ValueError("Source basis dependencies contain a cycle")
        if signature in complete:
            return
        parameters = json.loads(signature)
        factory, provider = _source_basis_factory(parameters.pop("type"))
        if provider is not None:
            description = {
                "name": provider.name,
                "factory": provider.value,
                "distribution": provider.dist.metadata["Name"],
                "version": provider.dist.version,
            }
            if any(not isinstance(value, str) or not value for value in description.values()):
                raise ValueError("Source provider metadata must contain nonempty strings")
            key = tuple(description.values())
            if key not in recorded:
                providers.append(description)
                recorded.add(key)
        active.add(signature)
        missing = object()
        hook = getattr(factory, "source_basis_dependencies", missing)
        if hook is not missing:
            if not callable(hook):
                raise TypeError("source_basis_dependencies must be callable")
            children = hook(parameters)
            if isinstance(children, (str, bytes, bytearray)) or not isinstance(children, Sequence):
                raise TypeError("Source dependencies must be a finite sequence of mappings")
            for child in children:
                visit(child)
        active.remove(signature)
        complete.add(signature)

    for root in (record, *additional_records):
        visit(root)
    return providers


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
