"""Средние нагрузки P1 и фиксированная известная предыстория."""
from dataclasses import dataclass

import numpy as np

from adrkit.contracts import MapCapabilities
from adrkit.sources import P1Basis
from adrkit.spaces import ArraySpace


def _real_array(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    result = np.array(value, dtype=float, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def interval_hat_integrals(basis, edges):
    """Интегрировать функции P1 по временным интервалам.

    Parameters
    ----------
    basis : P1Basis
        Временной кусочно-линейный базис.
    edges : array_like, shape (m+1,)
        Конечные строго возрастающие границы интервалов в единицах
        времени базиса.

    Returns
    -------
    integrals : ndarray, shape (m, basis.size)
        Интегралы функций базиса в единицах времени, без деления на
        длину интервала.

    Notes
    -----
    За интервалом базиса используется нулевое продолжение.
    """
    if not isinstance(basis, P1Basis):
        raise TypeError("basis must be P1Basis")
    edges = _real_array(edges, "interval edges")
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("interval edges must be strictly increasing")
    result = np.zeros((edges.size - 1, basis.size))
    knots = basis.knots
    for j, (start, end) in enumerate(zip(knots[:-1], knots[1:])):
        left = np.maximum(edges[:-1], start)
        right = np.minimum(edges[1:], end)
        active = right > left
        width = right[active] - left[active]
        fraction = ((left[active] - start) + (right[active] - start)) / (2*(end-start))
        result[active, j] += width*(1-fraction)
        result[active, j+1] += width*fraction
    return result


class IntervalAverageLoad:
    """Средняя временная нагрузка из P1 и известной предыстории.

    Parameters
    ----------
    basis : P1Basis
        Базис неизвестной части источника.
    state_times : array_like, shape (nt+1,)
        Строго возрастающие времена решателя от нуля, в единицах
        времени базиса.
    origin : float
        Начало физического времени решателя; сетка включает
        предысторию и заканчивается последним узлом базиса.
    q_reference : float
        Конечный положительный масштаб интенсивности.
    history_integral : callable
        Интеграл известного источника по двум физическим концам;
        результат в единицах интенсивность·время.
    history_initial : float
        Начальная интенсивность известной истории.
    source_unit : str
        Метка единицы интенсивности источника.

    Notes
    -----
    Коэффициенты P1 безразмерны. Интегралы истории вычисляются
    при создании; вне интервала P1 применяется нулевое продолжение.
    Начальный компонент нагрузки не задаёт начальную концентрацию.
    """

    def __init__(self, basis, state_times, *, origin, q_reference,
                 history_integral, history_initial, source_unit):
        if not isinstance(basis, P1Basis):
            raise TypeError("basis must be P1Basis")
        times = _real_array(state_times, "state times")
        scalars = _real_array([origin, q_reference, history_initial], "load parameters")
        origin, q_reference, history_initial = scalars
        if (times.ndim != 1 or len(times) < 2 or times[0] != 0
                or np.any(np.diff(times) <= 0) or q_reference <= 0):
            raise ValueError("increasing state times starting at zero and positive Qref required")
        edges = times + origin
        knots = basis.knots
        if (edges[0] > knots[0] or not np.isclose(edges[-1], knots[-1], rtol=0,
                atol=32*np.finfo(float).eps*max(1., abs(knots[-1])))):
            raise ValueError("state clock must include history and end at the unknown horizon")
        if not callable(history_integral):
            raise TypeError("history_integral must be an explicit definite-integral callable")
        dt = np.diff(times)
        matrix = np.zeros((len(times), basis.size))
        matrix[1:] = interval_hat_integrals(basis, edges) / dt[:, None]
        offset = np.zeros(len(times))
        offset[0] = history_initial
        for k, (left, right) in enumerate(zip(edges[:-1], edges[1:]), start=1):
            right = min(right, knots[0])
            if left < right:
                integral = _real_array(history_integral(float(left), float(right)), "history integral")
                if integral.shape != ():
                    raise ValueError("history integral must return a scalar")
                offset[k] = integral.item()/dt[k-1]
        self.domain = ArraySpace((basis.size,), axes=("coefficient",),
                                 coordinates=(knots,), units="dimensionless")
        self.codomain = ArraySpace((len(times),), axes=("time",),
                                   coordinates=(times,), units=source_unit)
        self._matrix, self._offset = matrix, offset
        self._matrix.flags.writeable = self._offset.flags.writeable = False
        self.q_reference = float(q_reference)

    @property
    def averages(self):
        """Копия матрицы средних P1 без масштаба интенсивности.

        Returns
        -------
        averages : ndarray, shape (nt+1, basis.size)
            Независимая матрица средних значений P1 без `q_reference`;
            значения безразмерны, начальная строка нулевая.
        """
        return self._matrix.copy()

    @property
    def history(self):
        """Копия значений известной истории в единицах интенсивности.

        Returns
        -------
        history : ndarray, shape (nt+1,)
            Независимая копия известной нагрузки в единицах интенсивности;
            начальный компонент равен `history_initial`.
        """
        return self._offset.copy()

    def predict(self, coefficients):
        """Вычислить среднюю интенсивность по временным шагам.

        Parameters
        ----------
        coefficients : array_like, shape (basis.size,)
            Безразмерные коэффициенты неизвестного P1-источника.

        Returns
        -------
        load : ndarray, shape (nt+1,)
            Средние интенсивности с известной предысторией в source_unit.
        """
        a = self.domain.array(coefficients)
        return self.codomain.array(self._offset + self.q_reference*(self._matrix @ a),
                                   name="load prediction", copy=True)

    def jvp(self, at, direction):
        """Применить производную средней нагрузки к направлению.

        Parameters
        ----------
        at, direction : array_like, shape (basis.size,)
            Точка и направление безразмерных коэффициентов.

        Returns
        -------
        tangent : ndarray, shape (nt+1,)
            Изменение интенсивности в source_unit; история не
            дифференцируется.
        """
        self.domain.array(at)
        return self.codomain.array(self.q_reference*(self._matrix @ self.domain.array(direction)),
                                   name="load JVP output", copy=True)

    def vjp(self, at, cotangent):
        """Перенести ковектор средней нагрузки к коэффициентам.

        Parameters
        ----------
        at : array_like, shape (basis.size,)
            Точка коэффициентов.
        cotangent : array_like, shape (nt+1,)
            Координатный ковектор нагрузки.

        Returns
        -------
        covector : ndarray, shape (basis.size,)
            Ковектор по безразмерным коэффициентам в евклидовых
            координатах.
        """
        self.domain.array(at)
        return self.domain.array(self.q_reference*(self._matrix.T @ self.codomain.array(cotangent)),
                                 name="load VJP output", copy=True)

    def capabilities(self):
        """Предоставить среднюю нагрузку и её JVP/VJP как MapCapabilities."""
        return MapCapabilities(self, self.domain, self.codomain, self, self)


@dataclass(frozen=True)
class DisclosedHistory:
    """Интегралы известного источника на заданных интервалах.

    Parameters
    ----------
    intervals : sequence of pairs
        Уникальные пары возрастающих конечных концов, не позже end.
    integrals : sequence of float
        Конечные интегралы источника в порядке intervals, в единицах
        интенсивность·время.
    initial : float
        Конечное значение источника в начале истории.
    end : float, optional
        Конечный физический момент доступной истории.

    Notes
    -----
    Хранятся числа, а не исходная функция источника.
    """
    intervals: tuple
    integrals: tuple
    initial: float
    end: float = 0.0

    def __post_init__(self):
        end = _history_scalar(self.end, "history end")
        object.__setattr__(self, "end", end)
        intervals, integrals = tuple(self.intervals), tuple(self.integrals)
        if len(intervals) != len(integrals):
            raise ValueError("One disclosed integral is required per history interval")
        owned_intervals = []
        for interval in intervals:
            pair = tuple(interval)
            if len(pair) != 2:
                raise ValueError("History intervals require two endpoints")
            left, right = (_history_scalar(value, "history endpoint") for value in pair)
            if left >= right or right > end + 1e-12:
                raise ValueError("History intervals must increase and contain no undisclosed future")
            owned_intervals.append((left, right))
        if len(set(owned_intervals)) != len(owned_intervals):
            raise ValueError("Disclosed history intervals must have unique endpoint pairs")
        object.__setattr__(self, "intervals", tuple(owned_intervals))
        object.__setattr__(self, "integrals", tuple(
            _history_scalar(value, "history integral") for value in integrals))
        object.__setattr__(self, "initial", _history_scalar(self.initial, "history initial value"))

    def integral(self, left, right):
        """Получить интеграл для сохранённой пары концов.

        Parameters
        ----------
        left, right : float
            Физические концы сохранённого интервала в единицах времени
            истории.

        Returns
        -------
        integral : float
            Сохранённый интеграл источника в единицах
            интенсивность·время.

        Notes
        -----
        Концы сравниваются с абсолютным допуском 1e-12.
        Интегралы незарегистрированных подынтервалов не вычисляются.
        """
        left, right = (_history_scalar(value, "history endpoint") for value in (left, right))
        if right > self.end + 1e-12:
            raise ValueError("Estimator requested undisclosed future source")
        for interval, value in zip(self.intervals, self.integrals):
            if np.allclose(interval, [left,right], rtol=0, atol=1e-12):
                return value
        raise ValueError("Unregistered disclosed history interval")


def _history_scalar(value, name):
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real scalar")
    return float(_real_array(value, name))


def disclose_history(source, state_times, origin, *, end=0.0):
    """Сохранить известную историю источника до заданного момента.

    Parameters
    ----------
    source : object
        Источник с методами integral(left, right) и value(time) в
        согласованных единицах.
    state_times : array_like, shape (nt+1,)
        Конечные строго возрастающие времена решателя от нуля.
    origin : float
        Конечное начало физического времени, не позже end.
    end : float, optional
        Конечный момент доступной истории.

    Returns
    -------
    history : DisclosedHistory
        Интегралы в доступных частях шагов и начальное значение
        источника; функции не сохраняются.
    """
    for method in ("integral", "value"):
        if not callable(getattr(source, method, None)):
            raise TypeError(f"History source must provide callable {method}()")
    raw_times = np.asarray(state_times)
    if raw_times.dtype.kind not in "iuf":
        raise ValueError("History state times must be finite real numbers")
    times = _real_array(state_times, "history state times")
    origin = _history_scalar(origin, "history time origin")
    end = _history_scalar(end, "history end")
    if (times.ndim != 1 or len(times) < 2 or times[0] != 0 or origin > end
            or np.any(np.diff(times) <= 0)):
        raise ValueError("History state times must increase from zero with origin no later than end")
    edges = _real_array(times+origin, "history physical clock")
    intervals = tuple((float(a), float(min(b,end))) for a,b in zip(edges[:-1],edges[1:])
                      if a < min(b,end))
    values = tuple(_history_scalar(source.integral(a,b), "history integral") for a,b in intervals)
    return DisclosedHistory(intervals, values,
        _history_scalar(source.value(origin), "history initial value"), end)


