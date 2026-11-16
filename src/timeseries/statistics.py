"""Медианное абсолютное отклонение и его локальная чувствительность.

Пропуски не удаляются неявно: вызывающая сторона передаёт конечный вектор
и сохраняет его соответствие исходным наблюдениям. Ранги и ветви вычисляются
над точными двоично-рациональными значениями входов float64. Значения и
производные возвращаются с округлением в float64.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np


def _finite_vector(values, name, *, size=None):
    if np.ma.isMaskedArray(values):
        raise ValueError(f"{name} must explicitly select finite values, not a masked array")
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "iuf" or not array.size:
        raise ValueError(f"{name} must be a nonempty real vector")
    array = np.array(array, dtype=float, copy=True)
    if not np.isfinite(array).all() or (size is not None and array.size != size):
        raise ValueError(f"{name} must contain finite values with the required size")
    return array


def _positive_normalization(value):
    if np.ma.isMaskedArray(value):
        raise ValueError("normalization must be an explicit unmasked real scalar")
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind not in "iuf":
        raise ValueError("normalization must be a positive finite real scalar")
    value = float(raw)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("normalization must be a positive finite real scalar")
    return value


def _exact(values):
    return tuple(Fraction(float(value)) for value in values)


def _represented(value, name):
    try:
        result = float(value)
    except OverflowError as error:
        raise FloatingPointError(f"{name} is not representable in float64") from error
    if not np.isfinite(result) or (result == 0 and value != 0):
        raise FloatingPointError(f"{name} is not representable in float64")
    return result


def _immutable(values):
    array = np.asarray(values, dtype=float)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _central_ranks(size):
    return (size - 1) // 2, size // 2


def _median(values):
    ordered = sorted(values)
    low, high = _central_ranks(len(ordered))
    return (ordered[low] + ordered[high]) / 2


def _mad_parts(values):
    center = _median(values)
    deviations = tuple(abs(value - center) for value in values)
    return center, deviations, _median(deviations)


def _mad_value(values, normalization):
    return _represented(normalization * _mad_parts(values)[2], "MAD")


def _difference_vectors(left, right):
    left = _finite_vector(left, "left")
    right = _finite_vector(right, "right", size=len(left))
    return left, right


def _exact_differences(left, right):
    return tuple(rvalue - lvalue for lvalue, rvalue in zip(_exact(left), _exact(right)))


def _median_directional(values, directions):
    """Дифференцировать нужные ранги только внутри их исходных блоков равенств."""
    ordered = sorted(values)

    def rank_direction(rank):
        pivot = ordered[rank]
        preceding = sum(value < pivot for value in values)
        block = sorted(direction for value, direction in zip(values, directions) if value == pivot)
        return block[rank - preceding]

    low, high = _central_ranks(len(values))
    return (rank_direction(low) + rank_direction(high)) / 2


def _mean_form(first, second):
    """Среднее разреженных форм a·d+b*dcenter; None означает отсутствие сертификата."""
    if first is None or second is None:
        return None
    coefficients = {}
    for expression in (first, second):
        for index, coefficient in expression[0].items():
            coefficients[index] = coefficients.get(index, Fraction(0)) + coefficient / 2
    return (
        {index: value for index, value in coefficients.items() if value},
        (first[1] + second[1]) / 2,
    )


def _median_form(values, expression):
    """Достаточный сертификат линейности выбранных медианных рангов.

    Внутри блока равенств допустимы одинаковые формы или среднее всего блока
    ровно из двух элементов. Остальные случаи остаются несертифицированными.
    Градиентные строки строятся только для нужных блоков, без матрицы n×n.
    """
    ordered = sorted(values)
    low, high = _central_ranks(len(values))

    def indices(pivot):
        return [index for index, value in enumerate(values) if value == pivot]

    def common(block):
        first = expression(block[0])
        if first is None:
            return None
        return first if all(expression(index) == first for index in block[1:]) else None

    if ordered[low] != ordered[high]:
        return _mean_form(common(indices(ordered[low])), common(indices(ordered[high])))
    block = indices(ordered[low])
    if low != high and len(block) == 2:
        return _mean_form(expression(block[0]), expression(block[1]))
    return common(block)


def _ordinary_gradient(values, center, deviations):
    """Проверить достаточный сертификат, включая сокращение нелинейного dcenter."""
    zero = Fraction(0)
    center_form = _median_form(values, lambda index: ({index: Fraction(1)}, zero))
    center_gradient = None if center_form is None else center_form[0]

    def expression(index):
        residual = values[index] - center
        if residual == 0:
            if center_gradient == {index: Fraction(1)}:
                return {}, zero
            return None
        sign = Fraction(1 if residual > 0 else -1)
        coefficients, center_coefficient = {index: sign}, -sign
        if center_gradient is not None:
            for coordinate, weight in center_gradient.items():
                coefficients[coordinate] = coefficients.get(coordinate, zero) - sign * weight
            coefficients = {
                coordinate: weight for coordinate, weight in coefficients.items() if weight
            }
            center_coefficient = zero
        return coefficients, center_coefficient

    result = _median_form(deviations, expression)
    if result is None or result[1] != 0:
        return None
    return tuple(result[0].get(index, zero) for index in range(len(values)))


def median_absolute_deviation(values, normalization=1.0):
    """Вычислить normalization * median(abs(values - median(values))).

    Parameters
    ----------
    values : array_like, shape (n,)
        Непустой конечный вещественный вектор. NaN, inf и masked arrays
        отклоняются; выбор доступных наблюдений выполняется вызывающим кодом.
    normalization : float, optional
        Положительный конечный множитель. По умолчанию 1, без автоматической
        нормировки для нормального распределения.

    Returns
    -------
    float
        MAD в единицах входного вектора, округлённое в float64. Непредставимый
        результат вызывает FloatingPointError, без отсечения или насыщения.

    Notes
    -----
    Для чётного числа элементов медиана — среднее двух центральных рангов.
    Внутренние Fraction сохраняют точные значения переданных binary64,
    включая порядок абсолютных отклонений при не представимом в float64
    центре. Это не оценка приборного шума без дополнительных предпосылок.
    """
    vector = _finite_vector(values, "values")
    factor = Fraction(_positive_normalization(normalization))
    return _mad_value(_exact(vector), factor)


def median_absolute_deviation_of_differences(left, right, normalization=1.0):
    """Вычислить MAD попарных разностей без предварительного округления разности.

    Parameters
    ----------
    left, right : array_like, shape (n,)
        Непустые конечные вещественные векторы одинаковой длины. Каждая
        разность равна точному Fraction(right[i])-Fraction(left[i]).
        Пропуски и masked arrays отклоняются, пары не сжимаются.
    normalization : float, optional
        Положительный конечный множитель, по умолчанию 1.

    Returns
    -------
    float
        normalization * MAD(right-left), округлённое в float64. Ранги и
        медиана вычисляются до этого округления; используется та же формула,
        что в median_absolute_deviation. Непредставимый результат вызывает
        FloatingPointError, включая потерю ненулевого результата в нуле.
    """
    left, right = _difference_vectors(left, right)
    factor = Fraction(_positive_normalization(normalization))
    return _mad_value(_exact_differences(left, right), factor)


@dataclass(frozen=True, eq=False)
class MADSensitivity:
    """Неизменяемое значение MAD и его направленная чувствительность.

    Attributes
    ----------
    value : float
        Значение той же функции median_absolute_deviation.
    ordinary_gradient : ndarray of float or None
        Собственный неизменяемый градиент при достаточном сертификате.
        None не доказывает недифференцируемость.
    status : {'ordinary_certified', 'ordinary_uncertified'}
        Сертификат относится к вещественной MAD на точных binary64-входах;
        выходные числа округляются в float64. Это не сертификат производной
        буквально округлённой программы, всего ADR-метода или его риска.

    Notes
    -----
    directional вычисляет Hadamard-производную для любого конечного
    направления. Она может быть неаддитивной; в таком случае одна матрица
    или VJP не подменяет её. Обычный VJP доступен только при сертификате.
    """

    value: float
    ordinary_gradient: np.ndarray | None
    status: str
    _values: tuple[Fraction, ...] = field(repr=False)
    _center: Fraction = field(repr=False)
    _deviations: tuple[Fraction, ...] = field(repr=False)
    _normalization: Fraction = field(repr=False)
    _gradient: tuple[Fraction, ...] | None = field(repr=False)

    def directional(self, direction):
        """Вычислить одностороннюю Hadamard-производную вдоль direction."""
        vector = _finite_vector(direction, "direction", size=len(self._values))
        return self._directional_exact(_exact(vector))

    def _directional_exact(self, exact):
        """Применить общий направленный алгоритм к внутренним точным направлениям."""
        center_direction = _median_directional(self._values, exact)
        deviations_direction = tuple(
            (
                abs(delta - center_direction)
                if value == self._center
                else (1 if value > self._center else -1) * (delta - center_direction)
            )
            for value, delta in zip(self._values, exact)
        )
        result = self._normalization * _median_directional(self._deviations, deviations_direction)
        return _represented(result, "MAD directional derivative")

    def vjp(self, cotangent):
        """Вернуть обычное транспонированное действие для конечного скаляра.

        При ordinary_uncertified возникает ValueError: направленная
        производная не объявляется линейным оператором.
        """
        if self._gradient is None:
            raise ValueError("An ordinary MAD gradient is not certified; use directional")
        if np.ma.isMaskedArray(cotangent):
            raise ValueError("cotangent must be an explicit unmasked real scalar")
        raw = np.asarray(cotangent)
        if raw.shape != () or raw.dtype.kind not in "iuf" or not np.isfinite(raw):
            raise ValueError("cotangent must be a finite real scalar")
        weight = Fraction(float(raw))
        return _immutable([_represented(weight * value, "MAD VJP") for value in self._gradient])


def _sensitivity_from_exact(exact, factor, value):
    center, deviations, _ = _mad_parts(exact)
    gradient = _ordinary_gradient(exact, center, deviations)
    if gradient is not None:
        gradient = tuple(factor * coefficient for coefficient in gradient)
    represented_gradient = (
        None
        if gradient is None
        else _immutable([_represented(entry, "MAD gradient") for entry in gradient])
    )
    return MADSensitivity(
        value,
        represented_gradient,
        "ordinary_uncertified" if gradient is None else "ordinary_certified",
        exact,
        center,
        deviations,
        factor,
        gradient,
    )


def mad_sensitivity(values, normalization=1.0):
    """Создать MAD с точной направленной формулой и достаточным обычным сертификатом.

    Parameters
    ----------
    values : array_like, shape (n,)
        Конечные исходные значения без неявного удаления пропусков.
    normalization : float, optional
        Положительный конечный множитель той же функции MAD.

    Returns
    -------
    MADSensitivity
        Собственные неизменяемые данные; ordinary_uncertified означает только
        отсутствие достаточного сертификата в реализованных правилах.

    Notes
    -----
    Для выбранного ранга внутри блока равных значений берётся соответствующий
    ранг направлений этого блока. Композиция учитывает медиану центра,
    производную abs (в нуле — abs направления) и внешнюю медиану. Сертификат
    допускает одинаковые линейные формы в блоке, среднее всего блока из двух
    элементов и алгебраическое сокращение коэффициента при нелинейной
    производной центра. Полная классификация всех ветвей не выполняется.
    """
    vector = _finite_vector(values, "values")
    factor = _positive_normalization(normalization)
    value = median_absolute_deviation(vector, factor)
    return _sensitivity_from_exact(_exact(vector), Fraction(factor), value)


@dataclass(frozen=True, eq=False)
class DifferenceMADSensitivity:
    """MAD разностей как функция исходных левых и правых значений пары.

    ordinary_gradient и vjp имеют форму (2,n): первая строка относится
    к left, вторая к right. directional принимает два направления той же
    длины и вычитает их в точной арифметике до выбора направленных рангов.
    ordinary_uncertified не доказывает недифференцируемость; обычный VJP
    тогда недоступен. Сертификат относится к вещественной композиции на
    точных binary64-входах, а не к буквально округлённой программе.
    """

    value: float
    ordinary_gradient: np.ndarray | None
    status: str
    _mad: MADSensitivity = field(repr=False)

    def directional(self, d_left, d_right):
        """Вычислить Hadamard-производную по исходным парным значениям."""
        size = len(self._mad._values)
        left = _finite_vector(d_left, "d_left", size=size)
        right = _finite_vector(d_right, "d_right", size=size)
        return self._mad._directional_exact(_exact_differences(left, right))

    def vjp(self, cotangent):
        """Вернуть две строки обычного ковектора при наличии сертификата."""
        pulled = self._mad.vjp(cotangent)
        return _immutable(np.stack((-pulled, pulled)))


def mad_of_differences_sensitivity(left, right, normalization=1.0):
    """Создать чувствительность MAD по значениям исходных пар.

    Parameters
    ----------
    left, right : array_like, shape (n,)
        Непустые конечные векторы одинаковой длины. Вычисляются точные
        парные разности, а не MAD от предварительно округлённого right-left.
    normalization : float, optional
        Положительный конечный множитель той же функции MAD.

    Returns
    -------
    DifferenceMADSensitivity
        Неизменяемый результат с общим MAD-алгоритмом и достаточным
        сертификатом. Значения, градиенты и действия выводятся в float64;
        непредставимые результаты не обрезаются и не заменяются нулями.
    """
    left, right = _difference_vectors(left, right)
    factor = _positive_normalization(normalization)
    value = median_absolute_deviation_of_differences(left, right, factor)
    mad = _sensitivity_from_exact(_exact_differences(left, right), Fraction(factor), value)
    gradient = (
        None
        if mad.ordinary_gradient is None
        else _immutable(np.stack((-mad.ordinary_gradient, mad.ordinary_gradient)))
    )
    return DifferenceMADSensitivity(mad.value, gradient, mad.status, mad)
