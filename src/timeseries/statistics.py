"""Статистики временных рядов и условная калибровка нормальной модели.

Пропуски не удаляются неявно: вызывающая сторона передаёт конечный вектор
и сохраняет его соответствие исходным наблюдениям. Для MAD ранги и ветви
вычисляются над точными двоично-рациональными входами; результаты округляются
в float64. Reference-калибровка нормального ряда использует арифметику FP64
и не предоставляет интервальных сертификатов.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.special import logsumexp


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


def _hour_positions(values, name, *, size=None):
    if np.ma.isMaskedArray(values):
        raise ValueError(f"{name} must explicitly select integer positions")
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in "iu" or not raw.size:
        raise ValueError(f"{name} must be a nonempty integer vector")
    integers = tuple(int(value) for value in raw)
    limits = np.iinfo(np.int64)
    if (
        (size is not None and len(integers) != size)
        or integers[0] < limits.min
        or integers[-1] > limits.max
        or any(left >= right for left, right in zip(integers, integers[1:]))
        or integers[-1] - integers[0] > limits.max
    ):
        raise ValueError(f"{name} requires increasing distinct positions in a safe int64 range")
    array = np.array(integers, dtype=np.int64)
    return np.frombuffer(array.tobytes(), dtype=array.dtype)


def _ar1_correlation(left, right, rho):
    span = max(int(left[-1]), int(right[-1])) - min(int(left[0]), int(right[0]))
    if span > np.iinfo(np.int64).max:
        raise ValueError("Combined hour positions exceed the safe integer-lag range")
    lags = np.abs(left[:, None] - right[None, :])
    if rho == 0:
        return (lags == 0).astype(float)
    # Sign uses exact integer parity even when a lag is larger than 2**53.
    # Rounding a tiny magnitude to zero is ordinary FP64, not a rho threshold.
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        correlation = np.power(abs(rho), lags)
    return np.where(lags % 2, -correlation, correlation) if rho < 0 else correlation


def _reference_weights(log_masses):
    if not np.isfinite(log_masses).all():
        raise FloatingPointError("Reference log-weights must remain finite")
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        weights = np.exp(log_masses - logsumexp(log_masses))
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise FloatingPointError("A positive reference weight is not representable in FP64")
    weights /= weights.sum()
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise FloatingPointError("Normalized reference weights must remain positive and finite")
    return weights, np.log(weights)


def _reference_covariance(matrix):
    if not np.isfinite(matrix).all():
        raise FloatingPointError("Reference predictive covariance must remain finite")
    # Remove only antisymmetric rounding; no eigenvalue or diagonal correction.
    symmetric = 0.5 * matrix + 0.5 * matrix.T
    try:
        cholesky(symmetric, lower=True, check_finite=False)
    except np.linalg.LinAlgError as error:
        raise FloatingPointError(
            "Reference predictive covariance lost positive definiteness"
        ) from error
    return symmetric


@dataclass(frozen=True, eq=False, init=False)
class AR1ReferenceCalibration:
    """Reference-калибровка одного коррелированного нормального ряда.

    Parameters
    ----------
    values, positions, rho_grid, prior_weights
        Те же явные входы, что у fit_ar1_reference. Конструктор выполняет
        единственную калибровку; фабрика не добавляет другого алгоритма.

    Attributes
    ----------
    values : numpy.ndarray, shape (n,)
        Конечный исходный ряд в выбранной пользователем единице, n>=4.
    positions : numpy.ndarray of int64, shape (n,)
        Действительные целочасовые координаты; пропуски не сжимают лаги.
    rho_grid : numpy.ndarray, shape (J,)
        Различные коэффициенты строго внутри (-1,1).
    prior_weights, posterior_weights, log_posterior_weights : numpy.ndarray, shape (J,)
        Нормированные положительные веса сетки и логарифмы posterior-весов.
    mean_location, mean_precision : numpy.ndarray, shape (J,)
        GLS-оценки среднего и tau=1.T@inv(R)@1. Первые имеют единицу ряда;
        tau безразмерно, поскольку R — корреляционная матрица.
    residual_sum_squares, variance_mean : numpy.ndarray, shape (J,)
        Q и E[v|y,rho]=Q/(n-3), в квадрате единицы ряда.
    degrees_of_freedom : int
        n-1, число степеней свободы каждой прогнозной Student-компоненты.

    Notes
    -----
    Объявляется рабочая модель y|mu,v,rho~N(mu*1,v*R), R_ij=rho**|t_i-t_j|,
    и одна общая мера prior_weights[j]*dmu*dv/v для всех узлов rho.
    Сетка и prior-веса задаются до использования values. Их подбор по тому же
    ряду задаёт иной эмпирический prior, а не posterior объявленного анализа.
    Это не установленная ковариация атмосферной или приборной ошибки.
    Параметр v — маргинальная дисперсия ряда, не дисперсия AR-инновации.

    Все массивы собственные и неизменяемые. Отбеливание и суммы вычисляются
    в FP64 без явной обратной матрицы, добавок к диагонали и усечения спектра.
    Очень малая корреляция на большом лаге может округлиться в ноль; исходный
    rho сохраняется. Непредставимые положительные Q, масштабы или веса вызывают
    FloatingPointError. FP64-проверки не являются интервальным сертификатом.
    """

    values: np.ndarray
    positions: np.ndarray
    rho_grid: np.ndarray
    prior_weights: np.ndarray
    posterior_weights: np.ndarray
    log_posterior_weights: np.ndarray
    mean_location: np.ndarray
    mean_precision: np.ndarray
    residual_sum_squares: np.ndarray
    variance_mean: np.ndarray
    degrees_of_freedom: int
    _mean_offsets: np.ndarray = field(repr=False)
    _cholesky: tuple = field(repr=False)
    _whitened_ones: tuple = field(repr=False)
    _whitened_residuals: tuple = field(repr=False)

    def __init__(self, values, positions, *, rho_grid, prior_weights):
        data = _finite_vector(values, "values")
        if len(data) < 4:
            raise ValueError("Reference predictive covariance requires at least four observations")
        if np.all(data == data[0]):
            raise ValueError("A constant series has Q=0 and an improper reference posterior")
        times = _hour_positions(positions, "positions", size=len(data))
        rho = _finite_vector(rho_grid, "rho_grid")
        prior = _finite_vector(prior_weights, "prior_weights", size=len(rho))
        if np.any(np.abs(rho) >= 1) or len(np.unique(rho)) != len(rho):
            raise ValueError("rho_grid requires distinct values strictly inside (-1,1)")
        if np.any(prior <= 0):
            raise ValueError("prior_weights must be strictly positive")
        prior, log_prior = _reference_weights(np.log(prior))
        locations, offsets, precisions, residual_sums, log_masses = [], [], [], [], []
        factors, white_ones, white_residuals = [], [], []
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            centered = data - data[0]
            if not np.isfinite(centered).all():
                raise FloatingPointError("Reference centering lost its numerical scale")
            for index, correlation in enumerate(rho):
                matrix = _ar1_correlation(times, times, correlation)
                try:
                    factor = cholesky(matrix, lower=True, check_finite=False)
                except np.linalg.LinAlgError as error:
                    raise FloatingPointError(
                        "AR(1) correlation factorization failed in FP64"
                    ) from error
                one = solve_triangular(factor, np.ones(len(data)), lower=True, check_finite=False)
                white = solve_triangular(factor, centered, lower=True, check_finite=False)
                precision = float(one @ one)
                if not np.isfinite(precision) or precision <= 0:
                    raise FloatingPointError("Reference mean precision lost its numerical scale")
                offset = float((one @ white) / precision)
                residual = solve_triangular(
                    factor, centered - offset, lower=True, check_finite=False
                )
                squared = float(residual @ residual)
                location = float(data[0] + offset)
                if not np.isfinite([squared, offset, location]).all() or squared <= 0:
                    raise FloatingPointError(
                        "A positive reference residual sum is not representable"
                    )
                locations.append(location)
                offsets.append(offset)
                precisions.append(precision)
                residual_sums.append(squared)
                log_masses.append(
                    log_prior[index]
                    - np.log(np.diag(factor)).sum()
                    - 0.5 * np.log(precision)
                    - 0.5 * (len(data) - 1) * np.log(squared)
                )
                factors.append(_immutable(factor))
                white_ones.append(_immutable(one))
                white_residuals.append(_immutable(residual))
            posterior, log_posterior = _reference_weights(np.asarray(log_masses))
            variance = np.asarray(residual_sums) / (len(data) - 3)
            if not np.isfinite(variance).all() or np.any(variance <= 0):
                raise FloatingPointError("Reference variance means are not representable")
        arrays = dict(
            values=data,
            rho_grid=rho,
            prior_weights=prior,
            posterior_weights=posterior,
            log_posterior_weights=log_posterior,
            mean_location=locations,
            mean_precision=precisions,
            residual_sum_squares=residual_sums,
            variance_mean=variance,
            _mean_offsets=offsets,
        )
        for name, array in arrays.items():
            object.__setattr__(self, name, _immutable(array))
        object.__setattr__(self, "positions", times)
        object.__setattr__(self, "degrees_of_freedom", len(data) - 1)
        object.__setattr__(self, "_cholesky", tuple(factors))
        object.__setattr__(self, "_whitened_ones", tuple(white_ones))
        object.__setattr__(self, "_whitened_residuals", tuple(white_residuals))

    def predictive_moments(self, positions, *, relation):
        """Вернуть общие среднее и ковариацию конечной прогнозной смеси.

        Parameters
        ----------
        positions : array_like of int, shape (m,)
            Различные возрастающие целочасовые координаты, m>=1.
        relation : {'replicate', 'continuation'}
            replicate задаёт новую независимую реализацию при общих mu,v,rho.
            continuation условится на исходном ряде того же AR(1)-процесса;
            её координаты должны быть непересекающимися с positions калибровки.

        Returns
        -------
        mean : numpy.ndarray, shape (m,)
            Собственное неизменяемое среднее в единице исходного ряда.
        covariance : numpy.ndarray, shape (m,m)
            Собственная неизменяемая ковариация в квадрате этой единицы,
            включая неопределённость mu,v и межкомпонентный разброс средних.

        Notes
        -----
        При фиксированном rho результат интегрирования mu,v является Student
        с n-1 степенями свободы; его матрица масштаба не равна ковариации.
        Смесь по rho в общем случае не является ни Student, ни нормальной.
        Для continuation используется Schur complement R_**-R_*o R_oo^-1 R_o*.
        Его mean-uncertainty член равен b*b.T/tau, b=1-R_*o R_oo^-1 1.
        Возвращается симметричная часть ковариации FP64; отрицательные
        собственные числа не исправляются, потеря SPD вызывает исключение.
        """
        if not isinstance(relation, str) or relation not in ("replicate", "continuation"):
            raise ValueError("relation must be explicitly 'replicate' or 'continuation'")
        future = _hour_positions(positions, "predictive positions")
        if relation == "continuation" and not set(future).isdisjoint(self.positions):
            raise ValueError("Continuation positions must be disjoint from training positions")
        count = len(future)
        means, covariances = [], []
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            for index, rho in enumerate(self.rho_grid):
                correlation = _ar1_correlation(future, future, rho)
                if relation == "replicate":
                    relative_mean = np.full(count, self._mean_offsets[index])
                    conditional = correlation
                    shift = np.ones(count)
                else:
                    cross = _ar1_correlation(future, self.positions, rho)
                    projected = solve_triangular(
                        self._cholesky[index], cross.T, lower=True, check_finite=False
                    )
                    relative_mean = (
                        self._mean_offsets[index] + projected.T @ self._whitened_residuals[index]
                    )
                    conditional = correlation - projected.T @ projected
                    shift = np.ones(count) - projected.T @ self._whitened_ones[index]
                component = self.variance_mean[index] * (
                    conditional + np.outer(shift, shift) / self.mean_precision[index]
                )
                if not np.isfinite(relative_mean).all():
                    raise FloatingPointError("Reference predictive mean lost its numerical scale")
                means.append(relative_mean)
                covariances.append(_reference_covariance(component))
            relative_mean = self.posterior_weights @ np.asarray(means)
            covariance = np.zeros((count, count))
            for weight, mean, component in zip(self.posterior_weights, means, covariances):
                difference = mean - relative_mean
                covariance += weight * (component + np.outer(difference, difference))
            mean = self.values[0] + relative_mean
            if not np.isfinite(mean).all():
                raise FloatingPointError("Reference mixture mean lost its numerical scale")
            covariance = _reference_covariance(covariance)
        return _immutable(mean), _immutable(covariance)


def fit_ar1_reference(values, positions, *, rho_grid, prior_weights):
    """Калибровать mu,v и конечную AR(1)-сетку под общей мерой dmu*dv/v.

    Parameters
    ----------
    values : array_like, shape (n,)
        Конечный неконстантный вещественный ряд в одной объявленной единице;
        n>=4 требуется для конечной прогнозной ковариации. Пропуски явно
        выбираются вызывающим кодом; маска фиксирована для всех узлов rho.
    positions : array_like of int, shape (n,)
        Различные возрастающие целочасовые координаты с безопасным int64 span.
    rho_grid : array_like, shape (J,)
        Явная непустая сетка различных коэффициентов строго внутри (-1,1).
        Задаётся до использования values, как и prior_weights.
    prior_weights : array_like, shape (J,)
        Явные конечные строго положительные массы prior сетки; нормируются.

    Returns
    -------
    AR1ReferenceCalibration
        Собственные неизменяемые данные, GLS/IG-статистики и posterior-веса.
        Календарное окно, физическая интерпретация остатков и перенос закона
        между эпизодами остаются явными решениями вызывающего приложения.

    Raises
    ------
    ValueError
        Нарушена форма, диапазон или математическая предпосылка, включая n<4,
        константный ряд, повторный rho или неположительный вес prior.
    FloatingPointError
        Допустимая модель потеряла численную представимость при факторизации,
        вычислении положительных статистик, весов или прогнозных моментов.
        Параметры не обрезаются, дополнительная дисперсия не вводится.
    """
    return AR1ReferenceCalibration(
        values, positions, rho_grid=rho_grid, prior_weights=prior_weights
    )
