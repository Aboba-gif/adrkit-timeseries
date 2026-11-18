"""Положительные аффинные подшаги и расщепление квадратичного стока."""

from dataclasses import dataclass, field
from operator import index
from importlib import import_module

import numpy as np
from scipy import sparse
from scipy.special import exprel, gammainc, gammaln, hyp1f1


def _scalar(value, name, *, nonnegative=False, positive=False):
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real scalar")
    value = float(raw)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real scalar")
    if (nonnegative and value < 0) or (positive and value <= 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return value


def _array_module(backend):
    if backend == "numpy":
        return np
    if backend != "cupy":
        raise ValueError("backend must be 'numpy' or 'cupy'")
    try:
        return import_module("cupy")
    except ImportError as error:
        raise ImportError("The 'cupy' backend requires a compatible CuPy installation") from error


def _readonly(value, xp):
    if xp is np:
        value.flags.writeable = False
    return value


def _owned(value, name, *, shape=None, nonnegative=False, xp=np):
    raw = xp.asarray(value)
    if raw.dtype.kind not in "iuf" or not xp.isfinite(raw).all():
        raise ValueError(f"{name} must contain finite real values")
    with np.errstate(over="ignore", invalid="ignore"):
        values = xp.array(raw, dtype=float, copy=True)
    if not xp.isfinite(values).all():
        raise ValueError(f"{name} must remain finite in double precision")
    if shape is not None and values.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if nonnegative and xp.any(values < 0):
        raise ValueError(f"{name} must be nonnegative")
    return _readonly(values, xp)


def _matrix(value, name, size=None, *, xp=np):
    matrix_module = sparse if xp is np else import_module("cupyx.scipy.sparse")
    if (
        np.dtype(
            value.dtype
            if sparse.issparse(value) or matrix_module.issparse(value)
            else xp.asarray(value).dtype
        ).kind
        not in "iuf"
    ):
        raise ValueError(f"{name} must be real")
    matrix = matrix_module.csr_matrix(value, dtype=float, copy=True)
    matrix.sum_duplicates()
    if xp is np:
        matrix.eliminate_zeros()
    # Explicit zeros do not change a device matrix's action. Keep its signed
    # entries intact without an unnecessary device compression operation.
    if (
        matrix.shape[0] != matrix.shape[1]
        or not matrix.shape[0]
        or (size is not None and matrix.shape != (size, size))
        or not xp.isfinite(matrix.data).all()
    ):
        raise ValueError(f"{name} must be a finite square matrix of the required size")
    return matrix


def _operator_directions(partials, directions, count, size, rate, *, xp=np):
    partials = tuple((partials or {}).values())
    matrices = tuple(_matrix(value, "operator partial", size, xp=xp) / rate for value in partials)
    if directions is None:
        if matrices:
            raise ValueError("Operator partials require explicit operator_directions")
        coefficients = xp.zeros((0, count))
    else:
        coefficients = _owned(
            directions, "operator_directions", shape=(len(matrices), count), xp=xp
        )
    active = [j for j, matrix in enumerate(matrices) if matrix.nnz and xp.any(coefficients[j])]
    return tuple(matrices[j] for j in active), coefficients[active]


def _decay(u, beta, step, *, xp=np):
    if beta == 0.0 or step == 0.0:
        return u.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        product = (beta * step) * u
        value = xp.asarray(u / (1.0 + product))
    if xp is not np:
        # Device selection avoids a host synchronization at every reaction step.
        log_u = xp.log(u)
        fallback = xp.exp(log_u - xp.logaddexp(0.0, np.log(beta) + np.log(step) + log_u))
        return xp.where(u == 0.0, 0.0, xp.where(xp.isfinite(product), value, fallback))
    overflow = ~np.isfinite(product) & (u > 0)
    if overflow.any():
        log_u = np.log(u[overflow])
        value[overflow] = np.exp(log_u - np.logaddexp(0.0, np.log(beta) + np.log(step) + log_u))
    value[u == 0.0] = 0.0
    return value


def _decay_ratio(u, value, *, xp=np):
    if xp is not np:
        return xp.where(u > 0.0, value / u, 1.0)
    ratio = np.ones_like(u)
    positive = u > 0.0
    ratio[positive] = value[positive] / u[positive]
    return ratio


def _decay_curvature_product(u, ratio, beta, step, tangent, cotangent, *, xp=np):
    if beta == 0.0 or step == 0.0:
        return xp.zeros_like(u)
    # R'' can overflow by itself while the weighted action is zero or finite.
    first = (beta * step) * ratio
    second = ratio * tangent
    third = (2.0 * ratio) * cotangent
    partial = first * second
    direct = -partial * third
    normal = xp.ones(u.shape, dtype=bool)
    for value in (first, second, third, partial, direct):
        normal &= xp.isfinite(value) & (xp.abs(value) >= np.finfo(float).tiny)
    log_k = np.log(beta) + np.log(step)
    log_denominator = xp.logaddexp(0.0, log_k + xp.log(u))
    log_magnitude = (
        np.log(2.0)
        + log_k
        + xp.log(xp.abs(tangent))
        + xp.log(xp.abs(cotangent))
        - 3.0 * log_denominator
    )
    fallback = -xp.sign(tangent) * xp.sign(cotangent) * xp.exp(log_magnitude)
    zero = (tangent == 0.0) | (cotangent == 0.0)
    return xp.where(zero, 0.0, xp.where(normal, direct, fallback))


def _decay_increment(u, shifted, increment, beta, step, *, xp=np):
    if beta == 0.0 or step == 0.0:
        return increment.copy()
    k = beta * step
    denominator = 1.0 + k * u
    shifted_denominator = 1.0 + k * shifted
    direct = (increment / denominator) / shifted_denominator
    valid = xp.isfinite(denominator) & xp.isfinite(shifted_denominator)
    if k == 0.0:
        valid = xp.zeros(u.shape, dtype=bool)
    log_k = np.log(beta) + np.log(step)
    log_denominator = xp.logaddexp(0.0, log_k + xp.log(u))
    log_shifted_denominator = xp.logaddexp(0.0, log_k + xp.log(shifted))
    fallback = xp.sign(increment) * xp.exp(
        xp.log(xp.abs(increment)) - log_denominator - log_shifted_denominator
    )
    return xp.where(increment == 0.0, 0.0, xp.where(valid, direct, fallback))


def _decay_pullback_tangent(
    u, value, beta, step, tangent, cotangent, cotangent_direction, dbeta, *, parameters, xp
):
    """Differentiate a reaction pullback, retaining weighted mixed beta terms."""
    ratio = _decay_ratio(u, value, xp=xp)
    slope = ratio**2
    pulled = slope * cotangent
    changed = slope * cotangent_direction + _decay_curvature_product(
        u, ratio, beta, step, tangent, cotangent, xp=xp
    )
    if not parameters:
        return pulled, changed, 0.0
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        log_u = xp.log(u)
        log_denominator = (
            xp.zeros_like(u)
            if beta == 0
            else xp.logaddexp(0.0, np.log(beta) + np.log(step) + log_u)
        )

        def weighted(log_coefficient, *factors, negative=False):
            log_magnitude = log_coefficient.copy()
            sign = -xp.ones_like(u) if negative else xp.ones_like(u)
            zero = u == 0
            for factor in factors:
                log_magnitude = log_magnitude + xp.log(xp.abs(factor))
                sign = sign * xp.sign(factor)
                zero = zero | (factor == 0)
            return xp.where(zero, 0.0, sign * xp.exp(log_magnitude))

        beta_log = np.log(step) + 2 * log_u - 2 * log_denominator
        mixed_log = np.log(2.0) + np.log(step) + log_u - 3 * log_denominator
        changed_beta = weighted(beta_log, cotangent_direction, negative=True)
        changed_beta += weighted(mixed_log, tangent, cotangent, negative=True)
        if dbeta != 0:
            changed += weighted(mixed_log, dbeta, cotangent, negative=True)
            beta_beta_log = np.log(2.0) + 2 * np.log(step) + 3 * log_u - 3 * log_denominator
            changed_beta += weighted(beta_beta_log, dbeta, cotangent)
    return pulled, changed, xp.sum(changed_beta)


def _decay_partials(u, value, step, *, xp=np):
    if xp is not np:
        derivative = _decay_ratio(u, value, xp=xp) ** 2
        return derivative, -(step * value) * value
    derivative = _decay_ratio(u, value) ** 2
    with np.errstate(over="ignore", invalid="ignore"):
        beta_derivative = np.asarray(-(step * value) * value)
    if not np.isfinite(beta_derivative).all():
        raise FloatingPointError("The quadratic-decay derivative is not representable")
    return derivative, beta_derivative


def quadratic_decay(u, beta, step):
    """Вычислить точный поток u'=-beta*u² покомпонентно.

    Parameters
    ----------
    u : array_like
        Конечное неотрицательное состояние.
    beta, step : float
        Неотрицательные коэффициент стока и длительность подшага.
        Если состояние имеет единицу C, а время — T, единица beta
        равна 1/(C*T), единица step — T.

    Returns
    -------
    values : ndarray
        u/(1+beta*step*u), той же формы, что u.
    """
    values = _owned(u, "u", nonnegative=True)
    return _decay(
        values,
        _scalar(beta, "beta", nonnegative=True),
        _scalar(step, "step", nonnegative=True),
    )


def quadratic_decay_partials(u, beta, step):
    """Получить диагональ производной потока по u и производную по beta.

    Parameters
    ----------
    u : array_like
        Конечное неотрицательное состояние.
    beta, step : float
        Неотрицательные коэффициент стока и длительность.
        Единица beta — 1/(C*T), где C — единица состояния, T — времени.

    Returns
    -------
    state_partial, beta_partial : ndarray
        Массивы формы u: (1+beta*step*u)^(-2) и -step*R(u)².
    """
    values = _owned(u, "u", nonnegative=True)
    beta, step = _scalar(beta, "beta", nonnegative=True), _scalar(step, "step", nonnegative=True)
    return _decay_partials(values, _decay(values, beta, step), step)


def poisson_degree(mean, tolerance):
    """Выбрать наименьшую степень K>=2 с 2*P(Poisson(mean)>K)<=tolerance.

    Parameters
    ----------
    mean : float
        Конечное неотрицательное среднее rate*step.
    tolerance : float
        Положительный локальный допуск нормы транспортного оператора.

    Returns
    -------
    degree : int
        Степень, фиксируемая до изменения параметров оператора.
    """
    mean, tolerance = _scalar(mean, "mean", nonnegative=True), _scalar(
        tolerance, "tolerance", positive=True
    )
    if 2 * gammainc(3.0, mean) <= tolerance:
        return 2
    low, high = 2, max(3, int(np.ceil(mean)))
    while 2 * gammainc(high + 1.0, mean) > tolerance:
        low, high = high, 2 * high + 1
    while high - low > 1:
        middle = (low + high) // 2
        if 2 * gammainc(middle + 1.0, mean) <= tolerance:
            high = middle
        else:
            low = middle
    return high


def _weights(alpha, step, rate, degree, *, second=False):
    k = np.arange(degree + 1, dtype=float)
    mean = rate * step
    alpha_step = alpha * step
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        growth = np.exp(alpha_step)
        if alpha < 0.0 and -alpha_step >= 1.0:
            total = -np.expm1(alpha_step) / (-alpha)
            log_total_derivative = np.log(gammainc(2.0, -alpha_step)) - 2 * np.log(-alpha)
            if second:
                log_total_second = (
                    np.log(2.0) + np.log(gammainc(3.0, -alpha_step)) - 3 * np.log(-alpha)
                )
        else:
            total = step * exprel(alpha_step)
            log_total_derivative = (
                2 * np.log(step) - np.log(2.0) + np.log(hyp1f1(2.0, 3.0, alpha_step))
            )
            if second:
                log_total_second = (
                    3 * np.log(step) - np.log(3.0) + np.log(hyp1f1(3.0, 4.0, alpha_step))
                )
        total_derivative = np.exp(log_total_derivative)
        if second:
            total_second = np.exp(log_total_second)
    if not np.isfinite([mean, growth, total, total_derivative]).all() or total <= 0:
        raise FloatingPointError("The affine weights are outside the representable range")
    if second and not np.isfinite(total_second):
        raise FloatingPointError("The source-weight second derivative is not representable")
    if mean == 0.0:
        transport = np.zeros(degree + 1)
        transport[0] = 1.0
        source, source_derivative = transport * total, transport * total_derivative
        result = growth, total, transport, source, source_derivative, 0.0
        return (*result, transport * total_second) if second else result
    log_power = k * np.log(mean)
    log_transport = log_power - gammaln(k + 1.0)
    transport = np.exp(log_transport - np.max(log_transport))
    transport /= transport.sum()
    difference = alpha - rate
    argument = difference * step if np.isfinite(difference) else alpha_step - mean
    with np.errstate(divide="ignore", invalid="ignore"):
        log_w = (
            np.log(step)
            + log_power
            - gammaln(k + 2.0)
            + np.log(hyp1f1(k + 1.0, k + 2.0, argument))
        )
        log_dw = (
            2 * np.log(step)
            + log_power
            - gammaln(k + 1.0)
            - np.log(k + 2.0)
            + np.log(hyp1f1(k + 2.0, k + 3.0, argument))
        )
        if second:
            log_ddw = (
                3 * np.log(step)
                + log_power
                - gammaln(k + 1.0)
                - np.log(k + 3.0)
                + np.log(hyp1f1(k + 3.0, k + 4.0, argument))
            )
    # При положительном затухании гамма-форма сохраняет точность и тогда,
    # когда гипергеометрическая функция конечна, но уже неточна.
    # Субнормальная вероятность не даёт нужной относительной точности:
    # для неё или непредставимого логарифма остаётся гипергеометрическая запись.
    if rate > alpha:
        log_loss = (
            np.logaddexp(np.log(rate), np.log(-alpha)) if alpha < 0.0 else np.log(rate - alpha)
        )
        probability = gammainc(k + 1.0, -argument)
        derivative_probability = gammainc(k + 2.0, -argument)
        if second:
            second_probability = gammainc(k + 3.0, -argument)
        with np.errstate(divide="ignore", invalid="ignore"):
            alternative = k * (np.log(rate) - log_loss) - log_loss + np.log(probability)
            alternative_dw = (
                np.log(k + 1.0)
                + k * (np.log(rate) - log_loss)
                - 2 * log_loss
                + np.log(derivative_probability)
            )
            if second:
                alternative_ddw = (
                    np.log(k + 1.0)
                    + np.log(k + 2.0)
                    + k * (np.log(rate) - log_loss)
                    - 3 * log_loss
                    + np.log(second_probability)
                )
        normal_minimum = np.finfo(float).tiny
        log_w = np.where(
            (probability >= normal_minimum) & np.isfinite(alternative), alternative, log_w
        )
        log_dw = np.where(
            (derivative_probability >= normal_minimum) & np.isfinite(alternative_dw),
            alternative_dw,
            log_dw,
        )
        if second:
            log_ddw = np.where(
                (second_probability >= normal_minimum) & np.isfinite(alternative_ddw),
                alternative_ddw,
                log_ddw,
            )
    if np.isnan(log_w).any() or np.isposinf(log_w).any() or not np.isfinite(log_w).any():
        raise FloatingPointError("The source weights are not representable")
    maximum = np.max(log_w)
    normalized = np.exp(log_w - maximum)
    normalization = normalized.sum()
    normalized /= normalization
    ratios = np.zeros_like(k)
    finite = np.isfinite(log_w) & np.isfinite(log_dw)
    ratios[finite] = np.exp(log_dw[finite] - log_w[finite])
    source = total * normalized
    # S*p_k can be normal even when p_k is subnormal or rounded to zero.
    # Recover that same S*I_k/sum(I) using the existing logarithmic normalizer.
    small_probability = (normalized < np.finfo(float).tiny) & np.isfinite(log_w)
    if small_probability.any():
        source[small_probability] = np.exp(
            np.log(total) + log_w[small_probability] - (maximum + np.log(normalization))
        )
    if np.any((source > 0) & ~finite):
        raise FloatingPointError("The source-weight derivative is not representable")
    source_derivative = source * (total_derivative / total + ratios - normalized @ ratios)
    if not np.isfinite(source_derivative).all():
        raise FloatingPointError("The normalized source derivative is not representable")
    result = (
        growth,
        total,
        transport,
        source,
        source_derivative,
        float(gammainc(degree + 1.0, mean)),
    )
    if not second:
        return result
    second_ratios = np.zeros_like(k)
    active = source > 0
    finite_second = np.isfinite(log_w) & np.isfinite(log_ddw)
    if np.any(active & ~finite_second):
        raise FloatingPointError("The source-weight second moment is not representable")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        second_ratios[active] = np.exp(log_ddw[active] - log_w[active])
        first_mean = normalized @ ratios
        second_mean = normalized @ second_ratios
        source_second = source * (
            total_second / total
            + 2 * (total_derivative / total - first_mean) * (ratios - first_mean)
            + second_ratios
            - second_mean
        )
    if not np.isfinite(source_second).all():
        raise FloatingPointError("The normalized source second derivative is not representable")
    return (*result, source_second)


@dataclass(frozen=True, init=False)
class PositiveAffineAction:
    r"""Положительный полином для u'=(Q+alpha*I)u+f.

    Parameters
    ----------
    Q : sparse matrix or array_like, shape (N, N)
        Матрица с неотрицательными внедиагональными элементами и нулевыми
        суммами строк. Проверка сумм учитывает округление сложения строки.
    alpha : float
        Пространственно постоянный коэффициент роста в обратных единицах времени.
    step, rate : float
        Положительный шаг и заранее фиксированная граница rate>=max(-diag(Q)).
        Единицы rate обратны единицам step.
    degree : int
        Заранее фиксированная степень не меньше двух.
    backend : {"numpy", "cupy"}, optional
        Место хранения и арифметики массивов. CuPy загружается только при
        явном выборе; подготовка генератора и полиномиальных весов общая.

    Attributes
    ----------
    size : int
        Число компонент состояния.
    transport_tail : float
        P(Poisson(rate*step)>degree); погрешность транспорта не выше удвоенного хвоста.
    source_weight_sum : float
        S=integral(exp(alpha*s), s=0..step), в единицах времени.

    Notes
    -----
    P=I+Q/rate. Транспортные пуассоновы веса нормируются к единице,
    интегральные источниковые — к S=step*exprel(alpha*step).
    rate и degree не дифференцируются. Все входные матрицы копируются.
    При backend='cupy' вызывающий код сохраняет текущее CUDA-устройство,
    выбранное при создании объекта, на всё время его использования.
    """

    size: int
    alpha: float
    step: float
    rate: float
    degree: int
    transport_tail: float
    source_weight_sum: float
    _growth: float = field(repr=False)
    _transition: object = field(repr=False)
    _a: object = field(repr=False)
    _b: object = field(repr=False)
    _db: object = field(repr=False)
    backend: str

    def __init__(self, Q, alpha, *, step, rate, degree, backend="numpy"):
        xp = _array_module(backend)
        operator = _matrix(Q, "Q")
        alpha, step, rate = (
            _scalar(alpha, "alpha"),
            _scalar(step, "step", positive=True),
            _scalar(rate, "rate", positive=True),
        )
        if isinstance(degree, (bool, np.bool_)):
            raise ValueError("degree must be an integer >=2")
        try:
            degree = index(degree)
        except TypeError as error:
            raise ValueError("degree must be an integer >=2") from error
        if degree < 2:
            raise ValueError("degree must be an integer >=2")
        diagonal = operator.diagonal()
        off = operator - sparse.diags(diagonal)
        row_norm = np.asarray(abs(operator).sum(axis=1)).ravel()
        count_eps = (operator.getnnz(axis=1) + 1) * np.finfo(float).eps
        row_error = count_eps / (1.0 - count_eps) * row_norm
        if (
            np.any(off.data < 0)
            or np.any(np.abs(np.asarray(operator.sum(axis=1)).ravel()) > row_error)
            or rate < np.max(-diagonal)
        ):
            raise ValueError("Q must be a row-zero Metzler matrix dominated by rate")
        transition = (sparse.eye(operator.shape[0], format="csr") + operator / rate).tocsr()
        if np.any(transition.data < 0):
            raise ValueError("The supplied rate does not give nonnegative transition entries")
        growth, total, a, b, db, tail = _weights(alpha, step, rate, degree)
        for values in (transition.data, transition.indices, transition.indptr):
            values.flags.writeable = False
        if xp is not np:
            transition = import_module("cupyx.scipy.sparse").csr_matrix(transition)
        for name, value in dict(
            size=operator.shape[0],
            alpha=alpha,
            step=step,
            rate=rate,
            degree=degree,
            transport_tail=tail,
            source_weight_sum=total,
            _growth=growth,
            _transition=transition,
            _a=_owned(a, "weights", xp=xp),
            _b=_owned(b, "weights", xp=xp),
            _db=_owned(db, "weight derivatives", xp=xp),
            backend=backend,
        ).items():
            object.__setattr__(self, name, value)

    @property
    def _xp(self):
        # Keep frames serializable: modules are resolved, never stored in them.
        return _array_module(self.backend)

    @property
    def array_module(self):
        """Модуль NumPy или CuPy, выбранный для арифметики этого подшага."""
        return self._xp

    def to_host(self, values):
        """Вернуть независимый NumPy-массив с явной синхронизацией устройства."""
        if self._xp is np:
            return np.array(values, copy=True)
        return self._xp.asnumpy(values)

    @property
    def transport_weights(self):
        """ndarray : Независимые нормированные пуассоновы веса."""
        return _owned(self._a, "weights", xp=self._xp)

    @property
    def source_weights(self):
        """ndarray : Независимые интегральные источниковые веса."""
        return _owned(self._b, "weights", xp=self._xp)

    @property
    def source_alpha_derivative(self):
        """ndarray : Производные источниковых весов по alpha."""
        return _owned(self._db, "weight derivatives", xp=self._xp)

    @property
    def source_alpha_second_derivative(self):
        """ndarray : Вторые производные нормированных источниковых весов по alpha.

        Вычисляются по запросу для той же фиксированной степени и rate.
        Производные общей интенсивности и нормировки включены. Массив
        независим, хранится в выбранном backend и не кэшируется в объекте.
        Непредставимая вторая производная вызывает FloatingPointError только
        при обращении к этому свойству; обычные apply и JVP её не вычисляют.
        """
        weights = _weights(self.alpha, self.step, self.rate, self.degree, second=True)[-1]
        return _owned(weights, "second weight derivatives", xp=self._xp)

    def _powers(self, u, f):
        up = self._xp.empty((self.degree + 1, self.size))
        fp = self._xp.empty_like(up)
        up[0], fp[0] = u, f
        for k in range(1, self.degree + 1):
            up[k] = self._transition @ up[k - 1]
            fp[k] = self._transition @ fp[k - 1]
        return up, fp

    def _value(self, powers):
        value = self._growth * (self._a @ powers[0]) + self._b @ powers[1]
        if self._xp is np and not np.isfinite(value).all():
            raise FloatingPointError("The affine substep is not representable")
        return value

    def _tangent(self, powers, du, df, dP, dalpha):
        partials = () if dP is None else (dP,)
        directions = self._xp.ones((len(partials), 1))
        return self._tangent_many(
            powers,
            du[:, None],
            df[:, None],
            partials,
            directions,
            self._xp.array([dalpha]),
        )[:, 0]

    def _tangent_many(self, powers, du, df, partials, directions, dalpha, *, plan=None):
        xp = self._xp
        result = xp.zeros_like(du)
        if plan is None:
            active = xp.flatnonzero(xp.any(directions, axis=0))
            plan = (
                bool(xp.any(dalpha)),
                slice(None) if active.size == du.shape[1] else active,
            )
        varying_alpha, selected = plan
        if varying_alpha or partials:
            up, fp = powers
        operator_coefficients = directions[:, selected]
        for k in range(self.degree + 1):
            result += self._growth * self._a[k] * du + self._b[k] * df
            if varying_alpha:
                result += xp.multiply.outer(
                    self.step * self._growth * self._a[k] * up[k] + self._db[k] * fp[k],
                    dalpha,
                )
            if k < self.degree:
                du, df = self._transition @ du, self._transition @ df
                if partials:
                    du[:, selected] += (
                        xp.column_stack([matrix @ up[k] for matrix in partials])
                        @ operator_coefficients
                    )
                    df[:, selected] += (
                        xp.column_stack([matrix @ fp[k] for matrix in partials])
                        @ operator_coefficients
                    )
        return result

    def _pullback(self, powers, cotangent, partials):
        xp = self._xp
        up, fp = powers
        u_bar, f_bar = xp.zeros(self.size), xp.zeros(self.size)
        operator_bar = {name: 0.0 for name in partials}
        alpha_bar = self.step * self._growth * (self._a @ (up @ cotangent)) + self._db @ (
            fp @ cotangent
        )
        if xp is np:
            alpha_bar = float(alpha_bar)
        for k in range(self.degree, -1, -1):
            u_bar += self._growth * self._a[k] * cotangent
            f_bar += self._b[k] * cotangent
            if k:
                for name, matrix in partials.items():
                    contribution = u_bar @ (matrix @ up[k - 1]) + f_bar @ (matrix @ fp[k - 1])
                    operator_bar[name] += float(contribution) if xp is np else contribution
                u_bar = self._transition.T @ u_bar
                f_bar = self._transition.T @ f_bar
        return {"u": u_bar, "f": f_bar, "operator": operator_bar, "alpha": alpha_bar}

    def _power_directions(self, powers, du, df, dP):
        xp = self._xp
        up, fp = powers
        dup, dfp = xp.empty_like(up), xp.empty_like(fp)
        dup[0], dfp[0] = du, df
        for k in range(1, self.degree + 1):
            dup[k] = self._transition @ dup[k - 1]
            dfp[k] = self._transition @ dfp[k - 1]
            if dP is not None:
                dup[k] += dP @ up[k - 1]
                dfp[k] += dP @ fp[k - 1]
        return dup, dfp

    def _pullback_directional(
        self,
        powers,
        power_directions,
        cotangent,
        cotangent_direction,
        partials,
        dP,
        dalpha,
        source_second,
    ):
        """Tangent-over-reverse of this fixed, normalized finite polynomial."""
        xp = self._xp
        up, fp = powers
        dup, dfp = power_directions
        u_bar, f_bar = xp.zeros(self.size), xp.zeros(self.size)
        du_bar, df_bar = xp.zeros(self.size), xp.zeros(self.size)
        operator_bar = {name: 0.0 for name in partials}
        alpha_bar = self.step * self._growth * (
            self._a @ (dup @ cotangent + up @ cotangent_direction)
        ) + self._db @ (dfp @ cotangent + fp @ cotangent_direction)
        if dalpha != 0:
            alpha_bar += dalpha * (
                (self.step * self._growth) * self.step * (self._a @ (up @ cotangent))
                + source_second @ (fp @ cotangent)
            )
        for k in range(self.degree, -1, -1):
            transport = self._growth * self._a[k]
            u_bar += transport * cotangent
            f_bar += self._b[k] * cotangent
            du_bar += transport * cotangent_direction
            df_bar += self._b[k] * cotangent_direction
            if dalpha != 0:
                du_bar += self.step * transport * dalpha * cotangent
                df_bar += self._db[k] * dalpha * cotangent
            if k:
                for name, matrix in partials.items():
                    operator_bar[name] += (
                        du_bar @ (matrix @ up[k - 1])
                        + u_bar @ (matrix @ dup[k - 1])
                        + df_bar @ (matrix @ fp[k - 1])
                        + f_bar @ (matrix @ dfp[k - 1])
                    )
                du_bar = self._transition.T @ du_bar
                df_bar = self._transition.T @ df_bar
                if dP is not None:
                    du_bar += dP.T @ u_bar
                    df_bar += dP.T @ f_bar
                u_bar = self._transition.T @ u_bar
                f_bar = self._transition.T @ f_bar
        return u_bar, {"u": du_bar, "f": df_bar, "operator": operator_bar, "alpha": alpha_bar}

    def apply(self, u, f):
        """Применить аффинный подшаг.

        Parameters
        ----------
        u, f : array_like, shape (N,)
            Неотрицательные состояние u и нагрузка f. Единицы f:
            состояние/время.

        Returns
        -------
        values : ndarray, shape (N,)
            Состояние после подшага.
        """
        result = self._value(
            self._powers(
                _owned(u, "u", shape=(self.size,), nonnegative=True, xp=self._xp),
                _owned(f, "f", shape=(self.size,), nonnegative=True, xp=self._xp),
            )
        )
        if self._xp is not np and not self._xp.isfinite(result).all():
            raise FloatingPointError("The affine substep is not representable")
        return result

    def jvp(self, u, f, du=None, df=None, dQ=None, dalpha=0.0):
        """Применить производную аффинного подшага.

        Parameters
        ----------
        u, f : array_like, shape (N,)
            Неотрицательные точка и нагрузка.
        du, df : array_like, shape (N,), or None, optional
            Направления состояния и нагрузки; None означает нуль.
        dQ : sparse matrix or None, optional
            Вариация Q при фиксированном rate.
        dalpha : float, optional
            Вариация коэффициента роста.

        Returns
        -------
        tangent : ndarray, shape (N,)
            Координатная производная результата по всем направлениям.
        """
        xp = self._xp
        u, f = (
            _owned(u, "u", shape=(self.size,), nonnegative=True, xp=xp),
            _owned(f, "f", shape=(self.size,), nonnegative=True, xp=xp),
        )
        du = xp.zeros(self.size) if du is None else _owned(du, "du", shape=(self.size,), xp=xp)
        df = xp.zeros(self.size) if df is None else _owned(df, "df", shape=(self.size,), xp=xp)
        result = self._tangent(
            self._powers(u, f),
            du,
            df,
            None if dQ is None else _matrix(dQ, "dQ", self.size, xp=xp) / self.rate,
            _scalar(dalpha, "dalpha"),
        )
        if not xp.isfinite(result).all():
            raise FloatingPointError("The affine tangent is not representable")
        return result

    def vjp(self, u, f, cotangent, operator_partials=None):
        """Перенести ковектор аффинного результата к входам и параметрам.

        Parameters
        ----------
        u, f : array_like, shape (N,)
            Неотрицательные точка и нагрузка.
        cotangent : array_like, shape (N,)
            Выходной координатный ковектор.
        operator_partials : Mapping or None, optional
            Именованные разреженные матрицы dQ/dtheta в данной точке.

        Returns
        -------
        covectors : dict
            u и f — массивы (N,), alpha — число, operator — именованные
            свёртки с dQ/dtheta. Используется евклидово спаривание координат.
        """
        xp = self._xp
        u, f = (
            _owned(u, "u", shape=(self.size,), nonnegative=True, xp=xp),
            _owned(f, "f", shape=(self.size,), nonnegative=True, xp=xp),
        )
        partials = {
            name: _matrix(value, name, self.size, xp=xp) / self.rate
            for name, value in (operator_partials or {}).items()
        }
        pulled = self._pullback(
            self._powers(u, f),
            _owned(cotangent, "cotangent", shape=(self.size,), xp=xp),
            partials,
        )
        scalars = xp.stack(
            [
                xp.asarray(pulled["alpha"]),
                *(xp.asarray(value) for value in pulled["operator"].values()),
            ]
        )
        if not xp.isfinite(xp.concatenate((pulled["u"], pulled["f"], scalars))).all():
            raise FloatingPointError("The affine pullback is not representable")
        pulled["alpha"] = float(pulled["alpha"])
        pulled["operator"] = {key: float(value) for key, value in pulled["operator"].items()}
        return pulled

    def jvp_many(
        self,
        u,
        f,
        du,
        df=None,
        *,
        operator_partials=None,
        operator_directions=None,
        dalpha=None,
    ):
        """Применить производную аффинного подшага к нескольким направлениям.

        Parameters
        ----------
        u, f : array_like, shape (N,)
            Неотрицательные исходные состояние и распределённая нагрузка.
        du, df : array_like, shape (N, K)
            Направления по столбцам. du обязателен и задаёт K; df=None
            означает нулевые направления нагрузки.
        operator_partials : Mapping, optional
            Именованные sparse-матрицы частных производных Q.
        operator_directions : array_like, shape (len(operator_partials), K)
            Коэффициенты линейных комбинаций частных производных в каждом
            направлении; порядок строк совпадает с порядком Mapping.
        dalpha : array_like, shape (K,), optional
            Направления коэффициента линейного роста; None означает нуль.

        Returns
        -------
        tangent : ndarray, shape (N, K)
            Те же производные, что отдельные JVP при фиксированных rate
            и degree; изменение весов по alpha учитывается. Базовые
            степени P общие для всех столбцов.
        """
        xp = self._xp
        u, f = _owned(u, "u", shape=(self.size,), nonnegative=True, xp=xp), _owned(
            f, "f", shape=(self.size,), nonnegative=True, xp=xp
        )
        du = _owned(du, "du", xp=xp)
        if du.ndim != 2 or du.shape[0] != self.size or du.shape[1] < 1:
            raise ValueError("du must have shape (N,K) with K>=1")
        count = du.shape[1]
        df = xp.zeros_like(du) if df is None else _owned(df, "df", shape=du.shape, xp=xp)
        dalpha = (
            xp.zeros(count) if dalpha is None else _owned(dalpha, "dalpha", shape=(count,), xp=xp)
        )
        partials, directions = _operator_directions(
            operator_partials, operator_directions, count, self.size, self.rate, xp=xp
        )
        result = self._tangent_many(self._powers(u, f), du, df, partials, directions, dalpha)
        if not xp.isfinite(result).all():
            raise FloatingPointError("The affine tangent is not representable")
        return result


@dataclass(frozen=True, init=False)
class StrangTrajectory:
    """Траектория и производные при фиксированных аффинном подшаге и нагрузке.

    Attributes
    ----------
    states : numpy.ndarray or cupy.ndarray, shape (nt+1, N)
        Состояния, включая начальное. NumPy возвращает заимствованный массив
        с выключенным WRITEABLE; потребитель сохраняет этот режим чтения.
        CuPy возвращает независимую копию на выбранном устройстве.
    polynomial_error_bound : float
        Накопленная оценка только усечения полиномов в бесконечной норме,
        в единице состояния.
        Не включает округление, расщепление, усреднение q или пространственную ошибку.
    affine : PositiveAffineAction
        Неизменяемый аффинный подшаг этой траектории.
    beta : float
        Фиксированный коэффициент квадратичного стока.
    """

    _states: object = field(repr=False)
    affine: PositiveAffineAction
    beta: float
    polynomial_error_bound: float
    _forcing: object = field(repr=False)
    _load: object = field(repr=False)

    def __init__(self, states, affine, beta, polynomial_error_bound, _forcing, _load):
        xp = affine.array_module
        for name, value in dict(
            _states=_owned(states, "states", xp=xp),
            affine=affine,
            beta=beta,
            polynomial_error_bound=polynomial_error_bound,
            _forcing=_forcing,
            _load=_load,
        ).items():
            object.__setattr__(self, name, value)

    @property
    def states(self):
        """Состояния: заимствованный readonly-массив CPU или независимая копия GPU."""
        if self.affine.array_module is np:
            # NumPy deepcopy/pickle may restore an owned array as writable.
            return _readonly(self._states, np)
        return self._states.copy()

    def _frame(self, n, direction=None):
        values = self._forcing if direction is None else direction
        return values[n] if self._load is None else self._load * values[n]

    def jvp(self, d_initial=None, d_forcing=None, dQ=None, dalpha=0.0, dbeta=0.0):
        """Применить производную полной композиции.

        Parameters
        ----------
        d_initial : array_like, shape (N,), or None, optional
            Начальное направление; по умолчанию нулевое.
        d_forcing : array_like or None, optional
            Направление той же формы, что исходная forcing.
        dQ : sparse matrix or None, optional
            Вариация генератора при фиксированных rate и degree.
        dalpha, dbeta : float, optional
            Направления коэффициентов роста и квадратичного стока.

        Returns
        -------
        tangent : ndarray, shape (nt+1, N)
            Касательная траектория. Пространственный load фиксирован.
        """
        xp = self.affine.array_module
        initial = (
            xp.zeros(self.affine.size)
            if d_initial is None
            else _owned(d_initial, "d_initial", shape=(self.affine.size,), xp=xp)
        )
        forcing = (
            None
            if d_forcing is None
            else _owned(d_forcing, "d_forcing", shape=self._forcing.shape, xp=xp)[..., None]
        )
        partials = None if dQ is None else {"direction": dQ}
        directions = None if dQ is None else xp.ones((1, 1))
        return xp.stack(
            [
                value[:, 0]
                for value in self.iter_jvp_many(
                    initial[:, None],
                    forcing,
                    operator_partials=partials,
                    operator_directions=directions,
                    dalpha=[_scalar(dalpha, "dalpha")],
                    dbeta=[_scalar(dbeta, "dbeta")],
                )
            ]
        )

    def iter_jvp_many(
        self,
        d_initial,
        d_forcing=None,
        *,
        operator_partials=None,
        operator_directions=None,
        dalpha=None,
        dbeta=None,
    ):
        """Выдавать матричные чувствительности по времени одним проходом.

        Parameters
        ----------
        d_initial : array_like, shape (N, K)
            Начальные направления; для фиксированного начального поля —
            нулевая матрица. Число K>=1 определяется её столбцами.
        d_forcing : array_like, optional
            Направления формы (*forcing.shape, K). При скалярной нагрузке
            с фиксированным load это (nt,K), при распределённой — (nt,N,K).
        operator_partials : Mapping, optional
            Частные производные Q, общие для всех временных шагов.
        operator_directions : array_like, shape (len(operator_partials), K)
            Коэффициенты операторных направлений в порядке Mapping.
        dalpha, dbeta : array_like, shape (K,), optional
            Направления коэффициентов; None означает нули.

        Yields
        ------
        sensitivities : numpy.ndarray or cupy.ndarray, shape (N, K)
            Собственные матрицы для начального и последующих временных узлов.
            NumPy запрещает запись; CuPy выдаёт независимые копии. Если
            потребитель не хранит прежние матрицы, выходная память не растёт
            как nt*N*K. Исходная траектория и входные направления остаются
            в памяти; распределённый d_forcing сам имеет форму (nt,N,K).
            На GPU накопленная проверка конечности завершается при полном
            исчерпании итератора. При beta=0 и dbeta=0 реакционные
            подшаги в производной тождественны. Если транспорт и линейная
            реакция также фиксированы, исходные степени полинома не нужны.
        """
        affine, half = self.affine, self.affine.step / 2
        xp = affine.array_module
        current = _owned(d_initial, "d_initial", xp=xp)
        if current.ndim != 2 or current.shape[0] != affine.size or current.shape[1] < 1:
            raise ValueError("d_initial must have shape (N,K) with K>=1")
        count = current.shape[1]
        forcing = (
            None
            if d_forcing is None
            else _owned(d_forcing, "d_forcing", shape=(*self._forcing.shape, count), xp=xp)
        )
        dalpha = (
            xp.zeros(count) if dalpha is None else _owned(dalpha, "dalpha", shape=(count,), xp=xp)
        )
        dbeta = xp.zeros(count) if dbeta is None else _owned(dbeta, "dbeta", shape=(count,), xp=xp)
        partials, directions = _operator_directions(
            operator_partials,
            operator_directions,
            count,
            affine.size,
            affine.rate,
            xp=xp,
        )
        active = xp.flatnonzero(xp.any(directions, axis=0))
        tangent_plan = (
            bool(xp.any(dalpha)),
            slice(None) if active.size == count else active,
        )
        linear_reaction = self.beta == 0.0 and not bool(xp.any(dbeta))
        needs_powers = not linear_reaction or tangent_plan[0] or bool(partials)
        valid = xp.asarray(True)
        yield current if xp is np else current.copy()
        for n in range(len(self._states) - 1):
            state = self._states[n]
            if linear_reaction:
                first, first_direction = state, current
            else:
                first = _decay(state, self.beta, half, xp=xp)
                du, db = _decay_partials(state, first, half, xp=xp)
                first_direction = du[:, None] * current + xp.multiply.outer(db, dbeta)
            powers = affine._powers(first, self._frame(n)) if needs_powers else None
            if forcing is None:
                forcing_direction = xp.zeros_like(current)
            else:
                forcing_direction = (
                    forcing[n] if self._load is None else xp.multiply.outer(self._load, forcing[n])
                )
            tangent = affine._tangent_many(
                powers,
                first_direction,
                forcing_direction,
                partials,
                directions,
                dalpha,
                plan=tangent_plan,
            )
            if linear_reaction:
                current = tangent
            else:
                middle = affine._value(powers)
                du, db = _decay_partials(middle, self._states[n + 1], half, xp=xp)
                current = du[:, None] * tangent + xp.multiply.outer(db, dbeta)
            if xp is np:
                if not np.isfinite(current).all():
                    raise FloatingPointError("The batched tangent is not representable")
                current.flags.writeable = False
                yield current
            else:
                valid &= xp.isfinite(current).all()
                yield current.copy()
        if xp is not np and not bool(valid):
            raise FloatingPointError("The batched tangent is not representable")

    def jvp_many(
        self,
        d_initial,
        d_forcing=None,
        *,
        operator_partials=None,
        operator_directions=None,
        dalpha=None,
        dbeta=None,
    ):
        """Собрать iter_jvp_many в массив формы (nt+1,N,K).

        Parameters
        ----------
        d_initial, d_forcing, operator_partials, operator_directions, dalpha, dbeta
            Те же аргументы и единицы, что у iter_jvp_many.

        Returns
        -------
        states : ndarray, shape (nt+1, N, K)
            Все чувствительности. Для наблюдений без хранения этого
            тензора следует непосредственно использовать iterator.
        """
        return self.affine.array_module.stack(
            list(
                self.iter_jvp_many(
                    d_initial,
                    d_forcing,
                    operator_partials=operator_partials,
                    operator_directions=operator_directions,
                    dalpha=dalpha,
                    dbeta=dbeta,
                )
            )
        )

    def vjp(self, cotangent, operator_partials=None):
        """Перенести ковектор всей траектории к её входам.

        Parameters
        ----------
        cotangent : array_like, shape (nt+1, N)
            Координатный ковектор всех состояний, включая начальное.
        operator_partials : Mapping or None, optional
            Именованные разреженные матрицы dQ/dtheta в исходной точке.

        Returns
        -------
        covectors : dict
            initial — массив (N,), forcing — формы исходной нагрузки,
            operator — свёртки с dQ/dtheta, alpha и beta — числа.
            Пространственный load фиксирован; спаривание евклидово.
        """
        affine, half = self.affine, self.affine.step / 2
        xp = affine.array_module
        cotangent = _owned(cotangent, "cotangent", shape=self._states.shape, xp=xp)
        partials = {
            name: _matrix(value, name, affine.size, xp=xp) / affine.rate
            for name, value in (operator_partials or {}).items()
        }
        operator_bar = {name: 0.0 for name in partials}
        forcing_bar = xp.zeros_like(self._forcing)
        state_bar, alpha_bar, beta_bar = cotangent[-1].copy(), 0.0, 0.0
        for n in range(len(self._states) - 2, -1, -1):
            first = _decay(self._states[n], self.beta, half, xp=xp)
            powers = affine._powers(first, self._frame(n))
            middle = affine._value(powers)
            du, db = _decay_partials(middle, self._states[n + 1], half, xp=xp)
            contribution = state_bar @ db
            beta_bar += float(contribution) if xp is np else contribution
            pulled = affine._pullback(powers, state_bar * du, partials)
            alpha_bar += pulled["alpha"]
            for name in operator_bar:
                operator_bar[name] += pulled["operator"][name]
            forcing_bar[n] = pulled["f"] if self._load is None else self._load @ pulled["f"]
            du, db = _decay_partials(self._states[n], first, half, xp=xp)
            contribution = pulled["u"] @ db
            beta_bar += float(contribution) if xp is np else contribution
            state_bar = cotangent[n] + pulled["u"] * du
        scalars = xp.stack(
            [
                xp.asarray(alpha_bar),
                xp.asarray(beta_bar),
                *(xp.asarray(value) for value in operator_bar.values()),
            ]
        )
        if not xp.isfinite(xp.concatenate((state_bar, forcing_bar.ravel(), scalars))).all():
            raise FloatingPointError("The trajectory pullback is not representable")
        alpha_bar, beta_bar = float(alpha_bar), float(beta_bar)
        operator_bar = {name: float(value) for name, value in operator_bar.items()}
        return {
            "initial": state_bar,
            "forcing": forcing_bar,
            "operator": operator_bar,
            "alpha": alpha_bar,
            "beta": beta_bar,
        }

    def initial_increment(self, direction):
        r"""Propagate a finite initial-state increment without subtracting trajectories.

        Parameters
        ----------
        direction : array_like, shape (N,)
            Finite signed initial increment. The shifted initial state must
            remain finite and nonnegative. It is checked without clipping.

        Returns
        -------
        increments : numpy.ndarray or cupy.ndarray, shape (nt+1, N)
            Independently owned increments on the trajectory's backend,
            including the supplied increment at layer zero.

        Notes
        -----
        Forcing, load, Q, alpha, beta and the complete polynomial plan remain
        fixed. At a reaction half-step the algebraic identity
        ``R(u+d)-R(u)=d/((1+k*u)*(1+k*(u+d)))`` is evaluated directly.
        The affine increment uses the existing linear state action; forcing
        cancels. Increments are never obtained by subtracting two states.

        One shifted positive state is advanced as a denominator anchor, with
        one additional affine forward action per time step. It avoids a
        possibly negative rounded reconstruction of u+d near d=-u and does
        not replace the separately propagated increment. The second full
        trajectory is not stored. This is a floating-point evaluation of an
        algebraic increment identity, not an interval certificate or the
        literal difference of two rounded forward solves. In particular a
        sub-ulp increment can be nonzero when that rounded difference is zero.
        """
        affine, half = self.affine, self.affine.step / 2
        xp = affine.array_module
        increment = _owned(direction, "direction", shape=(affine.size,), xp=xp)
        with np.errstate(over="ignore", invalid="ignore"):
            shifted = _owned(
                self._states[0] + increment,
                "shifted initial state",
                shape=(affine.size,),
                nonnegative=True,
                xp=xp,
            )
        result = xp.empty_like(self._states)
        result[0] = increment
        if not bool(xp.any(increment)):
            return xp.zeros_like(self._states)
        zero = xp.zeros((affine.size, 1))
        operator_directions = xp.empty((0, 1))
        alpha_direction = xp.zeros(1)
        valid = xp.asarray(True)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for n in range(len(self._states) - 1):
                state = self._states[n]
                first = _decay(state, self.beta, half, xp=xp)
                shifted_first = _decay(shifted, self.beta, half, xp=xp)
                first_increment = _decay_increment(
                    state, shifted, increment, self.beta, half, xp=xp
                )
                frame = self._frame(n)
                powers = affine._powers(first, frame)
                middle = affine._value(powers)
                shifted_middle = affine._value(affine._powers(shifted_first, frame))
                middle_increment = affine._tangent_many(
                    powers,
                    first_increment[:, None],
                    zero,
                    (),
                    operator_directions,
                    alpha_direction,
                    plan=(False, slice(0, 0)),
                )[:, 0]
                increment = _decay_increment(
                    middle, shifted_middle, middle_increment, self.beta, half, xp=xp
                )
                shifted = _decay(shifted_middle, self.beta, half, xp=xp)
                result[n + 1] = increment
                valid &= xp.isfinite(increment).all() & xp.isfinite(shifted).all()
        if not bool(valid):
            raise FloatingPointError("The initial-state increment is not representable")
        return result

    def initial_hvp(self, cotangent, direction):
        r"""Differentiate the initial-state pullback in a fixed initial direction.

        Parameters
        ----------
        cotangent : array_like, shape (nt+1, N)
            Fixed coordinate covector of all stored states, including layer 0.
            Either sign is permitted.
        direction : array_like, shape (N,)
            Initial-state direction. Either sign is permitted, including at
            zero components of the nonnegative base state.

        Returns
        -------
        product : numpy.ndarray or cupy.ndarray, shape (N,)
            The directional derivative of ``vjp(cotangent)['initial']``.
            The backend and coordinate pairing are those of this trajectory.
            No dense state Hessian is formed.

        Notes
        -----
        The scalar objective is the fixed linear pairing of cotangent with
        all trajectory states. Forcing, load, Q, alpha, beta, rate, degree and
        polynomial weights stay fixed. This differentiates the implemented
        discrete Strang composition, not the continuous PDE solution.

        For each reaction half-step, ``R(u)=u/(1+k*u)``,
        ``R'(u)=(1+k*u)^(-2)`` and ``R''(u)=-2*k*(1+k*u)^(-3)``.
        Tangent-over-reverse adds ``R''(u)*du*state_bar`` to the tangent
        pullback. The affine state substep has zero second derivative.
        The initial cotangent contributes a linear term only. When beta=0
        the initial-state Hessian is exactly zero.
        Reaction curvature is evaluated as a weighted product, with a
        logarithmic expression when direct intermediate products are outside
        the normal floating-point range. No beta bound or curvature clipping
        is introduced; zero direction or a purely initial pairing returns
        exact zero before propagating derivatives.
        """
        if direction is None:
            raise ValueError("direction must be a finite initial-state vector")
        return self._state_forcing_hvp(
            cotangent, d_initial=direction, d_forcing=None, collect_forcing=False
        )["initial"]

    def state_forcing_hvp(self, cotangent, *, d_initial=None, d_forcing=None):
        r"""Differentiate the state/forcing pullback in a joint control direction.

        Parameters
        ----------
        cotangent : array_like, shape (nt+1, N)
            Fixed coordinate covector of every trajectory state, including
            the initial layer. Either sign is permitted.
        d_initial : array_like, shape (N,), optional
            Initial-state direction; None means zero.
        d_forcing : array_like, optional
            Direction with exactly the original forcing shape: (nt,N) for
            distributed forcing, or (nt,) for intensity with fixed load.
            None means zero. Signed directions are permitted for both inputs.

        Returns
        -------
        products : dict
            ``initial`` has shape (N,), ``forcing`` has the original forcing
            shape. Both are independently owned arrays on this trajectory's
            backend. They are derivatives of the corresponding two outputs
            of vjp(cotangent), with cotangent held fixed.

        Notes
        -----
        Q, alpha, beta, load, rate, degree and polynomial weights remain
        fixed. This method differentiates the discrete composition only with
        respect to initial state and forcing, not its parameter pullbacks.
        For an affine step L*a+K*f, the differentiated forcing pullback is
        K.T@eta_b, additionally paired with load for scalar intensities.
        All reaction curvature uses the same weighted evaluation as
        initial_hvp. Beta=0 makes every state/forcing Hessian block zero.
        A fixed cotangent gives a weighted output Hessian; the Gauss-Newton
        term of a least-squares objective must be added separately.
        """
        return self._state_forcing_hvp(
            cotangent, d_initial=d_initial, d_forcing=d_forcing, collect_forcing=True
        )

    def _state_forcing_hvp(self, cotangent, *, d_initial, d_forcing, collect_forcing):
        return self._hvp_core(
            cotangent,
            {},
            d_initial=d_initial,
            d_forcing=d_forcing,
            dQ=None,
            dalpha=0.0,
            dbeta=0.0,
            collect_forcing=collect_forcing,
            collect_parameters=False,
        )

    def hvp(
        self,
        cotangent,
        operator_partials=None,
        *,
        d_initial=None,
        d_forcing=None,
        dQ=None,
        dalpha=0.0,
        dbeta=0.0,
    ):
        r"""Differentiate the complete trajectory pullback in one control direction.

        Parameters
        ----------
        cotangent : array_like, shape (nt+1,N)
            Fixed coordinate covector of all trajectory layers.
        operator_partials : Mapping or None, optional
            Fixed physical partial matrices dQ/dtheta, as in vjp. Their
            derivatives are zero; a nonlinear generator parametrization
            requires its own second chain rule in the caller.
        d_initial, d_forcing : array_like or None, optional
            Signed directions of the initial vector and original forcing.
        dQ : sparse matrix or array_like or None, optional
            Signed generator direction; rate and degree remain fixed.
        dalpha, dbeta : float, optional
            Physical directions of growth and quadratic reaction.

        Returns
        -------
        products : dict
            Directional derivatives of all vjp outputs: initial, forcing,
            operator, alpha and beta. Arrays retain the original backend
            and control shapes; scalar contractions are Python floats.

        Notes
        -----
        This is the weighted Hessian of the implemented finite polynomial
        and Strang composition. The covector is not differentiated. The
        Gauss-Newton term of a least-squares objective is added separately.
        Normalization derivatives of source weights and mixed beta terms
        are included. Beta=0 does not eliminate mixed parameter curvature.
        No dense state Hessian or matrix-valued Q cotangent is constructed.
        """
        return self._hvp_core(
            cotangent,
            operator_partials or {},
            d_initial=d_initial,
            d_forcing=d_forcing,
            dQ=dQ,
            dalpha=dalpha,
            dbeta=dbeta,
            collect_forcing=True,
            collect_parameters=True,
        )

    def _hvp_core(
        self,
        cotangent,
        operator_partials,
        *,
        d_initial,
        d_forcing,
        dQ,
        dalpha,
        dbeta,
        collect_forcing,
        collect_parameters,
    ):
        affine, half = self.affine, self.affine.step / 2
        xp = affine.array_module
        cotangent = _owned(cotangent, "cotangent", shape=self._states.shape, xp=xp)
        dalpha, dbeta = _scalar(dalpha, "dalpha"), _scalar(dbeta, "dbeta")
        generator_direction = None if dQ is None else _matrix(dQ, "dQ", affine.size, xp=xp)
        transition_direction = (
            None if generator_direction is None else generator_direction / affine.rate
        )
        partials = {
            name: _matrix(value, name, affine.size, xp=xp) / affine.rate
            for name, value in operator_partials.items()
        }
        initial_direction = (
            xp.zeros(affine.size)
            if d_initial is None
            else _owned(d_initial, "d_initial", shape=(affine.size,), xp=xp)
        )
        forcing_direction = (
            None
            if d_forcing is None
            else _owned(d_forcing, "d_forcing", shape=self._forcing.shape, xp=xp)
        )
        forcing_bar = xp.zeros_like(self._forcing) if collect_forcing else None
        operator_bar = {name: 0.0 for name in partials}
        alpha_bar, beta_bar = 0.0, 0.0
        if (
            (not collect_parameters and self.beta == 0.0)
            or not bool(xp.any(cotangent[1:]))
            or (
                not bool(xp.any(initial_direction))
                and (forcing_direction is None or not bool(xp.any(forcing_direction)))
                and (generator_direction is None or not bool(xp.any(generator_direction.data)))
                and dalpha == 0.0
                and dbeta == 0.0
            )
        ):
            result = {"initial": xp.zeros(affine.size)}
            if collect_forcing:
                result["forcing"] = forcing_bar
            if collect_parameters:
                result.update(operator=operator_bar, alpha=alpha_bar, beta=beta_bar)
            return result
        tangents = self.jvp(
            d_initial=initial_direction,
            d_forcing=forcing_direction,
            dQ=generator_direction,
            dalpha=dalpha,
            dbeta=dbeta,
        )
        source_second = (
            affine.source_alpha_second_derivative if collect_parameters and dalpha != 0 else None
        )
        state_bar = cotangent[-1].copy()
        tangent_bar = xp.zeros(affine.size)
        zero = xp.zeros(affine.size)
        operator_directions = xp.empty((0, 1))
        alpha_direction = xp.zeros(1)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for n in range(len(self._states) - 2, -1, -1):
                state = self._states[n]
                first = _decay(state, self.beta, half, xp=xp)
                first_ratio = _decay_ratio(state, first, xp=xp)
                first_slope = first_ratio**2
                powers = affine._powers(first, self._frame(n))
                middle = affine._value(powers)
                first_direction = first_slope * tangents[n]
                if dbeta != 0:
                    first_direction += _decay_partials(state, first, half, xp=xp)[1] * dbeta
                load_direction = (
                    zero if forcing_direction is None else self._frame(n, forcing_direction)
                )
                if collect_parameters:
                    power_directions = affine._power_directions(
                        powers, first_direction, load_direction, transition_direction
                    )
                    middle_direction = (
                        affine._growth * (affine._a @ power_directions[0])
                        + affine._b @ power_directions[1]
                    )
                    if dalpha != 0:
                        middle_direction += dalpha * (
                            affine.step * affine._growth * (affine._a @ powers[0])
                            + affine._db @ powers[1]
                        )
                else:
                    middle_direction = affine._tangent_many(
                        powers,
                        first_direction[:, None],
                        load_direction[:, None],
                        (),
                        operator_directions,
                        alpha_direction,
                        plan=(False, slice(0, 0)),
                    )[:, 0]
                middle_bar, middle_tangent_bar, beta_contribution = _decay_pullback_tangent(
                    middle,
                    self._states[n + 1],
                    self.beta,
                    half,
                    middle_direction,
                    state_bar,
                    tangent_bar,
                    dbeta,
                    parameters=collect_parameters,
                    xp=xp,
                )
                beta_bar += beta_contribution
                if collect_parameters:
                    first_bar, tangent_pulled = affine._pullback_directional(
                        powers,
                        power_directions,
                        middle_bar,
                        middle_tangent_bar,
                        partials,
                        transition_direction,
                        dalpha,
                        source_second,
                    )
                    alpha_bar += tangent_pulled["alpha"]
                    for name in operator_bar:
                        operator_bar[name] += tangent_pulled["operator"][name]
                else:
                    first_bar = affine._pullback(powers, middle_bar, {})["u"]
                    tangent_pulled = affine._pullback(powers, middle_tangent_bar, {})
                first_tangent_bar = tangent_pulled["u"]
                if collect_forcing:
                    forcing_bar[n] = (
                        tangent_pulled["f"]
                        if self._load is None
                        else self._load @ tangent_pulled["f"]
                    )
                state_bar, tangent_bar, beta_contribution = _decay_pullback_tangent(
                    state,
                    first,
                    self.beta,
                    half,
                    tangents[n],
                    first_bar,
                    first_tangent_bar,
                    dbeta,
                    parameters=collect_parameters,
                    xp=xp,
                )
                beta_bar += beta_contribution
                state_bar = cotangent[n] + state_bar
        finite = xp.isfinite(tangent_bar).all() & xp.isfinite(state_bar).all()
        if collect_forcing:
            finite &= xp.isfinite(forcing_bar).all()
        if collect_parameters:
            finite &= xp.isfinite(
                xp.stack(
                    [
                        xp.asarray(alpha_bar),
                        xp.asarray(beta_bar),
                        *(xp.asarray(value) for value in operator_bar.values()),
                    ]
                )
            ).all()
        if not bool(finite):
            raise FloatingPointError("The state/forcing Hessian action is not representable")
        result = {"initial": tangent_bar}
        if collect_forcing:
            result["forcing"] = forcing_bar
        if collect_parameters:
            result.update(
                operator={name: float(value) for name, value in operator_bar.items()},
                alpha=float(alpha_bar),
                beta=float(beta_bar),
            )
        return result


def solve_strang(affine, initial, forcing, *, beta=0.0, load=None):
    """Выполнить композицию R(step/2), affine, R(step/2).

    Parameters
    ----------
    affine : PositiveAffineAction
        Фиксированный аффинный подшаг.
    initial : array_like, shape (N,)
        Неотрицательное начальное состояние.
    forcing : array_like, shape (nt, N) or (nt,)
        Конечные неотрицательные средние нагрузки интервалов.
        При load=None имеют форму (nt, N) и единицу состояние/время.
        При заданном load передаются скалярные интенсивности формы (nt,);
        произведение load*forcing[n] имеет единицу состояние/время.
        Неиспользуемого начального элемента нет.
    beta : float, optional
        Конечный неотрицательный коэффициент квадратичного стока,
        в единицах 1/(состояние*время).
    load : array_like, shape (N,), or None, optional
        Фиксированное конечное неотрицательное пространственное распределение;
        на интервале применяется load*forcing[n]. Его единицы согласуются
        с единицами интенсивности; численное преобразование не выполняется.

    Returns
    -------
    trajectory : StrangTrajectory
        Состояния, производные и оценка усечения. Входные массивы копируются.
    """
    if not isinstance(affine, PositiveAffineAction):
        raise TypeError("affine must be PositiveAffineAction")
    xp = affine.array_module
    beta = _scalar(beta, "beta", nonnegative=True)
    initial = _owned(initial, "initial", shape=(affine.size,), nonnegative=True, xp=xp)
    forcing = _owned(forcing, "forcing", nonnegative=True, xp=xp)
    if load is None:
        if forcing.ndim != 2 or forcing.shape[1] != affine.size:
            raise ValueError("Distributed forcing must have shape (nt,N)")
    else:
        load = _owned(load, "load", shape=(affine.size,), nonnegative=True, xp=xp)
        if forcing.ndim != 1:
            raise ValueError("Load intensities must have shape (nt,)")
    states = xp.empty((len(forcing) + 1, affine.size))
    states[0] = initial
    error_bound = 0.0
    for n in range(len(forcing)):
        first = _decay(states[n], beta, affine.step / 2, xp=xp)
        frame = forcing[n] if load is None else load * forcing[n]
        states[n + 1] = _decay(
            affine._value(affine._powers(first, frame)), beta, affine.step / 2, xp=xp
        )
        local = (
            2
            * affine.transport_tail
            * (
                affine._growth * xp.linalg.norm(first, np.inf)
                + affine.source_weight_sum * xp.linalg.norm(frame, np.inf)
            )
        )
        error_bound = affine._growth * error_bound + local
    _readonly(states, xp)
    return StrangTrajectory(states, affine, beta, float(error_bound), forcing, load)
