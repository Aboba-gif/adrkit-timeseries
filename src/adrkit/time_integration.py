"""Положительные аффинные подшаги и расщепление квадратичного стока."""

from dataclasses import dataclass, field
from operator import index

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


def _owned(value, name, *, shape=None, nonnegative=False):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or not np.isfinite(raw).all():
        raise ValueError(f"{name} must contain finite real values")
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.array(raw, dtype=float, copy=True)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must remain finite in double precision")
    if shape is not None and values.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if nonnegative and np.any(values < 0):
        raise ValueError(f"{name} must be nonnegative")
    values.flags.writeable = False
    return values


def _matrix(value, name, size=None):
    if (
        np.dtype(value.dtype if sparse.issparse(value) else np.asarray(value).dtype).kind
        not in "iuf"
    ):
        raise ValueError(f"{name} must be real")
    matrix = sparse.csr_matrix(value, dtype=float, copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if (
        matrix.shape[0] != matrix.shape[1]
        or not matrix.shape[0]
        or (size is not None and matrix.shape != (size, size))
        or not np.isfinite(matrix.data).all()
    ):
        raise ValueError(f"{name} must be a finite square matrix of the required size")
    return matrix


def _operator_directions(partials, directions, count, size, rate):
    partials = tuple((partials or {}).values())
    matrices = tuple(_matrix(value, "operator partial", size) / rate for value in partials)
    if directions is None:
        if matrices:
            raise ValueError("Operator partials require explicit operator_directions")
        coefficients = np.zeros((0, count))
    else:
        coefficients = _owned(directions, "operator_directions", shape=(len(matrices), count))
    active = [j for j, matrix in enumerate(matrices) if matrix.nnz and np.any(coefficients[j])]
    return tuple(matrices[j] for j in active), coefficients[active]


def _decay(u, beta, step):
    if beta == 0.0 or step == 0.0:
        return u.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        product = (beta * step) * u
        value = np.asarray(u / (1.0 + product))
    overflow = ~np.isfinite(product) & (u > 0)
    if overflow.any():
        log_u = np.log(u[overflow])
        value[overflow] = np.exp(log_u - np.logaddexp(0.0, np.log(beta) + np.log(step) + log_u))
    value[u == 0.0] = 0.0
    return value


def _decay_partials(u, value, step):
    derivative = np.ones_like(u)
    positive = u > 0.0
    derivative[positive] = (value[positive] / u[positive]) ** 2
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
        values, _scalar(beta, "beta", nonnegative=True), _scalar(step, "step", nonnegative=True)
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


def _weights(alpha, step, rate, degree):
    k = np.arange(degree + 1, dtype=float)
    mean = rate * step
    alpha_step = alpha * step
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        growth = np.exp(alpha_step)
        if alpha < 0.0 and -alpha_step >= 1.0:
            total = -np.expm1(alpha_step) / (-alpha)
            log_total_derivative = np.log(gammainc(2.0, -alpha_step)) - 2 * np.log(-alpha)
        else:
            total = step * exprel(alpha_step)
            log_total_derivative = (
                2 * np.log(step) - np.log(2.0) + np.log(hyp1f1(2.0, 3.0, alpha_step))
            )
        total_derivative = np.exp(log_total_derivative)
    if not np.isfinite([mean, growth, total, total_derivative]).all() or total <= 0:
        raise FloatingPointError("The affine weights are outside the representable range")
    if mean == 0.0:
        transport = np.zeros(degree + 1)
        transport[0] = 1.0
        source, source_derivative = transport * total, transport * total_derivative
        return growth, total, transport, source, source_derivative, 0.0
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
    # Неполная гамма избегает потери exp(-mean) при больших средних.
    if rate > alpha:
        log_loss = (
            np.logaddexp(np.log(rate), np.log(-alpha)) if alpha < 0.0 else np.log(rate - alpha)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            alternative = (
                k * (np.log(rate) - log_loss) - log_loss + np.log(gammainc(k + 1.0, -argument))
            )
            alternative_dw = (
                np.log(k + 1.0)
                + k * (np.log(rate) - log_loss)
                - 2 * log_loss
                + np.log(gammainc(k + 2.0, -argument))
            )
        log_w = np.where(~np.isfinite(log_w) & np.isfinite(alternative), alternative, log_w)
        log_dw = np.where(
            ~np.isfinite(log_dw) & np.isfinite(alternative_dw), alternative_dw, log_dw
        )
    if np.isnan(log_w).any() or np.isposinf(log_w).any() or not np.isfinite(log_w).any():
        raise FloatingPointError("The source weights are not representable")
    normalized = np.exp(log_w - np.max(log_w))
    normalized /= normalized.sum()
    ratios = np.zeros_like(k)
    finite = np.isfinite(log_w) & np.isfinite(log_dw)
    ratios[finite] = np.exp(log_dw[finite] - log_w[finite])
    if np.any((normalized > 0) & ~finite):
        raise FloatingPointError("The source-weight derivative is not representable")
    source = total * normalized
    source_derivative = source * (total_derivative / total + ratios - normalized @ ratios)
    if not np.isfinite(source_derivative).all():
        raise FloatingPointError("The normalized source derivative is not representable")
    return growth, total, transport, source, source_derivative, float(gammainc(degree + 1.0, mean))


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
    _a: np.ndarray = field(repr=False)
    _b: np.ndarray = field(repr=False)
    _db: np.ndarray = field(repr=False)

    def __init__(self, Q, alpha, *, step, rate, degree):
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
            _a=_owned(a, "weights"),
            _b=_owned(b, "weights"),
            _db=_owned(db, "weight derivatives"),
        ).items():
            object.__setattr__(self, name, value)

    @property
    def transport_weights(self):
        """ndarray : Независимые нормированные пуассоновы веса."""
        return _owned(self._a, "weights")

    @property
    def source_weights(self):
        """ndarray : Независимые интегральные источниковые веса."""
        return _owned(self._b, "weights")

    @property
    def source_alpha_derivative(self):
        """ndarray : Производные источниковых весов по alpha."""
        return _owned(self._db, "weight derivatives")

    def _powers(self, u, f):
        up = np.empty((self.degree + 1, self.size))
        fp = np.empty_like(up)
        up[0], fp[0] = u, f
        for k in range(1, self.degree + 1):
            up[k] = self._transition @ up[k - 1]
            fp[k] = self._transition @ fp[k - 1]
        return up, fp

    def _value(self, powers):
        value = self._growth * (self._a @ powers[0]) + self._b @ powers[1]
        if not np.isfinite(value).all():
            raise FloatingPointError("The affine substep is not representable")
        return value

    def _tangent(self, powers, du, df, dP, dalpha):
        partials = () if dP is None else (dP,)
        directions = np.ones((len(partials), 1))
        return self._tangent_many(
            powers, du[:, None], df[:, None], partials, directions, np.array([dalpha])
        )[:, 0]

    def _tangent_many(self, powers, du, df, partials, directions, dalpha):
        up, fp = powers
        result = np.zeros_like(du)
        varying_alpha = np.any(dalpha)
        active = np.flatnonzero(np.any(directions, axis=0))
        selected = slice(None) if active.size == du.shape[1] else active
        operator_coefficients = directions[:, selected]
        for k in range(self.degree + 1):
            result += self._growth * self._a[k] * du + self._b[k] * df
            if varying_alpha:
                result += np.multiply.outer(
                    self.step * self._growth * self._a[k] * up[k] + self._db[k] * fp[k], dalpha
                )
            if k < self.degree:
                du, df = self._transition @ du, self._transition @ df
                if partials:
                    du[:, selected] += (
                        np.column_stack([matrix @ up[k] for matrix in partials])
                        @ operator_coefficients
                    )
                    df[:, selected] += (
                        np.column_stack([matrix @ fp[k] for matrix in partials])
                        @ operator_coefficients
                    )
        return result

    def _pullback(self, powers, cotangent, partials):
        up, fp = powers
        u_bar, f_bar = np.zeros(self.size), np.zeros(self.size)
        operator_bar = {name: 0.0 for name in partials}
        alpha_bar = float(
            self.step * self._growth * (self._a @ (up @ cotangent)) + self._db @ (fp @ cotangent)
        )
        for k in range(self.degree, -1, -1):
            u_bar += self._growth * self._a[k] * cotangent
            f_bar += self._b[k] * cotangent
            if k:
                for name, matrix in partials.items():
                    operator_bar[name] += float(
                        u_bar @ (matrix @ up[k - 1]) + f_bar @ (matrix @ fp[k - 1])
                    )
                u_bar = self._transition.T @ u_bar
                f_bar = self._transition.T @ f_bar
        return {"u": u_bar, "f": f_bar, "operator": operator_bar, "alpha": alpha_bar}

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
        return self._value(
            self._powers(
                _owned(u, "u", shape=(self.size,), nonnegative=True),
                _owned(f, "f", shape=(self.size,), nonnegative=True),
            )
        )

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
        u, f = (
            _owned(u, "u", shape=(self.size,), nonnegative=True),
            _owned(f, "f", shape=(self.size,), nonnegative=True),
        )
        du = np.zeros(self.size) if du is None else _owned(du, "du", shape=(self.size,))
        df = np.zeros(self.size) if df is None else _owned(df, "df", shape=(self.size,))
        return self._tangent(
            self._powers(u, f),
            du,
            df,
            None if dQ is None else _matrix(dQ, "dQ", self.size) / self.rate,
            _scalar(dalpha, "dalpha"),
        )

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
        u, f = (
            _owned(u, "u", shape=(self.size,), nonnegative=True),
            _owned(f, "f", shape=(self.size,), nonnegative=True),
        )
        partials = {
            name: _matrix(value, name, self.size) / self.rate
            for name, value in (operator_partials or {}).items()
        }
        return self._pullback(
            self._powers(u, f), _owned(cotangent, "cotangent", shape=(self.size,)), partials
        )

    def jvp_many(
        self, u, f, du, df=None, *, operator_partials=None, operator_directions=None, dalpha=None
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
            Те же производные, что отдельные JVP при фиксированных rate,
            degree и весах. Базовые степени P общие для всех столбцов.
        """
        u, f = _owned(u, "u", shape=(self.size,), nonnegative=True), _owned(
            f, "f", shape=(self.size,), nonnegative=True
        )
        du = _owned(du, "du")
        if du.ndim != 2 or du.shape[0] != self.size or du.shape[1] < 1:
            raise ValueError("du must have shape (N,K) with K>=1")
        count = du.shape[1]
        df = np.zeros_like(du) if df is None else _owned(df, "df", shape=du.shape)
        dalpha = np.zeros(count) if dalpha is None else _owned(dalpha, "dalpha", shape=(count,))
        partials, directions = _operator_directions(
            operator_partials, operator_directions, count, self.size, self.rate
        )
        return self._tangent_many(self._powers(u, f), du, df, partials, directions, dalpha)


@dataclass(frozen=True)
class StrangTrajectory:
    """Траектория и производные при фиксированных аффинном подшаге и нагрузке.

    Attributes
    ----------
    states : ndarray, shape (nt+1, N)
        Собственные состояния, включая начальное; запись запрещена.
    polynomial_error_bound : float
        Накопленная оценка только усечения полиномов в бесконечной норме,
        в единице состояния.
        Не включает округление, расщепление, усреднение q или пространственную ошибку.
    affine : PositiveAffineAction
        Неизменяемый аффинный подшаг этой траектории.
    beta : float
        Фиксированный коэффициент квадратичного стока.
    """

    states: np.ndarray
    affine: PositiveAffineAction
    beta: float
    polynomial_error_bound: float
    _forcing: np.ndarray = field(repr=False)
    _load: np.ndarray | None = field(repr=False)

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
        initial = (
            np.zeros(self.affine.size)
            if d_initial is None
            else _owned(d_initial, "d_initial", shape=(self.affine.size,))
        )
        forcing = (
            None
            if d_forcing is None
            else _owned(d_forcing, "d_forcing", shape=self._forcing.shape)[..., None]
        )
        partials = None if dQ is None else {"direction": dQ}
        directions = None if dQ is None else np.ones((1, 1))
        return np.stack(
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
        sensitivities : ndarray, shape (N, K)
            Собственные неизменяемые матрицы для начального и последующих
            временных узлов. Если потребитель не хранит прежние матрицы,
            память не растёт как nt*N*K. Прямой шаг не изменяется.
        """
        affine, half = self.affine, self.affine.step / 2
        current = _owned(d_initial, "d_initial")
        if current.ndim != 2 or current.shape[0] != affine.size or current.shape[1] < 1:
            raise ValueError("d_initial must have shape (N,K) with K>=1")
        count = current.shape[1]
        forcing = (
            None
            if d_forcing is None
            else _owned(d_forcing, "d_forcing", shape=(*self._forcing.shape, count))
        )
        dalpha = np.zeros(count) if dalpha is None else _owned(dalpha, "dalpha", shape=(count,))
        dbeta = np.zeros(count) if dbeta is None else _owned(dbeta, "dbeta", shape=(count,))
        partials, directions = _operator_directions(
            operator_partials, operator_directions, count, affine.size, affine.rate
        )
        yield current
        for n in range(len(self.states) - 1):
            first = _decay(self.states[n], self.beta, half)
            du, db = _decay_partials(self.states[n], first, half)
            powers = affine._powers(first, self._frame(n))
            middle = affine._value(powers)
            first_direction = du[:, None] * current + np.multiply.outer(db, dbeta)
            if forcing is None:
                forcing_direction = np.zeros_like(current)
            else:
                forcing_direction = (
                    forcing[n] if self._load is None else np.multiply.outer(self._load, forcing[n])
                )
            tangent = affine._tangent_many(
                powers, first_direction, forcing_direction, partials, directions, dalpha
            )
            du, db = _decay_partials(middle, self.states[n + 1], half)
            current = du[:, None] * tangent + np.multiply.outer(db, dbeta)
            if not np.isfinite(current).all():
                raise FloatingPointError("The batched tangent is not representable")
            current.flags.writeable = False
            yield current

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
        return np.stack(
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
        cotangent = _owned(cotangent, "cotangent", shape=self.states.shape)
        partials = {
            name: _matrix(value, name, affine.size) / affine.rate
            for name, value in (operator_partials or {}).items()
        }
        operator_bar = {name: 0.0 for name in partials}
        forcing_bar = np.zeros_like(self._forcing)
        state_bar, alpha_bar, beta_bar = cotangent[-1].copy(), 0.0, 0.0
        for n in range(len(self.states) - 2, -1, -1):
            first = _decay(self.states[n], self.beta, half)
            powers = affine._powers(first, self._frame(n))
            middle = affine._value(powers)
            du, db = _decay_partials(middle, self.states[n + 1], half)
            beta_bar += float(state_bar @ db)
            pulled = affine._pullback(powers, state_bar * du, partials)
            alpha_bar += pulled["alpha"]
            for name in operator_bar:
                operator_bar[name] += pulled["operator"][name]
            forcing_bar[n] = pulled["f"] if self._load is None else self._load @ pulled["f"]
            du, db = _decay_partials(self.states[n], first, half)
            beta_bar += float(pulled["u"] @ db)
            state_bar = cotangent[n] + pulled["u"] * du
        return {
            "initial": state_bar,
            "forcing": forcing_bar,
            "operator": operator_bar,
            "alpha": alpha_bar,
            "beta": beta_bar,
        }


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
    beta = _scalar(beta, "beta", nonnegative=True)
    initial = _owned(initial, "initial", shape=(affine.size,), nonnegative=True)
    forcing = _owned(forcing, "forcing", nonnegative=True)
    if load is None:
        if forcing.ndim != 2 or forcing.shape[1] != affine.size:
            raise ValueError("Distributed forcing must have shape (nt,N)")
    else:
        load = _owned(load, "load", shape=(affine.size,), nonnegative=True)
        if forcing.ndim != 1:
            raise ValueError("Load intensities must have shape (nt,)")
    states = np.empty((len(forcing) + 1, affine.size))
    states[0] = initial
    error_bound = 0.0
    for n in range(len(forcing)):
        first = _decay(states[n], beta, affine.step / 2)
        frame = forcing[n] if load is None else load * forcing[n]
        states[n + 1] = _decay(affine._value(affine._powers(first, frame)), beta, affine.step / 2)
        local = (
            2
            * affine.transport_tail
            * (
                affine._growth * np.linalg.norm(first, np.inf)
                + affine.source_weight_sum * np.linalg.norm(frame, np.inf)
            )
        )
        error_bound = affine._growth * error_bound + local
    states.flags.writeable = False
    return StrangTrajectory(states, affine, beta, float(error_bound), forcing, load)
