"""Целевые критерии аффинного восстановления и выбор конечной сети.

Матрицы влияния строятся вызывающей процедурой. Этот модуль не получает
истинных параметров, городских данных или модели переноса. Численные оценки
максимума в FP64 не являются интервальными сертификатами.
"""

from dataclasses import dataclass
from operator import index

import numpy as np
from scipy.linalg import solve_triangular

from ._numeric import real_array, real_scalar, symmetric_psd_matrix
from .linalg import _spd_factor, _symmetric_matrix, gram_root, whiten


def _matrix(value, name):
    result = real_array(value, name=name)
    if result.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    return result


def _owned(value):
    result = np.array(value, dtype=float, copy=True)
    result.flags.writeable = False
    return result


def _norm(value):
    scale = float(np.max(np.abs(value), initial=0.0))
    if scale == 0.0 or not np.isfinite(scale):
        return scale
    return scale * float(np.sqrt(np.sum((value / scale) ** 2)))


def _square_norm(value, name):
    norm = _norm(value)
    squared = real_scalar(norm * norm, name=name)
    if norm > 0.0 and squared == 0.0:
        raise ValueError(f"{name} underflowed; rescale the target units")
    return squared


@dataclass(frozen=True, init=False)
class CovarianceFactor:
    r"""Явный прямоугольный корень положительно полуопределённой ковариации.

    Parameters
    ----------
    factor : array_like, shape (n, r)
        Корень R в равенстве C=R R.T. Допустим r=0.

    Notes
    -----
    Создаётся собственный массив с запрещённой записью. Представление
    фактором позволяет задавать вырожденную ковариацию без исправления
    отрицательных вычисленных собственных значений.
    """

    factor: np.ndarray

    def __init__(self, factor):
        object.__setattr__(self, "factor", _owned(_matrix(factor, "covariance factor")))


@dataclass(frozen=True)
class InnovationBudget:
    """Разделение заданного общего ковариационного бюджета инноваций.

    physical_scale — omega, множитель физической ковариации; сами физические
    отклонения в данных И цели умножаются на sqrt(omega). max_ratio — Lambda,
    максимальная относительная физическая дисперсия до масштабирования.
    residual_factor задаёт C_nu=F@F.T; F не обязан быть симметричным.
    """

    physical_scale: float
    max_ratio: float
    fraction: float
    residual_factor: CovarianceFactor


def innovation_covariance_budget(physical_innovation_factor, innovation_covariance, *, fraction):
    r"""Распределить положительный бюджет инноваций между физикой и остатком.

    Parameters
    ----------
    physical_innovation_factor : array_like, shape (n, r)
        Уже вычисленная вызывающим кодом матрица P=L@O@R, C_xi=R@R.T.
        n>0, r>=0. Нулевая P, включая r=0, допустима.
    innovation_covariance : array_like, shape (n,n) or (n,)
        Строго положительно определённая общая ковариация C_I либо её
        строго положительная диагональ. Используется существующий контракт
        SPD-разложения, без сдвига диагонали и исправления собственных чисел.
    fraction : float
        Явная доля gamma, 0<gamma<1, не значение по умолчанию.

    Returns
    -------
    InnovationBudget
        Lambda=||A^-1@P||_2**2, A@A.T=C_I;
        omega=1 при P=0, иначе min(1,gamma/Lambda).
        residual_factor.factor=F, F@F.T=C_I-omega*P@P.T.

    Notes
    -----
    В точной арифметике C_nu >= (1-gamma)*C_I > 0. Тонкое SVD отбелённой
    P даёт Q,s; используется F=A-(A@Q*d)@Q.T,
    d=(omega*s**2)/(1+sqrt(1-omega*s**2)). Формула не вычитает близкие
    корни и не требует спектрального усечения. Возвращается плотный фактор
    формы (n,n), а не симметричный квадратный корень.

    Вызывающий код обязан умножить И физическую карту данных O, И карту
    истинной цели T на sqrt(omega), сохранить их перекрёстные моменты и
    согласовать средние. Нулевая P не означает нулевую физическую цель.
    Функция не оценивает L/O/T/R, закон ошибок или физическую определимость;
    это только распределение объявленного ковариационного бюджета.
    Проверки FP64 не являются интервальным сертификатом неравенства.
    """
    if any(
        np.ma.isMaskedArray(value)
        for value in (physical_innovation_factor, innovation_covariance, fraction)
    ):
        raise ValueError("Innovation budget inputs cannot be masked arrays")
    physical = _matrix(physical_innovation_factor, "physical innovation factor")
    if not physical.shape[0]:
        raise ValueError("Innovation budget requires n>0")
    gamma = real_scalar(fraction, name="innovation budget fraction")
    if not 0 < gamma < 1:
        raise ValueError("Innovation budget fraction must satisfy 0<fraction<1")
    factor = _spd_factor(innovation_covariance, physical.shape[0], "innovation covariance")
    if not np.any(physical):
        matrix = np.diag(factor) if factor.ndim == 1 else factor
        return InnovationBudget(1.0, 0.0, gamma, CovarianceFactor(matrix))
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
            white = (
                physical / factor[:, None]
                if factor.ndim == 1
                else solve_triangular(factor, physical, lower=True)
            )
            white = real_array(white, name="whitened physical innovation factor")
            directions, singular, _ = np.linalg.svd(white, full_matrices=False)
            ratio = _square_norm(singular[:1], "maximum physical innovation ratio")
            if ratio == 0:
                raise ValueError("A nonzero physical factor lost its whitened spectral scale")
            omega = (
                1.0
                if ratio <= gamma
                else real_scalar(np.divide(gamma, ratio), name="physical covariance scale")
            )
            if omega <= 0:
                raise ValueError("A positive physical covariance scale underflowed")
            allocated = omega * singular**2
            if not np.isfinite(allocated).all() or np.any(allocated < 0) or np.any(allocated >= 1):
                raise ValueError("Allocated physical spectrum must remain in [0,1)")
            correction = allocated / (1.0 + np.sqrt(1.0 - allocated))
            matrix = np.diag(factor) if factor.ndim == 1 else factor
            residual = matrix - ((matrix @ directions) * correction) @ directions.T
            residual = real_array(residual, name="residual innovation covariance factor")
    except FloatingPointError as error:
        raise ValueError(
            "Innovation budget arithmetic lost its scale; rescale the input units"
        ) from error
    return InnovationBudget(omega, ratio, gamma, CovarianceFactor(residual))


def _covariance_factor(value, size, name):
    if isinstance(value, CovarianceFactor):
        if value.factor.shape[0] != size:
            raise ValueError(f"{name} factor has incompatible row count")
        return value.factor
    matrix = real_array(value, name=name)
    if matrix.ndim == 1:
        if matrix.shape != (size,) or np.any(matrix < 0.0):
            raise ValueError(f"{name} requires a nonnegative diagonal of size {size}")
        return np.sqrt(matrix)
    if size == 0:
        if matrix.shape != (0, 0):
            raise ValueError(f"{name} must have shape (0, 0)")
        return matrix.copy()
    symmetric = _symmetric_matrix(matrix, size, name)
    symmetric, _ = symmetric_psd_matrix(symmetric, size, name=name)
    try:
        return gram_root(symmetric, size).T
    except ValueError:
        eigenvalues, vectors = np.linalg.eigh(symmetric)
        if np.any(eigenvalues < 0.0):
            raise ValueError(
                f"{name} has negative computed eigenvalues; "
                "supply an explicit CovarianceFactor for a singular covariance"
            ) from None
        return real_array(vectors * np.sqrt(eigenvalues)[None, :], name=f"{name} factor")


def _right_factor(matrix, factor):
    return matrix * factor[None, :] if factor.ndim == 1 else matrix @ factor


@dataclass(frozen=True)
class JointMomentRisk:
    """Средняя ошибка и два слагаемых её среднего квадрата.

    mean_error — собственный неизменяемый массив. squared_bias равен
    квадрату нормы среднего, variance — следу ковариации ошибки,
    value — их сумме. Это значения в FP64, не интервальные сертификаты.
    """

    mean_error: np.ndarray
    squared_bias: float
    variance: float
    value: float


def joint_moment_risk(K, offset, mean, covariance):
    r"""Вычислить E||offset+K X||² по совместным моментам X.

    Parameters
    ----------
    K : array_like, shape (k, p)
        Карта расширенного случайного входа в целевую ошибку. Целевая
        метрика уже включена в K и offset вызывающей процедурой.
    offset : array_like, shape (k,)
        Явное детерминированное смещение; нуль автоматически не назначается.
    mean : array_like, shape (p,)
        Полный совместный вектор средних. Нулевые средние не предполагаются.
    covariance : array_like or CovarianceFactor
        Совместная ковариация формы (p,p), её неотрицательная диагональ (p,)
        или явный прямоугольный фактор. Все перекрёстные связи находятся
        внутри этой ковариации; независимость компонент не предполагается.

    Returns
    -------
    JointMomentRisk
        mean_error=offset+K@mean, squared_bias=||mean_error||²,
        variance=tr(K@covariance@K.T), value=squared_bias+variance.

    Notes
    -----
    Тождество требует лишь конечных вторых моментов, без нормальности X.
    Например, X может включать исходные данные, нелинейную направленную
    MAD и изменяющуюся физическую цель. Моменты этого совместного вектора
    должен предоставить вызывающий код; функция не выводит их из MAD.
    Риск относится к переданной карте ошибки и не удостоверяет точность
    нелинейной PDE. Ковариация обрабатывается тем же способом, что и
    в affine_goal_risk; отрицательные собственные значения не обрезаются.
    """
    if any(np.ma.isMaskedArray(value) for value in (K, offset, mean, covariance)):
        raise ValueError("Joint moment inputs cannot be masked arrays")
    matrix = _matrix(K, "joint error map")
    offset = real_array(offset, name="joint error offset")
    mean = real_array(mean, name="joint input mean")
    if offset.shape != (matrix.shape[0],) or mean.shape != (matrix.shape[1],):
        raise ValueError("Joint mean and offset must match the error map dimensions")
    factor = _covariance_factor(covariance, matrix.shape[1], "joint covariance")
    try:
        with np.errstate(over="raise", invalid="raise", under="raise"):
            mean_error = real_array(offset + matrix @ mean, name="mean goal error")
            spread = real_array(_right_factor(matrix, factor), name="joint goal spread")
    except FloatingPointError as error:
        raise ValueError(
            "Joint moment arithmetic lost its scale; rescale the input units"
        ) from error
    squared_bias = _square_norm(mean_error, "squared mean goal error")
    variance = _square_norm(spread, "goal error variance")
    value = real_scalar(squared_bias + variance, name="joint moment risk")
    immutable_mean = np.frombuffer(mean_error.tobytes(), dtype=mean_error.dtype).reshape(
        mean_error.shape
    )
    return JointMomentRisk(immutable_mean, squared_bias, variance, value)


@dataclass(frozen=True)
class BallMaximum:
    """Численные оценки максимума квадрата нормы на шаре.

    Attributes
    ----------
    witness : ndarray
        Собственная точка модельного возмущения.
    attained_estimate, dual_upper_estimate : float
        Достигнутое значение и двойственная оценка.
    gap_estimate : float
        Подписанная разность двойственной и достигнутой оценок.
    status : str
        Результат численного решения, не статус научной валидации.
    iterations : int
        Число итераций скалярного поиска.

    Notes
    -----
    В точной арифметике допустимая точка и двойственная формула дают
    нижнюю и верхнюю границы. Здесь SVD и допустимость вычислены в FP64;
    ни интервал, ни размер разрыва не сертифицированы округлением.
    """

    witness: np.ndarray
    attained_estimate: float
    dual_upper_estimate: float
    gap_estimate: float
    status: str
    iterations: int

    @property
    def certified(self):
        return False

    @property
    def converged_estimate(self):
        return self.status in {
            "constant_estimate",
            "hard_case_estimate",
            "converged_estimate",
        }


def robust_ball_maximum(a, D, radius, *, rtol=1e-12, maxiter=256):
    r"""Оценить max ||a+D z||^2 при ||z||<=radius.

    Parameters
    ----------
    a : array_like, shape (k,)
        Центральное смещение целевой величины.
    D : array_like, shape (k, r)
        Влияние модельного возмущения.
    radius : float
        Конечный неотрицательный радиус.
    rtol : float, optional
        Относительный допуск разрыва в масштабированной задаче.
    maxiter : int, optional
        Положительный предел скалярных итераций.

    Returns
    -------
    result : BallMaximum
        Точка, две численные оценки, подписанный разрыв и статус.

    Notes
    -----
    Используется SVD D без формирования D.T@D. Скалярный параметр
    delta=lambda-s_max^2 сохраняет малые положительные сдвиги около
    вырожденного предельного случая. Малые ненулевые компоненты не
    объявляются нулевыми по допуску. Физическая допустимость классов
    возмущений и вероятностная интерпретация задаются вызывающим кодом.
    """
    a = real_array(a, name="a")
    D = _matrix(D, "D")
    if a.ndim != 1 or D.shape[0] != a.size:
        raise ValueError("Expected a.shape=(k,), D.shape=(k,r)")
    radius = real_scalar(radius, name="radius")
    rtol = real_scalar(rtol, name="rtol")
    if radius < 0.0 or rtol <= 0.0:
        raise ValueError("radius must be nonnegative and rtol must be positive")
    if isinstance(maxiter, (bool, np.bool_)):
        raise ValueError("maxiter must be a positive integer")
    try:
        maxiter = index(maxiter)
    except TypeError as error:
        raise ValueError("maxiter must be a positive integer") from error
    if maxiter < 1:
        raise ValueError("maxiter must be a positive integer")

    columns = D.shape[1]
    amax = float(np.max(np.abs(a), initial=0.0))
    dmax = float(np.max(np.abs(D), initial=0.0))
    product = real_scalar(radius * dmax, name="scaled model directions")
    scale = max(amax, product) or 1.0
    aa = a / scale
    matrix = (D / dmax) * (product / scale) if dmax else np.zeros_like(D)
    a2 = _square_norm(aa, "scaled offset norm")
    # Небольшой внутренний сдвиг точки; он не доказывает её FP64-допустимость.
    inward = 1.0 - 8.0 * np.finfo(float).eps

    def finish(point, upper, status, iterations):
        norm = _norm(point)
        if norm:
            point = point * (inward / norm)
        witness = real_array(radius * point, name="model witness")
        actual = witness / radius if radius else np.zeros(columns)
        residual = real_array(aa + matrix @ actual, name="scaled goal residual")
        attained = _square_norm(residual, "scaled attained value")
        value = real_scalar((attained * scale) * scale, name="attained value")
        dual = real_scalar((upper * scale) * scale, name="dual upper estimate")
        if (attained > 0.0 and value == 0.0) or (upper > 0.0 and dual == 0.0):
            raise ValueError("Objective underflowed; rescale the target units")
        gap = dual - value
        scaled_gap = upper - attained
        if gap < 0.0 or scaled_gap < 0.0:
            status = "numerical_inconsistency"
        return (
            BallMaximum(_owned(witness), value, dual, gap, status, iterations),
            scaled_gap,
        )

    if radius == 0.0 or columns == 0 or not np.any(matrix):
        status = "constant_estimate"
        if radius and columns and np.any(D):
            status = "numerically_zero_estimate"
        return finish(np.zeros(columns), a2, status, 0)[0]

    U, singular, Vt = np.linalg.svd(matrix, full_matrices=False)
    if not singular.size or singular[0] == 0.0:
        return finish(np.zeros(columns), a2, "numerically_zero_estimate", 0)[0]
    g = singular * (U.T @ aa)
    gaps = (singular[0] - singular) * (singular[0] + singular)
    top = gaps == 0.0
    t0 = np.zeros_like(singular)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        t0[~top] = g[~top] / gaps[~top]
    norm0 = _norm(t0)

    if np.all(g[top] == 0.0) and norm0 <= 1.0:
        point = t0.copy()
        first = int(np.flatnonzero(top)[0])
        point[first] = np.sqrt((1.0 - norm0) * (1.0 + norm0))
        upper = a2 + singular[0] ** 2 + float(g @ point)
        result, gap = finish(Vt.T @ point, upper, "hard_case_estimate", 0)
        if result.status != "numerical_inconsistency" and gap > rtol * max(1.0, abs(upper)):
            return BallMaximum(
                result.witness,
                result.attained_estimate,
                result.dual_upper_estimate,
                result.gap_estimate,
                "resolution_limit_estimate",
                0,
            )
        return result

    def coefficients(delta):
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            return g / (delta + gaps)

    hi = float(np.nextafter(_norm(g), np.inf))
    while _norm(coefficients(hi)) > 1.0:
        hi *= 2.0
        if not np.isfinite(hi):
            raise ValueError("Failed to bracket the secular equation")
    lo = max(float(np.nextafter(0.0, 1.0)), _norm(g[top]))
    if _norm(coefficients(lo)) <= 1.0:
        hi = lo

    for iteration in range(maxiter + 1):
        point = coefficients(hi)
        upper = a2 + singular[0] ** 2 + hi + float(g @ point)
        result, gap = finish(Vt.T @ point, upper, "iterating", iteration)
        if result.status == "numerical_inconsistency":
            return result
        if 0.0 <= gap <= rtol * max(1.0, abs(upper)):
            return BallMaximum(
                result.witness,
                result.attained_estimate,
                result.dual_upper_estimate,
                result.gap_estimate,
                "converged_estimate",
                iteration,
            )
        if iteration == maxiter:
            status = "iteration_limit_estimate"
        elif lo >= hi:
            status = "resolution_limit_estimate"
        else:
            middle = float(np.exp(0.5 * (np.log(lo) + np.log(hi))))
            status = None if lo < middle < hi else "resolution_limit_estimate"
        if status is not None:
            return BallMaximum(
                result.witness,
                result.attained_estimate,
                result.dual_upper_estimate,
                result.gap_estimate,
                status,
                iteration,
            )
        if _norm(coefficients(middle)) > 1.0:
            lo = middle
        else:
            hi = middle
    raise RuntimeError("Unreachable secular iteration state")


@dataclass(frozen=True)
class AffineRisk:
    """Слагаемые целевого риска фиксированного аффинного оценивателя."""

    parameter_variance: float
    noise_variance: float
    robust: BallMaximum
    value_estimate: float
    upper_estimate: float
    gap_estimate: float

    @property
    def converged_estimate(self):
        return self.robust.converged_estimate

    @property
    def certified(self):
        return False


def affine_goal_risk(E, offset, O, T, Cxi, Ce, L, B, Ceta, rho, *, rtol=1e-12, maxiter=256):
    r"""Оценить риск полной аффинной карты в произвольной целевой метрике.

    Parameters
    ----------
    E, O, T, L : array_like
        Матрицы форм (v,d), (d,p), (v,p), (k,v).
    offset : array_like, shape (v,)
        Центральное смещение выхода, до применения L.
    Cxi, Ce, Ceta : array_like or CovarianceFactor
        Ковариации вариаций параметров, совместного шума данных и
        модельного возмущения. Допустимы диагонали и прямоугольные факторы.
    B : array_like, shape (d,r)
        Карта модельного возмущения в данные.
    rho : float
        Неотрицательный радиус модельного класса.
    rtol, maxiter : optional
        Параметры robust_ball_maximum.

    Returns
    -------
    result : AffineRisk
        Средние квадратичные вклады и оценки наихудшего смещения.

    Notes
    -----
    Для d=d0+O xi+B eta+e, v=v0+T xi, vhat=vhat0+E(d-d0)
    используются нулевые средние xi,e и Cov(xi,e)=0. Ce содержит ВСЕ
    корреляции компонентов данных, включая историю и основной интервал.

    Критерий равен
    ||L(EO-T) Cxi^(1/2)||_F^2 + ||LE Ce^(1/2)||_F^2
    + max ||L offset + LE B Ceta^(1/2) z||^2, ||z||<=rho.

    Для нелинейного адаптивного оценивателя это риск объявленного
    аффинного приближения, не автоматически риск исходного метода.
    При задании ковариаций матрицами используется их проверенная
    симметричная часть. Отрицательные собственные значения не обрезаются.
    Диагональные ковариации применяются по столбцам без плотной матрицы корня.
    Произведения используют FP64: вызывающая процедура должна задавать
    согласованные масштабы переменных. Обнаруженная потеря диапазона
    отклоняется; контроль флагов не является верификацией всей BLAS-арифметики.
    """
    E, O, T, L, B = (
        _matrix(value, name) for value, name in ((E, "E"), (O, "O"), (T, "T"), (L, "L"), (B, "B"))
    )
    outputs, data = E.shape
    parameters = O.shape[1]
    if O.shape[0] != data or T.shape != (outputs, parameters):
        raise ValueError("E, O and T have incompatible spaces")
    if L.shape[1] != outputs or B.shape[0] != data:
        raise ValueError("L or B has incompatible spaces")
    offset = real_array(offset, name="offset")
    if offset.shape != (outputs,):
        raise ValueError("offset must belong to the output space")
    parameter_factor = _covariance_factor(Cxi, parameters, "Cxi")
    noise_factor = _covariance_factor(Ce, data, "Ce")
    model_factor = _covariance_factor(Ceta, B.shape[1], "Ceta")
    try:
        with np.errstate(over="raise", invalid="raise", under="raise"):
            state = real_array(
                L @ (E @ _right_factor(O, parameter_factor) - _right_factor(T, parameter_factor)),
                name="parameter effect",
            )
            noise = real_array(L @ _right_factor(E, noise_factor), name="noise effect")
            bias = real_array(L @ offset, name="goal offset")
            directions = real_array(L @ (E @ _right_factor(B, model_factor)), name="model effect")
    except FloatingPointError as error:
        raise ValueError("Affine arithmetic lost its scale; rescale the input units") from error
    parameter_variance = _square_norm(state, "parameter variance")
    noise_variance = _square_norm(noise, "noise variance")
    robust = robust_ball_maximum(bias, directions, rho, rtol=rtol, maxiter=maxiter)
    base = real_scalar(parameter_variance + noise_variance, name="stochastic contribution")
    value = real_scalar(base + robust.attained_estimate, name="affine risk")
    upper = real_scalar(base + robust.dual_upper_estimate, name="affine upper estimate")
    return AffineRisk(parameter_variance, noise_variance, robust, value, upper, upper - value)


@dataclass(frozen=True)
class ProfileInformation:
    """Информация после исключения численно разрешённых помеховых направлений.

    Attributes
    ----------
    information, projected : ndarray
        Матрица projected.T@projected и отбелённая проекция чувствительности.
    nuisance_rank : int
        Численный ранг помехового оператора.
    nuisance_threshold : float
        Абсолютный порог его сингулярных чисел.

    Notes
    -----
    Все массивы собственные и с запрещённой записью. При усечении ненулевых
    сингулярных направлений результат отличается от профиля полной
    свободной помехи; его положительность не доказывает идентифицируемость.
    """

    information: np.ndarray
    projected: np.ndarray
    nuisance_rank: int
    nuisance_threshold: float


def profile_information(A, nuisance, covariance, *, relative_threshold=1e-10):
    r"""Исключить свободные помеховые направления в метрике наблюдений.

    Parameters
    ----------
    A, nuisance : array_like, shape (m,n), (m,r)
        Чувствительность целевых и помеховых неизвестных в одинаковых строках.
    covariance : array_like
        Рабочая SPD-метрика, передаваемая существующей операции whiten.
        Она не обязана совпадать с реальной Ce в affine_goal_risk.
    relative_threshold : float, optional
        Относительный порог SVD помехового оператора.

    Returns
    -------
    result : ProfileInformation
        Информация данных без строк регуляризатора.

    Notes
    -----
    После отбеливания вычисляется Atilde-Q@(Q.T@Atilde), где Q содержит
    разрешённые левые сингулярные направления помехи. Матрица nuisance.T@
    nuisance не формируется; скрытого ridge нет.
    Масштаб столбцов помехи задаёт вызывающая процедура. Порог относится
    к этим координатам. Усечение малых ненулевых направлений сужает
    пространство помехи; результат не является сертификатом точного
    ранга или наблюдаемости полной задачи.
    """
    A = _matrix(A, "A")
    nuisance = _matrix(nuisance, "nuisance")
    if nuisance.shape[0] != A.shape[0]:
        raise ValueError("A and nuisance must use identical observation rows")
    ratio = real_scalar(relative_threshold, name="relative_threshold")
    if ratio < 0.0:
        raise ValueError("relative_threshold must be nonnegative")
    rows, columns = A.shape
    if rows == 0:
        cov = real_array(covariance, name="covariance")
        if cov.shape not in {(0,), (0, 0)}:
            raise ValueError("Empty observations require an empty covariance")
        return ProfileInformation(_owned(np.zeros((columns, columns))), _owned(A), 0, 0.0)
    combined = np.concatenate((A, nuisance), axis=1)
    white, _ = whiten(combined, np.zeros(rows), covariance)
    white = real_array(white, name="whitened design")
    target, other = white[:, :columns], white[:, columns:]
    if other.shape[1]:
        U, singular, _ = np.linalg.svd(other, full_matrices=False)
        threshold = real_scalar(ratio * singular[0], name="nuisance threshold")
        rank = int(np.count_nonzero(singular > threshold))
        Q = U[:, :rank]
        projected = target - Q @ (Q.T @ target)
    else:
        threshold, rank, projected = 0.0, 0, target.copy()
    projected = real_array(projected, name="profiled sensitivity")
    information = real_array(projected.T @ projected, name="profile information")
    return ProfileInformation(_owned(information), _owned(projected), rank, threshold)


def _named_values(values, name, *, positive=False):
    if not hasattr(values, "items") or not values:
        raise ValueError(f"{name} must be a nonempty mapping")
    result = []
    for key, value in values.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{name} keys must be nonempty strings")
        scalar = real_scalar(value, name=f"{name}[{key}]")
        if scalar < 0.0 or (positive and scalar == 0.0):
            raise ValueError(f"{name} values must be {'positive' if positive else 'nonnegative'}")
        result.append((key, scalar))
    return tuple(sorted(result))


@dataclass(frozen=True, init=False)
class DesignCandidate:
    """Одна конфигурация с заданной стоимостью и оценками произвольных целей."""

    identifier: str
    cost: float
    _risks: tuple

    def __init__(self, identifier, cost, risks):
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("identifier must be a nonempty string")
        cost = real_scalar(cost, name="cost")
        if cost < 0.0:
            raise ValueError("cost must be nonnegative")
        object.__setattr__(self, "identifier", identifier)
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "_risks", _named_values(risks, "risks"))

    @property
    def risks(self):
        return dict(self._risks)


@dataclass(frozen=True)
class DesignRank:
    """Нормированные цели и ключ сравнения одной конфигурации."""

    identifier: str
    cost: float
    _normalized: tuple
    maximum: float
    total: float

    @property
    def normalized(self):
        return dict(self._normalized)


@dataclass(frozen=True)
class BudgetSelection:
    """Результат выбора из предоставленной конечной таблицы при одном бюджете."""

    budget: float
    mode: str
    ranking: tuple
    winner: str | None
    _goal_winners: tuple
    pareto: tuple

    @property
    def goal_winners(self):
        return dict(self._goal_winners)


def select_designs(candidates, baselines, *, budgets, mode="exact"):
    r"""Выбрать конфигурации по общим нормировкам и ограничениям стоимости.

    Parameters
    ----------
    candidates : iterable of DesignCandidate
        Предоставленная конечная таблица. Отсутствующие сети не достраиваются.
    baselines : mapping of str to positive float
        Один общий положительный масштаб каждой цели для всех конфигураций.
    budgets : iterable of float
        Конечные неотрицательные значения бюджета, в порядке результата.
    mode : {"exact", "at_most"}, optional
        Точное равенство стоимости бюджету либо ограничение сверху.
        Стоимость сравнивается в заданных единицах без скрытого допуска.

    Returns
    -------
    results : tuple of BudgetSelection
        Упорядоченные конфигурации, победитель, отдельные минимумы целей
        и множество Парето по каждому бюджету. Если кандидатов нет,
        winner=None и соответствующие коллекции пусты.

    Notes
    -----
    Сравнение: максимум нормированных целей, затем сумма, затем identifier.
    Число целей произвольно. Это минимум предоставленных численных оценок;
    глобальная оптимальность среди нерассмотренных городских сетей и
    строгий порядок при неопределённой арифметической погрешности не следуют.
    Потеря положительного риска до нуля при нормировке отвергается:
    в таком случае необходимо изменить общий масштаб соответствующей цели.
    Оценочная истина не является входом этой процедуры.
    """
    if mode not in {"exact", "at_most"}:
        raise ValueError("mode must be 'exact' or 'at_most'")
    baseline = dict(_named_values(baselines, "baselines", positive=True))
    goals = tuple(baseline)
    rows = []
    identifiers = set()
    for candidate in candidates:
        if not isinstance(candidate, DesignCandidate):
            raise TypeError("candidates must contain DesignCandidate values")
        if candidate.identifier in identifiers:
            raise ValueError("Candidate identifiers must be unique")
        identifiers.add(candidate.identifier)
        risks = candidate.risks
        if set(risks) != set(goals):
            raise ValueError("All candidates must use exactly the baseline goals")
        normalized = tuple(
            (goal, real_scalar(risks[goal] / baseline[goal], name="normalized risk"))
            for goal in goals
        )
        if any(risks[goal] > 0.0 and value == 0.0 for goal, value in normalized):
            raise ValueError("A positive normalized risk underflows; rescale the goal")
        numbers = [value for _, value in normalized]
        total = real_scalar(sum(numbers), name="sum of normalized risks")
        rows.append(
            DesignRank(candidate.identifier, candidate.cost, normalized, max(numbers), total)
        )
    results = []
    for raw_budget in budgets:
        budget = real_scalar(raw_budget, name="budget")
        if budget < 0.0:
            raise ValueError("budget must be nonnegative")
        feasible = [
            row for row in rows if (row.cost == budget if mode == "exact" else row.cost <= budget)
        ]
        ranking = tuple(sorted(feasible, key=lambda row: (row.maximum, row.total, row.identifier)))
        goal_winners = (
            tuple(
                (
                    goal,
                    min(
                        ranking,
                        key=lambda row: (
                            row.normalized[goal],
                            row.total,
                            row.identifier,
                        ),
                    ).identifier,
                )
                for goal in goals
            )
            if ranking
            else ()
        )
        pareto = tuple(
            sorted(
                row.identifier
                for row in ranking
                if not any(
                    all(other.normalized[goal] <= row.normalized[goal] for goal in goals)
                    and any(other.normalized[goal] < row.normalized[goal] for goal in goals)
                    for other in ranking
                )
            )
        )
        results.append(
            BudgetSelection(
                budget,
                mode,
                ranking,
                ranking[0].identifier if ranking else None,
                goal_winners,
                pareto,
            )
        )
    return tuple(results)
