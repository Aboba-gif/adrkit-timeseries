"""Покомпонентные границы, проекция и анализ линейных наблюдений."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from scipy.linalg import norm, solve_triangular
from scipy.optimize import OptimizeResult, linprog
from scipy.sparse.linalg import LinearOperator, minres

from adrkit.contracts import require_capability, validate_map_capabilities

from ._numeric import real_array, real_scalar
from .linalg import _spd_factor


@dataclass(frozen=True)
class NonnegativeAnalysisLinearization:
    """Действие производной выбранной стационарной ветви анализа.

    Parameters
    ----------
    operator : scipy.sparse.linalg.LinearOperator
        Карта (delta_reference, delta_target) в delta_point и её транспонированное
        действие. Свободные линейные системы решаются с численным допуском.
    metadata : Mapping
        Неизменяемые настройки и численные остатки исходной стационарности.
        Они не удостоверяют гладкость ветви, невырожденность или минимум.
    """

    operator: LinearOperator
    metadata: Mapping


def gradient_mapping(theta, gradient):
    r"""Вычислить отображение проекционного градиента.

    Parameters
    ----------
    theta, gradient : array_like
        Точка и евклидов градиент одинаковой формы; theta
        неотрицательна.

    Returns
    -------
    mapping : ndarray
        Отображение с единичным шагом той же формы.

    Notes
    -----
    Для проекции :math:`P_+` на неотрицательный ортант:

    .. math::

        g_P(\theta)=\theta-P_+(\theta-g).

    Здесь :math:`g` — переданный gradient.
    """
    return np.asarray(theta) - np.maximum(0.0, np.asarray(theta) - gradient)


def box_bounds(bounds, size):
    """Подготовить покомпонентные границы для SciPy.

    Parameters
    ----------
    bounds : sequence of pairs or None
        Нижняя и верхняя границы каждой координаты; None задаёт
        отсутствие границы.
    size : int
        Число координат.

    Returns
    -------
    bounds : list of tuple
        Пары границ в порядке координат; обратные и недопустимые
        границы отклоняются.
    """
    if bounds is None:
        return [(None, None)] * size
    if len(bounds) != size:
        raise ValueError("One bound pair per coefficient required")
    result = []
    for low, high in bounds:
        low = -np.inf if low is None else float(low)
        high = np.inf if high is None else float(high)
        if np.isnan(low) or np.isnan(high) or low > high or low == np.inf or high == -np.inf:
            raise ValueError("Invalid coefficient bounds")
        result.append((low, high))
    return result


def project_nonnegative_equalities(
    reference,
    observation,
    target,
    mass,
    *,
    equality_tolerance=1e-10,
    kkt_tolerance=1e-10,
    maxiter=100,
    max_backtracks=50,
):
    r"""Проецировать поле на линейные равенства и неотрицательный ортант.

    Parameters
    ----------
    reference : array_like, shape (N,)
        Конечное вещественное опорное поле r.
    observation : array_like, shape (M, N)
        Матрица H фактических наблюдений, M,N > 0; знаковые строки допустимы.
    target : array_like, shape (M,)
        Значения d в равенствах H U = d.
    mass : array_like, shape (N,)
        Строго положительная диагональ метрики M. Полная матрица метрики
        этим алгоритмом не поддерживается.
    equality_tolerance, kkt_tolerance : float, optional
        Положительные абсолютные допуски исходных равенств и остатка ККТ.
    maxiter, max_backtracks : int, optional
        Положительные пределы двойственных шагов и сокращений шага.

    Returns
    -------
    result : scipy.optimize.OptimizeResult
        Проверенная точка x, множители equality_multipliers и
        lower_multipliers, цель fun, число шагов nit и diagnostics с
        отдельными остатками равенств, границ, стационарности и
        дополнительности. Только результат, прошедший оба допуска,
        возвращается с success=True.

    Raises
    ------
    ValueError
        Недопустимый вход или получен явно проверенный вектор разделения,
        свидетельствующий о несовместности.
    RuntimeError
        Заданные допуски не достигнуты либо вычисленный сертификат или
        цель не конечны. Это само по себе не означает несовместность.

    Notes
    -----
    Решается строго выпуклая задача
    ``min 0.5 * sum(mass * (U-reference)**2), H@U=target, U>=0``.
    Положительная часть ``U(mu)=max(0,r-H.T@mu/mass)`` является точным
    минимумом лагранжиана по ортантному ограничению; обрезания уже найденного
    решения равенств здесь нет. Двойственная минимизируемая функция равна
    ``0.5 * sum(mass * U(mu)**2) + mu@target`` и может быть отрицательной.
    Используются полугладкий шаг Ньютона в пространстве M множителей и
    проверка убывания. При вырожденной активной матрице применяется спуск
    по градиенту с глобальной липшицевой оценкой.

    Невязка исходных равенств проверяется независимо от критерия оптимизатора.
    Разность прямой и двойственной целей при недопустимой точке не выдаётся
    за сертификат оптимальности. Равенства не заменяются большим штрафом.
    """
    r = real_array(reference, name="reference", copy=True)
    h = real_array(observation, name="observation", copy=True)
    d = real_array(target, name="target", copy=True)
    m = real_array(mass, name="mass", copy=True)
    if (
        r.ndim != 1
        or not r.size
        or h.ndim != 2
        or not h.shape[0]
        or h.shape[1] != r.size
        or d.shape != (h.shape[0],)
        or m.shape != r.shape
        or np.any(m <= 0)
    ):
        raise ValueError("Projection requires H(M,N), r(N), d(M), positive diagonal mass(N)")
    equality_tolerance = real_scalar(equality_tolerance, name="equality_tolerance")
    kkt_tolerance = real_scalar(kkt_tolerance, name="kkt_tolerance")
    if min(equality_tolerance, kkt_tolerance) <= 0:
        raise ValueError("Projection tolerances must be positive")
    for name, value in (("maxiter", maxiter), ("max_backtracks", max_backtracks)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        gram = (h / m) @ h.T
        right = h @ r - d
    if not np.isfinite(gram).all() or not np.isfinite(right).all():
        raise ValueError("Projection metric and observation scaling overflow the dual Gram matrix")
    lipschitz = float(np.linalg.norm(gram, 2))
    mu = np.linalg.lstsq(gram, right, rcond=None)[0]

    def evaluate(multipliers):
        with np.errstate(over="ignore", invalid="ignore"):
            u = np.maximum(0.0, r - h.T @ multipliers / m)
            value = 0.5 * np.dot(m * u, u) + multipliers @ d
            gradient = d - h @ u
        return float(value), gradient, u

    def certify(u, multipliers, iteration):
        with np.errstate(over="ignore", invalid="ignore"):
            equality = h @ u - d
            difference = u - r
            lagrangian = m * difference + h.T @ multipliers
            lower = np.maximum(lagrangian, 0.0)
            stationarity = lagrangian - lower
            complementarity = u * lower
            objective = float(0.5 * np.dot(m * difference, difference))
        if not all(
            np.isfinite(value).all()
            for value in (
                u,
                multipliers,
                equality,
                lagrangian,
                lower,
                stationarity,
                complementarity,
            )
        ) or not np.isfinite(objective):
            raise RuntimeError(
                "Nonnegative equality projection is uncertified: nonfinite KKT certificate or objective"
            )
        diagnostics = {
            "equality_residual": equality.tolist(),
            "equality_residual_inf": float(np.max(np.abs(equality))),
            "minimum_state": float(np.min(u)),
            "minimum_lower_multiplier": float(np.min(lower)),
            "stationarity_residual_inf": float(np.max(np.abs(stationarity))),
            "complementarity_residual_inf": float(np.max(np.abs(complementarity))),
            "active_count": int(np.count_nonzero(u == 0)),
        }
        kkt = max(
            max(0.0, -diagnostics["minimum_state"]),
            diagnostics["stationarity_residual_inf"],
            diagnostics["complementarity_residual_inf"],
        )
        diagnostics["kkt_residual_inf"] = kkt
        if diagnostics["equality_residual_inf"] <= equality_tolerance and kkt <= kkt_tolerance:
            return OptimizeResult(
                x=u.copy(),
                equality_multipliers=multipliers.copy(),
                lower_multipliers=lower,
                fun=objective,
                nit=iteration,
                success=True,
                diagnostics=diagnostics,
            )
        return None

    iteration = 0
    for iteration in range(maxiter + 1):
        value, gradient, u = evaluate(mu)
        if not np.isfinite(value) or not np.isfinite(gradient).all() or not np.isfinite(u).all():
            raise RuntimeError(
                "Nonnegative equality projection is uncertified: nonfinite dual iterate"
            )
        result = certify(u, mu, iteration)
        if result is not None:
            return result
        if iteration == maxiter or lipschitz == 0:
            break
        positive = u > 0
        active_gram = (h[:, positive] / m[positive]) @ h[:, positive].T
        direction = np.linalg.lstsq(active_gram, -gradient, rcond=None)[0]
        descent = float(gradient @ direction)
        if (
            not np.isfinite(descent)
            or descent >= 0
            or np.linalg.norm(active_gram @ direction + gradient) > 0.5 * np.linalg.norm(gradient)
        ):
            direction = -gradient / lipschitz
            descent = float(gradient @ direction)
        step = 1.0
        for _ in range(max_backtracks):
            trial = mu + step * direction
            trial_value, _, _ = evaluate(trial)
            if np.isfinite(trial_value) and trial_value <= value + 1e-4 * step * descent:
                mu = trial
                break
            step *= 0.5
        else:
            break

    # Сходимость двойственного метода не является проверкой совместности.
    # Малое пространство множителей позволяет отдельно искать вектор Фаркаша.
    separation = linprog(
        np.zeros(h.shape[0]),
        A_ub=-h.T,
        b_ub=np.zeros(r.size),
        A_eq=d[None, :],
        b_eq=[-1.0],
        bounds=[(None, None)] * h.shape[0],
        method="highs",
    )
    if separation.success:
        vector = separation.x
        with np.errstate(over="ignore", invalid="ignore"):
            cone_product = h.T @ vector
            target_product = d @ vector
        if (
            not np.isfinite(vector).all()
            or not np.isfinite(cone_product).all()
            or not np.isfinite(target_product)
        ):
            raise RuntimeError(
                "Nonnegative equality projection is uncertified: nonfinite separation certificate"
            )
        # Ортант не ограничен сверху: даже малая отрицательная компонента
        # H.T@vector не является допустимым численным запасом Фаркаша.
        if np.min(cone_product) >= 0 and target_product < 0:
            raise ValueError(
                "Nonnegative equality projection is infeasible: verified separation vector"
            )
    residual = float(np.max(np.abs(h @ u - d))) if np.isfinite(u).all() else float("inf")
    raise RuntimeError(
        f"Nonnegative equality projection is uncertified after {iteration} steps; "
        f"equality residual={residual:.6g}"
    )


def analyze_nonnegative_observations(
    reference,
    observation,
    target,
    mass,
    covariance,
    *,
    kkt_tolerance=1e-10,
    maxiter=100,
    max_backtracks=50,
):
    r"""Согласовать неотрицательное состояние с неточными линейными наблюдениями.

    Parameters
    ----------
    reference : array_like, shape (N,)
        Конечное вещественное опорное состояние r.
    observation : array_like, shape (M, N)
        Матрица H; строки могут иметь оба знака. M может быть равным нулю.
    target : array_like, shape (M,)
        Конечные измеренные значения d в координатах H@U.
    mass : array_like, shape (N,)
        Строго положительная диагональ априорной точности. Это не обязательно
        геометрические площади узлов; её физический смысл задаёт вызывающий код.
    covariance : array_like, shape (M,) or (M, M)
        Положительная диагональ либо симметричная положительно определённая
        ковариация наблюдательной невязки. При M=0 передаётся пустой массив.
    kkt_tolerance : float, optional
        Положительный абсолютный допуск исходной стационарности,
        дополнительности и двойственной связи.
    maxiter, max_backtracks : int, optional
        Положительные пределы двойственных шагов и сокращений шага.

    Returns
    -------
    result : scipy.optimize.OptimizeResult
        Проверенное состояние x, observation_multipliers, lower_multipliers,
        стоимость fun, число шагов nit и исходные остатки в diagnostics.
        success=True возвращается только после проверки условий ККТ.

    Raises
    ------
    ValueError
        Размеры, числовые значения или метрики недопустимы, либо их
        преобразование в двойственную задачу переполняется.
    RuntimeError
        Не получено конечного состояния с заданной точностью ККТ.

    Notes
    -----
    Решается строго выпуклая задача
    ``min 0.5*sum(mass*(U-reference)**2)
    +0.5*(H@U-target).T@inv(covariance)@(H@U-target), U>=0``.
    Минимум существует и единственен для указанных положительных метрик.
    При отсутствии наблюдений результат равен положительной части reference.

    Двойственная функция в переменной mu равна
    ``0.5*sum(mass*U(mu)**2)+mu@target+0.5*mu@covariance@mu``,
    где ``U(mu)=max(0,reference-H.T@mu/mass)``. Она минимизируется после
    отбеливания наблюдений; постоянная составляющая удаляется из стоимости.
    Обрезание является точным решением ортантной подзадачи, а не исправлением
    найденного неограниченного анализа. Успех оптимизатора не заменяет
    отдельную проверку градиента исходного функционала.
    """
    r = real_array(reference, name="reference", copy=True)
    h = real_array(observation, name="observation", copy=True)
    d = real_array(target, name="target", copy=True)
    m = real_array(mass, name="mass", copy=True)
    cov = real_array(covariance, name="covariance", copy=True)
    if (
        r.ndim != 1
        or not r.size
        or h.ndim != 2
        or h.shape[1] != r.size
        or d.shape != (h.shape[0],)
        or m.shape != r.shape
        or np.any(m <= 0)
    ):
        raise ValueError("Analysis requires H(M,N), r(N), d(M), positive diagonal mass(N)")
    tolerance = real_scalar(kkt_tolerance, name="kkt_tolerance")
    if tolerance <= 0:
        raise ValueError("kkt_tolerance must be positive")
    for name, value in (("maxiter", maxiter), ("max_backtracks", max_backtracks)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    size = h.shape[0]
    factor = _spd_factor(cov, size, "covariance")

    def whiten(values):
        if factor.ndim == 1:
            return values / (factor[:, None] if values.ndim == 2 else factor)
        return solve_triangular(factor, values, lower=True, check_finite=False)

    def unwhiten_transpose(values):
        if factor.ndim == 1:
            return values / factor
        return solve_triangular(factor.T, values, lower=False, check_finite=False)

    def covariance_product(values):
        return factor * factor * values if factor.ndim == 1 else factor @ (factor.T @ values)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        hw, dw = whiten(h), whiten(d)
        gram = (hw / m) @ hw.T
        right = hw @ r - dw
    if not all(np.isfinite(value).all() for value in (hw, dw, gram, right)):
        raise ValueError("Analysis scaling overflows the dual system")
    identity = np.eye(size)
    if size:
        try:
            nu = np.linalg.solve(identity + gram, right)
        except np.linalg.LinAlgError:
            nu = np.zeros(size)
    else:
        nu = np.empty(0)
    base = np.maximum(r, 0.0)
    lipschitz = 1.0 + float(np.linalg.norm(gram, 2)) if size else 1.0

    def evaluate(point):
        with np.errstate(over="ignore", invalid="ignore"):
            u = np.maximum(0.0, r - hw.T @ point / m)
            difference = m * (u - base)
            value = (
                0.5 * (np.dot(difference, u) + np.dot(difference, base))
                + point @ dw
                + 0.5 * np.dot(point, point)
            )
            gradient = dw + point - hw @ u
        return float(value), gradient, u

    def certify(u, point, iteration):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            residual = h @ u - d
            whitened_residual = whiten(residual)
            data_multipliers = unwhiten_transpose(whitened_residual)
            multipliers = unwhiten_transpose(point)
            primal_gradient = m * (u - r) + h.T @ data_multipliers
            lower = np.maximum(primal_gradient, 0.0)
            stationarity = primal_gradient - lower
            complementarity = u * lower
            relation = residual - covariance_product(multipliers)
            difference = u - r
            objective = float(
                0.5 * np.dot(m * difference, difference)
                + 0.5 * np.dot(whitened_residual, whitened_residual)
            )
        if not all(
            np.isfinite(value).all()
            for value in (
                u,
                multipliers,
                data_multipliers,
                lower,
                stationarity,
                complementarity,
                relation,
            )
        ) or not np.isfinite(objective):
            raise RuntimeError("Nonnegative observation analysis is uncertified: nonfinite KKT")
        diagnostics = {
            "observation_residual": residual.tolist(),
            "observation_residual_inf": float(np.max(np.abs(residual), initial=0.0)),
            "dual_relation_residual_inf": float(np.max(np.abs(relation), initial=0.0)),
            "minimum_state": float(np.min(u)),
            "minimum_lower_multiplier": float(np.min(lower)),
            "stationarity_residual_inf": float(np.max(np.abs(stationarity))),
            "complementarity_residual_inf": float(np.max(np.abs(complementarity))),
            "active_count": int(np.count_nonzero(u == 0)),
        }
        kkt = max(
            max(0.0, -diagnostics["minimum_state"]),
            diagnostics["dual_relation_residual_inf"],
            diagnostics["stationarity_residual_inf"],
            diagnostics["complementarity_residual_inf"],
        )
        diagnostics["kkt_residual_inf"] = kkt
        if kkt <= tolerance:
            return OptimizeResult(
                x=u.copy(),
                observation_multipliers=multipliers.copy(),
                lower_multipliers=lower,
                fun=objective,
                nit=iteration,
                success=True,
                diagnostics=diagnostics,
            )
        return None

    for iteration in range(maxiter + 1):
        value, gradient, u = evaluate(nu)
        if not np.isfinite(value) or not np.isfinite(gradient).all() or not np.isfinite(u).all():
            raise RuntimeError(
                "Nonnegative observation analysis is uncertified: nonfinite iterate"
            )
        result = certify(u, nu, iteration)
        if result is not None:
            return result
        if iteration == maxiter or not size:
            break
        positive = u > 0
        hessian = identity + (hw[:, positive] / m[positive]) @ hw[:, positive].T
        try:
            direction = np.linalg.solve(hessian, -gradient)
        except np.linalg.LinAlgError:
            direction = -gradient / lipschitz
        descent = float(gradient @ direction)
        if not np.isfinite(descent) or descent >= 0:
            direction = -gradient / lipschitz
            descent = float(gradient @ direction)
        step = 1.0
        for _ in range(max_backtracks):
            trial = nu + step * direction
            trial_value, _, _ = evaluate(trial)
            if np.isfinite(trial_value) and trial_value <= value + 1e-4 * step * descent:
                nu = trial
                break
            step *= 0.5
        else:
            break
    raise RuntimeError(
        f"Nonnegative observation analysis is uncertified after {iteration} steps; "
        f"KKT residual exceeds {tolerance:.6g}"
    )


def analyze_nonnegative_map(
    reference,
    observation,
    target,
    mass,
    covariance,
    *,
    x0=None,
    kkt_tolerance=1e-8,
    maxiter=50,
    max_backtracks=30,
    armijo=1e-4,
    backtrack_factor=0.5,
    model_change_rtol=1e-12,
    linear_options=None,
):
    r"""Analyze a nonnegative vector using a nonlinear observation map.

    Parameters
    ----------
    reference : array_like, shape (N,)
        Finite prior centre r; negative entries are permitted.
    observation : MapCapabilities
        A map A with vector domain (N,) and codomain (M,). Prediction and
        explicitly declared VJP are required. JVP and Jacobian are unused.
    target : array_like, shape (M,)
        Finite observations d. Signed values are permitted.
    mass : array_like, shape (N,)
        Strictly positive diagonal prior precision m, not necessarily cell areas.
    covariance : array_like, shape (M,) or (M, M)
        Positive observation variance vector or symmetric positive definite R.
    x0 : array_like, shape (N,), optional
        Nonnegative initial iterate. The default is the positive part of r,
        the exact nonnegative minimizer of the prior term alone.
    kkt_tolerance : float, optional
        Positive absolute tolerance for the infinity norm of the original
        coordinate projected gradient x-P_+(x-grad Phi). This is not a
        tolerance for residuals, coefficient errors or objective optimality.
    maxiter, max_backtracks : int, optional
        Positive limits on accepted outer steps and trial step evaluations.
    armijo, backtrack_factor : float, optional
        Numbers strictly between zero and one. Line search tests the original
        nonlinear objective, with an initial step of one.
    model_change_rtol : float, optional
        Finite comparison allowance in [0,1) for the quadratic model change.
        Its absolute allowance is zero. This is a floating-point comparison
        rule, not an interval-arithmetic certificate.
    linear_options : dict or None, optional
        Only kkt_tolerance, maxiter and max_backtracks are forwarded to
        analyze_nonnegative_observations. Its tolerance controls the linear
        subproblem's own KKT residuals in their original units, separately
        from this function's projected-gradient tolerance.

    Returns
    -------
    result : scipy.optimize.OptimizeResult
        x, original objective fun and gradient jac, observation_multipliers,
        projected-gradient diagnostics, counts and objective_history. A true
        success means that the original projected-gradient test passed;
        it does not establish a global minimum or physical reconstruction.
        status is 0 for stationarity, 1 for the outer limit, 2 for an inner
        analysis failure, 3 for a rejected model/direction, 4 for line-search
        failure. Invalid inputs and invalid map outputs raise exceptions.

    Notes
    -----
    The objective is
    ``Phi(x)=0.5*sum(m*(x-r)**2)+0.5*(A(x)-d).T@inv(R)@(A(x)-d)``, x>=0.
    Each row of J=A'(x) is obtained by a VJP of a coordinate covector.
    The step endpoint v solves the native strictly convex linear analysis
    with arguments (r,J,d-A(x)+J@x,m,R). No dense N-by-N prior is formed.

    For p=v-x the explicit model change is
    ``g@p + 0.5*(sum(m*p*p) + ||R^(-1/2) Jp||**2)``. It must be at most
    ``model_change_rtol * (abs(g)@abs(p) + 0.5*curvature)``; additionally
    g@p must be strictly negative. There is no alternative search direction.
    An exact subproblem satisfies this descent property; a finite inner
    tolerance does not guarantee it, hence the independent checks.

    Trials are convex combinations of two feasible vectors, rather than
    clipping an unconstrained solution. Armijo and strict decrease are tested
    on Phi. Only the forward map is used during backtracking. Continuous
    differentiability and accurate declared derivatives are assumptions of
    the method, not facts established by optimizer termination.
    """
    validate_map_capabilities(observation, name="observation")
    domain, codomain = observation.domain, observation.codomain
    if len(domain.shape) != 1 or len(codomain.shape) != 1:
        raise ValueError("Nonlinear analysis requires vector domain and codomain spaces")
    derivative = require_capability(
        observation.vjp, operation="VJP", consumer="nonlinear observation analysis"
    )
    r = real_array(reference, name="reference", copy=True)
    d = real_array(target, name="target", copy=True)
    m = real_array(mass, name="mass", copy=True)
    cov = real_array(covariance, name="covariance", copy=True)
    if r.shape != domain.shape or d.shape != codomain.shape or m.shape != r.shape:
        raise ValueError("Analysis vectors must agree with the declared map spaces")
    if np.any(m <= 0):
        raise ValueError("Analysis mass must be strictly positive")
    factor = _spd_factor(cov, codomain.size, "covariance")
    tolerance = real_scalar(kkt_tolerance, name="kkt_tolerance")
    decrease = real_scalar(armijo, name="armijo")
    contraction = real_scalar(backtrack_factor, name="backtrack_factor")
    model_rtol = real_scalar(model_change_rtol, name="model_change_rtol")
    if tolerance <= 0 or not 0 < decrease < 1 or not 0 < contraction < 1:
        raise ValueError("Require positive kkt_tolerance and line-search parameters in (0,1)")
    if not 0 <= model_rtol < 1:
        raise ValueError("model_change_rtol must lie in [0,1)")
    for name, value in (("maxiter", maxiter), ("max_backtracks", max_backtracks)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if linear_options is not None and not isinstance(linear_options, dict):
        raise ValueError("linear_options must be a dictionary or None")
    linear = {} if linear_options is None else dict(linear_options)
    if set(linear) - {"kkt_tolerance", "maxiter", "max_backtracks"}:
        raise ValueError("linear_options supports only native linear analysis options")
    for name, value in linear.items():
        if name == "kkt_tolerance":
            if real_scalar(value, name="linear kkt_tolerance") <= 0:
                raise ValueError("Linear kkt_tolerance must be positive")
        elif type(value) is not int or value < 1:
            raise ValueError(f"Linear {name} must be a positive integer")
    x = np.maximum(r, 0.0) if x0 is None else real_array(x0, name="x0", copy=True)
    if x.shape != r.shape or np.any(x < 0):
        raise ValueError("x0 must be a nonnegative vector in the declared domain")

    nfev = njev = nvjp = linear_nit = 0
    last_model_change = last_model_allowance = last_directional = None
    last_step = 0.0

    def whiten(vector):
        if factor.ndim == 1:
            return vector / factor
        return solve_triangular(factor, vector, lower=True, check_finite=False)

    def whiten_transpose(vector):
        if factor.ndim == 1:
            return vector / factor
        return solve_triangular(factor.T, vector, lower=False, check_finite=False)

    def evaluate(point):
        nonlocal nfev
        nfev += 1
        values = observation.predict.predict(domain.array(point, copy=True))
        values = np.asarray(codomain.array(values, name="prediction", copy=True), dtype=float)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            residual = values - d
            white = whiten(residual)
            difference = point - r
            value = 0.5 * (np.dot(m * difference, difference) + np.dot(white, white))
        if not np.isfinite(value) or not np.isfinite(white).all():
            raise RuntimeError("Nonlinear analysis has a nonfinite original objective")
        return values, white, float(value)

    def linearize(point, white):
        nonlocal njev, nvjp
        njev += 1
        matrix = np.empty((codomain.size, domain.size))
        unit = np.zeros(codomain.size)
        for row in range(codomain.size):
            unit[row] = 1.0
            values = derivative.vjp(
                domain.array(point, copy=True), codomain.array(unit, copy=True)
            )
            matrix[row] = domain.array(values, name="observation VJP", copy=True)
            unit[row] = 0.0
            nvjp += 1
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            multiplier = whiten_transpose(white)
            gradient = m * (point - r) + matrix.T @ multiplier
        if not np.isfinite(gradient).all() or not np.isfinite(multiplier).all():
            raise RuntimeError("Nonlinear analysis has a nonfinite original gradient")
        return matrix, multiplier, gradient

    prediction, whitened_residual, fun = evaluate(x)
    history = [fun]
    iteration = 0
    gradient = multipliers = matrix = None
    optimality = float("inf")

    def result(status, message):
        return OptimizeResult(
            x=x.copy(),
            fun=fun,
            jac=gradient.copy(),
            observation_multipliers=multipliers.copy(),
            nit=iteration,
            nfev=nfev,
            njev=njev,
            nvjp=nvjp,
            linear_nit=linear_nit,
            success=status == 0,
            status=status,
            message=message,
            objective_history=np.asarray(history),
            diagnostics={
                "projected_gradient_inf": optimality,
                "kkt_residual_inf": optimality,
                "minimum_state": float(np.min(x)),
                "last_model_change": last_model_change,
                "last_model_allowance": last_model_allowance,
                "last_directional_derivative": last_directional,
                "last_step_size": last_step,
            },
        )

    for iteration in range(maxiter + 1):
        matrix, multipliers, gradient = linearize(x, whitened_residual)
        # For x>=0, min(x,g) equals x-P_+(x-g) without subtractive cancellation.
        optimality = float(np.max(np.abs(np.minimum(x, gradient))))
        if optimality <= tolerance:
            return result(
                0,
                "Original projected-gradient tolerance satisfied; stationarity is not "
                "a global minimum certificate.",
            )
        if iteration == maxiter:
            return result(1, "Maximum outer iterations reached before original stationarity")

        with np.errstate(over="ignore", invalid="ignore"):
            adjusted_target = d - prediction + matrix @ x
        if not np.isfinite(adjusted_target).all():
            return result(2, "Linearized target is nonfinite")
        try:
            inner = analyze_nonnegative_observations(r, matrix, adjusted_target, m, cov, **linear)
        except (RuntimeError, ValueError) as error:
            return result(2, "Linearized nonnegative analysis failed: " + str(error))
        endpoint = real_array(inner.x, name="linearized state", copy=True)
        if not inner.success or endpoint.shape != x.shape or np.any(endpoint < 0):
            return result(2, "Linearized analysis did not return a verified feasible endpoint")
        linear_nit += int(inner.nit)
        direction = endpoint - x
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            last_directional = float(gradient @ direction)
            white_tangent = whiten(matrix @ direction)
            curvature = float(
                np.dot(m * direction, direction) + np.dot(white_tangent, white_tangent)
            )
            last_model_change = last_directional + 0.5 * curvature
            scale = float(np.abs(gradient) @ np.abs(direction)) + 0.5 * curvature
            last_model_allowance = model_rtol * scale
        if (
            not np.isfinite([last_directional, curvature, last_model_change, scale]).all()
            or last_directional >= 0
            or last_model_change > last_model_allowance
        ):
            return result(3, "Linearized endpoint failed original descent or model-change test")
        step = 1.0
        last_step = 0.0
        for _ in range(max_backtracks):
            trial = (1.0 - step) * x + step * endpoint
            trial_prediction, trial_white, trial_fun = evaluate(trial)
            if trial_fun < fun and trial_fun <= fun + decrease * step * last_directional:
                x, prediction = trial, trial_prediction
                whitened_residual, fun = trial_white, trial_fun
                history.append(fun)
                last_step = step
                break
            step *= contraction
        else:
            return result(4, "Armijo search did not decrease the original nonlinear objective")
    raise RuntimeError("Unreachable nonlinear analysis state")


def nonnegative_analysis_linear_operator(
    reference,
    observation,
    target,
    mass,
    covariance,
    point,
    *,
    observation_hvp=None,
    curvature="exact",
    kkt_tolerance,
    active_margin,
    solve_rtol,
    maxiter,
):
    r"""Линеаризовать выбранную стационарную ветвь неотрицательного анализа.

    Parameters
    ----------
    reference, point : array_like, shape (N,)
        Фоновый центр r и выбранная неотрицательная точка x.
    observation : MapCapabilities
        Карта A из (N,) в (M,) с прогнозом и VJP. Якобиан J собирается M
        сопряжёнными действиями; JVP и явная матрица карты не требуются.
    target : array_like, shape (M,)
        Наблюдения d. Центр, наблюдения и масштабы копируются.
    mass : array_like, shape (N,)
        Положительная диагональ фоновой точности M.
    covariance : array_like, shape (M,) or (M, M)
        Положительные дисперсии либо симметричная положительно определённая R.
    observation_hvp : callable or None, optional
        Для curvature='exact' обязательное действие (zeta,v) ->
        D²<zeta,A(x)>v формы (N,). Callback должен быть привязан к собственной
        неизменяемой линеаризации именно в point. Библиотека не может заморозить
        произвольные внешние переменные его closure. zeta передаётся фиксированным;
        производная R^{-1}(A-d) уже учтена членом J.T@inv(R)@J.
        В режиме 'gauss-newton' callback не вызывается.
    curvature : {'exact', 'gauss-newton'}, optional
        Точный гессиан исходного функционала либо явно объявленное приближение
        без остаточной кривизны. Скрытой замены между ними нет.
    kkt_tolerance : float
        Положительный допуск исходного проекционного градиента и отдельно
        свободного градиента. Свободны ровно координаты point>0.
    active_margin : float
        Неотрицательная граница: на point==0 требуется gradient>active_margin.
        Это численная проверка запаса, а не доказательство строгой дополнительности.
    solve_rtol : float
        Относительный допуск в (0,1) фактического евклидова остатка свободной
        линейной системы. MINRES использует более строгий внутренний rtol.
    maxiter : int
        Положительный предел итераций каждой свободной системы.

    Returns
    -------
    linearization : NonnegativeAnalysisLinearization
        operator формы (N,N+M): [dr,dd] -> K(M@dr+J.T@inv(R)@dd), где
        K=Z@inv(Z.T@H@Z)@Z.T. Транспонированное действие возвращает
        [M@K@bar_x, inv(R)@J@K@bar_x]. metadata хранит исходные остатки и запасы.

    Notes
    -----
    Для exact используется H=M+J.T@inv(R)@J+sum(zeta_i*D²A_i),
    zeta=inv(R)@(A(point)-d). Плотная матрица (N,N) не создаётся.
    При пустом свободном множестве действие нулевое, гессиан не вычисляется.
    Предполагаются гладкая устойчивая активная ветвь, стационарность и
    невырожденность приведённого гессиана. Выполнение численных допусков
    этих предпосылок не доказывает. Это не производная фактически остановленного
    алгоритма и не производная выбора весов, центра, масок или гиперпараметров.

    MINRES допускает симметричный неопределённый гессиан. Успешное решение
    не удостоверяет положительную определённость, невырожденность или минимум.
    При неуспешной проверке остатка возникает ошибка; сдвига, дополнительного
    демпфирования, псевдообратной или запасного решателя нет. Конечные допуски
    дают приближённые действия и не обеспечивают точное сопряжённое тождество.
    """
    validate_map_capabilities(observation, name="observation")
    domain, codomain = observation.domain, observation.codomain
    if len(domain.shape) != 1 or len(codomain.shape) != 1:
        raise ValueError("Analysis linearization requires vector domain and codomain")
    pullback = require_capability(
        observation.vjp, operation="VJP", consumer="analysis linearization"
    )
    if curvature not in ("exact", "gauss-newton"):
        raise ValueError("curvature must be exact or gauss-newton")
    if curvature == "exact" and not callable(observation_hvp):
        raise ValueError("Exact analysis curvature requires an observation_hvp callback")
    r = real_array(reference, name="reference", copy=True)
    x = real_array(point, name="point", copy=True)
    d = real_array(target, name="target", copy=True)
    m = real_array(mass, name="mass", copy=True)
    cov = real_array(covariance, name="covariance", copy=True)
    if (
        r.shape != domain.shape
        or x.shape != domain.shape
        or d.shape != codomain.shape
        or m.shape != domain.shape
        or np.any(m <= 0)
        or np.any(x < 0)
    ):
        raise ValueError("Require r(N), x(N)>=0, d(M) and positive diagonal mass(N)")
    tolerance = real_scalar(kkt_tolerance, name="kkt_tolerance")
    margin = real_scalar(active_margin, name="active_margin")
    relative = real_scalar(solve_rtol, name="solve_rtol")
    if tolerance <= 0 or margin < 0 or not 0 < relative < 1:
        raise ValueError("Require kkt_tolerance>0, active_margin>=0 and 0<solve_rtol<1")
    if type(maxiter) is not int or maxiter < 1:
        raise ValueError("maxiter must be a positive integer")
    factor = _spd_factor(cov, codomain.size, "covariance")

    def inverse_covariance(values):
        if factor.ndim == 1:
            return (values / factor) / factor
        return solve_triangular(
            factor.T,
            solve_triangular(factor, values, lower=True, check_finite=False),
            lower=False,
            check_finite=False,
        )

    prediction = codomain.array(
        observation.predict.predict(x.copy()), name="prediction", copy=True
    )
    matrix = np.empty((codomain.size, domain.size))
    unit = np.zeros(codomain.size)
    for row in range(codomain.size):
        unit[row] = 1.0
        matrix[row] = domain.array(pullback.vjp(x.copy(), unit.copy()), name="observation VJP")
        unit[row] = 0.0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        zeta = inverse_covariance(prediction - d)
        gradient = m * (x - r) + matrix.T @ zeta
    zeta = real_array(zeta, name="observation multipliers", copy=True)
    gradient = real_array(gradient, name="analysis gradient", copy=True)
    free, active = np.flatnonzero(x > 0), np.flatnonzero(x == 0)
    projected_residual = float(np.max(np.abs(np.minimum(x, gradient))))
    root_residual = float(np.max(np.abs(gradient[free]))) if free.size else 0.0
    if projected_residual > tolerance or root_residual > tolerance:
        raise ValueError("Analysis point fails original projected or free stationarity tolerance")
    active_minimum = float(np.min(gradient[active])) if active.size else None
    if active_minimum is not None and active_minimum <= margin:
        raise ValueError("Analysis point has a weak or insufficiently separated active face")
    for values in (r, x, d, m, factor, matrix, zeta):
        values.flags.writeable = False

    def hessian(direction):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            value = m * direction + matrix.T @ inverse_covariance(matrix @ direction)
        if curvature == "exact":
            extra = real_array(
                observation_hvp(zeta.copy(), direction.copy()),
                name="observation Hessian product",
            )
            if extra.shape != domain.shape:
                raise ValueError("Observation Hessian product must have shape (N,)")
            value = value + extra
        return real_array(value, name="analysis Hessian product")

    def reduced_hessian(direction):
        full = np.zeros(domain.size)
        full[free] = direction
        return hessian(full)[free]

    reduced = (
        LinearOperator(
            (free.size, free.size), matvec=reduced_hessian, rmatvec=reduced_hessian, dtype=float
        )
        if free.size
        else None
    )

    def solve_free(right):
        result = np.zeros(domain.size)
        if not free.size:
            return result
        selected = real_array(right[free], name="free linear right-hand side")
        if not np.any(selected):
            return result
        inner_rtol = relative / np.sqrt(free.size)
        if inner_rtol == 0:
            raise ValueError("The requested internal MINRES tolerance is not representable")
        solution, info = minres(reduced, selected, rtol=inner_rtol, maxiter=maxiter, check=True)
        solution = real_array(solution, name="free linear solution")
        residual = real_array(reduced @ solution - selected, name="free linear residual")
        if info != 0 or norm(residual) > relative * norm(selected):
            raise RuntimeError("MINRES failed the requested true relative linear residual")
        result[free] = solution
        return result

    def matvec(direction):
        direction = real_array(direction, name="analysis input direction").reshape(-1)
        if direction.size != domain.size + codomain.size:
            raise ValueError("Analysis input direction must have length N+M")
        if not free.size:
            return np.zeros(domain.size)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            right = m * direction[: domain.size] + matrix.T @ inverse_covariance(
                direction[domain.size :]
            )
        return solve_free(real_array(right, name="analysis derivative right-hand side"))

    def rmatvec(cotangent):
        cotangent = real_array(cotangent, name="analysis output cotangent").reshape(-1)
        if cotangent.size != domain.size:
            raise ValueError("Analysis output cotangent must have length N")
        if not free.size:
            return np.zeros(domain.size + codomain.size)
        pulled = solve_free(cotangent)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            result = np.r_[m * pulled, inverse_covariance(matrix @ pulled)]
        return real_array(result, name="analysis transposed derivative")

    metadata = MappingProxyType(
        {
            "curvature": curvature,
            "projected_gradient_inf": projected_residual,
            "free_root_residual_inf": root_residual,
            "free_min": float(np.min(x[free])) if free.size else None,
            "active_gradient_min": active_minimum,
            "free_indices": tuple(int(index) for index in free),
            "active_indices": tuple(int(index) for index in active),
            "point": tuple(float(value) for value in x),
            "kkt_tolerance": tolerance,
            "active_margin": margin,
            "solve_rtol": relative,
        }
    )
    operator = LinearOperator(
        (domain.size, domain.size + codomain.size), matvec=matvec, rmatvec=rmatvec, dtype=float
    )
    return NonnegativeAnalysisLinearization(operator, metadata)
