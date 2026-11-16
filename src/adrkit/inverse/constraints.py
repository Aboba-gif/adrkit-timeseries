"""Покомпонентные границы, проекция и анализ линейных наблюдений."""

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import OptimizeResult, linprog

from ._numeric import real_array, real_scalar
from .linalg import _spd_factor


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
