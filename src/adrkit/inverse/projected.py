"""Подгонка неотрицательных коэффициентов с квадратичным штрафом."""
from typing import Protocol

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import nnls
from ._numeric import real_array, real_scalar, symmetric_psd_matrix


class FitPrediction(Protocol):
    """Прогноз и координатное транспонирование его производной."""

    def predict(self, point: np.ndarray) -> np.ndarray:
        """Вычислить прогноз по коэффициентам.

        Parameters
        ----------
        point : ndarray, shape (n,)
            Коэффициенты модели.

        Returns
        -------
        prediction : ndarray, shape (m,)
            Прогноз в единицах наблюдений.
        """
        ...

    def vjp(self, point: np.ndarray, cotangent: np.ndarray) -> np.ndarray:
        """Перенести ковектор наблюдений к коэффициентам.

        Parameters
        ----------
        point : ndarray, shape (n,)
            Точка коэффициентов.
        cotangent : ndarray, shape (m,)
            Координатный ковектор наблюдений.

        Returns
        -------
        covector : ndarray, shape (n,)
            Координатный ковектор коэффициентов.
        """
        ...


class FitMetric(Protocol):
    """Операции обратной ковариации и согласованного нормирования."""

    def whiten(self, values: np.ndarray) -> np.ndarray:
        """Привести невязку к единичной ковариации.

        Parameters
        ----------
        values : ndarray, shape (m,)
            Невязка в координатах наблюдений.

        Returns
        -------
        whitened : ndarray, shape (m,)
            Невязка в координатах с единичной ковариацией.
        """
        ...

    def whiten_matrix(self, values: np.ndarray) -> np.ndarray:
        """Преобразовать строки матрицы к единичной ковариации.

        Parameters
        ----------
        values : ndarray, shape (m, n)
            Матрица с наблюдениями по строкам.

        Returns
        -------
        whitened : ndarray, shape (m, n)
            Матрица после того же преобразования, что в whiten.
        """
        ...

    def apply_precision(self, values: np.ndarray) -> np.ndarray:
        """Применить обратную ковариацию к невязке.

        Parameters
        ----------
        values : ndarray, shape (m,)
            Координатная невязка наблюдений.

        Returns
        -------
        weighted : ndarray, shape (m,)
            Действие фиксированной обратной ковариации.
        """
        ...


def fit(prediction, observations, metric, gram, alpha, jacobian, offset,
        *, tolerance=1e-6, max_iterations=80, initial_point=None,
        active_tolerance=1e-10,
        nnls_max_iterations=1000, max_backtracks=24,
        backtrack_factor=0.5, armijo_coefficient=1e-4):
    r"""Подогнать неотрицательные коэффициенты с квадратичным штрафом.

    Parameters
    ----------
    prediction : FitPrediction
        Прогноз и его координатный VJP по коэффициентам.
    observations : ndarray, shape (m,)
        Конечные вещественные целевые наблюдения.
    metric : FitMetric
        Фиксированная обратная ковариация с согласованным whiten.
    gram : ndarray, shape (n, n)
        Конечная вещественная симметричная положительно полуопределённая
        матрица штрафа; n должно быть положительным.
    alpha : float
        Конечный вещественный неотрицательный вес штрафа.
    jacobian : ndarray, shape (m, n)
        Конечная вещественная матрица аффинной модели начального приближения.
    offset : ndarray, shape (m,)
        Конечный вещественный свободный член аффинной модели
        в единицах наблюдений.

    Other Parameters
    ----------------
    tolerance : float, optional
        Положительный допуск локальной стационарности и
        дополняемости.
    max_iterations : int, optional
        Неотрицательный предел обновлений коэффициентов.
    initial_point : array_like, shape (n,), or None, optional
        Конечная неотрицательная начальная точка; при None строится
        по аффинной модели.
    active_tolerance : float, optional
        Неотрицательный порог активной границы коэффициента.
    nnls_max_iterations : int, optional
        Положительный предел итераций аффинной неотрицательной
        подзадачи.
    max_backtracks : int, optional
        Положительное число проб длины шага.
    backtrack_factor : float, optional
        Множитель длины шага строго между нулём и единицей.
    armijo_coefficient : float, optional
        Коэффициент достаточного убывания строго между нулём и
        единицей.

    Returns
    -------
    result : dict
        Коэффициенты формы (n,), прогноз формы (m,), статус и
        показатели остановки. residual_norm — ковариационная норма
        невязки, penalty_norm — норма штрафа без веса alpha.

    Notes
    -----
    Минимизируется функционал при неотрицательных коэффициентах:

    .. math::

        J(a)=\frac12(F(a)-y)^\mathsf{T}W(F(a)-y)
        +\frac{\alpha}{2}a^\mathsf{T}Ga,\qquad a\geq0.

    Здесь :math:`F` — prediction, :math:`y` — observations,
    :math:`W` — обратная ковариация metric, :math:`G` — gram.
    Фиксированная матрица квадратичной подзадачи должна быть симметричной
    положительно определённой:

    .. math::

        H=J_{\mathrm{aff}}^\mathsf{T}WJ_{\mathrm{aff}}+\alpha G.

    Здесь :math:`J_{\mathrm{aff}}` — входная `jacobian`;
    она не пересчитывается по текущей точке нелинейного прогноза.
    Начальная точка каждого вызова строится независимо;
    заданный initial_point копируется. Статус accepted относится к локальным
    условиям остановки, а не к глобальной оптимальности.
    Численные входы проверяются до вызовов metric и prediction.
    Проверка симметрии и полуопределённости G использует те же допуски,
    что QuadraticPenalty; исходная матрица не симметризуется и не проецируется.
    Проверяются собственные значения симметричной части G,
    определяющей квадратичную форму.
    Только для вычисления penalty_norm отрицательная квадратичная форма
    заменяется нулём, если её модуль не превышает допуск собственных значений
    G, умноженный на квадрат евклидовой нормы коэффициентов.
    """
    parameters = {}
    for name, value in (("tolerance", tolerance), ("active_tolerance", active_tolerance),
                        ("backtrack_factor", backtrack_factor),
                        ("armijo_coefficient", armijo_coefficient)):
        parameters[name] = real_scalar(value, name=name)
        if parameters[name] < 0 or (name in ("tolerance", "backtrack_factor", "armijo_coefficient")
                                    and parameters[name] == 0):
            raise ValueError(f"{name} must be {'positive' if name == 'tolerance' else 'nonnegative'}")
    tolerance, active_tolerance = (
        parameters[name] for name in ("tolerance", "active_tolerance"))
    backtrack_factor, armijo_coefficient = (
        parameters[name] for name in ("backtrack_factor", "armijo_coefficient"))
    if backtrack_factor >= 1 or armijo_coefficient >= 1:
        raise ValueError("backtrack_factor and armijo_coefficient must lie strictly between zero and one")
    for name, value, minimum in (("max_iterations", max_iterations, 0),
                                ("nnls_max_iterations", nnls_max_iterations, 1),
                                ("max_backtracks", max_backtracks, 1)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or value < minimum):
            raise ValueError(f"{name} must be an integer >= {minimum}")
    alpha = real_scalar(alpha, name="alpha")
    if alpha < 0:
        raise ValueError("alpha must be nonnegative")
    observations = real_array(observations, name="observations", copy=True)
    jacobian = real_array(jacobian, name="jacobian", copy=True)
    offset = real_array(offset, name="offset", copy=True)
    if observations.ndim != 1:
        raise ValueError("observations must be a vector")
    if jacobian.ndim != 2 or jacobian.shape[0] != observations.size or jacobian.shape[1] == 0:
        raise ValueError("jacobian must have shape (observations.size, n) with n > 0")
    if offset.shape != observations.shape:
        raise ValueError("offset must have the same shape as observations")
    size = jacobian.shape[1]
    gram, gram_tolerance = symmetric_psd_matrix(gram, size, name="gram", copy=True)
    if initial_point is not None:
        point = real_array(initial_point, name="initial_point", copy=True)
        if point.shape != (size,) or np.any(point < 0):
            raise ValueError("initial_point must be a nonnegative coefficient vector of shape (n,)")
    white = metric.whiten_matrix(jacobian)
    hessian = white.T @ white + alpha*gram
    root = cholesky(hessian, lower=False)
    if initial_point is None:
        target = white.T @ metric.whiten(observations-offset)
        initial_rhs = solve_triangular(root.T, target, lower=True)
        point = nnls(root, initial_rhs, maxiter=nnls_max_iterations)[0]
    status = "iteration_limit"
    for iteration in range(max_iterations+1):
        predicted = prediction.predict(point)
        residual = predicted-observations
        precision = metric.apply_precision(residual)
        data = .5*float(residual @ precision)
        penalty = .5*alpha*float(point @ gram @ point)
        gd = prediction.vjp(point, precision)
        gp = alpha*(gram @ point)
        gradient = gd+gp
        projected = np.where(point <= active_tolerance, np.minimum(gradient, 0), gradient)
        scale = max(1., np.max(np.abs(gd)), np.max(np.abs(gp)))
        stationarity = float(np.max(np.abs(projected))/scale)
        multipliers = np.where(point <= active_tolerance, np.maximum(gradient, 0), 0)
        complementarity = float(np.max(np.abs(point*multipliers))/max(1., data+penalty))
        if max(stationarity, complementarity) <= tolerance:
            status = "accepted"
            break
        if iteration == max_iterations:
            break
        rhs = solve_triangular(root.T, hessian @ point-gradient, lower=True)
        destination = nnls(root, rhs, maxiter=nnls_max_iterations)[0]
        direction = destination-point
        slope = float(gradient @ direction)
        if slope >= 0:
            status = "non_descent"
            break
        for backtrack in range(max_backtracks):
            fraction = backtrack_factor**backtrack
            trial = point+fraction*direction
            r = prediction.predict(trial)-observations
            objective = .5*float(r @ metric.apply_precision(r))+ .5*alpha*float(trial @ gram @ trial)
            if objective <= data+penalty+armijo_coefficient*fraction*slope:
                point = trial
                break
        else:
            status = "line_search_failed"
            break
    # После отклонённого поиска шага в сохранённом состоянии может оставаться пробная точка.
    predicted = prediction.predict(point)
    residual = predicted-observations
    penalty_squared = float(point @ gram @ point)
    if penalty_squared < 0:
        roundoff_bound = gram_tolerance*float(point @ point)
        if penalty_squared < -roundoff_bound:
            raise ValueError("Penalty quadratic form is negative beyond the Gram tolerance")
        penalty_squared = 0.
    return dict(alpha=float(alpha), status=status,
        initialization='affine_nnls' if initial_point is None else 'explicit_point',
        accepted=bool(status == "accepted"),
        coefficients=point.copy(), prediction=predicted.copy(),
        residual_norm=float(np.linalg.norm(metric.whiten(residual))),
        penalty_norm=float(np.sqrt(penalty_squared)),
        stationarity=stationarity, complementarity=complementarity,
        iterations=iteration)


def lcurve_corner(path):
    """Выбрать максимальную положительную кривизну L-кривой.

    Parameters
    ----------
    path : sequence of Mapping
        Принятые результаты с положительными конечными alpha,
        residual_norm и penalty_norm; alpha строго возрастает.

    Returns
    -------
    index : int or None
        Индекс выбранной точки. None — менее пяти точек или
        отсутствие допустимой положительной кривизны.

    Notes
    -----
    Кривизна считается в логарифмах нормы невязки и нормы штрафа.
    Два отсчёта с каждого края исключаются. Выбирается положительный
    поворот против часовой стрелки.
    Кандидаты с численно неразрешимой касательной исключаются.
    Порог учитывает округление логарифмов норм и соседние шаги
    логарифмической сетки `alpha`.
    """
    if len(path) < 5 or any(not isinstance(row.get("accepted"), (bool, np.bool_))
                            or not row["accepted"] for row in path):
        return None
    values = np.array([[r["alpha"], r["residual_norm"], r["penalty_norm"]]
                       for r in path], dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        return None
    t, x, y = np.log(values).T
    if np.any(np.diff(t) <= 0):
        return None
    spacing = np.diff(t)

    def derivative(z):
        # Разности исключают вычитание больших абсолютных логарифмов на неравномерной сетке alpha;
        # постоянный участок даёт точно нулевой наклон.
        slopes = np.diff(z)/spacing
        result = np.empty_like(z)
        result[0], result[-1] = slopes[0], slopes[-1]
        result[1:-1] = (spacing[1:]*slopes[:-1]+spacing[:-1]*slopes[1:])/(spacing[:-1]+spacing[1:])
        return result

    dx, dy = derivative(x), derivative(y)
    # Положительная кривизна соответствует повороту против часовой стрелки
    # в координатах (логарифм нормы невязки, логарифм нормы штрафа).
    speed = (dx*dx+dy*dy)**1.5
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        curvature = (dx*derivative(dy)-dy*derivative(dx))/speed
    candidates = np.arange(2, len(path)-2)
    # Порог учитывает округление логарифмов и его усиление
    # на малых шагах сетки.
    local_spacing = np.minimum(np.r_[spacing[0], spacing], np.r_[spacing, spacing[-1]])
    resolution = 32*np.finfo(float).eps*np.maximum(1., np.maximum(abs(x), abs(y)))/local_spacing
    resolved = np.hypot(dx, dy) > resolution
    candidates = candidates[np.isfinite(curvature[candidates]) &
                            (curvature[candidates] > 0) & resolved[candidates]]
    return int(candidates[np.argmax(curvature[candidates])]) if candidates.size else None
