"""Гладкая оптимизация функционалов и нелинейных наименьших квадратов."""

from operator import index

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, OptimizeResult, lsq_linear, minimize

from adrkit.contracts import (
    MapCapabilities,
    ScalarCapabilities,
    require_capability,
    validate_scalar_capabilities,
)
from adrkit.spaces import ArraySpace
from .constraints import box_bounds
from ._numeric import real_scalar


def solve_lbfgsb(objective, initial, *, bounds=None, options=None, callback=None):
    """Минимизировать гладкий функционал методом L-BFGS-B.

    Parameters
    ----------
    objective : ScalarCapabilities
        Функционал с координатным cotangent; совместная операция
        используется при наличии.
    initial : array_like
        Конечная начальная точка формы objective.domain.shape.
    bounds : sequence of pairs or None, optional
        Покомпонентные границы в порядке C; None означает отсутствие
        границ.
    options : Mapping or None, optional
        Параметры scipy.optimize.minimize для L-BFGS-B.
    callback : callable or None, optional
        Передаётся без изменения в scipy.optimize.minimize.

    Returns
    -------
    result : scipy.optimize.OptimizeResult
        Оценка x и координатный градиент jac представлены плоскими
        массивами в порядке C.

    Notes
    -----
    Ковектор используется как евклидов градиент координат.
    Единицы аргумента и функционала задаются objective.
    """
    if not isinstance(objective, ScalarCapabilities):
        raise TypeError("objective must declare ScalarCapabilities")
    if not isinstance(objective.domain, ArraySpace):
        raise TypeError("objective domain must be an ArraySpace record")
    derivative = require_capability(
        objective.cotangent, operation="smooth cotangent", consumer="L-BFGS-B"
    )
    for operation, method in (
        (objective.value, "value"),
        (derivative, "cotangent"),
        (objective.value_and_cotangent, "value_and_cotangent"),
    ):
        if operation is not None and not callable(getattr(operation, method, None)):
            raise TypeError(f"L-BFGS-B {method} operation must provide callable {method}()")
    x = objective.domain.array(initial, copy=True)
    checked_bounds = box_bounds(bounds, x.size)

    def function(flat):
        point = objective.domain.array(flat.reshape(objective.domain.shape), copy=True)
        if objective.value_and_cotangent is not None:
            value, gradient = objective.value_and_cotangent.value_and_cotangent(point.copy())
            value = real_scalar(value, name="Smooth optimizer objective")
        else:
            value = real_scalar(
                objective.value.value(point.copy()), name="Smooth optimizer objective"
            )
            gradient = derivative.cotangent(point.copy())
        return value, objective.domain.array(gradient, name="covector", copy=True).ravel()

    return minimize(
        function,
        x.ravel(),
        method="L-BFGS-B",
        jac=True,
        bounds=checked_bounds,
        options=dict(options or {}),
        **({} if callback is None else {"callback": callback}),
    )


def solve_trust_constr(
    objective, initial, *, bounds=None, constraints=(), options=None, callback=None
):
    """Минимизировать гладкий функционал с линейными ограничениями.

    Parameters
    ----------
    objective : ScalarCapabilities
        Функционал с координатным cotangent; совместная операция
        значения и ковектора используется при наличии.
    initial : array_like
        Конечная точка формы objective.domain.shape, допустимая по bounds
        и сохраняемым неравенствам constraints.
    bounds : sequence of pairs or None, optional
        Покомпонентные границы в порядке C; None означает отсутствие
        границ. Они передаются с keep_feasible=True.
    constraints : scipy.optimize.LinearConstraint or sequence, optional
        Одно линейное ограничение или их последовательность. Столбцы
        матриц соответствуют координатам аргумента в порядке C.
        Флаги keep_feasible сохраняются без изменения.
    options : Mapping or None, optional
        Настройки trust-constr для scipy.optimize.minimize.
    callback : callable or None, optional
        Передаётся без изменения в scipy.optimize.minimize.

    Returns
    -------
    result : scipy.optimize.OptimizeResult
        Результат SciPy с плоскими x и grad в порядке C;
        jac содержит матрицы производных ограничений.

    Notes
    -----
    Используются аналитический координатный ковектор и аппроксимация
    гессиана BFGS. Аргументы функционала не проецируются и не обрезаются.
    keep_feasible сохраняет неравенства; допустимость равенств на всех
    промежуточных пробах SciPy не гарантирует.
    """
    validate_scalar_capabilities(objective, name="trust-constr objective")
    derivative = require_capability(
        objective.cotangent, operation="smooth cotangent", consumer="trust-constr"
    )
    x = objective.domain.array(initial, copy=True)
    flat = x.ravel()
    checked_bounds = None
    if bounds is not None:
        lower, upper = np.asarray(box_bounds(bounds, x.size), dtype=float).T
        if np.any(flat < lower) or np.any(flat > upper):
            raise ValueError("Initial point violates bounds")
        checked_bounds = Bounds(lower, upper, keep_feasible=True)
    if isinstance(constraints, LinearConstraint):
        constraints = (constraints,)
    elif isinstance(constraints, (tuple, list)):
        constraints = tuple(constraints)
    else:
        raise TypeError("constraints must contain LinearConstraint objects")
    for constraint in constraints:
        if not isinstance(constraint, LinearConstraint):
            raise TypeError("constraints must contain LinearConstraint objects")
        if constraint.A.shape[1] != flat.size:
            raise ValueError("Linear constraint columns must match the domain size")
        if (
            np.isnan(constraint.lb).any()
            or np.isnan(constraint.ub).any()
            or np.isposinf(constraint.lb).any()
            or np.isneginf(constraint.ub).any()
            or np.any(constraint.lb > constraint.ub)
        ):
            raise ValueError("Invalid linear constraint bounds")
        values = np.asarray(constraint.A @ flat).ravel()
        if not np.isfinite(values).all():
            raise ValueError("Linear constraint values must be finite at the initial point")
        required = constraint.keep_feasible & (constraint.lb != constraint.ub)
        if np.any(required & ((values < constraint.lb) | (values > constraint.ub))):
            raise ValueError("Initial point violates a keep_feasible linear constraint")

    def function(values):
        point = objective.domain.array(values.reshape(objective.domain.shape), copy=True)
        if objective.value_and_cotangent is not None:
            value, gradient = objective.value_and_cotangent.value_and_cotangent(point.copy())
        else:
            value = objective.value.value(point.copy())
            gradient = derivative.cotangent(point.copy())
        return (
            real_scalar(value, name="Smooth optimizer objective"),
            objective.domain.array(gradient, name="covector", copy=True).ravel(),
        )

    return minimize(
        function,
        flat,
        method="trust-constr",
        jac=True,
        bounds=checked_bounds,
        constraints=constraints,
        options=dict(options or {}),
        **({} if callback is None else {"callback": callback}),
    )


def solve_damped_gauss_newton(residual, initial, *, bounds=None, options=None, callback=None):
    r"""Минимизировать половину квадрата нормы невязки с границами.

    Parameters
    ----------
    residual : MapCapabilities
        Отображение невязки с predict и JVP между ArraySpace.
        Вес и нормировка должны входить в невязку и её производную.
    initial : array_like
        Конечная допустимая точка формы residual.domain.shape.
    bounds : sequence of pairs or None, optional
        Границы координат в порядке C. Равные границы фиксируют координату.
    options : Mapping or None, optional
        gtol (1e-8) — допуск отображения проекционного градиента;
        maxiter (200) — предел принятых шагов; armijo_coefficient (1e-4)
        принадлежит (0, 0.5); backtrack_factor (0.5) принадлежит (0, 1);
        max_backtracks (30) — предел уменьшений длины шага;
        qp_tolerance (1e-12) — абсолютный допуск KKT подзадачи;
        qp_maxiter (None) — предел итераций BVLS, None задаёт выбор SciPy.
        damping_rule ("objective") задаёт сдвиг J; "projected-gradient"
        задаёт min(J, optimality). Функционал J при этом не меняется.
        gtol допускает ноль; неизвестные настройки отклоняются.
    callback : callable or None, optional
        callback(result) вызывается в начальной и каждой принятой точке.
        Получает независимый OptimizeResult; StopIteration останавливает метод.

    Returns
    -------
    result : scipy.optimize.OptimizeResult
        x и градиент jac — плоские массивы в порядке C; fun — J,
        residual — вектор невязки, optimality — бесконечная норма
        отображения проекционного градиента. nit считает принятые шаги,
        nfev — прогнозы, njev — сборки якобиана, njvp — действия JVP.
        step_size и damping относятся к последней подзадаче;
        damping_rule содержит выбранное правило сдвига.
        status: 1 — нулевая невязка; 2 — стационарность; 0 — maxiter;
        -1 — не подтверждён шаг подзадачи; -2 — не выполнено Армихо;
        -3 — численный застой; 99 — остановка callback.
        success истинен только для status 1 или 2.

    Notes
    -----
    При :math:`J=\frac12\|r\|^2>0` матрица :math:`A=Dr(x)` собирается
    по столбцам JVP. Метод BVLS с плотным QR/SVD решает

    .. math::

        \min_{l-x\le p\le u-x}
        \tfrac12\|Ap+r\|^2+\tfrac\mu2\|p\|^2.

    По умолчанию :math:`\mu=J`; при projected-gradient используется
    :math:`\mu=\min(J,\operatorname{optimality})`. До подзадачи обе величины
    положительны. Для её точного ненулевого решения вариационное неравенство
    даёт :math:`\nabla J^Tp\le-\|Ap\|^2-\mu\|p\|^2<0`.
    Затем длина шага выбирается по условию Армихо для J, начиная с единицы.
    Допуск проверки KKT равен qp_tolerance плюс оценка округления плотных
    произведений; вариационное неравенство проверяется с соответствующей
    ошибкой. Проекция устраняет только округление допустимого шага на границе.
    Постоянный диагональный сдвиг и априорный штраф не добавляются.
    Стационарность не доказывает нулевую невязку или глобальность минимума.
    """
    if not isinstance(residual, MapCapabilities):
        raise TypeError("residual must declare MapCapabilities")
    if not isinstance(residual.domain, ArraySpace) or not isinstance(
        residual.codomain, ArraySpace
    ):
        raise TypeError("residual spaces must be ArraySpace records")
    derivative = require_capability(residual.jvp, operation="JVP", consumer="damped Gauss-Newton")
    for operation, method in ((residual.predict, "predict"), (derivative, "jvp")):
        if not callable(getattr(operation, method, None)):
            raise TypeError(f"Residual {method} operation must provide callable {method}()")
    if callback is not None and not callable(callback):
        raise TypeError("callback must be callable or None")
    settings = dict(
        gtol=1e-8,
        maxiter=200,
        armijo_coefficient=1e-4,
        backtrack_factor=0.5,
        max_backtracks=30,
        qp_tolerance=1e-12,
        qp_maxiter=None,
        damping_rule="objective",
    )
    supplied = dict(options or {})
    if set(supplied) - set(settings):
        raise ValueError(
            f"Unknown damped Gauss-Newton options: {sorted(set(supplied)-set(settings))}"
        )
    settings.update(supplied)
    if settings["damping_rule"] not in ("objective", "projected-gradient"):
        raise ValueError("damping_rule must be 'objective' or 'projected-gradient'")
    for name in ("gtol", "armijo_coefficient", "backtrack_factor", "qp_tolerance"):
        settings[name] = real_scalar(settings[name], name=name)
    if settings["gtol"] < 0 or settings["qp_tolerance"] <= 0:
        raise ValueError("gtol must be nonnegative and qp_tolerance positive")
    if not 0 < settings["armijo_coefficient"] < 0.5 or not 0 < settings["backtrack_factor"] < 1:
        raise ValueError("armijo_coefficient must lie in (0, .5), backtrack_factor in (0, 1)")
    for name in ("maxiter", "max_backtracks", "qp_maxiter"):
        if name == "qp_maxiter" and settings[name] is None:
            continue
        try:
            count = index(settings[name])
        except TypeError as error:
            raise ValueError(f"{name} must be an integer") from error
        if isinstance(settings[name], (bool, np.bool_)) or count < (
            1 if name == "qp_maxiter" else 0
        ):
            raise ValueError(f"Invalid iteration count: {name}")
        settings[name] = count

    domain, codomain = residual.domain, residual.codomain
    x = np.array(domain.array(initial, copy=True), dtype=float).ravel()
    pairs = box_bounds(bounds, x.size)
    lower = np.array([-np.inf if a is None else a for a, _ in pairs])
    upper = np.array([np.inf if b is None else b for _, b in pairs])
    if np.any(x < lower) or np.any(x > upper):
        raise ValueError("initial must satisfy the coefficient bounds")
    free = lower < upper
    nfev = njev = njvp = 0
    step_size = damping = 0.0
    qp_optimality = qp_error_bound = directional_derivative = step_curvature = None

    def evaluate(point):
        nonlocal nfev
        nfev += 1
        values = residual.predict.predict(domain.array(point.reshape(domain.shape), copy=True))
        values = np.asarray(
            codomain.array(values, name="residual", copy=True), dtype=float
        ).ravel()
        value = real_scalar(0.5 * (values @ values), name="Least-squares objective")
        return values, value

    def linearize(point, values):
        nonlocal njev, njvp
        njev += 1
        columns = []
        for j in range(domain.size):
            direction = np.zeros(domain.size)
            direction[j] = 1.0
            value = derivative.jvp(
                domain.array(point.reshape(domain.shape), copy=True),
                domain.array(direction.reshape(domain.shape), copy=True),
            )
            columns.append(
                np.asarray(
                    codomain.array(value, name="residual JVP", copy=True), dtype=float
                ).ravel()
            )
            njvp += 1
        matrix = np.column_stack(columns)
        return matrix, matrix.T @ values

    def result(status, message):
        return OptimizeResult(
            x=x.copy(),
            fun=fun,
            jac=gradient.copy(),
            residual=r.copy(),
            optimality=optimality,
            nit=iteration,
            nfev=nfev,
            njev=njev,
            njvp=njvp,
            success=status in (1, 2),
            status=status,
            message=message,
            step_size=step_size,
            damping=damping,
            damping_rule=settings["damping_rule"],
            qp_optimality=qp_optimality,
            qp_error_bound=qp_error_bound,
            directional_derivative=directional_derivative,
            step_curvature=step_curvature,
        )

    r, fun = evaluate(x)
    for iteration in range(settings["maxiter"] + 1):
        if np.all(r == 0):
            gradient = np.zeros_like(x)
        else:
            matrix, gradient = linearize(x, r)
        projected = np.where(
            gradient > 0, np.minimum(gradient, x - lower), np.maximum(gradient, x - upper)
        )
        optimality = float(np.linalg.norm(projected, np.inf))
        if callback is not None:
            try:
                callback(result(0, "Current accepted point"))
            except StopIteration:
                return result(99, "Stopped by callback")
        if np.all(r == 0):
            return result(1, "Zero residual")
        if optimality <= settings["gtol"]:
            return result(2, "Projected-gradient stationarity; residual may be nonzero")
        if iteration == settings["maxiter"]:
            return result(0, "Maximum number of iterations reached")
        if fun == 0:
            return result(-3, "Positive residual cost underflowed; rescale the residual")

        step_size = 0.0
        qp_optimality = qp_error_bound = directional_derivative = step_curvature = None
        damping = fun if settings["damping_rule"] == "objective" else min(fun, optimality)
        augmented = np.vstack((matrix[:, free], np.sqrt(damping) * np.eye(np.count_nonzero(free))))
        target = np.r_[-r, np.zeros(np.count_nonzero(free))]
        step_lower, step_upper = lower[free] - x[free], upper[free] - x[free]
        qp = lsq_linear(
            augmented,
            target,
            bounds=(step_lower, step_upper),
            method="bvls",
            lsq_solver="exact",
            tol=settings["qp_tolerance"],
            max_iter=settings["qp_maxiter"],
        )
        step = qp.x
        if (
            not qp.success
            or not np.isfinite(step).all()
            or np.any(step < step_lower)
            or np.any(step > step_upper)
        ):
            return result(
                -1,
                "The bounded least-squares subproblem did not produce a feasible converged step",
            )
        direction = np.zeros_like(x)
        direction[free] = step
        qp_gradient = augmented.T @ (augmented @ step - target)
        defect = qp_gradient.copy()
        at_lower, at_upper = step == step_lower, step == step_upper
        defect[at_lower] = np.minimum(defect[at_lower], 0.0)
        defect[at_upper] = np.maximum(defect[at_upper], 0.0)
        qp_optimality = float(np.linalg.norm(defect, np.inf))
        rounding_factor = (sum(augmented.shape) + 3) * np.finfo(float).eps
        rounding_factor /= 1.0 - rounding_factor
        product_scale = np.abs(augmented).T @ (np.abs(augmented) @ np.abs(step) + np.abs(target))
        qp_error_bound = settings["qp_tolerance"] + rounding_factor * np.linalg.norm(
            product_scale, np.inf
        )
        directional_derivative = float(gradient @ direction)
        mapped = matrix @ direction
        step_curvature = float(mapped @ mapped + damping * (direction @ direction))
        vi_error = qp_error_bound * np.sum(np.abs(step)) + rounding_factor * (
            np.sum(np.abs(gradient * direction)) + step_curvature
        )
        if (
            not np.isfinite(
                [qp_optimality, qp_error_bound, directional_derivative, step_curvature, vi_error]
            ).all()
            or qp_optimality > qp_error_bound
            or directional_derivative >= 0
            or directional_derivative + step_curvature > vi_error
        ):
            return result(
                -1, "The bounded least-squares step failed the optimality or descent check"
            )

        endpoint = x + direction
        if not np.isfinite(endpoint).all():
            return result(
                -3, "The next point is not representable; rescale the parameter coordinates"
            )
        feasibility_error = np.maximum(lower - endpoint, endpoint - upper)
        addition_error = (
            np.spacing(np.abs(x)) + np.spacing(np.abs(direction)) + np.spacing(np.abs(endpoint))
        )
        if np.any(feasibility_error > addition_error):
            return result(-1, "The bounded step did not produce a numerically feasible endpoint")
        # The subproblem is constrained; only addition roundoff is removed here.
        endpoint = np.clip(endpoint, lower, upper)
        step_size = 1.0
        for backtrack in range(settings["max_backtracks"] + 1):
            trial = (
                endpoint if step_size == 1.0 else np.clip(x + step_size * direction, lower, upper)
            )
            if np.array_equal(trial, x):
                return result(
                    -3, "The descent step is no longer representable at the current point"
                )
            if gradient @ (trial - x) < 0:
                trial_r, trial_fun = evaluate(trial)
                if (
                    trial_fun
                    <= fun + settings["armijo_coefficient"] * step_size * directional_derivative
                ):
                    x, r, fun = trial, trial_r, trial_fun
                    break
            if backtrack == settings["max_backtracks"]:
                return result(-2, "Armijo backtracking exhausted")
            step_size *= settings["backtrack_factor"]
