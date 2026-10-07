"""Неотрицательная регуляризация Тихонова для линейной модели."""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_triangular, qr
from scipy.optimize import lsq_linear
from . import linalg
from ._numeric import real_array, real_scalar


@dataclass
class PositiveResult:
    """Результат неотрицательной регуляризации Тихонова.

    Parameters
    ----------
    theta : ndarray
        Оценённый вектор коэффициентов.
    objective : float
        Значение функционала с весом регуляризации.
    residual : float
        Ковариационная норма невязки.
    penalty : float
        Норма отклонения от центра в метрике штрафа, без веса alpha.
    status : str
        Статус численного решения.
    success : bool
        Выполнение критериев остановки, условий Каруша—Куна—Таккера (KKT)
        и неотрицательности с заданными численными допусками.
    kkt_relative, feasibility : float
        Относительная невязка KKT и нарушение неотрицательности.
    iterations : int
        Число итераций ограниченного решения.
    """
    theta: np.ndarray
    objective: float
    residual: float
    penalty: float
    status: str
    success: bool
    kkt_relative: float
    feasibility: float
    iterations: int


class PositiveTikhonov:
    r"""Неотрицательная регуляризация фиксированной линейной модели.

    Parameters
    ----------
    A : array_like, shape (m, n)
        Конечная матрица линейного прогноза; n положительно.
    Sigma : array_like, shape (m, m) or (m,)
        Положительно определённая ковариация либо положительная
        диагональ.
    M : array_like, shape (n, n) or (n,)
        Положительно определённая матрица штрафа либо положительная
        диагональ.
    alpha : float
        Конечный положительный вес регуляризации.
    theta0 : array_like, shape (n,)
        Конечный центр штрафа в единицах коэффициентов.
    kkt_tolerance : float, optional
        Конечный положительный допуск относительных условий KKT.
    solve_tolerance : float, optional
        Конечный положительный допуск ограниченного решения.
    feasibility_tolerance : float, optional
        Конечный неотрицательный допуск нарушения границ.
    max_iterations : int or None, optional
        Положительный предел итераций; при None выбирается по числу
        коэффициентов.

    Notes
    -----
    Минимизируется функционал

    .. math::

        J(\theta)=\frac12\|A\theta-y\|_{\Sigma^{-1}}^2
        +\frac{\alpha}{2}\|\theta-\theta_0\|_M^2,
        \qquad \theta\geq0.

    Здесь :math:`y` — наблюдения, переданные в `solve`,
    :math:`\theta` — коэффициенты, :math:`\theta_0` — `theta0`.
    Матрицы :math:`\Sigma,M` и вес :math:`\alpha` заданы аргументами
    конструктора. Для симметричной положительно определённой матрицы
    :math:`Q` обозначение :math:`\|z\|_Q^2` означает
    :math:`z^\mathsf{T}Qz`.
    Матрицы и их разложения используются повторно для разных наблюдений.
    Успех результата означает выполнение численных условий остановки
    и допустимости с заданными допусками.
    """

    def __init__(self, A, Sigma, M, alpha, theta0, kkt_tolerance=1e-7, *,
                 solve_tolerance=1e-11, feasibility_tolerance=1e-10,
                 max_iterations=None):
        self.A = real_array(A, name="A", copy=True)
        self.theta0 = real_array(theta0, name="theta0", copy=True)
        alpha = real_scalar(alpha, name="alpha")
        kkt_tolerance = real_scalar(kkt_tolerance, name="KKT tolerance")
        solve_tolerance = real_scalar(solve_tolerance, name="Solve tolerance")
        feasibility_tolerance = real_scalar(feasibility_tolerance, name="Feasibility tolerance")
        if solve_tolerance <= 0 or feasibility_tolerance < 0:
            raise ValueError("Solve tolerance must be positive and feasibility tolerance nonnegative")
        if max_iterations is not None and (
            isinstance(max_iterations, (bool, np.bool_))
            or not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1
        ):
            raise ValueError("max_iterations must be a positive integer or None")
        if self.A.ndim != 2 or self.A.shape[1] == 0 or not np.isfinite(self.A).all():
            raise ValueError("A must be finite with at least one coefficient")
        if self.theta0.shape != (self.A.shape[1],) or not np.isfinite(self.theta0).all():
            raise ValueError("theta0 has incompatible shape or nonfinite entries")
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        if kkt_tolerance <= 0:
            raise ValueError("Positive KKT tolerance is required")
        self.alpha, self.kkt_tolerance = alpha, kkt_tolerance
        self.solve_tolerance, self.feasibility_tolerance = solve_tolerance, feasibility_tolerance
        self.max_iterations = (max(1000, 5*self.A.shape[1]) if max_iterations is None
                               else int(max_iterations))
        self.sigma_factor = linalg._spd_factor(Sigma, self.A.shape[0], "Sigma")
        self.R = linalg.gram_root(M, self.A.shape[1])
        self.B = self._whiten(self.A)
        self.augmented = np.vstack((self.B, np.sqrt(alpha)*self.R))
        # Положительная диагональная замена переменных сохраняет покомпонентную неотрицательность.
        self.column_scale = 1/np.linalg.norm(self.augmented, axis=0)
        self.C = self.augmented*self.column_scale
        self.Q, self.U = qr(self.C, mode="economic")
        self.penalty_target = np.sqrt(alpha)*self.R@self.theta0

    def _whiten(self, x):
        if self.sigma_factor.ndim == 1:
            return x/self.sigma_factor[:, None] if x.ndim == 2 else x/self.sigma_factor
        return solve_triangular(self.sigma_factor, x, lower=True)

    def solve(self, y):
        """Подогнать наблюдения с неотрицательными коэффициентами.

        Parameters
        ----------
        y : array_like, shape (m,)
            Конечные наблюдения в единицах линейного прогноза.

        Returns
        -------
        result : PositiveResult
            Оценка и показатели численной допустимости; KKT проверяется в
            масштабированных координатах.
        """
        y = real_array(y, name="y", copy=True)
        if y.shape != (self.A.shape[0],) or not np.isfinite(y).all():
            raise ValueError("Finite y with the required observation length is required")
        white_y = self._whiten(y)
        d = np.r_[white_y, self.penalty_target]
        rhs = self.C.T@d
        z = solve_triangular(self.U, self.Q.T@d)
        status, iterations = "unconstrained_feasible", 0
        if np.any(z < 0):
            result = lsq_linear(self.C, d, bounds=(0, np.inf), method="bvls",
                                tol=self.solve_tolerance, max_iter=self.max_iterations)
            z, iterations = result.x, result.nit
            status = f"bvls_{result.status}"
            optimizer_success = bool(result.success)
        else:
            optimizer_success = True
        theta = self.column_scale*z
        grad = self.C.T@(self.C@z-d)
        projected = z-np.maximum(0, z-grad)
        kkt = np.linalg.norm(projected, np.inf)/max(1, np.linalg.norm(rhs, np.inf))
        feasibility = max(0.0, -float(theta.min()))
        residual2 = float(np.sum((self.B@theta-white_y)**2))
        penalty2 = float(np.sum((self.R@(theta-self.theta0))**2))
        success = (optimizer_success and feasibility <= self.feasibility_tolerance
                   and kkt <= self.kkt_tolerance)
        if not success:
            status += "_failed_contract"
        return PositiveResult(theta, .5*(residual2+self.alpha*penalty2),
                              np.sqrt(residual2), np.sqrt(penalty2), status, success,
                              float(kkt), feasibility, int(iterations))


def solve_positive_tikhonov(A, y, Sigma, M, alpha, theta0):
    """Решить неотрицательную линейную задачу Тихонова.

    Parameters
    ----------
    A : array_like, shape (m, n)
        Конечная матрица линейного прогноза; n положительно.
    y : array_like, shape (m,)
        Конечные наблюдения в единицах линейного прогноза.
    Sigma : array_like, shape (m, m) or (m,)
        Положительно определённая ковариация либо положительная
        диагональ.
    M : array_like, shape (n, n) or (n,)
        Положительно определённая матрица штрафа либо положительная
        диагональ.
    alpha : float
        Конечный положительный вес регуляризации.
    theta0 : array_like, shape (n,)
        Конечный центр штрафа в единицах коэффициентов.

    Returns
    -------
    result : PositiveResult
        Оценка и показатели допустимости со стандартными настройками
        решателя.
    """
    return PositiveTikhonov(A, Sigma, M, alpha, theta0).solve(y)

