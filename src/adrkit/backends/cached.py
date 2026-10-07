"""ADR с постоянными коэффициентами методом неподвижной точки."""
import numpy as np
from scipy.sparse.linalg import splu
from .bounded import BoundedSolver, StateTrajectory, _integer_count
from adrkit.coefficients import is_scalar_constant


class CachedSolver(BoundedSolver):
    r"""Решатель ADR с постоянными коэффициентами и реакцией tanh.

    Parameters
    ----------
    *args
        Позиционные параметры BoundedSolver.
    residual_tolerance : float, optional
        Положительный допуск относительной невязки.
    max_fixed_point_iterations : int, optional
        Положительный предел обновлений метода неподвижной точки.
    **kwargs
        Остальные параметры BoundedSolver для постоянных
        коэффициентов.

    See Also
    --------
    adrkit.backends.bounded.BoundedSolver : Общие параметры модели.

    Notes
    -----
    Одно LU-разложение линейной матрицы используется повторно.
    Требуется

    .. math::

        \frac{\Delta t\,\gamma}{1+\Delta t\,\kappa} < 1,

    где :math:`\Delta t` — `self.dt`, :math:`\gamma` — действующий
    `self.gamma`. При `gamma=None` он берётся из
    `model.get("reaction_gamma", 0)`; иначе используется переданный `gamma`.
    Коэффициент :math:`\kappa` — `model["linear_loss"]`.
    Счётчик результата учитывает обновления
    неподвижной точки.
    """
    supports_field_coefficients = False

    def __init__(self, *args, residual_tolerance=1e-11, max_fixed_point_iterations=50,
                 **kwargs):
        iterations = _integer_count(max_fixed_point_iterations, minimum=1,
                                    name="max_fixed_point_iterations")
        if (not is_scalar_constant(residual_tolerance) or not np.isfinite(residual_tolerance)
                or residual_tolerance <= 0):
            raise ValueError("Positive finite residual tolerance required.")
        super().__init__(*args, **kwargs)
        self.residual_tolerance = float(residual_tolerance)
        self.max_fixed_point_iterations = iterations
        self.contraction_bound = self.dt*self.gamma/(
            1+self.dt*self.model['linear_loss'])
        if self.contraction_bound >= 1:
            raise ValueError("Cached fixed-point solver requires dt*gamma/(1+dt*kappa) < 1.")
        # LU линейного шага используется только при нулевой нелинейной реакции.
        self.cached_lu = self.linear_lu if self.linear_lu is not None else splu(self.base_matrix)

    def _derivative_solve(self, rhs, at_state, transpose=False, time_index=None):
        """Решить линеаризованную систему или её транспонирование итерациями."""
        diagonal = self.dt*self.reaction_derivative(at_state)
        matrix = self.base_matrix.T if transpose else self.base_matrix
        trans = 'T' if transpose else 'N'
        scale = 1+np.linalg.norm(rhs, np.inf)
        value = self.cached_lu.solve(rhs, trans=trans)
        for iteration in range(self.max_fixed_point_iterations+1):
            scaled = np.linalg.norm(matrix @ value+diagonal*value-rhs, np.inf)/scale
            if scaled <= self.residual_tolerance:
                return value
            if iteration == self.max_fixed_point_iterations:
                raise RuntimeError(f"Derivative fixed-point residual failed: {scaled}")
            value = self.cached_lu.solve(rhs-diagonal*value, trans=trans)

    def _solve_steps(self, contribution):
        states = np.zeros((self.nt+1, self.size))
        max_residual = 0.
        total_iterations = 0
        for n in range(self.nt):
            rhs = states[n]+contribution(n+1)
            scale = 1+np.linalg.norm(rhs, np.inf)
            state = self.cached_lu.solve(rhs)
            for iteration in range(self.max_fixed_point_iterations+1):
                residual = self.base_matrix @ state+self.dt*self.reaction(state)-rhs
                scaled = np.linalg.norm(residual, np.inf)/scale
                if scaled <= self.residual_tolerance:
                    break
                if iteration == self.max_fixed_point_iterations:
                    raise RuntimeError(f"State fixed-point failed at time step {n}: {scaled}")
                state = self.cached_lu.solve(rhs-self.dt*self.reaction(state))
                total_iterations += 1
            states[n+1] = state
            max_residual = max(max_residual, scaled)
        return StateTrajectory(states, float(max_residual), total_iterations)
