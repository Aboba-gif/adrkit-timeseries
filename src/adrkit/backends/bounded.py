"""Разностная ADR на прямоугольнике с нулевыми граничными данными."""
from copy import copy, deepcopy
from dataclasses import dataclass
from operator import index
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from adrkit.coefficients import (ADRUnits, ReactionLaw, is_scalar_constant,
                                is_vector_constant, sample_field)
from adrkit.errors import CapabilityError
from adrkit.pde import ADRProblem
from adrkit.state_residuals import StateResidual


def _integer_count(value, *, minimum, name):
    """Проверить целый размер или предел итераций."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        count = index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer >= {minimum}") from error
    if count < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return count


@dataclass
class StateTrajectory:
    r"""Состояния неявной схемы и показатели решения.

    Parameters
    ----------
    states : ndarray, shape (nt+1, size)
        Концентрация на временных узлах; начальная строка нулевая.
    max_scaled_residual : float
        Наибольшая относительная невязка шага в бесконечной норме.
    nonlinear_iterations : int
        Суммарное число обновлений Ньютона или неподвижной точки.

    Notes
    -----
    Для правой части шага :math:`b_k` и невязки :math:`r_k`:

    .. math::

        \varepsilon = \max_k
        \frac{\|r_k\|_\infty}{1 + \|b_k\|_\infty}.

    Здесь :math:`r_k` — невязка шага, :math:`b_k` — его правая часть.
    """
    states: np.ndarray
    max_scaled_residual: float
    nonlinear_iterations: int


class BoundedResidual:
    """Невязка неявной ADR-схемы на прямоугольнике.

    Parameters
    ----------
    model : ADRProblem or Mapping
        Данные уравнения с нулевыми начальными и граничными
        значениями.
    interior_points : int or tuple of int, optional
        Число внутренних узлов каждой оси; не меньше трёх.
    time_steps : int, optional
        Положительное число временных шагов.
    half_width : float, optional
        Половина стороны квадратной области при domain=None.
    gamma : float, ScalarField, callable or None, optional
        Коэффициент встроенной реакции вместо reaction_gamma из
        model.
    domain : array_like, shape (4,), or None, optional
        Границы xmin, xmax, ymin, ymax в единицах длины модели.

    Notes
    -----
    При заданном точечном источнике его положение должно совпадать
    с внутренним узлом сетки. Дискретная нагрузка равна обратной площади
    ячейки. Коэффициенты полей вычисляются на правых концах шагов.
    """
    supports_field_coefficients = True

    def __init__(self, model, interior_points=23, time_steps=80,
                 half_width=2., gamma=None, domain=None):
        # Функции могут владеть некопируемыми ресурсами. В снимке модели поля
        # заменяются числовыми слоями, а законы состояния копируются через snapshot().
        model = model.to_model() if isinstance(model, ADRProblem) else dict(model)
        self.model = model
        if np.ndim(interior_points) == 0:
            self.nx = self.ny = _integer_count(interior_points, minimum=3, name="interior_points")
        else:
            counts = tuple(interior_points)
            if len(counts) != 2:
                raise ValueError("interior_points must give one integer or two axis counts")
            self.nx, self.ny = (_integer_count(n, minimum=3, name="interior_points") for n in counts)
        self.nt = _integer_count(time_steps, minimum=1, name="time_steps")
        bounds = (-half_width, half_width, -half_width, half_width) if domain is None else domain
        self.bounds = np.array(bounds, dtype=float, copy=True)
        if self.bounds.shape != (4,) or not np.isfinite(self.bounds).all():
            raise ValueError("Finite rectangle bounds are required.")
        xmin, xmax, ymin, ymax = self.bounds
        self.hx = (xmax-xmin)/(self.nx+1)
        self.hy = (ymax-ymin)/(self.ny+1)
        if min(self.hx, self.hy) <= 0:
            raise ValueError("Rectangle bounds must increase on both axes.")
        self.cell_area = self.hx*self.hy
        self.dt = float(model['horizon']) / self.nt
        self.gamma = model.get('reaction_gamma', 0.) if gamma is None else gamma
        self.c_star = model.get('reaction_c_star', 1.)
        self.time_origin = model.get('time_origin', 0.)
        if (not is_scalar_constant(self.time_origin) or not np.isfinite(self.time_origin)
                or not np.isfinite(self.dt) or self.dt <= 0):
            raise ValueError("Positive finite horizon and finite physical time_origin required")
        self.time_origin = float(self.time_origin)
        self.units = model.get('units')
        if self.units is not None and not isinstance(self.units, ADRUnits):
            raise TypeError("units must be ADRUnits; unit labels do not convert values")
        if (model.get('diffusion_form', 'divergence') != 'divergence'
                or model.get('advection_form', 'advective') != 'advective'):
            raise CapabilityError("Bounded backend implements -div(D grad c) + v dot grad c only")
        reaction = model.get('reaction')
        if reaction is not None and not isinstance(reaction, ReactionLaw):
            raise TypeError("reaction must be an explicit ReactionLaw")
        state_residual = model.get('state_residual')
        if state_residual is not None and not isinstance(state_residual, StateResidual):
            raise TypeError("state_residual must be an explicit StateResidual")
        self._variable_fields = not (
            all(is_scalar_constant(v) for v in (model['diffusion'], model['linear_loss'],
                                                 self.gamma, self.c_star))
            and is_vector_constant(model['velocity']) and reaction is None and state_residual is None)
        if self._variable_fields and not self.supports_field_coefficients:
            raise CapabilityError(f"{type(self).__name__} supports constant coefficients and built-in tanh only; "
                                  "use BoundedSolver for prescribed fields, ReactionLaw or StateResidual")
        self._reaction_law = None if reaction is None else reaction.snapshot()
        self._state_residual_law = None if state_residual is None else state_residual.snapshot()
        if self._reaction_law is not None and self.units is not None:
            for label, expected in ((self._reaction_law.unit, self.units.reaction),
                                     (self._reaction_law.derivative_unit, self.units.loss)):
                if label is not None and label != expected:
                    raise ValueError(f"Reaction unit {label!r} does not match {expected!r}")
        if self._state_residual_law is not None and self.units is not None:
            for label, expected in ((self._state_residual_law.unit, self.units.reaction),
                                     (self._state_residual_law.derivative_unit, self.units.loss)):
                if label is not None and label != expected:
                    raise ValueError(f"StateResidual unit {label!r} does not match {expected!r}")
        if not self._variable_fields:
            velocity = np.asarray(model['velocity'], dtype=float)
            parameters = [self.dt, model['diffusion'], model['linear_loss'], self.gamma, self.c_star]
            if (not np.isfinite(velocity).all() or not np.isfinite(parameters).all() or
                    model['diffusion'] <= 0 or self.c_star <= 0 or
                    self.gamma < 0 or model['linear_loss'] < 0):
                raise ValueError("Finite transport, positive time/diffusion/scale, and nonnegative losses required.")
        if model.get('initial_state', 0.) != 0 or model.get('background', 0.) != 0:
            raise ValueError("This solver implements zero initial state and zero background only.")
        self.times = np.linspace(0., model['horizon'], self.nt + 1)
        xx, yy = np.meshgrid(xmin+self.hx*np.arange(1, self.nx+1),
                             ymin+self.hy*np.arange(1, self.ny+1))
        self.xy = np.column_stack((xx.ravel(), yy.ravel()))
        self.size = self.nx*self.ny
        source = model.get('source_position', [0., 0.])
        self.load = self.source_index = None
        if source is not None:
            source = np.asarray(source, dtype=float)
            if source.shape != (2,) or not np.isfinite(source).all():
                raise ValueError("A finite source position in two dimensions is required.")
            distances = np.max(np.abs(self.xy-source), axis=1)
            self.source_index = int(np.argmin(distances))
            if distances[self.source_index] > 1e-10*max(1., self.hx, self.hy):
                raise ValueError("The point source must coincide with an interior grid node.")
            self.load = np.zeros(self.size)
            self.load[self.source_index] = 1/self.cell_area

        if self._variable_fields:
            self._sample_coefficients()
            self._freeze_model_metadata()
            self.linear_lu = None
            self._step_cache_index = self._step_cache_matrix = self._step_cache_operator = None
            return

        self.model = model = deepcopy(model)
        self.gamma, self.c_star = deepcopy(self.gamma), deepcopy(self.c_star)

        def axis_operator(velocity, count, spacing):
            d = model['diffusion'] / spacing**2
            diagonal = np.full(count, 2*d + abs(velocity)/spacing)
            lower = np.full(count-1, -d - max(velocity, 0)/spacing)
            upper = np.full(count-1, -d + min(velocity, 0)/spacing)
            return sparse.diags([lower, diagonal, upper], [-1, 0, 1], format='csc')

        ax = axis_operator(velocity[0], self.nx, self.hx)
        ay = axis_operator(velocity[1], self.ny, self.hy)
        operator = sparse.kron(sparse.eye(self.ny), ax) + sparse.kron(ay, sparse.eye(self.nx))
        operator += model['linear_loss'] * sparse.eye(self.size)
        self.operator = operator.tocsc()
        self.base_matrix = (sparse.eye(self.size) + self.dt * operator).tocsc()
        self.linear_lu = splu(self.base_matrix) if self.gamma == 0 else None

    def _sample_coefficients(self):
        """Вычислить диффузию на гранях и остальные поля в узлах сетки."""
        xmin, xmax, ymin, ymax = self.bounds
        x = xmin+self.hx*np.arange(1, self.nx+1)
        y = ymin+self.hy*np.arange(1, self.ny+1)
        xf, yf = np.meshgrid(xmin+self.hx*(np.arange(self.nx+1)+.5), y)
        xg, yg = np.meshgrid(x, ymin+self.hy*(np.arange(self.ny+1)+.5))
        faces = np.concatenate((np.column_stack((xf.ravel(), yf.ravel())),
                                np.column_stack((xg.ravel(), yg.ravel()))))
        times = self.times[1:]+self.time_origin
        unit = lambda name: None if self.units is None else getattr(self.units, name)
        self._diffusion_frames = sample_field(self.model['diffusion'], faces, times,
            name='diffusion', positive=True, expected_unit=unit('diffusion'))
        self._velocity_frames = sample_field(self.model['velocity'], self.xy, times,
            name='velocity', components=2, expected_unit=unit('velocity'))
        self._loss_frames = sample_field(self.model['linear_loss'], self.xy, times,
            name='linear_loss', nonnegative=True, expected_unit=unit('loss'))
        if self._reaction_law is None:
            self._gamma_frames = sample_field(self.gamma, self.xy, times,
                name='reaction_gamma', nonnegative=True, expected_unit=unit('loss'))
            self._scale_frames = sample_field(self.c_star, self.xy, times,
                name='reaction_c_star', positive=True, expected_unit=unit('concentration'))

    def _freeze_model_metadata(self):
        """Сохранить вычисленные коэффициенты и снимки законов состояния."""
        field_keys = {'diffusion', 'velocity', 'linear_loss', 'reaction_gamma',
                      'reaction_c_star', 'reaction', 'state_residual'}
        self.model = deepcopy({key: value for key, value in self.model.items()
                               if key not in field_keys})
        self.model.update(diffusion=self._diffusion_frames, velocity=self._velocity_frames,
                          linear_loss=self._loss_frames)
        if self._reaction_law is None:
            self.gamma, self.c_star = self._gamma_frames, self._scale_frames
            self.model.update(reaction_gamma=self.gamma, reaction_c_star=self.c_star)
        else:
            self.gamma = self.c_star = None
            self.model['reaction'] = self._reaction_law
        if self._state_residual_law is not None:
            self.model['state_residual'] = self._state_residual_law

    def _step(self, time_index):
        if (not isinstance(time_index, (int, np.integer)) or isinstance(time_index, (bool, np.bool_))
                or not 0 <= time_index < self.nt):
            raise ValueError("Variable-field operations require time_index in [0, nt)")
        return int(time_index)

    def _base_at(self, time_index=None):
        if not self._variable_fields:
            return self.base_matrix
        step = self._step(time_index)
        if self._step_cache_index != step:
            d = self._diffusion_frames.at(step)
            split = self.ny*(self.nx+1)
            dx = d[:split].reshape(self.ny, self.nx+1)/self.hx**2
            dy = d[split:].reshape(self.ny+1, self.nx)/self.hy**2
            velocity = self._velocity_frames.at(step).reshape(self.ny, self.nx, 2)
            vx, vy = velocity[..., 0], velocity[..., 1]
            west = -dx[:, :-1]-np.maximum(vx, 0)/self.hx
            east = -dx[:, 1:]+np.minimum(vx, 0)/self.hx
            south = -dy[:-1]-np.maximum(vy, 0)/self.hy
            north = -dy[1:]+np.minimum(vy, 0)/self.hy
            diagonal = (dx[:, :-1]+dx[:, 1:]+dy[:-1]+dy[1:]
                        +np.abs(vx)/self.hx+np.abs(vy)/self.hy
                        +self._loss_frames.at(step).reshape(self.ny, self.nx))
            index = np.arange(self.size).reshape(self.ny, self.nx)
            rows = np.concatenate((index.ravel(), index[:, 1:].ravel(), index[:, :-1].ravel(),
                                   index[1:].ravel(), index[:-1].ravel()))
            cols = np.concatenate((index.ravel(), index[:, :-1].ravel(), index[:, 1:].ravel(),
                                   index[:-1].ravel(), index[1:].ravel()))
            values = np.concatenate((diagonal.ravel(), west[:, 1:].ravel(), east[:, :-1].ravel(),
                                     south[1:].ravel(), north[:-1].ravel()))
            operator = sparse.coo_matrix((values, (rows, cols)), shape=(self.size, self.size)).tocsc()
            self._step_cache_operator = operator
            self._step_cache_matrix = (sparse.eye(self.size)+self.dt*operator).tocsc()
            self._step_cache_index = step
        return self._step_cache_matrix

    def operator_at(self, time_index):
        """Получить пространственный линейный оператор шага.

        Parameters
        ----------
        time_index : int
            Индекс шага в диапазоне от нуля до nt−1 для переменных полей.

        Returns
        -------
        operator : scipy.sparse.csc_matrix, shape (size, size)
            Копия оператора диффузии, переноса и линейных потерь.

        Notes
        -----
        Локальная нелинейная реакция и StateResidual в матрицу не входят.
        """
        self._base_at(time_index)
        return (self._step_cache_operator if self._variable_fields else self.operator).copy()

    def _reaction_parameters(self, time_index):
        if not self._variable_fields:
            return self.gamma, self.c_star
        step = self._step(time_index)
        return self._gamma_frames.at(step), self._scale_frames.at(step)

    def reaction(self, state, time_index=None):
        """Вычислить реакцию и добавочный дискретный член.

        Parameters
        ----------
        state : ndarray, shape (size,)
            Концентрация во внутренних узлах.
        time_index : int or None, optional
            Индекс шага; требуется для переменных полей.

        Returns
        -------
        values : ndarray, shape (size,)
            Реакция с добавочным StateResidual в единицах
            концентрация/время.
        """
        if self._reaction_law is not None:
            step = self._step(time_index)
            result = self._reaction_law.evaluate(state, self.xy, self.time_origin+self.times[step+1])
        else:
            gamma, c_star = self._reaction_parameters(time_index)
            result = gamma*c_star*np.tanh(state/c_star)
        if self._state_residual_law is not None:
            step = self._step(time_index)
            result = result+self._state_residual_law.evaluate(
                state, self.xy, self.time_origin+self.times[step+1])
        return result

    def reaction_derivative(self, state, time_index=None):
        """Вычислить производную локальной реакции по концентрации.

        Parameters
        ----------
        state : ndarray, shape (size,)
            Концентрация во внутренних узлах.
        time_index : int or None, optional
            Индекс шага; требуется для переменных полей.

        Returns
        -------
        values : ndarray, shape (size,)
            Локальная производная в обратных единицах времени;
            StateResidual не включён.
        """
        if self._reaction_law is not None:
            step = self._step(time_index)
            return self._reaction_law.evaluate(state, self.xy, self.time_origin+self.times[step+1],
                                                derivative=True)
        gamma, c_star = self._reaction_parameters(time_index)
        return gamma*(1-np.tanh(state/c_star)**2)

    def jacobian(self, state, time_index=None):
        """Построить якобиан невязки по следующему состоянию.

        Parameters
        ----------
        state : ndarray, shape (size,)
            Состояние, в котором вычисляется производная.
        time_index : int or None, optional
            Индекс шага; требуется для переменных полей.

        Returns
        -------
        jacobian : scipy.sparse.csc_matrix, shape (size, size)
            Производная неявного шага с локальной реакцией и полным
            якобианом StateResidual.
        """
        derivative = self.reaction_derivative(state, time_index)
        matrix = (self._base_at(time_index) + sparse.diags(self.dt*derivative)).tocsc()
        if self._state_residual_law is not None:
            step = self._step(time_index)
            matrix = (matrix+self.dt*self._state_residual_law.evaluate(
                state, self.xy, self.time_origin+self.times[step+1], derivative=True)).tocsc()
        return matrix

    def residual(self, next_state, previous_state, control_right, time_index=None):
        """Вычислить невязку шага с точечным источником.

        Parameters
        ----------
        next_state, previous_state : ndarray, shape (size,)
            Концентрация в конце и начале временного шага.
        control_right : float
            Интенсивность на правом конце шага в единицах
            концентрация·длина²/время.
        time_index : int or None, optional
            Индекс шага; требуется для переменных полей.

        Returns
        -------
        residual : ndarray, shape (size,)
            Невязка неявного шага в единицах концентрации.
        """
        if self.load is None:
            raise CapabilityError("No point-source load configured; use residual_forcing")
        return (self._base_at(time_index) @ next_state + self.dt*self.reaction(next_state, time_index)
                - previous_state - self.dt*self.load*control_right)

    def residual_forcing(self, next_state, previous_state, forcing, time_index=None):
        """Вычислить невязку шага с распределённой нагрузкой.

        Parameters
        ----------
        next_state, previous_state : ndarray, shape (size,)
            Концентрация в конце и начале шага.
        forcing : array_like, shape (size,)
            Конечная нагрузка в единицах концентрация/время.
        time_index : int or None, optional
            Индекс шага; требуется для переменных полей.

        Returns
        -------
        residual : ndarray, shape (size,)
            Невязка неявного шага в единицах концентрации.
        """
        forcing = np.asarray(forcing)
        if (forcing.shape != (self.size,) or forcing.dtype.kind not in 'iuf'
                or not np.isfinite(forcing).all()):
            raise ValueError("A finite spatial forcing vector is required")
        return (self._base_at(time_index) @ next_state + self.dt*self.reaction(next_state, time_index)
                -previous_state-self.dt*forcing)


class BoundedSolver(BoundedResidual):
    """Решатель неявной ADR-схемы с методом Ньютона.

    Parameters
    ----------
    *args
        Позиционные параметры BoundedResidual.
    residual_tolerance : float, optional
        Положительный допуск относительной невязки шага.
    max_newton_iterations : int, optional
        Положительный предел числа обновлений Ньютона за шаг.
    max_backtracks : int, optional
        Положительный предел уменьшений шага Ньютона.
    backtrack_factor : float, optional
        Множитель уменьшения шага строго между нулём и единицей.
    **kwargs
        Остальные параметры BoundedResidual.

    See Also
    --------
    BoundedResidual : Данные модели и пространственно-временная сетка.

    Notes
    -----
    Для постоянной линейной модели используется одно разложение LU.
    Нелинейные шаги решаются методом Ньютона с уменьшением длины шага.
    """
    def __init__(self, *args, residual_tolerance=1e-11, max_newton_iterations=20,
                 max_backtracks=20, backtrack_factor=0.5, **kwargs):
        for name, value in (("residual_tolerance", residual_tolerance),
                            ("backtrack_factor", backtrack_factor)):
            if (not is_scalar_constant(value) or not np.isfinite(value)
                    or value <= 0):
                raise ValueError(f"{name} must be a finite positive number")
        if backtrack_factor >= 1:
            raise ValueError("backtrack_factor must be less than one")
        for name, value in (("max_newton_iterations", max_newton_iterations),
                            ("max_backtracks", max_backtracks)):
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, np.integer)) or value < 1):
                raise ValueError(f"{name} must be a positive integer")
        super().__init__(*args, **kwargs)
        self.residual_tolerance = float(residual_tolerance)
        self.max_newton_iterations = int(max_newton_iterations)
        self.max_backtracks = int(max_backtracks)
        self.backtrack_factor = float(backtrack_factor)

    def solve_controls(self, controls):
        """Решить уравнение с точечным источником.

        Parameters
        ----------
        controls : array_like, shape (nt+1,)
            Конечные интенсивности в единицах концентрация·длина²/время;
            компонент ноль проверяется, но в шагах не используется.

        Returns
        -------
        trajectory : StateTrajectory
            Концентрация на временной сетке и показатели решения.
        """
        if self.load is None:
            raise CapabilityError("No point source configured; use solve_forcing")
        if np.iscomplexobj(controls):
            raise ValueError("Controls must be real")
        controls = np.asarray(controls, dtype=float)
        if controls.shape != self.times.shape or not np.isfinite(controls).all():
            raise ValueError("One finite control value per state time node is required.")
        return self._solve_steps(lambda n: self.dt*self.load*controls[n])

    def solve_forcing(self, forcing):
        """Решить уравнение с распределённой нагрузкой.

        Parameters
        ----------
        forcing : array_like, shape (nt+1, size)
            Конечные нагрузки в единицах концентрация/время; строка ноль
            не используется.

        Returns
        -------
        trajectory : StateTrajectory
            Концентрация на временной сетке и показатели решения.
        """
        forcing = self._forcing_array(forcing)
        return self._solve_steps(lambda n: self.dt*forcing[n])

    def _forcing_array(self, forcing):
        if np.iscomplexobj(forcing):
            raise ValueError("Forcing must be real")
        forcing = np.asarray(forcing, dtype=float)
        if forcing.shape != (self.nt+1, self.size) or not np.isfinite(forcing).all():
            raise ValueError("Forcing must be finite with shape (nt+1, size)")
        return forcing

    def _solve_steps(self, contribution):
        states = np.zeros((self.nt+1, self.size))
        max_residual = 0.
        total_iterations = 0
        for n in range(self.nt):
            matrix = self._base_at(n)
            rhs = states[n] + contribution(n+1)
            scale = 1+np.linalg.norm(rhs, np.inf)
            if self.linear_lu is not None:
                state = self.linear_lu.solve(rhs)
                scaled = np.linalg.norm(matrix @ state-rhs, np.inf)/scale
            else:
                state = states[n].copy()
                for iteration in range(self.max_newton_iterations+1):
                    residual = matrix @ state + self.dt*self.reaction(state, n)-rhs
                    norm = np.linalg.norm(residual, np.inf)
                    scaled = norm/scale
                    if scaled <= self.residual_tolerance:
                        break
                    if iteration == self.max_newton_iterations:
                        raise RuntimeError(f"Newton failed at time step {n}: {scaled}")
                    delta = splu(self.jacobian(state, time_index=n)).solve(-residual)
                    fraction = 1.
                    for backtrack in range(self.max_backtracks+1):
                        trial = state+fraction*delta
                        candidate = matrix @ trial + self.dt*self.reaction(trial, n)-rhs
                        if np.linalg.norm(candidate, np.inf) < norm:
                            state = trial
                            break
                        if backtrack == self.max_backtracks:
                            raise RuntimeError(f"Newton line search failed at time step {n}")
                        fraction *= self.backtrack_factor
                    total_iterations += 1
            states[n+1] = state
            max_residual = max(max_residual, scaled)
        return StateTrajectory(states, float(max_residual), total_iterations)

    def _derivative_solve(self, rhs, at_state, transpose=False, time_index=None):
        lu = self.linear_lu if self.linear_lu is not None else splu(
            self.jacobian(at_state, time_index=time_index))
        return lu.solve(rhs, trans='T' if transpose else 'N')

    def tangent_controls(self, forward, controls):
        """Применить производную состояния по точечному источнику.

        Parameters
        ----------
        forward : StateTrajectory
            Траектория точки, в которой вычисляется производная.
        controls : array_like, shape (nt+1,)
            Направление интенсивности точечного источника в единицах
            концентрация·длина²/время; начальный компонент не используется.

        Returns
        -------
        tangent : ndarray, shape (nt+1, size)
            Изменение концентрации при фиксированных коэффициентах
            уравнения.
        """
        if self.load is None:
            raise CapabilityError("No point source configured; use tangent_forcing")
        if np.iscomplexobj(controls):
            raise ValueError("Control directions must be real")
        controls = np.asarray(controls, dtype=float)
        if controls.shape != self.times.shape or not np.isfinite(controls).all():
            raise ValueError("One finite direction per state time node is required.")
        states = np.zeros_like(forward.states)
        for n in range(self.nt):
            rhs = states[n]+self.dt*self.load*controls[n+1]
            states[n+1] = self._derivative_solve(rhs, forward.states[n+1], time_index=n)
        return states

    def tangent_forcing(self, forward, forcing):
        """Применить производную состояния по распределённой нагрузке.

        Parameters
        ----------
        forward : StateTrajectory
            Траектория точки, в которой вычисляется производная.
        forcing : array_like, shape (nt+1, size)
            Направление распределённой нагрузки в единицах
            концентрация/время; начальная строка не используется.

        Returns
        -------
        tangent : ndarray, shape (nt+1, size)
            Изменение концентрации при фиксированных коэффициентах
            уравнения.
        """
        forcing = self._forcing_array(forcing)
        states = np.zeros_like(forward.states)
        for n in range(self.nt):
            rhs = states[n]+self.dt*forcing[n+1]
            states[n+1] = self._derivative_solve(rhs, forward.states[n+1], time_index=n)
        return states

    def adjoint_controls(self, forward, state_derivative):
        """Перенести ковектор состояния к точечному источнику.

        Parameters
        ----------
        forward : StateTrajectory
            Траектория точки линеаризации.
        state_derivative : array_like, shape (nt+1, size)
            Координатный ковектор по состоянию.

        Returns
        -------
        covector : ndarray, shape (nt+1,)
            Координатный ковектор интенсивности точечного источника;
            начальный компонент нулевой.

        Notes
        -----
        Транспонирование определяется евклидовым скалярным произведением
        координат состояния и нагрузки.
        """
        if self.load is None:
            raise CapabilityError("No point source configured; use adjoint_forcing")
        if np.iscomplexobj(state_derivative):
            raise ValueError("State cotangent must be real")
        state_derivative = np.asarray(state_derivative, dtype=float)
        if state_derivative.shape != forward.states.shape or not np.isfinite(state_derivative).all():
            raise ValueError("State cotangent must match the trajectory.")
        lam = state_derivative[-1].copy()
        gradient = np.zeros(self.nt+1)
        for n in range(self.nt-1, -1, -1):
            dual = self._derivative_solve(lam, forward.states[n+1], transpose=True, time_index=n)
            gradient[n+1] = self.dt*np.dot(self.load, dual)
            lam = state_derivative[n]+dual
        return gradient

    def adjoint_forcing(self, forward, state_derivative):
        """Перенести ковектор состояния к распределённой нагрузке.

        Parameters
        ----------
        forward : StateTrajectory
            Траектория точки линеаризации.
        state_derivative : array_like, shape (nt+1, size)
            Координатный ковектор по состоянию.

        Returns
        -------
        covector : ndarray, shape (nt+1, size)
            Координатный ковектор распределённой нагрузки;
            начальная строка нулевая.

        Notes
        -----
        Транспонирование определяется евклидовым скалярным произведением
        координат состояния и нагрузки.
        """
        state_derivative = self._forcing_array(state_derivative)
        lam = state_derivative[-1].copy()
        gradient = np.zeros_like(forward.states)
        for n in range(self.nt-1, -1, -1):
            dual = self._derivative_solve(lam, forward.states[n+1], transpose=True, time_index=n)
            gradient[n+1] = self.dt*dual
            lam = state_derivative[n]+dual
        return gradient

    def snapshot(self):
        """Получить независимые данные и настройки решателя.

        Returns
        -------
        solver : BoundedSolver
            Копия численных слоёв и настроек; законы состояния копируются
            своими фабриками снимка.
        """
        result = copy(self)
        # Числовые слои копируются с сохранением общих ссылок; законы — через snapshot().
        memo = {id(law): law for law in (self._reaction_law, self._state_residual_law)
                if law is not None}
        result.__dict__ = deepcopy({k: v for k, v in self.__dict__.items()
                                   if k not in ('linear_lu', 'cached_lu')}, memo)
        result.linear_lu = splu(result.base_matrix) if self.linear_lu is not None else None
        if self._variable_fields:
            for key, value in result.__dict__.items():
                if key.endswith('_frames'):
                    for frame in value.values:
                        frame.flags.writeable = False
        if self._reaction_law is not None:
            result._reaction_law = self._reaction_law.snapshot()
            result.model['reaction'] = result._reaction_law
        if self._state_residual_law is not None:
            result._state_residual_law = self._state_residual_law.snapshot()
            result.model['state_residual'] = result._state_residual_law
        if hasattr(self, 'cached_lu'):
            result.cached_lu = (result.linear_lu if result.linear_lu is not None
                                else splu(result.base_matrix))
        return result


