"""Композиция нагрузки, состояния ADR и линейных наблюдений."""
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from adrkit.coefficients import ADRUnits
from adrkit.contracts import MapCapabilities
from adrkit.observations.grid import GridObservation
from adrkit.selection import Selection
from adrkit.spaces import ArraySpace


def solver_state_space(solver, *, state_unit=None):
    """Описать пространство сеточной концентрации решателя.

    Parameters
    ----------
    solver : object
        Решатель с nt, size, times, xy и units.
    state_unit : str or None, optional
        Единица концентрации; при None берётся из solver.units либо
        задаётся как unknown.

    Returns
    -------
    space : ArraySpace
        Пространство формы (nt+1, size) с временными и
        пространственными координатами решателя.
    """
    if state_unit is None:
        state_unit = solver.units.concentration if solver.units is not None else "unknown"
    return ArraySpace((solver.nt+1, solver.size), axes=("time", "node"),
                      coordinates=(solver.times, solver.xy), units=state_unit)


def grid_observation(space_weights, time_weights, *, state_space, codomain):
    """Создать линейные функционалы сеточного состояния.

    Parameters
    ----------
    space_weights : array_like, shape (R, N)
        Пространственные веса приёмников, включая нужную площадь и
        нормировку.
    time_weights : array_like, shape (K, T)
        Веса временных функционалов, включая нужную длительность и
        нормировку.
    state_space : ArraySpace
        Пространство состояния формы (T, N).
    codomain : ArraySpace
        Пространство наблюдений формы (R*K,) и требуемые единицы.

    Returns
    -------
    observation : GridObservation
        Оператор со строками в порядке приёмника, затем временного
        функционала.
    """
    s, w = np.asarray(space_weights), np.asarray(time_weights)
    if s.ndim != 2 or w.ndim != 2 or min(*s.shape, *w.shape) == 0:
        raise ValueError("Nonempty spatial and temporal matrices required")
    if not isinstance(state_space, ArraySpace):
        raise TypeError("Explicit state ArraySpace required")
    if not isinstance(codomain, ArraySpace):
        raise TypeError("Explicit observation ArraySpace required")
    return GridObservation(s, w, domain=state_space, codomain=codomain)


@dataclass(frozen=True)
class StateCertificate:
    """Показатели сохранённого решения без массива состояния.

    Parameters
    ----------
    max_scaled_residual : float
        Наибольшая относительная невязка шага.
    nonlinear_iterations : int
        Суммарное число нелинейных обновлений решателя.
    """
    max_scaled_residual: float
    nonlinear_iterations: int


def _validate_components(solver, load, observation):
    """Проверить пространства нагрузки, состояния и наблюдения."""
    if not isinstance(observation, GridObservation):
        raise TypeError("Explicit GridObservation required")
    for method in ("predict", "jvp", "vjp"):
        provider = getattr(load, method) if isinstance(load, MapCapabilities) else load
        if not callable(getattr(provider, method, None)):
            raise TypeError(f"GridPrediction requires callable {method}()")
    for method in ("solve_controls", "tangent_controls", "adjoint_controls", "snapshot"):
        if not callable(getattr(solver, method, None)):
            raise TypeError(f"GridPrediction requires callable {method}()")
    if not isinstance(load.domain, ArraySpace) or not isinstance(load.codomain, ArraySpace):
        raise TypeError("Load domain and codomain must be ArraySpace records")
    for name in ("nt", "size"):
        value = getattr(solver, name, None)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"Solver {name} must be a positive integer")
    times, points = np.asarray(solver.times), np.asarray(solver.xy)
    if (times.shape != (solver.nt+1,) or times.dtype.kind not in "iuf"
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
        raise ValueError("Solver must declare finite increasing time nodes")
    if (points.shape != (solver.size, 2) or points.dtype.kind not in "iuf"
            or not np.isfinite(points).all()):
        raise ValueError("Solver must declare finite ordered two-dimensional geometry")
    if (load.codomain.shape != (solver.nt+1,)
            or observation.domain.shape != (solver.nt+1, solver.size)
            or len(load.domain.shape) != 1
            or load.domain.dtype != np.dtype("float64")
            or load.codomain.dtype != load.domain.dtype
            or observation.domain.dtype != load.domain.dtype):
        raise ValueError("Control, state and observation layouts must agree")
    if (load.codomain.coordinates[0] != tuple(times)
            or observation.domain.coordinates[0] != tuple(times)
            or observation.domain.coordinates[1] != tuple(map(tuple, points))):
        raise ValueError("Actual control/state times and ordered geometry must agree")
    if observation.codomain.units != observation.domain.units:
        raise ValueError("Observation and concentration units must agree")
    if solver.units is not None:
        if not isinstance(solver.units, ADRUnits):
            raise TypeError("Solver units must be ADRUnits or None")
        if (load.codomain.units != solver.units.source
                or observation.domain.units != solver.units.concentration):
            raise ValueError("Declared units must agree with the PDE solver")
    return load if isinstance(load, MapCapabilities) else MapCapabilities(
        load, load.domain, load.codomain, load, load)


class GridPrediction:
    """Прогноз концентрации из параметров временной нагрузки.

    Parameters
    ----------
    solver : object
        ADR-решатель с точечной нагрузкой, производными и методом
        snapshot.
    load : MapCapabilities or object
        Отображение параметров в интенсивности на временной сетке с
        JVP/VJP.
    observation : GridObservation
        Линейный оператор подгонки сеточной концентрации.

    Notes
    -----
    Данные решателя и нагрузки копируются при создании. Для последней
    точки сохраняется одна траектория; другие линейные наблюдения этого
    состояния доступны через project.
    """

    def __init__(self, solver, load, observation):
        _validate_components(solver, load, observation)
        owned_solver, owned_load = solver.snapshot(), deepcopy(load)
        owned_capabilities = _validate_components(owned_solver, owned_load, observation)
        load.domain.require_compatible(owned_load.domain)
        load.codomain.require_compatible(owned_load.codomain)
        self._solver, self._load, self._observation = owned_solver, owned_load, observation
        self._load_capabilities = owned_capabilities
        self.domain, self.codomain = owned_load.domain, observation.codomain
        self._point = self._value = self._trajectory = None

    @property
    def observation(self):
        """Фиксированный оператор наблюдения для подгонки."""
        return self._observation

    @property
    def trajectory(self):
        """StateCertificate последнего решения либо None до вычисления.

        Returns
        -------
        certificate : StateCertificate or None
            Показатели последнего решения; `None` до первого расчёта
            или после `invalidate`. Массив состояния не возвращается.
        """
        if self._trajectory is None:
            return None
        return StateCertificate(float(self._trajectory.max_scaled_residual),
                                int(self._trajectory.nonlinear_iterations))

    def invalidate(self):
        """Освободить сохранённую точку, прогноз и траекторию."""
        self._point = self._value = self._trajectory = None

    def observe(self, states):
        """Применить оператор подгонки к сеточному состоянию.

        Parameters
        ----------
        states : array_like
            Состояние формы observation.domain.shape в единицах
            концентрации.

        Returns
        -------
        prediction : ndarray
            Наблюдения формы codomain.shape в единицах codomain.
        """
        return self.observation.predict(states)

    def predict(self, point):
        """Вычислить концентрационные наблюдения по параметрам.

        Parameters
        ----------
        point : array_like
            Параметры формы domain.shape в единицах domain.

        Returns
        -------
        prediction : ndarray
            Независимая копия прогноза формы codomain.shape в единицах
            codomain.

        Notes
        -----
        В последней вычисленной точке сохранённая траектория используется повторно.
        """
        point = self.domain.array(point, name="parameter point", copy=True)
        if self._point is None or not np.array_equal(point, self._point):
            trajectory = self._solver.solve_controls(self._load_capabilities.predict.predict(point))
            value = self.observe(trajectory.states)
            self._trajectory, self._value = trajectory, value
            self._point = point.copy()
        return self._value.copy()

    def project(self, point, observation):
        """Вычислить другой линейный функционал сохранённого состояния.

        Parameters
        ----------
        point : array_like
            Параметры формы domain.shape в единицах domain.
        observation : GridObservation
            Другой оператор на том же пространстве состояния.

        Returns
        -------
        prediction : ndarray
            Результат формы observation.codomain.shape в его единицах.

        Notes
        -----
        В последней точке новое прямое решение не требуется.
        """
        if not isinstance(observation, GridObservation):
            raise TypeError("Explicit GridObservation required")
        self.observation.domain.require_compatible(observation.domain)
        self.predict(point)
        return observation.predict(self._trajectory.states)

    def jvp(self, point, direction):
        """Применить производную прогноза к направлению параметров.

        Parameters
        ----------
        point : array_like
            Параметры формы domain.shape в единицах domain.
        direction : array_like
            Направление формы domain.shape.

        Returns
        -------
        tangent : ndarray
            Действие производной формы codomain.shape при фиксированных
            коэффициентах уравнения.
        """
        point = self.domain.array(point, name="parameter point", copy=True)
        direction = self.domain.array(direction, name="direction", copy=True)
        self.predict(point)
        tangent = self._solver.tangent_controls(
            self._trajectory, self._load_capabilities.jvp.jvp(point, direction))
        return self.observe(tangent)

    def vjp(self, point, cotangent):
        """Перенести ковектор наблюдений к параметрам нагрузки.

        Parameters
        ----------
        point : array_like
            Параметры формы domain.shape в единицах domain.
        cotangent : array_like
            Координатный ковектор формы codomain.shape.

        Returns
        -------
        covector : ndarray
            Координатный ковектор формы domain.shape.
        """
        point = self.domain.array(point, name="parameter point", copy=True)
        cotangent = self.codomain.array(cotangent, name="cotangent", copy=True)
        self.predict(point)
        dual = self.observation.vjp(self._trajectory.states, cotangent)
        controls = self._solver.adjoint_controls(self._trajectory, dual)
        return self._load_capabilities.vjp.vjp(point, controls)

    def jacobian(self, point):
        """Собрать матрицу производной прогноза по параметрам.

        Parameters
        ----------
        point : array_like
            Параметры формы domain.shape в единицах domain.

        Returns
        -------
        jacobian : ndarray, shape (codomain.size, domain.size)
            Строки следуют наблюдениям, столбцы — параметрам.
        """
        return np.column_stack([self.jvp(point, d) for d in np.eye(self.domain.size)])

    def projected_jacobians(self, point, observations):
        """Собрать производные нескольких наблюдений в одной точке.

        Parameters
        ----------
        point : array_like
            Параметры формы domain.shape в единицах domain.
        observations : sequence of GridObservation
            Непустой набор операторов на пространстве состояния.

        Returns
        -------
        jacobians : tuple of ndarray
            Матрицы формы (obs.codomain.size, domain.size) в порядке
            observations.

        Notes
        -----
        Направление состояния для каждого параметра вычисляется один раз
        и применяется ко всем операторам наблюдения.
        """
        observations = tuple(observations)
        if not observations:
            raise ValueError("At least one observation required")
        for obs in observations:
            if not isinstance(obs, GridObservation):
                raise TypeError("Explicit GridObservation required")
            self.observation.domain.require_compatible(obs.domain)
        self.predict(point)
        matrices = [np.empty((obs.codomain.size, self.domain.size)) for obs in observations]
        for j, d in enumerate(np.eye(self.domain.size)):
            tangent = self._solver.tangent_controls(
                self._trajectory, self._load_capabilities.jvp.jvp(point, d))
            for matrix, obs in zip(matrices, observations):
                matrix[:, j] = obs.predict(tangent)
        return tuple(matrices)

    def capabilities(self):
        """Предоставить прогноз и JVP/VJP как MapCapabilities."""
        return MapCapabilities(self, self.domain, self.codomain, self, self)


class SelectedPrediction:
    """Прогноз с упорядоченным выбором строк наблюдений.

    Parameters
    ----------
    prediction : object
        Полный прогноз с domain, codomain и нужными операциями
        производной.
    rows : array_like of int, shape (k,)
        Уникальные индексы строк полного векторного прогноза.
    """

    def __init__(self, prediction, rows):
        domain = getattr(prediction, "domain", None)
        codomain = getattr(prediction, "codomain", None)
        if not isinstance(domain, ArraySpace) or not isinstance(codomain, ArraySpace):
            raise TypeError("Prediction domain and codomain must be ArraySpace records")
        self.base = prediction
        self._selection = Selection(rows, codomain)
        self.domain = domain
        self.codomain = self._selection.selected_layout

    @property
    def rows(self):
        """Независимая копия выбранных индексов в порядке наблюдений."""
        return self._selection.indices

    @property
    def trajectory(self):
        """Показатели последней траектории полного прогноза."""
        return self.base.trajectory

    def predict(self, point):
        """Вычислить выбранные наблюдения по параметрам.

        Parameters
        ----------
        point : array_like
            Параметры формы domain.shape в единицах domain.

        Returns
        -------
        prediction : ndarray
            Независимая копия выбранного прогноза формы codomain.shape.
        """
        point = self.domain.array(point, name="parameter point", copy=True)
        return self.codomain.array(self._selection.take(self.base.predict(point)),
                                   name="selected prediction", copy=True)

    def vjp(self, point, cotangent):
        """Перенести ковектор выбранных наблюдений к параметрам прогноза.

        Parameters
        ----------
        point : array_like
            Параметры формы domain.shape в единицах domain.
        cotangent : array_like
            Координатный ковектор формы codomain.shape.

        Returns
        -------
        covector : ndarray
            Координатный ковектор формы domain.shape.
        """
        point = self.domain.array(point, name="parameter point", copy=True)
        return self.base.vjp(point, self._selection.scatter(cotangent))

    def jvp(self, point, direction):
        """Применить производную прогноза к направлению параметров.

        Parameters
        ----------
        point : array_like
            Параметры формы domain.shape в единицах domain.
        direction : array_like
            Направление формы domain.shape.

        Returns
        -------
        tangent : ndarray
            Действие производной полного прогноза после выбора строк;
            форма `codomain.shape`.
        """
        point = self.domain.array(point, name="parameter point", copy=True)
        direction = self.domain.array(direction, name="direction", copy=True)
        return self.codomain.array(self._selection.take(self.base.jvp(point, direction)),
                                   name="selected JVP output", copy=True)

    def jacobian(self, point):
        """Собрать матрицу производной прогноза по параметрам.

        Parameters
        ----------
        point : array_like
            Параметры формы domain.shape в единицах domain.

        Returns
        -------
        jacobian : ndarray, shape (codomain.size, domain.size)
            Строки следуют наблюдениям, столбцы — параметрам.
        """
        point = self.domain.array(point, name="parameter point", copy=True)
        return self.base.jacobian(point)[self.rows]

    def invalidate(self):
        """Освободить сохранённую траекторию полного прогноза."""
        self.base.invalidate()
