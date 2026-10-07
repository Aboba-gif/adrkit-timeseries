"""Отображения входа решателя в состояние и его производные."""
from copy import deepcopy

import numpy as np
from scipy.sparse.linalg import LinearOperator

from adrkit.coefficients import ADRUnits
from adrkit.contracts import MapCapabilities, require_capability
from adrkit.spaces import ArraySpace


def _states_data(trajectory):
    return trajectory.states


class _TrajectoryMap:
    """Адаптер решателя с сохранением траектории последней точки."""

    def __init__(self, *, domain, codomain, solve, jvp, vjp, states,
                 copy_trajectory, owned_trajectory=False, fixed_actions=None):
        self.domain, self.codomain = domain, codomain
        self._solve, self._tangent, self._pullback = solve, jvp, vjp
        self._states, self._copy_trajectory = states, copy_trajectory
        self._owned_trajectory = owned_trajectory
        self._fixed_actions = fixed_actions
        self._key = self._trajectory = None

    def _at(self, at):
        at = self.domain.array(at, name="at", copy=True)
        key = at.tobytes(order="C")
        if key != self._key:
            trajectory = self._solve(at)
            if not self._owned_trajectory:
                trajectory = self._copy_trajectory(trajectory)
            self.codomain.array(self._states(trajectory), name="state trajectory")
            self._trajectory, self._key = trajectory, key
        return self._trajectory

    def predict(self, x):
        """Вычислить сеточное состояние по аргументу.

        Parameters
        ----------
        x : array_like
            Аргумент формы domain.shape.

        Returns
        -------
        state : ndarray
            Копия состояния формы codomain.shape в единицах codomain.
        """
        return self.codomain.array(self._states(self._at(x)), name="prediction", copy=True)

    def jvp(self, at, direction):
        """Применить производную состояния к направлению.

        Parameters
        ----------
        at : array_like
            Точка формы domain.shape.
        direction : array_like
            Направление формы domain.shape.

        Returns
        -------
        tangent : ndarray
            Действие производной формы codomain.shape.
        """
        action = require_capability(self._tangent, operation="JVP", consumer="state map")
        direction = self.domain.array(direction, name="direction", copy=True)
        return self.codomain.array(action(self._at(at), direction), name="JVP output", copy=True)

    def vjp(self, at, cotangent):
        """Перенести ковектор состояния к аргументу.

        Parameters
        ----------
        at : array_like
            Точка формы domain.shape.
        cotangent : array_like
            Ковектор формы codomain.shape.

        Returns
        -------
        covector : ndarray
            Ковектор аргумента формы domain.shape.
        """
        action = require_capability(self._pullback, operation="VJP", consumer="state map")
        cotangent = self.codomain.array(cotangent, name="cotangent", copy=True)
        return self.domain.array(action(self._at(at), cotangent), name="VJP output", copy=True)

    def linear_operator(self, at):
        """Зафиксировать производную в заданной траектории.

        Parameters
        ----------
        at : array_like
            Точка формы domain.shape; нужны обе производные JVP и VJP.

        Returns
        -------
        operator : scipy.sparse.linalg.LinearOperator
            Оператор формы (codomain.size, domain.size) в порядке
            координат C, с собственной копией траектории.
        """
        tangent = require_capability(self._tangent, operation="JVP", consumer="linear operator")
        pullback = require_capability(self._pullback, operation="VJP", consumer="linear operator")
        trajectory = self._copy_trajectory(self._at(at))
        if self._fixed_actions is not None:
            tangent, pullback = self._fixed_actions()

        def matvec(h):
            h = self.domain.array(np.asarray(h).reshape(self.domain.shape),
                                  name="direction", copy=True)
            return self.codomain.array(tangent(trajectory, h), name="JVP output", copy=True).ravel()

        def rmatvec(z):
            z = self.codomain.array(np.asarray(z).reshape(self.codomain.shape),
                                    name="cotangent", copy=True)
            return self.domain.array(pullback(trajectory, z), name="VJP output", copy=True).ravel()

        return LinearOperator((self.codomain.size, self.domain.size),
                              dtype=np.result_type(self.domain.dtype, self.codomain.dtype),
                              matvec=matvec, rmatvec=rmatvec)


def _validate_callbacks(*, domain, codomain, solve, jvp, vjp, states, copy_trajectory):
    if not isinstance(domain, ArraySpace) or not isinstance(codomain, ArraySpace):
        raise TypeError("domain and codomain must be ArraySpace records")
    for name, action in (("solve", solve), ("states", states),
                         ("copy_trajectory", copy_trajectory)):
        if not callable(action):
            raise TypeError(f"{name} must be callable")
    for name, action in (("jvp", jvp), ("vjp", vjp)):
        if action is not None and not callable(action):
            raise TypeError(f"{name} must be callable or None")


def _capabilities(action):
    return MapCapabilities(action, action.domain, action.codomain,
                           action if action._tangent is not None else None,
                           action if action._pullback is not None else None)


def state_map_from_solver(*, domain, codomain, solve, jvp=None, vjp=None,
                          states=_states_data, copy_trajectory=deepcopy):
    """Связать внешний решатель с пространствами массивов.

    Parameters
    ----------
    domain, codomain : ArraySpace
        Пространства входа решателя и состояния.
    solve : callable
        Принимает массив формы domain.shape и возвращает траекторию.
    jvp : callable or None, optional
        Принимает траекторию и направление формы domain.shape;
        возвращает массив формы codomain.shape.
    vjp : callable or None, optional
        Принимает траекторию и ковектор формы codomain.shape;
        возвращает ковектор формы domain.shape.
    states : callable, optional
        Извлекает из траектории состояние формы codomain.shape; по
        умолчанию возвращает атрибут states.
    copy_trajectory : callable, optional
        Создаёт независимую копию траектории; по умолчанию deepcopy.

    Returns
    -------
    mapping : MapCapabilities
        Прогноз и явно заданные производные по входу решателя.

    Notes
    -----
    Последняя траектория используется повторно. Настройки переданных
    функций должны оставаться неизменными во время применения отображения.
    """
    _validate_callbacks(domain=domain, codomain=codomain, solve=solve,
                        jvp=jvp, vjp=vjp, states=states, copy_trajectory=copy_trajectory)
    return _capabilities(_TrajectoryMap(domain=domain, codomain=codomain,
        solve=solve, jvp=jvp, vjp=vjp, states=states, copy_trajectory=copy_trajectory))


def _bounded_methods(model, *, forcing):
    suffix = "forcing" if forcing else "controls"
    names = (f"solve_{suffix}", f"tangent_{suffix}", f"adjoint_{suffix}", "snapshot")
    missing = [name for name in names if not callable(getattr(model, name, None))]
    if missing:
        raise TypeError(f"A bounded state backend requires callable methods: {', '.join(missing)}")
    return names[:3]


def _bounded_layout(model, domain, codomain, *, forcing):
    if not isinstance(domain, ArraySpace) or not isinstance(codomain, ArraySpace):
        raise TypeError("ArraySpace domain and codomain required")
    missing = [name for name in ("nt", "size", "times", "xy", "units")
               if not hasattr(model, name)]
    if missing:
        raise TypeError(f"A bounded state backend requires metadata: {', '.join(missing)}")
    for name in ("nt", "size"):
        value = getattr(model, name)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"Bounded backend {name} must be a positive integer")
    times, xy = np.asarray(model.times), np.asarray(model.xy)
    if (times.shape != (model.nt+1,) or times.dtype.kind not in "iuf"
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
        raise ValueError("Bounded backend must declare finite increasing time nodes")
    if (xy.shape != (model.size, 2) or xy.dtype.kind not in "iuf" or not np.isfinite(xy).all()):
        raise ValueError("Bounded backend must declare finite x-fast geometry")
    expected_domain = (model.nt+1, model.size) if forcing else (model.nt+1,)
    if (domain.shape != expected_domain or codomain.shape != (model.nt+1, model.size)
            or domain.dtype != np.dtype("float64") or codomain.dtype != domain.dtype):
        raise ValueError("State map requires float64 time controls and time-by-node states")
    if domain.coordinates[0] != tuple(times) or codomain.coordinates[0] != tuple(times):
        raise ValueError("Input coordinates must match backend time nodes")
    if forcing and domain.coordinates != codomain.coordinates:
        raise ValueError("Forcing and state coordinates must match")
    node_coordinates, node_indices = tuple(map(tuple, xy)), tuple(range(model.size))
    if codomain.coordinates[1] not in (node_coordinates, node_indices):
        raise ValueError("State node coordinates must match backend x-fast geometry or ordered node indices")
    if model.units is not None:
        if not isinstance(model.units, ADRUnits):
            raise TypeError("Bounded backend units must be ADRUnits or None")
        if codomain.units != model.units.concentration:
            raise ValueError("State unit must match the backend concentration unit")
        if forcing and domain.units != model.units.reaction:
            raise ValueError("Forcing unit must match backend concentration/time")
        if not forcing and domain.units != model.units.source:
            raise ValueError("Point-source control unit must match backend concentration*length^2/time")


def _bounded_state_map(model, domain, codomain, *, forcing):
    names = _bounded_methods(model, forcing=forcing)
    _bounded_layout(model, domain, codomain, forcing=forcing)
    geometry = np.array(model.xy, copy=True)
    units = model.units
    snapshot = model.snapshot()
    _bounded_methods(snapshot, forcing=forcing)
    _bounded_layout(snapshot, domain, codomain, forcing=forcing)
    if not np.array_equal(snapshot.xy, geometry):
        raise ValueError("Solver snapshot must preserve x-fast geometry")
    if snapshot.units != units:
        raise ValueError("Solver snapshot must preserve declared units")
    solve_name, tangent_name, pullback_name = names
    # При вызове использовать операции независимой копии решателя.
    action = _TrajectoryMap(domain=domain, codomain=codomain,
        solve=lambda x: getattr(snapshot, solve_name)(x),
        jvp=lambda trajectory, h: getattr(snapshot, tangent_name)(trajectory, h),
        vjp=lambda trajectory, z: getattr(snapshot, pullback_name)(trajectory, z),
        states=_states_data, copy_trajectory=deepcopy, owned_trajectory=True,
        fixed_actions=lambda: (getattr(snapshot, tangent_name), getattr(snapshot, pullback_name)))
    return _capabilities(action)


def state_map(model, *, domain, codomain):
    """Связать точечную нагрузку с состоянием ADR.

    Parameters
    ----------
    model : object
        ADR-решатель с прямым решением, производными и независимым
        snapshot.
    domain : ArraySpace
        Пространство формы (nt+1,) с dtype float64. Интенсивности
        точечного источника в единицах концентрация·длина²/время.
    codomain : ArraySpace
        Пространство концентрации формы (nt+1, size) с dtype float64.
        Время совпадает с решателем; узлы заданы координатами или
        порядковыми индексами.

    Returns
    -------
    mapping : MapCapabilities
        Прогноз и производные по нагрузке при фиксированных данных
        уравнения.

    Notes
    -----
    Решатель копируется при создании. Начальная нагрузка не влияет
    на состояние; её производная нулевая.
    """
    return _bounded_state_map(model, domain, codomain, forcing=False)


def forcing_state_map(model, *, domain, codomain):
    """Связать распределённую нагрузку с состоянием ADR.

    Parameters
    ----------
    model : object
        ADR-решатель с прямым решением, производными и независимым
        snapshot.
    domain : ArraySpace
        Пространство формы (nt+1, size) с dtype float64.
        Распределённая нагрузка в единицах концентрация/время.
    codomain : ArraySpace
        Пространство концентрации формы (nt+1, size) с dtype float64.
        Время совпадает с решателем; узлы заданы координатами или
        порядковыми индексами.

    Returns
    -------
    mapping : MapCapabilities
        Прогноз и производные по нагрузке при фиксированных данных
        уравнения.

    Notes
    -----
    Решатель копируется при создании. Начальная нагрузка не влияет
    на состояние; её производная нулевая.
    """
    return _bounded_state_map(model, domain, codomain, forcing=True)


__all__ = ["state_map_from_solver", "state_map", "forcing_state_map"]
