"""Линейные пространственные и временные наблюдения сеточного состояния."""
import numpy as np

from adrkit.spaces import _LinearMap


def apply_observation(states, space_weights, time_weights):
    """Применить линейные пространственные и временные веса.

    Parameters
    ----------
    states : ndarray, shape (T, N)
        Состояние в порядке времени и пространственного узла.
    space_weights : ndarray, shape (R, N)
        Пространственные веса приёмников, включая нужную площадь и
        нормировку.
    time_weights : ndarray, shape (K, T)
        Веса временных функционалов, включая нужную длительность и
        нормировку.

    Returns
    -------
    observations : ndarray, shape (R*K,)
        Результаты в порядке приёмника, затем временного функционала;
        единицы определяются весами.
    """
    spatial_series = states @ space_weights.T
    return (spatial_series.T @ time_weights.T).ravel()


def transpose_observation(vector, space_weights, time_weights):
    """Перенести ковектор наблюдений к сеточному состоянию.

    Parameters
    ----------
    vector : ndarray, shape (R*K,)
        Ковектор в порядке приёмника, затем временного функционала.
    space_weights : ndarray, shape (R, N)
        Пространственные веса приёмников, включая нужную площадь и
        нормировку.
    time_weights : ndarray, shape (K, T)
        Веса временных функционалов, включая нужную длительность и
        нормировку.

    Returns
    -------
    covector : ndarray, shape (T, N)
        Координатный ковектор состояния в евклидовом скалярном
        произведении.
    """
    receivers = vector.reshape(len(space_weights), len(time_weights))
    return time_weights.T @ receivers.T @ space_weights


class GridObservation(_LinearMap):
    """Линейное наблюдение сеточного состояния.

    Parameters
    ----------
    space_weights : array_like, shape (R, N)
        Пространственные веса приёмников, включая нужную площадь и
        нормировку.
    time_weights : array_like, shape (K, T)
        Веса временных функционалов, включая нужную длительность и
        нормировку.
    domain : ArraySpace
        Пространство состояния формы (T, N).
    codomain : ArraySpace
        Пространство наблюдений формы (R*K,) с требуемыми единицами.

    Notes
    -----
    Матрицы весов копируются. Знаковые веса допустимы; единицы результата
    и нормировка задаются весами и codomain. Порядок строк: приёмник,
    затем временной функционал.
    """

    def __init__(self, space_weights, time_weights, *, domain, codomain):
        if np.iscomplexobj(space_weights) or np.iscomplexobj(time_weights):
            raise ValueError("Observation weights must be real")
        self._space_weights = np.array(space_weights, dtype=domain.dtype, copy=True)
        self._time_weights = np.array(time_weights, dtype=domain.dtype, copy=True)
        s, w = self._space_weights, self._time_weights
        if (s.ndim != 2 or w.ndim != 2 or not np.isfinite(s).all()
                or not np.isfinite(w).all() or domain.shape != (w.shape[1], s.shape[1])
                or codomain.shape != (s.shape[0]*w.shape[0],)):
            raise ValueError("S/W shapes must match state and receiver-major observation spaces")
        super().__init__(domain, codomain, self._matvec, self._rmatvec)

    def _matvec(self, x):
        return apply_observation(x.reshape(self.domain.shape),
                                 self._space_weights, self._time_weights)

    def _rmatvec(self, z):
        return transpose_observation(z.reshape(-1),
                                     self._space_weights, self._time_weights).ravel()

    @property
    def space_weights(self):
        """Независимая копия пространственных весов формы (R, N)."""
        return self._space_weights.copy()

    @property
    def time_weights(self):
        """Независимая копия временных весов формы (K, T)."""
        return self._time_weights.copy()


def observation_map(space_weights, time_weights, *, domain, codomain):
    """Создать линейный оператор наблюдения с производными.

    Parameters
    ----------
    space_weights : array_like, shape (R, N)
        Пространственные веса приёмников, включая нужную площадь и
        нормировку.
    time_weights : array_like, shape (K, T)
        Веса временных функционалов, включая нужную длительность и
        нормировку.
    domain, codomain : ArraySpace
        Пространства состояния формы (T, N) и наблюдений формы
        (R*K,).

    Returns
    -------
    mapping : MapCapabilities
        Прогноз и JVP/VJP оператора GridObservation.
    """
    return GridObservation(space_weights, time_weights,
                           domain=domain, codomain=codomain).capabilities()
