"""Наблюдения линейной ADR на плоскости по гауссову ядру."""
from dataclasses import dataclass, replace
import numpy as np
from numpy.polynomial.legendre import leggauss
from adrkit.spaces import MatrixMap


def _constant(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real constant")
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "iuf" or not np.isfinite(array):
        raise ValueError(f"{name} must be a finite real constant")
    return float(array)


@dataclass(frozen=True)
class Transport:
    """Постоянные коэффициенты ADR и гауссового наблюдения.

    Parameters
    ----------
    diffusion : float
        Положительная диффузия в единицах длина²/время.
    velocity : array_like, shape (2,)
        Конечная скорость в единицах длина/время.
    reaction : float
        Неотрицательные линейные потери в обратных единицах времени.
    footprint : float
        Положительное стандартное отклонение пространственного
        усреднения по каждой координате в единицах длины.
    """
    diffusion: float
    velocity: tuple[float, float]
    reaction: float
    footprint: float

    def __post_init__(self):
        for name in ("diffusion", "reaction", "footprint"):
            object.__setattr__(self, name, _constant(getattr(self, name), name))
        try:
            velocity = tuple(self.velocity)
        except TypeError as error:
            raise ValueError("A finite real two-component velocity is required") from error
        if len(velocity) != 2:
            raise ValueError("A finite real two-component velocity is required")
        object.__setattr__(self, "velocity", tuple(_constant(value, "velocity") for value in velocity))
        if self.diffusion <= 0 or self.footprint <= 0:
            raise ValueError("Finite positive diffusion and footprint are required")
        if self.reaction < 0:
            raise ValueError("Nonnegative reaction and a 2D velocity are required")

    def normalized(self, a, time_scale):
        r"""Пересчитать коэффициенты при смене координат и времени.

        Parameters
        ----------
        a : float
            Положительный множитель пространственных координат.
        time_scale : float
            Положительный делитель временной координаты.

        Returns
        -------
        transport : Transport
            Коэффициенты в координатах, заданных новыми масштабами.

        Notes
        -----
        Для исходных координат :math:`x,t` новые координаты равны

        .. math::

            x' = a x,\qquad t' = t / T,

        где :math:`T` — `time_scale`.
        """
        if a <= 0 or time_scale <= 0:
            raise ValueError("Positive coordinate/time scales are required")
        return replace(self, diffusion=time_scale*a*a*self.diffusion,
                       velocity=tuple(time_scale*a*np.asarray(self.velocity)),
                       reaction=time_scale*self.reaction, footprint=a*self.footprint)


def _spatial_point(value, name):
    """Проверить конечные вещественные координаты точки на плоскости."""
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    point = np.asarray(value, dtype=float)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError(f"{name} must be a finite two-component point")
    return point


def gaussian_kernel(lag, receiver, source, model):
    r"""Вычислить ядро ADR с гауссовым пространственным усреднением.

    Parameters
    ----------
    lag : array_like
        Конечное вещественное неотрицательное запаздывание в единицах времени
        model.
    receiver, source : array_like, shape (2,)
        Конечные вещественные координаты приёмника и точечного источника в единицах длины
        model.
    model : Transport
        Постоянные коэффициенты уравнения и ширина усреднения.

    Returns
    -------
    kernel : ndarray or numpy.floating
        Ядро формы lag в обратных единицах площади.

    Notes
    -----
    Для :math:`\sigma^2=w^2+2D\ell`:

    .. math::

        K(\ell,r,x_0)=\frac{1}{2\pi\sigma^2}
        \exp\!\left(-\kappa\ell
        -\frac{\|r-x_0-v\ell\|^2}{2\sigma^2}\right).

    Здесь :math:`\ell` — `lag`, :math:`r` — `receiver`,
    :math:`x_0` — `source`, :math:`w` — `model.footprint`.
    Коэффициенты :math:`D,v,\kappa` соответствуют `model.diffusion`,
    `model.velocity` и `model.reaction`.
    """
    if np.iscomplexobj(lag):
        raise ValueError("Lag must be real")
    lag = np.asarray(lag, dtype=float)
    if not np.isfinite(lag).all() or np.any(lag < 0):
        raise ValueError("Lag must be finite and nonnegative")
    receiver = _spatial_point(receiver, "receiver")
    source = _spatial_point(source, "source")
    r = receiver-source
    variance = model.footprint**2+2*model.diffusion*lag
    dx = r[0]-model.velocity[0]*lag
    dy = r[1]-model.velocity[1]*lag
    return np.exp(-model.reaction*lag-(dx*dx+dy*dy)/(2*variance))/(2*np.pi*variance)


def observation_matrix(times, receivers, source, edges, model, order=24):
    """Построить наблюдения кусочно-постоянного источника.

    Parameters
    ----------
    times : array_like, shape (m,)
        Конечные времена наблюдений.
    receivers : array_like, shape (m, 2)
        Координаты приёмника для каждой строки наблюдения.
    source : array_like, shape (2,)
        Положение точечного источника.
    edges : array_like, shape (n+1,)
        Конечные строго возрастающие границы интервалов
        интенсивности.
    model : Transport
        Коэффициенты в согласованных единицах времени и длины.
    order : int, optional
        Порядок квадратуры Гаусса — Лежандра, не меньше двух.

    Returns
    -------
    matrix : ndarray, shape (m, n)
        Столбцы соответствуют интервалам edges, строки — парам
        времени и приёмника; единицы время/площадь.

    Notes
    -----
    Временные интегралы ядра приближённо вычисляются квадратурой.
    Интервалы после времени наблюдения дают нулевой вклад.
    """
    times, receivers, source, edges = map(
        lambda x: np.asarray(x, dtype=float), (times, receivers, source, edges))
    if times.ndim != 1 or receivers.shape != (len(times), 2) or source.shape != (2,):
        raise ValueError("Incompatible observation arrays")
    if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("Strictly increasing bin edges are required")
    if not all(np.isfinite(x).all() for x in (times, receivers, source, edges)):
        raise ValueError("All observation inputs must be finite")
    if isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer)) or order < 2:
        raise ValueError("Quadrature order must be an integer >= 2")
    nodes, weights = leggauss(order)
    A = np.zeros((len(times), len(edges)-1))
    for receiver in np.unique(receivers, axis=0):
        sel = np.all(receivers == receiver, axis=1)
        t = times[sel, None]
        lower = np.broadcast_to(edges[:-1], (len(t), len(edges)-1))
        upper = np.minimum(t, edges[1:])
        width = np.maximum(0, upper-lower)
        s = lower[..., None]+0.5*width[..., None]*(nodes+1)
        lag = np.maximum(0, t[..., None]-s)
        A[sel] = 0.5*width*np.sum(gaussian_kernel(lag, receiver, source, model)*weights, axis=-1)
    return A


class WholeSpacePredictor(MatrixMap):
    """Линейный прогноз концентрации на плоскости.

    Parameters
    ----------
    times : array_like, shape (m,)
        Конечные времена наблюдений в единицах времени `model`.
    receivers : array_like, shape (m, 2)
        Координаты приёмника каждой строки в единицах длины `model`.
    source : array_like, shape (2,)
        Положение источника в тех же единицах длины.
    edges : array_like, shape (n+1,)
        Конечные строго возрастающие границы интервалов интенсивности
        в тех же единицах времени.
    model : Transport
        Постоянные коэффициенты переноса и усреднения.
    domain : ArraySpace
        Вектор коэффициентов формы (len(edges)−1,) в единицах
        интенсивности.
    codomain : ArraySpace
        Вектор наблюдений формы (len(times),) в единицах
        концентрации.
    order : int, optional
        Порядок квадратуры, не меньше двух.

    See Also
    --------
    observation_matrix : Порядок строк, столбцов и единицы матрицы.
    """
    def __init__(self, times, receivers, source, edges, model, *, domain, codomain, order=24):
        matrix = observation_matrix(times, receivers, source, edges, model, order)
        if domain.shape != (matrix.shape[1],) or codomain.shape != (matrix.shape[0],):
            raise ValueError("P0 predictor requires coefficient and observation vectors")
        super().__init__(matrix, domain, codomain)
