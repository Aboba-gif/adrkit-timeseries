"""Линейные наблюдения источника P1 по ядру ADR на плоскости."""
from dataclasses import dataclass
import numpy as np
from numpy.polynomial.legendre import leggauss

from adrkit.backends.whole_space import Transport, gaussian_kernel
from adrkit.sources import P1Basis
from adrkit.spaces import matrix_map


@dataclass(frozen=True, init=False)
class SensorSpec:
    """Времена, приёмники и окна наблюдения.

    Parameters
    ----------
    times : array_like, shape (m,)
        Непустой вектор конечных времён в time_unit.
    receivers : array_like, shape (m, 2)
        Конечные координаты приёмника каждой строки в length_unit.
    windows : array_like, shape (m,), or None, optional
        Положительные ширины предшествующих окон в time_unit; None
        задаёт мгновенные наблюдения.
    time_unit, length_unit : str, optional
        Обозначения согласованных единиц времени и длины.

    Notes
    -----
    Порядок строк сохраняется. Каждое временное окно заканчивается
    в момент соответствующего наблюдения; среднее делится на полную ширину.
    """
    times: tuple
    receivers: tuple
    windows: tuple | None
    time_unit: str
    length_unit: str

    def __init__(self, times, receivers, *, windows=None, time_unit="s", length_unit="m"):
        times, receivers = np.asarray(times, float), np.asarray(receivers, float)
        if (times.ndim != 1 or not len(times) or receivers.shape != (len(times), 2)
                or not np.isfinite(times).all() or not np.isfinite(receivers).all()):
            raise ValueError("Finite times and paired 2D receivers required")
        if not time_unit or not length_unit:
            raise ValueError("Explicit time and length units required")
        if windows is not None:
            windows = np.asarray(windows, float)
            if (windows.shape != times.shape or not np.isfinite(windows).all()
                    or np.any(windows <= 0)):
                raise ValueError("One positive backward window width per row required")
            windows = tuple(windows.tolist())
        object.__setattr__(self, 'times', tuple(times.tolist()))
        object.__setattr__(self, 'receivers', tuple(map(tuple, receivers.tolist())))
        object.__setattr__(self, 'windows', windows)
        object.__setattr__(self, 'time_unit', time_unit)
        object.__setattr__(self, 'length_unit', length_unit)


@dataclass(frozen=True)
class SpatialSource:
    """Фиксированное положение точечного источника на плоскости.

    Parameters
    ----------
    position : array_like, shape (2,)
        Конечные координаты источника.
    length_unit : str, optional
        Единица координат position.
    """
    position: tuple
    length_unit: str = "m"

    def __post_init__(self):
        position = np.asarray(self.position, float)
        if position.shape != (2,) or not np.isfinite(position).all() or not self.length_unit:
            raise ValueError("A finite fixed spatial delta position and length unit required")
        object.__setattr__(self, 'position', tuple(position.tolist()))


@dataclass(frozen=True)
class Quadrature:
    """Порядок временной квадратуры Гаусса — Лежандра.

    Parameters
    ----------
    order : int, optional
        Число узлов на подынтервале; не меньше двух.
    """
    order: int = 24

    def __post_init__(self):
        if isinstance(self.order, bool) or not isinstance(self.order, (int, np.integer)) or self.order < 2:
            raise ValueError("Gauss order must be an integer >= 2")


def p1_response(basis, sensors, transport, spatial_source, *, quadrature,
                coefficient_scale, domain, codomain):
    r"""Создать линейный отклик кусочно-линейного источника.

    Parameters
    ----------
    basis : P1Basis
        Временной базис с коэффициентами в порядке узлов.
    sensors : SensorSpec
        Пары времени и приёмника с необязательными окнами.
    transport : Transport
        Постоянные коэффициенты ADR и положительная ширина
        пространственного усреднения.
    spatial_source : SpatialSource
        Положение точечного источника на плоскости.
    quadrature : Quadrature
        Порядок приближённого временного интегрирования.
    coefficient_scale : float
        Конечный положительный масштаб интенсивности источника.
    domain : ArraySpace
        Вектор формы (basis.size,) с координатами узлов базиса.
    codomain : ArraySpace
        Вектор формы (len(sensors.times),) в порядке строк
        наблюдений.

    Returns
    -------
    response : MapCapabilities
        Прогноз и JVP применяют матрицу отклика; VJP — её
        транспонирование.

    Notes
    -----
    Для мгновенного наблюдения матрица приближает интеграл

    .. math::

        A_{ij} \approx s\int_{t_0}^{\min(t_i,t_N)}
            K(t_i-u,r_i,x_0)\phi_j(u)\,du,\qquad t_i>t_0.

    Здесь :math:`s` — `coefficient_scale`, :math:`\phi_j` —
    функция `basis`, :math:`K` — `gaussian_kernel`,
    :math:`t_0,t_N` — крайние узлы `basis.knots`.
    Время :math:`t_i` и положение :math:`r_i` задаёт строка `sensors`,
    :math:`x_0` — `spatial_source.position`; :math:`u` — время источника.
    До начала интервала базиса отклик нулевой. Оконные наблюдения
    дополнительно усредняются по времени с делением на полную ширину.
    Численные единицы времени, длины и интенсивности согласует вызывающий код.
    """
    if not isinstance(basis, P1Basis) or not isinstance(sensors, SensorSpec):
        raise TypeError("P1Basis and SensorSpec required")
    if not isinstance(transport, Transport) or not isinstance(spatial_source, SpatialSource):
        raise TypeError("Transport and fixed SpatialSource required")
    if not isinstance(quadrature, Quadrature):
        raise TypeError("Explicit Quadrature required")
    if (basis.time_unit != sensors.time_unit
            or sensors.length_unit != spatial_source.length_unit):
        raise ValueError("Source and sensors must use matching time/length units")
    if (isinstance(coefficient_scale, (bool, np.bool_))
            or not np.isfinite(coefficient_scale) or coefficient_scale <= 0):
        raise ValueError("Positive finite coefficient scale required")
    if (domain.shape != (basis.size,) or codomain.shape != (len(sensors.times),)
            or domain.coordinates != (tuple(basis.knots),)):
        raise ValueError("Spaces must match ordered P1 knots and observation rows")
    nodes, weights = leggauss(quadrature.order)
    knots = basis.knots

    def snapshot(time, receiver):
        upper = np.minimum(time, knots[1:])
        widths = np.maximum(0., upper-knots[:-1])
        source_times = knots[:-1, None]+.5*widths[:, None]*(nodes+1)
        kernel = gaussian_kernel(np.maximum(0., time-source_times), receiver,
                                 spatial_source.position, transport)
        weighted = .5*widths[:, None]*weights*kernel
        fraction = (source_times-knots[:-1, None])/np.diff(knots)[:, None]
        values = np.zeros(basis.size)
        values[:-1] += np.sum(weighted*(1-fraction), axis=1)
        values[1:] += np.sum(weighted*fraction, axis=1)
        return values

    matrix = np.zeros((len(sensors.times), basis.size))
    for row, (time, receiver) in enumerate(zip(sensors.times, sensors.receivers)):
        if sensors.windows is None:
            matrix[row] = snapshot(time, receiver)
        else:
            width = sensors.windows[row]
            lower = max(knots[0], time-width)
            if lower >= time:
                continue
            breaks = np.unique(np.r_[lower, knots[(knots > lower) & (knots < time)], time])
            for left, right in zip(breaks[:-1], breaks[1:]):
                obs_times = left+.5*(right-left)*(nodes+1)
                responses = np.array([snapshot(t, receiver) for t in obs_times])
                matrix[row] += .5*(right-left)*(weights @ responses)/width
    return matrix_map(coefficient_scale*matrix, domain=domain, codomain=codomain)
