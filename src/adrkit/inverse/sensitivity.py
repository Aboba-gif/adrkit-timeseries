"""Спектр локальной чувствительности в метриках коэффициентов и наблюдений."""
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.linalg import solve_triangular

from adrkit.contracts import MapCapabilities, require_capability
from adrkit.errors import CapabilityError
from adrkit.spaces import ArraySpace, matrix_map
from .misfit import covariance_metric
from .regularization import CoefficientMetric
from ._numeric import real_array, real_scalar


class CoefficientWhitening(Protocol):
    r"""Корень фиксированной метрики пространства коэффициентов.

    Attributes
    ----------
    domain : ArraySpace
        Фиксированное пространство коэффициентов.
    root : ndarray, shape (domain.size, domain.size)
        Верхнетреугольный корень с положительной диагональю.

    Notes
    -----
    Для симметричной положительно определённой матрицы метрики
    :math:`G` и верхнетреугольной матрицы :math:`R` из `root`:

    .. math::

        G=R^\mathsf{T}R.
    """
    domain: ArraySpace
    root: np.ndarray


class ObservationWhitening(Protocol):
    r"""Преобразование фиксированной ковариации к единичной.

    Attributes
    ----------
    layout : ArraySpace
        Фиксированное пространство наблюдений в установленном порядке
        строк.

    Notes
    -----
    Пусть :math:`\Sigma` — фиксированная симметричная положительно
    определённая ковариация, :math:`L` — её нижний фактор Холецкого,
    :math:`A` — входная матрица. Операция `whiten_matrix` задаёт

    .. math::

        \Sigma=LL^\mathsf{T},\qquad \widetilde A=L^{-1}A.

    Порядок строк сохраняется.
    """
    layout: ArraySpace

    def whiten_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Привести строки матрицы к единичной ковариации.

        Parameters
        ----------
        matrix : ndarray, shape (layout.size, n)
            Матрица с наблюдениями по строкам.

        Returns
        -------
        whitened : ndarray, shape (layout.size, n)
            Матрица в том же порядке строк и столбцов.
        """
        ...


@dataclass(frozen=True)
class SensitivityLimits:
    """Пределы построения плотного спектра чувствительности.

    Parameters
    ----------
    max_columns : int
        Положительный предел числа параметров.
    max_entries : int
        Положительный предел числа элементов плотной матрицы.
    max_jvp_calls : int
        Положительный предел вызовов JVP.
    """
    max_columns: int
    max_entries: int
    max_jvp_calls: int

    def __post_init__(self):
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1
               for v in (self.max_columns, self.max_entries, self.max_jvp_calls)):
            raise ValueError("Positive integer dense sensitivity limits required")


@dataclass(frozen=True)
class SpectrumResult:
    """Спектр чувствительности и численные характеристики ранга.

    Parameters
    ----------
    singular_values : ndarray
        Сингулярные числа в убывающем порядке.
    rank, nullity : int
        Число разрешённых направлений и размер численного ядра.
    threshold : float
        Абсолютный порог разрешения сингулярных чисел.
    condition : float or None
        Отношение крайних разрешённых сингулярных чисел; None при
        нулевом ранге или неконечном отношении.
    condition_reason : str or None
        Причина отсутствующей condition.
    at : ndarray
        Точка, в которой вычислен спектр, в пространстве
        coefficient_space.
    coefficient_space, observation_space : ArraySpace
        Пространства коэффициентов и наблюдений.

    Notes
    -----
    В фактическом результате sensitivity_spectrum массивы спектра
    и точки имеют запрещённую запись.
    """
    singular_values: np.ndarray
    rank: int
    nullity: int
    threshold: float
    condition: float | None
    condition_reason: str | None
    at: np.ndarray
    coefficient_space: ArraySpace
    observation_space: ArraySpace

    def as_tuple(self):
        """Получить численные характеристики спектра кортежем.

        Returns
        -------
        spectrum : ndarray
            Независимая копия сингулярных чисел.
        rank, nullity : int
            Численный ранг и размер ядра.
        threshold : float
            Порог разрешения.
        condition : float
            Разрешённая обусловленность; бесконечность при отсутствующей
            condition.
        """
        return (self.singular_values.copy(), self.rank, self.nullity, self.threshold,
                float("inf") if self.condition is None else self.condition)


def sensitivity_spectrum(prediction, at, *, coefficient_metric,
                         observation_metric, relative_threshold=1e-10, limits):
    r"""Вычислить спектр локальной чувствительности в двух метриках.

    Parameters
    ----------
    prediction : MapCapabilities
        Прогноз с явно заданным JVP.
    at : array_like
        Конечная точка формы prediction.domain.shape.
    coefficient_metric : CoefficientWhitening
        Метрика на пространстве параметров prediction.
    observation_metric : ObservationWhitening
        Преобразование на пространстве наблюдений prediction.
    relative_threshold : float, optional
        Неотрицательная доля наибольшего сингулярного числа для
        оценки ранга.
    limits : SensitivityLimits or dict
        Явно заданные пределы плотного построения.

    Returns
    -------
    result : SpectrumResult
        Сингулярные числа, разрешённый ранг и обусловленность выше
        порога.

    Notes
    -----
    Спектр относится к матрице

    .. math::

        TJR^{-1},\qquad G=R^\mathsf{T}R,

    Здесь :math:`J` — якобиан `prediction` в точке `at`,
    :math:`G` — матрица метрики коэффициентов,
    :math:`R` — её верхнетреугольный корень `coefficient_metric.root`,
    :math:`T` — действие `observation_metric.whiten_matrix`.
    Координаты входных массивов развёрнуты в порядке C.
    Ранг определяется числом сингулярных чисел строго выше порога;
    это численный ранг, а не доказательство точного алгебраического ранга.
    """
    if not isinstance(prediction, MapCapabilities):
        raise TypeError("prediction must declare MapCapabilities")
    jvp = require_capability(prediction.jvp, operation="JVP", consumer="sensitivity")
    if not isinstance(getattr(coefficient_metric, "domain", None), ArraySpace):
        raise TypeError("Coefficient whitening must declare an ArraySpace domain")
    if (not isinstance(getattr(observation_metric, "layout", None), ArraySpace)
            or not callable(getattr(observation_metric, "whiten_matrix", None))):
        raise CapabilityError("sensitivity requires an explicit SPD whitening metric")
    prediction.domain.require_compatible(coefficient_metric.domain)
    prediction.codomain.require_compatible(observation_metric.layout)
    point = prediction.domain.array(at, copy=True)
    threshold_ratio = real_scalar(relative_threshold, name="Relative threshold")
    if not np.isfinite(threshold_ratio) or threshold_ratio < 0:
        raise ValueError("Finite nonnegative relative threshold required")
    if isinstance(limits, dict):
        limits = SensitivityLimits(**limits)
    if not isinstance(limits, SensitivityLimits):
        raise TypeError("Explicit SensitivityLimits required")
    m, n = prediction.codomain.size, prediction.domain.size
    if n > limits.max_columns or n > limits.max_jvp_calls or m*n > limits.max_entries:
        raise ValueError("Dense sensitivity exceeds caller limits")
    root = real_array(coefficient_metric.root, name="Coefficient whitening root", copy=True)
    if (root.shape != (n, n) or not np.array_equal(root, np.triu(root))
            or np.any(np.diag(root) <= 0)):
        raise ValueError("Coefficient whitening requires a finite positive upper-triangular root")
    matrix = np.empty((m, n), dtype=prediction.domain.dtype)
    for j in range(n):
        direction = np.zeros(prediction.domain.shape, dtype=prediction.domain.dtype)
        direction.ravel()[j] = 1.
        matrix[:, j] = prediction.codomain.array(
            jvp.jvp(point.copy(), direction), name="JVP output").ravel()
    white = real_array(observation_metric.whiten_matrix(matrix.copy()),
                       name="Whitened sensitivity")
    if white.shape != (m, n):
        raise ValueError("Observation whitening must preserve sensitivity row and column layout")
    kernel = solve_triangular(root.T, white.T, lower=True).T
    singular = np.linalg.svd(kernel, compute_uv=False)
    threshold = float(threshold_ratio*singular[0]) if len(singular) else 0.
    rank = int(np.sum(singular > threshold))
    condition = float(singular[0]/singular[rank-1]) if rank else None
    reason = "zero resolved rank" if rank == 0 else None
    if condition is not None and not np.isfinite(condition):
        condition, reason = None, "nonfinite resolved condition"
    singular.flags.writeable = point.flags.writeable = False
    return SpectrumResult(singular, rank, n-rank, threshold, condition, reason,
                          point, prediction.domain, prediction.codomain)


def matrix_spectrum(A, Sigma, M, relative_threshold=1e-10):
    """Вычислить спектр линейной матрицы в заданных метриках.

    Parameters
    ----------
    A : array_like, shape (m, n)
        Конечная вещественная матрица прогноза.
    Sigma : array_like, shape (m, m) or (m,)
        Положительно определённая ковариация наблюдений либо
        положительная диагональ.
    M : array_like, shape (n, n) or (n,)
        Положительно определённая метрика коэффициентов либо
        положительная диагональ.
    relative_threshold : float, optional
        Неотрицательный относительный порог ранга.

    Returns
    -------
    result : SpectrumResult
        Спектр линейной чувствительности в метриках Sigma и M.
    """
    A = real_array(A, name="A")
    if A.ndim != 2:
        raise ValueError("A must be a finite real matrix")
    m, n = A.shape
    domain = ArraySpace((n,), axes=("coefficient",), coordinates=(tuple(range(n)),), units="unknown")
    layout = ArraySpace((m,), axes=("observation",), coordinates=(tuple(range(m)),), units="unknown")
    return sensitivity_spectrum(matrix_map(A, domain=domain, codomain=layout), np.zeros(n),
        coefficient_metric=CoefficientMetric(M, domain=domain),
        observation_metric=covariance_metric(Sigma, layout=layout),
        relative_threshold=relative_threshold,
        limits=SensitivityLimits(n, m*n, n))
