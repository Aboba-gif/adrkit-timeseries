"""Невязка наблюдений в заданной метрике."""
import numpy as np
from scipy.linalg import cho_solve, norm, solve_triangular, svd

from adrkit.contracts import ScalarCapabilities
from . import linalg
from ._numeric import real_array, real_scalar


def _whitening_matrix(matrix, layout):
    """Проверить вещественную матрицу до и после смены dtype."""
    converted = real_array(matrix, name="Whitening matrix", dtype=layout.dtype)
    if converted.ndim != 2 or converted.shape[0] != layout.size:
        raise ValueError("Whitening matrix must have finite real rows matching the selected layout")
    return converted


class CovarianceMetric:
    """Метрика наблюдений с фиксированной ковариацией.

    Parameters
    ----------
    covariance : array_like, shape (layout.size, layout.size) or (layout.size,)
        Симметричная положительно определённая ковариация или
        вектор положительных диагональных элементов.
    layout : ArraySpace
        Пространство наблюдений; ковариация задаётся в порядке
        координат C.

    Notes
    -----
    Преобразование whiten приводит ковариацию к единичной матрице.
    Обратная ковариация применяется к координатному вектору невязки.
    """
    def __init__(self, covariance, *, layout):
        if np.iscomplexobj(covariance):
            raise ValueError("Covariance must be real")
        self._layout = layout
        self._factor = linalg._spd_factor(covariance, layout.size, "covariance")

    @property
    def layout(self):
        """Пространство наблюдений метрики."""
        return self._layout

    def apply_precision(self, residual):
        """Применить обратную ковариацию к невязке.

        Parameters
        ----------
        residual : array_like
            Конечная невязка формы layout.shape.

        Returns
        -------
        result : ndarray
            Координатное действие обратной ковариации. Форма layout.shape
            сохраняется.
        """
        r = self.layout.array(residual).ravel()
        result = (r/self._factor**2 if self._factor.ndim == 1 else
                  cho_solve((self._factor, True), r))
        return result.reshape(self.layout.shape)

    def whiten(self, residual):
        """Привести невязку к единичной ковариации.

        Parameters
        ----------
        residual : array_like
            Конечная невязка формы layout.shape.

        Returns
        -------
        result : ndarray
            Невязка в координатах с единичной ковариацией. Форма
            layout.shape сохраняется.
        """
        r = self.layout.array(residual).ravel()
        return (r/self._factor if self._factor.ndim == 1 else
                solve_triangular(self._factor, r, lower=True)).reshape(self.layout.shape)

    def whiten_matrix(self, matrix):
        """Преобразовать матрицу в координатах наблюдений.

        Parameters
        ----------
        matrix : array_like, shape (layout.size, k)
            Конечная вещественная матрица в порядке наблюдений.

        Returns
        -------
        transformed : ndarray, shape (layout.size, k)
            Матрица, умноженная слева на обратный нижний фактор
            Холецкого ковариации; для диагональной ковариации строки
            делятся на стандартные отклонения.
        """
        a = _whitening_matrix(matrix, self.layout)
        return (a/self._factor[:, None] if self._factor.ndim == 1 else
                solve_triangular(self._factor, a, lower=True))

    @property
    def logdet(self):
        """Логарифм определителя фиксированной ковариации."""
        diagonal = self._factor if self._factor.ndim == 1 else np.diag(self._factor)
        return float(2*np.log(diagonal).sum())

    @property
    def covariance_factor(self):
        """Получить копию фактора ковариации.

        Returns
        -------
        factor : ndarray
            Для диагональной ковариации — стандартные отклонения формы
            (layout.size,); иначе нижний фактор Холецкого.
        """
        return self._factor.copy()

    @property
    def covariance(self):
        """Полная ковариация формы (layout.size, layout.size) в независимом массиве."""
        return (np.diag(self._factor**2) if self._factor.ndim == 1 else
                self._factor @ self._factor.T)


def covariance_metric(covariance, *, layout):
    """Создать метрику фиксированной ковариации.

    Parameters
    ----------
    covariance : array_like, shape (layout.size, layout.size) or (layout.size,)
        Симметричная положительно определённая ковариация или
        вектор положительных диагональных элементов.
    layout : ArraySpace
        Пространство наблюдений; ковариация задаётся в порядке
        координат C.

    Returns
    -------
    metric : CovarianceMetric
        Метрика на пространстве layout.
    """
    return CovarianceMetric(covariance, layout=layout)


class WorkingPrecision:
    """Действие рабочей обратной ковариации наблюдений.

    Parameters
    ----------
    action : callable
        Фиксированное линейное симметричное положительно определённое
        действие. Принимает и возвращает массив формы layout.shape.
    layout : ArraySpace
        Пространство наблюдений.

    Notes
    -----
    Свойства линейности, симметрии и положительной определённости
    задаёт автор action. При вызове проверяются форма и конечность массива.
    """
    def __init__(self, action, *, layout):
        if not callable(action):
            raise ValueError("Fixed precision action required")
        self._action, self._layout = action, layout

    @property
    def layout(self):
        """Пространство наблюдений рабочей метрики."""
        return self._layout

    def apply_precision(self, residual):
        """Применить заданную рабочую обратную ковариацию.

        Parameters
        ----------
        residual : array_like
            Конечная невязка формы layout.shape.

        Returns
        -------
        result : ndarray
            Координатное действие заданной матрицы весов. Форма
            layout.shape сохраняется.
        """
        r = self.layout.array(residual, copy=True)
        return self.layout.array(self._action(r), name="precision output", copy=True)


def working_precision(action, *, layout):
    """Создать действие рабочей обратной ковариации.

    Parameters
    ----------
    action : callable
        Фиксированное линейное симметричное положительно определённое
        действие. Принимает и возвращает массив формы layout.shape.
    layout : ArraySpace
        Пространство наблюдений.

    Returns
    -------
    metric : WorkingPrecision
        Фиксированное действие на пространстве layout.
    """
    return WorkingPrecision(action, layout=layout)


class DiagonalWorkingMetric(WorkingPrecision):
    """Рабочая метрика с положительными диагональными весами.

    Parameters
    ----------
    weights : array_like
        Положительные конечные веса формы layout.shape, в единицах
        обратной ковариации.
    layout : ArraySpace
        Пространство наблюдений.
    """
    def __init__(self, weights, *, layout):
        weights = layout.array(weights, copy=True)
        if np.any(weights <= 0):
            raise ValueError("Select active rows before constructing working precision")
        self._weights = weights
        super().__init__(lambda r: weights*r, layout=layout)

    def whiten(self, residual):
        """Привести невязку к единичной рабочей ковариации.

        Parameters
        ----------
        residual : array_like
            Конечная невязка формы layout.shape.

        Returns
        -------
        result : ndarray
            Невязка, умноженная на квадратные корни диагональных весов.
            Форма layout.shape сохраняется.
        """
        return np.sqrt(self._weights)*self.layout.array(residual)

    def whiten_matrix(self, matrix):
        """Преобразовать матрицу в координатах наблюдений.

        Parameters
        ----------
        matrix : array_like, shape (layout.size, k)
            Конечная вещественная матрица в порядке наблюдений.

        Returns
        -------
        transformed : ndarray, shape (layout.size, k)
            Матрица, строки которой умножены на квадратные корни весов.
        """
        a = _whitening_matrix(matrix, self.layout)
        return np.sqrt(self._weights.ravel())[:, None]*a


class QuadraticMisfit:
    r"""Квадратичная невязка с фиксированными весами.

    Parameters
    ----------
    target : array_like
        Конечные наблюдения формы metric.layout.shape.
    metric : object
        Метрика с `layout` и `apply_precision(residual)`.
        Действие задаёт фиксированную линейную симметричную матрицу
        весов :math:`W`, независимую от `x`. Для неотрицательной
        квадратичной невязки :math:`W` должна быть положительно
        полуопределённой; положительная определённость не требуется.
    normalization : float, optional
        Конечный положительный делитель функционала.

    Notes
    -----
    Для невязки :math:`r=x-y`, матрицы весов :math:`W`
    и делителя :math:`n`:

    .. math::

        J(x)=\frac{r^\mathsf{T}Wr}{2n},\qquad
        dJ(x)=\frac{Wr}{n}.

    Здесь :math:`x` — прогноз, :math:`y` — `target`,
    :math:`n` — `normalization`. Матрица :math:`W` задаётся
    `metric.apply_precision`. В формулах массивы развёрнуты в порядке C;
    :math:`dJ` обозначает координатный ковектор, без преобразования Рисса.
    """
    def __init__(self, target, metric, *, normalization=1.):
        self._domain = metric.layout
        self._target = self.domain.array(target, name="selected target", copy=True)
        self._metric = metric
        self._normalization = real_scalar(normalization, name="Normalization")
        if not np.isfinite(self.normalization) or self.normalization <= 0:
            raise ValueError("Normalization must be finite and positive")

    @property
    def domain(self):
        """Пространство прогнозируемых наблюдений."""
        return self._domain

    @property
    def metric(self):
        """Фиксированная метрика наблюдений."""
        return self._metric

    @property
    def normalization(self):
        """Положительный делитель квадратичной невязки."""
        return self._normalization

    @property
    def target(self):
        """Независимая копия целевых наблюдений формы domain.shape."""
        return self._target.copy()

    def _residual(self, x):
        return self.domain.array(x)-self._target

    def value(self, x):
        """Вычислить квадратичную невязку.

        Parameters
        ----------
        x : array_like
            Конечный прогноз формы domain.shape в единицах target.

        Returns
        -------
        result : float
            Значение функционала с заданной нормировкой.
        """
        residual = self._residual(x)
        return .5*float(residual.ravel() @ self.metric.apply_precision(residual).ravel())/self.normalization

    def cotangent(self, x):
        """Вычислить координатный ковектор невязки.

        Parameters
        ----------
        x : array_like
            Конечный прогноз формы domain.shape в единицах target.

        Returns
        -------
        result : ndarray
            Производная формы domain.shape.
        """
        return self.metric.apply_precision(self._residual(x))/self.normalization

    def capabilities(self):
        """Предоставить значение и ковектор как ScalarCapabilities."""
        return ScalarCapabilities(self, self.domain, self)


class GaussianLikelihood(QuadraticMisfit):
    r"""Гауссов функционал наблюдений с фиксированной ковариацией.

    Parameters
    ----------
    target : array_like
        Конечные наблюдения формы metric.layout.shape.
    metric : CovarianceMetric
        Положительно определённая ковариация наблюдений.

    Notes
    -----
    Отрицательный логарифм плотности равен

    .. math::

        J(x)=\frac12\left[(x-y)^\mathsf{T}\Sigma^{-1}(x-y)
        +m\log(2\pi)+\log\det\Sigma\right],

    Здесь :math:`x` — прогноз, :math:`y` — `target`,
    :math:`\Sigma` — фиксированная симметричная положительно
    определённая ковариация `metric`, :math:`m` — `metric.layout.size`.
    Массивы :math:`x,y` развёрнуты в порядке C. Это плотность
    наблюдений при среднем :math:`x` в выбранных численных координатах.
    """
    def __init__(self, target, metric):
        if not isinstance(metric, CovarianceMetric):
            raise TypeError("Gaussian density requires a covariance metric")
        super().__init__(target, metric, normalization=1.)
        self._constant = .5*(self.domain.size*np.log(2*np.pi)+metric.logdet)

    def value(self, x):
        """Вычислить отрицательный логарифм гауссовой плотности.

        Parameters
        ----------
        x : array_like
            Конечный прогноз формы domain.shape в единицах target.

        Returns
        -------
        result : float
            Значение с нормировочной константой плотности.
        """
        return super().value(x)+self._constant


def working_quadratic(target, weights, *, layout, normalization=1.):
    """Создать квадратичную невязку с диагональными весами.

    Parameters
    ----------
    target, weights : array_like
        Целевые наблюдения и положительные конечные веса формы
        layout.shape.
    layout : ArraySpace
        Пространство наблюдений.
    normalization : float, optional
        Конечный положительный делитель функционала.

    Returns
    -------
    misfit : QuadraticMisfit
        Невязка с фиксированной DiagonalWorkingMetric.
    """
    return QuadraticMisfit(target, DiagonalWorkingMetric(
        weights, layout=layout), normalization=normalization)

def support_whitener(lift, variance):
    r"""Построить преобразование ковариации на её носителе.

    Parameters
    ----------
    lift : array_like, shape (m, n)
        Конечная вещественная матрица полного столбцового ранга.
    variance : array_like, shape (n,)
        Положительные конечные дисперсии исходных наблюдений.

    Returns
    -------
    whitener : ndarray, shape (n, m)
        Преобразование к единичной ковариации на носителе.
    support : ndarray, shape (m, n)
        Ортонормированный базис носителя ковариации.

    Notes
    -----
    Для :math:`B` — lift и :math:`v` — variance:

    .. math::

        C=B\,\operatorname{diag}(v)B^\mathsf{T},\qquad
        WCW^\mathsf{T}=I_n.

    Здесь :math:`W` — возвращаемая матрица `whitener`,
    :math:`I_n` — единичная матрица порядка :math:`n`.
    Матрица :math:`C` может быть вырожденной в полном пространстве.
    Равенство с :math:`I_n` относится к точной арифметике;
    полный столбцовый ранг проверяется по численному порогу SVD.
    """
    lift = real_array(lift, name="Lift")
    variance = real_array(variance, name="Variance")
    if lift.ndim != 2 or variance.shape != (lift.shape[1],) or np.any(variance <= 0):
        raise ValueError("Positive variance per original observation required")
    if not np.isfinite(lift).all() or not np.isfinite(variance).all():
        raise ValueError("Finite inputs required")
    U, singular, _ = svd(lift * np.sqrt(variance), full_matrices=False)
    tol = max(lift.shape) * np.finfo(float).eps * singular[0]
    if len(singular) != lift.shape[1] or np.min(singular) <= tol:
        raise ValueError("Lift must have full column rank")
    return U.T / singular[:, None], U


def on_support(values, support, tolerance=1e-10):
    r"""Проверить расстояние массива до заданного подпространства.

    Parameters
    ----------
    values : array_like, shape (m,) or (m, k)
        Конечные вещественные значения для проверки.
    support : array_like, shape (m, n)
        Ортонормированный базис подпространства.
    tolerance : float, optional
        Конечный неотрицательный относительный допуск.

    Returns
    -------
    values : ndarray
        Проверенные исходные значения той же формы, без проекции.

    Notes
    -----
    Пусть :math:`X` — `values`, :math:`U` — `support` с
    ортонормированными столбцами, :math:`\varepsilon` — `tolerance`.
    Проверяется условие

    .. math::

        \|X-UU^\mathsf{T}X\|\leq
        \varepsilon\max(1,\|X\|).

    Для вектора используется евклидова норма, для матрицы — норма Фробениуса.
    При нулевом допуске требуется точное машинное совпадение
    с вычисленной проекцией.
    """
    values = real_array(values, name="Support values")
    support = real_array(support, name="Support basis")
    tolerance = real_scalar(tolerance, name="Tolerance")
    if tolerance < 0:
        raise ValueError("Tolerance must be nonnegative")
    if tolerance == 0:
        outside = not np.array_equal(values, support @ (support.T @ values))
    else:
        scale = max(1., np.max(np.abs(values), initial=0.))
        normalized = values / scale
        defect = norm((normalized - support @ (support.T @ normalized)).ravel())
        outside = defect > tolerance * max(1. / scale, norm(normalized.ravel()))
    if outside:
        raise ValueError("Residual outside pushed covariance support")
    return values


class SupportWhitening:
    """Преобразование ковариации на её носителе.

    Parameters
    ----------
    lift : array_like, shape (m, n)
        Конечная вещественная матрица полного столбцового ранга.
    variance : array_like, shape (n,)
        Положительные конечные дисперсии исходных наблюдений.

    Notes
    -----
    Входы должны принадлежать носителю; проверка выполняется
    перед умножением на сохранённую матрицу преобразования.
    """
    def __init__(self, lift, variance):
        self._whitener, self._support = support_whitener(lift, variance)

    def whiten(self, values):
        """Преобразовать вектор на носителе ковариации.

        Parameters
        ----------
        values : array_like, shape (m,)
            Конечный вектор из носителя ковариации.

        Returns
        -------
        transformed : ndarray, shape (n,)
            Вектор в координатах с единичной ковариацией.
        """
        return self._whitener @ on_support(values, self._support)

    def whiten_matrix(self, matrix):
        """Преобразовать столбцы матрицы на носителе ковариации.

        Parameters
        ----------
        matrix : array_like, shape (m, k)
            Конечная матрица со столбцами из носителя ковариации.

        Returns
        -------
        transformed : ndarray, shape (n, k)
            Столбцы в координатах с единичной ковариацией.
        """
        return self._whitener @ on_support(matrix, self._support)

    @property
    def support(self):
        """Независимая копия ортонормированного базиса носителя ковариации."""
        return self._support.copy()
