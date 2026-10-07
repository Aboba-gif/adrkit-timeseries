"""Штрафы коэффициентов и явно заданные метрики Рисса."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from adrkit.contracts import ScalarCapabilities
from adrkit.sources import P1Basis
from . import linalg
from ._numeric import real_array, real_scalar, symmetric_psd_matrix


def p1_matrices(basis):
    """Собрать матрицы массы и жёсткости базиса P1.

    Parameters
    ----------
    basis : P1Basis
        Временной базис со свободными крайними коэффициентами.

    Returns
    -------
    mass : ndarray, shape (basis.size, basis.size)
        Интегралы произведений функций базиса в единицах времени.
    stiffness : ndarray, shape (basis.size, basis.size)
        Интегралы произведений производных в обратных единицах
        времени.
    """
    if not isinstance(basis, P1Basis):
        raise TypeError("p1_matrices requires P1Basis")
    mass = np.zeros((basis.size, basis.size))
    stiffness = np.zeros_like(mass)
    for j, h in enumerate(np.diff(basis.knots)):
        mass[j:j+2, j:j+2] += h/6*np.array([[2., 1.], [1., 2.]])
        stiffness[j:j+2, j:j+2] += np.array([[1., -1.], [-1., 1.]])/h
    return mass, stiffness


class QuadraticPenalty:
    r"""Квадратичный штраф коэффициентов с заданным центром.

    Parameters
    ----------
    center : array_like, shape (n,)
        Конечный центр штрафа в единицах коэффициентов.
    gram : array_like, shape (n, n)
        Конечная симметричная положительно полуопределённая матрица
        штрафа.
    alpha : float
        Конечный неотрицательный вес штрафа.

    Notes
    -----
    Для центра :math:`a_0`, матрицы :math:`G` и веса :math:`\alpha`:

    .. math::

        P(a)=\frac{\alpha}{2}(a-a_0)^\mathsf{T}G(a-a_0).

    Здесь :math:`a` — коэффициенты, :math:`a_0` — `center`,
    :math:`G` — `gram`, :math:`\alpha` — `alpha`.
    Координаты центра и матрица сохраняются в собственных копиях.
    Симметрия и положительная полуопределённость проверяются с численными
    допусками; исходная матрица не заменяется её симметризацией или проекцией.
    Собственные значения проверяются у симметричной части матрицы,
    определяющей квадратичную форму.
    """
    def __init__(self, center, gram, alpha):
        if np.iscomplexobj(center) or np.iscomplexobj(gram):
            raise ValueError("Penalty coordinates must be real")
        self._center = real_array(center, name="Penalty center", copy=True)
        self._alpha = real_scalar(alpha, name="Penalty alpha")
        n = self._center.size
        if self._center.shape != (n,) or n == 0 or self._alpha < 0:
            raise ValueError("Finite nonempty center and nonnegative alpha required")
        self._gram, _ = symmetric_psd_matrix(gram, n, name="Penalty Gram", copy=True)

    @property
    def center(self):
        """Независимая копия центра штрафа."""
        return self._center.copy()

    @property
    def gram(self):
        """Независимая копия матрицы штрафа."""
        return self._gram.copy()

    @property
    def alpha(self):
        """Неотрицательный вес квадратичного штрафа."""
        return self._alpha

    def _difference(self, x):
        if np.iscomplexobj(x):
            raise ValueError("Penalty input must be real")
        x = real_array(x, name="Penalty input")
        if x.shape != self._center.shape or not np.isfinite(x).all():
            raise ValueError("Penalty input must match center")
        return x-self._center

    def value(self, x):
        """Вычислить квадратичный штраф коэффициентов.

        Parameters
        ----------
        x : array_like
            Конечная точка формы center.shape в единицах центра.

        Returns
        -------
        result : float
            Значение с включённым весом alpha.
        """
        d = self._difference(x)
        return .5*self.alpha*float(d @ self._gram @ d)

    def cotangent(self, x):
        """Вычислить координатный ковектор штрафа.

        Parameters
        ----------
        x : array_like
            Конечная точка формы center.shape в единицах центра.

        Returns
        -------
        result : ndarray
            Ковектор той же формы, что center.
        """
        d = self._difference(x)
        return self.alpha*(self._gram @ d)

    def capabilities(self, domain):
        """Объявить штраф на пространстве коэффициентов.

        Parameters
        ----------
        domain : ArraySpace
            Пространство, совместимое с формой центра штрафа.

        Returns
        -------
        functional : ScalarCapabilities
            Значение и координатный ковектор штрафа.
        """
        domain.array(self._center, name="penalty center")
        return ScalarCapabilities(self, domain, self)


class CoefficientMetric:
    """Положительно определённая метрика коэффициентов.

    Parameters
    ----------
    gram : array_like, shape (domain.size, domain.size) or (domain.size,)
        Симметричная положительно определённая матрица либо
        положительная диагональ.
    domain : ArraySpace
        Пространство коэффициентов; матрица задаётся в порядке C.

    Notes
    -----
    apply применяет матрицу метрики; solve преобразует
    координатный ковектор в градиент Рисса.
    """
    def __init__(self, gram, *, domain):
        if np.iscomplexobj(gram):
            raise ValueError("Coefficient metric must be real")
        self._domain = domain
        root = linalg.gram_root(gram, domain.size)
        self._gram = root.T @ root
        self._root = root
        self._factor = cho_factor(self._gram, lower=True)

    @property
    def domain(self):
        """Пространство коэффициентов метрики."""
        return self._domain

    @property
    def gram(self):
        """Независимая копия положительно определённой матрицы метрики."""
        return self._gram.copy()

    @property
    def root(self):
        r"""Получить верхнетреугольный корень метрики.

        Returns
        -------
        root : ndarray, shape (domain.size, domain.size)
            Независимая копия корня с положительной диагональю.

        Notes
        -----
        Для матрицы метрики :math:`G` и возвращаемого корня :math:`R`
        в точной арифметике:

        .. math::

            G=R^\mathsf{T}R.
        """
        return self._root.copy()

    def apply(self, x):
        """Применить матрицу метрики к коэффициентам.

        Parameters
        ----------
        x : array_like
            Конечный массив формы domain.shape.

        Returns
        -------
        result : ndarray
            Действие метрики той же формы.
        """
        x = self.domain.array(x)
        return (self._gram @ x.ravel()).reshape(self.domain.shape)

    def solve(self, covector):
        """Преобразовать ковектор в градиент Рисса.

        Parameters
        ----------
        covector : array_like
            Конечный координатный ковектор формы domain.shape.

        Returns
        -------
        gradient : ndarray
            Градиент формы domain.shape в заданной метрике.
        """
        g = self.domain.array(covector)
        return cho_solve(self._factor, g.ravel()).reshape(self.domain.shape)


def build_penalty(kind, basis, center, alpha, *, tau=0., domain):
    r"""Построить временной штраф L² или H¹ для P1.

    Parameters
    ----------
    kind : {'l2', 'h1'}
        Вид квадратичного штрафа.
    basis : P1Basis
        Временной базис.
    center : array_like, shape (basis.size,)
        Конечный центр в единицах коэффициентов.
    alpha : float
        Конечный неотрицательный вес.
    tau : float, optional
        Конечный неотрицательный масштаб производной в единицах
        времени; для L² равен нулю.
    domain : ArraySpace
        Пространство коэффициентов, совместимое с центром.

    Returns
    -------
    penalty : ScalarCapabilities
        Значение и ковектор штрафа с включённым весом alpha.

    Notes
    -----
    Для матриц массы :math:`M` и жёсткости :math:`K`:

    .. math::

        G_{L^2}=M,\qquad G_{H^1}=M+\tau^2K.

    Здесь :math:`\tau` — `tau`, матрицы :math:`M,K` получены
    из `p1_matrices(basis)`. При `kind="h1"` и положительном `tau`
    учитываются значение профиля и его временная производная.
    Допустимый в этой функции случай `tau=0` совпадает с L²-штрафом.
    """
    tau = real_scalar(tau, name="Temporal penalty scale")
    if kind not in ("l2", "h1") or not np.isfinite(tau) or tau < 0:
        raise ValueError("Expected l2/h1 and finite nonnegative tau")
    if kind == "l2" and tau != 0:
        raise ValueError("L2 does not use a temporal derivative scale")
    mass, stiffness = p1_matrices(basis)
    gram = mass if kind == "l2" else mass+tau**2*stiffness
    return QuadraticPenalty(center, gram, alpha).capabilities(domain)
