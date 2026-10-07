"""Разложения ковариаций и матриц Грама."""
import numpy as np
from scipy.linalg import cholesky, solve_triangular

from ._numeric import real_array


def _spd_factor(matrix, size, name):
    """Разложить положительно определённую матрицу или её диагональ."""
    matrix = real_array(matrix, name=name)
    if matrix.ndim == 1:
        if matrix.shape != (size,) or not np.isfinite(matrix).all() or np.any(matrix <= 0):
            raise ValueError(f"{name} diagonal must be finite and strictly positive")
        return np.sqrt(matrix)
    if matrix.shape != (size, size) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must have the required finite square shape")
    if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-14):
        raise ValueError(f"{name} must be symmetric")
    try:
        return cholesky(matrix, lower=True)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite") from exc


def whiten(A, y, Sigma):
    r"""Привести ковариацию наблюдений к единичной матрице.

    Parameters
    ----------
    A : ndarray, shape (m, n)
        Матрица прогноза в координатах наблюдений.
    y : ndarray, shape (m,)
        Вектор наблюдений в тех же единицах.
    Sigma : array_like, shape (m, m) or (m,)
        Симметричная положительно определённая ковариация или её
        положительная диагональ.

    Returns
    -------
    matrix : ndarray, shape (m, n)
        Преобразованная матрица прогноза.
    observations : ndarray, shape (m,)
        Преобразованные наблюдения.

    Notes
    -----
    Пусть :math:`L` — нижний фактор Холецкого ковариации
    :math:`\Sigma`. Преобразование определяется равенствами

    .. math::

        \Sigma=LL^\mathsf{T},\qquad
        \widetilde A=L^{-1}A,\qquad \widetilde y=L^{-1}y.

    Здесь :math:`A,y` — входные матрица прогноза и наблюдения.
    Для диагональной ковариации применяется покомпонентное деление
    на стандартные отклонения.
    """
    factor = _spd_factor(Sigma, A.shape[0], "Sigma")
    if factor.ndim == 1:
        return A/factor[:, None], y/factor
    return (solve_triangular(factor, A, lower=True),
            solve_triangular(factor, y, lower=True))


def gram_root(M, size):
    r"""Вычислить верхнетреугольный корень положительной метрики.

    Parameters
    ----------
    M : array_like, shape (size, size) or (size,)
        Симметричная положительно определённая матрица либо
        положительная диагональ.
    size : int
        Размер пространства коэффициентов.

    Returns
    -------
    root : ndarray, shape (size, size)
        Верхнетреугольная матрица с положительной диагональю.

    Notes
    -----
    Для возвращаемой матрицы :math:`R` и входной метрики :math:`M`
    в точной арифметике выполняется

    .. math::

        M=R^\mathsf{T}R.
    """
    factor = _spd_factor(M, size, "M")
    return np.diag(factor) if factor.ndim == 1 else factor.T
