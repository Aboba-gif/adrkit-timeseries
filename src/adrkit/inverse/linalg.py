"""Разложения ковариаций и матриц Грама."""

import numpy as np
from scipy.linalg import cholesky, solve_triangular

from ._numeric import real_array


def _symmetric_matrix(matrix, size, name):
    """Проверить относительную асимметрию и построить одну симметричную матрицу."""
    matrix = real_array(matrix, name=name)
    if matrix.shape != (size, size):
        raise ValueError(f"{name} must have the required finite square shape")
    if np.array_equal(matrix, matrix.T):
        return matrix
    scale = float(np.max(np.abs(matrix)))
    scaled = matrix / scale
    if np.max(np.abs(scaled - scaled.T)) > 1e-12:
        raise ValueError(f"{name} must be symmetric relative to its scale")
    symmetric = matrix.copy()
    for row in range(size):
        low = np.minimum(matrix[row, :row], matrix[:row, row])
        high = np.maximum(matrix[row, :row], matrix[:row, row])
        mean = low + 0.5 * (high - low)
        symmetric[row, :row] = mean
        symmetric[:row, row] = mean
    return symmetric


def _spd_factor(matrix, size, name):
    """Разложить положительно определённую симметричную часть или диагональ.

    Для плотной матрицы M с масштабом s=max(abs(M))>0 требуется
    max(abs(M/s - M.T/s)) <= 1e-12. Допустимая асимметрия заменяется
    симметричной частью перед проверкой положительной определённости.
    Точно симметричные элементы сохраняются без усреднения; добавки
    к диагонали и исправления собственных значений не применяются.
    """
    matrix = real_array(matrix, name=name)
    if matrix.ndim == 1:
        if matrix.shape != (size,) or not np.isfinite(matrix).all() or np.any(matrix <= 0):
            raise ValueError(f"{name} diagonal must be finite and strictly positive")
        return np.sqrt(matrix)
    matrix = _symmetric_matrix(matrix, size, name)
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
    Пусть :math:`L` — нижний фактор Холецкого проверенной симметричной
    ковариации :math:`\Sigma`. Преобразование определяется равенствами

    .. math::

        \Sigma=LL^\mathsf{T},\qquad
        \widetilde A=L^{-1}A,\qquad \widetilde y=L^{-1}y.

    Здесь :math:`A,y` — входные матрица прогноза и наблюдения.
    Для диагональной ковариации применяется покомпонентное деление
    на стандартные отклонения. Для плотной матрицы допускается асимметрия
    не более 1e-12 от её наибольшего по модулю элемента; используется
    симметричная часть. Форма y должна быть именно (m,), без расширения
    размерностей при делении.
    """
    A = real_array(A, name="A")
    if A.ndim != 2:
        raise ValueError("A must be a two-dimensional matrix")
    y = real_array(y, name="y")
    if y.shape != (A.shape[0],):
        raise ValueError("y must be a vector matching the rows of A")
    factor = _spd_factor(Sigma, A.shape[0], "Sigma")
    if factor.ndim == 1:
        return A / factor[:, None], y / factor
    return (solve_triangular(factor, A, lower=True), solve_triangular(factor, y, lower=True))


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

    Для плотного входа допускается относительная асимметрия не более
    1e-12, как в whiten. В этом случае M в равенстве обозначает
    симметричную часть входной матрицы.
    """
    factor = _spd_factor(M, size, "M")
    return np.diag(factor) if factor.ndim == 1 else factor.T
