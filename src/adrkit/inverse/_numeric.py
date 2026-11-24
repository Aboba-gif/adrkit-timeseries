"""Проверки вещественных числовых входов для операций обратной задачи."""

import numpy as np

from .._numeric import real_array, real_scalar


def symmetric_psd_matrix(value, size, *, name, copy=False):
    """Проверить конечную симметричную положительно полуопределённую матрицу.

    Parameters
    ----------
    value : array_like, shape (size, size)
        Вещественная матрица; размер size должен быть положительным.
    size : int
        Число строк и столбцов.
    name : str
        Имя матрицы в сообщении об ошибке.
    copy : bool, optional
        Создать независимую копию.

    Returns
    -------
    matrix : ndarray
        Матрица, прошедшая проверку с численными допусками.
        Исходная матрица не заменяется её симметричной частью;
        проекция собственных значений не выполняется.
    tolerance : float
        Абсолютный допуск отрицательной части собственных значений.

    Notes
    -----
    Собственные значения проверяются у симметричной части матрицы,
    поскольку она определяет квадратичную форму. Допуск симметрии
    допускает малое различие между верхним и нижним треугольниками.
    """
    matrix = real_array(value, name=name, copy=copy)
    if matrix.shape != (size, size) or size == 0:
        raise ValueError(f"{name} must have nonempty shape ({size}, {size})")
    if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-14):
        raise ValueError(f"{name} must be symmetric")
    # Допуск учитывает округление собственных значений; исходная матрица сохраняется.
    tolerance = 100 * np.finfo(float).eps * max(1.0, np.linalg.norm(matrix, 2))
    symmetric_part = 0.5 * matrix + 0.5 * matrix.T
    if np.linalg.eigvalsh(symmetric_part).min() < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    return matrix, tolerance
