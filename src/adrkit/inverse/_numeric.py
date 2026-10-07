"""Проверки вещественных числовых входов для операций обратной задачи."""
import numpy as np


def real_array(value, *, name, dtype=float, finite=True, copy=False):
    """Преобразовать вещественный числовой массив.

    Parameters
    ----------
    value : array_like
        Вещественные числа; bool, комплексные и нечисловые массивы
        отклоняются.
    name : str
        Имя входа в сообщении об ошибке.
    dtype : data-type, optional
        Тип результата.
    finite : bool, optional
        Требовать конечность до и после преобразования.
    copy : bool, optional
        Создать независимую копию.

    Returns
    -------
    array : ndarray
        Массив исходной формы в заданном dtype.
    """
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or (finite and not np.isfinite(raw).all()):
        qualifier = "finite real" if finite else "real"
        raise ValueError(f"{name} must contain {qualifier} numeric values")
    with np.errstate(over="ignore", invalid="ignore"):
        converted = np.array(raw, dtype=dtype, copy=True) if copy else np.asarray(raw, dtype=dtype)
    if finite and not np.isfinite(converted).all():
        raise ValueError(f"{name} must remain finite in the requested dtype")
    return converted


def real_scalar(value, *, name, finite=True):
    """Преобразовать вещественное скалярное число.

    Parameters
    ----------
    value : scalar
        Вещественное число; bool, комплексные значения и строки
        отклоняются.
    name : str
        Имя входа в сообщении об ошибке.
    finite : bool, optional
        Требовать конечное значение; при False допустимы NaN и
        бесконечности.

    Returns
    -------
    scalar : float
        Значение после преобразования.
    """
    qualifier = "finite real scalar" if finite else "real scalar"
    try:
        scalar = real_array(value, name=name, finite=finite)
    except ValueError as error:
        raise ValueError(f"{name} must be a {qualifier}") from error
    if scalar.shape != ():
        raise ValueError(f"{name} must be a {qualifier}")
    return float(scalar)


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
    tolerance = 100*np.finfo(float).eps*max(1., np.linalg.norm(matrix, 2))
    symmetric_part = .5*matrix+.5*matrix.T
    if np.linalg.eigvalsh(symmetric_part).min() < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    return matrix, tolerance
