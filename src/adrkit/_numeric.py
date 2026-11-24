"""Проверки вещественных числовых входов для численных компонентов ADRKit."""

from collections.abc import Sequence

import numpy as np


def _contains_masked_values(value):
    """Проверить маску до того, как NumPy отбросит её у массива или вложенной строки."""
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview, range)
    ):
        return any(_contains_masked_values(item) for item in value)
    return np.ma.is_masked(value)


def real_array(value, *, name, dtype=float, finite=True, copy=False):
    """Преобразовать вещественный числовой массив.

    Parameters
    ----------
    value : array_like
        Вещественные числа; пропуски под маской, bool, комплексные и нечисловые массивы
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
    if _contains_masked_values(value):
        raise ValueError(f"{name} contains masked values")
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
