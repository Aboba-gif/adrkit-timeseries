"""Дополнительный дискретный член по состоянию и его полный якобиан."""
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import sparse


def _arguments(state, points, time):
    points = np.asarray(points)
    state = np.asarray(state)
    if (points.ndim != 2 or points.shape[1] != 2 or not len(points)
            or points.dtype.kind not in "iuf" or not np.isfinite(points).all()):
        raise ValueError("StateResidual requires finite real points with shape (n,2)")
    if (state.shape != (len(points),) or state.dtype.kind not in "iuf"
            or not np.isfinite(state).all()):
        raise ValueError("StateResidual requires a finite real state with shape (n,)")
    if (isinstance(time, (bool, np.bool_))
            or not isinstance(time, (int, float, np.integer, np.floating))
            or not np.isfinite(time)):
        raise ValueError("StateResidual requires a finite real physical time")
    return np.array(state, dtype=float, copy=True), np.array(points, dtype=float, copy=True), float(time)


@dataclass(frozen=True)
class StateResidual:
    """Дискретный член состояния с полным якобианом.

    Parameters
    ----------
    value, jacobian : callable
        Принимают состояние формы (n,), точки формы (n, 2) и
        физическое время. value возвращает массив формы (n,),
        jacobian — плотную или разреженную матрицу формы (n, n).
    pure : {True}
        Неизменное поведение функций при решении и вычислении
        производных.
    snapshot_factory : callable or None, optional
        Фабрика независимого StateResidual без собственной фабрики
        снимка для изменяемых настроек.
    unit, derivative_unit : str or None, optional
        Метки единиц добавочного члена и его производной по
        концентрации.
    """

    value: Callable
    jacobian: Callable
    pure: bool
    snapshot_factory: Callable | None = None
    unit: str | None = None
    derivative_unit: str | None = None

    def __post_init__(self):
        if not callable(self.value) or not callable(self.jacobian):
            raise TypeError("StateResidual value and full Jacobian must be explicit callables")
        if self.pure is not True:
            raise ValueError("StateResidual requires the explicit pure=True lifetime contract")
        if self.snapshot_factory is not None and not callable(self.snapshot_factory):
            raise TypeError("snapshot_factory must be callable")
        for label in (self.unit, self.derivative_unit):
            if label is not None and (not isinstance(label, str) or not label.strip()):
                raise ValueError("StateResidual unit labels must be nonempty strings")

    def snapshot(self):
        """Получить снимок настроек закона.

        Returns
        -------
        law : StateResidual
            Результат `snapshot_factory`; при отсутствии фабрики
            возвращается `self` без копирования.

        Notes
        -----
        Неизменность поведения функций задаётся контрактом `pure=True`;
        значение этого флага не проверяет замыкания пользовательских функций.
        """
        if self.snapshot_factory is None:
            return self
        result = self.snapshot_factory()
        if not isinstance(result, StateResidual) or result.snapshot_factory is not None:
            raise TypeError("snapshot_factory must return a self-contained pure StateResidual")
        return result

    def evaluate(self, state, points, time, *, derivative=False):
        """Вычислить дискретный член состояния или его якобиан.

        Parameters
        ----------
        state : array_like, shape (n,)
            Конечная концентрация в узлах.
        points : array_like, shape (n, 2)
            Конечные двумерные координаты узлов.
        time : float
            Конечный физический момент.
        derivative : bool, optional
            Вернуть полный якобиан вместо значения.

        Returns
        -------
        result : ndarray or scipy.sparse.csc_matrix
            Значение формы (n,) в unit либо якобиан формы (n, n) в
            derivative_unit. Результат копируется.
        """
        state, points, time = _arguments(state, points, time)
        count = len(points)
        operation = self.jacobian if derivative else self.value
        result = operation(state, points, time)
        if not derivative:
            array = np.asarray(result)
            if (array.shape != (count,) or array.dtype.kind not in "iuf"
                    or not np.isfinite(array).all()):
                raise ValueError("StateResidual value must be finite real values with shape (n,)")
            return np.array(array, dtype=float, copy=True)
        if sparse.issparse(result):
            if result.shape != (count, count) or result.dtype.kind not in "iuf":
                raise ValueError("StateResidual Jacobian must be a finite real (n,n) matrix")
            matrix = sparse.csc_matrix(result, dtype=float, copy=True)
        else:
            array = np.asarray(result)
            if (array.shape != (count, count) or array.dtype.kind not in "iuf"
                    or not np.isfinite(array).all()):
                raise ValueError("StateResidual Jacobian must be a finite real (n,n) matrix")
            matrix = sparse.csc_matrix(np.array(array, dtype=float, copy=True))
        matrix.sum_duplicates()
        if not np.isfinite(matrix.data).all():
            raise ValueError("StateResidual Jacobian must be a finite real (n,n) matrix")
        return matrix


__all__ = ["StateResidual"]
