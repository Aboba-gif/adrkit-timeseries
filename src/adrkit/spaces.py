"""Пространства массивов и линейные отображения в евклидовых координатах."""
from dataclasses import dataclass
from math import prod

import numpy as np
from scipy.sparse.linalg import LinearOperator

from adrkit.contracts import MapCapabilities


def _coordinate(value):
    """Привести координаты к строкам, конечным числам и кортежам."""
    if isinstance(value, (tuple, list, np.ndarray)):
        return tuple(_coordinate(item) for item in value)
    if isinstance(value, np.generic):
        value = value.item()
    if type(value) not in (str, int, float):
        raise ValueError("Coordinates must be strings, numbers or nested tuples")
    if isinstance(value, (int, float)) and not np.isfinite(value):
        raise ValueError("Coordinates must be finite")
    return value


@dataclass(frozen=True, init=False)
class ArraySpace:
    """Форма, координаты и единицы вещественного массива.

    Parameters
    ----------
    shape : tuple of int
        Размеры осей; пустой кортеж задаёт скаляр.
    axes : sequence of str
        Непустые уникальные имена осей в порядке shape.
    coordinates : sequence of sequences
        Уникальные координаты каждой оси; длины совпадают с shape.
        Допустимы строки, конечные числа и вложенные кортежи.
    units : str
        Непустое обозначение единиц массива.
    scale : float, optional
        Положительный масштаб координат как метаданные.
    pairing : {'euclidean'}, optional
        Евклидово скалярное произведение координат.
    dtype : numpy.dtype or data-type, optional
        Вещественный тип с плавающей точкой.
    binding : str or None, optional
        Дополнительное обозначение привязки пространства.

    Notes
    -----
    Единицы и масштаб не преобразуют численные значения.
    Совместимость требует совпадения всех полей пространства.
    """

    shape: tuple
    axes: tuple
    coordinates: tuple
    units: str
    scale: float
    pairing: str
    dtype: np.dtype
    binding: str | None

    def __init__(self, shape, *, axes, coordinates, units, scale=1.,
                 pairing="euclidean", dtype=np.float64, binding=None):
        shape = tuple(shape)
        if any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer))
               or n < 1 for n in shape):
            raise ValueError("Shape entries must be positive integers")
        shape = tuple(int(n) for n in shape)
        axes = tuple(axes)
        coordinates = tuple(_coordinate(axis) for axis in coordinates)
        if (len(axes) != len(shape) or len(set(axes)) != len(axes)
                or any(not isinstance(a, str) or not a.strip() for a in axes)):
            raise ValueError("Each axis needs a distinct nonempty name")
        if (len(coordinates) != len(shape)
                or any(not isinstance(c, tuple) or len(c) != n
                       for c, n in zip(coordinates, shape))):
            raise ValueError("Coordinates must match every axis length")
        if any(len(set(c)) != len(c) for c in coordinates):
            raise ValueError("Axis coordinate IDs must be unique")
        if not isinstance(units, str) or not units.strip():
            raise ValueError("An explicit unit label (or 'unknown') is required")
        if isinstance(scale, (bool, np.bool_)) or not np.isfinite(scale) or scale <= 0:
            raise ValueError("Coordinate scale must be finite and positive")
        if pairing != "euclidean":
            raise ValueError("Only Euclidean coordinate pairing is implemented")
        dtype = np.dtype(dtype)
        if dtype.kind != "f":
            raise ValueError("Only real floating point array spaces are supported")
        if binding is not None and (not isinstance(binding, str) or not binding):
            raise ValueError("binding must be a nonempty string or None")
        for key, value in dict(shape=shape, axes=axes, coordinates=coordinates,
                               units=units, scale=float(scale), pairing=pairing,
                               dtype=dtype, binding=binding).items():
            object.__setattr__(self, key, value)

    @property
    def size(self):
        """Число координат массива; для скаляра равно единице."""
        return prod(self.shape)

    def array(self, value, *, name="array", copy=False, finite=True):
        """Привести массив к типу и форме пространства.

        Parameters
        ----------
        value : array_like
            Вещественные значения точной формы shape.
        name : str, optional
            Имя входа в сообщении об ошибке.
        copy : bool, optional
            Создать независимую копию результата.
        finite : bool, optional
            Требовать конечные значения после преобразования.

        Returns
        -------
        array : ndarray
            Массив формы shape и типа dtype; единицы задаёт пространство.
        """
        if np.iscomplexobj(value):
            raise ValueError(f"{name} must be real")
        result = np.asarray(value, dtype=self.dtype)
        if result.shape != self.shape or (finite and not np.isfinite(result).all()):
            raise ValueError(f"{name} must have shape {self.shape} and valid values")
        return result.copy() if copy else result

    def require_compatible(self, other):
        """Проверить совместимость двух пространств.

        Parameters
        ----------
        other : ArraySpace
            Пространство с теми же формой, координатами, единицами и
            остальными полями.

        Notes
        -----
        При несовпадении полей вызывается ValueError.
        """
        if not isinstance(other, ArraySpace) or self != other:
            raise ValueError("Incompatible array spaces (coordinates/units/binding)")


class _LinearMap:
    """Линейные действия между пространствами в порядке координат C."""

    def __init__(self, domain, codomain, matvec, rmatvec):
        if not isinstance(domain, ArraySpace) or not isinstance(codomain, ArraySpace):
            raise TypeError("domain and codomain must be ArraySpace records")
        if domain.dtype != codomain.dtype:
            raise ValueError("A numerical map requires one explicit common dtype")
        self._domain, self._codomain = domain, codomain
        self._operator = LinearOperator((codomain.size, domain.size),
                                        matvec=matvec, rmatvec=rmatvec,
                                        dtype=domain.dtype)

    @property
    def domain(self):
        """Пространство аргументов линейного отображения."""
        return self._domain

    @property
    def codomain(self):
        """Пространство результатов линейного отображения."""
        return self._codomain

    def capabilities(self):
        """Предоставить прогноз и производные как MapCapabilities."""
        return MapCapabilities(self, self.domain, self.codomain, self, self)

    def predict(self, x):
        """Применить линейный оператор к массиву.

        Parameters
        ----------
        x : array_like
            Аргумент формы domain.shape в единицах domain.

        Returns
        -------
        prediction : ndarray
            Результат формы codomain.shape в единицах codomain.
        """
        x = self.domain.array(x, name="input")
        result = np.asarray(self._operator.matvec(x.ravel(order="C")),
                            dtype=self.codomain.dtype).reshape(self.codomain.shape)
        return self.codomain.array(result, name="prediction", copy=True)

    def jvp(self, at, direction):
        """Применить линейный оператор к направлению.

        Parameters
        ----------
        at : array_like
            Точка формы domain.shape; используется для проверки входа.
        direction : array_like
            Направление формы domain.shape.

        Returns
        -------
        tangent : ndarray
            Действие производной формы codomain.shape.
        """
        self.domain.array(at, name="at")
        return self.predict(direction)

    def vjp(self, at, cotangent):
        """Применить транспонированный оператор к ковектору.

        Parameters
        ----------
        at : array_like
            Точка формы domain.shape.
        cotangent : array_like
            Ковектор результата формы codomain.shape.

        Returns
        -------
        covector : ndarray
            Ковектор аргумента формы domain.shape в евклидовых
            координатах.
        """
        self.domain.array(at, name="at")
        z = self.codomain.array(cotangent, name="cotangent")
        result = np.asarray(self._operator.rmatvec(z.ravel(order="C")),
                            dtype=self.domain.dtype).reshape(self.domain.shape)
        return self.domain.array(result, name="VJP output", copy=True)

    def linear_operator(self):
        """Вернуть прямое и транспонированное действия через SciPy.

        Returns
        -------
        operator : scipy.sparse.linalg.LinearOperator
            Оператор формы (codomain.size, domain.size); массивы
            разворачиваются в порядке C.
        """
        return LinearOperator(self._operator.shape, matvec=self._operator.matvec,
                              rmatvec=self._operator.rmatvec, dtype=self.domain.dtype)


class MatrixMap(_LinearMap):
    """Линейное отображение по собственной копии матрицы.

    Parameters
    ----------
    matrix : array_like, shape (codomain.size, domain.size)
        Конечная вещественная матрица координатного действия.
    domain, codomain : ArraySpace
        Пространства входа и выхода с одним вещественным dtype.

    Notes
    -----
    Координаты массивов разворачиваются в порядке C.
    Производная применяет матрицу, перенос ковектора — её транспонирование.
    """

    def __init__(self, matrix, domain, codomain):
        if np.iscomplexobj(matrix):
            raise ValueError("Matrix must be real")
        self._matrix = np.array(matrix, dtype=domain.dtype, copy=True)
        if (self._matrix.shape != (codomain.size, domain.size)
                or not np.isfinite(self._matrix).all()):
            raise ValueError("Finite matrix must match domain/codomain")
        super().__init__(domain, codomain, self._matvec, self._rmatvec)

    def _matvec(self, x):
        return self._matrix @ x

    def _rmatvec(self, z):
        return self._matrix.T @ z

    @property
    def matrix(self):
        """Независимая копия матрицы формы (codomain.size, domain.size)."""
        return self._matrix.copy()


def matrix_map(matrix, *, domain, codomain):
    """Создать линейное отображение из матрицы.

    Parameters
    ----------
    matrix : array_like, shape (codomain.size, domain.size)
        Конечная вещественная матрица; сохраняется собственная копия.
    domain, codomain : ArraySpace
        Пространства массивов с одинаковым dtype.

    Returns
    -------
    mapping : MapCapabilities
        Прогноз и JVP применяют матрицу; VJP применяет её
        транспонирование.
    """
    if not isinstance(domain, ArraySpace) or not isinstance(codomain, ArraySpace):
        raise TypeError("domain and codomain must be ArraySpace records")
    return MatrixMap(matrix, domain, codomain).capabilities()
