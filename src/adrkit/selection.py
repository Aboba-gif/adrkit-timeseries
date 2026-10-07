"""Упорядоченный выбор координат и дополнение ковектора нулями."""
import numpy as np

from adrkit.spaces import ArraySpace


class Selection:
    """Упорядоченный выбор координат векторного пространства.

    Parameters
    ----------
    indices : array_like of int, shape (k,)
        Непустые уникальные индексы в диапазоне полного пространства.
    full_layout : ArraySpace
        Одномерное пространство исходных наблюдений.
    """

    def __init__(self, indices, full_layout):
        if not isinstance(full_layout, ArraySpace) or len(full_layout.shape) != 1:
            raise ValueError("Selection requires a one-dimensional ArraySpace layout")
        indices = np.asarray(indices)
        if (indices.ndim != 1 or indices.dtype.kind not in "iu" or not len(indices)
                or len(np.unique(indices)) != len(indices)
                or np.any(indices < 0) or np.any(indices >= full_layout.size)):
            raise ValueError("Selection indices must be nonempty, unique, integer and in range")
        self._indices = indices.astype(np.intp, copy=True)
        self._full_layout = full_layout
        self._selected_layout = ArraySpace(
            (len(indices),), axes=full_layout.axes,
            coordinates=(tuple(full_layout.coordinates[0][i] for i in indices),),
            units=full_layout.units, scale=full_layout.scale, pairing=full_layout.pairing,
            dtype=full_layout.dtype, binding=full_layout.binding)

    @property
    def indices(self):
        """Независимая копия индексов в порядке выбора."""
        return self._indices.copy()

    @property
    def full_layout(self):
        """Полное пространство исходных наблюдений."""
        return self._full_layout

    @property
    def selected_layout(self):
        """Пространство выбранных наблюдений с их порядком координат."""
        return self._selected_layout

    def take(self, x):
        # Скрытые ответы могут быть NaN; строки выбираются до проверки конечности и вычисления
        # невязки.
        """Выбрать значения в установленном порядке.

        Parameters
        ----------
        x : array_like
            Массив формы full_layout.shape; NaN и бесконечности
            допустимы.

        Returns
        -------
        selected : ndarray
            Независимая копия формы selected_layout.shape с теми же
            единицами.
        """
        return self.full_layout.array(x, finite=False)[self._indices].copy()

    def scatter(self, z):
        """Дополнить ковектор нулями в полном пространстве.

        Parameters
        ----------
        z : array_like
            Конечный ковектор формы selected_layout.shape.

        Returns
        -------
        full : ndarray
            Ковектор формы full_layout.shape; невыбранные координаты
            нулевые.
        """
        z = self.selected_layout.array(z, name="selected cotangent")
        result = np.zeros(self.full_layout.shape, dtype=self.full_layout.dtype)
        result[self._indices] = z
        return result

    def take_covariance(self, covariance):
        """Выбрать главную подматрицу ковариации.

        Parameters
        ----------
        covariance : array_like, shape (full_layout.size, full_layout.size)
            Вещественная матрица ковариации полного вектора.

        Returns
        -------
        selected : ndarray, shape (k, k)
            Независимая копия в порядке indices, где k — число выбранных
            координат.
        """
        if np.iscomplexobj(covariance):
            raise ValueError("Covariance must be real")
        covariance = np.asarray(covariance, dtype=self.full_layout.dtype)
        if covariance.shape != (self.full_layout.size, self.full_layout.size):
            raise ValueError("Covariance must match full observation layout")
        return covariance[np.ix_(self._indices, self._indices)].copy()
