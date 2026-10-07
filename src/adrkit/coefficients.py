"""Поля коэффициентов ADR, локальная реакция и числовые слои."""
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class ADRUnits:
    """Обозначения единиц длины, времени и концентрации.

    Parameters
    ----------
    length, time, concentration : str
        Непустые обозначения базовых единиц; численные значения не
        преобразуются.
    """
    length: str
    time: str
    concentration: str

    def __post_init__(self):
        for value in (self.length, self.time, self.concentration):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Nonempty length, time and concentration units required")

    @property
    def diffusion(self):
        """Обозначение единицы диффузии: квадрат длины за время."""
        return f"{self.length}^2/{self.time}"

    @property
    def velocity(self):
        """Обозначение единицы скорости: длина за время."""
        return f"{self.length}/{self.time}"

    @property
    def loss(self):
        """Обозначение единицы коэффициента потерь: обратное время."""
        return f"1/{self.time}"

    @property
    def reaction(self):
        """Обозначение единицы реакции: концентрация за время."""
        return f"{self.concentration}/{self.time}"

    @property
    def source(self):
        """Обозначение единицы точечного источника: концентрация·площадь за время."""
        return f"{self.concentration}*{self.length}^2/{self.time}"


def _evaluator(value):
    if callable(value):
        return value
    operation = getattr(value, "evaluate", None)
    if callable(operation):
        return operation
    raise TypeError("A field requires a callable or an object with evaluate(points, time)")


@dataclass(frozen=True)
class ScalarField:
    """Скалярное поле координат и физического времени.

    Parameters
    ----------
    evaluator : callable or object
        Функция или объект с evaluate(points, time). Принимает точки
        формы (n, 2) и скаляр времени; возвращает число или массив
        формы (n,).
    unit : str or None, optional
        Обозначение единицы значений поля.
    time_dependent : bool, optional
        Вычислять поле в каждый момент. При False используется один
        временной слой.
    """
    evaluator: object
    unit: str | None = None
    time_dependent: bool = True

    def __post_init__(self):
        _evaluator(self.evaluator)
        if self.unit is not None and (not isinstance(self.unit, str) or not self.unit.strip()):
            raise ValueError("A field unit must be a nonempty string")
        if type(self.time_dependent) is not bool:
            raise TypeError("time_dependent must be an explicit bool")


@dataclass(frozen=True)
class VectorField(ScalarField):
    """Двумерное векторное поле координат и времени.

    Parameters
    ----------
    evaluator : callable or object
        Функция или evaluate(points, time); результат формы (2,) или
        (n, 2).
    unit : str or None, optional
        Обозначение единицы компонентов поля.
    time_dependent : bool, optional
        Вычислять поле в каждый момент; при False сохраняется один
        слой.
    """


def is_scalar_constant(value):
    """Определить, является ли значение вещественной числовой константой.

    Parameters
    ----------
    value : object
        Проверяемое значение.

    Returns
    -------
    is_constant : bool
        `True` для вещественного числа или скалярного числового ndarray;
        логические значения отклоняются.

    Notes
    -----
    Конечность не проверяется. Для произвольного объекта преобразование
    в массив не вызывается.
    """
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return True
    # Проверка скалярных массивов не вызывает __array__ произвольного объекта функции.
    return isinstance(value, np.ndarray) and value.shape == () and value.dtype.kind in 'iuf'


def is_vector_constant(value):
    """Определить, является ли значение постоянным вектором длины два.

    Parameters
    ----------
    value : object
        Проверяемое значение.

    Returns
    -------
    is_constant : bool
        `True` для значения, приводимого к вещественному числовому
        массиву формы (2,). Поля, функции и объекты с `evaluate`
        отклоняются.

    Notes
    -----
    Конечность компонентов не проверяется.
    """
    if isinstance(value, (ScalarField, VectorField)) or callable(value) or hasattr(value, "evaluate"):
        return False
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    return array.shape == (2,) and array.dtype.kind in "iuf" and array.dtype.kind != "b"


def _owned_values(value, count, components, name):
    original = np.asarray(value)
    if original.dtype.kind not in "iuf" or np.iscomplexobj(original):
        raise ValueError(f"{name} must contain real numeric values, not bool/complex/object")
    array = np.array(value, dtype=float, copy=True)
    shape = (count,) if components == 1 else (count, components)
    constant_shape = () if components == 1 else (components,)
    if array.shape == constant_shape:
        array = np.broadcast_to(array, shape).copy()
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must return finite values with shape {shape} or {constant_shape}")
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class SampledField:
    """Сохранённые числовые слои поля.

    Parameters
    ----------
    values : tuple of ndarray
        Слои поля в порядке моментов вычисления.

    Notes
    -----
    При прямом создании массивы не копируются.
    Функция sample_field создаёт независимые массивы с запрещённой записью.
    """
    values: tuple

    def at(self, step):
        """Получить сохранённый слой поля.

        Parameters
        ----------
        step : int
            Индекс слоя. При единственном слое не используется.

        Returns
        -------
        values : ndarray
            Сохранённый массив без копирования, в единицах исходного
            поля.
        """
        return self.values[0] if len(self.values) == 1 else self.values[step]


def sample_field(value, points, times, *, name, components=1,
                 positive=False, nonnegative=False, expected_unit=None):
    """Вычислить поле в точках и физических моментах.

    Parameters
    ----------
    value : float, array_like, ScalarField, VectorField or callable
        Постоянное значение либо функция поля. Постоянный вектор
        имеет форму (2,); функция принимает points и скаляр времени.
    points : array_like, shape (n, 2)
        Конечные двумерные координаты в согласованных единицах длины,
        выбранных вызывающим кодом.
    times : array_like, shape (m,)
        Непустой вектор конечных физических моментов.
    name : str
        Имя коэффициента в сообщениях об ошибке.
    components : {1, 2}, optional
        Одно скалярное значение либо две векторные компоненты.
    positive, nonnegative : bool, optional
        Требовать соответственно строгую положительность или
        неотрицательность в точках вычисления.
    expected_unit : str or None, optional
        Ожидаемая метка единиц; сравнивается с явно указанной
        единицей поля.

    Returns
    -------
    sampled : SampledField
        Независимые слои формы (n,) или (n, 2) с запрещённой записью.

    Notes
    -----
    Для константы или поля с time_dependent=False сохраняется один слой.
    Численные единицы не преобразуются; условия знака проверяются в выборке.
    """
    points = np.array(points, dtype=float, copy=True)
    times = np.asarray(times, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all()
            or times.ndim != 1 or not len(times) or not np.isfinite(times).all()):
        raise ValueError("Finite two-dimensional points and times required")
    constant = is_scalar_constant(value) if components == 1 else is_vector_constant(value)
    if constant:
        operation, selected_times = None, times[:1]
    else:
        if isinstance(value, ScalarField):
            if (components == 1 and isinstance(value, VectorField)) or (
                    components != 1 and not isinstance(value, VectorField)):
                raise TypeError(f"{name} requires a {'scalar' if components == 1 else 'vector'} field")
            if expected_unit is not None and value.unit is not None and value.unit != expected_unit:
                raise ValueError(f"{name} unit {value.unit!r} does not match {expected_unit!r}")
            operation = _evaluator(value.evaluator)
            selected_times = times if value.time_dependent else times[:1]
        else:
            operation = _evaluator(value)
            selected_times = times
    frames = []
    for time in selected_times:
        # Функция может менять эту временную копию, но не сетку решателя.
        result = value if operation is None else operation(points.copy(), float(time))
        array = _owned_values(result, len(points), components, name)
        if positive and np.any(array <= 0):
            raise ValueError(f"{name} must be strictly positive at every sampled point/time")
        if nonnegative and np.any(array < 0):
            raise ValueError(f"{name} must be nonnegative at every sampled point/time")
        frames.append(array)
    return SampledField(tuple(frames))


@dataclass(frozen=True)
class ReactionLaw:
    """Локальная реакция и производная по состоянию.

    Parameters
    ----------
    value, derivative : callable
        Принимают состояние формы (n,), точки формы (n, 2) и
        физическое время. Возвращают число или массив формы (n,).
    pure : {True}
        Поведение функций неизменно при вычислении траектории и её
        производных.
    snapshot_factory : callable or None, optional
        Фабрика независимого ReactionLaw без собственной фабрики
        снимка. Нужна для изменяемых настроек функций.
    unit, derivative_unit : str or None, optional
        Метки единиц реакции и её производной по концентрации.
    """
    value: Callable
    derivative: Callable
    pure: bool
    snapshot_factory: Callable | None = None
    unit: str | None = None
    derivative_unit: str | None = None

    def __post_init__(self):
        if not callable(self.value) or not callable(self.derivative):
            raise TypeError("Reaction value and state derivative must be explicit callables")
        if self.pure is not True:
            raise ValueError("ReactionLaw requires the explicit pure=True lifetime contract")
        if self.snapshot_factory is not None and not callable(self.snapshot_factory):
            raise TypeError("snapshot_factory must be callable")
        for label in (self.unit, self.derivative_unit):
            if label is not None and (not isinstance(label, str) or not label.strip()):
                raise ValueError("Reaction unit labels must be nonempty strings")

    def snapshot(self):
        """Получить снимок настроек закона.

        Returns
        -------
        law : ReactionLaw
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
        if not isinstance(result, ReactionLaw) or result.snapshot_factory is not None:
            raise TypeError("snapshot_factory must return a self-contained pure ReactionLaw")
        return result

    def evaluate(self, state, points, time, *, derivative=False):
        """Вычислить локальную реакцию или её производную.

        Parameters
        ----------
        state : array_like, shape (n,)
            Концентрация в узлах.
        points : array_like, shape (n, 2)
            Координаты узлов в единицах длины, принятых вызывающим кодом.
        time : float
            Физический момент в единицах времени, принятых вызывающим кодом.
        derivative : bool, optional
            Вычислить производную по концентрации вместо значения
            реакции.

        Returns
        -------
        values : ndarray, shape (n,)
            Независимый вещественный массив в unit или derivative_unit;
            запись запрещена.
        """
        operation = self.derivative if derivative else self.value
        return _owned_values(operation(np.array(state, copy=True),
                                       np.array(points, copy=True), float(time)),
                             len(points), 1, "reaction derivative" if derivative else "reaction")


__all__ = ["ADRUnits", "ScalarField", "VectorField", "ReactionLaw"]
