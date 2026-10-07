"""Уравнение и коэффициенты двумерной адвекции, диффузии и реакции."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from typing import TypeVar

import numpy as np

from adrkit.coefficients import ADRUnits, ReactionLaw, ScalarField, VectorField
from adrkit.errors import CapabilityError
from adrkit.state_residuals import StateResidual


_Solver = TypeVar("_Solver")


def _number(value: object, *, name: str, positive: bool = False,
            nonnegative: bool = False) -> float:
    """Проверить конечное вещественное число и ограничения знака."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real scalar")
    if isinstance(value, (int, float, np.integer, np.floating)):
        result = float(value)
    elif isinstance(value, np.ndarray) and value.shape == () and value.dtype.kind in "iuf":
        result = float(value)
    else:
        raise ValueError(f"{name} must be a finite real scalar")
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0:
        raise ValueError(f"{name} must be strictly positive")
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _vector(value: object, *, name: str) -> tuple[float, float]:
    """Скопировать две конечные вещественные координаты в кортеж."""
    if not isinstance(value, (tuple, list, np.ndarray)):
        raise ValueError(f"{name} must be a finite real two-vector")
    array = np.asarray(value)
    if array.shape != (2,) or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real two-vector")
    # Проверяются исходные элементы: преобразование смешанного bool/числового списка в float не
    # должно скрывать bool.
    return tuple(_number(item, name=name) for item in value)


def _coefficient(value: object, *, name: str, vector: bool = False,
                 positive: bool = False, nonnegative: bool = False,
                 expected_unit: str | None = None) -> object:
    if isinstance(value, ScalarField):
        if vector != isinstance(value, VectorField):
            raise TypeError(f"{name} requires a {'vector' if vector else 'scalar'} field")
        if expected_unit is not None and value.unit is not None and value.unit != expected_unit:
            raise ValueError(f"{name} unit {value.unit!r} does not match {expected_unit!r}")
        return value
    if callable(value) or callable(getattr(value, "evaluate", None)):
        return value
    if vector:
        return _vector(value, name=name)
    return _number(value, name=name, positive=positive, nonnegative=nonnegative)


@dataclass(frozen=True, kw_only=True)
class ADRProblem:
    r"""Данные двумерного уравнения адвекции, диффузии и реакции.

    Parameters
    ----------
    diffusion : float, ScalarField or callable
        Положительный коэффициент диффузии в единицах длина²/время.
    velocity : array_like, VectorField or callable
        Скорость формы (2,) либо поле со значениями формы (n, 2), в
        единицах длина/время.
    linear_loss : float, ScalarField or callable
        Неотрицательный коэффициент линейных потерь в обратных
        единицах времени.
    horizon : float
        Конечная положительная длительность решения.
    reaction_gamma : float, ScalarField or callable, optional
        Неотрицательный коэффициент встроенной реакции tanh, в
        обратных единицах времени.
    reaction_c_star : float, ScalarField or callable, optional
        Положительный масштаб концентрации встроенной реакции.
    reaction : ReactionLaw or None, optional
        Пользовательская локальная реакция вместо встроенной.
    state_residual : StateResidual or None, optional
        Дополнительный дискретный член по состоянию.
    source_position : array_like, shape (2,), or None, optional
        Положение точечного источника; None оставляет только
        распределённую нагрузку.
    time_origin : float, optional
        Начало физического времени: поля получают time_origin плюс
        время решателя.
    units : ADRUnits or None, optional
        Метки согласованных единиц коэффициентов и концентрации.
    initial_state, background : {0.0}, optional
        Нулевые начальные данные и граничный фон.
    diffusion_form : {'divergence'}, optional
        Дивергентная форма диффузии.
    advection_form : {'advective'}, optional
        Перенос в форме скорости, умноженной на градиент.

    Notes
    -----
    На прямоугольной области используется уравнение

    .. math::

        \partial_t c - \nabla\!\cdot(D\nabla c)
        + v\cdot\nabla c + \kappa c + R(c,x,t) = f.

    Здесь :math:`c` — концентрация, :math:`x` — пространственная
    координата, :math:`t` — физическое время, :math:`f` — нагрузка.
    Коэффициенты :math:`D,v,\kappa` соответствуют `diffusion`,
    `velocity` и `linear_loss`. Локальная реакция :math:`R` задаётся
    `reaction`; при её отсутствии используется

    .. math::

        R(c,x,t)=\gamma(x,t)c_*(x,t)
                 \tanh\!\left(c/c_*(x,t)\right).

    Здесь :math:`\gamma,c_*` — `reaction_gamma,reaction_c_star`.
    Начальные и граничные значения нулевые. Добавочный `StateResidual`
    задаётся в дискретной задаче.
    """

    diffusion: object
    velocity: object
    linear_loss: object
    horizon: float
    reaction_gamma: object = 0.0
    reaction_c_star: object = 1.0
    reaction: ReactionLaw | None = None
    state_residual: StateResidual | None = None
    source_position: tuple[float, float] | None = (0.0, 0.0)
    time_origin: float = 0.0
    units: ADRUnits | None = None
    initial_state: float = 0.0
    background: float = 0.0
    diffusion_form: str = "divergence"
    advection_form: str = "advective"

    def __post_init__(self) -> None:
        if self.units is not None and not isinstance(self.units, ADRUnits):
            raise TypeError("units must be ADRUnits; labels do not convert values")
        if self.diffusion_form != "divergence" or self.advection_form != "advective":
            raise CapabilityError("ADRProblem implements -div(D grad c) + v dot grad c only")
        for name, positive in (("horizon", True), ("time_origin", False),
                               ("initial_state", False), ("background", False)):
            object.__setattr__(self, name, _number(getattr(self, name), name=name,
                                                 positive=positive))
        if self.initial_state != 0 or self.background != 0:
            raise CapabilityError("ADRProblem implements zero initial state and background only")
        if self.source_position is not None:
            object.__setattr__(self, "source_position", _vector(self.source_position,
                                                              name="source_position"))
        declarations = (
            ("diffusion", False, True, False, "diffusion"),
            ("velocity", True, False, False, "velocity"),
            ("linear_loss", False, False, True, "loss"),
            ("reaction_gamma", False, False, True, "loss"),
            ("reaction_c_star", False, True, False, "concentration"),
        )
        for name, vector, positive, nonnegative, unit in declarations:
            expected = None if self.units is None else getattr(self.units, unit)
            value = _coefficient(getattr(self, name), name=name, vector=vector,
                                 positive=positive, nonnegative=nonnegative,
                                 expected_unit=expected)
            object.__setattr__(self, name, value)
        if self.reaction is not None:
            if not isinstance(self.reaction, ReactionLaw):
                raise TypeError("reaction must be an explicit ReactionLaw")
            if self.reaction.pure is not True:
                raise ValueError("reaction requires the explicit pure=True lifetime contract")
            if self.units is not None:
                for label, expected in ((self.reaction.unit, self.units.reaction),
                                        (self.reaction.derivative_unit, self.units.loss)):
                    if label is not None and label != expected:
                        raise ValueError(f"Reaction unit {label!r} does not match {expected!r}")
        if self.state_residual is not None:
            if not isinstance(self.state_residual, StateResidual):
                raise TypeError("state_residual must be an explicit StateResidual")
            if self.state_residual.pure is not True:
                raise ValueError("state_residual requires the explicit pure=True lifetime contract")
            if self.units is not None:
                for label, expected in ((self.state_residual.unit, self.units.reaction),
                                        (self.state_residual.derivative_unit, self.units.loss)):
                    if label is not None and label != expected:
                        raise ValueError(f"StateResidual unit {label!r} does not match {expected!r}")

    @classmethod
    def from_model(cls, model: Mapping[str, object]) -> ADRProblem:
        """Создать данные уравнения из словаря.

        Parameters
        ----------
        model : Mapping
            Поля конструктора ADRProblem; неизвестные ключи отклоняются.

        Returns
        -------
        problem : ADRProblem
            Проверенные данные уравнения.
        """
        if not isinstance(model, Mapping):
            raise TypeError("model must be a mapping")
        allowed = {item.name for item in fields(cls)}
        unknown = set(model) - allowed
        if unknown:
            raise ValueError(f"Unknown ADR model keys: {sorted(unknown, key=str)!r}")
        return cls(**dict(model))

    def to_model(self) -> dict[str, object]:
        """Представить данные уравнения словарём.

        Returns
        -------
        model : dict
            Значения всех полей; объекты коэффициентов и законов
            сохраняются по ссылке.
        """
        return {item.name: getattr(self, item.name) for item in fields(self)}

    def make_solver(self, factory: Callable[..., _Solver], /, **numerical_options: object) -> _Solver:
        """Создать решатель указанной фабрикой.

        Parameters
        ----------
        factory : callable
            Принимает словарь данных ADR первым аргументом.
        **numerical_options
            Численные настройки, передаваемые фабрике без изменения.

        Returns
        -------
        solver : object
            Результат пользовательской фабрики.
        """
        if not callable(factory):
            raise TypeError("solver factory must be callable")
        return factory(self.to_model(), **numerical_options)


__all__ = ["ADRProblem"]
