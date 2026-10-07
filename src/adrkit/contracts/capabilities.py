"""Прогнозы, производные и скалярные операции компонентов."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from adrkit.errors import CapabilityError

if TYPE_CHECKING:
    from adrkit.spaces import ArraySpace


class Predict(Protocol):
    """Прогноз по аргументу отображения."""
    def predict(self, x: Any) -> Any:
        """Вычислить прогноз по аргументу.

        Parameters
        ----------
        x : object
            Аргумент в пространстве отображения.

        Returns
        -------
        prediction : object
            Результат в пространстве значений отображения;
            форму и единицы задаёт реализация.
        """
        ...


class JVP(Protocol):
    """Действие производной в точке на направление."""
    def jvp(self, at: Any, direction: Any) -> Any:
        """Применить производную к направлению.

        Parameters
        ----------
        at, direction : object
            Точка и направление в пространстве аргументов.

        Returns
        -------
        tangent : object
            Направление в пространстве результата.
        """
        ...


class VJP(Protocol):
    """Перенос ковектора транспонированной производной."""

    def vjp(self, at: Any, cotangent: Any) -> Any:
        """Перенести ковектор в пространство аргументов.

        Parameters
        ----------
        at : object
            Точка в пространстве аргументов.
        cotangent : object
            Ковектор в пространстве результатов.

        Returns
        -------
        covector : object
            Координатный ковектор аргумента; метрический градиент
            строится отдельно.
        """
        ...


class ScalarValue(Protocol):
    """Значение скалярного функционала."""
    def value(self, x: Any) -> float:
        """Вычислить значение функционала.

        Parameters
        ----------
        x : object
            Точка области определения.

        Returns
        -------
        value : float
            Скалярное значение; единицы задаёт функционал.
        """
        ...


class ScalarCotangent(Protocol):
    """Координатная производная скалярного функционала."""

    def cotangent(self, x: Any) -> Any:
        """Вычислить координатный ковектор функционала.

        Parameters
        ----------
        x : object
            Точка области определения.

        Returns
        -------
        covector : object
            Координаты производной в пространстве аргумента;
            преобразование Рисса не выполняется.
        """
        ...


class ScalarValueAndCotangent(Protocol):
    """Совместное вычисление значения функционала и ковектора."""

    def value_and_cotangent(self, x: Any) -> tuple[float, Any]:
        """Вычислить значение и ковектор одним вызовом.

        Parameters
        ----------
        x : object
            Точка области определения.

        Returns
        -------
        value : float
            Значение функционала.
        covector : object
            Координатная производная в той же точке.
        """
        ...


class Prox(Protocol):
    """Проксимальная операция функционала с заданным шагом."""
    def prox(self, x: Any, step: float) -> Any:
        """Применить проксимальную операцию с заданным шагом.

        Parameters
        ----------
        x : object
            Точка области определения.
        step : float
            Положительный параметр шага.

        Returns
        -------
        point : object
            Результат в области определения; скалярное произведение
            задаёт реализация.
        """
        ...


class TruthSource(Protocol):
    """Значения заданного профиля источника."""

    def value(self, times: Any) -> Any:
        """Вычислить заданный профиль источника в моменты времени.

        Parameters
        ----------
        times : array_like
            Моменты в единицах времени профиля.

        Returns
        -------
        values : array_like
            Значения профиля в порядке times, в единицах интенсивности
            источника.
        """
        ...


class TruthDerivative(Protocol):
    """Производная заданного профиля источника по времени."""

    def derivative(self, times: Any) -> Any:
        """Вычислить производную заданного профиля по времени.

        Parameters
        ----------
        times : array_like
            Моменты, в которых производная профиля определена.

        Returns
        -------
        values : array_like
            Производная в порядке times, в единицах интенсивности за
            время.
        """
        ...


@dataclass(frozen=True)
class MapCapabilities:
    """Операции отображения между пространствами массивов.

    Parameters
    ----------
    predict : Predict
        Прогноз в пространстве codomain по аргументу из domain.
    domain, codomain : ArraySpace
        Пространства аргумента и результата.
    jvp : JVP or None, optional
        Действие производной на направление.
    vjp : VJP or None, optional
        Действие транспонированной производной на ковектор.
    """
    predict: Predict
    domain: ArraySpace
    codomain: ArraySpace
    jvp: JVP | None = None
    vjp: VJP | None = None

    def __post_init__(self) -> None:
        require_capability(self.predict, operation="predict", consumer="map")


@dataclass(frozen=True)
class ScalarCapabilities:
    """Операции скалярного функционала на пространстве массивов.

    Parameters
    ----------
    value : ScalarValue
        Значение функционала.
    domain : ArraySpace
        Пространство аргумента.
    cotangent : ScalarCotangent or None, optional
        Координатная производная.
    prox : Prox or None, optional
        Проксимальная операция.
    prox_pairing : str or None, optional
        Непустое описание скалярного произведения для prox; задаётся
        только вместе с prox.
    value_and_cotangent : ScalarValueAndCotangent or None, optional
        Совместная операция; требует также явно заданный cotangent.
    """
    value: ScalarValue
    domain: ArraySpace
    cotangent: ScalarCotangent | None = None
    prox: Prox | None = None
    prox_pairing: str | None = None
    value_and_cotangent: ScalarValueAndCotangent | None = None

    def __post_init__(self) -> None:
        require_capability(self.value, operation="value", consumer="scalar")
        if self.value_and_cotangent is not None and self.cotangent is None:
            raise ValueError("joint evaluation requires an explicitly declared cotangent")
        if self.prox is None and self.prox_pairing is not None:
            raise ValueError("prox_pairing requires an explicit prox")
        if self.prox is not None and (
            not isinstance(self.prox_pairing, str) or not self.prox_pairing.strip()
        ):
            raise ValueError("prox requires an explicit pairing")


_Capability = TypeVar("_Capability")


def require_capability(
    capability: _Capability | None, *, operation: str, consumer: str
) -> _Capability:
    """Получить явно предоставленную операцию компонента.

    Parameters
    ----------
    capability : object or None
        Операция компонента; None означает её отсутствие.
    operation, consumer : str
        Название операции и её потребителя для сообщения об ошибке.

    Returns
    -------
    capability : object
        Переданный объект без вызова.

    Notes
    -----
    При отсутствии операции вызывается CapabilityError.
    """
    if capability is None:
        raise CapabilityError(f"{consumer} requires {operation}")
    return capability
