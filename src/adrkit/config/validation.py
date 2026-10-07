"""Проверки значений JSON для явно переданных пользовательских схем."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any

from adrkit.errors import ConfigError


def closed_object(value: Any, allowed: set[str], required: set[str] = frozenset(),
                  *, where: str = "object") -> dict:
    """Проверить разрешённые и обязательные ключи JSON-объекта.

    Parameters
    ----------
    value : dict
        JSON-объект со строковыми ключами.
    allowed : collection of str
        Разрешённые имена полей.
    required : collection of str, optional
        Обязательные имена полей.
    where : str, optional
        Имя объекта в сообщении об ошибке.

    Returns
    -------
    object : dict
        Переданный словарь без копирования.
    """
    if type(value) is not dict:
        raise ConfigError(f"{where}: expected a JSON object")
    if any(type(k) is not str for k in value):
        raise ConfigError(f"{where}: keys must be strings")
    unknown, missing = set(value) - allowed, required - set(value)
    if unknown or missing:
        raise ConfigError(f"{where}: unknown={sorted(unknown)}, missing={sorted(missing)}")
    return value


def string(value: Any, *, where: str = "string") -> str:
    """Проверить непустую строку JSON.

    Parameters
    ----------
    value : str
        Строка с непробельным содержимым, представимая в UTF-8.
    where : str, optional
        Имя поля в сообщении об ошибке.

    Returns
    -------
    string : str
        Переданная строка без изменения.
    """
    if type(value) is not str or not value.strip():
        raise ConfigError(f"{where}: expected a nonempty string")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise ConfigError(f"{where}: invalid Unicode") from exc
    return value


def integer(value: Any, *, minimum: int | None = None, where: str = "integer") -> int:
    """Проверить целое значение JSON и нижнюю границу.

    Parameters
    ----------
    value : int
        Значение именно типа int; bool отклоняется.
    minimum : int or None, optional
        Включённая нижняя граница при её наличии.
    where : str, optional
        Имя поля в сообщении об ошибке.

    Returns
    -------
    integer : int
        Переданное целое значение.
    """
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ConfigError(f"{where}: expected integer >= {minimum}")
    return value


def number(value: Any, *, minimum: float | None = None,
           positive: bool = False, where: str = "number") -> float:
    """Преобразовать конечное число JSON к float.

    Parameters
    ----------
    value : int or float
        Число; преобразование целого должно сохранять точное
        значение.
    minimum : float or None, optional
        Включённая нижняя граница.
    positive : bool, optional
        Требовать строгую положительность.
    where : str, optional
        Имя поля в сообщении об ошибке.

    Returns
    -------
    number : float
        Конечное число; оба знака вещественного нуля нормализуются к
        положительному.
    """
    if type(value) not in (int, float):
        raise ConfigError(f"{where}: expected finite number (not bool/string)")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ConfigError(f"{where}: unrepresentable float") from exc
    if not math.isfinite(result) or (type(value) is int and int(result) != value):
        raise ConfigError(f"{where}: nonfinite or lossy integer-to-float conversion")
    if (minimum is not None and result < minimum) or (positive and result <= 0):
        raise ConfigError(f"{where}: out of range")
    return 0.0 if result == 0 else result


def sha256(value: Any, *, where: str = "sha256") -> str:
    """Проверить запись контрольной суммы SHA-256.

    Parameters
    ----------
    value : str
        64 шестнадцатеричных символа в нижнем регистре.
    where : str, optional
        Имя поля в сообщении об ошибке.

    Returns
    -------
    digest : str
        Проверенная исходная строка.
    """
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ConfigError(f"{where}: expected lowercase SHA256")
    return value


def _json_value(value: Any) -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        return number(value)
    if type(value) is list:
        return [_json_value(v) for v in value]
    if type(value) is dict and all(type(k) is str for k in value):
        return {k: _json_value(v) for k, v in value.items()}
    raise ConfigError("expected only JSON values and string object keys")


def canonical_bytes(value: Any) -> bytes:
    """Нормализовать JSON-значение в UTF-8 байты.

    Parameters
    ----------
    value : JSON value
        Скаляр JSON, список или словарь со строковыми ключами; числа
        float конечны.

    Returns
    -------
    data : bytes
        Компактное UTF-8 представление с сортировкой ключей и
        сохранением порядка списков.

    Notes
    -----
    Оба знака вещественного нуля нормализуются к 0.0.
    """
    try:
        return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ConfigError(f"invalid canonical JSON: {exc}") from exc


def digest(value: Any) -> str:
    """Вычислить SHA-256 нормализованного JSON-значения.

    Parameters
    ----------
    value : JSON value
        Значение, допустимое для canonical_bytes.

    Returns
    -------
    digest : str
        Шестнадцатеричная контрольная сумма нормализованных байтов.
    """
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def strict_json(data: bytes | str) -> Any:
    """Прочитать JSON без повторов ключей и неконечных чисел.

    Parameters
    ----------
    data : bytes or str
        UTF-8 байты либо текст JSON.

    Returns
    -------
    value : JSON value
        Скаляр, список или словарь; некорректный ввод вызывает
        ConfigError.
    """
    def pairs(items: list[tuple[str, Any]]) -> dict:
        result = {}
        for key, val in items:
            if key in result:
                raise ConfigError(f"duplicate JSON key: {key!r}")
            result[key] = val
        return result

    def constant(value: str) -> None:
        raise ConfigError(f"nonfinite JSON literal: {value}")

    try:
        if type(data) is bytes:
            data = data.decode("utf-8", errors="strict")
        if type(data) is not str:
            raise ConfigError("JSON input must be UTF-8 bytes or text")
        value = json.loads(data, object_pairs_hook=pairs, parse_constant=constant)
        canonical_bytes(value)
        return value
    except (ValueError, UnicodeError, RecursionError) as exc:
        if isinstance(exc, ConfigError):
            raise
        raise ConfigError(f"invalid UTF-8 JSON: {exc}") from exc


@dataclass(frozen=True, init=False)
class JSONRecord:
    """Неизменяемое нормализованное представление JSON.

    Parameters
    ----------
    value : JSON value
        Значение, допустимое для canonical_bytes.

    Notes
    -----
    Хранятся нормализованные байты. Преобразование обратно
    в изменяемые контейнеры создаёт новые объекты.
    """

    _bytes: bytes

    def __init__(self, value: Any):
        object.__setattr__(self, "_bytes", canonical_bytes(value))

    def to_dict(self) -> dict:
        """Получить запись в виде независимого словаря.

        Returns
        -------
        value : dict
            Словарь JSON; для записи другого типа вызывается ConfigError.
        """
        value = json.loads(self._bytes)
        if type(value) is not dict:
            raise ConfigError("record is not an object")
        return value

    def to_value(self) -> Any:
        """Получить сохранённое значение JSON.

        Returns
        -------
        value : JSON value
            Новый скаляр либо независимые JSON-контейнеры.
        """
        return json.loads(self._bytes)

    def to_bytes(self) -> bytes:
        """Получить нормализованные байты записи.

        Returns
        -------
        data : bytes
            Сохранённое неизменяемое UTF-8 представление.
        """
        return self._bytes

    @property
    def sha256(self) -> str:
        """Шестнадцатеричная SHA-256 сохранённых нормализованных байтов."""
        return hashlib.sha256(self._bytes).hexdigest()
