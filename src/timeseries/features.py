"""Признаки из видимых значений, их наличия и календарного времени."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .contracts import (DataContractError, MaskedInputs, PanelSpec, clock_timestamp,
                        snapshot, strings)

__all__ = ["FeatureValues", "FeatureSet", "feature_from_function", "combine_features",
           "channel_features", "availability_features", "lag_features", "lead_features",
           "calendar_features"]


@dataclass(frozen=True)
class FeatureValues:
    """Хранить численные признаки с именами, единицами и временем.

    Parameters
    ----------
    panel_spec : PanelSpec
        Привязка сетки и порядка постов.
    times : tuple of pandas.Timestamp
        Все моменты объявленной сетки.
    names : sequence of str
        Различные имена признаков в порядке последней оси.
    units : sequence of str
        Единица каждого признака в порядке ``names``.
    values : array_like, shape (n_times, n_sites, n_features)
        Вещественные признаки в объявленных единицах; ``NaN`` обозначает
        недоступное значение.
    context : {'retrospective', 'causal'}
        Временной контекст построения признаков.

    Attributes
    ----------
    availability : ndarray of bool
        Маска конечных значений формы ``values``.
    sites : tuple of str
        Посты в порядке ``panel_spec``.

    Notes
    -----
    Численные значения копируются в буфер без права записи.
    """
    panel_spec: PanelSpec
    times: tuple
    names: tuple
    units: tuple
    values: np.ndarray
    context: str

    def __post_init__(self):
        if not isinstance(self.panel_spec, PanelSpec):
            raise DataContractError("features require a PanelSpec binding")
        names = strings(self.names, "feature names")
        units = tuple(self.units)
        if len(units) != len(names) or any(type(u) is not str or not u.strip() for u in units):
            raise DataContractError("one explicit unit per feature is required")
        if self.context not in ("retrospective", "causal"):
            raise DataContractError("feature context must be retrospective or causal")
        times = tuple(self.times)
        layout = self.panel_spec.to_dict()
        expected = tuple(pd.date_range(clock_timestamp(layout["start"], layout),
                         clock_timestamp(layout["stop"], layout),
                         freq=layout["frequency"], inclusive="left"))
        if times != expected:
            raise DataContractError("feature times must match the declared panel grid")
        shape = (len(times), len(layout["sites"]), len(names))
        raw = np.asarray(self.values)
        if raw.shape != shape or raw.dtype.kind not in "buif":
            raise DataContractError(f"features require real numeric values with shape {shape}")
        values = snapshot(raw, dtype=float, shape=shape, name="features")
        if np.isinf(values).any():
            raise DataContractError("feature values must be finite or explicitly unavailable (NaN)")
        for key, value in (("names", names), ("units", units), ("times", times), ("values", values)):
            object.__setattr__(self, key, value)

    @property
    def availability(self):
        """Вернуть маску конечных значений признаков."""
        return np.isfinite(self.values)

    @property
    def sites(self):
        """Вернуть посты в порядке спецификации панели."""
        return tuple(self.panel_spec.to_dict()["sites"])


@dataclass(frozen=True)
class _Block:
    """Хранить одну функцию признаков и её объявленные зависимости."""
    names: tuple
    units: tuple
    channels: tuple
    transform: object
    uses_future: bool | None


@dataclass(frozen=True)
class FeatureSet:
    """Объединить функции признаков в объявленном порядке.

    Parameters
    ----------
    _blocks : tuple of _Block
        Блоки, созданные фабриками признаков; имена всех признаков
        должны различаться.

    Attributes
    ----------
    names : tuple of str
        Имена в порядке блоков, затем признаков внутри каждого блока.
    uses_future : bool or None
        ``True`` при наличии будущей зависимости, ``None`` при неизвестной
        зависимости хотя бы одного блока, иначе ``False``.

    Notes
    -----
    Для создания используйте ``channel_features``, ``lag_features``,
    ``calendar_features``, ``feature_from_function`` и ``combine_features``.
    """
    _blocks: tuple

    def __post_init__(self):
        blocks = tuple(self._blocks)
        if not blocks or any(not isinstance(b, _Block) for b in blocks):
            raise DataContractError("FeatureSet requires explicit feature constructors")
        strings(tuple(n for b in blocks for n in b.names), "feature names")
        object.__setattr__(self, "_blocks", blocks)

    @property
    def names(self):
        """Вернуть имена признаков в порядке их вычисления."""
        return tuple(n for b in self._blocks for n in b.names)

    @property
    def uses_future(self):
        """Вернуть общую декларацию зависимости от будущих значений."""
        declared = tuple(b.uses_future for b in self._blocks)
        return True if True in declared else None if None in declared else False

    def transform(self, inputs, *, context="retrospective"):
        """Вычислить признаки из видимых входов панели.

        Parameters
        ----------
        inputs : MaskedInputs
            Входы без контрольных ответов; скрытые значения равны ``NaN``.
        context : {'retrospective', 'causal'}, optional
            ``causal`` допускает только блоки с ``uses_future=False``; ``retrospective``
            разрешает будущую или неизвестную зависимость.

        Returns
        -------
        features : FeatureValues
            Массив формы (n_times, n_sites, n_features) с объявленными
            именами и единицами.

        Notes
        -----
        Для внешней функции ``uses_future`` является обязательством автора.
        Выбор ``causal`` не исследует её код и внешнее состояние и не устанавливает
        причинность всей процедуры обучения и прогноза.
        """
        if not isinstance(inputs, MaskedInputs):
            raise DataContractError("features require MaskedInputs, never panel/truth")
        if context not in ("retrospective", "causal"):
            raise DataContractError("feature context must be retrospective or causal")
        if context == "causal" and self.uses_future is not False:
            raise DataContractError("causal features require an explicit no-future dependency")
        layout = inputs.spec.to_dict()
        required = {c for b in self._blocks for c in b.channels}
        if not required <= set(layout["channels"]):
            raise DataContractError("features require every declared panel channel")
        parts = []
        for block in self._blocks:
            units = tuple(layout["units"][u[1]] if u[0] == "channel" else u[1] for u in block.units)
            checked = FeatureValues(inputs.spec, inputs.times, block.names, units,
                                    block.transform(inputs), context)
            parts.append(checked)
        return FeatureValues(inputs.spec, inputs.times, self.names,
                             tuple(u for p in parts for u in p.units),
                             np.concatenate([p.values for p in parts], axis=2), context)


def feature_from_function(*, names, units, transform, uses_future=None):
    """Создать набор признаков из пользовательской функции.

    Parameters
    ----------
    names, units : sequence of str
        Различные имена и по одной буквальной единице на каждый признак.
    transform : callable
        Принимает ``MaskedInputs`` и возвращает вещественный массив
        (n_times, n_sites, len(``names``)) в исходном порядке. ``NaN`` означает
        недоступный признак.
    uses_future : bool or None, optional
        Объявленная зависимость от будущих входов. Для ``causal`` требуется
        ``False``; ``None`` означает неизвестную зависимость.

    Returns
    -------
    features : FeatureSet
        Один блок пользовательских признаков.
    """
    names = strings(names, "feature names")
    units = tuple(units)
    if (len(units) != len(names) or any(type(u) is not str or not u.strip() for u in units)
            or not callable(transform) or uses_future not in (None, True, False)
            or (uses_future is not None and type(uses_future) is not bool)):
        raise DataContractError("features require callable transform, explicit units and dependency")
    return FeatureSet((_Block(names, tuple(("literal", u) for u in units), (), transform, uses_future),))


def combine_features(*features):
    """Соединить наборы признаков в порядке аргументов.

    Parameters
    ----------
    *features : FeatureSet
        Непустая последовательность наборов; повторные имена
        отклоняются.

    Returns
    -------
    combined : FeatureSet
        Набор с порядком блоков и единиц каждого исходного набора.
    """
    if not features or any(not isinstance(f, FeatureSet) for f in features):
        raise DataContractError("combine_features requires FeatureSet records")
    return FeatureSet(tuple(b for f in features for b in f._blocks))


def _channels(channels):
    """Проверить и вернуть упорядоченные имена каналов."""
    return strings(channels, "feature channels")


def _selected(inputs, channels):
    """Скопировать видимые значения выбранных каналов."""
    indices = inputs.spec.to_dict()["channels"]
    return inputs.values[:, :, [indices.index(c) for c in channels]].copy()


def channel_features(channels):
    """Создать признаки из видимых значений выбранных каналов.

    Parameters
    ----------
    channels : sequence of str
        Различные имена каналов в требуемом порядке; каналы должны
        присутствовать во входной панели.

    Returns
    -------
    features : FeatureSet
        Признаки с именами ``'<канал>:value'`` и единицами исходных каналов.
        Пропуски и скрытые значения сохраняют ``NaN``.
    """
    channels = _channels(channels)
    return FeatureSet((_Block(tuple(f"{c}:value" for c in channels),
                              tuple(("channel", c) for c in channels), channels,
                              lambda x: _selected(x, channels), False),))


def availability_features(channels):
    """Создать индикаторы видимости выбранных каналов.

    Parameters
    ----------
    channels : sequence of str
        Различные имена каналов в требуемом порядке; каналы должны
        присутствовать во входной панели.

    Returns
    -------
    features : FeatureSet
        Признаки ``'<канал>:available'`` с единицей ``'1'``: 1 для конечного
        входа, 0 для ``NaN``.
    """
    channels = _channels(channels)
    return FeatureSet((_Block(tuple(f"{c}:available" for c in channels),
                              tuple(("literal", "1") for _ in channels), channels,
                              lambda x: np.isfinite(_selected(x, channels)).astype(float), False),))


def _shift_features(channels, steps, *, future):
    """Создать упорядоченные сдвиги каналов в прошлое или будущее."""
    channels = _channels(channels)
    if type(steps) not in (tuple, list) or not steps or any(type(s) is not int or s <= 0 for s in steps) or len(set(steps)) != len(steps):
        raise DataContractError("feature steps require ordered unique positive integers")
    steps = tuple(steps)
    role = "lead" if future else "lag"
    names = tuple(f"{c}:{role}{s}" for c in channels for s in steps)

    def transform(inputs):
        """Вычислить сдвиги видимых каналов, оставляя края и пропуски ``NaN``."""
        raw = _selected(inputs, channels)
        values = np.full((*raw.shape[:2], len(names)), np.nan)
        for j in range(len(channels)):
            for k, step in enumerate(steps):
                if step < len(raw):
                    if future:
                        values[:-step, :, j*len(steps)+k] = raw[step:, :, j]
                    else:
                        values[step:, :, j*len(steps)+k] = raw[:-step, :, j]
        return values

    return FeatureSet((_Block(names, tuple(("channel", c) for c in channels for _ in steps),
                              channels, transform, future),))


def lag_features(channels, steps):
    """Создать признаки из прошлых видимых значений каналов.

    Parameters
    ----------
    channels : sequence of str
        Различные имена каналов в требуемом порядке; каналы должны
        присутствовать во входной панели.
    steps : sequence of int
        Различные положительные сдвиги в шагах сетки, в требуемом
        порядке.

    Returns
    -------
    features : FeatureSet
        Признаки в порядке канал, затем сдвиг; единицы исходного канала
        и ``uses_future=False``.

    Notes
    -----
    Сдвиг s использует значение на s шагов сетки раньше текущей точки.
    Перед началом панели и для пропущенного исходного значения остаётся
    ``NaN``; прогнозы заполнения не используются.
    """
    return _shift_features(channels, steps, future=False)


def lead_features(channels, steps):
    """Создать признаки из будущих видимых значений каналов.

    Parameters
    ----------
    channels : sequence of str
        Различные имена каналов в требуемом порядке; каналы должны
        присутствовать во входной панели.
    steps : sequence of int
        Различные положительные сдвиги в шагах сетки, в требуемом
        порядке.

    Returns
    -------
    features : FeatureSet
        Признаки в порядке канал, затем сдвиг; единицы исходного канала
        и ``uses_future=True``.

    Notes
    -----
    Сдвиг s использует значение на s шагов сетки позже текущей точки.
    После конца панели и для пропущенного исходного значения остаётся ``NaN``.
    В контексте ``causal`` этот набор отклоняется.
    """
    return _shift_features(channels, steps, future=True)


def calendar_features(components):
    """Создать календарные признаки по времени панели.

    Parameters
    ----------
    components : sequence of str
        Различные компоненты в требуемом порядке: ``hour``, ``hour_sin``,
        ``hour_cos``, ``dayofyear``, ``dayofyear_sin``, ``dayofyear_cos``, ``weekday`` или
        ``month``.

    Returns
    -------
    features : FeatureSet
        Календарные признаки для каждого поста. ``hour`` имеет единицу ``'h'``,
        ``dayofyear`` — ``'d'``, остальные — ``'1'``.

    Notes
    -----
    Время берётся в объявленном часовом поясе панели. Номер дня недели
    начинается с 0 для понедельника; месяц — с 1. Годовая синусоидальная
    фаза использует среднюю длину года 365.2425 дня.
    """
    components = strings(components, "calendar components")
    allowed = {"hour", "hour_sin", "hour_cos", "dayofyear", "dayofyear_sin",
               "dayofyear_cos", "weekday", "month"}
    if not set(components) <= allowed:
        raise DataContractError("unsupported calendar feature")

    def transform(inputs):
        """Вычислить календарные столбцы и повторить их для каждого поста."""
        index = pd.DatetimeIndex(inputs.times)
        hour = index.hour.to_numpy() + index.minute.to_numpy()/60 + index.second.to_numpy()/3600
        day = index.dayofyear.to_numpy()
        values = {"hour": hour, "hour_sin": np.sin(2*np.pi*hour/24),
                  "hour_cos": np.cos(2*np.pi*hour/24), "dayofyear": day,
                  "dayofyear_sin": np.sin(2*np.pi*(day-1)/365.2425),
                  "dayofyear_cos": np.cos(2*np.pi*(day-1)/365.2425),
                  "weekday": index.dayofweek.to_numpy(), "month": index.month.to_numpy()}
        matrix = np.column_stack([values[c] for c in components])
        return np.broadcast_to(matrix[:, None, :], (len(index), len(inputs.spec.to_dict()["sites"]), len(components))).copy()

    units = tuple(("literal", "h" if c == "hour" else "d" if c == "dayofyear" else "1") for c in components)
    return FeatureSet((_Block(tuple(f"calendar:{c}" for c in components), units, (), transform, False),))
