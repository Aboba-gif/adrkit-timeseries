"""Панели временных рядов, маски и результаты заполнения."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import re

import numpy as np
import pandas as pd


class DataContractError(ValueError):
    """Несогласованные параметры, данные или метаданные панели."""


class MissingInputError(DataContractError):
    """Отсутствующий обязательный файл, столбец, пост или канал."""


class SourceIntegrityError(DataContractError):
    """Несовпадение SHA-256 файла с указанной контрольной суммой."""


class DuplicateObservationError(DataContractError):
    """Повторное значение для одной точки времени, поста и канала."""


def closed(value, fields, *, optional=(), where="record"):
    """Проверить обязательные и допустимые ключи словаря.

    Возвращает исходный словарь; ``optional`` задаёт необязательные поля,
    ``where`` — имя записи в сообщении об ошибке.
    """
    if type(value) is not dict or any(type(k) is not str for k in value):
        raise DataContractError(f"{where}: expected an object")
    unknown, missing = set(value) - set(fields), set(fields) - set(optional) - set(value)
    if unknown or missing:
        raise DataContractError(f"{where}: unknown={sorted(unknown)}, missing={sorted(missing)}")
    return value


def text(value, where):
    """Вернуть непустую строку либо отклонить значение.

    ``where`` задаёт имя поля в сообщении об ошибке.
    """
    if type(value) is not str or not value.strip():
        raise DataContractError(f"{where}: expected nonempty string")
    return value


def choice(value, allowed, where):
    """Вернуть строку из допустимого набора либо отклонить значение.

    ``allowed`` содержит допустимые строки, ``where`` обозначает поле ошибки.
    """
    if type(value) is not str or value not in allowed:
        raise DataContractError(f"{where}: expected one of {sorted(allowed)}")
    return value


def strings(value, where, *, nonempty=True):
    """Вернуть упорядоченный кортеж различных непустых строк.

    ``nonempty=False`` разрешает пустую входную последовательность;
    ``where`` обозначает проверяемое поле.
    """
    if type(value) not in (list, tuple) or (nonempty and not value):
        raise DataContractError(f"{where}: expected an ordered sequence")
    result = tuple(text(v, where) for v in value)
    if len(result) != len(set(result)):
        raise DataContractError(f"{where}: duplicate entries")
    return result


def _clock(value, where):
    """Проверить описание часов и вернуть шаг сетки и её начало."""
    choice(value["clock"], {"naive", "localized"}, where + ".clock")
    choice(value["calendar"], {"proleptic_gregorian"}, where + ".calendar")
    zone = value["timezone"]
    if value["clock"] == "naive":
        if zone is not None:
            raise DataContractError(f"{where}: naive clock requires timezone=null")
    else:
        text(zone, where + ".timezone")
    try:
        step = pd.Timedelta(text(value["frequency"], where + ".frequency"))
        if step <= pd.Timedelta(0) or pd.isna(step):
            raise ValueError("nonpositive frequency")
        stamp = clock_timestamp(value["origin"], value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise DataContractError(f"{where}: invalid fixed grid") from exc
    return step, stamp


def clock_timestamp(value, clock):
    """Разобрать временную метку в часовом поясе панели.

    Parameters
    ----------
    value : str
        Временная метка границы или начала сетки.
    clock : dict
        Поля ``clock`` и ``timezone`` из описания панели или источника.

    Returns
    -------
    stamp : pandas.Timestamp
        Метка без часового пояса для ``naive``-часов или в объявленном
        часовом поясе для ``localized``-часов.

    Notes
    -----
    Для ``localized``-часов метка без зоны локализуется, а метка с зоной
    переводится в объявленную зону. Неоднозначные и несуществующие
    границы DST отклоняются. ``naive``-часы требуют метку без смещения.
    """
    try:
        stamp = pd.Timestamp(text(value, "timestamp"))
        if pd.isna(stamp):
            raise DataContractError("NaT is not a grid boundary")
        if clock["clock"] == "naive":
            if stamp.tz is not None:
                raise DataContractError("offset timestamp cannot enter a naive clock")
        else:
            if stamp.tz is None:
                stamp = stamp.tz_localize(clock["timezone"], ambiguous="raise", nonexistent="raise")
            else:
                stamp = stamp.tz_convert(clock["timezone"])
    except DataContractError:
        raise
    except Exception as exc:
        # Классы ошибок DST и часовых зон зависят от реализации pandas.
        raise DataContractError("invalid clock timestamp/timezone/DST boundary") from exc
    return stamp


class _JSONSpec:
    """Общее преобразование неизменяемой JSON-записи в словарь."""
    def to_dict(self):
        """Вернуть независимый словарь полей спецификации."""
        return json.loads(self._json)

    @classmethod
    def from_dict(cls, value):
        """Создать спецификацию из словаря полей её конструктора."""
        return cls(value)


@dataclass(frozen=True, init=False)
class InputSourceSpec(_JSONSpec):
    """Описать CSV-источник, время и единицы его значений.

    Parameters
    ----------
    value : dict
        Описание с полями ``id``, ``path``, ``sha256``, ``format``, ``columns``, ``time``, ``site``,
        ``units``, ``missing_values`` и ``route``. ``columns.time`` содержит один
        заголовок времени либо пару даты и времени; ``columns.values``
        связывает каналы с заголовками CSV. ``site`` задаёт один пост или
        столбец его имени. ``units`` задаёт единицу каждого канала.

    Notes
    -----
    ``format`` выбирает CSV, его кодировку и разделитель. ``time`` задаёт
    разбор локального времени, часы, календарь, зону, DST, шаг и ``origin``.
    Разбор ``pandas_infer``, а также DST ``shift``/``drop`` допускаются только при
    ``route='quarantine'``. ``missing_values`` содержит буквальные CSV-токены.
    SHA-256 задаётся 64 строчными шестнадцатеричными цифрами.

    Относительный ``path`` разрешается от абсолютного ``anchor`` в ``build_data``.
    Создание записи не читает файл; ``to_dict`` возвращает независимую копию.
    """
    _json: str

    def __init__(self, value):
        fields = {"id", "path", "sha256", "format", "columns", "time", "site",
                  "units", "missing_values", "route"}
        closed(value, fields, where="source")
        for key in ("id", "path"):
            text(value[key], "source." + key)
        if type(value["sha256"]) is not str or not re.fullmatch(r"[0-9a-f]{64}", value["sha256"]):
            raise DataContractError("source.sha256: expected lowercase SHA256")
        fmt = closed(value["format"], {"kind", "parameters"}, where="format")
        choice(fmt["kind"], {"csv"}, "format.kind")
        params = closed(fmt["parameters"], {"separator", "encoding"}, where="csv.parameters")
        if len(text(params["separator"], "separator")) != 1 or params["separator"] in "\r\n\x00":
            raise DataContractError("separator must be one non-newline character")
        text(params["encoding"], "encoding")
        columns = closed(value["columns"], {"time", "values"}, where="columns")
        times = strings(columns["time"], "columns.time")
        if len(times) not in (1, 2):
            raise DataContractError("columns.time requires timestamp or date/time")
        vals = columns["values"]
        if type(vals) is not dict or not vals:
            raise DataContractError("columns.values requires channel-to-header mapping")
        strings(list(vals), "channels")
        headers = strings(list(vals.values()), "value headers")
        if set(times) & set(headers):
            raise DataContractError("time/value columns overlap")
        site = value["site"]
        if type(site) is not dict:
            raise DataContractError("site requires explicit kind")
        kind = choice(site.get("kind"), {"fixed", "column"}, "site.kind")
        site_field = "id" if kind == "fixed" else "column"
        closed(site, {"kind", site_field}, where="site")
        text(site[site_field], "site." + site_field)
        if kind == "column" and site[site_field] in set(times) | set(headers):
            raise DataContractError("site/time/value columns overlap")
        units = closed(value["units"], set(vals), where="source.units")
        for unit in units.values():
            text(unit, "unit (use explicit 'unknown' when unknown)")
        missing = value["missing_values"]
        if (type(missing) is not list or any(type(v) is not str for v in missing)
                or len(missing) != len(set(missing))):
            raise DataContractError("missing_values requires unique literal string tokens")
        route = choice(value["route"], {"strict", "quarantine"}, "route")
        time = closed(value["time"], {"parser", "format", "timezone", "dst_policy", "calendar",
                                        "clock", "frequency", "origin"}, where="time")
        choice(time["parser"], {"format", "pandas_infer"}, "time.parser")
        choice(time["dst_policy"], {"fail", "shift", "drop"}, "time.dst_policy")
        if time["parser"] == "format":
            text(time["format"], "time.format")
            if "%z" in time["format"] or "%Z" in time["format"]:
                raise DataContractError("source timestamps must be wall time; timezone is explicit")
        elif time["format"] is not None or route != "quarantine":
            raise DataContractError("pandas_infer requires quarantine route and format=null")
        if time["clock"] == "naive" and time["dst_policy"] != "fail":
            raise DataContractError("DST policy for a naive clock must be fail")
        if route == "strict" and time["dst_policy"] != "fail":
            raise DataContractError("DST shift/drop requires explicit quarantine route")
        _clock(time, "source.time")
        object.__setattr__(self, "_json", json.dumps(value, sort_keys=True, allow_nan=False))


@dataclass(frozen=True, init=False)
class PanelSpec(_JSONSpec):
    """Описать регулярную сетку и порядок осей панели.

    Parameters
    ----------
    value : dict
        Поля ``start``, ``stop``, ``frequency``, ``origin``, ``clock``, ``timezone``, ``calendar``,
        ``sites``, ``channels`` и ``units``; необязательное ``duplicates`` выбирает
        ``'reject'`` или ``'first_with_quarantine'``. В ``sites`` и ``channels`` задан
        порядок координат, ``units`` связывает канал с его единицей.

    Notes
    -----
    Диапазон [``start``, ``stop``) непуст и выровнен относительно ``origin``.
    Шаг ``frequency`` — фиксированный положительный ``pandas.Timedelta``.
    Поддерживается ``proleptic_gregorian``: ``naive`` требует ``timezone=None``,
    ``localized`` — явный часовой пояс. Единицы служат метками данных;
    перевод значений между ними не выполняется.
    """
    _json: str

    def __init__(self, value):
        closed(value, {"start", "stop", "frequency", "origin", "clock", "timezone",
                       "calendar", "sites", "channels", "units", "duplicates"},
               optional={"duplicates"}, where="panel")
        sites = strings(value["sites"], "panel.sites")
        channels = strings(value["channels"], "panel.channels")
        units = closed(value["units"], set(channels), where="panel.units")
        for unit in units.values():
            text(unit, "panel.unit")
        duplicate = choice(value.get("duplicates", "reject"), {"reject", "first_with_quarantine"}, "duplicates")
        step, origin = _clock(value, "panel")
        start, stop = (clock_timestamp(value[k], value) for k in ("start", "stop"))
        if stop <= start or (start - origin) % step or (stop - origin) % step:
            raise DataContractError("panel range must be nonempty, ordered and grid aligned")
        normalized = {**value, "sites": list(sites), "channels": list(channels), "duplicates": duplicate}
        object.__setattr__(self, "_json", json.dumps(normalized, sort_keys=True, allow_nan=False))


def source_resource_pointers(specs, *, prefix):
    """Вернуть адреса пути и контрольной суммы источников в JSON.

    Parameters
    ----------
    specs : sequence of InputSourceSpec or dict
        Непустая упорядоченная последовательность описаний источников.
    prefix : str
        JSON Pointer списка источников, например ``'/data/sources'``;
        шаблоны не допускаются.

    Returns
    -------
    pointers : tuple of tuple of str
        Пары JSON Pointer к ``path`` и ``sha256`` для каждого источника в
        исходном порядке.
    """
    if (type(prefix) is not str or not prefix.startswith("/")
            or re.search(r"~(?![01])", prefix) or "*" in prefix):
        raise DataContractError("prefix must be a non-root JSON Pointer without wildcards")
    if type(specs) not in (list, tuple) or not specs:
        raise DataContractError("sources must be nonempty and ordered")
    for spec in specs:
        if not isinstance(spec, InputSourceSpec):
            InputSourceSpec(spec)
    return tuple((f"{prefix}/{i}/path", f"{prefix}/{i}/sha256") for i in range(len(specs)))


def snapshot(value, *, dtype, shape, name):
    """Скопировать массив заданной формы в неизменяемый буфер.

    Parameters
    ----------
    value : array_like
        Исходные значения или булева маска.
    dtype : data-type
        Тип выходного массива. Для float требуется вещественный числовой
        вход; для bool — булева маска.
    shape : tuple of int
        Точная ожидаемая форма массива; неявное согласование форм
        массивов не выполняется.
    name : str
        Имя массива в сообщении об ошибке.

    Returns
    -------
    array : ndarray
        Независимый массив формы ``shape`` с типом ``dtype``; право записи
        нельзя включить обратно.
    """
    array = np.asarray(value)
    if dtype is float and array.dtype.kind not in "buif":
        raise DataContractError(f"{name}: real numeric values required")
    if dtype is bool and array.dtype != np.dtype(bool):
        raise DataContractError(f"{name}: boolean mask required")
    if array.shape != shape:
        raise DataContractError(f"{name}: expected shape {shape}, got {array.shape}")
    array = np.asarray(array, dtype=dtype)
    # Буфер bytes не позволяет снова включить запись в массив.
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(shape)


@dataclass(frozen=True)
class RowLocator:
    """Указать происхождение одного CSV-значения.

    Parameters
    ----------
    source_id : str
        Идентификатор источника.
    sha256 : str
        SHA-256 прочитанных байтов файла.
    row_number : int
        Номер CSV-записи: заголовок имеет номер 1, данные начинаются с
        2.
    site, channel : str
        Пост и физический канал этого значения.
    raw_time, raw_value : str
        Исходный текст времени и значения до числового разбора.
    """
    source_id: str
    sha256: str
    row_number: int
    site: str
    channel: str
    raw_time: str
    raw_value: str


@dataclass(frozen=True)
class QuarantinedValue:
    """Сохранить адрес исключённого значения и причину исключения.

    Parameters
    ----------
    row : RowLocator
        Адрес исходного значения.
    reason : str
        Причина исключения, например ``off_grid`` или ``duplicate_timestamp``.
    """
    row: RowLocator
    reason: str


@dataclass(frozen=True)
class AcceptedValue:
    """Связать принятое CSV-значение с координатой панели.

    Parameters
    ----------
    row : RowLocator
        Адрес исходного значения, в том числе принятого обозначения
        пропуска.
    index : tuple of int, length 3
        Индексы времени, поста и канала в порядке осей панели.
    """
    row: RowLocator
    index: tuple[int, int, int]


@dataclass(frozen=True)
class SourceProvenance:
    """Сохранить описание и адрес прочитанного файла.

    Parameters
    ----------
    spec : InputSourceSpec
        Объявленные параметры источника.
    resolved_path : str
        Разрешённый абсолютный путь файла.
    byte_count : int
        Число прочитанных байтов.
    """
    spec: InputSourceSpec
    resolved_path: str
    byte_count: int


def _layout(record):
    """Проверить временную сетку записи и вернуть форму её панели."""
    if not isinstance(record.spec, PanelSpec):
        raise DataContractError("layout requires PanelSpec")
    spec = record.spec.to_dict()
    times = tuple(record.times)
    expected = tuple(pd.date_range(clock_timestamp(spec["start"], spec),
                                   clock_timestamp(spec["stop"], spec),
                                   freq=spec["frequency"], inclusive="left"))
    if times != expected:
        raise DataContractError("times differ from explicit panel grid")
    object.__setattr__(record, "times", times)
    return (len(times), len(spec["sites"]), len(spec["channels"]))


@dataclass(frozen=True)
class NetworkPanel:
    """Хранить исходные наблюдения и маски присутствия значений.

    Parameters
    ----------
    spec : PanelSpec
        Сетка, порядок постов и каналов и единицы значений.
    times : tuple of pandas.Timestamp
        Все узлы объявленной сетки в её часовом поясе.
    values : array_like, shape (n_times, n_sites, n_channels)
        Конечные исходные значения в единицах каналов; ``NaN`` обозначает
        отсутствие наблюдения.
    row_present : array_like of bool, shape (n_times, n_sites, n_channels)
        Принятый CSV-элемент присутствует, включая объявленный пропуск.
    original_observed : array_like of bool, shape (n_times, n_sites, n_channels)
        Маска конечных исходных наблюдений.
    raw_missing : array_like of bool, shape (n_times, n_sites, n_channels)
        Принятый CSV-элемент является обозначением пропуска.
    row_origins : tuple of AcceptedValue, optional
        Происхождение принятых значений в порядке источник, CSV-запись,
        канал.

    Notes
    -----
    Массивы копируются в буферы без права записи. ``original_observed``
    совпадает с конечностью ``values``; ``raw_missing`` выбирает присутствующие
    элементы без исходного наблюдения. Отсутствующая строка и принятый
    CSV-пропуск остаются разными случаями.
    """
    spec: PanelSpec
    times: tuple
    values: np.ndarray
    row_present: np.ndarray
    original_observed: np.ndarray
    raw_missing: np.ndarray
    # Порядок загрузки: источник, CSV-запись, канал; обозначения пропуска тоже учитываются.
    
    row_origins: tuple[AcceptedValue, ...] = ()

    def __post_init__(self):
        shape = _layout(self)
        for name in ("values", "row_present", "original_observed", "raw_missing"):
            object.__setattr__(self, name, snapshot(getattr(self, name), shape=shape,
                              dtype=float if name == "values" else bool, name=name))
        if (np.any(np.isinf(self.values)) or
                not np.array_equal(np.isfinite(self.values), self.original_observed) or
                np.any(self.original_observed & ~self.row_present) or
                not np.array_equal(self.raw_missing, self.row_present & ~self.original_observed)):
            raise DataContractError("inconsistent original/presence/missing masks")
        origins = tuple(self.row_origins)
        if any(not isinstance(row, AcceptedValue) for row in origins):
            raise DataContractError("row_origins require AcceptedValue records")
        object.__setattr__(self, "row_origins", origins)


@dataclass(frozen=True)
class LoadedSources:
    """Объединить загруженную панель и происхождение её значений.

    Parameters
    ----------
    panel : NetworkPanel
        Исходные значения и маски на регулярной сетке.
    quarantine : tuple of QuarantinedValue
        Исключённые CSV-значения и их причины.
    provenance : tuple of SourceProvenance
        Описания реально прочитанных файлов.
    """
    panel: NetworkPanel
    quarantine: tuple[QuarantinedValue, ...]
    provenance: tuple[SourceProvenance, ...]


@dataclass(frozen=True)
class MaskedInputs:
    """Хранить видимые входы метода отдельно от скрытых ответов.

    Parameters
    ----------
    spec : PanelSpec
        Сетка, порядок постов и каналов и единицы значений.
    times : tuple of pandas.Timestamp
        Все узлы объявленной сетки в её часовом поясе.
    values : array_like, shape (n_times, n_sites, n_channels)
        Видимые значения в единицах каналов; скрытые и отсутствующие
        наблюдения равны ``NaN``.
    row_present, original_observed : array_like of bool, shape (n_times, n_sites, n_channels)
        Маски принятого CSV-элемента и конечного исходного наблюдения до
        скрытия.
    artificial_mask, mit_mask : array_like of bool, shape (n_times, n_sites, n_channels)
        Непересекающиеся маски скрытия исходных наблюдений. ``mit_mask``
        допускается только для ``train``.
    protocol_id, population_id : str
        Идентификаторы протокола и выборки.
    population_kind : {'train', 'calibration', 'selection', 'test', 'export'}
        Назначение выборки.

    Attributes
    ----------
    visible : ndarray of bool
        Маска исходных наблюдений, не скрытых ни одной из двух масок.

    Notes
    -----
    Массивы копируются без права записи. Функции признаков, ``fit`` и ``predict``
    получают эту запись, без значений ``EvaluationTruth``.
    """
    spec: PanelSpec
    times: tuple
    values: np.ndarray
    row_present: np.ndarray
    original_observed: np.ndarray
    artificial_mask: np.ndarray
    mit_mask: np.ndarray
    protocol_id: str
    population_id: str
    population_kind: str

    def __post_init__(self):
        shape = _layout(self)
        for key in ("protocol_id", "population_id"):
            text(getattr(self, key), key)
        choice(self.population_kind, {"train", "calibration", "selection", "test", "export"}, "population_kind")
        for key in ("values", "row_present", "original_observed", "artificial_mask", "mit_mask"):
            object.__setattr__(self, key, snapshot(getattr(self, key), shape=shape,
                              dtype=float if key == "values" else bool, name=key))
        hidden = self.artificial_mask | self.mit_mask
        visible = self.original_observed & ~hidden
        if self.mit_mask.any() and self.population_kind != "train":
            raise DataContractError("MIT supervision is restricted to original training observations")
        if (np.any(self.artificial_mask & self.mit_mask) or np.any(hidden & ~self.original_observed)
                or np.any(self.original_observed & ~self.row_present) or np.any(np.isinf(self.values))
                or not np.array_equal(visible, np.isfinite(self.values))):
            raise DataContractError("inconsistent concealment/observed/presence masks")

    @property
    def visible(self):
        """Вернуть маску исходных наблюдений, оставшихся видимыми."""
        return self.original_observed & ~(self.artificial_mask | self.mit_mask)


@dataclass(frozen=True)
class EvaluationTruth:
    """Хранить контрольные ответы для оценки заполнения.

    Parameters
    ----------
    spec : PanelSpec
        Сетка, порядок постов и каналов и единицы значений.
    times : tuple of pandas.Timestamp
        Все узлы объявленной сетки в её часовом поясе.
    values : array_like, shape (n_times, n_sites, n_channels)
        Ответы в единицах каналов. Вне ``target_mask`` требуется ``NaN``; внутри
        ``NaN`` обозначает неизвестный ответ.
    target_mask : array_like of bool, shape (n_times, n_sites, n_channels)
        Точки, выбранные для проверки.
    protocol_id, population_id : str
        Идентификаторы протокола и выборки.
    population_kind : {'train', 'calibration', 'selection', 'test', 'export'}
        Назначение выборки.

    Notes
    -----
    Массивы копируются без права записи. Эта запись передаётся оценке
    метрик отдельно от входов обучения и прогноза.
    """
    spec: PanelSpec
    times: tuple
    values: np.ndarray
    target_mask: np.ndarray
    protocol_id: str
    population_id: str
    population_kind: str

    def __post_init__(self):
        shape = _layout(self)
        for key in ("protocol_id", "population_id"):
            text(getattr(self, key), key)
        choice(self.population_kind, {"train", "calibration", "selection", "test", "export"}, "population_kind")
        mask = snapshot(self.target_mask, dtype=bool, shape=shape, name="target_mask")
        values = snapshot(self.values, dtype=float, shape=shape, name="truth")
        if np.any(np.isfinite(values) & ~mask) or np.any(np.isinf(values)):
            raise DataContractError("truth must contain selected targets or NaN; no values outside target_mask")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "target_mask", mask)


def conceal(panel, *, artificial_mask, mit_mask, protocol_id, population_id, population_kind):
    """Разделить панель на видимые входы и скрытые контрольные ответы.

    Parameters
    ----------
    panel : NetworkPanel
        Исходная панель с конечными наблюдениями.
    artificial_mask, mit_mask : array_like of bool, shape (n_times, n_sites, n_channels)
        Непересекающиеся маски исходных наблюдений с формой
        ``panel.values``. ``mit_mask`` допустима только при
        ``population_kind='train'``.
    protocol_id, population_id : str
        Идентификаторы протокола и выборки.
    population_kind : {'train', 'calibration', 'selection', 'test', 'export'}
        Назначение выборки.

    Returns
    -------
    inputs : MaskedInputs
        Входы с ``NaN`` вместо выбранных наблюдений; единицы и оси исходной
        панели сохраняются.
    truth : EvaluationTruth
        Исходные ответы на объединении масок скрытия и ``NaN`` вне него.
    """
    if not isinstance(panel, NetworkPanel):
        raise DataContractError("conceal requires NetworkPanel")
    shape = panel.values.shape
    artificial = snapshot(artificial_mask, dtype=bool, shape=shape, name="artificial_mask")
    mit = snapshot(mit_mask, dtype=bool, shape=shape, name="mit_mask")
    hidden = artificial | mit
    inputs = MaskedInputs(panel.spec, panel.times, np.where(hidden, np.nan, panel.values),
                          panel.row_present, panel.original_observed, artificial, mit,
                          protocol_id, population_id, population_kind)
    truth = EvaluationTruth(panel.spec, panel.times, np.where(hidden, panel.values, np.nan), hidden,
                            protocol_id, population_id, population_kind)
    return inputs, truth


@dataclass(frozen=True)
class ImputationResult:
    """Хранить прогноз заполнения и его привязку к выборке.

    Parameters
    ----------
    prediction : array_like, shape (n_times, n_sites, n_channels)
        Прогноз в единицах каналов; конечные значения могут быть
        отрицательными, ``NaN`` означает недоступный прогноз.
    availability : array_like of bool, shape (n_times, n_sites, n_channels)
        Маска конечных прогнозов.
    status : array_like of str, shape (n_times, n_sites, n_channels)
        ``'observed'``, ``'imputed'`` или ``'unavailable'``; последний статус
        соответствует недоступному прогнозу.
    uncertainty : array_like, shape (n_times, n_sites, n_channels), optional
        Стандартное отклонение в единицах соответствующего канала:
        неотрицательные числа или ``NaN``. ``None`` означает отсутствие оценки.
    spec : PanelSpec
        Сетка, порядок постов и каналов и единицы значений.
    times : tuple of pandas.Timestamp
        Все узлы объявленной сетки в её часовом поясе.
    protocol_id, population_id : str
        Идентификаторы протокола и выборки.
    population_kind : {'train', 'calibration', 'selection', 'test', 'export'}
        Назначение выборки.

    Notes
    -----
    Массивы копируются без права записи. Смысл стандартного отклонения
    задаёт метод, возвращающий результат; автоматической калибровки нет.
    Совпадение осей, времени, протокола и выборки проверяет ``require_binding``.
    """
    prediction: np.ndarray
    availability: np.ndarray
    status: np.ndarray
    uncertainty: np.ndarray | None = None
    spec: PanelSpec = field(kw_only=True)
    times: tuple = field(kw_only=True)
    protocol_id: str = field(kw_only=True)
    population_id: str = field(kw_only=True)
    population_kind: str = field(kw_only=True)

    def __post_init__(self):
        shape = _layout(self)
        for key in ("protocol_id", "population_id"):
            text(getattr(self, key), key)
        choice(self.population_kind, {"train", "calibration", "selection", "test", "export"}, "population_kind")
        pred = snapshot(self.prediction, dtype=float, shape=shape, name="prediction")
        available = snapshot(self.availability, dtype=bool, shape=shape, name="availability")
        status = snapshot(self.status, dtype=str, shape=shape, name="status")
        if (np.any(np.isinf(pred)) or not np.array_equal(np.isfinite(pred), available)
                or np.any(~np.isin(status, ["observed", "imputed", "unavailable"]))
                or not np.array_equal(status != "unavailable", available)):
            raise DataContractError("inconsistent prediction/availability/status")
        for key, value in (("prediction", pred), ("availability", available), ("status", status)):
            object.__setattr__(self, key, value)
        if self.uncertainty is not None:
            uncertainty = snapshot(self.uncertainty, dtype=float, shape=shape, name="uncertainty")
            if np.any(np.isinf(uncertainty)) or np.any(uncertainty < 0):
                raise DataContractError("uncertainty must be nonnegative or unavailable (NaN)")
            object.__setattr__(self, "uncertainty", uncertainty)

    def require_binding(self, reference):
        """Проверить принадлежность прогноза указанной выборке.

        Parameters
        ----------
        reference : MaskedInputs or EvaluationTruth
            Запись с той же панелью, временем, ``protocol_id``, ``population_id`` и
            ``population_kind``.

        Returns
        -------
        None
            При несовпадении привязки вызывается ``DataContractError``.
        """
        if (not isinstance(reference, (MaskedInputs, EvaluationTruth)) or
                any(getattr(self, key) != getattr(reference, key) for key in
                    ("spec", "times", "protocol_id", "population_id", "population_kind"))):
            raise DataContractError("prediction panel/population binding mismatch")


def result_from_inputs(inputs, prediction, *, availability=None, status=None, uncertainty=None):
    """Создать результат заполнения с метаданными входной панели.

    Parameters
    ----------
    inputs : MaskedInputs
        Видимые значения и привязка выборки.
    prediction : array_like, shape (n_times, n_sites, n_channels)
        Прогноз в единицах каналов с точным сохранением ``inputs.values`` в
        видимых позициях.
    availability : array_like of bool, optional
        Маска формы ``prediction``; по умолчанию выводится из конечности
        прогноза.
    status : array_like of str, optional
        Статусы формы ``prediction``; по умолчанию выводятся из видимости и
        доступности.
    uncertainty : array_like, optional
        Стандартное отклонение формы ``prediction`` в единицах каналов, либо
        ``None``. ``NaN`` оставляет оценку недоступной.

    Returns
    -------
    result : ImputationResult
        Независимая запись с осью времени, панелью и идентификаторами
        ``inputs``.

    Notes
    -----
    Статус ``observed`` должен совпадать с ``inputs.visible``. Ни шкала данных,
    ни знак допустимых конечных прогнозов не преобразуются.
    """
    if not isinstance(inputs, MaskedInputs):
        raise DataContractError("prediction requires MaskedInputs, never panel/truth")
    raw = np.asarray(prediction)
    if raw.shape != inputs.values.shape or raw.dtype.kind not in "buif":
        raise DataContractError("prediction requires real numeric values with the input panel shape")
    if availability is None:
        availability = np.isfinite(raw)
    if status is None:
        status = np.where(inputs.visible, "observed", np.where(availability, "imputed", "unavailable"))
    result = ImputationResult(prediction, availability, status, uncertainty,
        spec=inputs.spec, times=inputs.times, protocol_id=inputs.protocol_id,
        population_id=inputs.population_id, population_kind=inputs.population_kind)
    if (not np.array_equal(result.prediction[inputs.visible], inputs.values[inputs.visible]) or
            not np.array_equal(result.status == "observed", inputs.visible)):
        raise DataContractError("predict must retain original visible values and status")
    return result
