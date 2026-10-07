"""Загрузка CSV-источников на регулярную сетку панели."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def localize_timestamps(ts: pd.Series, *, timezone: str | None, dst_policy: str) -> pd.Series:
    """Локализовать временные метки по объявленному правилу DST.

    Parameters
    ----------
    ts : pandas.Series
        Метки локального времени без часового пояса.
    timezone : str or None
        Часовой пояс pandas; ``None`` оставляет исходную Series.
    dst_policy : {'fail', 'shift', 'drop'}
        ``fail`` отклоняет неоднозначное/несуществующее время; ``shift``
        выбирает ``infer`` и ``shift_forward``; ``drop`` заменяет такое время на
        ``NaT``.

    Returns
    -------
    timestamps : pandas.Series
        Временные метки с объявленной зоной либо исходная Series при
        ``timezone=None``.

    Notes
    -----
    Правило ``infer`` требует порядка наблюдений, позволяющего различить
    неоднозначные метки; иначе локализация завершается ошибкой.
    """
    if timezone is None:
        return ts
    if dst_policy == "fail":
        return ts.dt.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
    if dst_policy == "shift":
        return ts.dt.tz_localize(timezone, ambiguous="infer", nonexistent="shift_forward")
    if dst_policy == "drop":
        return ts.dt.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT")
    raise ValueError(f"unknown dst_policy={dst_policy!r}")

@dataclass(frozen=True)
class DataSource:
    """Сохранить описание отложенной загрузки CSV.

    Parameters
    ----------
    specs : tuple of InputSourceSpec
        Источники в порядке чтения.
    panel_spec : PanelSpec
        Оси, сетка и единицы общей панели.
    anchor : pathlib.Path or None
        Абсолютная основа относительных путей источников.

    Notes
    -----
    Создавайте запись через ``build_data`` для проверки согласованности
    источников. Метод ``load`` читает файлы и возвращает ``LoadedSources``.
    """
    specs: tuple
    panel_spec: object
    anchor: Path | None

    def load(self):
        """Прочитать источники и собрать панель наблюдений.

        Returns
        -------
        loaded : LoadedSources
            Панель с исходными единицами, карантином и происхождением
            прочитанных байтов.
        """
        return load_sources(self.specs, panel_spec=self.panel_spec, anchor=self.anchor)


def build_data(specs, *, panel_spec, anchor=None) -> DataSource:
    """Собрать описание загрузки согласованных источников на панель.

    Parameters
    ----------
    specs : sequence of InputSourceSpec or dict
        Непустой упорядоченный список источников с различными
        идентификаторами.
    panel_spec : PanelSpec or dict
        Общая регулярная сетка, порядок постов/каналов и их единицы.
    anchor : str or pathlib.Path, optional
        Абсолютная основа относительных путей; необходима, если путь
        источника относительный.

    Returns
    -------
    source : DataSource
        Описание согласованных источников с отложенным чтением через
        ``load``.

    Notes
    -----
    Часы, календарь, зона, ``frequency``, ``origin`` и единицы каждого источника
    должны совпадать с панелью. Неявный перевод времени или чисел между
    единицами не выполняется. Файлы на этом этапе не читаются.
    """
    from .contracts import DataContractError, InputSourceSpec, PanelSpec

    if type(specs) not in (list, tuple) or not specs:
        raise DataContractError("sources must be a nonempty ordered list")
    sources = tuple(s if isinstance(s, InputSourceSpec) else InputSourceSpec(s) for s in specs)
    panel = panel_spec if isinstance(panel_spec, PanelSpec) else PanelSpec(panel_spec)
    data = [s.to_dict() for s in sources]
    layout = panel.to_dict()
    ids = [s["id"] for s in data]
    if len(ids) != len(set(ids)):
        raise DataContractError("duplicate source id")
    if anchor is not None:
        anchor = Path(anchor)
        if not anchor.is_absolute():
            raise DataContractError("anchor must be absolute")
    for source in data:
        if not Path(source["path"]).is_absolute() and anchor is None:
            raise DataContractError("relative source path requires explicit anchor")
        for key in ("clock", "calendar", "timezone", "frequency", "origin"):
            if source["time"][key] != layout[key]:
                raise DataContractError(f"source/panel {key} mismatch; no implicit conversion")
        if source["site"]["kind"] == "fixed" and source["site"]["id"] not in layout["sites"]:
            raise DataContractError("source site not in panel")
        for channel, unit in source["units"].items():
            if channel not in layout["channels"] or unit != layout["units"][channel]:
                raise DataContractError("source/panel channel or units mismatch")
        if layout["duplicates"] == "first_with_quarantine" and source["route"] != "quarantine":
            raise DataContractError("first_with_quarantine requires explicit quarantine route for every source")
    return DataSource(sources, panel, anchor)


def load_sources(specs, *, panel_spec, anchor=None):
    """Загрузить CSV-источники на объявленную сетку панели.

    Parameters
    ----------
    specs : sequence of InputSourceSpec or dict
        Непустой упорядоченный список источников с различными
        идентификаторами.
    panel_spec : PanelSpec or dict
        Общая регулярная сетка, порядок постов/каналов и их единицы.
    anchor : str or pathlib.Path, optional
        Абсолютная основа относительных путей; необходима, если путь
        источника относительный.

    Returns
    -------
    loaded : LoadedSources
        Панель формы (n_times, n_sites, n_channels) в исходных единицах,
        исключённые значения и сведения об источниках.

    Notes
    -----
    Проверяется SHA-256 прочитанных байтов и структура CSV. Дубли
    времени, поста и канала проверяются до попадания времени на сетку.
    Маршрут ``strict`` отклоняет неподходящее время; ``quarantine`` сохраняет
    исходный адрес и причину исключения до разбора числа. У принятых
    значений нечисловой текст допустим только как объявленное обозначение
    пропуска.
    Порядок постов и каналов задаёт ``PanelSpec``.
    """
    import csv
    import hashlib
    import io
    from math import isfinite

    from .contracts import (
        AcceptedValue, DataContractError, DuplicateObservationError, LoadedSources,
        MissingInputError, NetworkPanel, QuarantinedValue, RowLocator,
        SourceIntegrityError, SourceProvenance, clock_timestamp,
    )

    assembly = build_data(specs, panel_spec=panel_spec, anchor=anchor)
    layout = assembly.panel_spec.to_dict()
    start, stop = (clock_timestamp(layout[k], layout) for k in ("start", "stop"))
    step = pd.Timedelta(layout["frequency"])
    origin = clock_timestamp(layout["origin"], layout)
    grid = pd.date_range(start, stop, freq=step, inclusive="left")
    site_index = {site: i for i, site in enumerate(layout["sites"])}
    channel_index = {channel: i for i, channel in enumerate(layout["channels"])}
    shape = (len(grid), len(site_index), len(channel_index))
    values = np.full(shape, np.nan)
    present = np.zeros(shape, dtype=bool)
    seen, resolved, quarantine, provenance, accepted = {}, set(), [], [], []
    sites_seen, channels_seen = set(), set()
    paths = []
    for record in assembly.specs:
        source = record.to_dict()
        path = Path(source["path"])
        if not path.is_absolute():
            path = assembly.anchor / path
        path = path.resolve()
        if path in resolved:
            raise DataContractError(f"same resolved file under multiple source IDs: {path}")
        resolved.add(path)
        paths.append(path)

    for record, path in zip(assembly.specs, paths):
        source = record.to_dict()
        try:
            content = path.read_bytes()
        except (FileNotFoundError, IsADirectoryError) as exc:
            raise MissingInputError(f"missing input file: {path}") from exc
        actual_hash = hashlib.sha256(content).hexdigest()
        if actual_hash != source["sha256"]:
            raise SourceIntegrityError(f"{source['id']}: source SHA256 mismatch")
        fmt = source["format"]["parameters"]
        try:
            reader = csv.reader(io.StringIO(content.decode(fmt["encoding"]), newline=""),
                                delimiter=fmt["separator"], strict=True)
            headers = next(reader, [])
            rows = list(reader)
        except (UnicodeError, LookupError, csv.Error) as exc:
            raise DataContractError(f"{source['id']}: invalid CSV encoding/structure") from exc
        if not headers or len(headers) != len(set(headers)):
            raise DataContractError(f"{source['id']}: empty/duplicate CSV headers")
        columns, site_spec = source["columns"], source["site"]
        required = set(columns["time"]) | set(columns["values"].values())
        if site_spec["kind"] == "column":
            required.add(site_spec["column"])
        if not required <= set(headers):
            raise MissingInputError(f"{source['id']}: missing columns {sorted(required-set(headers))}")
        if any(len(row) != len(headers) for row in rows):
            raise DataContractError(f"{source['id']}: ragged CSV record")
        raw_rows = [dict(zip(headers, row)) for row in rows]
        raw_times = [" ".join(row[col] for col in columns["time"]) for row in raw_rows]
        time = source["time"]
        try:
            stamps = pd.Series(pd.to_datetime(raw_times, errors="coerce",
                               format=time["format"] if time["parser"] == "format" else None))
            if stamps.dt.tz is not None:
                raise DataContractError("source timestamps must be wall time, not offset/UTC timestamps")
            
            if time["timezone"] is not None:
                stamps = localize_timestamps(stamps, timezone=time["timezone"],
                                             dst_policy=time["dst_policy"])
        except Exception as exc:
            raise DataContractError(f"{source['id']}: incompatible timestamps/DST") from exc
        provenance.append(SourceProvenance(record, str(path), len(content)))
        # После проверки дублей strict вызывает ошибку времени, quarantine исключает значение.
        
        grid_indices = grid.get_indexer(pd.DatetimeIndex(stamps))
        for row_number, (row, raw_time, stamp, grid_index) in enumerate(
                zip(raw_rows, raw_times, stamps, grid_indices), 2):
            site = site_spec["id"] if site_spec["kind"] == "fixed" else row[site_spec["column"]]
            if site not in site_index:
                raise DataContractError(f"{source['id']}:{row_number}: undeclared site {site!r}")
            missing_stamp = pd.isna(stamp)
            time_checked, time_reason, time_index = False, None, None
            
            for channel, header in columns["values"].items():
                locator = RowLocator(source["id"], actual_hash, row_number, site, channel, raw_time, row[header])
                reason = None
                if missing_stamp:
                    reason = "unparseable_or_dst_dropped"
                else:
                    key = (stamp, site, channel)
                    if key in seen:
                        if layout["duplicates"] == "reject":
                            raise DuplicateObservationError(f"{locator}: overlaps {seen[key]}")
                        reason = "duplicate_timestamp"
                    else:
                        seen[key] = locator
                        # Время общее для всех каналов записи, но дубли проверяются отдельно по каналу.
                        
                        if not time_checked:
                            if grid_index >= 0:
                                time_index = int(grid_index)
                            elif (stamp - origin) % step:
                                time_reason = "off_grid"
                            elif stamp < start or stamp >= stop:
                                time_reason = "outside_panel"
                            time_checked = True
                        reason = time_reason
                if reason:
                    if source["route"] == "strict":
                        raise DataContractError(f"{locator}: {reason}")
                    quarantine.append(QuarantinedValue(locator, reason))
                    continue
                if row[header] in source["missing_values"]:
                    value = np.nan
                else:
                    try:
                        value = float(row[header])
                    except ValueError as exc:
                        raise DataContractError(f"{locator}: undeclared missing token or nonnumeric value") from exc
                    if not isfinite(value):
                        raise DataContractError(f"{locator}: nonfinite value is not a declared missing token")
                index = (time_index, site_index[site], channel_index[channel])
                values[index], present[index] = value, True
                accepted.append(AcceptedValue(locator, index))
                sites_seen.add(site)
                channels_seen.add(channel)
    if sites_seen != set(site_index) or channels_seen != set(channel_index):
        raise MissingInputError("one or more required panel sites/channels have no accepted input rows")
    observed = np.isfinite(values)
    panel = NetworkPanel(assembly.panel_spec, tuple(grid), values, present, observed,
                         present & ~observed, tuple(accepted))
    return LoadedSources(panel, tuple(quarantine), tuple(provenance))
