"""Сборка загрузчика данных и метода заполнения из параметров."""
from dataclasses import dataclass

from .contracts import DataContractError, closed, text
from .data import DataSource, build_data
from .imputation import build_imputer

__all__ = ["ConfiguredImputation", "assemble"]


@dataclass(frozen=True)
class ConfiguredImputation:
    """Объединить загрузчик, метод заполнения и идентификаторы выборки.

    Parameters
    ----------
    data : DataSource
        Описание отложенной загрузки данных.
    imputer : object
        Метод с операцией ``predict(inputs, fitted=...)``.
    protocol_id, population_id : str
        Идентификаторы протокола и выборки.
    """
    data: DataSource
    imputer: object
    protocol_id: str
    population_id: str

    def __post_init__(self):
        if not isinstance(self.data, DataSource):
            raise DataContractError("configured data must be a DataSource")
        if not callable(getattr(self.imputer, "predict", None)):
            raise DataContractError("configured imputer must provide predict")
        text(self.protocol_id, "protocol_id")
        text(self.population_id, "population_id")


def assemble(spec, *, families=None, anchor=None):
    """Собрать загрузчик и метод заполнения по конфигурации.

    Parameters
    ----------
    spec : dict
        Поля ``data``, ``imputer``, ``protocol_id`` и ``population_id``. ``data`` содержит
        ``sources``/``panel``, ``imputer`` — ``kind``/``parameters``.
    families : dict of str to ImputerFamily, optional
        Дополнительные семейства методов, передаваемые в ``build_imputer``.
    anchor : str or pathlib.Path, optional
        Абсолютная основа относительных путей источников.

    Returns
    -------
    configured : ConfiguredImputation
        Описание данных и созданный метод; маски и стадии обработки
        задаются отдельно.

    Notes
    -----
    Сборка не читает файлы, не обучает метод и не вычисляет прогноз.
    """
    closed(spec, {"data", "imputer", "protocol_id", "population_id"},
           where="configuration")
    protocol_id = text(spec["protocol_id"], "protocol_id")
    population_id = text(spec["population_id"], "population_id")
    data_spec = closed(spec["data"], {"sources", "panel"}, where="data")
    data = build_data(data_spec["sources"], panel_spec=data_spec["panel"], anchor=anchor)
    imputer = build_imputer(spec["imputer"], families=families)
    return ConfiguredImputation(data, imputer, protocol_id, population_id)
