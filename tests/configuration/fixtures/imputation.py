"""Тестовая JSON-конфигурация интерполяции linear/pchip без обучения."""
import numpy as np

from adrkit.config import JSONRecord, ResourcePath, ScienceSchema, ValidationContext, compose_config, read_configs, string, verify_resources
from adrkit.config.schema import fields
from adrkit.errors import ConfigError
from timeseries.config import assemble as assemble_components
from tests.configuration.fixtures._results import output_destination, write_report


def imputation_schema():
    # Зависимости данных загружаются только для выбранного порядка обработки.
    from timeseries.contracts import InputSourceSpec, PanelSpec, source_resource_pointers
    from timeseries.imputation import validate_imputer_spec

    def normalize(science, context):
        def partial(path):
            return context.partial and not any(path == p or path.startswith(p + "/") for p in context.complete_paths)

        fields(science, {"data", "imputer", "protocol_id", "population_id"}, partial=partial("/science"))
        s = JSONRecord(science).to_dict()
        for key in ("protocol_id", "population_id"):
            if key in s:
                string(s[key], where=key)
        if "data" in s:
            data = fields(s["data"], {"sources", "panel"}, partial=partial("/science/data"))
            if "sources" in data:
                source_resource_pointers(data["sources"], prefix="/science/data/sources")  # Массивы заменяются целиком.
                data["sources"] = [InputSourceSpec(v).to_dict() for v in data["sources"]]
            if "panel" in data:
                # Дополнения панели задают полные записи: последующее дополнение не должно скрывать
                # некорректное промежуточное описание времени или единиц.
                original = data["panel"]
                normalized = PanelSpec(original).to_dict()
                data["panel"] = {k: normalized[k] for k in original}
        if "imputer" in s:
            spec = fields(s["imputer"], {"kind", "parameters"}, partial=partial("/science/imputer"))
            if "kind" in spec and spec["kind"] not in ("linear", "pchip"):
                raise ConfigError("this caller supports no-fit linear/pchip imputer kinds only")
            if "parameters" in spec:
                if "kind" not in spec:
                    raise ConfigError("imputer parameters require kind")
                params = fields(spec["parameters"], {"edge_policy"}, partial=partial("/science/imputer/parameters"))
                if "edge_policy" in params:
                    validate_imputer_spec(spec)
        return s

    def resources(s):
        specs = s.get("data", {}).get("sources")
        return () if specs is None else tuple(ResourcePath(*p) for p in source_resource_pointers(
            specs, prefix="/science/data/sources"))

    return ScienceSchema("imputation", "adrkit.imputation/1", normalize, resources)


def load(files, *, inputs=None, overrides=(), config_root=None, output_root=None):
    if inputs:
        raise ConfigError("composition families do not apply to imputation")
    documents = read_configs(files, config_root=config_root)
    return compose_config(documents, schema=imputation_schema(), overrides=overrides,
                          config_root=config_root, output_root=output_root)


def assemble(config, *, inputs=None):
    if inputs:
        raise ConfigError("composition families do not apply to imputation")
    tree = config.to_dict()
    science = imputation_schema().normalize(tree["science"], ValidationContext(partial=False))
    return assemble_components(science)


def predict_export(assembled):
    from timeseries.contracts import conceal
    data, imputer = assembled.data, assembled.imputer
    loaded = data.load()  # При загрузке проверяется контрольная сумма прочитанных байтов.
    panel = loaded.panel
    empty = np.zeros(panel.values.shape, dtype=bool)
    inputs, _truth = conceal(panel, artificial_mask=empty, mit_mask=empty,
        protocol_id=assembled.protocol_id, population_id=assembled.population_id, population_kind="export")
    result = imputer.predict(inputs, fitted=imputer.no_fit_state)
    result.require_binding(inputs)
    return {"prediction": np.where(result.availability, result.prediction, None).tolist(),
            "availability": result.availability.tolist(), "status": result.status.tolist(),
            "quarantined_values": len(loaded.quarantine), "fit": False}


def run(config, *, mode, inputs=None, output_root=None):
    """Проверить сборку либо интерполировать заданную экспортную панель."""
    if mode not in ("validate", "smoke", "run"):
        raise ConfigError("supported engineering modes: validate, smoke, run")
    assembled = assemble(config, inputs=inputs)
    report = {"effective_config": config.to_dict(),
              "effective_science_sha256": config.effective_science_sha256}
    if mode == "validate":
        report["validated"] = True
        return report
    destination = output_destination(config, output_root)
    verify_resources(config)
    report["result"] = predict_export(assembled)
    write_report(config, report, destination)
    return report


