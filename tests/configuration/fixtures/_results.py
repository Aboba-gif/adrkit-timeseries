"""Запись результатов конфигурационных тестов."""
import json
from pathlib import Path

from adrkit.errors import ConfigError


def output_destination(config, output_root):
    root = output_root if output_root is not None else config.to_dict()["execution"]["output_root"]
    destination = None if root is None else Path(root)
    if destination is not None and (not destination.is_absolute() or destination.exists()):
        raise ConfigError("output_root must be an absolute, new directory")
    return destination


def write_report(config, report, destination):
    if destination is None:
        return
    payloads = {"result.json": report, "config-record.json": config.to_record().to_dict()}
    encoded = {name: json.dumps(value, indent=2, allow_nan=False) + "\n" for name, value in payloads.items()}
    destination.mkdir(parents=True, exist_ok=False)
    for name, content in encoded.items():
        (destination / name).write_text(content, encoding="utf-8")
