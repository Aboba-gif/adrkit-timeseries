"""Чтение, нормализация и объединение JSON-конфигураций."""

from .loader import (
    ConfigDocument, EffectiveConfig, ResourceBinding, ResourcePath, ScienceSchema,
    ValidationContext, compose_config, load_config, read_configs, resolve_locator,
    verify_resources,
)
from .validation import (
    JSONRecord, canonical_bytes, closed_object, digest, integer, number, sha256,
    strict_json, string,
)

__all__ = [
    "ConfigDocument", "EffectiveConfig", "ResourceBinding", "ResourcePath",
    "ScienceSchema", "ValidationContext", "compose_config", "load_config",
    "read_configs", "resolve_locator", "verify_resources", "JSONRecord", "canonical_bytes",
    "closed_object", "digest", "integer", "number", "sha256", "strict_json", "string",
]
