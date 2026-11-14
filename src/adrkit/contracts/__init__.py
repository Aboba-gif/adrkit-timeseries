"""Протоколы прогнозов, производных и скалярных функционалов."""

from .capabilities import (
    JVP,
    Jacobian,
    VJP,
    MapCapabilities,
    Predict,
    Prox,
    ScalarCapabilities,
    ScalarCotangent,
    ScalarValue,
    ScalarValueAndCotangent,
    TruthDerivative,
    TruthSource,
    require_capability,
    validate_map_capabilities,
    validate_scalar_capabilities,
)

__all__ = [
    "Predict",
    "JVP",
    "Jacobian",
    "VJP",
    "ScalarValue",
    "ScalarCotangent",
    "ScalarValueAndCotangent",
    "Prox",
    "TruthSource",
    "TruthDerivative",
    "MapCapabilities",
    "ScalarCapabilities",
    "require_capability",
    "validate_map_capabilities",
    "validate_scalar_capabilities",
]
