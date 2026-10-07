"""Протоколы прогнозов, производных и скалярных функционалов."""

from .capabilities import (
    JVP, VJP, MapCapabilities, Predict, Prox, ScalarCapabilities,
    ScalarCotangent, ScalarValue, ScalarValueAndCotangent, TruthDerivative, TruthSource, require_capability,
)

__all__ = [
    "Predict", "JVP", "VJP", "ScalarValue", "ScalarCotangent", "ScalarValueAndCotangent", "Prox",
    "TruthSource", "TruthDerivative", "MapCapabilities", "ScalarCapabilities",
    "require_capability",
]
