"""Ссылки на входные файлы используют явно выбранное расположение конфигурации."""
from pathlib import Path

import pytest

from timeseries.contracts import DataContractError, source_resource_pointers


def test_source_pointers_support_another_configuration_location(csv_case, monkeypatch):
    specs, _ = csv_case(3)

    def forbidden(*args, **kwargs):
        pytest.fail("constructing source pointers must not read sources")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    prefix = "/datasets/sensor~1array~0raw"
    actual = source_resource_pointers(specs, prefix=prefix)
    assert actual == tuple((f"{prefix}/{i}/path", f"{prefix}/{i}/sha256")
                           for i in range(len(specs)))


@pytest.mark.parametrize("prefix", [None, True, 12, "", "sources", "/source~", "/source~2", "/sources/*"])
def test_invalid_source_pointer_prefix_precedes_source_validation(prefix, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid prefix must fail before reading a source")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    with pytest.raises(DataContractError, match="prefix"):
        source_resource_pointers([object()], prefix=prefix)


def test_source_pointer_prefix_is_required():
    with pytest.raises(TypeError, match="prefix"):
        source_resource_pointers([])
