"""Научные входы и каталог результатов в общей JSON-конфигурации."""
import hashlib
import json

import pytest

from adrkit.config import (
    ResourcePath, ScienceSchema, closed_object, load_config, number,
    verify_resources,
)
from adrkit.errors import ConfigError


def schema():
    def normalize(science, context):
        fields = {"gain", "data"}
        closed_object(science, fields, set() if context.partial else {"gain"})
        result = dict(science)
        if "gain" in result:
            result["gain"] = number(result["gain"])
        if "data" in result:
            closed_object(result["data"], {"path", "sha256"},
                          set() if context.partial else {"path", "sha256"})
        return result

    def resources(science):
        if "path" in science.get("data", {}):
            return (ResourcePath("/science/data/path", "/science/data/sha256"),)
        return ()

    return ScienceSchema("test", "test/1", normalize, resources)


def write_config(path, *, science=None, execution=None, document_role=None):
    value = {
        "schema_version": "adrkit.experiment/1",
        "pipeline": "test",
        "science_schema_version": "test/1",
        "science": {"gain": 2} if science is None else science,
    }
    if execution is not None:
        value["execution"] = execution
    if document_role is not None:
        value["document_role"] = document_role
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_output_destination_keeps_scientific_identity(tmp_path):
    path = write_config(tmp_path / "config.json")
    direct = load_config([path], schema=schema())
    destination = tmp_path / "results"
    redirected = load_config([path], schema=schema(), output_root=destination)
    assert direct.to_dict()["execution"] == {"output_root": None}
    assert redirected.to_dict()["execution"] == {"output_root": str(destination)}
    assert direct.science_identity.to_dict() == redirected.science_identity.to_dict()
    assert direct.effective_science_sha256 == redirected.effective_science_sha256
    assert redirected.provenance.to_dict()["overrides"][0]["pointer"] == "/execution/output_root"
    assert not destination.exists()


def test_input_and_output_paths_keep_their_file_origins(tmp_path, monkeypatch):
    base = tmp_path / "base"
    overlay = tmp_path / "overlay"
    base.mkdir()
    overlay.mkdir()
    raw = b"time,value\n0,1\n1,2\n"
    source = base / "data.csv"
    source.write_bytes(raw)
    path = write_config(base / "config.json", science={
        "gain": 2, "data": {"path": "data.csv", "sha256": hashlib.sha256(raw).hexdigest()},
    })
    addition = overlay / "output.json"
    addition.write_text(json.dumps({
        "schema_version": "adrkit.experiment/1", "document_role": "config",
        "execution": {"output_root": "results"},
    }), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    config = load_config([path, addition], schema=schema())
    assert config.resources[0].locator == str(source)
    assert config.to_dict()["execution"] == {"output_root": str(overlay / "results")}
    assert verify_resources(config) == config.resources
    assert [d["document_role"] for d in config.provenance.to_dict()["inputs"]] == ["config", "config"]
    source.write_bytes(b"changed source")
    with pytest.raises(ConfigError, match="resource content mismatch"):
        verify_resources(config)


def test_output_path_cannot_replace_a_scientific_input(tmp_path):
    source = tmp_path / "data.csv"
    source.write_bytes(b"0,1\n")
    path = write_config(tmp_path / "config.json", science={
        "gain": 2, "data": {
            "path": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        },
    })
    with pytest.raises(ConfigError, match="collides with an input locator"):
        load_config([path], schema=schema(), output_root=source)


@pytest.mark.parametrize("field,value", [
    ("schema_version", "adrkit.execution/1"), ("profile", "local"),
    ("workers", 1), ("threads", 1), ("limits", None), ("input_root", None),
    ("replay_source", None), ("resume", None), ("environment_receipt", None),
    ("implementation_receipt", None), ("budget_reference", None),
])
def test_unknown_execution_fields_are_rejected(tmp_path, field, value):
    path = write_config(tmp_path / "config.json", execution={field: value})
    with pytest.raises(ConfigError, match="execution: unknown"):
        load_config([path], schema=schema())


@pytest.mark.parametrize("value", [False, 3, "", "*.json", "file?.json", "bad\x00name"])
def test_invalid_output_destination_is_rejected(tmp_path, value):
    path = write_config(tmp_path / "config.json", execution={"output_root": value})
    with pytest.raises(ConfigError):
        load_config([path], schema=schema())


def test_execution_profile_is_not_a_config_document(tmp_path):
    path = write_config(tmp_path / "config.json", document_role="execution_profile")
    with pytest.raises(ConfigError, match="document_role"):
        load_config([path], schema=schema())


def test_unknown_execution_prefix_cannot_be_repaired(tmp_path):
    path = write_config(tmp_path / "config.json", execution={"workers": 1})
    overlay = tmp_path / "overlay.json"
    overlay.write_text(json.dumps({"schema_version": "adrkit.experiment/1",
                                   "execution": {"output_root": None}}), encoding="utf-8")
    with pytest.raises(ConfigError, match="execution: unknown"):
        load_config([path, overlay], schema=schema())


def test_complete_output_override_and_conflict(tmp_path):
    path = write_config(tmp_path / "config.json")
    config = load_config([path], schema=schema(), overrides=[
        '/execution={"output_root":"results"}',
    ], config_root=tmp_path)
    assert config.to_dict()["execution"] == {"output_root": str(tmp_path / "results")}
    with pytest.raises(ConfigError, match="conflicts"):
        load_config([path], schema=schema(), overrides=[
            '/execution={"output_root":null}',
        ], output_root=tmp_path / "another")
    with pytest.raises(ConfigError, match="missing"):
        load_config([path], schema=schema(), overrides=['/execution={}'])


@pytest.mark.parametrize("literal", ["$SOURCE", "~", "[ab]", "closing]"])
def test_literal_paths_keep_file_overlay_and_cli_anchors(tmp_path, monkeypatch, literal):
    anchor = tmp_path / "anchor"
    base = anchor / literal
    source = base / literal / "data.csv"
    source.parent.mkdir(parents=True)
    raw = b"time,value\n0,1\n1,2\n"
    source.write_bytes(raw)
    relative_source = f"{literal}/data.csv"
    config_path = write_config(base / "config.json", science={
        "gain": 2, "data": {
            "path": relative_source, "sha256": hashlib.sha256(raw).hexdigest(),
        },
    })
    expanded = anchor / "expanded"
    expanded.mkdir()
    write_config(expanded / "config.json", science={"gain": 99})
    (expanded / "data.csv").write_bytes(b"different data")
    monkeypatch.setenv("SOURCE", str(expanded))
    monkeypatch.setenv("USERPROFILE", str(expanded))
    for parent in (anchor, base):
        decoy = parent / "a"
        decoy.mkdir()
        write_config(decoy / "config.json", science={"gain": 99})
        (decoy / "data.csv").write_bytes(b"different data")
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)

    direct = load_config([config_path], schema=schema())
    anchored = load_config([f"{literal}/config.json"], schema=schema(), config_root=anchor)
    assert direct.to_dict() == anchored.to_dict()
    assert anchored.to_dict()["science"]["gain"] == 2
    assert anchored.resources[0].locator == str(source)
    assert verify_resources(anchored) == anchored.resources
    assert anchored.provenance.to_dict()["paths"]["/science/data/path"] == {
        "original": relative_source, "anchor": str(base), "resolved": str(source),
    }

    overlay = tmp_path / "overlay"
    overlay.mkdir()
    addition = overlay / "output.json"
    addition.write_text(json.dumps({
        "schema_version": "adrkit.experiment/1",
        "execution": {"output_root": f"{literal}/results"},
    }), encoding="utf-8")
    overlaid = load_config([config_path, addition], schema=schema())
    assert overlaid.to_dict()["execution"]["output_root"] == str(overlay / literal / "results")
    assert not (overlay / literal / "results").exists()

    cli = load_config([config_path], schema=schema(), config_root=base,
        overrides=['/science/data/path=' + json.dumps(relative_source)],
        output_root=f"{literal}/cli-results")
    assert cli.resources[0].locator == str(source)
    assert verify_resources(cli) == cli.resources
    assert cli.provenance.to_dict()["origins"]["/science/data/path"]["kind"] == "cli"
    assert cli.to_dict()["execution"]["output_root"] == str(base / literal / "cli-results")
    assert not (base / literal / "cli-results").exists()
    assert direct.effective_science_sha256 == anchored.effective_science_sha256
    assert direct.effective_science_sha256 == overlaid.effective_science_sha256 == cli.effective_science_sha256

