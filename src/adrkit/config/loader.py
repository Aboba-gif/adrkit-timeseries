"""Чтение и объединение JSON-конфигураций с происхождением значений."""

from __future__ import annotations

import hashlib
import platform
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from adrkit.errors import ConfigError
from .validation import (
    JSONRecord, closed_object, digest, sha256, strict_json, string,
)

SCHEMA_VERSION = "adrkit.experiment/1"


@dataclass(frozen=True)
class ValidationContext:
    """Режим нормализации части конфигурации.

    Parameters
    ----------
    partial : bool
        Допустить отсутствие обязательных полей вне complete_paths.
    complete_paths : tuple of str, optional
        JSON Pointer объектов, которые должны проверяться как полные.
    """
    partial: bool
    complete_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResourcePath:
    """Пути ресурса и ожидаемой контрольной суммы в JSON.

    Parameters
    ----------
    pointer : str
        JSON Pointer имени файла ресурса.
    sha256_pointer : str
        JSON Pointer ожидаемой SHA-256 его содержимого.
    """

    pointer: str
    sha256_pointer: str


@dataclass(frozen=True)
class ScienceSchema:
    """Схема параметров пользовательской научной постановки.

    Parameters
    ----------
    pipeline, version : str
        Непустые идентификаторы постановки и версии схемы.
    normalize : callable
        Принимает словарь и ValidationContext; возвращает
        нормализованный словарь.
    resources : callable
        Принимает параметры; возвращает кортеж ResourcePath для
        объявленных ресурсов.
    defaults : JSONRecord, optional
        Значения по умолчанию, применяемые к отсутствующим ключам.
    """

    pipeline: str
    version: str
    normalize: Callable[[dict, ValidationContext], dict]
    resources: Callable[[dict], tuple[ResourcePath, ...]]
    defaults: JSONRecord = field(default_factory=lambda: JSONRecord({}))

    def __post_init__(self) -> None:
        string(self.pipeline, where="pipeline")
        string(self.version, where="science_schema_version")
        if not callable(self.normalize) or not callable(self.resources):
            raise ConfigError("science schema requires explicit Python validators")
        if not isinstance(self.defaults, JSONRecord):
            raise ConfigError("schema defaults require JSONRecord")
        self.defaults.to_dict()


@dataclass(frozen=True)
class ConfigDocument:
    """Прочитанный JSON-документ конфигурации.

    Parameters
    ----------
    ordinal : int
        Индекс документа в порядке чтения, начиная с нуля.
    locator : str
        Абсолютный путь прочитанного файла.
    sha256 : str
        SHA-256 исходных байтов файла.
    document_role : str
        Роль документа; загрузчик поддерживает config.
    tree : JSONRecord
        Сохранённое содержимое JSON.
    original : str
        Исходное имя файла, переданное загрузчику.
    anchor : str or None
        Корень разрешения относительного исходного имени.
    """
    ordinal: int
    locator: str
    sha256: str
    document_role: str
    tree: JSONRecord
    original: str
    anchor: str | None


@dataclass(frozen=True)
class ResourceBinding:
    """Файл ресурса с ожидаемой контрольной суммой.

    Parameters
    ----------
    pointer : str
        JSON Pointer ресурса в конфигурации.
    locator : str
        Абсолютный разрешённый путь файла.
    sha256 : str
        Ожидаемая SHA-256 содержимого.
    """
    pointer: str
    locator: str
    sha256: str


@dataclass(frozen=True)
class EffectiveConfig:
    """Объединённая конфигурация с происхождением параметров.

    Parameters
    ----------
    config : JSONRecord
        Действующая конфигурация с разрешёнными путями.
    science_identity : JSONRecord
        Научные параметры без параметров выполнения; пути ресурсов
        заменены ожидаемыми контрольными суммами.
    provenance : JSONRecord
        Происхождение значений, входные документы и
        последовательность замен.
    resources : tuple of ResourceBinding
        Объявленные внешние файлы; проверка содержимого выполняется
        отдельно.
    """

    config: JSONRecord
    science_identity: JSONRecord
    provenance: JSONRecord
    resources: tuple[ResourceBinding, ...]

    @property
    def effective_science_sha256(self) -> str:
        """SHA-256 нормализованной записи science_identity."""
        return self.science_identity.sha256

    def to_dict(self) -> dict:
        """Получить действующую конфигурацию словарём.

        Returns
        -------
        config : dict
            Независимые JSON-контейнеры действующей конфигурации.
        """
        return self.config.to_dict()

    def to_record(self) -> JSONRecord:
        """Получить конфигурацию вместе с её происхождением.

        Returns
        -------
        record : JSONRecord
            Действующие параметры, их научная SHA-256 и provenance.
        """
        return JSONRecord({"effective_config": self.to_dict(),
                           "effective_science_sha256": self.effective_science_sha256,
                           "provenance": self.provenance.to_dict()})


def _tokens(pointer: str) -> list[str]:
    if type(pointer) is not str or not pointer.startswith("/"):
        raise ConfigError("expected a non-root JSON Pointer beginning with /")
    if re.search(r"~(?![01])", pointer) or "*" in pointer:
        raise ConfigError(f"invalid JSON Pointer: {pointer!r}")
    return [part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")]


def _pointer(parent: str, key: str | int) -> str:
    return parent + "/" + str(key).replace("~", "~0").replace("/", "~1")


def _get(tree: dict, pointer: str):
    value = tree
    for token in _tokens(pointer):
        try:
            if type(value) is list:
                if not re.fullmatch(r"0|[1-9][0-9]*", token):
                    raise ConfigError(f"invalid array index in {pointer}")
                value = value[int(token)]
            else:
                value = value[token]
        except (KeyError, TypeError, IndexError) as exc:
            raise ConfigError(f"unknown path: {pointer}") from exc
    return value


def _set(tree: dict, pointer: str, value, *, arrays: bool = False) -> None:
    tokens = _tokens(pointer)
    parent = tree
    for token in tokens[:-1]:
        try:
            if arrays and type(parent) is list:
                parent = parent[int(token)]
            elif type(parent) is dict:
                parent = parent[token]
            else:
                raise ConfigError("array-index patches are forbidden")
        except (KeyError, TypeError, IndexError, ValueError) as exc:
            raise ConfigError(f"missing/invalid parent: {pointer}") from exc
    if type(parent) is dict:
        parent[tokens[-1]] = value
    elif arrays and type(parent) is list:
        parent[int(tokens[-1])] = value
    else:
        raise ConfigError("array-index patches are forbidden")


def resolve_locator(value: str | Path, *, anchor: str | Path | None = None) -> str:
    """Разрешить имя файла в абсолютный путь.

    Parameters
    ----------
    value : str or pathlib.Path
        Непустое абсолютное или относительное имя без звёздочки,
        вопросительного и нулевого символов.
    anchor : str, pathlib.Path or None, optional
        Абсолютный корень для относительного имени.

    Returns
    -------
    locator : str
        Абсолютный путь; существование файла не требуется.

    Notes
    -----
    Символы $, ~, [ и ] читаются буквально; подстановка окружения
    и раскрытие шаблонов не выполняются.
    """
    raw = str(value) if isinstance(value, Path) else value
    string(raw, where="path")
    if any(c in raw for c in ("*", "?", "\x00")):
        raise ConfigError(f"path contains a forbidden character: {raw!r}")
    path = Path(raw)
    if not path.is_absolute():
        if anchor is None or not Path(anchor).is_absolute():
            raise ConfigError("relative path requires an explicit absolute anchor")
        path = Path(anchor) / path
    try:
        return str(path.resolve(strict=False))
    except (OSError, RuntimeError, ValueError) as exc:
        raise ConfigError(f"cannot resolve locator: {raw!r}") from exc


def _root(config_root: str | Path | None) -> str | None:
    if config_root is None:
        return None
    if not isinstance(config_root, (str, Path)):
        raise ConfigError("config_root must be an absolute path")
    if not Path(config_root).is_absolute():
        raise ConfigError("config_root must be absolute")
    return resolve_locator(config_root)


def read_configs(files, *, config_root: str | Path | None = None) -> tuple[ConfigDocument, ...]:
    """Прочитать JSON-конфигурации в заданном порядке.

    Parameters
    ----------
    files : sequence of str or pathlib.Path
        Непустая последовательность имён файлов; строка не заменяет
        последовательность.
    config_root : str, pathlib.Path or None, optional
        Абсолютный корень для относительных имён.

    Returns
    -------
    documents : tuple of ConfigDocument
        Документы с SHA-256 исходных байтов в порядке files.
    """
    root = _root(config_root)
    if isinstance(files, (str, bytes, Path)):
        raise ConfigError("files must be an ordered sequence of locators")
    documents = []
    for ordinal, raw in enumerate(files):
        locator = resolve_locator(raw, anchor=root)
        try:
            data = Path(locator).read_bytes()
        except OSError as exc:
            raise ConfigError(f"cannot read config: {locator}") from exc
        tree = strict_json(data)
        _envelope(tree, first=ordinal == 0)
        documents.append(ConfigDocument(ordinal, locator, hashlib.sha256(data).hexdigest(),
                                        tree.get("document_role", "config"), JSONRecord(tree),
                                        str(raw), root))
    if not documents:
        raise ConfigError("at least one config is required")
    return tuple(documents)


def _envelope(tree: dict, *, first: bool) -> None:
    fields = {"schema_version", "pipeline", "science_schema_version", "science",
              "execution", "document_role"}
    required = {"schema_version"}
    if first:
        required |= {"pipeline", "science_schema_version"}
    closed_object(tree, fields, required, where="envelope")
    if tree["schema_version"] != SCHEMA_VERSION:
        raise ConfigError("unsupported schema_version")
    role = tree.get("document_role", "config")
    if role != "config":
        raise ConfigError("unsupported document_role")
    for name in ("pipeline", "science_schema_version"):
        if name in tree:
            string(tree[name], where=name)
    for name in ("science", "execution"):
        if name in tree and type(tree[name]) is not dict:
            raise ConfigError(f"{name}: expected object")


def _execution(tree: dict, *, partial: bool) -> dict:
    fields = {"output_root"}
    closed_object(tree, fields, set() if partial else fields, where="execution")
    if tree.get("output_root") is not None:
        string(tree["output_root"], where="output_root")
    return dict(tree)


_EXECUTION_DEFAULTS = {"output_root": None}


def _leaves(value, pointer=""):
    if type(value) in (dict, list) and value:
        items = value.items() if type(value) is dict else enumerate(value)
        for key, item in items:
            yield from _leaves(item, _pointer(pointer, key))
    else:
        yield pointer


def compose_config(documents, *, schema: ScienceSchema, overrides=(), config_root=None,
                   output_root=None) -> EffectiveConfig:
    """Объединить конфигурационные документы по заданной схеме.

    Parameters
    ----------
    documents : sequence of ConfigDocument
        Непустая последовательность с исходными упорядоченными
        ordinal.
    schema : ScienceSchema
        Явная схема научных параметров и ресурсов.
    overrides : sequence of str, optional
        Замены полей в формате /pointer=JSON; порядок сохраняется.
    config_root : str, pathlib.Path or None, optional
        Абсолютный корень относительных путей из overrides.
    output_root : str, pathlib.Path or None, optional
        Переопределённый каталог результата; относительный путь
        разрешается через config_root.

    Returns
    -------
    config : EffectiveConfig
        Действующие параметры, происхождение и объявленные ресурсы.

    Notes
    -----
    Поздние значения заменяют ранние; словари объединяются, списки
    заменяются целиком. При смене kind нужен полный объект. Значения
    по умолчанию применяются к отсутствующим ключам. Явный null
    сохраняется. Содержимое ресурсов проверяет verify_resources.
    """
    root = _root(config_root)
    documents = tuple(documents)
    if not documents:
        raise ConfigError("at least one config is required")
    if not isinstance(schema, ScienceSchema):
        raise ConfigError("an explicit ScienceSchema is required")
    # Константы проверяются и при последующем замещении входными параметрами.
    declared_defaults = schema.normalize(schema.defaults.to_dict(), ValidationContext(True))
    if type(declared_defaults) is not dict:
        raise ConfigError("science defaults validator must return an object")
    declared_defaults = JSONRecord(declared_defaults).to_dict()
    origins, history, cli = {}, [], []
    tree = {}

    def assign_origin(value, pointer, origin):
        for key in list(origins):
            if key == pointer or key.startswith(pointer + "/"):
                del origins[key]
        for key in _leaves(value, pointer):
            origins[key] = dict(origin)

    def replacement(old, new, pointer, origin, *, existed):
        history.append({"pointer": pointer, "old_sha256": digest(old) if existed else None,
                        "new_sha256": digest(new), "origin": dict(origin)})
        assign_origin(new, pointer, origin)
        return new

    def merge(old, new, pointer, origin, complete):
        if type(old) is dict and type(new) is dict:
            if "kind" in new and old.get("kind") != new["kind"]:
                complete.append(pointer)
                return replacement(old, new, pointer, origin, existed=True)
            result = dict(old)
            for key, value in new.items():
                ptr = _pointer(pointer, key)
                if key in result:
                    result[key] = merge(result[key], value, ptr, origin, complete)
                else:
                    result[key] = replacement(None, value, ptr, origin, existed=False)
            return result
        if old is not None and new is not None and type(old) is not type(new):
            if not (type(old) in (int, float) and type(new) in (int, float)):
                raise ConfigError(f"incompatible merge types at {pointer}")
        if type(new) is dict:
            complete.append(pointer)
        return replacement(old, new, pointer, origin, existed=True)

    def normalize(*, partial, complete=()):
        _envelope(tree, first=True)
        if tree["pipeline"] != schema.pipeline or tree["science_schema_version"] != schema.version:
            raise ConfigError("pipeline/science schema mismatch or changed version")
        science = schema.normalize(JSONRecord(tree.get("science", {})).to_dict(),
                                   ValidationContext(partial, tuple(complete)))
        if type(science) is not dict:
            raise ConfigError("science validator must return an object")
        JSONRecord(science)
        tree["science"] = science
        tree["execution"] = _execution(tree.get("execution", {}), partial=partial)

    def locators():
        paths = tuple(schema.resources(JSONRecord(tree["science"]).to_dict()))
        if any(not isinstance(p, ResourcePath) for p in paths):
            raise ConfigError("resource declarations must be ResourcePath records")
        if len({p.pointer for p in paths}) != len(paths):
            raise ConfigError("duplicate resource locator declaration")
        for item in paths:
            if not item.pointer.startswith("/science/") or not item.sha256_pointer.startswith("/science/"):
                raise ConfigError("science resource pointers must be within /science")
        return paths

    def check_paths():
        pointers = [p.pointer for p in locators()]
        if tree["execution"].get("output_root") is not None:
            pointers.append("/execution/output_root")
        for ptr in pointers:
            if ptr not in origins:
                raise ConfigError(f"validator added undeclared field/default: {ptr}")
            resolve_locator(_get(tree, ptr), anchor=origins[ptr].get("anchor"))

    for ordinal, doc in enumerate(documents):
        if not isinstance(doc, ConfigDocument) or doc.ordinal != ordinal:
            raise ConfigError("config documents must retain their ordered ordinals")
        sha256(doc.sha256, where="config content hash")
        if not Path(doc.locator).is_absolute():
            raise ConfigError("config document locator must be absolute")
        content = doc.tree.to_dict()
        _envelope(content, first=ordinal == 0)
        if content.pop("document_role", "config") != doc.document_role:
            raise ConfigError("document role binding mismatch")
        for key in ("schema_version", "pipeline", "science_schema_version"):
            if key in content and key in tree and content[key] != tree[key]:
                raise ConfigError(f"immutable envelope field changed: {key}")
        origin = {"kind": "file", "ordinal": ordinal, "locator": doc.locator,
                  "sha256": doc.sha256, "anchor": str(Path(doc.locator).parent)}
        complete = []
        tree = merge(tree, content, "", origin, complete)
        normalize(partial=True, complete=complete)
        check_paths()

    # Значения по умолчанию применяются только к отсутствующим ключам, не к null.
    def defaults(current, declared, pointer):
        for key, value in declared.items():
            ptr = _pointer(pointer, key)
            if key not in current:
                current[key] = replacement(None, value, ptr,
                    {"kind": "default", "anchor": None}, existed=False)
            elif type(current[key]) is dict and type(value) is dict:
                defaults(current[key], value, ptr)

    defaults(tree["science"], declared_defaults, "/science")
    defaults(tree["execution"], _EXECUTION_DEFAULTS, "/execution")
    if isinstance(overrides, str):
        raise ConfigError("overrides must be an ordered sequence")
    parsed = []
    for index, spec in enumerate(overrides):
        if type(spec) is not str or "=" not in spec:
            raise ConfigError("override must be /pointer=JSON")
        ptr, raw = spec.split("=", 1)
        tokens = _tokens(ptr)
        if tokens[0] not in ("science", "execution") or tokens[-1] == "kind":
            raise ConfigError("override requires a science/execution field; replace the full kind spec")
        parsed.append((index, ptr, raw, strict_json(raw)))
    if output_root is not None and any(
        p == "/execution" or p == "/execution/output_root" for _, p, _, _ in parsed
    ):
        raise ConfigError("--output-root conflicts with an execution/output_root override")
    if output_root is not None:
        parsed.append((len(parsed), "/execution/output_root", None, str(output_root)))
    for index, ptr, raw, value in parsed:
        # Родительский путь разрешается до изменения; массивы и неявное создание родителей не
        # допускаются.
        prior = JSONRecord(tree).to_dict()
        _set(tree, ptr, value)
        try:
            old = _get(prior, ptr)
            existed = True
        except ConfigError:
            old, existed = None, False
        origin = {"kind": "cli", "index": index, "anchor": root}
        replacement(old, value, ptr, origin, existed=existed)
        cli.append({"index": index, "pointer": ptr, "json": raw, "value": value})
        normalize(partial=True, complete=(ptr,) if type(value) in (dict, list) else ())
        # Общие поля и execution имеют закрытую схему; замена объекта должна быть полной.
        if ptr == "/execution":
            _execution(tree["execution"], partial=False)
        check_paths()
    normalize(partial=False)
    check_paths()

    projection = {key: JSONRecord(tree[key]).to_value()
                  for key in ("schema_version", "pipeline", "science_schema_version", "science")}
    paths = locators()
    resolved, bindings = {}, []
    for path in paths:
        expected = sha256(_get(tree, path.sha256_pointer), where=path.sha256_pointer)
        raw = _get(tree, path.pointer)
        origin = origins[path.pointer]
        locator = resolve_locator(raw, anchor=origin.get("anchor"))
        resolved[path.pointer] = {"original": raw, "anchor": origin.get("anchor"), "resolved": locator}
        _set(tree, path.pointer, locator, arrays=True)
        if path.pointer.startswith("/science/"):
            _set(projection, path.pointer, {"content_sha256": expected}, arrays=True)
        bindings.append(ResourceBinding(path.pointer, locator, expected))
    ptr = "/execution/output_root"
    raw = tree["execution"]["output_root"]
    if raw is not None:
        anchor = origins[ptr].get("anchor")
        locator = resolve_locator(raw, anchor=anchor)
        resolved[ptr] = {"original": raw, "anchor": anchor, "resolved": locator}
        tree["execution"]["output_root"] = locator
    execution = tree["execution"]
    out = execution["output_root"]
    inputs = [d.locator for d in documents] + [b.locator for b in bindings]
    if out is not None and out in inputs:
        raise ConfigError("output_root collides with an input locator")
    return EffectiveConfig(JSONRecord(tree), JSONRecord(projection), JSONRecord({
        "inputs": [{"ordinal": d.ordinal, "locator": d.locator, "sha256": d.sha256,
                    "original": d.original, "anchor": d.anchor,
                    "document_role": d.document_role} for d in documents],
        "overrides": cli, "origins": origins, "replacements": history, "paths": resolved,
        "canonicalization": {"version": "adrkit-json/1", "python": platform.python_version(),
                             "serializer": "stdlib.json", "unicode_normalization": "none"},
    }), tuple(bindings))


def load_config(files, *, schema: ScienceSchema, overrides=(), config_root=None,
                output_root=None) -> EffectiveConfig:
    """Прочитать и объединить конфигурации заданной схемы.

    Parameters
    ----------
    files : sequence of str or pathlib.Path
        Непустая упорядоченная последовательность файлов.
    schema : ScienceSchema
        Явная схема научных параметров и ресурсов.
    overrides : sequence of str, optional
        Замены полей в формате /pointer=JSON; порядок сохраняется.
    config_root : str, pathlib.Path or None, optional
        Абсолютный корень относительных имён `files` и путей
        из `overrides`.
    output_root : str, pathlib.Path or None, optional
        Переопределённый каталог результата; относительный путь
        разрешается через config_root.

    Returns
    -------
    config : EffectiveConfig
        Действующая конфигурация с происхождением и ресурсами.
    """
    return compose_config(read_configs(files, config_root=config_root), schema=schema,
                          overrides=overrides, config_root=config_root, output_root=output_root)


def verify_resources(config: EffectiveConfig) -> tuple[ResourceBinding, ...]:
    """Проверить контрольные суммы объявленных ресурсов.

    Parameters
    ----------
    config : EffectiveConfig
        Конфигурация с абсолютными путями и ожидаемыми SHA-256.

    Returns
    -------
    resources : tuple of ResourceBinding
        Исходные привязки config.resources после успешной проверки.

    Notes
    -----
    Несовпадение суммы или ошибка чтения вызывает ConfigError.
    """
    for binding in config.resources:
        hasher = hashlib.sha256()
        try:
            with open(binding.locator, "rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(chunk)
        except OSError as exc:
            raise ConfigError(f"missing/unreadable resource: {binding.pointer}") from exc
        if hasher.hexdigest() != binding.sha256:
            raise ConfigError(f"resource content mismatch: {binding.pointer}")
    return config.resources
