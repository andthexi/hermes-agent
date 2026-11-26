"""Validation and composition for declarative plugin configuration defaults.

Defaults are a read-time layer only. This module deliberately knows nothing about plugin
execution or profile discovery so a caller can compose already validated manifest data without
importing plugin code.
"""

from __future__ import annotations

import copy
import datetime as _datetime
import json
import logging
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

CONFIG_DEFAULTS_VERSION = 1
_FORBIDDEN_PATHS = frozenset({
    "plugins", "_config_version", "config_backend", "remote_config", "deployment",
    "deployment_id", "config_plane", "config_url", "config_token", "config_key",
})


class PluginDefaultsError(ValueError):
    """A value-free error in an enabled plugin's config-defaults declaration."""

    def __init__(self, plugin: str, path: str, reason: str):
        self.plugin = plugin
        self.path = path
        self.reason = reason
        super().__init__(f"Plugin '{plugin}' config_defaults{'.' + path if path else ''}: {reason}")


@dataclass(frozen=True)
class PluginDefaultDeclaration:
    """Validated, detached config-default values from one plugin manifest."""

    plugin_key: str
    values: dict[str, Any]
    merge: dict[str, str]


def _fail(plugin: str, path: str, reason: str) -> PluginDefaultsError:
    return PluginDefaultsError(plugin, path, reason)


def _validate_json_value(plugin: str, value: Any, path: str, active: set[int]) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _fail(plugin, path, "values must contain only finite JSON numbers")
        return value
    if isinstance(value, (_datetime.date, _datetime.datetime, _datetime.time)):
        raise _fail(plugin, path, "YAML dates and times are not JSON-compatible")
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise _fail(plugin, path, "recursive YAML aliases are not supported")
        active.add(identity)
        result: dict[str, Any] = {}
        try:
            for key, child in value.items():
                if not isinstance(key, str):
                    raise _fail(plugin, path, "mapping keys must be strings")
                child_path = f"{path}.{key}" if path else key
                result[key] = _validate_json_value(plugin, child, child_path, active)
        finally:
            active.remove(identity)
        return result
    if isinstance(value, list):
        identity = id(value)
        if identity in active:
            raise _fail(plugin, path, "recursive YAML aliases are not supported")
        active.add(identity)
        try:
            return [_validate_json_value(plugin, child, path, active) for child in value]
        finally:
            active.remove(identity)
    raise _fail(plugin, path, f"value type {type(value).__name__} is not JSON-compatible")


def _path_parts(path: str) -> list[str]:
    """Split a dotted config path, honoring the established backslash-escaped dot spelling."""
    parts: list[str] = []
    current: list[str] = []
    index = 0
    while index < len(path):
        char = path[index]
        if char == "\\" and path[index + 1:index + 2] == ".":
            current.append(".")
            index += 2
        elif char == ".":
            parts.append("".join(current))
            current = []
            index += 1
        else:
            current.append(char)
            index += 1
    parts.append("".join(current))
    return parts


def _escaped_path(parts: tuple[str, ...]) -> str:
    return ".".join(part.replace("\\", "\\\\").replace(".", "\\.") for part in parts)


def _path_is_covered(path: str, known_paths: set[str]) -> bool:
    return any(
        path == known or path.startswith(known + ".") or known.startswith(path + ".")
        for known in known_paths
    )


def _config_schemas() -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Known user/runtime schema data, without loading active plugin discovery."""
    from hermes_cli.config import DEFAULT_CONFIG
    from hermes_cli.cli_config_load import _cli_config_defaults

    return DEFAULT_CONFIG, _cli_config_defaults()


def _schema_value(parts: tuple[str, ...]) -> Any:
    """Return a known schema node from the config or classic-CLI runtime schema."""
    schemas = _config_schemas()
    if parts == ("agent", "max_turns"):
        # The unlimited default is null; finite configured turn caps remain integers.
        return (int,)
    if parts[0] == "model":
        schemas = (schemas[1], schemas[0])  # DEFAULT_CONFIG's model root is a legacy string shorthand.
    saw_untyped_path = False
    for schema in schemas:
        node: Any = schema
        for part in parts:
            if not isinstance(node, Mapping):
                node = _MISSING
                break
            if not node:
                return Mapping
            if part not in node:
                node = _MISSING
                break
            node = node[part]
        if node is not _MISSING:
            if parts == ("model",) and isinstance(node, str):
                return (str, Mapping)
            if len(parts) == 2 and parts[0] == "model" and parts[1] in {"default", "model", "name"}:
                return (str, Mapping)
            if node is None:
                saw_untyped_path = True
                continue
            return node
    return None if saw_untyped_path else _MISSING


def _known_schema_path(parts: tuple[str, ...], declared_paths: set[str]) -> bool:
    from hermes_cli.config import _validate_schema_key
    from hermes_cli.config_runtime_keys import CORE_RUNTIME_KEYS

    path = _escaped_path(parts)
    if _validate_schema_key(path)[0]:
        return True
    return _path_is_covered(path, declared_paths | set(CORE_RUNTIME_KEYS))


def _matches_schema_value(value: Any, schema: Any) -> bool:
    if value is None or schema is _MISSING or schema is None:
        return True
    if isinstance(schema, tuple):
        return isinstance(value, schema) and not (isinstance(value, bool) and bool not in schema)
    if isinstance(schema, Mapping):
        return isinstance(value, Mapping)
    if isinstance(schema, bool):
        return isinstance(value, bool)
    if isinstance(schema, int):
        return isinstance(value, int) and not isinstance(value, bool)
    if isinstance(schema, float):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    if isinstance(schema, str):
        return isinstance(value, str)
    if isinstance(schema, list):
        return isinstance(value, list)
    return True


def _validate_secret_fields_in_values(plugin: str, value: Any, path: str) -> None:
    """Inspect JSON containers, including list entries, for credential-shaped field names."""
    from hermes_cli.config import _is_secret_config_key

    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            if _is_secret_config_key(child_path):
                raise _fail(plugin, child_path, "credential-value config keys cannot be defaulted")
            _validate_secret_fields_in_values(plugin, child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_secret_fields_in_values(plugin, child, f"{path}[{index}]")


def _validate_paths(
    plugin: str, values: Mapping[str, Any], known_config_paths: set[str],
) -> None:
    """Reject config controls, credential values, unknown runtime paths, and known-shape mismatches."""
    from hermes_cli.config import _is_secret_config_key

    declared_paths = set(known_config_paths)

    def visit(node: Mapping[str, Any], prefix: tuple[str, ...] = ()) -> None:
        for key, value in node.items():
            path = (*prefix, key)
            dotted = _escaped_path(path)
            if ((not prefix and key in _FORBIDDEN_PATHS) or key == "_config_version"):
                raise _fail(plugin, dotted, "config/plugin-loading controls cannot be defaulted")
            if _is_secret_config_key(dotted) and not (not prefix and key == "secrets"):
                raise _fail(plugin, dotted, "credential-value config keys cannot be defaulted")
            if not _known_schema_path(path, declared_paths):
                raise _fail(plugin, dotted, "unknown Hermes config path")
            if not _matches_schema_value(value, _schema_value(path)):
                raise _fail(plugin, dotted, "value shape does not match the known Hermes config setting")
            if isinstance(value, Mapping):
                visit(value, path)

    visit(values)


def _get_path(mapping: Mapping[str, Any], parts: list[str]) -> tuple[bool, Any]:
    current: Any = mapping
    for part in parts:
        if not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    return True, current


def parse_default_declaration(
    plugin_key: str, block: Any, *, known_config_paths: set[str] | frozenset[str] = frozenset(),
) -> PluginDefaultDeclaration:
    """Validate and detach one manifest ``config_defaults`` mapping."""
    if not isinstance(block, Mapping):
        raise _fail(plugin_key, "", "declaration must be a mapping")
    version = block.get("version")
    if type(version) is not int or version != CONFIG_DEFAULTS_VERSION:
        raise _fail(plugin_key, "version", f"unsupported declaration version (supported: {CONFIG_DEFAULTS_VERSION})")
    raw_values = block.get("values")
    if not isinstance(raw_values, Mapping):
        raise _fail(plugin_key, "values", "must be a mapping")
    values = _validate_json_value(plugin_key, raw_values, "values", set())
    _validate_paths(plugin_key, values, set(known_config_paths))
    _validate_secret_fields_in_values(plugin_key, values, "values")
    raw_merge = block.get("merge", {})
    if not isinstance(raw_merge, Mapping):
        raise _fail(plugin_key, "merge", "must be a mapping")
    merge: dict[str, str] = {}
    for path, mode in raw_merge.items():
        if not isinstance(path, str) or not path or any(not p for p in _path_parts(path)):
            raise _fail(plugin_key, "merge", "merge paths must be non-empty dotted config paths")
        if mode != "append_unique":
            raise _fail(plugin_key, f"merge.{path}", "only the append_unique merge mode is supported")
        parts = _path_parts(path)
        found, value = _get_path(values, parts)
        if not found or not isinstance(value, list):
            raise _fail(plugin_key, f"merge.{path}", "append_unique must refer to a list in this declaration")
        if parts[0] in _FORBIDDEN_PATHS:
            raise _fail(plugin_key, f"merge.{path}", "config/plugin-loading controls cannot be defaulted")
        merge[path] = mode
    return PluginDefaultDeclaration(plugin_key, values, merge)


def collect_profile_default_layers(
    home: str | os.PathLike[str], raw_bootstrap: Mapping[str, Any], managed_bootstrap: Mapping[str, Any],
) -> tuple[list[PluginDefaultDeclaration], str]:
    """Resolve one profile's declarations from config data already read by its owner."""
    from hermes_cli.config import _normalize_root_model_keys
    from hermes_cli.plugins_config_defaults import collect_config_default_layers

    normalized = _normalize_root_model_keys(copy.deepcopy(dict(raw_bootstrap)))
    return collect_config_default_layers(home, normalized, managed_bootstrap)


def cached_profile_config(
    path_key: str, config_path, home: str | os.PathLike[str], managed: Mapping[str, Any], *,
    want_deepcopy: bool,
) -> Any | None:
    """Return a warm effective config only when its raw profile and manifest sources are current."""
    from hermes_cli.config import (
        _LOAD_CONFIG_CACHE, _load_config_cache_hit, _load_config_cache_sig, _raw_config_cache_hit,
    )

    if path_key not in _LOAD_CONFIG_CACHE:
        return None
    user_sig, base_cache_sig = _load_config_cache_sig(config_path)
    raw = {} if user_sig is None else _raw_config_cache_hit(path_key, user_sig)
    if raw is None:
        return None
    _layers, plugin_signature = collect_profile_default_layers(home, raw, managed)
    cache_sig = (*base_cache_sig, plugin_signature) if base_cache_sig is not None else None
    hit = _load_config_cache_hit(path_key, cache_sig)
    if hit is None:
        return None
    return copy.deepcopy(hit) if want_deepcopy else hit


def _leaf_items(node: Mapping[str, Any], prefix: tuple[str, ...] = ()):
    for key, value in node.items():
        path = (*prefix, key)
        if isinstance(value, Mapping):
            yield from _leaf_items(value, path)
        else:
            yield path, value


def resolve_default_merge_modes(
    declarations: list[PluginDefaultDeclaration],
) -> dict[tuple[str, ...], tuple[str, str, bool]]:
    """Resolve each leaf's earliest contributor mode for both composition and UI metadata."""
    owners: dict[tuple[str, ...], tuple[str, str, bool]] = {}
    for declaration in declarations:
        declared_modes = {tuple(_path_parts(path)): mode for path, mode in declaration.merge.items()}
        for path, value in _leaf_items(declaration.values):
            mode = declared_modes.get(path, "replace")
            owner = owners.get(path)
            if owner is None:
                owners[path] = (declaration.plugin_key, mode, isinstance(value, list))
            elif owner[1] != mode:
                logger.warning(
                    "Plugin defaults merge mode conflict for %s between plugins %s and %s; keeping %s",
                    ".".join(path), owner[0], declaration.plugin_key, owner[1],
                )
    return owners


def _ensure_mapping_path(target: dict[str, Any], parts: tuple[str, ...]) -> dict[str, Any] | None:
    current = target
    for part in parts:
        value = current.get(part, _MISSING)
        if value is _MISSING:
            value = {}
            current[part] = value
        if not isinstance(value, dict):
            return None
        current = value
    return current


_MISSING = object()


def _item_identity(value: Any) -> str:
    """Canonical JSON identity that keeps bool distinct from numbers and preserves numeric spelling."""
    return f"{type(value).__name__}:{json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}"


def _stable_unique(values: list[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for value in values:
        identity = _item_identity(value)
        if identity not in seen:
            seen.add(identity)
            result.append(copy.deepcopy(value))
    return result


def compose_user_layer(raw: Mapping[str, Any], declarations: list[PluginDefaultDeclaration]) -> dict[str, Any]:
    """Return plugin defaults beneath a raw user mapping, preserving explicit presence and null ancestors."""
    if not isinstance(raw, Mapping):
        raise TypeError("raw user config must be a mapping")
    result = copy.deepcopy(dict(raw))
    mode_owners = resolve_default_merge_modes(declarations)
    contributions: dict[tuple[str, ...], list[tuple[str, Any, str]]] = {}
    for declaration in declarations:
        declared_modes = {tuple(_path_parts(path)): mode for path, mode in declaration.merge.items()}
        for path, value in _leaf_items(declaration.values):
            first_plugin, first_mode, _is_list = mode_owners[path]
            mode = declared_modes.get(path, "replace")
            if mode != first_mode or (mode == "replace" and declaration.plugin_key != first_plugin):
                continue
            contributions.setdefault(path, []).append((declaration.plugin_key, value, mode))

    for path, entries in contributions.items():
        raw_found, raw_value = _get_path(result, list(path))
        if raw_found and (raw_value is None or not isinstance(raw_value, list) or entries[0][2] != "append_unique"):
            continue
        if entries[0][2] == "append_unique":
            base = raw_value if raw_found else []
            merged: list[Any] = list(base)
            for _plugin, value, _mode in entries:
                merged.extend(value)
            destination = _ensure_mapping_path(result, path[:-1])
            if destination is not None:
                destination[path[-1]] = _stable_unique(merged)
            continue
        destination = _ensure_mapping_path(result, path[:-1])
        if destination is not None and path[-1] not in destination:
            destination[path[-1]] = copy.deepcopy(entries[0][1])
    return result
