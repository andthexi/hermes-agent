"""Read and cache enabled plugin default declarations without executing plugin code."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Mapping

from hermes_cli._startup_fast import project_root_str
from hermes_cli.config_plugin_defaults import (
    PluginDefaultDeclaration, PluginDefaultsError, parse_default_declaration,
)
from hermes_cli.plugins_manifest import manifest_key, resolve_plugin_load_order

logger = logging.getLogger(__name__)
_BUNDLED_PLUGINS_DIR = Path(project_root_str()) / "plugins"
_CACHE: dict[str, tuple[str, list[PluginDefaultDeclaration]]] = {}
_CACHE_LOCK = threading.RLock()


def _effective_plugin_bootstrap(raw: Mapping[str, Any], managed: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    """Return explicit enabled/disabled names after the managed plugin section overlays raw values."""
    raw_plugins = raw.get("plugins")
    plugins = dict(raw_plugins) if isinstance(raw_plugins, Mapping) else {}
    managed_plugins = managed.get("plugins")
    if "plugins" in managed:
        if not isinstance(managed_plugins, Mapping):
            plugins = {}
        else:
            plugins.update(managed_plugins)
    enabled_raw = plugins.get("enabled")
    disabled_raw = plugins.get("disabled")
    enabled = {item for item in enabled_raw if isinstance(item, str)} if isinstance(enabled_raw, list) else set()
    disabled = {item for item in disabled_raw if isinstance(item, str)} if isinstance(disabled_raw, list) else set()
    return enabled, disabled


def _source_manifest_signature(home: Path, project_enabled: bool) -> str:
    """Content-aware source signature without parsing unchanged manifest YAML."""
    from utils import file_signature

    digest = hashlib.sha256()
    sources = [
        (_BUNDLED_PLUGINS_DIR, "bundled", {"memory", "context_engine", "model-providers", "cron_providers", "computer_use", "config_backends"}),
        (_BUNDLED_PLUGINS_DIR / "model-providers", "bundled", None),
        (home / "plugins", "user", None),
    ]
    if project_enabled:
        sources.append((Path.cwd() / ".hermes" / "plugins", "project", None))
    for root, source, skip_names in sources:
        for plugin_dir in _manifest_dirs(root, skip_names=skip_names, include_portable=True):
            manifest_path = next((candidate for candidate in (
                plugin_dir / "plugin.yaml", plugin_dir / "plugin.yml", plugin_dir / "plugin.json"
            ) if candidate.is_file()), None)
            if manifest_path is None:
                continue
            try:
                stat = manifest_path.stat()
                content_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                identity = (source, str(manifest_path.resolve()), *file_signature(stat), content_hash)
            except OSError:
                identity = (source, str(manifest_path.resolve()), "unreadable")
            digest.update(json.dumps(identity, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def _manifest_file(plugin_dir: Path) -> Path | None:
    return next((path for path in (plugin_dir / "plugin.yaml", plugin_dir / "plugin.yml") if path.is_file()), None)


def _manifest_dirs(
    root: Path, *, skip_names: set[str] | None = None, include_portable: bool = False,
) -> list[Path]:
    """Plugin directories at the same flat/category depth as ``scan_directory``."""
    from hermes_cli.plugins_discovery import is_skipped_plugin_dir

    found: list[Path] = []
    if not root.is_dir():
        return found
    try:
        children = sorted(root.iterdir())
    except OSError:
        return found
    for child in children:
        if is_skipped_plugin_dir(child.name) or (skip_names and child.name in skip_names):
            continue
        try:
            if not child.is_dir():
                continue
            if _manifest_file(child) is not None or (include_portable and (child / "plugin.json").is_file()):
                found.append(child)
                continue
            for nested in sorted(child.iterdir()):
                if nested.is_dir() and not is_skipped_plugin_dir(nested.name) and (
                    _manifest_file(nested) is not None or (include_portable and (nested / "plugin.json").is_file())
                ):
                    found.append(nested)
        except OSError:
            continue
    return found


def _manifest_name(manifest_file: Path, fallback: str) -> str:
    """Best-effort identity recovery from malformed YAML; never surface a manifest value in errors."""
    try:
        from utils import fast_safe_load
        data = fast_safe_load(manifest_file.read_text(encoding="utf-8-sig"))
        if isinstance(data, Mapping) and isinstance(data.get("name"), str) and data["name"].strip():
            return data["name"].strip()
    except Exception:  # health: allow BLE001 -- parser diagnostics may contain private manifest scalars
        logger.debug("could not parse a manifest identity from %s", manifest_file)
    try:
        match = re.search(r"(?m)^name:\s*([A-Za-z0-9_.-]+)\s*(?:#.*)?$", manifest_file.read_text(encoding="utf-8-sig"))
        return match.group(1) if match else fallback
    except (OSError, UnicodeDecodeError):
        logger.debug("could not read a manifest identity from %s", manifest_file)
        return fallback


def _invalid_manifest_candidate(root: Path, plugin_dir: Path, source: str):
    """Identity-only placeholder so an enabled malformed file keeps its place in source precedence."""
    from hermes_cli.plugins_manifest import PluginManifest

    relative = plugin_dir.relative_to(root)
    leaf = plugin_dir.name
    name = _manifest_name(_manifest_file(plugin_dir) or (plugin_dir / "plugin.yaml"), leaf)
    key = name if len(relative.parts) == 1 else "/".join(relative.parts)
    return PluginManifest(name=name, source=source, path=str(plugin_dir), key=key), key


def _source_manifests(home: Path):
    """Read winning manifests as data and retain parse-failed identities for enabled-plugin errors."""
    from hermes_cli.plugins_discovery import resolve_manifest_winners, scan_directory

    sources = [
        (_BUNDLED_PLUGINS_DIR, "bundled", {"memory", "context_engine", "model-providers", "cron_providers", "computer_use", "config_backends"}, ""),
        (_BUNDLED_PLUGINS_DIR / "model-providers", "bundled", None, "model-providers"),
        (home / "plugins", "user", None, ""),
    ]
    project_enabled = os.environ.get("HERMES_ENABLE_PROJECT_PLUGINS") == "1"
    if project_enabled:
        sources.append((Path.cwd() / ".hermes" / "plugins", "project", None, ""))

    manifests = []
    invalid_paths: dict[str, str] = {}
    for root, source, skip_names, prefix in sources:
        valid = scan_directory(root, source, skip_names=skip_names, prefix=prefix)
        valid_by_path = {str(Path(item.path).resolve()): item for item in valid if item.path}
        candidates = list(valid)
        for plugin_dir in _manifest_dirs(root, skip_names=skip_names):
            path_key = str(plugin_dir.resolve())
            if path_key in valid_by_path:
                continue
            placeholder, registry_key = _invalid_manifest_candidate(root, plugin_dir, source)
            if prefix and len(plugin_dir.relative_to(root).parts) == 1:
                placeholder.key = f"{prefix}/{plugin_dir.name}"
                registry_key = placeholder.key
            candidates.append(placeholder)
            invalid_paths[path_key] = registry_key
        candidates.sort(key=lambda item: str(Path(item.path).resolve()) if item.path else "")
        manifests.extend(candidates)

    winners = resolve_manifest_winners(manifests)
    invalid_winners = {
        key for key, manifest in winners.items()
        if manifest.path and str(Path(manifest.path).resolve()) in invalid_paths
    }
    return winners, invalid_winners, project_enabled


def collect_config_default_layers(
    home: str | os.PathLike[str], raw_bootstrap: Mapping[str, Any], managed_bootstrap: Mapping[str, Any],
) -> tuple[list[PluginDefaultDeclaration], str]:
    """Collect valid defaults from explicitly enabled winning manifests, bound to *home*.

    Discovery reads YAML only: provider/general plugin Python modules are never imported here. The
    signature binds declarations to the profile, the effective enable/disable lists, and the exact
    winning manifest membership/content.
    """
    home_path = Path(home).expanduser().resolve()
    enabled, disabled = _effective_plugin_bootstrap(raw_bootstrap, managed_bootstrap)
    project_enabled = os.environ.get("HERMES_ENABLE_PROJECT_PLUGINS") == "1"
    bootstrap = json.dumps({
        "enabled": sorted(enabled), "disabled": sorted(disabled), "project_enabled": project_enabled,
    }, sort_keys=True, separators=(",", ":"))
    home_key = str(home_path)
    if not enabled:
        signature = hashlib.sha256((home_key + "\0" + bootstrap + "\0no-enabled-plugin").encode()).hexdigest()
        cached = _CACHE.get(home_key)
        if cached is not None and cached[0] == signature:
            return list(cached[1]), signature
        with _CACHE_LOCK:
            _CACHE[home_key] = (signature, [])
        return [], signature

    identity = _source_manifest_signature(home_path, project_enabled)
    signature = hashlib.sha256(
        (home_key + "\0" + bootstrap + "\0" + identity).encode("utf-8")
    ).hexdigest()
    cached = _CACHE.get(home_key)
    if cached is not None and cached[0] == signature:
        return list(cached[1]), signature

    winners, invalid_winners, _project_enabled = _source_manifests(home_path)
    explicit: dict[str, Any] = {}
    enabled_manifests: dict[str, Any] = {}
    known_config_paths: set[str] = set()
    for key, manifest in winners.items():
        names = {key, manifest.name}
        if manifest.key:
            names.add(manifest.key)
        if manifest.path:
            names.add(Path(manifest.path).name)
        if names & disabled or not names & enabled:
            continue
        if key in invalid_winners:
            raise PluginDefaultsError(key, "manifest", "enabled plugin manifest could not be parsed")
        # Preserve existing source winner resolution and model-provider availability. This gate also
        # rejects removed plugin records and impossible legacy Relay registrations before defaults apply.
        from hermes_cli.plugins_discovery import gate_manifest
        verdict = gate_manifest(manifest, disabled, enabled)
        if verdict.error and verdict.error != "disabled via config":
            if "removed" in verdict.error.lower() or "relay" in verdict.error.lower():
                logger.warning("Plugin %s defaults are unavailable: %s", key, verdict.error)
                continue
        from hermes_cli.plugins_manifest import requires_hermes_error
        incompatibility = requires_hermes_error(manifest)
        if incompatibility:
            logger.warning("Plugin %s defaults are unavailable: %s", key, incompatibility)
            continue
        known_config_paths.update(manifest.config_keys)
        enabled_manifests[key] = manifest
        if manifest.has_config_defaults:
            explicit[key] = manifest

    ordered_keys = resolve_plugin_load_order(enabled_manifests)
    declarations: list[PluginDefaultDeclaration] = []
    for key in ordered_keys:
        manifest = explicit.get(key)
        if manifest is not None:
            declarations.append(parse_default_declaration(
                key, manifest.config_defaults, known_config_paths=known_config_paths))
    with _CACHE_LOCK:
        _CACHE[home_key] = (signature, declarations)
    return list(declarations), signature
