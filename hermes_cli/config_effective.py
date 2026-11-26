"""The effective USER config: config.yaml + managed overlay + ``${VAR}`` expansion, no defaults.

``load_config()`` merges ``DEFAULT_CONFIG`` first, which is wrong for readers that treat a
missing key as "unset" (the gateway's presence-sensitive env bridge, ``cfg == {}`` sentinels,
cron model pinning) — so nine surfaces used to hand-roll raw-read → overlay → expand in
differing orders and none of them replayed the model-key canonicalization or the last-known-good
recovery ``load_config()`` gained. This module is that one primitive.

Order matches ``_load_config_impl``: the user layer is expanded BEFORE the managed overlay so a
managed ``${VAR}`` resolves against the process environment only (``apply_managed_overlay``
expands it) and can never be re-resolved through a profile's secret scope
(docs/design/managed-scope.md §4.1). ``read_user_config_raw`` stays the write-back primitive.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from hermes_cli import config as _config
from hermes_cli import managed_scope
from hermes_cli.config_backend import read_config_doc
from hermes_cli.config_read_errors import _warn_config_parse_failure

# path -> raw user mapping from the last successful parse in this process; served (through the
# normal pipeline) when the file is later found mid-edit as broken YAML.
_LAST_GOOD_USER_RAW: dict[str, dict[str, Any]] = {}
# path -> (*user_signature, *managed_signature, effective, env_snapshot, root_is_mapping);
# see utils.file_signature.
_EFFECTIVE_CACHE: dict[str, tuple[Any, ...]] = {}


def _effective(raw: dict[str, Any]) -> dict[str, Any]:
    expanded = _config._expand_env_vars(raw)
    merged = managed_scope.apply_managed_overlay(expanded if isinstance(expanded, dict) else {})
    return _config._normalize_root_model_keys(merged if isinstance(merged, dict) else {})


def _recover_user_raw(config_path: Path, path_key: str, exc: Exception) -> dict[str, Any]:
    """Last-known-good raw user mapping after a parse failure: this process's last good parse,
    else the newest ``good`` copy in backups/config/, else ``{}`` (warned as defaults)."""
    raw = _LAST_GOOD_USER_RAW.get(path_key)
    fallback = "last-known-good"
    if raw is None:
        from hermes_cli.config_backups import load_newest_good_backup
        raw = load_newest_good_backup(config_path)
        fallback = "last-known-good-backup"
    _warn_config_parse_failure(config_path, exc, fallback=fallback if raw is not None else "defaults")
    return copy.deepcopy(raw) if raw is not None else {}


def load_user_config_effective(config_path: Optional[Path] = None, *, fail_closed: bool = False) -> dict[str, Any]:
    """Raw user settings plus enabled plugin defaults and managed policy, without built-in defaults.

    Plugin values are composed beneath explicitly present user values and managed policy.
    Broken YAML: ``fail_closed=True`` raises the parse error, including a non-mapping YAML root
    (for callers that keep their own last-good state); otherwise the last successfully parsed
    user file — in-process first, then
    the newest ``backups/config/*.good.*`` copy — is served through the same pipeline, so a
    mid-edit torn write never silently drops user overrides (same contract as ``load_config``).
    Cached on the user + managed file signatures and the values of every referenced env var."""
    from agent.safe_worker_policy import worker_config_snapshot

    snapshot = worker_config_snapshot()
    if snapshot is not None:
        return _effective(snapshot)
    if config_path is None:
        config_path = _config.get_config_path()
    path_key = str(config_path)
    with _config._CONFIG_LOCK:
        user_sig, source_cache_sig = _config._load_config_cache_sig(config_path)
        raw: dict[str, Any] = {}
        recovered = False
        root_is_mapping: Optional[bool] = True if user_sig is None else None
        raw_hit = _config._raw_config_cache_hit(path_key, user_sig) if user_sig is not None else None
        if raw_hit is not None:
            raw = copy.deepcopy(raw_hit)
            _LAST_GOOD_USER_RAW.setdefault(path_key, copy.deepcopy(raw))
        elif user_sig is not None:
            try:
                loaded = read_config_doc(config_path)
            except Exception as exc:
                if fail_closed:
                    raise
                raw, recovered = _recover_user_raw(config_path, path_key, exc), True
            else:
                root_is_mapping = loaded is None or isinstance(loaded, dict)
                if fail_closed and not root_is_mapping:
                    raise TypeError("config.yaml root is not a mapping")
                raw = loaded if isinstance(loaded, dict) else {}
                _config._raw_config_cache_store(path_key, user_sig, copy.deepcopy(raw))
                _LAST_GOOD_USER_RAW[path_key] = copy.deepcopy(raw)
                if config_path == _config.get_config_path():
                    from hermes_cli.config_backups import backup_config
                    backup_config(config_path, "good")

        if fail_closed and root_is_mapping is None:
            # Raw-config cache hits contain only normalized dicts, not the root shape.
            loaded = read_config_doc(config_path)
            if loaded is not None and not isinstance(loaded, dict):
                raise TypeError("config.yaml root is not a mapping")
            root_is_mapping = True

        env_snapshot = _config._env_ref_snapshot(raw)
        managed = managed_scope.load_managed_config()
        normalized = _config._normalize_root_model_keys(copy.deepcopy(raw))
        from hermes_cli.config_plugin_defaults import compose_user_layer
        from hermes_cli.plugins_config_defaults import collect_config_default_layers

        layers, plugin_signature = collect_config_default_layers(config_path.parent, normalized, managed or {})
        plugin_user = compose_user_layer(normalized, layers)
        cache_sig = (*source_cache_sig, plugin_signature) if source_cache_sig is not None else None
        cached = _EFFECTIVE_CACHE.get(path_key)
        n = len(cache_sig) if cache_sig is not None else 0
        if cached is not None and cache_sig is not None and cached[:n] == cache_sig:
            if all(_config._env_ref_lookup(k) == v for k, v in cached[n + 1].items()):
                return copy.deepcopy(cached[n])

        env_snapshot = _config._env_ref_snapshot(plugin_user)
        if managed:
            _config._env_ref_snapshot(managed, env_snapshot)
        effective = _effective(plugin_user)
        if cache_sig is not None and not recovered:
            _EFFECTIVE_CACHE[path_key] = (
                *cache_sig, copy.deepcopy(effective), env_snapshot, root_is_mapping,
            )
        else:
            _EFFECTIVE_CACHE.pop(path_key, None)
        return effective
