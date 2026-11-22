"""Raw-source save helpers shared by config mutations and setup flows."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Optional


_SECURITY_COMMENT = """
# ── Security ──────────────────────────────────────────────────────────
# Secret redaction is ON by default — strings that look like API keys,
# tokens, and passwords are masked in tool output, logs, and chat
# responses before the model or user ever sees them. Set redact_secrets
# to false to disable (e.g. when developing the redactor itself).
#
# security:
#   redact_secrets: true
"""

_FALLBACK_COMMENT = """
# ── Fallback Model ────────────────────────────────────────────────────
# Automatic provider failover when primary is unavailable.
# Uncomment and configure to enable. Triggers on rate limits (429),
# overload (529), service errors (503), or connection failures.
#
# Supported providers:
#   openrouter   (OPENROUTER_API_KEY)  — routes to any model
#   openai-codex (OAuth — hermes auth) — OpenAI Codex
#   nous         (OAuth — hermes auth) — Nous Portal
#   zai          (ZAI_API_KEY)         — Z.AI / GLM
#   kimi-coding  (KIMI_API_KEY)        — Kimi / Moonshot
#   kimi-coding-cn (KIMI_CN_API_KEY)   — Kimi / Moonshot (China)
#   minimax      (MINIMAX_API_KEY)     — MiniMax
#   minimax-cn   (MINIMAX_CN_API_KEY)  — MiniMax (China)
#   bedrock      (AWS IAM / boto3)     — AWS Bedrock (Converse API)
#
# For custom OpenAI-compatible endpoints, add base_url and key_env.
#
# fallback_model:
#   provider: openrouter
#   model: anthropic/claude-sonnet-4
"""


def explicit_config_paths(config: dict[str, Any]) -> set[tuple[str, ...]]:
    """Leaf paths explicitly present in a raw config document."""
    paths: set[tuple[str, ...]] = set()

    def walk(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, path + (key,))
        elif path:
            paths.add(path)

    walk(config, ())
    return paths


def commented_sections_for_save(normalized: dict[str, Any]) -> Optional[str]:
    """Commented-out example blocks for features that are off or unconfigured."""
    parts = []
    if (normalized.get("security") or {}).get("redact_secrets") is None:
        parts.append(_SECURITY_COMMENT)
    fallback = normalized.get("fallback_model", {})
    entries = fallback if isinstance(fallback, list) else [fallback]
    if not any(isinstance(entry, dict) and entry.get("provider") and entry.get("model") for entry in entries):
        parts.append(_FALLBACK_COMMENT)
    return "".join(parts) or None


def restore_unedited_plugin_defaults(
    config: dict[str, Any], raw: dict[str, Any], explicit_paths: set[tuple[str, ...]], *,
    home: Path, managed: dict[str, Any],
) -> dict[str, Any]:
    """Keep inherited manifest leaves out of saves unless their path was explicitly edited."""
    if not isinstance(raw, dict):
        return config
    from hermes_cli.config import _MISSING, _get_nested, _normalize_root_model_keys, _set_nested, _unset_nested
    from hermes_cli.config_plugin_defaults import PluginDefaultsError
    from hermes_cli.plugins_config_defaults import collect_config_default_layers

    try:
        normalized_raw = _normalize_root_model_keys(copy.deepcopy(raw))
        layers, _signature = collect_config_default_layers(home, normalized_raw, managed)
    except PluginDefaultsError:
        # Raw config recovery must remain usable while a malformed enabled declaration is repaired.
        return config

    def walk(values: dict[str, Any], prefix: tuple[str, ...] = ()):
        for key, value in values.items():
            path = (*prefix, key)
            if isinstance(value, dict):
                yield from walk(value, path)
            else:
                yield path

    for declaration in layers:
        for path in walk(declaration.values):
            if path in explicit_paths:
                continue
            dotted = ".".join(part.replace(".", "\\.") for part in path)
            raw_block: tuple[int, Any] | None = None
            for depth in range(1, len(path)):
                ancestor = ".".join(part.replace(".", "\\.") for part in path[:depth])
                ancestor_value = _get_nested(normalized_raw, ancestor)
                if ancestor_value is not _MISSING and not isinstance(ancestor_value, dict):
                    raw_block = (depth, ancestor_value)
                    break
            if raw_block is not None:
                depth, ancestor_value = raw_block
                ancestor_parts = path[:depth]
                ancestor = ".".join(part.replace(".", "\\.") for part in ancestor_parts)
                written_ancestor = _get_nested(config, ancestor)
                if written_ancestor is _MISSING or not isinstance(written_ancestor, dict):
                    continue  # the caller deleted or replaced the ancestor
                authored_below = any(intent[:depth] == ancestor_parts for intent in explicit_paths)
                if authored_below:
                    _unset_nested(config, dotted)
                else:
                    _set_nested(config, ancestor, copy.deepcopy(ancestor_value))
                continue

            config_blocks_path = False
            for depth in range(1, len(path) + 1):
                written_path = ".".join(part.replace(".", "\\.") for part in path[:depth])
                written_value = _get_nested(config, written_path)
                if written_value is _MISSING:
                    config_blocks_path = True  # an omitted raw path or ancestor is an explicit deletion
                    break
                if depth < len(path) and not isinstance(written_value, dict):
                    config_blocks_path = True  # an explicit scalar or null ancestor blocks descendants
                    break
            if config_blocks_path:
                continue
            source_value = _get_nested(normalized_raw, dotted)
            if source_value is _MISSING:
                _unset_nested(config, dotted)
            else:
                _set_nested(config, dotted, copy.deepcopy(source_value))
    return config
