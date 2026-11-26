"""Task-scoped route-validation contracts against existing provider owners."""

from pathlib import Path
from unittest.mock import patch

import hermes_yaml as yaml
import pytest

from providers.base import ProviderProfile
from tools.delegate_tool_validation import _catalog_verdict, _reasoning_verdict


class Parent:
    model = "parent-switched-model"
    provider = "custom"


def test_parent_grant_snapshot_failure_log_redacts_plugin_exception(caplog, monkeypatch):
    import logging
    from types import SimpleNamespace
    from tools.registry import registry
    from tools.delegate_tool_validation import _parent_tool_grant

    sentinel = "PLUGIN_GRANT_LOOKUP_SECRET_SENTINEL"
    class PluginAvailabilityFailure(Exception):
        pass

    class ParentWithUnverifiableTools(SimpleNamespace):
        @property
        def platform(self):
            raise PluginAvailabilityFailure(sentinel)

    name = "mcp-grant-redaction-probe"
    schema = {"name": name, "description": "Synthetic availability probe",
              "parameters": {"type": "object", "properties": {}}}
    registry.register(name=name, toolset="mcp-grant-redaction-probe", schema=schema,
                      handler=lambda *_args, **_kwargs: "unused", check_fn=lambda: True)
    parent = ParentWithUnverifiableTools(
        _session_db=None, enabled_toolsets=["mcp-grant-redaction-probe"],
        _configured_disabled_toolsets=None, disabled_toolsets=None,
        _tool_snapshot_selection=None, _delegation_launch_policy=None,
    )
    try:
        with patch("tools.connectors.gateway.config.connectors_available", return_value=False), \
             patch("tools.tool_search.load_config", return_value=SimpleNamespace(
                 effective_defer_tools=frozenset(), enabled="off")), \
             patch("tools.tool_search.load_config_readonly", return_value=SimpleNamespace(
                 enabled="off", effective_defer_tools=frozenset())), \
             patch("tools.connectors.gateway.config.connectors_available", return_value=False), \
             caplog.at_level(logging.WARNING, logger="tools.delegate_tool"):
            grant = _parent_tool_grant(parent)
    finally:
        registry.deregister(name)

    assert grant is None
    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "Unable to snapshot owning-parent concrete tool grant (PluginAvailabilityFailure)" in logs
    assert sentinel not in logs
    assert not any(record.exc_info for record in caplog.records)


def test_provider_only_saved_default_does_not_resolve_alias_provider_credentials(tmp_path, monkeypatch):
    from tools.delegate_tool_routes import TaskRoute
    from tools.delegate_tool_validation import resolve_task_options, resolve_task_routes

    host_home = tmp_path / "os"
    root = host_home / ".hermes"
    root.mkdir(parents=True)
    coder = root / "profiles" / "coder"
    coder.mkdir(parents=True)
    (root / "config.yaml").write_text(
        yaml.safe_dump({"model": {"provider": "custom", "default": "launch-model"}}), encoding="utf-8")
    (coder / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": "openai", "default": "saved-relay"},
        "model_aliases": {"saved-relay": {"model": "profile-alias-model", "provider": "openai",
                                            "base_url": "http://127.0.0.1:9731/v1",
                                            "key_env": "MISSING_ALIAS_KEY"}},
        "providers": {"target-relay": {"api": "http://127.0.0.1:9732/v1"}},
    }), encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: host_home)
    monkeypatch.setenv("HOME", str(host_home))
    monkeypatch.setenv("HERMES_HOME", str(root))
    from hermes_constants import named_profile_has_identity
    assert named_profile_has_identity(coder)
    calls = []

    def runtime_provider(**kwargs):
        calls.append(kwargs["requested"])
        if kwargs["requested"] == "target-relay":
            return {"provider": "custom", "requested_provider": "target-relay",
                    "base_url": "http://127.0.0.1:9732/v1", "api_key": "target-profile-key",
                    "api_mode": "chat_completions", "request_overrides": {}, "command": None, "args": []}
        raise ValueError("profile alias provider credentials are missing")

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", runtime_provider)
    task = {"goal": "profile alias baseline with replacement provider", "profile": "coder", "provider": "target-relay"}
    options, errors = resolve_task_options([task])
    (route,) = resolve_task_routes([task], {}, Parent(), [TaskRoute({}, {})], options, errors)

    assert route.error is None
    assert route.creds["model"] == "profile-alias-model"
    assert route.creds["provider"] == "target-relay"
    assert route.creds["api_key"] == "target-profile-key"
    assert calls.count("target-relay") == 1
    assert "openai" not in calls


def test_authoritative_catalog_rejection_refuses_exact_pair():
    with patch("hermes_cli.models_validate.static_model_provider_conflict", return_value=None), \
         patch("hermes_cli.models_validate.validate_requested_model", return_value={
             "accepted": False, "recognized": False, "message": "Model not in authoritative live catalog"}):
        assert _catalog_verdict("bad-exact-id", "openai", {"api_key": "fixture"}) == (
            True, "Model not in authoritative live catalog", False)


def test_accepted_unknown_pair_remains_unverified():
    with patch("hermes_cli.models_validate.static_model_provider_conflict", return_value=None), \
         patch("hermes_cli.models_validate.validate_requested_model", return_value={
             "accepted": True, "recognized": False, "message": "catalog unavailable"}):
        assert _catalog_verdict("exact-id", "custom", {}) == (False, "catalog unavailable", False)


def test_kimi_k3_explicit_none_uses_the_route_thinking_toggle():
    assert _reasoning_verdict("kimi-k3", "kimi-coding", {"api_mode": "chat_completions"}, "none") is None


def test_openai_compat_wire_vocab_does_not_override_responses_model_ladder():
    assert _reasoning_verdict("gpt-6-luna", "openai-codex", {"api_mode": "codex_responses"}, "max") is None
    verdict = _reasoning_verdict("gpt-6-luna", "openai-codex", {"api_mode": "codex_responses"}, "ultra")
    assert verdict is not None and verdict[0] == "refused"


def test_kimi_k3_owner_recognizes_high_and_refuses_unsupported_medium():
    assert _reasoning_verdict("kimi-k3", "kimi-coding", {"api_mode": "chat_completions"}, "high") is None
    verdict = _reasoning_verdict("kimi-k3", "kimi-coding", {"api_mode": "chat_completions"}, "medium")
    assert verdict is not None and verdict[0] == "refused"
    assert "unsupported" in verdict[1]


def test_tokenhub_owner_recognizes_high_and_refuses_unsupported_ultra():
    route = {"api_mode": "chat_completions"}
    assert _reasoning_verdict("hy-model", "tencent-tokenhub", route, "high") is None
    verdict = _reasoning_verdict("hy-model", "tencent-tokenhub", route, "ultra")
    assert verdict is not None and verdict[0] == "refused"
    assert "unsupported" in verdict[1]


def test_tokenhub_explicit_none_is_refused_when_builder_omits_disable():
    # Runtime imports model_tools before route validation, which triggers provider discovery.
    from importlib import import_module

    import_module("model_tools")
    import providers
    from agent.transports.chat_completions import ChatCompletionsTransport

    profile = providers.get_provider_profile("tencent-tokenhub")
    assert profile is not None and profile.name == "tencent-tokenhub"
    route = {
        "api_mode": "chat_completions",
        "base_url": "https://tokenhub.tencentmaas.com/v1",
    }
    with pytest.raises(ValueError, match="cannot be carried exactly"):
        ChatCompletionsTransport().build_kwargs(
            model="hy-model",
            messages=[{"role": "user", "content": "ping"}],
            tools=None,
            provider_profile=profile,
            reasoning_config={"enabled": False, "effort": "none"},
            strict_reasoning_effort=True,
            is_tokenhub=True,
            base_url=route["base_url"],
        )

    verdict = _reasoning_verdict("hy-model", "tencent-tokenhub", route, "none")
    assert verdict is not None and verdict[0] == "refused"
    assert "none" in verdict[1]

def test_codex_app_server_uses_route_effort_vocabulary_and_preserves_responses_ladder():
    route = {"api_mode": "codex_app_server"}
    assert _reasoning_verdict("gpt-6-luna", "openai-codex", route, "ultra") is None

    unsupported = _reasoning_verdict("gpt-5.5", "openai-codex", route, "ultra")
    assert unsupported is not None and unsupported[0] == "refused"
    assert "unsupported" in unsupported[1]

    # Responses API does not accept the app-server-only ultra name.
    responses = _reasoning_verdict("gpt-6-luna", "openai-codex", {"api_mode": "codex_responses"}, "ultra")
    assert responses is not None and responses[0] == "refused"


def test_custom_openai_compat_profile_does_not_claim_unknown_effort_support():
    with (
        patch("providers.get_provider_profile", return_value=None),
        patch("providers.profile_for_host", return_value=None),
        patch("agent.models_dev.get_model_capabilities", return_value=None),
    ):
        verdict = _reasoning_verdict("relay-model", "custom", {"api_mode": "chat_completions"}, "max")
    assert verdict is not None and verdict[0] == "unverified"


def test_custom_responses_mode_does_not_infer_openai_compat_efforts():
    with (
        patch("providers.get_provider_profile", return_value=None),
        patch("providers.profile_for_host", return_value=None),
        patch("agent.models_dev.get_model_capabilities", return_value=None),
    ):
        verdict = _reasoning_verdict("relay-model", "custom", {"api_mode": "codex_responses"}, "max")
    assert verdict is not None and verdict[0] == "unverified"


def test_unknown_responses_route_does_not_infer_openai_compat_efforts():
    with (
        patch("providers.get_provider_profile", return_value=None),
        patch("providers.profile_for_host", return_value=None),
        patch("agent.models_dev.get_model_capabilities", return_value=None),
    ):
        verdict = _reasoning_verdict("kimi-k3", "unknown-relay", {"api_mode": "codex_responses"}, "high")
    assert verdict is not None and verdict[0] == "unverified"


def test_anthropic_reasoning_effort_is_unknown_not_generic_chat_vocab():
    verdict = _reasoning_verdict("claude-model", "anthropic", {"api_mode": "anthropic_messages"}, "high")
    assert verdict is not None and verdict[0] == "unverified"


def test_explicit_none_is_not_claimed_for_no_disable_codex_tier():
    verdict = _reasoning_verdict("gpt-6.1-sol", "openai-codex", {"api_mode": "codex_responses"}, "none")
    assert verdict is not None and verdict[0] == "refused"


def test_profile_declared_efforts_are_enforced():
    profile = type("RouteProfile", (ProviderProfile,), {
        "supported_reasoning_efforts": lambda self, model: ("low", "high"),
    })(name="fixture-owner")
    with patch("providers.get_provider_profile", return_value=profile), patch("providers.profile_for_host", return_value=None):
        verdict = _reasoning_verdict("relay-model", "fixture-owner", {"api_mode": "chat_completions"}, "medium")
    assert verdict is not None and verdict[0] == "refused"


def test_reasoning_none_is_refused_when_profile_vocabulary_omits_it():
    profile = type("NoNoneProfile", (ProviderProfile,), {
        "supported_reasoning_efforts": lambda self, model: ("low", "high"),
    })(name="fixture-owner")
    with patch("providers.get_provider_profile", return_value=profile), patch("providers.profile_for_host", return_value=None):
        verdict = _reasoning_verdict("relay-model", "fixture-owner", {"api_mode": "chat_completions"}, "none")
    assert verdict is not None and verdict[0] == "refused"


def test_empty_profile_effort_set_refuses_enabled_efforts_and_explicit_none():
    profile = type("NoEffortProfile", (ProviderProfile,), {
        "supported_reasoning_efforts": lambda self, model: (),
    })(name="fixture-owner")
    with patch("providers.get_provider_profile", return_value=profile), patch("providers.profile_for_host", return_value=None):
        enabled = _reasoning_verdict("plain-model", "fixture-owner", {"api_mode": "chat_completions"}, "low")
        disabled = _reasoning_verdict("plain-model", "fixture-owner", {"api_mode": "chat_completions"}, "none")
    assert enabled is not None and enabled[0] == "refused"
    assert disabled is not None and disabled[0] == "refused"


def test_reasoning_unknown_support_reports_unverified_status():
    profile = type("UnknownProfile", (ProviderProfile,), {
        "supported_reasoning_efforts": lambda self, model: None,
    })(name="fixture-owner")
    with patch("providers.get_provider_profile", return_value=profile), patch("providers.profile_for_host", return_value=None):
        verdict = _reasoning_verdict("relay-model", "fixture-owner", {"api_mode": "chat_completions"}, "ultra")
    assert verdict is not None and verdict[0] == "unverified"


def test_registered_mode_hook_failure_invalidates_declared_reasoning_capabilities(caplog):
    import logging
    import providers

    sentinel = "MODE_HOOK_SECRET_SENTINEL"
    calls = []

    class RaisingModeProfile(ProviderProfile):
        def supported_reasoning_efforts(self, model):
            calls.append(("effort", model))
            return ("high",)

        def api_mode_for_model(self, model, *, api_key=None):
            calls.append(("mode", model))
            raise RuntimeError(sentinel)

    profile = RaisingModeProfile(name="fixture-mode-hook-failure", api_mode="chat_completions")
    providers.list_providers()
    providers.register_provider(profile)
    try:
        with patch("agent.safe_worker_policy.safe_worker_enabled", return_value=False):
            with patch("hermes_cli.config.load_config_readonly", return_value={"model": {"provider": profile.name}}):
                with caplog.at_level(logging.DEBUG, logger="tools.delegate_tool"):
                    assert providers.get_provider_profile(profile.name) is profile
                    verdict = _reasoning_verdict(
                        "exact-child-model", profile.name, {"api_mode": "chat_completions"}, "high",
                    )
    finally:
        providers._REGISTRY.pop(profile.name, None)
        providers._SOURCES.pop(profile.name, None)
        providers._PROVIDER_LIST_CACHE = None

    assert calls == [("effort", "exact-child-model"), ("mode", "exact-child-model")]
    assert verdict is not None and verdict[0] == "unverified"
    assert verdict[1] == (
        "Reasoning support is unknown for fixture-mode-hook-failure/exact-child-model; "
        "preserving the requested effort exactly. Provider capability hook failed (RuntimeError)."
    )
    assert sentinel not in "\n".join(record.getMessage() for record in caplog.records)
    assert not any(record.exc_info for record in caplog.records)


def test_reasoning_none_with_unknown_support_is_not_assumed_verified():
    profile = type("UnknownProfile", (ProviderProfile,), {
        "supported_reasoning_efforts": lambda self, model: None,
    })(name="fixture-owner")
    with patch("providers.get_provider_profile", return_value=profile), patch("providers.profile_for_host", return_value=None):
        verdict = _reasoning_verdict("relay-model", "fixture-owner", {"api_mode": "chat_completions"}, "none")
    assert verdict is not None and verdict[0] == "unverified"
    assert "unknown" in verdict[1]


def test_registered_effort_hook_failure_is_unverified_and_redacts_exception(caplog):
    import logging
    import providers

    secret = "EFFORT_HOOK_SECRET_SENTINEL"
    class ProviderHookFailure(Exception):
        pass

    class RaisingEffortProfile(ProviderProfile):
        def supported_reasoning_efforts(self, model):
            raise ProviderHookFailure(secret)

    profile = RaisingEffortProfile(name="fixture-effort-failure", api_mode="chat_completions")
    providers.list_providers()
    providers.register_provider(profile)
    try:
        with patch("providers.profile_for_host", return_value=None), \
             patch("agent.models_dev.get_model_capabilities", return_value=None), \
             caplog.at_level(logging.DEBUG, logger="tools.delegate_tool"):
            verdict = _reasoning_verdict(
                "exact-child-model", profile.name, {"api_mode": "chat_completions"}, "high",
            )
    finally:
        providers._REGISTRY.pop(profile.name, None)
        providers._SOURCES.pop(profile.name, None)
        providers._PROVIDER_LIST_CACHE = None

    assert verdict is not None and verdict[0] == "unverified"
    assert "ProviderHookFailure" in verdict[1]
    assert "preserving the requested effort exactly" in verdict[1]
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "ProviderHookFailure" in logged
    assert secret not in logged
    assert not any(record.exc_info for record in caplog.records)


def test_registered_mode_hook_failure_names_only_exception_class():
    import providers

    secret = "MODE_HOOK_DIAGNOSTIC_SECRET_SENTINEL"
    class ModeHookFailure(Exception):
        pass

    class RaisingModeProfile(ProviderProfile):
        def supported_reasoning_efforts(self, model):
            return ("high",)

        def api_mode_for_model(self, model, *, api_key=None):
            raise ModeHookFailure(secret)

    profile = RaisingModeProfile(name="fixture-mode-diagnostic", api_mode="chat_completions")
    providers.list_providers()
    providers.register_provider(profile)
    try:
        with patch("providers.profile_for_host", return_value=None), \
             patch("agent.models_dev.get_model_capabilities", return_value=None):
            verdict = _reasoning_verdict(
                "exact-child-model", profile.name, {"api_mode": "chat_completions"}, "high",
            )
    finally:
        providers._REGISTRY.pop(profile.name, None)
        providers._SOURCES.pop(profile.name, None)
        providers._PROVIDER_LIST_CACHE = None

    assert verdict is not None and verdict[0] == "unverified"
    assert "ModeHookFailure" in verdict[1]
    assert "preserving the requested effort exactly" in verdict[1]
    assert secret not in verdict[1]


def test_reasoning_cache_failure_reports_class_without_exception_text(caplog):
    import logging

    secret = "CAPABILITY_CACHE_SECRET_SENTINEL"
    class CapabilityCacheFailure(Exception):
        pass

    with (
        patch("providers.get_provider_profile", return_value=None),
        patch("providers.profile_for_host", return_value=None),
        patch("agent.models_dev.get_model_capabilities", side_effect=CapabilityCacheFailure(secret)),
        caplog.at_level(logging.DEBUG, logger="tools.delegate_tool"),
    ):
        verdict = _reasoning_verdict(
            "exact-child-model", "fixture-cache-provider", {"api_mode": "chat_completions"}, "high",
        )

    assert verdict is not None and verdict[0] == "unverified"
    assert "CapabilityCacheFailure" in verdict[1]
    assert "preserving the requested effort exactly" in verdict[1]
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "CapabilityCacheFailure" in logged
    assert secret not in logged and secret not in verdict[1]
    assert not any(record.exc_info for record in caplog.records)


def test_catalog_validation_failure_is_inconclusive_and_redacts_exception(caplog):
    import logging

    secret = "CATALOG_VALIDATOR_SECRET_SENTINEL"
    class CatalogValidatorFailure(Exception):
        pass

    with (
        patch("hermes_cli.models_validate.static_model_provider_conflict", return_value=None),
        patch("hermes_cli.models_validate.validate_requested_model",
              side_effect=CatalogValidatorFailure(secret)),
        caplog.at_level(logging.DEBUG, logger="tools.delegate_tool"),
    ):
        refused, message, recognized = _catalog_verdict("exact-selected-id", "fixture-provider", {})

    assert (refused, recognized) == (False, False)
    assert "CatalogValidatorFailure" in message
    assert "exact route will be attempted" in message
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "CatalogValidatorFailure" in logged
    assert secret not in logged and secret not in message
    assert not any(record.exc_info for record in caplog.records)


def test_catalog_probe_error_has_explicit_unverified_message():
    with patch("hermes_cli.models_validate.static_model_provider_conflict", return_value=None), \
         patch("hermes_cli.models_validate.validate_requested_model", side_effect=OSError("unreachable")):
        refused, message, recognized = _catalog_verdict("exact-id", "custom", {})
    assert refused is False and recognized is False
    assert "inconclusive" in message and "OSError" in message


def _profile_test_scope(monkeypatch, host_home, root):
    from pathlib import Path
    monkeypatch.setattr(Path, "home", lambda: host_home)
    monkeypatch.setenv("HOME", str(host_home))
    monkeypatch.setenv("HERMES_HOME", str(root))
    from hermes_constants import named_profile_has_identity
    assert named_profile_has_identity(root / "profiles" / "coder")


def test_resolve_task_routes_provider_only_uses_saved_profile_alias_not_parent_model(tmp_path, monkeypatch):
    from agent.secret_scope import get_secret
    from tools.delegate_tool_routes import TaskRoute
    from tools.delegate_tool_validation import resolve_task_options, resolve_task_routes

    host_home = tmp_path / "os"
    root = host_home / ".hermes"
    root.mkdir(parents=True)
    coder = root / "profiles" / "coder"
    coder.mkdir(parents=True)
    (root / "config.yaml").write_text("model:\n  provider: custom\n  default: launch-model\n", encoding="utf-8")
    (coder / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": "custom", "default": "saved-alias"},
        "model_aliases": {"saved-alias": {"model": "exact-profile-model", "provider": "custom",
                                            "base_url": "http://127.0.0.1:9731/v1",
                                            "key_env": "OLD_PROVIDER_KEY"}},
        "providers": {"target-relay": {"api": "http://127.0.0.1:9732/v1",
                                         "key_env": "TARGET_PROVIDER_KEY"}},
    }), encoding="utf-8")
    _profile_test_scope(monkeypatch, host_home, root)
    monkeypatch.delenv("OLD_PROVIDER_KEY", raising=False)
    monkeypatch.delenv("TARGET_PROVIDER_KEY", raising=False)
    def runtime_provider(**kwargs):
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly()
        if kwargs.get("requested") == "target-relay":
            assert config["providers"]["target-relay"]["api"] == "http://127.0.0.1:9732/v1"
            assert config["model"]["default"] == "saved-alias"
            return {"provider": "custom", "requested_provider": "target-relay",
                    "base_url": "http://127.0.0.1:9732/v1", "api_key": "profile-target-secret",
                    "api_mode": "chat_completions", "request_overrides": {}, "command": None, "args": []}
        if kwargs.get("requested") == "openai":
            raise ValueError("old provider credentials are unavailable")
        return {"provider": "custom", "requested_provider": kwargs.get("requested"),
                "base_url": kwargs.get("explicit_base_url") or "http://127.0.0.1:9731/v1",
                "api_key": kwargs.get("explicit_api_key") or "legacy-profile-secret",
                "api_mode": "chat_completions", "request_overrides": {}, "command": None, "args": []}
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", runtime_provider)

    task = {"goal": "provider only", "profile": "coder", "provider": "target-relay"}
    options, errors = resolve_task_options([task])
    (route,) = resolve_task_routes([task], {}, Parent(), [TaskRoute({}, {})], options, errors)
    assert route.error is None
    assert route.creds["model"] == "exact-profile-model"
    assert route.creds["provider"] == "target-relay"
    assert route.creds["base_url"] == "http://127.0.0.1:9732/v1"
    assert route.creds["api_key"] == "profile-target-secret"
    assert route.routing_cfg["fallback_providers"] == []
    assert get_secret("TARGET_PROVIDER_KEY") is None
    assert get_secret("OLD_PROVIDER_KEY") is None


def test_resolve_task_routes_explicit_model_provider_uses_selected_profile_key(tmp_path, monkeypatch):
    from agent.secret_scope import get_secret
    from tools.delegate_tool_routes import TaskRoute
    from tools.delegate_tool_validation import resolve_task_options, resolve_task_routes

    host_home = tmp_path / "os"
    root = host_home / ".hermes"
    root.mkdir(parents=True)
    coder = root / "profiles" / "coder"
    coder.mkdir(parents=True)
    (root / "config.yaml").write_text("model:\n  provider: custom\n  default: launch-model\n", encoding="utf-8")
    (coder / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": "custom", "default": "profile-default"},
        "providers": {"target-relay": {"api": "http://127.0.0.1:9732/v1",
                                         "key_env": "TARGET_PROVIDER_KEY"}},
    }), encoding="utf-8")
    (coder / ".env").write_text("TARGET_PROVIDER_KEY=profile-target-secret\n", encoding="utf-8")
    from pathlib import Path
    monkeypatch.setattr(Path, "home", lambda: host_home)
    monkeypatch.setenv("HOME", str(host_home))
    monkeypatch.setenv("HERMES_HOME", str(root))
    from hermes_constants import named_profile_has_identity
    assert named_profile_has_identity(coder)
    monkeypatch.delenv("TARGET_PROVIDER_KEY", raising=False)
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda **kwargs: {
        "provider": "custom", "requested_provider": kwargs.get("requested"),
        "base_url": "http://127.0.0.1:9732/v1", "api_key": "profile-target-secret",
        "api_mode": "chat_completions", "request_overrides": {}, "command": None, "args": [],
    })

    task = {"goal": "explicit model provider", "profile": "coder",
            "model": "exact-explicit-model", "provider": "target-relay"}
    options, errors = resolve_task_options([task])
    (route,) = resolve_task_routes([task], {}, Parent(), [TaskRoute({}, {})], options, errors)

    assert route.error is None
    assert route.creds["model"] == "exact-explicit-model"
    assert route.creds["provider"] == "target-relay"
    assert route.creds["base_url"] == "http://127.0.0.1:9732/v1"
    assert route.creds["api_key"] == "profile-target-secret"
    assert get_secret("TARGET_PROVIDER_KEY") is None


def test_resolve_task_routes_missing_saved_default_refuses_without_inference(tmp_path, monkeypatch):
    from tools.delegate_tool_routes import TaskRoute
    from tools.delegate_tool_validation import resolve_task_options, resolve_task_routes
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda **kwargs: {
        "provider": "custom", "requested_provider": kwargs.get("requested"),
        "base_url": "http://127.0.0.1:9732/v1", "api_key": "sibling-provider-key",
        "api_mode": "chat_completions", "request_overrides": {}, "command": None, "args": [],
    })

    host_home = tmp_path / "os"
    root = host_home / ".hermes"
    root.mkdir(parents=True)
    coder = root / "profiles" / "coder"
    sibling = root / "profiles" / "sibling"
    coder.mkdir(parents=True)
    sibling.mkdir(parents=True)
    (coder / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    (sibling / "config.yaml").write_text("model:\n  provider: custom\n  default: sibling-model\n", encoding="utf-8")
    from hermes_constants import named_profile_has_identity
    assert named_profile_has_identity(sibling)
    _profile_test_scope(monkeypatch, host_home, root)

    tasks = [{"goal": "provider only", "profile": "coder", "provider": "target-relay"},
             {"goal": "valid sibling", "profile": "sibling", "provider": "target-relay"}]
    options, errors = resolve_task_options(tasks)
    routes = resolve_task_routes(tasks, {}, Parent(), [TaskRoute({}, {}), TaskRoute({}, {})], options, errors)

    assert routes[0].error is not None
    assert "saved profile default model" in routes[0].error
    assert routes[0].creds.get("model") is None
    assert routes[1].error is None
    assert routes[1].creds["model"] == "sibling-model"


def test_explicit_provider_precedes_baseline_provider_for_direct_model(monkeypatch):
    from tools.delegate_tool_options import TaskRouteOptions
    from tools.delegate_tool_validation import _task_model_provider

    task = {"goal": "direct model with explicit provider", "model": "exact-model",
            "provider": "target-provider"}
    option, error = TaskRouteOptions.parse(task, 0)
    assert error is None and option is not None
    calls = []

    def resolve_runtime_provider(**kwargs):
        calls.append(kwargs["requested"])
        if kwargs["requested"] == "old-provider":
            raise ValueError("old provider credentials are unavailable")
        return {"provider": "custom", "requested_provider": "target-provider",
                "base_url": "http://127.0.0.1:9732/v1", "api_key": "target-key",
                "api_mode": "chat_completions", "request_overrides": {}}

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", resolve_runtime_provider)
    parent = type("Parent", (), {"provider": "old-provider", "base_url": None, "api_key": None})()
    route, model = _task_model_provider(
        task, option, {"model": "old-model", "provider": "old-provider", "base_url": None, "api_key": None},
        {}, 0, parent,
    )

    assert calls == ["target-provider"]
    assert model == "exact-model"
    assert route["provider"] == "target-provider"
    assert route["api_key"] == "target-key"


def test_provider_only_static_preflight_uses_saved_default_not_switched_credentials():
    from pathlib import Path
    from tools.delegate_tool_options import TaskRouteOptions
    from tools.delegate_tool_routes import TaskRoute
    from tools.delegate_tool_validation import _preflight_route_pair
    from hermes_cli.models_validate import static_model_provider_conflict

    task = {"goal": "provider only", "provider": "gemini"}
    option, error = TaskRouteOptions.parse(task, 0)
    assert error is None and option is not None
    profile_config = {"model": {"provider": "gemini", "default": "gemini-2.5-pro",
                                 "base_url": "http://127.0.0.1:9731/v1", "api_key": "saved-profile-key"}}
    switched_route = TaskRoute({"model": "gpt-6-luna", "provider": "custom"}, {}, profile_home=Path("/profile"))

    assert static_model_provider_conflict("gpt-6-luna", "gemini") is not None
    assert static_model_provider_conflict("gemini-2.5-pro", "gemini") is None
    _preflight_route_pair(option, switched_route.creds, profile_config, switched_route, Parent(), 0)


def test_provider_only_saved_default_static_conflict_precedes_credentials(tmp_path, monkeypatch):
    from tools.delegate_tool_routes import TaskRoute
    from tools.delegate_tool_validation import resolve_task_options, resolve_task_routes

    host_home = tmp_path / "os"
    root = host_home / ".hermes"
    root.mkdir(parents=True)
    (root / "config.yaml").write_text("model:\n  provider: custom\n  default: launch-model\n", encoding="utf-8")
    worker = root / "profiles" / "worker"
    worker.mkdir(parents=True)
    (worker / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": "custom", "default": "gpt-6-luna"},
        "providers": {"gemini": {"api": "http://127.0.0.1:9731/v1", "key_env": "MISSING_GEMINI_KEY"}},
    }), encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: host_home)
    monkeypatch.setenv("HOME", str(host_home))
    monkeypatch.setenv("HERMES_HOME", str(root))
    from hermes_constants import named_profile_has_identity
    assert named_profile_has_identity(worker)
    from tools import delegate_tool_options
    credentials = []
    def resolve_credentials(*args, **kwargs):
        credentials.append(args[0] if args else kwargs.get("provider"))
        raise AssertionError("invalid saved default must refuse before credential resolution")
    monkeypatch.setattr(delegate_tool_options, "_provider_credentials", resolve_credentials)

    task = {"goal": "static conflict beats auth", "profile": "worker", "provider": "gemini"}
    options, errors = resolve_task_options([task])
    (route,) = resolve_task_routes([task], {}, Parent(), [TaskRoute({}, {})], options, errors)

    assert route.error is not None
    assert "gpt-6-luna" in route.error and "gemini" in route.error
    assert credentials == []
