"""Explicit reasoning request/recovery tests use real transport payload builders."""

from pathlib import Path
from types import SimpleNamespace

import hermes_yaml as yaml
import pytest

from agent.transports.chat_completions import ChatCompletionsTransport
from agent.transports.codex import ResponsesApiTransport


def test_anthropic_messages_refuses_mapped_effort_under_strict_mode():
    from agent.transports.anthropic import AnthropicTransport

    with pytest.raises(ValueError, match="cannot be carried"):
        AnthropicTransport().build_kwargs(
            "claude-opus-4-7", [{"role": "user", "content": "go"}], None,
            reasoning_config={"enabled": True, "effort": "ultra"}, strict_reasoning_effort=True,
        )


def test_anthropic_messages_strict_guard_preserves_supported_effort():
    from agent.transports.anthropic import AnthropicTransport

    payload = AnthropicTransport().build_kwargs(
        "claude-opus-4-7", [{"role": "user", "content": "go"}], None,
        reasoning_config={"enabled": True, "effort": "high"}, strict_reasoning_effort=True,
    )
    assert payload["output_config"]["effort"] == "high"


def test_anthropic_messages_builder_forwards_strict_flag():
    from agent.chat_completion_helpers import _build_anthropic_kwargs
    from agent.transports.anthropic import AnthropicTransport

    agent = SimpleNamespace(
        _get_transport=lambda: AnthropicTransport(), model="claude-opus-4-7", max_tokens=2048,
        context_compressor=None, _prepare_anthropic_messages_for_api=lambda messages: messages,
        _is_anthropic_oauth=False, _anthropic_preserve_dots=lambda: False, _anthropic_base_url=None,
        _oauth_1m_beta_disabled=False, _strict_reasoning_effort=True, provider="custom",
    )
    with pytest.raises(ValueError, match="cannot be carried"):
        _build_anthropic_kwargs(
            agent, [{"role": "user", "content": "go"}], None,
            {"enabled": True, "effort": "ultra"}, {},
        )


def test_bedrock_converse_refuses_mapped_effort_under_strict_mode():
    from agent.transports.bedrock import BedrockTransport

    with pytest.raises(ValueError, match="cannot be carried"):
        BedrockTransport().build_kwargs(
            "amazon.nova-2-lite", [{"role": "user", "content": "go"}], None,
            reasoning_config={"enabled": True, "effort": "xhigh"}, strict_reasoning_effort=True,
        )


def test_bedrock_converse_strict_guard_preserves_supported_effort():
    from agent.transports.bedrock import BedrockTransport

    payload = BedrockTransport().build_kwargs(
        "amazon.nova-2-lite", [{"role": "user", "content": "go"}], None,
        reasoning_config={"enabled": True, "effort": "high"}, strict_reasoning_effort=True,
    )
    assert payload["additionalModelRequestFields"]["reasoningConfig"]["maxReasoningEffort"] == "high"


def test_bedrock_converse_strict_guard_refuses_unproven_disable():
    from agent.transports.bedrock import BedrockTransport

    with pytest.raises(ValueError, match="cannot be carried"):
        BedrockTransport().build_kwargs(
            "unknown-reasoning-model", [{"role": "user", "content": "go"}], None,
            reasoning_config={"enabled": False, "effort": "none"}, strict_reasoning_effort=True,
        )


def test_bedrock_builder_forwards_strict_flag():
    from agent.chat_completion_helpers import _build_bedrock_kwargs
    from agent.transports.bedrock import BedrockTransport

    agent = SimpleNamespace(
        _get_transport=lambda: BedrockTransport(), model="amazon.nova-2-lite", max_tokens=2048,
        _bedrock_region="us-east-1", _bedrock_guardrail_config=None, _strict_reasoning_effort=True,
    )
    with pytest.raises(ValueError, match="cannot be carried"):
        _build_bedrock_kwargs(
            agent, [{"role": "user", "content": "go"}], None,
            {"enabled": True, "effort": "xhigh"},
        )


def test_anthropic_messages_strict_none_requires_actual_disable():
    from agent.transports.anthropic import AnthropicTransport

    payload = AnthropicTransport().build_kwargs(
        "claude-opus-4-7", [{"role": "user", "content": "go"}], None,
        reasoning_config={"enabled": False, "effort": "none"}, strict_reasoning_effort=True,
    )
    assert payload["thinking"] == {"type": "disabled"}


def test_bedrock_converse_preserves_anthropic_high_effort():
    from agent.transports.bedrock import BedrockTransport

    payload = BedrockTransport().build_kwargs(
        "us.anthropic.claude-opus-4-7-v1:0", [{"role": "user", "content": "go"}], None,
        reasoning_config={"enabled": True, "effort": "high"}, strict_reasoning_effort=True,
    )
    assert payload["additionalModelRequestFields"]["output_config"]["effort"] == "high"


def test_bedrock_converse_refuses_anthropic_effort_translation():
    from agent.transports.bedrock import BedrockTransport

    with pytest.raises(ValueError, match="cannot be carried"):
        BedrockTransport().build_kwargs(
            "us.anthropic.claude-opus-4-7-v1:0", [{"role": "user", "content": "go"}], None,
            reasoning_config={"enabled": True, "effort": "ultra"}, strict_reasoning_effort=True,
        )


def test_chat_profile_refuses_final_extra_body_effort_override(monkeypatch):
    from providers import get_provider_profile

    profile = get_provider_profile("custom")
    assert profile is not None
    monkeypatch.setattr(profile, "supported_reasoning_efforts", lambda model: ("low", "high"))
    monkeypatch.setattr(
        profile, "build_api_kwargs_extras",
        lambda **kwargs: ({"reasoning": {"effort": "high"}}, {}),
    )
    with pytest.raises(ValueError, match="preserve"):
        ChatCompletionsTransport().build_kwargs(
            "relay-model", [{"role": "user", "content": "go"}], None,
            base_url="http://127.0.0.1:8000/v1", provider_name="custom", provider_profile=profile,
            supports_reasoning=True, reasoning_config={"enabled": True, "effort": "high"},
            strict_reasoning_effort=True,
            request_overrides={"extra_body": {"reasoning": {"effort": "low"}}},
            max_tokens=100, max_tokens_param_fn=lambda n: {"max_tokens": n},
        )


def test_chat_profile_preserves_supported_final_effort(monkeypatch):
    from providers import get_provider_profile

    profile = get_provider_profile("custom")
    assert profile is not None
    monkeypatch.setattr(profile, "supported_reasoning_efforts", lambda model: ("low", "high"))
    monkeypatch.setattr(
        profile, "build_api_kwargs_extras",
        lambda **kwargs: ({"reasoning": {"effort": "high"}}, {}),
    )
    payload = ChatCompletionsTransport().build_kwargs(
        "relay-model", [{"role": "user", "content": "go"}], None,
        base_url="http://127.0.0.1:8000/v1", provider_name="custom", provider_profile=profile,
        supports_reasoning=True, reasoning_config={"enabled": True, "effort": "high"},
        strict_reasoning_effort=True, max_tokens=100, max_tokens_param_fn=lambda n: {"max_tokens": n},
    )
    assert payload["extra_body"]["reasoning"]["effort"] == "high"


def test_chat_profile_refuses_override_that_removes_effort(monkeypatch):
    from providers import get_provider_profile

    profile = get_provider_profile("custom")
    assert profile is not None
    monkeypatch.setattr(profile, "supported_reasoning_efforts", lambda model: ("low", "high"))
    monkeypatch.setattr(
        profile, "build_api_kwargs_extras",
        lambda **kwargs: ({"reasoning": {"effort": "high"}}, {}),
    )
    with pytest.raises(ValueError, match="preserve"):
        ChatCompletionsTransport().build_kwargs(
            "relay-model", [{"role": "user", "content": "go"}], None,
            base_url="http://127.0.0.1:8000/v1", provider_name="custom", provider_profile=profile,
            supports_reasoning=True, reasoning_config={"enabled": True, "effort": "high"},
            strict_reasoning_effort=True, request_overrides={"extra_body": {"reasoning": {}}},
            max_tokens=100, max_tokens_param_fn=lambda n: {"max_tokens": n},
        )


def test_tokenhub_strict_effort_refuses_legacy_translation():
    with pytest.raises(ValueError, match="final request"):
        ChatCompletionsTransport().build_kwargs(
            "tokenhub-model", [{"role": "user", "content": "go"}], None,
            provider_name="unregistered", is_tokenhub=True, supports_reasoning=False,
            reasoning_config={"enabled": True, "effort": "xhigh"}, strict_reasoning_effort=True,
        )


def test_xai_responses_refuses_unsupported_effort_instead_of_clamping():
    with pytest.raises(ValueError, match="cannot be carried"):
        ResponsesApiTransport().build_kwargs(
            "grok-4.5", [{"role": "user", "content": "go"}], None,
            base_url="https://api.x.ai/v1", is_xai_responses=True,
            reasoning_config={"enabled": True, "effort": "xhigh"}, strict_reasoning_effort=True,
        )


def test_xai_responses_preserves_supported_effort():
    payload = ResponsesApiTransport().build_kwargs(
        "grok-4.5", [{"role": "user", "content": "go"}], None,
        base_url="https://api.x.ai/v1", is_xai_responses=True,
        reasoning_config={"enabled": True, "effort": "high"}, strict_reasoning_effort=True,
    )
    assert payload["reasoning"]["effort"] == "high"


def test_xai_responses_refuses_final_effort_override():
    with pytest.raises(ValueError, match="final Responses request"):
        ResponsesApiTransport().build_kwargs(
            "grok-4.5", [{"role": "user", "content": "go"}], None,
            base_url="https://api.x.ai/v1", is_xai_responses=True,
            reasoning_config={"enabled": True, "effort": "high"}, strict_reasoning_effort=True,
            request_overrides={"reasoning": {"effort": "low"}},
        )


def test_kimi_strict_effort_refuses_legacy_translation():
    with pytest.raises(ValueError, match="final request"):
        ChatCompletionsTransport().build_kwargs(
            "kimi-k3", [{"role": "user", "content": "go"}], None,
            provider_name="unknown-kimi-relay", is_kimi=True, supports_reasoning=False,
            reasoning_config={"enabled": True, "effort": "xhigh"}, strict_reasoning_effort=True,
        )


def test_chat_completions_wire_keeps_explicit_effort_verbatim():
    transport = ChatCompletionsTransport()
    payload = transport.build_kwargs(
        "custom-model", [{"role": "user", "content": "go"}], None,
        base_url="http://127.0.0.1:8000/v1", provider_name="custom", supports_reasoning=True,
        reasoning_config={"enabled": True, "effort": "xhigh"}, strict_reasoning_effort=True,
        max_tokens=100, max_tokens_param_fn=lambda n: {"max_tokens": n},
    )
    assert payload["extra_body"]["reasoning"]["effort"] == "xhigh"


def test_chat_completions_fails_instead_of_clamping_explicit_effort(monkeypatch):
    from providers import get_provider_profile
    profile = get_provider_profile("custom")
    monkeypatch.setattr(profile, "supported_reasoning_efforts", lambda model: ("low", "medium"))
    with pytest.raises(ValueError, match="unsupported"):
        ChatCompletionsTransport().build_kwargs(
            "relay-model", [{"role": "user", "content": "go"}], None,
            base_url="http://127.0.0.1:8000/v1", provider_name="custom", provider_profile=profile,
            reasoning_config={"enabled": True, "effort": "high"}, strict_reasoning_effort=True,
            max_tokens=100, max_tokens_param_fn=lambda n: {"max_tokens": n},
        )


def test_codex_refuses_none_on_declared_no_disable_route():
    with pytest.raises(ValueError, match="cannot be carried"):
        ResponsesApiTransport().build_kwargs(
            "gpt-6-astra", [{"role": "user", "content": "go"}], None,
            base_url="https://api.openai.com/v1", is_codex_backend=True,
            reasoning_config={"enabled": False, "effort": "none"}, strict_reasoning_effort=True,
        )


def test_chat_completions_unknown_profile_attempts_exact_extended_effort(monkeypatch):
    from providers import get_provider_profile
    profile = get_provider_profile("custom")
    assert profile is not None
    profile.default_reasoning_config = lambda model=None: {"enabled": True, "effort": "medium"}
    monkeypatch.setattr(profile, "supported_reasoning_efforts", lambda model: None)
    monkeypatch.setattr(profile, "build_api_kwargs_extras", lambda **kwargs: (
        {"reasoning": {"enabled": True, "effort": kwargs["reasoning_config"]["effort"]}}, {},
    ))
    payload = ChatCompletionsTransport().build_kwargs(
        "relay-model", [{"role": "user", "content": "go"}], None,
        base_url="http://127.0.0.1:8000/v1", provider_name="custom", provider_profile=profile,
        supports_reasoning=True, reasoning_config={"enabled": True, "effort": "ultra"}, strict_reasoning_effort=True,
        max_tokens=100, max_tokens_param_fn=lambda n: {"max_tokens": n},
    )
    assert payload["extra_body"]["reasoning"]["effort"] == "ultra"


def test_chat_completions_rejects_profile_hook_that_changes_explicit_effort(monkeypatch):
    from providers import get_provider_profile
    profile = get_provider_profile("custom")
    assert profile is not None
    monkeypatch.setattr(profile, "supported_reasoning_efforts", lambda model: ("low", "high"))
    monkeypatch.setattr(profile, "build_api_kwargs_extras", lambda **kwargs: ({}, {}))
    with pytest.raises(ValueError, match="preserve"):
        ChatCompletionsTransport().build_kwargs(
            "relay-model", [{"role": "user", "content": "go"}], None,
            base_url="http://127.0.0.1:8000/v1", provider_name="custom", provider_profile=profile,
            supports_reasoning=True, reasoning_config={"enabled": True, "effort": "high"}, strict_reasoning_effort=True,
            max_tokens=100, max_tokens_param_fn=lambda n: {"max_tokens": n},
        )


def test_lmstudio_strict_effort_reaches_its_declared_wire_field():
    from providers import get_provider_profile

    payload = ChatCompletionsTransport().build_kwargs(
        "gpt-oss", [{"role": "user", "content": "go"}], None,
        provider_name="lmstudio", provider_profile=get_provider_profile("lmstudio"),
        supports_reasoning=True, lmstudio_reasoning_options=["off", "low", "medium", "high"],
        reasoning_config={"enabled": True, "effort": "high"}, strict_reasoning_effort=True,
        max_tokens=100, max_tokens_param_fn=lambda n: {"max_tokens": n},
    )
    assert payload["reasoning_effort"] == "high"


def test_codex_wire_fails_instead_of_clamping_explicit_effort():
    with pytest.raises(ValueError, match="cannot be carried"):
        ResponsesApiTransport().build_kwargs(
            "gpt-5.5", [{"role": "user", "content": "go"}], None,
            base_url="https://chatgpt.com/backend-api/codex", is_codex_backend=True,
            reasoning_config={"enabled": True, "effort": "max"}, strict_reasoning_effort=True,
        )


    # Legacy callers retain the old wire clamp while strict requests still fail.
    payload = ResponsesApiTransport().build_kwargs(
        "gpt-5.6-codex", [{"role": "user", "content": "go"}], None,
        base_url="https://chatgpt.com/backend-api/codex",
        reasoning_config={"enabled": True, "effort": "ultra"},
    )
    assert payload["reasoning"]["effort"] == "max"


def test_reasoning_rejection_does_not_retry_explicit_child_effort():
    from agent.turn_recovery import recover_before_classification

    agent = SimpleNamespace(
        _strict_reasoning_effort=True, _buffer_vprint=lambda *_: None, _image_rejecting_models=set(),
        model="custom-model", provider="custom", base_url="http://127.0.0.1/v1", api_mode="chat_completions",
    )
    recovered, _ = recover_before_classification(
        agent, ValueError("unsupported reasoning effort"), messages=[], api_messages=[], api_kwargs={},
        active_system_prompt="prompt",
    )
    assert recovered is False


def test_codex_app_server_public_delegate_refuses_grants_and_preserves_ultra(tmp_path, monkeypatch):
    """Public delegation exercises refusal before startup and the exact app-server effort wire."""
    import json
    from agent.delegation_launch import DelegationLaunchPolicy
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from tests.fakes.providers.codex_app_server import FakeCodex
    from tools.delegate_tool import delegate_task

    home = tmp_path / "codex-profile" / ".hermes"
    home.mkdir(parents=True)
    fake = FakeCodex(tmp_path, [{"steps": [{"kind": "message", "text": "actual app-server child completed"}]}])
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-openai-key")
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    (home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": "openai-codex", "default": "gpt-6-luna",
                 "openai_runtime": "codex_app_server", "codex_bin": str(fake.bin)},
        "delegation": {"fallback_providers": []},
        "agent": {"api_max_retries": 1},
    }), encoding="utf-8")
    db = SessionDB(db_path=home / "state.db")
    db.create_session("parent-session", source="cli")
    parent = None
    try:
        parent = AIAgent(
            api_key="fixture-openai-key", base_url="https://api.openai.com/v1", provider="openai-codex",
            requested_provider="openai-codex", api_mode="codex_app_server", model="gpt-6-luna",
            platform="cli", quiet_mode=True, skip_context_files=True, skip_memory=True,
            save_trajectories=False, enabled_toolsets=["file", "delegation"],
            session_id="parent-session", session_db=db,
        )

        for task in (
            {"goal": "Refuse an explicitly empty app-server grant", "tools": []},
            {"goal": "Refuse an explicitly narrow app-server grant", "tools": ["read_file"]},
        ):
            refused = json.loads(delegate_task(tasks=[task], parent_agent=parent))
            assert "error" in refused and "Codex app-server" in refused["error"], refused
            assert fake.spawned_pids() == []
            assert fake.requests("turn/start") == []

        parent.__dict__["_delegation_launch_policy"] = DelegationLaunchPolicy(
            tools=("read_file",), parent_tool_grant=("read_file",), admitted_tool_names=("read_file",))
        inherited_refusal = json.loads(delegate_task(
            tasks=[{"goal": "Refuse the inherited exact ceiling"}], parent_agent=parent))
        assert "error" in inherited_refusal
        assert "inherited exact tool ceiling" in inherited_refusal["error"]
        assert fake.spawned_pids() == []
        assert fake.requests("turn/start") == []

        parent.__dict__.pop("_delegation_launch_policy")
        result = json.loads(delegate_task(tasks=[{
            "goal": "Run one real app-server turn with exact ultra effort", "reasoning_effort": "ultra",
        }], parent_agent=parent))
        assert "results" in result, result
        (entry,) = result["results"]
        assert entry["status"] == "completed", entry
        assert entry["summary"] == "actual app-server child completed"
        assert entry["api_calls"] == 1
        assert len(fake.spawned_pids()) == 1
        starts = fake.requests("turn/start")
        assert len(starts) == 1
        params = starts[0]["msg"]["params"]
        assert params["model"] == "gpt-6-luna"
        assert params["effort"] == "ultra"
        fake.assert_wire_clean()
    finally:
        if parent is not None:
            parent.close()
        db.close()


def test_replayed_reasoning_content_rejection_uses_its_own_recovery():
    from agent.turn_recovery import recover_before_classification

    agent = SimpleNamespace(
        _strict_reasoning_effort=True, _buffer_vprint=lambda *_: None, _vprint=lambda *_a, **_kw: None,
        log_prefix="", _image_rejecting_models=set(), _reasoning_rejecting_routes={}, model="custom-model", provider="custom",
        base_url="http://relay.example/v1", api_mode="chat_completions", session_id=None,
    )
    replay = [{"role": "assistant", "reasoning_details": [{"type": "reasoning.text", "text": "old"}],
               "reasoning_content": "old"}]
    recovered, _ = recover_before_classification(
        agent, ValueError("Unknown field reasoning_content"), messages=[], api_messages=replay,
        api_kwargs={"messages": []}, active_system_prompt="prompt",
    )
    assert recovered is True
    assert "reasoning_details" in replay[0]
    assert "reasoning_content" in replay[0]
    assert not getattr(agent, "_strict_reasoning_effort_rejected", False)


def test_wire_builder_keeps_explicit_effort_when_legacy_rejection_and_ephemeral_flags_are_set():
    from agent.reasoning_params import _reasoning_config_for_wire

    config = {"enabled": True, "effort": "high"}
    agent = SimpleNamespace(
        _strict_reasoning_effort=True, reasoning_config=config,
        _reasoning_effort_rejected=True, _reasoning_disable_rejected=True,
        _ephemeral_reasoning_off=True, _reasoning_floor_required=True,
    )
    assert _reasoning_config_for_wire(agent) == config
    assert agent._wire_reasoning_config == config


def test_strict_rejected_explicit_effort_cannot_build_a_default_request():
    from agent.reasoning_params import _reasoning_config_for_wire

    agent = SimpleNamespace(
        _strict_reasoning_effort=True, _strict_reasoning_effort_rejected=True,
        reasoning_config={"enabled": True, "effort": "high"},
    )
    with pytest.raises(ValueError, match="already rejected"):
        _reasoning_config_for_wire(agent)


def test_strict_child_image_corrupt_recovery_is_not_blocked():
    from agent.error_classifier import classify_api_error
    from agent.turn_recovery import recover_after_classification
    from agent.turn_retry_state import TurnRetryState

    class ImageError(Exception):
        status_code = 400
        body = {"error": {"message": "Invalid PNG image."}}

    class RecoveryAgent(SimpleNamespace):
        pass

    image_messages = [{"role": "user", "content": [
        {"type": "text", "text": "inspect"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,broken"}},
    ]}]
    agent = RecoveryAgent(
        _strict_reasoning_effort=True, _strict_requested_reasoning_effort="high",
        _strict_reasoning_effort_rejected=False,
        _wire_reasoning_config={"enabled": True, "effort": "high"},
        provider="xai", model="grok-5", base_url="https://api.x.ai/v1", api_mode="chat_completions",
        log_prefix="", _image_rejecting_models=set(), _fallback_chain=[], _fallback_index=0,
        _credential_pool=None, _retry_state=TurnRetryState(),
        _recover_with_credential_pool=lambda **_: (False, False),
        _vprint=lambda *_args, **_kwargs: None, _buffer_vprint=lambda *_args: None,
        _try_shrink_image_parts_in_messages=lambda *_args, **_kwargs: False,
    )
    error = ImageError("corrupt image")
    classified = classify_api_error(error, provider=agent.provider, model=agent.model)
    retry, pool = recover_after_classification(
        agent, error, classified, agent._retry_state, status_code=400, error_context={},
        messages=image_messages, api_messages=image_messages,
    )
    assert retry is True and pool is False
    assert "image_url" not in str(image_messages)
    assert agent._wire_reasoning_config == {"enabled": True, "effort": "high"}
    assert agent._strict_reasoning_effort_rejected is False


def test_route_report_ignores_synthetic_diagnostics_but_reports_real_metadata():
    from unittest.mock import MagicMock

    from tools.delegate_tool_child_run import _route_report

    child = SimpleNamespace(
        _delegate_route={"backend": "fixture"}, provider="custom", model="fixture-model",
        reasoning_config={"enabled": True, "effort": "high"}, _delegate_route_diagnostics=MagicMock(),
    )
    assert "validation" not in _route_report(child)
    child._delegate_route_diagnostics = (("validation", "unverified"), ("warning", "catalog missing"))
    report = _route_report(child)
    assert report["validation"] == "unverified"
    assert report["diagnostic"] == "catalog missing"


def test_chat_profile_rejects_enabled_reasoning_with_disabled_final_state():
    from providers import get_provider_profile

    profile = get_provider_profile("custom")
    with pytest.raises(ValueError, match="disabled final request"):
        ChatCompletionsTransport().build_kwargs(
            "relay-model", [{"role": "user", "content": "go"}], None,
            base_url="http://127.0.0.1:8000/v1", provider_name="custom", provider_profile=profile,
            reasoning_config={"enabled": True}, strict_reasoning_effort=True,
            request_overrides={"extra_body": {"reasoning": {"enabled": False}}},
        )


def test_chat_profile_rejects_enabled_effort_with_final_disabled_carrier():
    from providers import get_provider_profile

    profile = get_provider_profile("custom")
    payload_overrides = {"extra_body": {"reasoning": {"enabled": False, "effort": "high"}}}
    with pytest.raises(ValueError, match="preserve"):
        ChatCompletionsTransport().build_kwargs(
            "relay-model", [{"role": "user", "content": "go"}], None,
            base_url="http://127.0.0.1:8000/v1", provider_name="custom", provider_profile=profile,
            reasoning_config={"enabled": True, "effort": "high"}, strict_reasoning_effort=True,
            request_overrides=payload_overrides,
        )


@pytest.mark.parametrize("route", ["kimi", "tokenhub"])
def test_unregistered_reasoning_route_rejects_overridden_effort_carrier(route):
    route_params = {
        "kimi": {"model": "kimi-k3", "provider_name": "unknown-kimi", "is_kimi": True},
        "tokenhub": {"model": "tokenhub-model", "provider_name": "unknown-tokenhub", "is_tokenhub": True},
    }[route]
    with pytest.raises(ValueError, match="final request"):
        ChatCompletionsTransport().build_kwargs(
            route_params["model"], [{"role": "user", "content": "go"}], None,
            provider_name=route_params["provider_name"], is_kimi=route_params.get("is_kimi", False),
            is_tokenhub=route_params.get("is_tokenhub", False), supports_reasoning=False,
            reasoning_config={"enabled": True, "effort": "high"}, strict_reasoning_effort=True,
            request_overrides={"extra_body": {"reasoning_effort": "low"}},
        )


def test_chat_profile_rejects_none_when_final_extra_body_reenables_reasoning():
    from providers import get_provider_profile

    profile = get_provider_profile("custom")
    with pytest.raises(ValueError, match="cannot be carried"):
        ChatCompletionsTransport().build_kwargs(
            "relay-model", [{"role": "user", "content": "go"}], None,
            base_url="http://127.0.0.1:8000/v1", provider_name="custom", provider_profile=profile,
            reasoning_config={"enabled": False, "effort": "none"}, strict_reasoning_effort=True,
            request_overrides={"extra_body": {"reasoning": {"enabled": True, "effort": "high"}}},
        )


def test_gemini_hidden_thoughts_keeps_exact_enabled_level():
    from providers import get_provider_profile

    profile = get_provider_profile("gemini")
    payload = ChatCompletionsTransport().build_kwargs(
        "gemini-3.8-flash", [{"role": "user", "content": "go"}], None,
        provider_name="gemini", provider_profile=profile, base_url=profile.base_url,
        reasoning_config={"enabled": True, "effort": "high"}, strict_reasoning_effort=True,
        request_overrides={"extra_body": {"thinking_config": {"includeThoughts": False, "thinkingLevel": "high"}}},
    )
    assert payload["extra_body"]["thinking_config"] == {"includeThoughts": False, "thinkingLevel": "high"}


def test_gemini_openai_compat_hidden_thoughts_keeps_exact_enabled_level():
    from providers import get_provider_profile

    profile = get_provider_profile("gemini")
    payload = ChatCompletionsTransport().build_kwargs(
        "gemini-3.8-flash", [{"role": "user", "content": "go"}], None,
        provider_name="gemini", provider_profile=profile,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        reasoning_config={"enabled": True, "effort": "high"}, strict_reasoning_effort=True,
        request_overrides={"extra_body": {"extra_body": {"google": {"thinking_config": {
            "include_thoughts": False, "thinking_level": "high",
        }}}}},
    )
    assert payload["extra_body"]["extra_body"]["google"]["thinking_config"] == {
        "include_thoughts": False, "thinking_level": "high",
    }


def test_gemini_hidden_thoughts_alone_cannot_prove_reasoning_disabled():
    from providers import get_provider_profile

    profile = get_provider_profile("gemini")
    with pytest.raises(ValueError, match="cannot be carried"):
        ChatCompletionsTransport().build_kwargs(
            "gemini-2.5-flash", [{"role": "user", "content": "go"}], None,
            provider_name="gemini", provider_profile=profile, base_url=profile.base_url,
            reasoning_config={"enabled": False, "effort": "none"}, strict_reasoning_effort=True,
            request_overrides={"extra_body": {"thinking_config": {"includeThoughts": False}}},
        )


def test_gemini_none_uses_actual_budget_zero_only_where_supported():
    from providers import get_provider_profile

    profile = get_provider_profile("gemini")
    payload = ChatCompletionsTransport().build_kwargs(
        "gemini-2.5-flash", [{"role": "user", "content": "go"}], None,
        provider_name="gemini", provider_profile=profile, base_url=profile.base_url,
        reasoning_config={"enabled": False, "effort": "none"}, strict_reasoning_effort=True,
    )
    assert payload["extra_body"]["thinking_config"] == {"includeThoughts": False, "thinkingBudget": 0}


def test_gemini_low_thinking_level_does_not_claim_reasoning_is_disabled():
    from providers import get_provider_profile

    profile = get_provider_profile("gemini")
    with pytest.raises(ValueError, match="cannot be carried"):
        ChatCompletionsTransport().build_kwargs(
            "gemini-3.8-flash", [{"role": "user", "content": "go"}], None,
            provider_name="gemini", provider_profile=profile, base_url=profile.base_url,
            reasoning_config={"enabled": False, "effort": "none"}, strict_reasoning_effort=True,
        )


def test_classified_effort_rejection_is_terminal_for_strict_child():
    from agent.error_classifier import FailoverReason, classify_api_error
    from agent.turn_api_error import handle_api_error
    from agent.turn_retry_state import TurnRetryState

    class ProviderError(Exception):
        status_code = 400
        body = {"error": {"param": "reasoning.effort", "error_code": "invalid_reasoning_effort", "retryable": False,
                           "message": "reasoning_effort unsupported"}}
        message = "reasoning_effort unsupported"

    class RecoveryAgent(SimpleNamespace):
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    agent = RecoveryAgent(
        _strict_reasoning_effort=True, _strict_requested_reasoning_effort="max",
        _strict_reasoning_effort_rejected=False, _wire_reasoning_config={"enabled": True, "effort": "max"},
        _recover_with_credential_pool=lambda **_: (False, False), _vprint=lambda *_args, **_kwargs: None,
        _buffer_vprint=lambda *_: None, log_prefix="", provider="custom", model="relay-model",
        base_url="http://relay.example/v1", api_mode="chat_completions", thinking_callback=None,
        _image_rejecting_models=set(), _fallback_chain=[], _fallback_index=0, _credential_pool=None,
        _retry_state=TurnRetryState(),
    )
    error = ProviderError("provider rejected explicit effort")
    classified = classify_api_error(error, provider=agent.provider, model=agent.model)
    assert classified.reason == FailoverReason.reasoning_mandatory
    from agent.turn_recovery import recover_after_classification
    from agent.route_verdicts import route_visit
    assert recover_after_classification(
        agent, error, classified, agent._retry_state, status_code=400, error_context={}, messages=[], api_messages=[],
    )[0] is False
    assert agent._strict_reasoning_effort_rejected is True
    agent._retry_state.reasoning_mandatory_retry_visit = route_visit(agent)
    verdict = handle_api_error(
        agent, api_error=error, _retry=agent._retry_state, thinking_spinner=None, messages=[], api_messages=[],
        api_kwargs={"messages": []}, system_message="", active_system_prompt="", conversation_history=None,
        approx_tokens=10, retry_count=0, max_retries=1, compression_attempts=0, max_compression_attempts=0,
        api_call_count=1, api_request_id="req", api_start_time=0.0, effective_task_id=None, turn_id="turn",
    )
    assert verdict.action == "return"
    assert verdict.result is not None
    assert verdict.result["completed"] is False
    assert "final_response" in verdict.result
    assert agent._wire_reasoning_config is None
