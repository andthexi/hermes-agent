"""S1 route/profile/provider behaviors at real option and route boundaries."""

import json
from hermes_constants import get_hermes_home
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from unittest.mock import patch

import hermes_yaml as yaml

from tools.delegate_tool_options import _profile_default_model, _provider_credentials
from tools.delegate_tool_routes import TaskRoute
from tools.delegate_tool_validation import _catalog_verdict, resolve_task_options, resolve_task_routes


class Parent:
    model = "parent-switched-model"
    provider = "custom"
    api_mode = "chat_completions"
    reasoning_config = {"enabled": True, "effort": "medium"}
    _fallback_chain = [{"provider": "custom", "model": "launch-fallback"}]


def test_real_profile_config_scope_selects_provider_and_alias_credentials(tmp_path, monkeypatch):
    from tools.delegate_tool_validation import resolve_task_options, resolve_task_routes

    root = tmp_path / "os" / "hermes-test-home"
    root.mkdir(parents=True)
    (root / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": "custom", "default": "parent-model", "base_url": "http://127.0.0.1:9710/v1"},
        "model_aliases": {"local": {"model": "exact-alias-model", "base_url": "http://127.0.0.1:9711/v1"}},
    }), encoding="utf-8")
    coder = root / "profiles" / "coder"
    coder.mkdir(parents=True)
    (coder / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": "custom", "default": "coder-saved-model", "base_url": "http://127.0.0.1:9712/v1"},
        "model_aliases": {"local": {"model": "exact-alias-model", "base_url": "http://127.0.0.1:9713/v1",
                                       "key_env": "ALIAS_KEY"}},
    }), encoding="utf-8")
    (coder / ".env").write_text("ALIAS_KEY=coder-alias-key\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path / "os"))
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("ALIAS_KEY", "launch-profile-key")
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name == "coder")
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: coder)

    def resolve(**kwargs):
        return {"model": kwargs.get("target_model"), "provider": "custom", "requested_provider": "custom",
                "base_url": kwargs.get("explicit_base_url") or "http://127.0.0.1:9712/v1",
                "api_key": kwargs.get("explicit_api_key") or "no-key-required", "api_mode": "chat_completions",
                "request_overrides": {}, "command": None, "args": []}

    from hermes_cli.runtime_provider import _FROZEN_RUNTIME_CONFIG
    from agent.secret_scope import get_secret
    from hermes_cli.config import load_config_readonly

    parent = Parent()
    base = TaskRoute({"model": "parent-model", "provider": "custom", "base_url": "http://127.0.0.1:9710/v1",
                      "api_key": "parent-key", "api_mode": "chat_completions"}, {})
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", resolve)
    monkeypatch.setattr("tools.delegate_tool_validation._catalog_verdict", lambda *a, **k: (False, "", True))

    tasks = [{"goal": "use selected profile alias", "profile": "coder", "model": "local"}]
    options, errors = resolve_task_options(tasks)
    (route,) = resolve_task_routes(tasks, {}, parent, [base], options, errors)

    assert route.error is None
    assert (route.creds["model"], route.creds["base_url"], route.creds["api_key"]) == (
        "exact-alias-model", "http://127.0.0.1:9713/v1", "coder-alias-key")
    assert _FROZEN_RUNTIME_CONFIG.get() is None
    assert str(load_config_readonly()["model"]["default"]) == "parent-model"
    assert get_secret("ALIAS_KEY") == "launch-profile-key"


def test_runtime_profile_scope_selects_real_alias_credentials_and_resets_scope(tmp_path, monkeypatch):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from agent.credential_pool import custom_provider_pool_key_candidates
    from hermes_cli.runtime_provider import _FROZEN_RUNTIME_CONFIG
    from agent.secret_scope import get_secret
    from hermes_cli.config import load_config_readonly
    from tools.delegate_tool_validation import resolve_task_options, resolve_task_routes

    class CatalogHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"data": [{"id": "exact-alias-model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), CatalogHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        root = tmp_path / "os" / "hermes-test-home"
        root.mkdir(parents=True)
        monkeypatch.setenv("HOME", str(tmp_path / "os"))
        monkeypatch.setenv("HERMES_HOME", str(root))
        parent_url = f"http://127.0.0.1:{server.server_port}/parent/v1"
        alias_url = f"http://127.0.0.1:{server.server_port}/alias/v1"
        coder = root / "profiles" / "coder"
        coder.mkdir(parents=True)
        (root / "config.yaml").write_text(yaml.safe_dump({
            "model": {"provider": "custom", "default": "parent-model", "base_url": parent_url},
            "custom_providers": [{"name": "parent-relay", "base_url": parent_url, "api_key": "parent-route-key"}],
        }), encoding="utf-8")
        (coder / "config.yaml").write_text(yaml.safe_dump({
            "model": {"provider": "custom", "default": "coder-saved-model", "base_url": parent_url},
            "model_aliases": {"local": {"model": "exact-alias-model", "base_url": alias_url,
                                           "key_env": "ALIAS_KEY"}},
        }), encoding="utf-8")
        (coder / ".env").write_text("ALIAS_KEY=coder-alias-key\n", encoding="utf-8")
        monkeypatch.setenv("ALIAS_KEY", "launch-profile-key")
        monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name == "coder")
        monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: coder)

        class Parent:
            model, provider, base_url, api_key = "switched-model", "custom", parent_url, "launch-route-key"
            reasoning_config = {"enabled": True, "effort": "medium"}

        base = TaskRoute({"model": "backend-model", "provider": "custom", "base_url": parent_url,
                          "api_key": "backend-key", "api_mode": "chat_completions"}, {})
        tasks = [{"goal": "use target-profile alias", "profile": "coder", "model": "local"}]
        options, errors = resolve_task_options(tasks)
        (route,) = resolve_task_routes(tasks, {}, Parent(), [base], options, errors)

        assert route.error is None
        assert (route.creds["model"], route.creds["base_url"], route.creds["api_key"]) == (
            "exact-alias-model", alias_url, "coder-alias-key")
        assert route.launch_policy is not None
        assert dict(route.launch_policy.diagnostics)["validation"] == "verified"
        assert route.routing_cfg.get("fallback_providers") == []
        assert load_config_readonly()["model"]["default"] == "parent-model"
        assert get_secret("ALIAS_KEY") == "launch-profile-key"
        assert _FROZEN_RUNTIME_CONFIG.get() is None
        assert custom_provider_pool_key_candidates(route.creds["base_url"]) != \
            custom_provider_pool_key_candidates(parent_url)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_dispatch_reports_unverified_model_pair_and_runs_exact_route(tmp_path, monkeypatch):
    from tools.delegate_tool import _routes_for
    from tools.delegate_tool_validation import _catalog_verdict

    root = tmp_path / "os" / "hermes-test-home"
    root.mkdir(parents=True)
    endpoint = "http://127.0.0.1:9721/v1"
    (root / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": "custom", "default": "saved-model", "base_url": endpoint},
        "delegation": {"fallback_providers": [{"provider": "custom", "model": "fallback-model"}]},
    }), encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path / "os"))
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr("tools.delegate_tool_validation._catalog_verdict",
                        lambda *args, **kwargs: (False, "fixture catalog unavailable", False))

    class Parent:
        model, provider, base_url, api_key = "switched-model", "custom", endpoint, "parent-key"

    tasks = [{"goal": "exact unknown model", "model": "unlisted-exact-id"}]
    (route,), error = _routes_for(tasks, {}, None, Parent())

    assert error is None and route.error is None
    assert route.creds["model"] == "unlisted-exact-id"
    assert route.routing_cfg["fallback_providers"] == []
    from tools.delegate_tool_options import TaskRouteOptions
    route_error = TaskRouteOptions.parse({"goal": "", "model": "fixture"}, 0)[1]
    assert route_error is None
    diagnostics = dict(route.launch_policy.diagnostics)
    assert diagnostics["validation"] == "unverified"
    assert "attempting the exact selection" in diagnostics["warning"]


def test_dispatch_preserves_unverified_reasoning_none_in_route_and_policy(tmp_path, monkeypatch):
    from tools.delegate_tool import _routes_for

    root = tmp_path / "os" / "hermes-test-home"
    root.mkdir(parents=True)
    endpoint = "http://127.0.0.1:9722/v1"
    (root / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": "custom", "default": "saved-model", "base_url": endpoint},
    }), encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path / "os"))
    monkeypatch.setenv("HERMES_HOME", str(root))
    profile = type("UnknownProfile", (), {"supported_reasoning_efforts": lambda self, model: None})()
    monkeypatch.setattr("providers.get_provider_profile", lambda provider: profile)
    monkeypatch.setattr("providers.profile_for_host", lambda host: None)
    monkeypatch.setattr("agent.models_dev.get_model_capabilities", lambda *a, **k: None)
    monkeypatch.setattr("tools.delegate_tool_validation._catalog_verdict", lambda *a, **k: (False, "", True))

    class SelectedParent:
        model, provider, base_url, api_key = "switched-model", "custom", endpoint, "fixture-parent-key"
        reasoning_config = {"enabled": True, "effort": "medium"}

    from tools.delegate_tool_routes import TaskRoute
    tasks = [{"goal": "disable unknown reasoning", "reasoning_effort": "none"}]
    base = TaskRoute({"model": "saved-model", "provider": "custom", "base_url": endpoint,
                      "api_key": "fixture-parent-key", "api_mode": "chat_completions"}, {})
    from tools.delegate_tool_validation import resolve_task_options, resolve_task_routes
    options, option_errors = resolve_task_options(tasks)
    (route,) = resolve_task_routes(tasks, {}, SelectedParent(), [base], options, option_errors)
    assert route.error is None
    diagnostics = dict(route.launch_policy.diagnostics)
    assert diagnostics["validation"] == "unverified"
    assert "support is unknown" in diagnostics["warning"]
    assert route.diagnostics == diagnostics
    assert route.report()["validation"] == "unverified"
    assert "support is unknown" in route.report()["diagnostic"]


def test_profile_only_pin_carries_requested_and_resolved_route_report(tmp_path, monkeypatch):
    from tools.delegate_tool_routes import TaskRoute

    root = tmp_path / "os" / "hermes-test-home"
    root.mkdir(parents=True)
    coder = root / "profiles" / "coder"
    coder.mkdir(parents=True)
    endpoint = "http://127.0.0.1:9723/v1"
    (root / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": "custom", "default": "launch-model", "base_url": endpoint},
    }), encoding="utf-8")
    (coder / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": "custom", "default": "coder-saved-model", "base_url": endpoint,
                   "key_env": "CODER_PROFILE_KEY"},
    }), encoding="utf-8")
    (coder / ".env").write_text("CODER_PROFILE_KEY=coder-profile-key\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path / "os"))
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("CODER_PROFILE_KEY", "launch-environment-key")
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name == "coder")
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: coder)
    monkeypatch.setattr("tools.delegate_tool_validation._catalog_verdict", lambda *a, **k: (False, "", True))

    class SelectedParent:
        model, provider, base_url, api_key = "switched-model", "custom", endpoint, "launch-route-key"
        reasoning_config = {"enabled": True, "effort": "medium"}

    tasks = [{"goal": "run as coder", "profile": "coder"}]
    base = TaskRoute({"model": "launch-model", "provider": "custom", "base_url": endpoint,
                      "api_key": "launch-route-key", "api_mode": "chat_completions"}, {})
    from tools.delegate_tool_validation import resolve_task_options, resolve_task_routes
    options, option_errors = resolve_task_options(tasks)
    (route,) = resolve_task_routes(tasks, {}, SelectedParent(), [base], options, option_errors)
    assert route.error is None
    assert route.creds["model"] == "coder-saved-model"
    assert route.creds["api_key"] == "coder-profile-key"
    assert route.diagnostics == {
        "requested_provider": None, "requested_model": None, "requested_profile": "coder",
        "resolved_provider": "custom", "resolved_model": "coder-saved-model", "validation": "verified",
    }
    assert route.launch_policy is not None
    assert dict(route.launch_policy.diagnostics) == route.diagnostics
    assert route.report() == {"backend": "default", **route.diagnostics}


def test_delegate_task_runs_real_profile_child_on_loopback_and_reports_exact_pin(tmp_path, monkeypatch):
    from tools.delegate_tool_options import TaskRouteOptions
    from tools.delegate_tool_validation import resolve_task_options

    malformed = [{"goal": "bad", "tools": [" file_read"]}]
    options, errors = resolve_task_options(malformed)
    assert errors[0] and "surrounding whitespace" in errors[0]
    assert options[0].tools is None
    valid, valid_error = TaskRouteOptions.parse({"goal": "good", "tools": ["file_read", "file_read"]}, 0)
    assert valid_error is None and valid is not None and valid.tools == ("file_read",)

    from agent.secret_scope import get_secret
    from hermes_cli.config import load_config_readonly
    from hermes_state import SessionDB
    from run_agent import AIAgent

    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    requests = []

    class InferenceHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"data": [{"id": "coder-inference-model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, self.headers.get("Authorization"), payload))
            response = {"id": "fixture-chatcmpl", "object": "chat.completion", "created": 1,
                        "model": payload.get("model"),
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": "child finished"},
                                     "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12}}
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream" if payload.get("stream") else "application/json")
            self.end_headers()
            if payload.get("stream"):
                chunks = [
                    {"id": response["id"], "object": "chat.completion.chunk", "created": 1,
                     "model": payload.get("model"), "choices": [{"index": 0,
                         "delta": {"role": "assistant", "content": "child finished"}, "finish_reason": None}]},
                    {"id": response["id"], "object": "chat.completion.chunk", "created": 1,
                     "model": payload.get("model"), "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                ]
                for chunk in chunks:
                    self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                self.wfile.write(json.dumps(response).encode())

        def log_message(self, format, *args):
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), InferenceHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    agents = []
    observed_launch = []
    observed_policies = {}
    db = None
    try:
        host_home = tmp_path / "os"
        root = tmp_path / "hermes-root"
        (root / "profiles").mkdir(parents=True)
        coder = root / "profiles" / "coder"
        coder.mkdir(parents=True)
        search_skill = root / "skills" / "search"
        search_skill.mkdir(parents=True)
        (search_skill / "SKILL.md").write_text(
            "---\nname: search\ndescription: real selected-profile launch fixture\n---\n"
            "Search skill fixture for delegated preload policy.\n",
            encoding="utf-8",
        )
        endpoint = f"http://127.0.0.1:{server.server_port}/v1"
        launch_config = {"model": {"provider": "custom", "default": "launch-saved-model", "base_url": endpoint,
                                   "key_env": "LAUNCH_PROFILE_KEY"},
                         "tools": {"tool_search": {"enabled": "on", "defer": ["todo_list"]}},
                         "delegation": {"fallback_providers": [{"provider": "custom", "model": "launch-fallback"}]}}
        coder_config = {"model": {"provider": "custom", "default": "coder-inference-model", "base_url": endpoint,
                                  "key_env": "CODER_PROFILE_KEY"}}
        (root / "config.yaml").write_text(yaml.safe_dump(launch_config), encoding="utf-8")
        (root / ".env").write_text("LAUNCH_PROFILE_KEY=launch-config-key\n", encoding="utf-8")
        (coder / "config.yaml").write_text(yaml.safe_dump(coder_config), encoding="utf-8")
        (coder / ".env").write_text("CODER_PROFILE_KEY=coder-config-key\n", encoding="utf-8")
        (tmp_path / "config.yaml").write_text(yaml.safe_dump(launch_config), encoding="utf-8")
        launch_config_bytes = (root / "config.yaml").read_bytes()
        monkeypatch.setenv("HOME", str(host_home))
        monkeypatch.setenv("HERMES_HOME", str(root))
        monkeypatch.setenv("LAUNCH_PROFILE_KEY", "launch-process-key")
        monkeypatch.setenv("CODER_PROFILE_KEY", "launch-process-key-must-not-leak")
        from tools.delegate_tool import delegate_task as real_delegate_task
        monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name == "coder")
        monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: coder)
        monkeypatch.setattr("tools.delegate_tool_config._load_config", lambda: launch_config["delegation"])
        import tools.delegate_tool_config as delegation_config
        monkeypatch.setattr(delegation_config, "_load_config", lambda: launch_config["delegation"])
        monkeypatch.setattr("tools.delegate_tool._create_isolated_worktree", lambda *a, **k: None, raising=False)
        db = SessionDB(db_path=get_hermes_home() / "state.db")
        parent = AIAgent(api_key="launch-parent-key", base_url=endpoint, provider="custom", api_mode="chat_completions",
                         model="launch-switched-model", platform="cli", quiet_mode=True,
                         skip_context_files=True, skip_memory=True, save_trajectories=False,
                         enabled_toolsets=["delegation", "todo"], session_id="s1-loopback-parent", session_db=db)
        agents.append(parent)
        parent._ensure_db_session()
        # A malformed sibling is refused in preflight; valid siblings use actual AIAgents.
        from agent import agent_init
        real_load_tools = agent_init._load_tools

        tasks = [
            {"goal": "must not infer", "profile": "missing-profile"},
            {"goal": "reply with a short confirmation", "profile": "coder", "reasoning_effort": "low"},
            {"goal": "persona-only policy probe", "system_prompt": "persona-only-marker"},
            {"goal": "tools-only policy probe", "tools": []},
            {"goal": "skills-only policy probe", "skills": ["search"]},
        ]
        options, option_errors = resolve_task_options(tasks)
        from tools.delegate_tool_routes import route_for_config
        baseline_route = route_for_config(launch_config["delegation"], parent)
        (launch_route,) = resolve_task_routes(tasks[:1], launch_config["delegation"], parent,
                                              [baseline_route], options[:1], option_errors[:1])
        assert launch_route.launch_policy.parent_tool_grant is not None
        assert "todo_list" in launch_route.launch_policy.parent_tool_grant
        assert "file_read" not in launch_route.launch_policy.parent_tool_grant
        assert "todo_list" not in parent.valid_tool_names
        assert "file_read" not in parent.valid_tool_names

        def observe_early_init(agent, enabled_toolsets, disabled_toolsets):
            policy = getattr(agent, "_delegation_launch_policy", None)
            if policy is not None:
                observed_launch.append((agent, policy))
                observed_policies[policy.system_prompt or ("tools" if policy.route.tools_set else "skills")] = policy
            return real_load_tools(agent, enabled_toolsets, disabled_toolsets)

        monkeypatch.setattr(agent_init, "_load_tools", observe_early_init)
        output = json.loads(real_delegate_task(tasks=tasks, parent_agent=parent))
        assert [item["status"] for item in output["results"]] == ["error", "completed", "completed", "completed", "completed"]
        expected_goals = ["must not infer", "reply with a short confirmation", "persona-only policy probe",
                          "tools-only policy probe", "skills-only policy probe"]
        assert len(output["results"]) == len(expected_goals)
        results_by_goal = dict(zip(expected_goals, output["results"]))
        assert set(results_by_goal) == {"must not infer", "reply with a short confirmation", "persona-only policy probe",
                                        "tools-only policy probe", "skills-only policy probe"}
        refused = results_by_goal["must not infer"]
        completed = results_by_goal["reply with a short confirmation"]
        assert refused["api_calls"] == 0 and "missing-profile" in refused["error"]
        assert completed["summary"] == "child finished"
        assert completed["route"]["requested_profile"] == "coder"
        assert completed["route"]["requested_model"] is None
        assert completed["route"]["resolved_model"] == "coder-inference-model"
        assert completed["route"]["resolved_provider"] == "custom"
        assert completed["route"]["validation"] == "verified"
        assert completed["route"]["reasoning_effort"] == "low"
        child, early_policy = next(pair for pair in observed_launch if pair[1].reasoning_effort == "low")
        assert child._delegation_launch_policy == early_policy
        persona_policy = observed_policies["persona-only-marker"]
        tools_policy = observed_policies["tools"]
        skills_policy = observed_policies["skills"]
        assert persona_policy.system_prompt_mode == "replace"
        assert tools_policy.tools == ()
        assert skills_policy.skills == ("search",)
        for policy in (persona_policy, tools_policy, skills_policy):
            child_for_policy, observed_policy = next(
                (agent, early) for agent, early in observed_launch
                if early.system_prompt == policy.system_prompt and early.route.tools_set == policy.route.tools_set
                and early.route.skills_set == policy.route.skills_set
            )
            assert child_for_policy._delegation_launch_policy == observed_policy
            assert child_for_policy._delegation_launch_policy.route == policy.route
        assert early_policy.profile_home == str(coder)
        assert child._delegate_profile_home == str(coder)
        from tools.delegate_tool_progress import _resolve_workspace_hint
        assert early_policy.workspace == _resolve_workspace_hint(parent)
        assert early_policy.parent_tool_grant is not None
        assert early_policy.diagnostics
        assert child._strict_reasoning_effort is True
        assert child._delegate_route_diagnostics
        assert dict(early_policy.diagnostics) == child._delegate_route_diagnostics
        assert all(results_by_goal[goal]["api_calls"] == 1 for goal in (
            "reply with a short confirmation", "persona-only policy probe", "tools-only policy probe",
            "skills-only policy probe"))
        inference_requests = [request for request in requests if request[0] == "/v1/chat/completions"]
        assert len(inference_requests) == 4
        launch_requests = [request for request in inference_requests if request[2]["model"] == "launch-switched-model"
                           and request[1] == "Bearer launch-parent-key"]
        assert len(launch_requests) == 3
        profile_requests = [request for request in inference_requests if request[2]["model"] == "coder-inference-model"
                            and request[1] == "Bearer coder-config-key"]
        assert len(profile_requests) == 1
        _, _, payload = profile_requests[0]
        assert payload["reasoning_effort"] == "low"
        from hermes_constants import set_hermes_home_override, reset_hermes_home_override
        from hermes_cli.config import load_config_readonly
        parent_home_token = set_hermes_home_override(root)
        try:
            parent_config = load_config_readonly()
        finally:
            reset_hermes_home_override(parent_home_token)
        assert parent_config["model"]["default"] == "launch-saved-model"
        assert (root / "config.yaml").read_bytes() == launch_config_bytes
        assert get_secret("CODER_PROFILE_KEY") == "launch-process-key-must-not-leak"
        assert get_secret("LAUNCH_PROFILE_KEY") == "launch-process-key"
        serialized = json.dumps(completed)
        assert "coder-config-key" not in serialized and "launch-process-key" not in serialized
    finally:
        for agent in reversed(agents):
            agent.close()
        if db is not None:
            db.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_delegate_task_provider_only_uses_profile_saved_alias_in_real_child(tmp_path, monkeypatch):
    """A provider pin replaces the selected profile's saved alias before old auth is needed."""
    from agent.secret_scope import get_secret
    from hermes_cli.config import load_config_readonly
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from tools.delegate_tool import delegate_task

    requests = []

    class InferenceHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"data": [{"id": "profile-alias-model"},
                                         {"id": "explicit-profile-model"},
                                         {"id": "steady-profile-model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, self.headers.get("Authorization"), payload))
            response = {"id": "fixture-chatcmpl", "object": "chat.completion", "created": 1,
                        "model": payload.get("model"),
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": "child finished"},
                                     "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12}}
            self.send_response(200)
            if payload.get("stream"):
                self.send_header("Content-Type", "text/event-stream")
            else:
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(json.dumps(response))))
            self.end_headers()
            if payload.get("stream"):
                chunks = [
                    {"id": response["id"], "object": "chat.completion.chunk", "created": 1,
                     "model": payload.get("model"), "choices": [{"index": 0,
                         "delta": {"role": "assistant", "content": "child finished"}, "finish_reason": None}]},
                    {"id": response["id"], "object": "chat.completion.chunk", "created": 1,
                     "model": payload.get("model"), "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                ]
                for chunk in chunks:
                    self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                self.wfile.write(json.dumps(response).encode())

        def log_message(self, format, *args):
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), InferenceHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    agents = []
    db = None
    try:
        host_home = tmp_path / "os"
        root = host_home / "hermes-test-home"
        coder = root / "profiles" / "coder"
        coder.mkdir(parents=True)
        endpoint = f"http://127.0.0.1:{server.server_port}/v1"
        launch_config = {"model": {"provider": "custom", "default": "launch-model", "base_url": endpoint}}
        coder_config = {
            "model": {"provider": "openai", "default": "legacy-alias",
                      "aliases": {"legacy-alias": "openai/profile-alias-model"}},
            "providers": {"openai": {"api": endpoint, "key_env": "MISSING_OLD_ALIAS_KEY"},
                          "target-relay": {"api": endpoint, "key_env": "TARGET_RELAY_KEY"}},
        }
        (root / "config.yaml").write_text(yaml.safe_dump(launch_config), encoding="utf-8")
        (coder / "config.yaml").write_text(yaml.safe_dump(coder_config), encoding="utf-8")
        (coder / ".env").write_text("TARGET_RELAY_KEY=profile-target-key\n", encoding="utf-8")
        monkeypatch.setenv("HOME", str(host_home))
        monkeypatch.setenv("HERMES_HOME", str(root))
        monkeypatch.delenv("MISSING_OLD_ALIAS_KEY", raising=False)
        monkeypatch.delenv("TARGET_RELAY_KEY", raising=False)
        monkeypatch.setattr("tools.delegate_tool_config._load_config", lambda: {"max_iterations": 2, "fallback_providers": []})
        import tools.delegate_tool_config as delegation_config
        monkeypatch.setattr(delegation_config, "_load_config", lambda: {"max_iterations": 2, "fallback_providers": []})

        db = SessionDB(db_path=get_hermes_home() / "state.db")
        parent = AIAgent(api_key="launch-parent-key", base_url=endpoint, provider="custom", api_mode="chat_completions",
                         model="launch-switched-model", platform="cli", quiet_mode=True,
                         skip_context_files=True, skip_memory=True, save_trajectories=False,
                         enabled_toolsets=["delegation"], session_id="s1-provider-priority", session_db=db)
        agents.append(parent)
        parent._ensure_db_session()

        output = json.loads(delegate_task(tasks=[{
            "goal": "reply with a short confirmation", "profile": "coder", "provider": "target-relay",
        }], parent_agent=parent))

        (completed,) = output["results"]
        assert completed["status"] == "completed"
        assert completed["summary"] == "child finished"
        assert completed["route"]["resolved_model"] == "profile-alias-model"
        assert completed["route"]["resolved_provider"] == "target-relay"
        inference_requests = [request for request in requests if request[0] == "/v1/chat/completions"]
        assert len(inference_requests) == 1
        _, authorization, payload = inference_requests[0]
        assert authorization == "Bearer profile-target-key"
        assert payload["model"] == "profile-alias-model"
        assert load_config_readonly()["model"]["default"] == "launch-model"
        assert get_secret("TARGET_RELAY_KEY") is None
        assert all(auth != "Bearer MISSING_OLD_ALIAS_KEY" for _, auth, _ in requests)
    finally:
        for agent in reversed(agents):
            agent.close()
        if db is not None:
            db.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_delegate_task_explicit_model_provider_uses_profile_credentials_in_real_child(tmp_path, monkeypatch):
    """An explicit model/provider pair is resolved inside the selected profile and used on wire."""
    from agent.secret_scope import get_secret
    from hermes_cli.config import load_config_readonly
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from tools.delegate_tool import delegate_task

    requests = []

    class InferenceHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"data": [{"id": "explicit-profile-model"},
                                         {"id": "direct-profile-model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, self.headers.get("Authorization"), payload))
            response = {"id": "fixture-chatcmpl", "object": "chat.completion", "created": 1,
                        "model": payload.get("model"),
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": "explicit route finished"},
                                     "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11}}
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunks = [
                {"id": response["id"], "object": "chat.completion.chunk", "created": 1,
                 "model": payload.get("model"), "choices": [{"index": 0,
                     "delta": {"role": "assistant", "content": "explicit route finished"}, "finish_reason": None}]},
                {"id": response["id"], "object": "chat.completion.chunk", "created": 1,
                 "model": payload.get("model"), "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            ]
            for chunk in chunks:
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")

        def log_message(self, format, *args):
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), InferenceHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    agents = []
    db = None
    try:
        host_home = tmp_path / "os"
        root = host_home / "hermes-test-home"
        coder = root / "profiles" / "coder"
        empty = root / "profiles" / "empty"
        coder.mkdir(parents=True)
        empty.mkdir(parents=True)
        endpoint = f"http://127.0.0.1:{server.server_port}/v1"
        launch_config = {
            "model": {"provider": "custom", "default": "launch-model", "base_url": endpoint},
            "providers": {"target-relay": {"api": endpoint + "/target/v1", "key_env": "ROOT_TARGET_RELAY_KEY"}},
        }
        target_endpoint = endpoint + "/target/v1"
        coder_config = {
            "model": {"provider": "openai", "default": "steady-profile-model", "base_url": endpoint,
                      "context_length": 8192, "key_env": "MISSING_OLD_ALIAS_KEY"},
            "model_aliases": {"explicit-route-alias": {
                "model": "explicit-profile-model", "provider": "openai",
                "base_url": endpoint + "/old-provider/v1", "key_env": "MISSING_OLD_ALIAS_KEY",
            }},
            "providers": {"target-relay": {"api": target_endpoint, "key_env": "TARGET_RELAY_KEY"}},
        }
        (root / "config.yaml").write_text(yaml.safe_dump(launch_config), encoding="utf-8")
        (root / ".env").write_text("ROOT_TARGET_RELAY_KEY=root-target-key\n", encoding="utf-8")
        (coder / "config.yaml").write_text(yaml.safe_dump(coder_config), encoding="utf-8")
        (coder / ".env").write_text("TARGET_RELAY_KEY=profile-explicit-key\n", encoding="utf-8")
        (empty / "config.yaml").write_text("model: {}\n", encoding="utf-8")
        monkeypatch.setattr(Path, "home", lambda: host_home)
        monkeypatch.setenv("HOME", str(host_home))
        monkeypatch.setenv("HERMES_HOME", str(root))
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from hermes_constants import named_profile_has_identity
        assert named_profile_has_identity(coder)
        assert named_profile_has_identity(empty)
        monkeypatch.delenv("UNUSED_PROFILE_DEFAULT_KEY", raising=False)
        monkeypatch.delenv("MISSING_OLD_ALIAS_KEY", raising=False)
        monkeypatch.delenv("TARGET_RELAY_KEY", raising=False)
        monkeypatch.delenv("ROOT_TARGET_RELAY_KEY", raising=False)
        from hermes_cli.runtime_provider import resolve_runtime_provider as resolve_runtime_provider_impl
        provider_resolutions = []

        def observe_provider_resolution(**kwargs):
            provider_resolutions.append(kwargs)
            if kwargs.get("requested") == "openai" and kwargs.get("target_model") in {
                "explicit-profile-model", "direct-profile-model",
            }:
                raise ValueError("old provider credentials are unavailable")
            return resolve_runtime_provider_impl(**kwargs)

        monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", observe_provider_resolution)
        monkeypatch.setattr("model_tools._active_model_config", lambda: (None, {}))
        monkeypatch.setattr("tools.delegate_tool_config._load_config", lambda: {"max_iterations": 2, "fallback_providers": []})
        import tools.delegate_tool_config as delegation_config
        monkeypatch.setattr(delegation_config, "_load_config", lambda: {"max_iterations": 2, "fallback_providers": []})

        db = SessionDB(db_path=get_hermes_home() / "state.db")
        parent = AIAgent(api_key="launch-parent-key", base_url=endpoint, provider="custom", api_mode="chat_completions",
                         model="launch-switched-model", platform="cli", quiet_mode=True,
                         skip_context_files=True, skip_memory=True, save_trajectories=False,
                         enabled_toolsets=["delegation"], session_id="s1-explicit-route", session_db=db)
        agents.append(parent)
        parent._ensure_db_session()
        output = json.loads(delegate_task(tasks=[
            {"goal": "must not infer from sibling", "profile": "empty", "provider": "target-relay"},
            {"goal": "reply with a short confirmation", "profile": "coder",
             "model": "explicit-route-alias", "provider": "target-relay"},
            {"goal": "reply with a short confirmation", "profile": "coder",
             "model": "direct-profile-model", "provider": "target-relay"},
        ], parent_agent=parent))

        refused, completed, direct_completed = output["results"]
        assert refused["status"] == "error"
        assert refused["api_calls"] == 0
        assert "saved profile default model" in refused["error"]
        assert completed["status"] == "completed"
        assert completed["summary"] == "explicit route finished"
        assert completed["route"]["requested_model"] == "explicit-route-alias"
        assert completed["route"]["resolved_model"] == "explicit-profile-model"
        assert completed["route"]["resolved_provider"] == "target-relay"
        assert direct_completed["status"] == "completed", direct_completed
        assert direct_completed["summary"] == "explicit route finished"
        assert direct_completed["route"]["requested_model"] == "direct-profile-model"
        assert direct_completed["route"]["resolved_model"] == "direct-profile-model"
        assert direct_completed["route"]["resolved_provider"] == "target-relay"
        assert direct_completed["api_calls"] == 1
        inference_requests = [request for request in requests if request[0] == "/v1/target/v1/chat/completions"]
        assert len(inference_requests) == 2
        inference_by_model = {request[2]["model"]: request for request in inference_requests}
        _, authorization, payload = inference_by_model["explicit-profile-model"]
        assert authorization == "Bearer profile-explicit-key"
        assert payload["model"] == "explicit-profile-model"
        _, direct_authorization, direct_payload = inference_by_model["direct-profile-model"]
        assert direct_authorization == "Bearer profile-explicit-key"
        assert direct_payload["model"] == "direct-profile-model"
        serialized_result = json.dumps(output)
        assert "profile-explicit-key" not in serialized_result
        assert "launch-parent-key" not in serialized_result
        old_provider_resolutions = [call for call in provider_resolutions
                                    if call.get("requested") == "openai" and call.get("target_model") in {
                                        "explicit-profile-model", "direct-profile-model",
                                    }]
        target_provider_resolutions = [call for call in provider_resolutions if call.get("requested") == "target-relay"]
        assert old_provider_resolutions == []
        assert len(target_provider_resolutions) == 2
        assert target_provider_resolutions[0]["requested"] == "target-relay"
        assert target_provider_resolutions[1]["requested"] == "target-relay"
        old_provider_requests = [request for request in requests if "/old-provider/" in request[0]]
        assert old_provider_requests == []
        assert "MISSING_OLD_ALIAS_KEY" not in (coder / ".env").read_text(encoding="utf-8")
        assert load_config_readonly()["model"]["default"] == "launch-model"
        assert get_secret("TARGET_RELAY_KEY") is None
    finally:
        for agent in reversed(agents):
            agent.close()
        if db is not None:
            db.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_delegate_task_model_only_keeps_selected_profile_baseline_provider(tmp_path, monkeypatch):
    """A model override in an explicit profile does not borrow the parent's switched provider."""
    from agent.secret_scope import get_secret
    from hermes_cli.config import load_config_readonly
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from tools.delegate_tool import delegate_task

    requests = []

    class InferenceHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"data": [{"id": "steady-profile-model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, self.headers.get("Authorization"), payload))
            response = {"id": "fixture-chatcmpl", "object": "chat.completion", "created": 1,
                        "model": payload.get("model"),
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": "baseline route finished"},
                                     "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11}}
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunks = [
                {"id": response["id"], "object": "chat.completion.chunk", "created": 1,
                 "model": payload.get("model"), "choices": [{"index": 0,
                     "delta": {"role": "assistant", "content": "baseline route finished"}, "finish_reason": None}]},
                {"id": response["id"], "object": "chat.completion.chunk", "created": 1,
                 "model": payload.get("model"), "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            ]
            for chunk in chunks:
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")

        def log_message(self, format, *args):
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), InferenceHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    agents = []
    db = None
    try:
        host_home = tmp_path / "os"
        root = host_home / "hermes-test-home"
        coder = root / "profiles" / "coder"
        coder.mkdir(parents=True)
        endpoint = f"http://127.0.0.1:{server.server_port}/v1"
        launch_config = {"model": {"provider": "custom", "default": "launch-model", "base_url": endpoint}}
        coder_config = {
            "model": {"provider": "custom", "default": "steady-profile-model", "base_url": endpoint,
                      "key_env": "STEADY_PROFILE_KEY"},
        }
        (root / "config.yaml").write_text(yaml.safe_dump(launch_config), encoding="utf-8")
        (coder / "config.yaml").write_text(yaml.safe_dump(coder_config), encoding="utf-8")
        (coder / ".env").write_text("STEADY_PROFILE_KEY=steady-profile-key\n", encoding="utf-8")
        monkeypatch.setenv("HOME", str(host_home))
        monkeypatch.setenv("HERMES_HOME", str(root))
        monkeypatch.delenv("STEADY_PROFILE_KEY", raising=False)
        monkeypatch.setattr("tools.delegate_tool_config._load_config", lambda: {"max_iterations": 2, "fallback_providers": []})
        import tools.delegate_tool_config as delegation_config
        monkeypatch.setattr(delegation_config, "_load_config", lambda: {"max_iterations": 2, "fallback_providers": []})

        db = SessionDB(db_path=get_hermes_home() / "state.db")
        parent = AIAgent(api_key="launch-parent-key", base_url=endpoint, provider="custom", api_mode="chat_completions",
                         model="launch-switched-model", platform="cli", quiet_mode=True,
                         skip_context_files=True, skip_memory=True, save_trajectories=False,
                         enabled_toolsets=["delegation"], session_id="s1-model-only", session_db=db)
        agents.append(parent)
        parent._ensure_db_session()
        output = json.loads(delegate_task(tasks=[{
            "goal": "reply with a short confirmation", "profile": "coder", "model": "steady-profile-model",
        }], parent_agent=parent))

        (completed,) = output["results"]
        assert completed["status"] == "completed"
        assert completed["summary"] == "baseline route finished"
        assert completed["route"]["resolved_model"] == "steady-profile-model"
        assert completed["route"]["resolved_provider"] == "custom"
        inference_requests = [request for request in requests if request[0] == "/v1/chat/completions"]
        assert len(inference_requests) == 1
        _, authorization, payload = inference_requests[0]
        assert authorization == "Bearer steady-profile-key"
        assert payload["model"] == "steady-profile-model"
        assert load_config_readonly()["model"]["default"] == "launch-model"
    finally:
        for agent in reversed(agents):
            agent.close()
        if db is not None:
            db.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _schema_definition():
    from tools.registry import registry
    return registry.get_definitions({"delegate_task"})[0]["function"]


def _schema_task_props(definition):
    return definition["parameters"]["properties"]["tasks"]["items"]["properties"]


def _schema_recorded(monkeypatch):
    from tools import delegate_tool as dt
    import types

    built = []

    def build(**kwargs):
        built.append(kwargs)
        return types.SimpleNamespace(tool_progress_callback=None, model=kwargs.get("model"))

    monkeypatch.setattr(dt, "_build_child_preserving_parent_tools", build)
    monkeypatch.setattr(dt, "_run_single_child", lambda task_index, goal, child=None, parent_agent=None, **kw: {
        "task_index": task_index, "status": "completed", "summary": "ok", "api_calls": 1, "duration_seconds": 0})
    return built


def _schema_parent():
    import types
    return types.SimpleNamespace(session_id="schema-parent", model="parent-model", provider="openrouter",
                                 tool_progress_callback=None, _delegate_spinner=None, request_overrides={},
                                 _safe_print=lambda line: None)


def _temporary_schema_backend(tmp_path, monkeypatch, *, name, plugin_source):
    from hermes_cli import plugins as plugins_mod
    from hermes_cli.plugins import PluginManager

    home = tmp_path / "hermes-test-home"
    plugin_dir = home / "plugins" / name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(yaml.safe_dump({"name": name, "version": "1.0"}), encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(plugin_source, encoding="utf-8")
    marker = tmp_path / "route-called"
    (home / "config.yaml").write_text(yaml.safe_dump({
        "plugins": {"enabled": [name]},
        "delegation": {"backend": name, "backends": {name: {"marker": str(marker)}}},
    }), encoding="utf-8")
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    bundled.joinpath("delegation").symlink_to(Path(__file__).resolve().parents[2] / "plugins" / "delegation",
                                                 target_is_directory=True)
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: bundled)
    manager = PluginManager()
    manager.discover_and_load()
    return manager, marker


def test_incompatible_core_schema_is_visible_and_refuses_public_spawn(tmp_path, monkeypatch):
    from tools.delegate_tool import DELEGATE_TASK_SCHEMA, delegate_task

    built = _schema_recorded(monkeypatch)
    fields = ("profile", "model", "provider", "reasoning_effort", "system_prompt", "system_prompt_mode", "tools", "skills")
    invalid = {name: {"type": "integer"} for name in fields}
    plugin_source = f'''
from pathlib import Path
from agent.delegation_provider import DelegationProvider
class Collision(DelegationProvider):
    @property
    def name(self): return "schema-collision"
    def task_parameters(self, config):
        return {invalid!r}
    def route_task(self, task, config):
        Path(self.settings(config)["marker"]).write_text("called", encoding="utf-8")
        return {{"model": "must-not-run"}}
def register(ctx): ctx.register_delegation_provider(Collision())
'''
    manager, marker = _temporary_schema_backend(
        tmp_path, monkeypatch, name="schema-collision", plugin_source=plugin_source)
    try:
        definition = _schema_definition()
        props = _schema_task_props(definition)
        core_props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]["properties"]
        assert all(props[name] == core_props[name] for name in fields)
        assert "WARNING: incompatible core task parameter declarations" in definition["description"]
        assert all(name in definition["description"] for name in fields)

        result = json.loads(delegate_task(tasks=[{"goal": "do not infer", "model": "requested-model"}],
                                          parent_agent=_schema_parent()))
        assert "error" in result
        assert "has incompatible core task parameter declarations" in result["error"]
        assert all(name in result["error"] for name in fields)
        assert not marker.exists()
        assert built == []
    finally:
        manager.unload()


def test_compatible_core_declaration_and_nonconflicting_backend_param_still_route(
        tmp_path, monkeypatch):
    from tools.delegate_tool import DELEGATE_TASK_SCHEMA, delegate_task

    built = _schema_recorded(monkeypatch)
    plugin_source = '''
from pathlib import Path
from agent.delegation_provider import DelegationProvider
class Compatible(DelegationProvider):
    @property
    def name(self): return "schema-compatible"
    def task_parameters(self, config):
        from tools.delegate_tool import DELEGATE_TASK_SCHEMA
        core = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]["properties"]
        return {"model": core["model"], "option": {"type": "string", "enum": ["fast"]}}
    def route_task(self, task, config):
        Path(self.settings(config)["marker"]).write_text("called", encoding="utf-8")
        return {"model": "fast-model", "provider": "openrouter", "base_url": "http://127.0.0.1:9/v1",
                "api_key": "fixture-key", "label": task["option"]}
def register(ctx): ctx.register_delegation_provider(Compatible())
'''
    manager, marker = _temporary_schema_backend(
        tmp_path, monkeypatch, name="schema-compatible", plugin_source=plugin_source)
    try:
        definition = _schema_definition()
        props = _schema_task_props(definition)
        core_props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]["properties"]
        assert props["model"] == core_props["model"]
        assert props["option"]["enum"] == ["fast"]
        output = json.loads(delegate_task(tasks=[{"goal": "use compatible route", "option": "fast"}],
                                          parent_agent=_schema_parent()))
        assert output["results"][0]["status"] == "completed"
        assert marker.read_text(encoding="utf-8") == "called"
        assert (built[-1]["model"], built[-1]["override_api_key"]) == ("fast-model", "fixture-key")
    finally:
        manager.unload()


def test_no_override_keeps_the_backend_route_object_unchanged():
    route = TaskRoute({"model": "backend-default-model", "provider": "custom"}, {"fallback_providers": []})
    tasks = [{"goal": "use the backend route"}]
    options, errors = resolve_task_options(tasks)

    (resolved,) = resolve_task_routes(tasks, {}, Parent(), [route], options, errors)

    assert resolved is route


def test_profile_default_model_alias_is_resolved_by_existing_startup_owner():
    config = {"model": {"provider": "custom", "default": "primary"},
              "model_aliases": {"primary": {"model": "vendor/exact-v4", "provider": "custom",
                                            "base_url": "http://127.0.0.1:9300/v1"}}}
    saved = _profile_default_model(config, "other")
    assert saved["model"] == "vendor/exact-v4"
    assert saved["base_url"] == "http://127.0.0.1:9300/v1"


def test_provider_resolver_uses_owner_signature_and_preserves_named_custom_provider(monkeypatch):
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda **kwargs: {
        "provider": "custom", "base_url": "http://127.0.0.1:9400/v1", "api_key": "selected-key",
        "api_mode": "chat_completions", "request_overrides": {"temperature": 0.4}})
    resolved = _provider_credentials("named-relay", "exact-id", {"providers": {"named-relay": {}}})
    assert resolved["provider"] == "named-relay"
    assert resolved["api_key"] == "selected-key" and resolved["api_mode"] == "chat_completions"
    assert resolved["request_overrides"] == {"temperature": 0.4}


def test_catalog_explicit_rejection_is_refused_and_unreachable_is_unverified():
    with patch("hermes_cli.models_validate.static_model_provider_conflict", return_value=None), \
         patch("hermes_cli.models_validate.validate_requested_model", return_value={
             "accepted": False, "recognized": False, "message": "rejected by live catalog"}):
        assert _catalog_verdict("miss", "custom", {}) == (True, "rejected by live catalog", False)
    with patch("hermes_cli.models_validate.static_model_provider_conflict", return_value=None), \
         patch("hermes_cli.models_validate.validate_requested_model", side_effect=OSError("offline")):
        mismatch, diagnostic, recognized = _catalog_verdict("exact", "custom", {})
    assert not mismatch and not recognized and "inconclusive" in diagnostic


def test_profile_route_owns_credential_reasoning_and_fallback(monkeypatch, tmp_path):
    from tools.delegate_tool_routes import TaskRoute
    from tools.delegate_tool_validation import resolve_task_options, resolve_task_routes
    profile = tmp_path / "profiles" / "coder"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text(yaml.safe_dump({"model": {"provider": "custom", "default": "coder-model"}}),
                                         encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path / "os"))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name == "coder")
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: str(profile))
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda **kwargs: {
        "model": kwargs.get("target_model"), "provider": "custom", "requested_provider": kwargs.get("requested"),
        "base_url": kwargs.get("explicit_base_url") or "http://127.0.0.1:9601/v1",
        "api_key": kwargs.get("explicit_api_key") or "coder-key", "api_mode": "chat_completions",
        "request_overrides": {}, "command": None, "args": [],
    })
    monkeypatch.setattr("tools.delegate_tool_validation._catalog_verdict", lambda *a, **k: (False, "", True))
    tasks = [{"goal": "coder", "profile": "coder", "provider": "custom", "model": "route-model"}]
    options, errors = resolve_task_options(tasks)
    base = TaskRoute({"model": "base-model", "provider": "custom", "base_url": "http://127.0.0.1:9600/v1",
                      "api_key": "launch-key", "api_mode": "chat_completions", "request_overrides": {}},
                     {"fallback_providers": [{"provider": "custom", "model": "old-fallback"}]})
    (route,) = resolve_task_routes(tasks, {}, Parent(), [base], options, errors)
    assert route.error is None and route.creds["model"] == "route-model"
    assert route.creds["api_key"] == "coder-key" and route.creds["base_url"] == "http://127.0.0.1:9601/v1"
    assert route.reasoning_effort is not None
    assert route.routing_cfg["fallback_providers"] == [] and route.profile_home == profile


def test_reasoning_none_parser_and_schema_agree():
    from tools.delegate_tool import DELEGATE_TASK_SCHEMA
    from tools.delegate_tool_options import TaskRouteOptions
    from hermes_constants import VALID_REASONING_EFFORTS
    assert {"none", *VALID_REASONING_EFFORTS}.issuperset({"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"})
    option, error = TaskRouteOptions.parse({"goal": "", "reasoning_effort": "none"}, 0)
    assert error is None and option is not None
    assert option.reasoning_effort == "none" and option.reasoning_effort_set
    fields = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]["properties"]
    assert "none" in fields["reasoning_effort"]["enum"]


def test_public_unknown_task_keys_are_refused_on_internal_credentials_route(tmp_path, monkeypatch):
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from tests.fakes.fake_llm_provider import FakeLLMServer, Text, write_hermes_home
    from tools.delegate_tool import delegate_task

    with FakeLLMServer([Text("child finished")], api_key="internal-loopback-key") as server:
        home = tmp_path / "hermes-test-home"
        write_hermes_home(
            home, server.base_url, api_key="internal-loopback-key",
            extra_config="delegation:\n  fallback_providers: []\n  worktree_isolation: false\n",
        )
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_HOME", str(home))
        db = SessionDB(db_path=get_hermes_home() / "state.db")
        parent = AIAgent(
            api_key="internal-loopback-key", base_url=server.base_url, provider="custom",
            api_mode="chat_completions", model="fake-model", platform="cli", quiet_mode=True,
            skip_context_files=True, skip_memory=True, save_trajectories=False,
            enabled_toolsets=["delegation"], session_id="internal-route-parent", session_db=db,
        )
        parent._ensure_db_session()
        try:
            parent.__dict__["_delegation_backend"] = "unavailable-backend-must-not-be-discovered"
            result = json.loads(delegate_task(tasks=[
                {"goal": "Reject unknown key regardless of value", "unknown_task_key": None},
                {"goal": "Preserve internal reserved fields", "role": "leaf", "group": "review"},
            ], credentials_cfg={"model": "fake-model", "provider": "custom", "base_url": server.base_url,
                                "api_key": "internal-loopback-key", "api_mode": "chat_completions",
                                "fallback_providers": []}, parent_agent=parent))
            assert [entry["status"] for entry in result["results"]] == ["error", "completed"]
            assert [entry["api_calls"] for entry in result["results"]] == [0, 1]
            assert "unknown_task_key" in result["results"][0]["error"]
            assert len(server.requests) == 1
        finally:
            parent.close()
            db.close()

def test_public_delegate_refuses_null_and_value_unknown_keys_default_and_internal(tmp_path, monkeypatch):
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from tests.fakes.fake_llm_provider import FakeLLMServer, Text, write_hermes_home
    from tools.delegate_tool import delegate_task

    with FakeLLMServer([Text("child finished")], api_key="parent-loopback-key") as server:
        home = tmp_path / "hermes-test-home"
        write_hermes_home(
            home, server.base_url, api_key="parent-loopback-key",
            extra_config="delegation:\n  fallback_providers: []\n  worktree_isolation: false\n",
        )
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_HOME", str(home))
        db = SessionDB(db_path=get_hermes_home() / "state.db")
        parent = AIAgent(
            api_key="parent-loopback-key", base_url=server.base_url, provider="custom",
            api_mode="chat_completions", model="fake-model", platform="cli", quiet_mode=True,
            skip_context_files=True, skip_memory=True, save_trajectories=False,
            enabled_toolsets=["delegation"], session_id="unknown-key-parent", session_db=db,
        )
        parent._ensure_db_session()
        try:
            ordinary = json.loads(delegate_task(tasks=[
                {"goal": "Reject unknown non-null key", "unknown_task_key": "secret-value"},
                {"goal": "Reject unknown null key", "unknown_task_key": None},
                {"goal": "Preserve legacy metadata", "role": "leaf", "group": "legacy"},
            ], parent_agent=parent))
            assert [entry["status"] for entry in ordinary["results"]] == ["error", "error", "completed"]
            assert [entry["api_calls"] for entry in ordinary["results"]] == [0, 0, 1]
            assert "secret-value" not in json.dumps(ordinary)
            assert len(server.requests) == 1

            parent.__dict__["_delegation_backend"] = "must-not-discover-backend"
            internal = json.loads(delegate_task(tasks=[
                {"goal": "Reject internal non-null key", "unknown_task_key": "secret-value"},
                {"goal": "Reject internal null key", "unknown_task_key": None},
                {"goal": "Preserve internal legacy metadata", "role": "leaf", "group": "internal"},
            ], credentials_cfg={"model": "fake-model", "provider": "custom", "base_url": server.base_url,
                                "api_key": "parent-loopback-key", "api_mode": "chat_completions",
                                "fallback_providers": []}, parent_agent=parent))
            assert [entry["status"] for entry in internal["results"]] == ["error", "error", "completed"]
            assert [entry["api_calls"] for entry in internal["results"]] == [0, 0, 1]
            assert "secret-value" not in json.dumps(internal)
            assert len(server.requests) == 2
            assert all(request["body"]["model"] == "fake-model" for request in server.requests)
        finally:
            parent.close()
            db.close()


def test_raw_route_fields_are_refused_before_default_and_internal_child_inference(tmp_path, monkeypatch, caplog):
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from tests.fakes.fake_llm_provider import FakeLLMServer, Text, write_hermes_home
    from tools.delegate_tool import delegate_task

    raw_fields = ("api_key", "base_url", "api_mode")
    sentinels = {name: f"SYNTHETIC-RAW-{name}-NEVER-LOG" for name in raw_fields}
    with FakeLLMServer(lambda _request: Text("valid sibling completed"),
                       api_key=["parent-loopback-key", "internal-loopback-key"]) as server:
        home = tmp_path / "hermes-test-home"
        write_hermes_home(
            home, server.base_url, api_key="parent-loopback-key",
            extra_config="delegation:\n  fallback_providers: []\n  worktree_isolation: false\n",
        )
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_HOME", str(home))
        db = SessionDB(db_path=get_hermes_home() / "state.db")
        parent = AIAgent(
            api_key="parent-loopback-key", base_url=server.base_url, provider="custom",
            api_mode="chat_completions", model="fake-model", platform="cli", quiet_mode=True,
            skip_context_files=True, skip_memory=True, save_trajectories=False,
            enabled_toolsets=["delegation"], session_id="raw-route-fields-parent", session_db=db,
        )
        parent._ensure_db_session()
        try:
            cases = []
            for route_kind, key in (("default", "parent-loopback-key"), ("internal", "internal-loopback-key")):
                credentials_cfg = None if route_kind == "default" else {
                    "model": "internal-selected-model", "provider": "custom", "base_url": server.base_url,
                    "api_key": key, "api_mode": "chat_completions", "fallback_providers": [],
                }
                for supplied in ("synthetic-value", None):
                    for field in raw_fields:
                        value = sentinels[field] if supplied is not None else None
                        cases.append((route_kind, key, credentials_cfg,
                                      [{"goal": f"reject {field}", field: value},
                                       {"goal": f"valid {route_kind} sibling"}]))

            for route_kind, key, credentials_cfg, tasks in cases:
                if credentials_cfg is None:
                    parent.__dict__.pop("_delegation_backend", None)
                else:
                    parent.__dict__["_delegation_backend"] = "unavailable-internal-backend"
                result = json.loads(delegate_task(tasks=tasks, credentials_cfg=credentials_cfg, parent_agent=parent))
                assert "results" in result, result
                rejected, sibling = result["results"]
                assert "not started" in rejected["error"]
                assert rejected.get("api_calls", 0) == 0
                assert sibling["status"] == "completed"
                assert sibling["api_calls"] == 1
                assert all(sentinel not in json.dumps(result) for sentinel in sentinels.values())

            assert all(sentinel not in caplog.text for sentinel in sentinels.values())
            requests = list(server.requests)
            assert len(requests) == len(cases)
            default_requests = [request for request in requests if request["body"]["model"] == "fake-model"]
            internal_requests = [request for request in requests
                                 if request["body"]["model"] == "internal-selected-model"]
            assert len(default_requests) == len(internal_requests) == 6
            assert all(request["auth"] == "Bearer parent-loopback-key" for request in default_requests)
            assert all(request["auth"] == "Bearer internal-loopback-key" for request in internal_requests)
        finally:
            parent.close()
            db.close()
