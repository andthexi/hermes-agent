"""Real child route pinning under an actual loopback provider failure."""

import json
import os
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import hermes_yaml as yaml

from run_agent import AIAgent
from tests.fakes.fake_llm_provider import Error, FakeLLMServer, Text, write_hermes_home
from tests.fakes.providers.gemini_native import HERMES_ENV, MODEL_ID, GeminiFake, Text as GeminiText
from tools.delegate_tool import delegate_task


def test_provider_only_uses_saved_profile_default_not_parent_switched_model(tmp_path, monkeypatch):
    """A valid saved Gemini default must survive preflight despite the parent's GPT model."""
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    with GeminiFake(tmp_path / "gemini-fake", [GeminiText("saved Gemini default completed")]) as server:
        home = tmp_path / ".hermes"
        monkeypatch.setenv("HERMES_HOME", str(home))
        home.mkdir(parents=True)
        (home / "config.yaml").write_text(yaml.safe_dump({
            "model": {"provider": "gemini", "default": MODEL_ID},
            "agent": {"api_max_retries": 1},
            "delegation": {"fallback_providers": []},
            "telemetry": {"shared_metrics": {"enabled": False, "send": False}},
        }), encoding="utf-8")
        (home / ".env").write_text("GOOGLE_API_KEY=" + HERMES_ENV["GEMINI_API_KEY"] + "\n", encoding="utf-8")
        protected = {path: path.read_bytes() for path in (home / "config.yaml", home / ".env")}
        proxy_env = server.child_env()
        for key, value in proxy_env.items():
            monkeypatch.setenv(key, value)
        parent = AIAgent(
            provider="custom", model="gpt-6-luna", base_url="http://127.0.0.1:9731/v1", api_key="parent-secret",
            enabled_toolsets=["file"], quiet_mode=True, skip_context_files=True, skip_memory=True,
            save_trajectories=False,
        )
        try:
            raw = delegate_task(
                tasks=[{"goal": "Use the saved profile default", "provider": "gemini"}],
                parent_agent=parent)
        finally:
            parent.close()

        result = json.loads(raw)
        (entry,) = result["results"]
        assert entry["status"] == "completed", entry
        assert entry["summary"] == "saved Gemini default completed"
        assert entry["route"]["resolved_model"] == MODEL_ID
        assert entry["route"]["resolved_provider"] == "gemini"
        (request,) = server.main_calls()
        assert request.model == MODEL_ID
        assert all(host not in {"generativelanguage.googleapis.com", "aiplatform.googleapis.com"}
                   for host in server.refused_hosts)
        assert all(secret not in json.dumps(result) for secret in ("parent-secret", HERMES_ENV["GEMINI_API_KEY"]))
        assert {path: path.read_bytes() for path in protected} == protected


class _BackendRouteServer:
    """Loopback backend with a declared model catalog and recorded inference requests."""

    def __init__(self):
        self.requests = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path.rstrip("/").endswith("/models"):
                    body = json.dumps({"data": [
                        {"id": "backend-default-model"}, {"id": "task-only-model"},
                    ]}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_error(404)

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.requests.append({"path": self.path, "auth": self.headers.get("Authorization"),
                                       "model": payload["model"], "messages": payload["messages"]})
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                chunks = [
                    {"id": "fixture-backend-completion", "object": "chat.completion.chunk", "created": 1,
                     "model": payload["model"], "choices": [{"index": 0,
                         "delta": {"role": "assistant", "content": f"backend completed {payload['model']}"},
                         "finish_reason": None}]},
                    {"id": "fixture-backend-completion", "object": "chat.completion.chunk", "created": 1,
                     "model": payload["model"], "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                ]
                for chunk in chunks:
                    self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

            def log_message(self, format, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.worker = Thread(target=self.server.serve_forever, daemon=True)
        self.base_url = f"http://127.0.0.1:{self.server.server_port}/v1"

    def __enter__(self):
        self.worker.start()
        return self

    def __exit__(self, *_exc):
        self.server.shutdown()
        self.server.server_close()
        self.worker.join(timeout=5)


def test_model_only_override_keeps_selected_backend_provider_and_credentials(tmp_path, monkeypatch):
    """A model-only task override changes only the model on the real backend's route."""
    from agent.delegation_provider import DelegationProvider
    from hermes_cli import plugins as plugins_mod
    from hermes_cli.plugins import PluginManager

    repo_plugins = Path(__file__).resolve().parents[2] / "plugins"
    with (
        _BackendRouteServer() as backend,
        FakeLLMServer([Text("saved route was used")], api_key="saved-route-secret") as saved,
        FakeLLMServer([Text("switched route was used")], api_key="switched-route-secret") as switched,
    ):
        os_home = tmp_path / "os-home"
        home = os_home / ".hermes"
        monkeypatch.setenv("HOME", str(os_home))
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("BACKEND_ROUTE_KEY", "backend-route-secret")
        monkeypatch.setenv("SAVED_ROUTE_KEY", "saved-route-secret")

        plugin_dir = home / "plugins" / "fixture-route-backend"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.yaml").write_text(
            "name: fixture-route-backend\nversion: 1.0\n", encoding="utf-8",
        )
        (plugin_dir / "__init__.py").write_text(
            "from agent.delegation_provider import DelegationProvider\n"
            "class FixtureRouteBackend(DelegationProvider):\n"
            "    @property\n"
            "    def name(self): return 'fixture-route-backend'\n"
            "    def route_task(self, task, config):\n"
            "        return {'provider': 'fixture-backend-provider', 'model': 'backend-default-model'}\n"
            "def register(ctx): ctx.register_delegation_provider(FixtureRouteBackend())\n",
            encoding="utf-8",
        )
        bundled = tmp_path / "bundled"
        bundled.mkdir()
        (bundled / "delegation").symlink_to(repo_plugins / "delegation", target_is_directory=True)
        monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: bundled)

        (home / "config.yaml").write_text(yaml.safe_dump({
            "plugins": {"enabled": ["fixture-route-backend"]},
            "model": {"provider": "saved-parent-provider", "default": "saved-parent-model"},
            "providers": {
                "saved-parent-provider": {"base_url": saved.base_url, "api_key_env": "SAVED_ROUTE_KEY"},
                "fixture-backend-provider": {"base_url": backend.base_url,
                                             "api_key_env": "BACKEND_ROUTE_KEY"},
            },
            "agent": {"api_max_retries": 1},
            "delegation": {"backend": "fixture-route-backend"},
        }), encoding="utf-8")
        (home / ".env").write_text(
            "SAVED_ROUTE_KEY=saved-route-secret\nBACKEND_ROUTE_KEY=backend-route-secret\n",
            encoding="utf-8",
        )
        protected = {path: path.read_bytes() for path in (home / "config.yaml", home / ".env")}

        manager = PluginManager()
        manager.discover_and_load()
        parent = AIAgent(
            provider="custom", model="switched-parent-model", base_url=switched.base_url,
            api_key="switched-route-secret", enabled_toolsets=["file"], quiet_mode=True,
            skip_context_files=True, skip_memory=True, save_trajectories=False,
        )
        parent_before = tuple(getattr(parent, name, None) for name in (
            "provider", "model", "base_url", "api_key", "_cached_system_prompt",
        ))
        try:
            raw = delegate_task(tasks=[
                {"goal": "Use backend model without an override"},
                {"goal": "Use backend provider with this model override", "model": "task-only-model"},
            ], parent_agent=parent)
        finally:
            parent.close()
            manager.unload()

        result = json.loads(raw)
        assert "results" in result, result
        entries = {entry["task_index"]: entry for entry in result["results"]}
        assert {index: entry["status"] for index, entry in entries.items()} == {
            0: "completed", 1: "completed",
        }, entries
        assert entries[0]["summary"] == "backend completed backend-default-model"
        assert entries[1]["summary"] == "backend completed task-only-model"
        assert entries[0]["route"]["backend"] == "fixture-route-backend"
        assert entries[1]["route"]["backend"] == "fixture-route-backend"
        assert entries[1]["route"]["resolved_provider"] == "fixture-backend-provider"
        assert entries[1]["route"]["resolved_model"] == "task-only-model"

        requests_by_goal = {
            goal: [request for request in backend.requests
                   if goal in json.dumps(request["messages"])]
            for goal in ("Use backend model without an override",
                         "Use backend provider with this model override")
        }
        assert {goal: len(requests) for goal, requests in requests_by_goal.items()} == {
            "Use backend model without an override": 1,
            "Use backend provider with this model override": 1,
        }
        assert {goal: requests[0]["model"] for goal, requests in requests_by_goal.items()} == {
            "Use backend model without an override": "backend-default-model",
            "Use backend provider with this model override": "task-only-model",
        }
        assert {goal: requests[0]["auth"] for goal, requests in requests_by_goal.items()} == {
            "Use backend model without an override": "Bearer backend-route-secret",
            "Use backend provider with this model override": "Bearer backend-route-secret",
        }
        assert not saved.main_requests()
        assert not switched.main_requests()
        assert all(secret not in json.dumps(result) for secret in (
            "backend-route-secret", "saved-route-secret", "switched-route-secret",
        ))
        assert {path: path.read_bytes() for path in protected} == protected
        assert tuple(getattr(parent, name, None) for name in (
            "provider", "model", "base_url", "api_key", "_cached_system_prompt",
        )) == parent_before


def test_profile_route_override_stays_pinned_on_failure(
    tmp_path, monkeypatch,
):
    """A task-level explicit route pin must refuse fallback after a real provider failure."""
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    def fail_explicit_but_finish_profile_primary(request):
        if request["body"]["model"] == "explicit-model-pin":
            return Error(status=401, message="invalid api key")
        return Text("profile primary completed")

    with (
        FakeLLMServer(fail_explicit_but_finish_profile_primary) as primary,
        FakeLLMServer([Text("declared fallback completed")], api_key="fallback-child-secret") as fallback,
    ):
        home = tmp_path / ".hermes"
        monkeypatch.setenv("HERMES_HOME", str(home))
        write_hermes_home(home, primary.base_url, api_key="parent-secret")
        profile_home = home / "profiles" / "worker"
        profile_home.mkdir(parents=True)
        (profile_home / "config.yaml").write_text(
            "model:\n"
            "  provider: custom\n"
            f"  base_url: {primary.base_url}\n"
            "  default: profile-primary-model\n"
            "  context_length: 128000\n"
            "agent:\n"
            "  api_max_retries: 1\n"
            "fallback_providers:\n"
            "  - provider: custom\n"
            "    model: declared-fallback-model\n"
            f"    base_url: {fallback.base_url}\n"
            "    api_key: fallback-child-secret\n",
            encoding="utf-8",
        )
        (profile_home / ".env").write_text("OPENAI_API_KEY=profile-primary-secret\n", encoding="utf-8")
        (profile_home / "SOUL.md").write_text("Worker profile identity\n", encoding="utf-8")
        protected = {
            path: path.read_bytes()
            for path in (home / "config.yaml", home / ".env", profile_home / "config.yaml", profile_home / ".env")
        }
        parent = AIAgent(
            provider="custom", model="fake-model", base_url=primary.base_url, api_key="parent-secret",
            enabled_toolsets=["file"], quiet_mode=True, skip_context_files=True, skip_memory=True,
            save_trajectories=False,
        )
        try:
            raw = delegate_task(
                tasks=[
                    {"goal": "Take the explicit route and fail", "profile": "worker", "provider": "custom", "model": "explicit-model-pin"},
                    {"goal": "Use the profile route and its declared fallback", "profile": "worker"},
                ],
                parent_agent=parent,
            )
        finally:
            parent.close()

        result = json.loads(raw)
        assert "results" in result, result
        entries = {entry["task_index"]: entry for entry in result["results"]}
        assert entries[0]["status"] == "failed"
        assert entries[0]["route"]["model"] == "explicit-model-pin"
        assert entries[1]["status"] == "completed", entries[1]
        assert entries[1]["summary"] == "profile primary completed"
        assert entries[1]["route"]["model"] == "profile-primary-model"
        assert not entries[1]["route"].get("fallback_activated")

        primary_models = sorted(request["model"] for request in primary.main_requests())
        assert primary_models == ["explicit-model-pin", "profile-primary-model"]
        assert not fallback.main_requests()
        assert all(key not in json.dumps(result) for key in (
            "parent-secret", "profile-primary-secret", "fallback-child-secret",
        ))
        assert {path: path.read_bytes() for path in protected} == protected


def test_legacy_fallback_activates_unpinned_and_explicit_route_pin_suppresses_same_fault(
    tmp_path, monkeypatch,
):
    """A billing failure activates the inherited legacy chain unless the task pins its route."""
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    def reject_primary(_request):
        return Error(status=402, message="Insufficient credits")

    with (
        FakeLLMServer(reject_primary) as primary,
        FakeLLMServer([Text("legacy fallback completed")], api_key="fallback-child-secret") as fallback,
    ):
        home = tmp_path / ".hermes"
        monkeypatch.setenv("HERMES_HOME", str(home))
        extra_config = (
            "delegation:\n"
            "  fallback_providers:\n"
            "    - provider: custom\n"
            "      model: legacy-fallback-model\n"
            f"      base_url: {fallback.base_url}\n"
            "      api_key: fallback-child-secret\n"
        )
        write_hermes_home(home, primary.base_url, api_key="parent-secret", extra_config=extra_config)
        protected = {path: path.read_bytes() for path in (home / "config.yaml", home / ".env")}
        parent = AIAgent(
            provider="custom", model="primary-model", base_url=primary.base_url, api_key="parent-secret",
            enabled_toolsets=["file"], quiet_mode=True, skip_context_files=True, skip_memory=True,
            save_trajectories=False,
        )
        parent_before = tuple(getattr(parent, name, None) for name in (
            "provider", "model", "base_url", "api_key", "_cached_system_prompt",
        ))
        try:
            control = json.loads(delegate_task(
                goal="Use the inherited route and its legacy fallback", parent_agent=parent,
            ))
            pinned = json.loads(delegate_task(
                tasks=[{"goal": "Keep this explicit model pin and fail", "model": "explicit-model-pin"}],
                parent_agent=parent,
            ))
        finally:
            parent.close()

        (control_entry,) = control["results"]
        (pinned_entry,) = pinned["results"]
        assert control_entry["status"] == "completed", control_entry
        assert control_entry["summary"] == "legacy fallback completed"
        assert control_entry["model"] == "legacy-fallback-model"
        assert pinned_entry["status"] == "failed", pinned_entry
        assert pinned_entry["route"]["requested_model"] == "explicit-model-pin"
        assert pinned_entry["route"]["model"] == "explicit-model-pin"
        assert not pinned_entry["route"].get("fallback_activated")

        primary_requests = primary.main_requests()
        assert [request["model"] for request in primary_requests] == ["primary-model", "explicit-model-pin"]
        messages_by_model = {request["model"]: json.dumps(request["messages"]) for request in primary_requests}
        assert "Use the inherited route" in messages_by_model["primary-model"]
        assert "Keep this explicit model pin" in messages_by_model["explicit-model-pin"]
        fallback_requests = fallback.main_requests()
        assert len(fallback_requests) == 1
        assert fallback_requests[0]["model"] == "legacy-fallback-model"
        fallback_record = next(request for request in fallback.requests if request["kind"] == "main")
        assert fallback_record["auth"] == "Bearer fallback-child-secret"
        assert all(value not in json.dumps((control, pinned)) for value in (
            "parent-secret", "fallback-child-secret",
        ))
        assert {path: path.read_bytes() for path in protected} == protected
        assert tuple(getattr(parent, name, None) for name in (
            "provider", "model", "base_url", "api_key", "_cached_system_prompt",
        )) == parent_before


def test_authoritative_profile_catalog_rejects_before_child_inference(tmp_path, monkeypatch):
    """An owner catalog miss refuses one task while its recognized sibling runs."""
    requests = []

    class CatalogHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append((self.path, self.headers.get("Authorization")))
            if self.path == "/catalog":
                body = json.dumps({"data": [{"id": "catalog-known-model"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(404)

        def do_POST(self):
            if not self.path.endswith("/chat/completions"):
                self.send_error(404)
                return
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, self.headers.get("Authorization"), payload))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            chunks = [
                {"id": "fixture-catalog-completion", "object": "chat.completion.chunk", "created": 1,
                 "model": payload["model"], "choices": [{"index": 0,
                     "delta": {"role": "assistant", "content": "recognized catalog route completed"},
                     "finish_reason": None}]},
                {"id": "fixture-catalog-completion", "object": "chat.completion.chunk", "created": 1,
                 "model": payload["model"], "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            ]
            for chunk in chunks:
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), CatalogHandler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        home = tmp_path / ".hermes"
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
        base = f"http://127.0.0.1:{server.server_port}/inference"
        catalog_url = f"http://127.0.0.1:{server.server_port}/catalog"
        profile_dir = home / "profiles" / "catalog-worker"
        plugin_dir = profile_dir / "plugins" / "model-providers" / "fixture-catalog"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.yaml").write_text(
            "name: fixture-authoritative-catalog\nkind: model-provider\nversion: 1.0.0\n", encoding="utf-8",
        )
        (plugin_dir / "__init__.py").write_text(
            "from providers import register_provider\n"
            "from providers.base import ProviderProfile\n"
            "register_provider(ProviderProfile(\n"
            " name='fixture-authoritative-catalog', auth_type='api_key',\n"
            " env_vars=('FIXTURE_CATALOG_KEY',),\n"
            f" base_url={base!r}, models_url={catalog_url!r},\n"
            "))\n", encoding="utf-8",
        )
        (profile_dir / "config.yaml").write_text(yaml.safe_dump({
            "model": {"provider": "fixture-authoritative-catalog", "default": "catalog-known-model",
                      "base_url": base, "key_env": "FIXTURE_CATALOG_KEY"},
            "agent": {"api_max_retries": 1},
        }), encoding="utf-8")
        (profile_dir / ".env").write_text("FIXTURE_CATALOG_KEY=fixture-secret\n", encoding="utf-8")
        protected = {path: path.read_bytes() for path in (profile_dir / "config.yaml", profile_dir / ".env")}
        parent = AIAgent(
            provider="custom", model="fake-model", base_url=base, api_key="parent-secret",
            enabled_toolsets=["file"], quiet_mode=True, skip_context_files=True, skip_memory=True,
            save_trajectories=False,
        )
        try:
            raw = delegate_task(tasks=[
                {"goal": "This must not run", "profile": "catalog-worker", "model": "catalog-missing-exact",
                 "provider": "fixture-authoritative-catalog"},
                {"goal": "Run the recognized catalog route", "profile": "catalog-worker",
                 "model": "catalog-known-model", "provider": "fixture-authoritative-catalog"},
            ], parent_agent=parent)
        finally:
            parent.close()

        result = json.loads(raw)
        assert "results" in result, result
        entries = {entry["task_index"]: entry for entry in result["results"]}
        assert entries[0]["status"] == "error"
        assert entries[0]["api_calls"] == 0
        assert entries[0]["route"]["refused"] is True
        assert "not found in this provider's catalog" in entries[0]["error"]
        assert entries[1]["status"] == "completed", entries[1]
        assert entries[1]["summary"] == "recognized catalog route completed"
        inference_requests = [request for request in requests
                              if len(request) == 3 and request[0].endswith("/chat/completions")]
        assert len(inference_requests) == 1
        assert inference_requests[0][2]["model"] == "catalog-known-model"
        assert any(path == "/catalog" and auth == "Bearer fixture-secret"
                   for request in requests if len(request) == 2 for path, auth in [request])
        assert all(value not in json.dumps(result) for value in (
            "fixture-secret", "parent-secret",
        ))
        assert {path: path.read_bytes() for path in protected} == protected
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def _write_profile_auth_provider(profile_dir, base_url):
    plugin_dir = profile_dir / "plugins" / "model-providers" / "fixture-profile-auth"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        "name: fixture-profile-auth\nkind: model-provider\nversion: 1.0.0\n", encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(
        "from providers import register_provider\n"
        "from providers.base import ProviderProfile\n"
        "register_provider(ProviderProfile(\n"
        " name='fixture-profile-auth', auth_type='api_key',\n"
        " env_vars=('FIXTURE_PROFILE_API_KEY',),\n"
        f" base_url={base_url!r},\n"
        "))\n", encoding="utf-8",
    )
    (profile_dir / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": "fixture-profile-auth", "default": "fake-model", "base_url": base_url},
        "agent": {"api_max_retries": 1},
    }), encoding="utf-8")


def test_missing_selected_profile_credential_refuses_before_inference_and_sibling_runs(tmp_path, monkeypatch):
    """A selected profile's required key cannot fall through to the parent's environment key."""
    with FakeLLMServer([Text("valid sibling completed")], api_key="valid-sibling-key") as server:
        home = tmp_path / ".hermes"
        write_hermes_home(home, server.base_url, api_key="parent-only-key")
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("FIXTURE_PROFILE_API_KEY", "parent-only-key")
        monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)

        missing_profile = home / "profiles" / "missing-worker"
        valid_profile = home / "profiles" / "valid-worker"
        missing_profile.mkdir(parents=True)
        valid_profile.mkdir(parents=True)
        _write_profile_auth_provider(missing_profile, server.base_url)
        _write_profile_auth_provider(valid_profile, server.base_url)
        (missing_profile / ".env").write_text("", encoding="utf-8")
        (valid_profile / ".env").write_text("FIXTURE_PROFILE_API_KEY=valid-sibling-key\n", encoding="utf-8")
        (missing_profile / "SOUL.md").write_text("Missing-key worker identity\n", encoding="utf-8")
        (valid_profile / "SOUL.md").write_text("Valid-key worker identity\n", encoding="utf-8")
        protected = {path: path.read_bytes() for path in (
            missing_profile / "config.yaml", missing_profile / ".env",
            valid_profile / "config.yaml", valid_profile / ".env",
        )}

        from agent.secret_scope import get_secret
        from tools.delegate_tool_routes import delegated_profile_scope
        assert (missing_profile / ".env").read_text(encoding="utf-8") == ""
        assert __import__("os").environ["FIXTURE_PROFILE_API_KEY"] == "parent-only-key"
        with delegated_profile_scope(missing_profile):
            assert get_secret("FIXTURE_PROFILE_API_KEY") is None

        parent = AIAgent(
            provider="custom", model="fake-model", base_url=server.base_url, api_key="parent-only-key",
            enabled_toolsets=["file"], quiet_mode=True, skip_context_files=True, skip_memory=True,
            save_trajectories=False,
        )
        try:
            raw = delegate_task(tasks=[
                {"goal": "Must be refused: selected profile has no required key", "profile": "missing-worker"},
                {"goal": "Run under the profile with its own key", "profile": "valid-worker"},
            ], parent_agent=parent)
        finally:
            parent.close()

        result = json.loads(raw)
        assert "results" in result, result
        entries = {entry["task_index"]: entry for entry in result["results"]}
        assert entries[0]["status"] == "error", entries[0]
        assert entries[0]["api_calls"] == 0
        assert entries[0]["route"]["refused"] is True
        assert "missing_api_key" in entries[0]["error"] or "credential" in entries[0]["error"].lower()
        assert entries[1]["status"] == "completed", entries[1]
        assert entries[1]["summary"] == "valid sibling completed"
        main = [request for request in server.requests if request["kind"] == "main"]
        assert len(main) == 1
        assert main[0]["auth"] == "Bearer valid-sibling-key"
        assert main[0]["body"]["model"] == "fake-model"
        assert all(secret not in json.dumps(result) for secret in ("parent-only-key", "valid-sibling-key"))
        assert {path: path.read_bytes() for path in protected} == protected


def test_registered_backend_authorization_refusal_survives_route_overrides_and_sibling_runs(tmp_path, monkeypatch):
    """A real registered backend refusal comes before explicit model/provider overrides."""
    from hermes_cli import plugins as plugins_mod
    from hermes_cli.plugins import PluginManager
    from pathlib import Path

    home = tmp_path / ".hermes"
    backend_dir = home / "plugins" / "auth-guard"
    backend_dir.mkdir(parents=True)
    (backend_dir / "plugin.yaml").write_text("name: auth-guard\nversion: 1.0\n", encoding="utf-8")
    (backend_dir / "__init__.py").write_text(
        "from agent.delegation_provider import DelegationProvider, DelegationRouteError\n"
        "class Guard(DelegationProvider):\n"
        " @property\n"
        " def name(self): return 'auth-guard'\n"
        " def route_task(self, task, config):\n"
        "  if task.get('goal') == 'deny this task now':\n"
        "   raise DelegationRouteError('Backend authorization denied this task.')\n"
        "  return {}\n"
        "def register(ctx): ctx.register_delegation_provider(Guard())\n", encoding="utf-8",
    )
    (home / "config.yaml").write_text(yaml.safe_dump({
        "plugins": {"enabled": ["auth-guard"]},
        "delegation": {"backend": "auth-guard", "model": "parent-model"},
    }), encoding="utf-8")
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    (bundled / "delegation").symlink_to(
        Path(__file__).resolve().parents[2] / "plugins" / "delegation", target_is_directory=True,
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: bundled)
    manager = PluginManager()
    manager.discover_and_load()
    try:
        with FakeLLMServer([Text("authorized sibling completed")], api_key="parent-route-key") as server:
            (home / "config.yaml").write_text(yaml.safe_dump({
                "plugins": {"enabled": ["auth-guard"]},
                "model": {"provider": "custom", "default": "parent-model", "base_url": server.base_url,
                          "api_key": "parent-route-key"},
                "delegation": {"backend": "auth-guard", "model": "parent-model"},
            }), encoding="utf-8")
            parent = AIAgent(
                provider="custom", model="parent-model", base_url=server.base_url, api_key="parent-route-key",
                enabled_toolsets=["file"], quiet_mode=True, skip_context_files=True, skip_memory=True,
                save_trajectories=False,
            )
            try:
                raw = delegate_task(tasks=[
                    {"goal": "deny this task now", "model": "explicit-denied-model"},
                    {"goal": "allow this sibling task", "model": "explicit-sibling-model"},
                ], parent_agent=parent)
            finally:
                parent.close()

            result = json.loads(raw)
            assert "results" in result, result
            entries = {entry.get("task_index", index): entry for index, entry in enumerate(result["results"])}
            assert entries[0]["status"] == "error", entries[0]
            assert entries[0]["api_calls"] == 0
            assert entries[0]["route"]["backend"] == "auth-guard"
            assert entries[0]["route"]["refused"] is True
            assert entries[0]["error"] == "Task 0: Backend authorization denied this task."
            assert entries[1]["status"] == "completed", entries[1]
            assert entries[1]["summary"] == "authorized sibling completed"
            main = [request for request in server.requests if request["kind"] == "main"]
            assert len(main) == 1
            assert main[0]["auth"] == "Bearer parent-route-key"
            assert main[0]["body"]["model"] == "explicit-sibling-model"
            assert "explicit-denied-model" not in json.dumps(main)
            assert "parent-route-key" not in json.dumps(result)
    finally:
        manager.unload()


def test_inconclusive_profile_catalog_runs_exact_pair_and_reports_warning(tmp_path, monkeypatch):
    """A down owner catalog stays unverified; the exact id/provider runs on the inference endpoint."""
    requests = []

    class CatalogHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append((self.path, self.headers.get("Authorization")))
            self.send_error(503)

        def do_POST(self):
            if not self.path.endswith("/chat/completions"):
                self.send_error(404)
                return
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, self.headers.get("Authorization"), payload))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            chunks = [
                {"id": "fixture-catalog-completion", "object": "chat.completion.chunk", "created": 1,
                 "model": payload["model"], "choices": [{"index": 0,
                     "delta": {"role": "assistant", "content": "exact unverified route completed"},
                     "finish_reason": None}]},
                {"id": "fixture-catalog-completion", "object": "chat.completion.chunk", "created": 1,
                 "model": payload["model"], "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            ]
            for chunk in chunks:
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), CatalogHandler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        home = tmp_path / ".hermes"
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
        base = f"http://127.0.0.1:{server.server_port}/inference"
        catalog_url = f"http://127.0.0.1:{server.server_port}/catalog"
        profile_dir = home / "profiles" / "catalog-worker"
        plugin_dir = profile_dir / "plugins" / "model-providers" / "fixture-catalog"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.yaml").write_text(
            "name: fixture-authoritative-catalog\nkind: model-provider\nversion: 1.0.0\n", encoding="utf-8",
        )
        (plugin_dir / "__init__.py").write_text(
            "from providers import register_provider\n"
            "from providers.base import ProviderProfile\n"
            "register_provider(ProviderProfile(\n"
            " name='fixture-authoritative-catalog', auth_type='api_key',\n"
            " env_vars=('FIXTURE_CATALOG_KEY',),\n"
            f" base_url={base!r}, models_url={catalog_url!r},\n"
            "))\n", encoding="utf-8",
        )
        (profile_dir / "config.yaml").write_text(yaml.safe_dump({
            "model": {"provider": "fixture-authoritative-catalog", "default": "catalog-known-model",
                      "base_url": base, "key_env": "FIXTURE_CATALOG_KEY"},
            "agent": {"api_max_retries": 1},
        }), encoding="utf-8")
        (profile_dir / ".env").write_text("FIXTURE_CATALOG_KEY=fixture-secret\n", encoding="utf-8")
        protected = {path: path.read_bytes() for path in (profile_dir / "config.yaml", profile_dir / ".env")}
        parent = AIAgent(
            provider="custom", model="fake-model", base_url=base, api_key="parent-secret",
            enabled_toolsets=["file"], quiet_mode=True, skip_context_files=True, skip_memory=True,
            save_trajectories=False,
        )
        try:
            raw = delegate_task(tasks=[{
                "goal": "Reply with a short confirmation", "profile": "catalog-worker",
                "model": "catalog-unknown-exact", "provider": "fixture-authoritative-catalog",
            }], parent_agent=parent)
        finally:
            parent.close()

        result = json.loads(raw)
        assert "results" in result, result
        (entry,) = result["results"]
        assert entry["status"] == "completed", entry
        assert entry["summary"] == "exact unverified route completed"
        assert entry["api_calls"] == 1
        assert entry["route"]["validation"] == "unverified"
        assert "attempting the exact selection" in entry["route"]["diagnostic"]
        assert "catalog-unknown-exact" in entry["route"]["diagnostic"]
        assert entry["route"]["model"] == "catalog-unknown-exact"
        assert entry["route"]["provider"] == "fixture-authoritative-catalog"
        inference_requests = [request for request in requests
                              if len(request) == 3 and request[0].endswith("/chat/completions")]
        assert len(inference_requests) == 1
        path, authorization, payload = inference_requests[0]
        assert path.endswith("/chat/completions")
        assert authorization == "Bearer fixture-secret"
        assert payload["model"] == "catalog-unknown-exact"
        assert any(path == "/catalog" and auth == "Bearer fixture-secret" for request in requests
                   if len(request) == 2 for path, auth in [request])
        assert all(value not in json.dumps(result) for value in (
            "fixture-secret", "parent-secret",
        ))
        assert {path: path.read_bytes() for path in protected} == protected
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)



def test_selected_profile_pool_isolated_through_real_child_lease_and_loopback(tmp_path, monkeypatch):
    """Children on one shared gateway acquire and send only their selected profile pool key."""
    from agent.credential_pool import CredentialPool, load_pool
    from agent.secret_scope import get_secret
    from hermes_constants import get_hermes_home
    from tools.delegate_tool_routes import delegated_profile_scope, _profile_route

    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    os_home = tmp_path / "os-home"
    launch = os_home / ".hermes"
    profile_a, profile_b = launch / "profiles" / "worker-a", launch / "profiles" / "worker-b"
    provider = "fixture-gateway"
    pool_keys = {"launch": "fixture-launch-pool", "worker-a": "fixture-profile-a-pool",
                 "worker-b": "fixture-profile-b-pool"}
    fixed_keys = {"launch": "fixture-launch-fixed", "worker-a": "fixture-a-fixed",
                  "worker-b": "fixture-b-fixed"}
    scopes = {"launch": "fixture-launch-scope", "worker-a": "fixture-a-scope",
              "worker-b": "fixture-b-scope"}

    profiles = ((launch, "launch", "launch-model"), (profile_a, "worker-a", "fake-model"),
                (profile_b, "worker-b", "fake-model"))
    with FakeLLMServer(lambda req: Text(f"completed {req['body']['model']}"),
                       api_key=tuple(pool_keys.values()) + tuple(fixed_keys.values()), record_get=True) as server:
        for home, name, model in profiles:
            home.mkdir(parents=True, exist_ok=True)
            (home / "config.yaml").write_text(yaml.safe_dump({
                "model": {"provider": provider, "default": model},
                "providers": {provider: {"api": server.base_url, "key_env": "FIXTURE_GATEWAY_KEY"}},
                "agent": {"api_max_retries": 1},
            }), encoding="utf-8")
            (home / ".env").write_text(f"FIXTURE_GATEWAY_KEY={fixed_keys[name]}\nFIXTURE_SCOPE={scopes[name]}\n", encoding="utf-8")
            row = {"id": name, "label": name, "auth_type": "api_key", "priority": 0,
                   "source": "manual", "access_token": pool_keys[name], "base_url": server.base_url}
            (home / "auth.json").write_text(json.dumps({"version": 1, "credential_pool": {provider: [row]}}),
                                            encoding="utf-8")
        monkeypatch.setenv("HOME", str(tmp_path / "os-home-anchor"))
        monkeypatch.setenv("HERMES_HOME", str(launch))
        monkeypatch.setenv("FIXTURE_GATEWAY_KEY", fixed_keys["launch"])
        monkeypatch.setenv("FIXTURE_SCOPE", scopes["launch"])

        with delegated_profile_scope(launch):
            parent_route = _profile_route(launch)
            assert [entry.access_token for entry in load_pool(provider).entries()] == [pool_keys["launch"]]
            parent = AIAgent(provider=parent_route.creds["provider"], requested_provider=provider,
                             model=parent_route.creds["model"], base_url=parent_route.creds["base_url"],
                             api_key=parent_route.creds["api_key"], api_mode=parent_route.creds["api_mode"],
                             credential_pool=load_pool(provider),
                             enabled_toolsets=["file"], quiet_mode=True, skip_context_files=True, skip_memory=True,
                             save_trajectories=False)
        assert [entry.access_token for entry in parent._credential_pool.entries()] == [pool_keys["launch"]]
        protected = {path: path.read_bytes() for home in (launch, profile_a, profile_b)
                     for path in (home / "config.yaml", home / ".env", home / "auth.json")}

        leases = []
        acquire = CredentialPool.acquire_lease
        def observe_acquire(pool, credential_id=None):
            lease_id = acquire(pool, credential_id)
            if lease_id:
                entry = next(row for row in pool.entries() if row.id == lease_id)
                leases.append((str(get_hermes_home()), pool.provider, entry.access_token, get_secret("FIXTURE_SCOPE")))
            return lease_id
        monkeypatch.setattr(CredentialPool, "acquire_lease", observe_acquire)
        try:
            result = json.loads(delegate_task(tasks=[
                {"goal": "Identify the worker-a pool acquisition", "profile": "worker-a"},
                {"goal": "Identify the worker-b pool acquisition", "profile": "worker-b"},
            ], parent_agent=parent))
        finally:
            parent.close()

        assert "results" in result, result
        entries = {entry["task_index"]: entry for entry in result["results"]}
        assert {i: entry["status"] for i, entry in entries.items()} == {0: "completed", 1: "completed"}, entries
        assert [entry["summary"] for entry in entries.values()] == ["completed fake-model", "completed fake-model"]
        lease_scopes = {
            home: {(token, scope) for seen_home, _provider, token, scope in leases if seen_home == home}
            for home in (str(profile_a), str(profile_b))
        }
        assert lease_scopes == {
            str(profile_a): {(pool_keys["worker-a"], scopes["worker-a"])},
            str(profile_b): {(pool_keys["worker-b"], scopes["worker-b"])},
        }
        assert all(_provider == provider for _home, _provider, _token, _scope in leases)
        main_requests = [req for req in server.requests if req["kind"] == "main"]
        requests_by_profile = {
            name: [req for req in main_requests if goal in json.dumps(req["body"]["messages"])]
            for name, goal in (("worker-a", "Identify the worker-a pool acquisition"),
                               ("worker-b", "Identify the worker-b pool acquisition"))
        }
        assert {name: len(requests) for name, requests in requests_by_profile.items()} == {
            "worker-a": 1, "worker-b": 1,
        }
        assert {name: requests[0]["auth"] for name, requests in requests_by_profile.items()} == {
            "worker-a": f"Bearer {pool_keys['worker-a']}",
            "worker-b": f"Bearer {pool_keys['worker-b']}",
        }
        assert {path: path.read_bytes() for path in protected} == protected
        assert [entry.access_token for entry in load_pool(provider).entries()] == [pool_keys["launch"]]
        selected = {}
        for home in (profile_a, profile_b):
            with delegated_profile_scope(home):
                selected[str(home)] = [entry.access_token for entry in load_pool(provider).entries()]
        assert selected == {str(profile_a): [pool_keys["worker-a"]], str(profile_b): [pool_keys["worker-b"]]}
        assert get_hermes_home() == launch and get_secret("FIXTURE_SCOPE") == scopes["launch"]
        assert all(secret not in json.dumps(result) for secret in (*pool_keys.values(), *fixed_keys.values()))


def test_strict_child_reasoning_effort_is_not_dropped_after_real_provider_400(tmp_path, monkeypatch):
    """An actual reasoning-effort 400 ends only the strict child; sibling control still completes."""
    from run_agent import AIAgent

    def reject_strict_effort(request):
        body = request["body"]
        if body["model"] == "explicit-strict-model":
            return Error(status=400, message="Unrecognized request argument supplied: reasoning_effort")
        return Text("valid control completed")

    with FakeLLMServer(reject_strict_effort, api_key="fixture-parent-key") as server:
        home = tmp_path / ".hermes"
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
        write_hermes_home(home, server.base_url, api_key="fixture-parent-key")
        protected = {path: path.read_bytes() for path in (home / "config.yaml", home / ".env")}
        parent = AIAgent(
            provider="custom", model="parent-model", base_url=server.base_url, api_key="fixture-parent-key",
            enabled_toolsets=["file"], quiet_mode=True, skip_context_files=True, skip_memory=True,
            save_trajectories=False,
        )
        parent_before = tuple(getattr(parent, name, None) for name in (
            "provider", "model", "base_url", "api_key", "_cached_system_prompt", "reasoning_config",
        ))
        try:
            raw = delegate_task(tasks=[
                {"goal": "Reject this explicit high effort", "model": "explicit-strict-model",
                 "reasoning_effort": "high"},
                {"goal": "Complete the no-explicit-effort control", "model": "valid-control-model"},
            ], parent_agent=parent)
        finally:
            parent.close()

        result = json.loads(raw)
        assert "results" in result, result
        entries = {entry["task_index"]: entry for entry in result["results"]}
        strict, control = entries[0], entries[1]
        assert strict["status"] == "failed", strict
        assert strict["route"]["reasoning_effort"] == "high"
        assert strict["api_calls"] == 1
        assert "reasoning_effort" in strict["error"]
        assert strict.get("failure_reason") == "reasoning_mandatory"
        assert control["status"] == "completed", control
        assert control["summary"] == "valid control completed"

        requests_by_goal = {
            goal: [request for request in server.main_requests()
                   if goal in json.dumps(request["messages"])]
            for goal in ("Reject this explicit high effort", "Complete the no-explicit-effort control")
        }
        assert {goal: len(requests) for goal, requests in requests_by_goal.items()} == {
            "Reject this explicit high effort": 1,
            "Complete the no-explicit-effort control": 1,
        }
        strict_body = requests_by_goal["Reject this explicit high effort"][0]
        assert strict_body["model"] == "explicit-strict-model"
        carriers = [strict_body.get("reasoning_effort"),
                    (strict_body.get("extra_body") or {}).get("reasoning"),
                    strict_body.get("reasoning")]
        assert any(carrier == "high" or (isinstance(carrier, dict) and carrier.get("effort") == "high")
                   for carrier in carriers), strict_body
        control_body = requests_by_goal["Complete the no-explicit-effort control"][0]
        assert control_body["model"] == "valid-control-model"
        control_carriers = [control_body.get("reasoning_effort"),
                            (control_body.get("extra_body") or {}).get("reasoning"),
                            control_body.get("reasoning")]
        assert all(carrier != "high" and not (isinstance(carrier, dict) and carrier.get("effort") == "high")
                   for carrier in control_carriers), control_body
        assert all(secret not in json.dumps(result) for secret in ("fixture-parent-key",))
        assert {path: path.read_bytes() for path in protected} == protected
        assert tuple(getattr(parent, name, None) for name in (
            "provider", "model", "base_url", "api_key", "_cached_system_prompt", "reasoning_config",
        )) == parent_before


def test_disabled_unavailable_and_static_conflict_routes_refuse_before_inference_sibling_runs(tmp_path, monkeypatch):
    """Disabled/unavailable providers and known static pair conflicts refuse independently."""
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with FakeLLMServer([Text("valid sibling completed")], api_key="valid-sibling-key") as server:
        home = tmp_path / ".hermes"
        write_hermes_home(home, server.base_url, api_key="parent-secret")
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_HOME", str(home))

        profiles = {
            "disabled-worker": ("disabled-secret", {"gemini": {"enabled": False}}),
            "unavailable-worker": (None, {}),
            "conflict-worker": ("conflict-secret", {}),
            "valid-worker": ("valid-sibling-key", {}),
        }
        protected = {}
        for name, (key, providers) in profiles.items():
            profile = home / "profiles" / name
            profile.mkdir(parents=True)
            if name == "valid-worker":
                _write_profile_auth_provider(profile, server.base_url)
                env_name = "FIXTURE_PROFILE_API_KEY"
            else:
                (profile / "config.yaml").write_text(yaml.safe_dump({
                    "model": {"provider": "custom", "default": "fake-model", "base_url": server.base_url},
                    "providers": providers,
                    "agent": {"api_max_retries": 1},
                }), encoding="utf-8")
                env_name = "ANTHROPIC_API_KEY" if name in {"disabled-worker", "conflict-worker"} else "OPENAI_API_KEY"
            (profile / ".env").write_text(f"{env_name}={key}\n" if key else "", encoding="utf-8")
            protected[profile / "config.yaml"] = (profile / "config.yaml").read_bytes()
            protected[profile / ".env"] = (profile / ".env").read_bytes()

        parent = AIAgent(
            provider="custom", model="fake-model", base_url=server.base_url, api_key="parent-secret",
            enabled_toolsets=["file"], quiet_mode=True, skip_context_files=True, skip_memory=True,
            save_trajectories=False,
        )
        try:
            raw = delegate_task(tasks=[
                {"goal": "Disabled provider must not run", "profile": "disabled-worker",
                 "model": "fake-model", "provider": "gemini"},
                {"goal": "Unavailable provider must not run", "profile": "unavailable-worker",
                 "model": "fake-model", "provider": "not-a-real-provider-mrefusal1"},
                {"goal": "Known static conflict must not run", "profile": "conflict-worker",
                 "model": "gpt-6-luna", "provider": "gemini"},
                {"goal": "Valid sibling runs", "profile": "valid-worker"},
            ], parent_agent=parent)
        finally:
            parent.close()

        result = json.loads(raw)
        assert "results" in result, result
        entries = {entry["task_index"]: entry for entry in result["results"]}
        for index in (0, 1, 2):
            assert entries[index]["status"] == "error", entries[index]
            assert entries[index]["api_calls"] == 0, entries[index]
            assert entries[index]["route"]["refused"] is True, entries[index]
        assert "disabled" in entries[0]["error"].lower()
        assert "not-a-real-provider-mrefusal1" in json.dumps(entries[1])
        assert "gpt-6-luna" in entries[2]["error"] and "gemini" in entries[2]["error"]
        assert entries[3]["status"] == "completed", entries[3]
        assert entries[3]["summary"] == "valid sibling completed"

        main = [request for request in server.requests if request["kind"] == "main"]
        assert len(main) == 1
        assert main[0]["auth"] == "Bearer valid-sibling-key"
        assert main[0]["body"]["model"] == "fake-model"
        assert "Disabled provider must not run" not in json.dumps(main)
        assert "Unavailable provider must not run" not in json.dumps(main)
        assert "Known static conflict must not run" not in json.dumps(main)
        assert all(secret not in json.dumps(result) for secret in (
            "parent-secret", "disabled-secret", "conflict-secret", "valid-sibling-key",
        ))
        assert {path: path.read_bytes() for path in protected} == protected


def test_registered_mode_hook_failure_preserves_exact_effort_and_valid_sibling(tmp_path, monkeypatch, caplog):
    """A failed registered mode hook leaves one exact child unverified without blocking its sibling."""
    import logging
    import providers
    from providers.base import ProviderProfile

    sentinel = "MODE_HOOK_INTEGRATION_SECRET_SENTINEL"
    responses = {
        "exact-m-hook-model": "exact child completed",
        "valid-sibling-model": "valid sibling completed",
    }

    class RaisingModeProfile(ProviderProfile):
        def supported_reasoning_efforts(self, model):
            return ("high",)

        def api_mode_for_model(self, model, *, api_key=None):
            raise RuntimeError(sentinel)

        def build_api_kwargs_extras(self, *, reasoning_config=None, **context):
            if reasoning_config is not None:
                return {}, {"reasoning_effort": reasoning_config["effort"]}
            return {}, {}

    monkeypatch.setattr("agent.safe_worker_policy.safe_worker_enabled", lambda: False)
    providers.list_providers()
    previous_profile = providers._REGISTRY.get("custom")
    previous_source = providers._SOURCES.get("custom")
    profile = RaisingModeProfile(name="custom", api_mode="chat_completions")
    providers.register_provider(profile)
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))

    try:
        with FakeLLMServer(lambda request: Text(responses[request["body"]["model"]]),
                           api_key="fixture-parent-secret") as server:
            write_hermes_home(home, server.base_url, api_key="fixture-parent-secret")
            protected = {path: path.read_bytes() for path in (home / "config.yaml", home / ".env")}
            parent = AIAgent(
                provider="custom", model="parent-model", base_url=server.base_url, api_key="fixture-parent-secret",
                enabled_toolsets=["file"], quiet_mode=True, skip_context_files=True, skip_memory=True,
                save_trajectories=False,
            )
            parent_before = tuple(getattr(parent, name, None) for name in (
                "provider", "model", "base_url", "api_key", "_cached_system_prompt", "reasoning_config",
            ))
            try:
                with caplog.at_level(logging.DEBUG):
                    raw = delegate_task(tasks=[
                        {"goal": "Keep this exact high effort", "model": "exact-m-hook-model",
                         "reasoning_effort": "high"},
                        {"goal": "Run the valid sibling", "model": "valid-sibling-model"},
                    ], parent_agent=parent)
            finally:
                parent.close()

            result = json.loads(raw)
            entries = {entry.get("task_index", index): entry for index, entry in enumerate(result["results"])}
            exact, sibling = entries[0], entries[1]
            assert exact["status"] == "completed", exact
            assert exact["summary"] == "exact child completed"
            assert exact["route"]["validation"] == "unverified"
            assert "preserving the requested effort exactly" in exact["route"]["diagnostic"]
            assert exact["route"]["requested_model"] == "exact-m-hook-model"
            assert exact["route"]["resolved_model"] == "exact-m-hook-model"
            assert exact["route"]["resolved_provider"] == "custom"
            assert exact["route"]["reasoning_effort"] == "high"
            assert sibling["status"] == "completed", sibling
            assert sibling["summary"] == "valid sibling completed"

            requests = server.main_requests()
            assert len(requests) == 2
            request_by_model = {request["model"]: request for request in requests}
            exact_request = request_by_model["exact-m-hook-model"]
            carriers = [exact_request.get("reasoning_effort"),
                        (exact_request.get("extra_body") or {}).get("reasoning"),
                        exact_request.get("reasoning")]
            assert any(carrier == "high" or (isinstance(carrier, dict) and carrier.get("effort") == "high")
                       for carrier in carriers), exact_request
            serialized = json.dumps((result, requests, [record.getMessage() for record in caplog.records]))
            assert sentinel not in serialized
            assert "fixture-parent-secret" not in serialized
            assert not any(record.exc_info for record in caplog.records)
            assert {path: path.read_bytes() for path in protected} == protected
            assert tuple(getattr(parent, name, None) for name in (
                "provider", "model", "base_url", "api_key", "_cached_system_prompt", "reasoning_config",
            )) == parent_before
    finally:
        if previous_profile is None:
            providers._REGISTRY.pop("custom", None)
        else:
            providers._REGISTRY["custom"] = previous_profile
        if previous_source is None:
            providers._SOURCES.pop("custom", None)
        else:
            providers._SOURCES["custom"] = previous_source
        providers._PROVIDER_LIST_CACHE = None


def test_registered_effort_hook_failure_preserves_exact_wire_and_sibling(tmp_path, monkeypatch, caplog):
    """A registered effort-hook failure stays unverified without changing the request or sibling."""
    from unittest.mock import patch

    import logging
    import providers
    from providers.base import ProviderProfile

    sentinel = "EFFORT_HOOK_INTEGRATION_SECRET_SENTINEL"
    hook_calls = []
    class EffortHookFailure(Exception):
        pass

    class RaisingEffortProfile(ProviderProfile):
        def supported_reasoning_efforts(self, model):
            hook_calls.append(model)
            if len(hook_calls) == 1:
                raise EffortHookFailure(sentinel)
            return ("high",)

        def api_mode_for_model(self, model, *, api_key=None):
            return "chat_completions"

        def build_api_kwargs_extras(self, *, reasoning_config=None, **context):
            return ({}, {"reasoning_effort": reasoning_config["effort"]}) if reasoning_config else ({}, {})

    monkeypatch.setattr("agent.safe_worker_policy.safe_worker_enabled", lambda: False)
    providers.list_providers()
    previous_profile = providers._REGISTRY.get("custom")
    previous_source = providers._SOURCES.get("custom")
    profile = RaisingEffortProfile(name="custom", api_mode="chat_completions")
    providers.register_provider(profile)
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    try:
        with FakeLLMServer(lambda request: Text(f"completed {request['body']['model']}"),
                           api_key="fixture-effort-parent-secret") as server:
            write_hermes_home(home, server.base_url, api_key="fixture-effort-parent-secret")
            parent = AIAgent(
                provider="custom", model="parent-model", base_url=server.base_url,
                api_key="fixture-effort-parent-secret", enabled_toolsets=["file"], quiet_mode=True,
                skip_context_files=True, skip_memory=True, save_trajectories=False,
            )
            try:
                with caplog.at_level(logging.DEBUG):
                    raw = delegate_task(tasks=[
                        {"goal": "Preserve this exact effort", "model": "exact-effort-hook-model",
                         "reasoning_effort": "high"},
                        {"goal": "Complete the unaffected sibling", "model": "effort-hook-sibling-model"},
                    ], parent_agent=parent)
            finally:
                parent.close()

            result = json.loads(raw)
            entries = {entry.get("task_index", index): entry for index, entry in enumerate(result["results"])}
            exact, sibling = entries[0], entries[1]
            assert exact["status"] == "completed", exact
            assert exact["route"]["validation"] == "unverified"
            assert exact["route"]["reasoning_effort"] == "high"
            assert "EffortHookFailure" in exact["route"]["diagnostic"]
            assert "preserving the requested effort exactly" in exact["route"]["diagnostic"]
            assert sibling["status"] == "completed", sibling

            requests = server.main_requests()
            assert {request["model"] for request in requests} == {
                "exact-effort-hook-model", "effort-hook-sibling-model",
            }
            exact_request = next(request for request in requests if request["model"] == "exact-effort-hook-model")
            carriers = [exact_request.get("reasoning_effort"),
                        (exact_request.get("extra_body") or {}).get("reasoning"),
                        exact_request.get("reasoning")]
            assert any(carrier == "high" or (isinstance(carrier, dict) and carrier.get("effort") == "high")
                       for carrier in carriers), exact_request
            serialized = json.dumps((result, requests, [record.getMessage() for record in caplog.records]))
            assert sentinel not in serialized
            assert "fixture-effort-parent-secret" not in serialized
            assert not any(record.exc_info for record in caplog.records)
    finally:
        if previous_profile is None:
            providers._REGISTRY.pop("custom", None)
        else:
            providers._REGISTRY["custom"] = previous_profile
        if previous_source is None:
            providers._SOURCES.pop("custom", None)
        else:
            providers._SOURCES["custom"] = previous_source
        providers._PROVIDER_LIST_CACHE = None


def test_catalog_validator_exception_attempts_exact_route_without_leaking(tmp_path, monkeypatch, caplog):
    """A catalog owner's exception is inconclusive; exact selected model runs without error leakage."""
    import logging
    from unittest.mock import patch

    sentinel = "CATALOG_VALIDATOR_INTEGRATION_SECRET_SENTINEL"
    class CatalogOwnerFailure(Exception):
        pass

    monkeypatch.setenv("HOME", str(tmp_path))
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_constants import named_profile_has_identity
    from tests.fakes.fake_llm_provider import write_hermes_home as setup_home
    with FakeLLMServer([Text("exact unverified route completed")], api_key="fixture-catalog-parent-secret") as server:
        setup_home(home, server.base_url, api_key="fixture-catalog-parent-secret")
        assert named_profile_has_identity(home)
        parent = AIAgent(
            provider="custom", model="parent-model", base_url=server.base_url,
            api_key="fixture-catalog-parent-secret", enabled_toolsets=["file"], quiet_mode=True,
            skip_context_files=True, skip_memory=True, save_trajectories=False,
        )
        try:
            with patch("hermes_cli.models_validate.validate_requested_model",
                       side_effect=CatalogOwnerFailure(sentinel)), caplog.at_level(logging.DEBUG):
                raw = delegate_task(tasks=[{
                    "goal": "Run this exact inconclusive route", "model": "exact-catalog-owner-model",
                }], parent_agent=parent)
        finally:
            parent.close()

        result = json.loads(raw)
        (entry,) = result["results"]
        assert entry["status"] == "completed", entry
        assert entry["summary"] == "exact unverified route completed"
        assert entry["route"]["validation"] == "unverified"
        assert entry["route"]["model"] == "exact-catalog-owner-model"
        assert "CatalogOwnerFailure" in entry["route"]["diagnostic"]
        assert "exact route will be attempted" in entry["route"]["diagnostic"]
        requests = server.main_requests()
        assert len(requests) == 1 and requests[0]["model"] == "exact-catalog-owner-model"
        serialized = json.dumps((result, requests, [record.getMessage() for record in caplog.records]))
        assert sentinel not in serialized
        assert "fixture-catalog-parent-secret" not in serialized
        assert not any(record.exc_info for record in caplog.records)


# Combined dynamic-option acceptance fixtures -------------------------------------------------


def _write_combined_profile(home: Path, endpoint: str, key: str, *, provider: str = "custom", model: str = "fixture-model"):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": provider, "default": model, "base_url": endpoint},
        "agent": {"api_max_retries": 1},
        "skills": {"disabled": []},
        "telemetry": {"shared_metrics": {"enabled": False, "send": False}},
    }), encoding="utf-8")
    (home / ".env").write_text(f"COMBINED_ROUTE_KEY={key}\n", encoding="utf-8")


def _combined_parent(home: Path, endpoint: str, db, *, session_id: str = "combined-parent", tools=None,
                    parent_key: str = "parent-loopback-secret"):
    agent = AIAgent(
        provider="custom", model="fixture-model", base_url=endpoint, api_key=parent_key,
        api_mode="chat_completions", enabled_toolsets=["delegation", "file"], quiet_mode=True,
        skip_context_files=True, skip_memory=True, save_trajectories=False, platform="cli",
        session_id=session_id, session_db=db, cwd=str(home),
    )
    agent._ensure_db_session()
    if tools is not None:
        from agent.delegation_launch import DelegationLaunchPolicy
        agent._delegation_launch_policy = DelegationLaunchPolicy(
            tools=tuple(tools), parent_tool_grant=tuple(tools), admitted_tool_names=tuple(tools))
        agent.tools = []
        agent.valid_tool_names = set(tools)
    return agent


def _write_combined_skill(home: Path, name: str, body: str):
    skill = home / "skills" / "research" / name
    skill.mkdir(parents=True, exist_ok=True)
    text = f"---\nname: {name}\ndescription: combined fixture\n---\n{body}\n"
    (skill / "SKILL.md").write_text(text, encoding="utf-8")
    return text


def _wait_for_completion(delegation_id: str, timeout: float = 10.0):
    """Read one real async completion and restore unrelated queued events."""
    import queue
    import time
    from tools.process_registry import process_registry

    completion_queue = process_registry.completion_queue
    deadline = time.monotonic() + timeout
    unmatched = []
    try:
        while time.monotonic() < deadline:
            try:
                event = completion_queue.get(timeout=min(0.05, max(0.0, deadline - time.monotonic())))
            except queue.Empty:
                continue
            if event.get("delegation_id") == delegation_id:
                return event
            unmatched.append(event)
    finally:
        for event in unmatched:
            completion_queue.put(event)
    raise AssertionError(f"delegation {delegation_id} did not complete")


def test_combined_joined_overrides_pin_wire_prompt_grant_and_transcript_owner(tmp_path, monkeypatch):
    """Joined real child carries every task override to the wire and writes only its own DB transcript."""
    from hermes_state import SessionDB
    with FakeLLMServer([Text('{"city":"Oslo"}'), Text('{"city":"Oslo","zip":"0170"}')],
                       api_key="combined-profile-key") as server:
        home = tmp_path / ".hermes"
        parent_home = home
        profile = home / "profiles" / "researcher"
        _write_combined_profile(home, server.base_url, "parent-config-secret")
        profile.mkdir(parents=True)
        (profile / "config.yaml").write_text(yaml.safe_dump({
            "model": {"provider": "custom", "default": "fixture-model", "base_url": server.base_url,
                      "key_env": "COMBINED_ROUTE_KEY"},
            "agent": {"api_max_retries": 1}, "skills": {"disabled": []},
        }), encoding="utf-8")
        (profile / ".env").write_text("COMBINED_ROUTE_KEY=combined-profile-key\n", encoding="utf-8")
        (profile / "SOUL.md").write_text("selected profile identity marker", encoding="utf-8")
        skill_wire = _write_combined_skill(profile, "combined-guide", "FROZEN COMBINED SKILL BODY")
        monkeypatch.setenv("HOME", str(tmp_path / "host-home"))
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)

        import hermes_state
        monkeypatch.setattr(hermes_state, "_STATE_DB_GUARD_EXTRA_DENY_ROOTS", ())
        db = SessionDB(db_path=parent_home / "state.db")
        parent = _combined_parent(home, server.base_url, db, session_id="combined-joined-parent",
                                  parent_key="parent-config-secret")
        from tools.delegate_tool import oneshot_delegate_function
        # The joined variant is the session's actual pinned tool schema; it remains joined even
        # when an explicit background request is passed to the public Python entry point.
        from tools.delegate_tool import DELEGATE_TASK_SCHEMA
        parent.tools = [{"type": "function", "function": oneshot_delegate_function({
            **DELEGATE_TASK_SCHEMA,
            "description": "Parallel children are joined and their results return in this tool call.\n\n",
        })}]
        parent._delegation_inline = True
        selected_tools = ["read_file"]
        goal = "Return a city object with a short explanation"
        persona = "\nExact task-specific persona marker 611\n"
        schema = {"type": "object", "properties": {"city": {"type": "string"},
                  "zip": {"type": "string"}}, "required": ["city", "zip"]}
        protected = {p: p.read_bytes() for p in (home / "config.yaml", home / ".env",
                                                  profile / "config.yaml", profile / ".env", profile / "SOUL.md",
                                                  profile / "skills" / "research" / "combined-guide" / "SKILL.md")}
        try:
            result = json.loads(delegate_task(tasks=[{
                "goal": goal, "profile": "researcher", "model": "fixture-model", "provider": "custom",
                "reasoning_effort": "low", "system_prompt": persona, "system_prompt_mode": "append",
                "skills": ["combined-guide"], "tools": selected_tools, "output_schema": schema,
            }], background=True, parent_agent=parent))
            assert result["results"][0]["status"] == "completed", (result, server.main_requests())
            entry = result["results"][0]
            assert entry["summary"] == '{"city":"Oslo","zip":"0170"}'
            assert entry["route"]["requested_profile"] == "researcher"
            assert entry["route"]["resolved_model"] == "fixture-model"
            assert entry["route"]["resolved_provider"] == "custom"
            assert entry["route"]["reasoning_effort"] == "low"
            assert entry["schema_valid"] is True and entry["schema_retries"] == 1
            assert result.get("note") is None
            assert len(server.main_requests()) == 2
            requests = server.main_requests()
            assert all(req["model"] == "fixture-model" for req in requests)
            assert all(req.get("reasoning_effort") == "low" for req in requests)
            authenticated = [request for request in server.requests if request["kind"] == "main"]
            assert len(authenticated) == 2
            assert all(request["auth"] == "Bearer combined-profile-key" for request in authenticated)
            first_prompt = requests[0]["messages"][0]["content"]
            retry_prompt = requests[1]["messages"][0]["content"]
            for prompt in (first_prompt, retry_prompt):
                assert persona in prompt
                assert "selected profile identity marker" in prompt
                assert skill_wire in prompt
            assert any(tool["function"]["name"] == "read_file" for tool in requests[0].get("tools", []))
            assert all(tool["function"]["name"] == "read_file" for tool in requests[0].get("tools", []))
            assert "combined-profile-key" not in json.dumps((result, requests))
            assert "parent-config-secret" not in json.dumps((result, requests))
            assert {p: p.read_bytes() for p in protected} == protected
            children = [row for row in db.list_sessions_rich(source="subagent", include_subagents=True) if row["source"] == "subagent"]
            assert len(children) == 1, children
            child_sid = children[0]["id"]
            child_row = db.get_session(child_sid)
            assert child_row["parent_session_id"] == parent.session_id
            assert child_row["source"] == "subagent"
            transcript = db.get_messages_as_conversation(child_sid)
            assert any(message.get("role") == "user" and goal in str(message.get("content")) for message in transcript)
            assert any(message.get("role") == "assistant" and entry["summary"] in str(message.get("content"))
                       for message in transcript)
            assert db.get_messages_as_conversation(parent.session_id) == []
        finally:
            parent.close()
            db.close()


def test_combined_mixed_refusals_keep_valid_sibling_and_redact_diagnostics(tmp_path, monkeypatch):
    """Unsupported strict effort and exact-grant refusals remain local beside one valid sibling."""
    from hermes_state import SessionDB
    import providers
    from providers.base import ProviderProfile

    class RefusalProfile(ProviderProfile):
        def supported_reasoning_efforts(self, model):
            return () if model == "fixture-model" else None

        def api_mode_for_model(self, model, *, api_key=None):
            return "chat_completions"

    providers.list_providers()
    old_profile, old_source = providers._REGISTRY.get("custom"), providers._SOURCES.get("custom")
    profile = RefusalProfile(name="custom", api_mode="chat_completions")
    providers.register_provider(profile)
    try:
        with FakeLLMServer([Text("valid sibling completed")],
                           api_key=("mixed-parent-key", "valid-profile-key")) as server:
            home = tmp_path / ".hermes"
            _write_combined_profile(home, server.base_url, "mixed-parent-key")
            valid_profile = home / "profiles" / "valid"
            valid_profile.mkdir(parents=True)
            (valid_profile / "config.yaml").write_text(yaml.safe_dump({
                "model": {"provider": "custom", "default": "fixture-model", "base_url": server.base_url,
                          "key_env": "COMBINED_ROUTE_KEY"},
                "agent": {"api_max_retries": 1},
            }), encoding="utf-8")
            (valid_profile / ".env").write_text("COMBINED_ROUTE_KEY=valid-profile-key\n", encoding="utf-8")
            _write_combined_skill(home / "skills", "not-used", "unused")
            monkeypatch.setenv("HOME", str(tmp_path / "host-home"))
            monkeypatch.setenv("HERMES_HOME", str(home))
            monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)

            import hermes_state
            monkeypatch.setattr(hermes_state, "_STATE_DB_GUARD_EXTRA_DENY_ROOTS", ())
            db = SessionDB(db_path=home / "state.db")
            parent = AIAgent(
                provider="custom", model="fixture-model", base_url=server.base_url,
                api_key="mixed-parent-key", api_mode="chat_completions",
                enabled_toolsets=["delegation", "file"], quiet_mode=True, skip_context_files=True,
                skip_memory=True, save_trajectories=False, platform="cli", session_id="mixed-parent", session_db=db,
            )
            parent._ensure_db_session()
            from agent.delegation_launch import DelegationLaunchPolicy
            parent._delegation_launch_policy = DelegationLaunchPolicy(
                tools=("read_file",), parent_tool_grant=("read_file",), admitted_tool_names=("read_file",))
            parent.valid_tool_names = {"read_file"}
            try:
                result = json.loads(delegate_task(tasks=[
                    {"goal": "unsupported strict effort refusal", "model": "fixture-model", "reasoning_effort": "high"},
                    {"goal": "ungranted exact tool refusal", "tools": ["write_file"]},
                    {"goal": "valid sibling", "profile": "valid", "model": "fixture-model"},
                ], parent_agent=parent))
                assert "results" in result, result
                entries = {entry["task_index"]: entry for entry in result["results"]}
                assert [entries[i]["status"] for i in range(2)] == ["error"] * 2
                assert all(entries[i]["api_calls"] == 0 for i in range(2))
                assert all(entries[i]["route"]["refused"] is True for i in range(2))
                assert "unsupported" in entries[0]["error"].lower()
                assert "outside" in entries[1]["error"].lower() or "parent" in entries[1]["error"].lower()
                assert entries[2]["status"] == "completed" and entries[2]["summary"] == "valid sibling completed"
                authenticated = [request for request in server.requests if request["kind"] == "main"]
                assert len(authenticated) == 1
                assert authenticated[0]["auth"] == "Bearer valid-profile-key"
                assert all(secret not in json.dumps(result) for secret in (
                    "mixed-parent-key", "valid-profile-key", "launch-process-key"))
                child_rows = [row for row in db.list_sessions_rich(source="subagent", include_subagents=True) if row["source"] == "subagent"]
                assert len(child_rows) == 1 and child_rows[0]["parent_session_id"] == parent.session_id
            finally:
                parent.close()
                db.close()
    finally:
        if old_profile is None:
            providers._REGISTRY.pop("custom", None)
        else:
            providers._REGISTRY["custom"] = old_profile
        if old_source is None:
            providers._SOURCES.pop("custom", None)
        else:
            providers._SOURCES["custom"] = old_source
        providers._PROVIDER_LIST_CACHE = None


def _prepare_combined_background(home, endpoint, db, monkeypatch, *, session_ids):
    """Build real owner agents and enable the public detached-delegation entry point."""
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": "custom", "default": "fixture-model", "base_url": endpoint,
                  "api_key": "parent-loopback-secret"},
        "agent": {"api_max_retries": 1},
        "delegation": {"worktree_isolation": False},
    }), encoding="utf-8")
    (home / ".env").write_text("", encoding="utf-8")
    host_home = home.parent / "host-home"
    host_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(host_home))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    import hermes_state
    monkeypatch.setattr(hermes_state, "_STATE_DB_GUARD_EXTRA_DENY_ROOTS", ())
    parents = []
    from tools.delegate_tool import DELEGATE_TASK_SCHEMA, oneshot_delegate_function
    from tools.delegate_tool import delegate_task as public_delegate
    for session_id in session_ids:
        parent = _combined_parent(home, endpoint, db, session_id=session_id,
                                  parent_key="parent-loopback-secret")
        assert "delegate_task" in parent.valid_tool_names
        parent.tools = [{"type": "function", "function": oneshot_delegate_function(DELEGATE_TASK_SCHEMA)}]
        parent._delegation_inline = False
        parents.append(parent)
    return parents, public_delegate


def test_combined_background_target_stop_is_local_and_preserves_sibling(tmp_path, monkeypatch):
    """The public stop control interrupts only its target; a sibling completes normally."""
    import threading
    from hermes_state import SessionDB
    from tests.fakes.fake_llm_provider import ToolCall

    target_tool_entered, release_target_tool, sibling_done = threading.Event(), threading.Event(), threading.Event()
    from tools.registry import registry
    previous_read_file = registry.snapshot_registration("read_file")
    assert previous_read_file is not None

    def respond(request):
        goal = " ".join(str(message.get("content", "")) for message in request["body"]["messages"]
                        if message.get("role") == "user")
        if "target child to stop" in goal:
            return ToolCall("read_file", {"path": "fixture-path"}, text="partial summary before stop")
        if "sibling must finish" in goal:
            sibling_done.set()
            return Text("sibling completed without stop")
        return Text("unexpected request")

    def blocked_read(_args, **_kwargs):
        target_tool_entered.set()
        assert release_target_tool.wait(timeout=15)
        return "target tool result after stop"

    with FakeLLMServer(respond, api_key=("parent-loopback-secret", "no-key-required")) as server:
        registry.register(name="read_file", toolset="file", schema=previous_read_file.schema,
                          handler=blocked_read, override=True)
        home = tmp_path / ".hermes"
        db = SessionDB(db_path=home / "state.db")
        parents, public_delegate = _prepare_combined_background(
            home, server.base_url, db, monkeypatch, session_ids=("target-stop-parent",),
        )
        parent, = parents
        protected = (home / "config.yaml").read_bytes()
        try:
            handle = json.loads(public_delegate(tasks=[
                {"goal": "target child to stop", "model": "fixture-model"},
                {"goal": "sibling must finish", "model": "fixture-model"},
            ], background=True, parent_agent=parent))
            assert handle["status"] == "dispatched" and len(handle["subagent_ids"]) == 2, handle
            target_id, sibling_id = handle["subagent_ids"]
            assert target_tool_entered.wait(timeout=10)
            assert sibling_done.wait(timeout=10)
            stopped = json.loads(public_delegate(action="stop", subagent_id=target_id, parent_agent=parent))
            assert stopped["status"] == "interrupt_requested"
            release_target_tool.set()
            event = _wait_for_completion(handle["delegation_id"], timeout=20)
            entries = {entry["task_index"]: entry for entry in event["results"]}
            assert entries[0]["status"] == "interrupted", entries
            assert entries[0]["summary"] == "partial summary before stop"
            assert entries[0]["route"]["resolved_model"] == "fixture-model"
            assert entries[1]["status"] == "completed" and entries[1]["summary"] == "sibling completed without stop"
            assert len(server.main_requests()) == 2
            target_requests = [request for request in server.main_requests()
                               if any(message.get("role") == "user" and "target child to stop" in str(message.get("content"))
                                      for message in request["messages"])]
            assert len(target_requests) == 1
            assert target_id != sibling_id
            assert db.get_session(parent.session_id) is not None
            rows = [row for row in db.list_sessions_rich(source="subagent", include_subagents=True)
                    if row["source"] == "subagent"]
            assert len(rows) == 2 and all(row["parent_session_id"] == parent.session_id for row in rows)
            assert (home / "config.yaml").read_bytes() == protected
            assert "parent-loopback-secret" not in json.dumps(event)
        finally:
            release_target_tool.set()
            for parent in parents:
                parent.close()
            db.close()
            current_registration = registry.snapshot_registration("read_file")
            if current_registration is not None:
                registry.restore_registration("read_file", current_registration, previous_read_file)


def test_combined_background_session_cancel_stops_owned_batch_not_foreign_owner(tmp_path, monkeypatch):
    """The actual owner-session interrupt cancels its batch, not another parent's active batch."""
    import threading
    from hermes_state import SessionDB

    gates = {name: threading.Event() for name in ("owned one", "owned two", "foreign")}
    entered = {name: threading.Event() for name in gates}

    def respond(request):
        goal = " ".join(str(message.get("content", "")) for message in request["body"]["messages"]
                        if message.get("role") == "user")
        name = next(name for name in gates if name in goal)
        entered[name].set()
        assert gates[name].wait(timeout=15)
        return Text(f"{name} finished")

    with FakeLLMServer(respond, api_key=("parent-loopback-secret", "no-key-required")) as server:
        home = tmp_path / ".hermes"
        db = SessionDB(db_path=home / "state.db")
        parents, public_delegate = _prepare_combined_background(
            home, server.base_url, db, monkeypatch, session_ids=("owned-session-parent", "foreign-session-parent"),
        )
        owned_parent, foreign_parent = parents
        try:
            owned = json.loads(public_delegate(tasks=[
                {"goal": "owned one waits at its cancellation barrier", "model": "fixture-model"},
                {"goal": "owned two waits at its cancellation barrier", "model": "fixture-model"},
            ], background=True, parent_agent=owned_parent))
            foreign = json.loads(public_delegate(tasks=[
                {"goal": "foreign owner child remains active", "model": "fixture-model"},
            ], background=True, parent_agent=foreign_parent))
            assert owned.get("status") == foreign.get("status") == "dispatched", (owned, foreign)
            assert all(event.wait(timeout=10) for event in entered.values())
            from tools.async_delegation import interrupt_for_session
            assert interrupt_for_session(parent_session_id=owned_parent.session_id) == 1
            # Keep the owner-cancelled children at the real barrier until their interrupt is observed;
            # the foreign child remains free to continue independently.
            from tools.delegate_tool_registry import _active_subagents
            assert all(_active_subagents[sid]["agent"]._interrupt_requested for sid in owned["subagent_ids"])
            assert foreign["subagent_ids"][0] in _active_subagents
            assert not _active_subagents[foreign["subagent_ids"][0]]["agent"]._interrupt_requested
            for gate in gates.values():
                gate.set()
            owned_event = _wait_for_completion(owned["delegation_id"], timeout=20)
            foreign_event = _wait_for_completion(foreign["delegation_id"], timeout=20)
            assert [entry["status"] for entry in owned_event["results"]] == ["interrupted", "interrupted"]
            assert foreign_event["results"][0]["status"] == "completed"
            assert len(server.main_requests()) == 3
            assert all(db.get_session(parent.session_id) is not None for parent in parents)
            rows = [row for row in db.list_sessions_rich(source="subagent", include_subagents=True)
                    if row["source"] == "subagent"]
            assert len(rows) == 3
            assert sum(row["parent_session_id"] == owned_parent.session_id for row in rows) == 2
            assert sum(row["parent_session_id"] == foreign_parent.session_id for row in rows) == 1
        finally:
            for gate in gates.values():
                gate.set()
            for parent in parents:
                parent.close()
            db.close()


def test_combined_background_steer_and_completion_keep_route_metadata_and_ownership(tmp_path, monkeypatch):
    """A live steer enters the next real model request once; completion and durable ownership retain overrides."""
    import queue
    import sqlite3
    import threading
    from hermes_state import SessionDB
    from tests.fakes.fake_llm_provider import ToolCall
    from tests.waiting import wait_until
    from tools.registry import registry

    tool_entered, release_tool = threading.Event(), threading.Event()
    with FakeLLMServer([ToolCall("read_file", {"path": "fixture-path"}),
                        Text('{"answer":"steer incorporated"}')],
                       api_key=("combined-profile-key", "parent-loopback-secret", "no-key-required")) as server:
        home = tmp_path / ".hermes"
        db = SessionDB(db_path=home / "state.db")
        parents, public_delegate = _prepare_combined_background(
            home, server.base_url, db, monkeypatch, session_ids=("combined-background-parent",),
        )
        parent, = parents
        from tools.process_registry import process_registry
        local_completions = queue.Queue()
        monkeypatch.setattr(process_registry, "completion_queue", local_completions)
        unrelated_event = {"type": "async_delegation", "delegation_id": "unrelated-completion"}
        local_completions.put(unrelated_event)
        profile = home / "profiles" / "researcher"
        profile.mkdir(parents=True)
        (profile / "config.yaml").write_text(yaml.safe_dump({
            "model": {"provider": "custom", "default": "fixture-model", "base_url": server.base_url,
                      "key_env": "COMBINED_ROUTE_KEY"},
            "agent": {"api_max_retries": 1},
            "platform_toolsets": {"cli": ["file", "delegation"]},
        }), encoding="utf-8")
        (profile / ".env").write_text("COMBINED_ROUTE_KEY=combined-profile-key\n", encoding="utf-8")
        (profile / "SOUL.md").write_text("BACKGROUND PROFILE IDENTITY", encoding="utf-8")
        skill_wire = _write_combined_skill(profile, "background-guide", "BACKGROUND FROZEN SKILL")
        from hermes_cli.profiles import profile_exists
        assert profile_exists("researcher")
        previous_registration = registry.snapshot_registration("read_file")
        assert previous_registration is not None

        def blocked_read(_args, **_kwargs):
            tool_entered.set()
            assert release_tool.wait(timeout=15)
            return "STEER_TOOL_RESULT"

        registry.register(name="read_file", toolset="file", schema=previous_registration.schema,
                          handler=blocked_read, override=True)
        protected = {path: path.read_bytes() for path in (
            home / "config.yaml", profile / "config.yaml", profile / ".env", profile / "SOUL.md",
            profile / "skills" / "research" / "background-guide" / "SKILL.md",
        )}
        try:
            schema = {"type": "object", "properties": {"answer": {"type": "string"}},
                      "required": ["answer"]}
            handle = json.loads(public_delegate(tasks=[{
                "goal": "return the requested JSON answer", "profile": "researcher", "provider": "custom",
                "model": "fixture-model", "reasoning_effort": "low", "system_prompt": "BACKGROUND APPENDED PERSONA",
                "system_prompt_mode": "append", "skills": ["background-guide"], "tools": ["read_file"],
                "output_schema": schema,
            }], background=True, parent_agent=parent))
            assert handle["status"] == "dispatched" and len(handle["subagent_ids"]) == 1, handle
            child_id = handle["subagent_ids"][0]
            assert tool_entered.wait(timeout=10)
            steered = json.loads(public_delegate(action="steer", subagent_id=child_id,
                                                  message="Include the requested answer verbatim.",
                                                  parent_agent=parent))
            assert steered["status"] == "queued"
            release_tool.set()
            event = _wait_for_completion(handle["delegation_id"], timeout=25)
            assert local_completions.get_nowait() == unrelated_event
            (entry,) = event["results"]
            assert entry["status"] == "completed" and entry["summary"] == '{"answer":"steer incorporated"}'
            assert entry["route"]["requested_profile"] == "researcher"
            assert entry["route"]["resolved_model"] == "fixture-model"
            assert entry["route"]["resolved_provider"] == "custom"
            assert entry["route"]["reasoning_effort"] == "low"
            assert entry["schema_valid"] is True and "schema_retries" not in entry
            assert event["parent_session_id"] == parent.session_id
            assert event["model"] == "fixture-model"
            requests = server.main_requests()
            assert len(requests) == 2 and all(request["model"] == "fixture-model" for request in requests)
            assert all(request.get("reasoning_effort") == "low" for request in requests)
            assert all(request["tools"] and {tool["function"]["name"] for tool in request["tools"]} == {"read_file"}
                       for request in requests)
            assert all("BACKGROUND APPENDED PERSONA" in request["messages"][0]["content"]
                       and "BACKGROUND PROFILE IDENTITY" in request["messages"][0]["content"]
                       and skill_wire in request["messages"][0]["content"] for request in requests)
            second_messages = requests[1]["messages"]
            tool_index = next(i for i, message in enumerate(second_messages)
                              if message.get("role") == "tool" and "STEER_TOOL_RESULT" in str(message.get("content")))
            steer_rows = [(i, message) for i, message in enumerate(second_messages)
                          if message.get("role") == "user" and "Include the requested answer verbatim." in str(message.get("content"))]
            assert len(steer_rows) == 1 and steer_rows[0][0] > tool_index
            assert json.dumps(second_messages).count("Include the requested answer verbatim.") == 1
            child_rows = [row for row in db.list_sessions_rich(source="subagent", include_subagents=True)
                          if row["source"] == "subagent"]
            assert len(child_rows) == 1 and child_rows[0]["parent_session_id"] == parent.session_id
            transcript = db.get_messages_as_conversation(child_rows[0]["id"])
            assert any(message.get("role") == "user" and "Include the requested answer verbatim." in str(message.get("content"))
                       for message in transcript)
            assert db.get_messages_as_conversation(parent.session_id) == []
            with sqlite3.connect(home / "state.db") as connection:
                durable = connection.execute(
                    "SELECT state,event_json,result_json,parent_session_id,origin_session_id "
                    "FROM async_delegations WHERE delegation_id=?", (handle["delegation_id"],),
                ).fetchone()
            assert durable is not None and durable[0] == "completed"
            assert durable[3] == parent.session_id
            durable_event, durable_result = json.loads(durable[1]), json.loads(durable[2])
            assert durable_event["status"] == "completed"
            assert durable_event["model"] == "fixture-model"
            assert durable_event["results"][0]["route"]["requested_profile"] == "researcher"
            assert durable_event["results"][0]["route"]["reasoning_effort"] == "low"
            assert durable_event["parent_session_id"] == parent.session_id
            assert durable_event["origin_session_id"] == durable[4]
            assert durable_result["results"][0]["route"]["requested_profile"] == "researcher"
            assert durable_result["results"][0]["schema_valid"] is True
            assert all(secret not in json.dumps((event, durable_event, durable_result, requests))
                       for secret in ("combined-profile-key", "parent-loopback-secret"))
            assert {path: path.read_bytes() for path in protected} == protected
            assert wait_until(lambda: child_id not in __import__("tools.delegate_tool_registry", fromlist=["_active_subagents"])._active_subagents,
                              timeout=5)
        finally:
            release_tool.set()
            parent.close()
            db.close()
            current_registration = registry.snapshot_registration("read_file")
            if current_registration is not None:
                registry.restore_registration("read_file", current_registration, previous_registration)

def test_external_process_refuses_unenforceable_task_controls_before_spawn(tmp_path, monkeypatch):
    """ACP cannot enforce task persona/tool/skill/reasoning launch controls; valid sibling runs."""
    from tests.fakes.providers import fake_acp_agent as acp

    monkeypatch.setenv("HOME", str(tmp_path))
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    profile_home = home / "profiles" / "acp-worker"
    profile_home.mkdir(parents=True)
    fake = acp.AcpFake(tmp_path / "acp-state", [
        [acp.message("persona control escaped preflight")],
        [acp.message("tool control escaped preflight")],
        [acp.message("skill control escaped preflight")],
        [acp.message("reasoning control escaped preflight")],
    ])
    monkeypatch.setenv(acp.COMMAND_ENV_VAR, str(fake.launcher))
    monkeypatch.setenv(acp.ARGS_ENV_VAR, f"--acp --stdio --state {fake.state_dir}")
    acp.install_provider_plugin(profile_home)
    (profile_home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": acp.PROVIDER, "default": acp.PLACEHOLDER_MODEL},
        "skills": {"disabled": []},
        "agent": {"api_max_retries": 1},
    }), encoding="utf-8")
    (profile_home / ".env").write_text("\n".join(f"{key}={value}" for key, value in fake.env().items()) + "\n",
                                    encoding="utf-8")
    skill = profile_home / "skills" / "research" / "visible"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: visible\ndescription: readable synthetic skill\n---\nVALID SKILL BODY\n", encoding="utf-8")
    from hermes_constants import named_profile_has_identity
    assert named_profile_has_identity(profile_home)

    with FakeLLMServer([Text("valid HTTP sibling completed")], api_key="parent-route-secret") as server:
        write_hermes_home(home, server.base_url, api_key="parent-route-secret")
        with FakeLLMServer([Text("explicit API profile completed")], api_key="profile-api-secret") as profile_server:
            (profile_home / "config.yaml").write_text(yaml.safe_dump({
                "model": {"provider": acp.PROVIDER, "default": acp.PLACEHOLDER_MODEL},
                "providers": {"fixture-api": {"api": profile_server.base_url,
                                                "api_key": "profile-api-secret"}},
                "skills": {"disabled": []},
                "agent": {"api_max_retries": 1},
            }), encoding="utf-8")
            protected = {path: path.read_bytes() for path in (home / "config.yaml", home / ".env",
                                                               profile_home / "config.yaml", profile_home / ".env",
                                                               skill / "SKILL.md")}
            parent = AIAgent(
                provider="custom", model="fake-model", base_url=server.base_url, api_key="parent-route-secret",
                enabled_toolsets=["file"], quiet_mode=True, skip_context_files=True, skip_memory=True,
                save_trajectories=False,
            )
            parent_prompt = parent._build_system_prompt().encode("utf-8")
            parent_tools = json.dumps(parent.tools, sort_keys=True, separators=(",", ":")).encode("utf-8")
            try:
                raw = delegate_task(tasks=[
                    {"goal": "persona must be refused", "profile": "acp-worker", "system_prompt": "persona marker"},
                    {"goal": "exact tools must be refused", "profile": "acp-worker", "tools": ["read_file"]},
                    {"goal": "readable skill must be refused", "profile": "acp-worker", "skills": ["visible"]},
                    {"goal": "strict reasoning must be refused", "profile": "acp-worker", "reasoning_effort": "high"},
                    {"goal": "valid HTTP sibling runs"},
                    {"goal": "explicit API route clears ACP command", "profile": "acp-worker",
                     "provider": "fixture-api", "model": "fake-model"},
                ], parent_agent=parent)
            finally:
                parent.close()

            result = json.loads(raw)
            assert "results" in result, result
            entries = {entry["task_index"]: entry for entry in result["results"]}
            for index, field in enumerate(("persona", "tool", "skill", "reasoning")):
                assert entries[index]["status"] == "error", entries[index]
                assert entries[index]["api_calls"] == 0, entries[index]
                assert entries[index]["route"]["refused"] is True, entries[index]
                assert field in entries[index]["error"].lower(), entries[index]
            assert entries[4]["status"] == "completed", entries[4]
            assert entries[4]["summary"] == "valid HTTP sibling completed"
            assert entries[5]["status"] == "completed", entries[5]
            assert entries[5]["summary"] == "explicit API profile completed"
            assert entries[5]["route"]["resolved_provider"] == "fixture-api"
            assert entries[5]["route"]["resolved_model"] == "fake-model"
            assert len(server.main_requests()) == 1
            server_request = server.requests[0]
            assert server_request["kind"] == "main"
            assert server_request["auth"] == "Bearer parent-route-secret"
            assert len(profile_server.main_requests()) == 1
            profile_request = profile_server.requests[0]
            assert profile_request["kind"] == "main"
            assert profile_request["auth"] == "Bearer profile-api-secret"
            assert fake.pids() == []
            assert fake.inbound() == []
            assert parent._build_system_prompt().encode("utf-8") == parent_prompt
            assert json.dumps(parent.tools, sort_keys=True, separators=(",", ":")).encode("utf-8") == parent_tools
            assert {path: path.read_bytes() for path in protected} == protected
            serialized = json.dumps((result, fake.records(), server.main_requests()))
            assert "parent-route-secret" not in serialized
            assert "profile-api-secret" not in serialized
            assert "VALID SKILL BODY" not in serialized


def test_ordinary_external_process_task_still_runs_over_real_acp(tmp_path, monkeypatch):
    """An ACP-selected profile without task launch overrides remains legacy-compatible."""
    from tests.fakes.providers import fake_acp_agent as acp

    monkeypatch.setenv("HOME", str(tmp_path))
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    fake = acp.AcpFake(tmp_path / "acp-legacy", [[acp.message("ordinary ACP route completed")]])
    monkeypatch.setenv(acp.COMMAND_ENV_VAR, str(fake.launcher))
    monkeypatch.setenv(acp.ARGS_ENV_VAR, f"--acp --stdio --state {fake.state_dir}")
    profile_home = home / "profiles" / "acp-worker"
    profile_home.mkdir(parents=True)
    acp.install_provider_plugin(profile_home)
    (profile_home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": acp.PROVIDER, "default": acp.PLACEHOLDER_MODEL},
    }), encoding="utf-8")
    (profile_home / ".env").write_text("\n".join(f"{key}={value}" for key, value in fake.env().items()) + "\n",
                                    encoding="utf-8")
    from hermes_constants import named_profile_has_identity
    assert named_profile_has_identity(profile_home)
    with FakeLLMServer([], api_key="parent-route-secret") as server:
        write_hermes_home(home, server.base_url, api_key="parent-route-secret")
        parent = AIAgent(
            provider="custom", model="fake-model", base_url=server.base_url, api_key="parent-route-secret",
            enabled_toolsets=["file"], quiet_mode=True, skip_context_files=True, skip_memory=True,
            save_trajectories=False,
        )
        try:
            raw = delegate_task(tasks=[{"goal": "ordinary ACP task", "profile": "acp-worker"}],
                                parent_agent=parent)
        finally:
            parent.close()
        prompts = fake.inbound("session/prompt")
        pids = fake.pids()

    result = json.loads(raw)
    (entry,) = result["results"]
    assert entry["status"] == "completed", entry
    assert entry["summary"] == "ordinary ACP route completed"
    assert entry["api_calls"] == 1
    assert len(pids) == 1
    assert len(prompts) == 1
    assert "ordinary ACP task" in acp.prompt_text(prompts[0])
    assert not server.main_requests()


def test_external_process_cannot_bypass_inherited_exact_tool_ceiling(tmp_path, monkeypatch):
    """Strict ancestor ceilings compose into ACP refusal before any inference."""
    from agent.delegation_launch import DelegationLaunchPolicy
    from tests.fakes.providers import fake_acp_agent as acp

    monkeypatch.setenv("HOME", str(tmp_path))
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    fake = acp.AcpFake(tmp_path / "acp-inherited-ceiling", [[acp.message("inherited ceiling escaped")]])
    monkeypatch.setenv(acp.COMMAND_ENV_VAR, str(fake.launcher))
    monkeypatch.setenv(acp.ARGS_ENV_VAR, f"--acp --stdio --state {fake.state_dir}")
    profile_home = home / "profiles" / "acp-worker"
    profile_home.mkdir(parents=True)
    acp.install_provider_plugin(profile_home)
    (profile_home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": acp.PROVIDER, "default": acp.PLACEHOLDER_MODEL},
    }), encoding="utf-8")
    (profile_home / ".env").write_text(
        "\n".join(f"{key}={value}" for key, value in fake.env().items()) + "\n", encoding="utf-8")
    from hermes_constants import named_profile_has_identity
    assert named_profile_has_identity(profile_home)

    with FakeLLMServer([], api_key="parent-route-secret") as server:
        write_hermes_home(home, server.base_url, api_key="parent-route-secret")
        parent = AIAgent(
            provider="custom", model="fake-model", base_url=server.base_url, api_key="parent-route-secret",
            enabled_toolsets=["file"], quiet_mode=True, skip_context_files=True, skip_memory=True,
            save_trajectories=False,
            delegation_launch_policy=DelegationLaunchPolicy(
                tools=("read_file",), parent_tool_grant=("read_file",), admitted_tool_names=("read_file",),
            ),
        )
        try:
            raw = delegate_task(tasks=[{"goal": "inherited exact grant must be refused", "profile": "acp-worker"}],
                                parent_agent=parent)
        finally:
            parent.close()

    result = json.loads(raw)
    assert "exact tool ceiling" in result.get("error", "").lower(), result
    assert fake.pids() == []
    assert fake.inbound() == []
    assert not server.main_requests()
