"""Real child startup and local inference for an explicitly empty tool list."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer



def test_exact_context_tool_scope_guards_registry_dispatch_and_resets():
    from tools.registry import registry
    from tools.thread_context import exact_tool_scope, current_exact_tool_scope
    import model_tools

    name = "exact_scope_fixture_tool"
    denied_name = "exact_scope_denied_fixture_tool"
    effects = []
    for registered_name in (name, denied_name):
        registry.register(name=registered_name, toolset="exact-scope-fixture", schema={
            "name": registered_name, "description": "Registered scope fixture",
            "parameters": {"type": "object", "properties": {}},
        }, handler=lambda args, _name=registered_name, **kwargs: effects.append(_name)
        or json.dumps({"ok": True}))
    try:
        assert current_exact_tool_scope() is None
        with exact_tool_scope({name}):
            assert current_exact_tool_scope() == frozenset({name})
            assert json.loads(registry.dispatch(name, {})) == {"ok": True}
            denied = json.loads(model_tools.handle_function_call(denied_name, {}))
            assert "outside this child's exact tool grant" in denied["error"]
            assert effects == [name]
        assert current_exact_tool_scope() is None
        with exact_tool_scope(set()):
            assert current_exact_tool_scope() == frozenset()
            denied = json.loads(model_tools.handle_function_call(name, {}))
            assert "empty exact tool grant" in denied["error"]
        assert effects == [name]
    finally:
        registry.deregister(name)
        registry.deregister(denied_name)


def test_exact_public_dispatch_rejects_ungranted_literal_before_legacy_alias():
    """A legacy alias cannot smuggle a different literal through an exact-name grant."""
    import json

    from tools.registry import registry
    from tools.thread_context import exact_tool_scope
    import model_tools

    name = "process_manage"
    prior = registry.snapshot_registration(name)
    effects = []
    assert prior is not None
    try:
        registry.register(
            name=name,
            toolset="terminal",
            schema={"name": name, "description": "Exact alias fixture",
                    "parameters": {"type": "object", "properties": {}}},
            handler=lambda args, **kwargs: effects.append(dict(args)) or json.dumps({"ok": True}),
            override=True,
        )
        with exact_tool_scope({name}):
            result = json.loads(model_tools.handle_function_call("process", {}))
        assert "outside this child's exact tool grant" in result["error"]
        assert effects == []
        # Ordinary sessions retain the old compatibility alias.
        assert json.loads(model_tools.handle_function_call("process", {})) == {"ok": True}
        assert effects == [{}]
    finally:
        current = registry.snapshot_registration(name)
        if current is not None:
            registry.restore_registration(name, current, prior)


def test_admitted_ceiling_is_authoritative_and_bridge_grants_are_explicit():
    from agent.delegation_launch import DelegationLaunchPolicy, task_tool_ceiling, tool_name_in_grant

    agent = type("Agent", (), {"_delegation_launch_policy": DelegationLaunchPolicy(
        tools=("read_file", "write_file", "tool_call"),
        admitted_tool_names=("read_file",),
    )})()
    assert task_tool_ceiling(agent) == frozenset({"read_file"})
    assert tool_name_in_grant(agent, "read_file")
    assert not tool_name_in_grant(agent, "write_file")
    assert not tool_name_in_grant(agent, "tool_call")


def test_direct_registered_mcp_grant_does_not_require_bridge_tools(tmp_path, monkeypatch):
    import json

    import hermes_yaml as yaml
    from agent.delegation_launch import DelegationLaunchPolicy
    from run_agent import AIAgent
    from tools.delegate_tool_options import TaskRouteOptions
    from tools.delegate_tool_validation import _exact_builtin_tool_grant_error
    from tools.registry import registry

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump({
        "tools": {"tool_search": {"enabled": "on", "defer": []}},
        "platform_toolsets": {"subagent": ["mcp-bridge-fixture"]},
    }), encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    name = "mcp__bridge_fixture__echo"
    prior = registry.snapshot_registration(name)
    effects = []
    assert prior is None
    registry.register(name=name, toolset="mcp-bridge-fixture", schema={
        "name": name, "description": "Fixture MCP echo", "parameters": {
            "type": "object", "properties": {"value": {"type": "string"}},
        },
    }, handler=lambda args, **kwargs: effects.append(args["value"]) or json.dumps({"echo": args["value"]}))
    parent = AIAgent(
        api_key="fixture-key", base_url="http://127.0.0.1:9/v1", provider="custom",
        api_mode="chat_completions", model="fixture-model", enabled_toolsets=["mcp-bridge-fixture"],
        quiet_mode=True, skip_context_files=True, skip_memory=True, save_trajectories=False,
        platform="subagent",
    )
    try:
        option, parse_error = TaskRouteOptions.parse({"tools": [name]}, 0)
        assert parse_error is None and option is not None
        route = type("Route", (), {"profile_home": None})()
        error = _exact_builtin_tool_grant_error(0, option, (name,), route, {}, parent)
        assert error is None, error
        child = AIAgent(
            api_key="fixture-key", base_url="http://127.0.0.1:9/v1", provider="custom",
            api_mode="chat_completions", model="fixture-model", enabled_toolsets=["mcp-bridge-fixture"],
            quiet_mode=True, skip_context_files=True, skip_memory=True, save_trajectories=False,
            platform="subagent", delegation_launch_policy=DelegationLaunchPolicy(
                tools=(name,), parent_tool_grant=(name,), admitted_tool_names=(name,),
            ),
        )
        try:
            assert child.valid_tool_names == {name}
            assert json.loads(child._invoke_tool(name, {"value": "direct"}, "direct-mcp")) == {"echo": "direct"}
            assert effects == ["direct"]
        finally:
            child.close()
    finally:
        parent.close()
        current = registry.snapshot_registration(name)
        if current is not None:
            registry.restore_registration(name, current, prior)


from contextvars import copy_context
from threading import Event, get_ident
from concurrent.futures import ThreadPoolExecutor


def test_exact_context_scopes_propagate_intersect_and_reset_on_reused_executor():
    from tools.thread_context import current_exact_tool_scope, exact_tool_scope, propagate_context_to_thread

    barrier = Event()
    entered = Event()
    observed = {}

    def worker(label, inner):
        observed[(label, "outer")] = current_exact_tool_scope()
        with exact_tool_scope(inner):
            observed[(label, "inner")] = current_exact_tool_scope()
        observed[(label, "restored")] = current_exact_tool_scope()
        if label == "first":
            entered.set()
            assert barrier.wait(timeout=3)
        return get_ident()

    with ThreadPoolExecutor(max_workers=1) as pool:
        with exact_tool_scope({"read_file", "write_file"}):
            first = pool.submit(propagate_context_to_thread(worker), "first", {"read_file"})
        assert entered.wait(timeout=3)
        with exact_tool_scope({"read_file", "terminal"}):
            second = pool.submit(copy_context().run, propagate_context_to_thread(worker), "second", {"terminal"})
        barrier.set()
        assert first.result(timeout=3) == second.result(timeout=3)
        legacy = pool.submit(propagate_context_to_thread(lambda: current_exact_tool_scope())).result(timeout=3)

    assert observed == {
        ("first", "outer"): frozenset({"read_file", "write_file"}),
        ("first", "inner"): frozenset({"read_file"}),
        ("first", "restored"): frozenset({"read_file", "write_file"}),
        ("second", "outer"): frozenset({"read_file", "terminal"}),
        ("second", "inner"): frozenset({"terminal"}),
        ("second", "restored"): frozenset({"read_file", "terminal"}),
    }
    assert legacy is None
    assert current_exact_tool_scope() is None


def test_exact_scope_resets_after_callback_exception():
    from tools.thread_context import current_exact_tool_scope, exact_tool_scope, propagate_context_to_thread

    def fail():
        assert current_exact_tool_scope() == frozenset({"read_file"})
        raise RuntimeError("expected callback failure")

    with exact_tool_scope({"read_file"}):
        callback = propagate_context_to_thread(fail)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(callback)
        try:
            future.result(timeout=3)
        except RuntimeError as exc:
            assert str(exc) == "expected callback failure"
        else:
            raise AssertionError("exception callback returned")
        assert pool.submit(propagate_context_to_thread(current_exact_tool_scope)).result(timeout=3) is None
    assert current_exact_tool_scope() is None


def test_public_connector_batch_applies_exact_grants_per_entry(monkeypatch):
    import model_tools
    from tools.connectors.gateway import bridge, config
    from tools.thread_context import current_exact_tool_scope, exact_tool_scope
    from tools.registry import invalidate_check_fn_cache

    allowed = "connectors__gmail__FETCH_EMAILS"
    denied = "connectors__slack__POST_MESSAGE"
    executed = []
    grant = frozenset({"tool_call", "manage_connections", allowed})
    monkeypatch.setattr(config, "connectors_available", lambda: True)
    monkeypatch.setattr(bridge, "connectors_available", lambda: True)
    invalidate_check_fn_cache()

    class ExternalClient:
        def schemas(self, slugs):
            return {"schemas": {slug: {"input_schema": {}} for slug in slugs}}

        def execute(self, planned, **kwargs):
            executed.extend(planned)
            return [{"data": {"tool": item.name}, "error": None} for item in planned]

    monkeypatch.setattr(bridge, "_default_client_factory", lambda call=None: ExternalClient())
    with exact_tool_scope(grant):
        result = json.loads(model_tools.handle_function_call(
            "tool_call", {"calls": [
                {"name": allowed, "arguments": {"q": "yes"}},
                {"name": denied, "arguments": {"body": "no"}},
            ]}, enabled_toolsets=["connections"], session_id="connector-session", tool_call_id="batch-id",
            skip_pre_tool_call_hook=True, skip_tool_request_middleware=True,
            skip_tool_execution_middleware=True))

    assert "results" in result, result
    assert [(entry["index"], entry["name"]) for entry in result["results"]] == [(0, allowed), (1, denied)]
    assert result["results"][0]["response"] == {"tool": allowed}
    assert result["results"][1]["error"]["code"] == "TOOL_FORBIDDEN"
    assert [plan.name for plan in executed] == [allowed]
    assert current_exact_tool_scope() is None


def test_public_mcp_bridge_search_describe_invoke_use_exact_registered_grants(tmp_path, monkeypatch):
    """Real registered MCP handlers prove bridge catalogs and effects honor literal grants."""
    import json

    import hermes_yaml as yaml
    from agent.delegation_launch import DelegationLaunchPolicy
    from run_agent import AIAgent
    from tools.registry import registry

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump({
        "platform_toolsets": {"subagent": ["mcp-bridge-boundary"]},
        "tools": {"tool_search": {"enabled": "on", "defer": []}},
    }), encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    registry.register_toolset_alias("mcp-bridge-boundary", "mcp-bridge-boundary")
    allowed_name = "mcp__bridge_boundary__echo"
    denied_name = "mcp__bridge_boundary__secret"
    effects = []
    prior = {name: registry.snapshot_registration(name) for name in (allowed_name, denied_name)}
    for name in (allowed_name, denied_name):
        registry.register(name=name, toolset="mcp-bridge-boundary", schema={
            "name": name, "description": "Echo secret bridge-boundary fixture", "parameters": {
                "type": "object", "properties": {"value": {"type": "string"}},
            },
        }, handler=lambda args, _name=name, **kwargs: effects.append((_name, args.get("value")))
        or json.dumps({"called": _name, "value": args.get("value")}))
    agent = None
    try:
        grant = (allowed_name, "tool_search", "tool_describe", "tool_call")
        agent = AIAgent(
            api_key="fixture-key", base_url="http://127.0.0.1:9/v1", provider="custom",
            api_mode="chat_completions", model="fixture-model", enabled_toolsets=["mcp-bridge-boundary"],
            quiet_mode=True, skip_context_files=True, skip_memory=True, save_trajectories=False,
            platform="subagent", delegation_launch_policy=DelegationLaunchPolicy(
                tools=grant, parent_tool_grant=grant, admitted_tool_names=grant,
            ),
        )
        assert set(agent.valid_tool_names) == {allowed_name, "tool_search", "tool_describe", "tool_call"}
        from tools.thread_context import exact_tool_scope
        import model_tools
        with exact_tool_scope(grant):
            search = json.loads(model_tools.handle_function_call(
                "tool_search", {"queries": ["echo secret"]}, task_id="bridge-search",
                enabled_toolsets=["mcp-bridge-boundary"]))
            assert search["total_available"] == 1
            assert list(search["tools"]) == [allowed_name]
            described = json.loads(model_tools.handle_function_call(
                "tool_describe", {"names": [allowed_name, denied_name]}, task_id="bridge-describe",
                enabled_toolsets=["mcp-bridge-boundary"]))
            assert list(described["tools"]) == [allowed_name]
            assert denied_name in described["errors"]
            denied = json.loads(model_tools.handle_function_call(
                "tool_call", {"name": denied_name, "arguments": {"value": "no"}},
                task_id="bridge-denied", enabled_toolsets=["mcp-bridge-boundary"]))
            assert "outside this task's exact tool grant" in denied["error"]
            allowed = json.loads(model_tools.handle_function_call(
                "tool_call", {"name": allowed_name, "arguments": {"value": "yes"}},
                task_id="bridge-allowed", enabled_toolsets=["mcp-bridge-boundary"]))
            assert allowed == {"called": allowed_name, "value": "yes"}
        assert effects == [(allowed_name, "yes")]
    finally:
        if agent is not None:
            agent.close()
        for name, previous in prior.items():
            current = registry.snapshot_registration(name)
            if current is not None:
                registry.restore_registration(name, current, previous)


def test_empty_child_runtime_dispatch_refuses_direct_and_executor_calls(tmp_path, monkeypatch):
    from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall
    from agent.delegation_launch import DelegationLaunchPolicy
    import agent.agent_runtime_helpers as runtime_helpers
    from agent.tool_executor import execute_tool_calls_concurrent, execute_tool_calls_sequential
    from run_agent import AIAgent

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    agent = AIAgent(
        api_key="fixture-key", base_url="http://127.0.0.1:9/v1", provider="custom",
        api_mode="chat_completions", model="fixture-model", enabled_toolsets=["todo"],
        quiet_mode=True, skip_context_files=True, skip_memory=True, save_trajectories=False,
        platform="subagent", delegation_launch_policy=DelegationLaunchPolicy(tools=()),
    )
    todo_store = agent._todo_store
    legacy_agent = AIAgent(
        api_key="fixture-key", base_url="http://127.0.0.1:9/v1", provider="custom",
        api_mode="chat_completions", model="fixture-model", enabled_toolsets=["todo"],
        quiet_mode=True, skip_context_files=True, skip_memory=True, save_trajectories=False,
        platform="subagent", delegation_launch_policy=DelegationLaunchPolicy(tools=None),
    )
    legacy_todo_store = legacy_agent._todo_store
    legacy_agent.valid_tool_names.add("todo_list")
    middleware_calls = []
    from hermes_cli import middleware
    real_apply_tool_request_middleware = middleware.apply_tool_request_middleware

    def count_middleware(*args, **kwargs):
        middleware_calls.append(args[0])
        return real_apply_tool_request_middleware(*args, **kwargs)

    monkeypatch.setattr(middleware, "apply_tool_request_middleware", count_middleware)

    def tool_message(call_id):
        call = ChatCompletionMessageToolCall.model_validate({
            "id": call_id, "type": "function", "function": {
                "name": "todo_list", "arguments": json.dumps({
                    "todos": [{"id": "blocked", "content": "must not change todo state"}],
                }),
            },
        })
        return ChatCompletionMessage(role="assistant", content=None, tool_calls=[call])

    try:
        direct = runtime_helpers.invoke_tool(
            agent, "todo_list", {"todos": [{"id": "blocked", "content": "no effect"}]}, "empty-child",
        )
        assert "empty tool grant" in direct
        assert todo_store.read() == []
        assert todo_store.snapshot()["revision"] == 0

        for executor in (execute_tool_calls_sequential, execute_tool_calls_concurrent):
            messages = []
            executor(agent, tool_message(f"{executor.__name__}-call"), messages, "empty-child", finalize=False)
            assert len(messages) == 1
            assert "empty tool grant" in messages[0]["content"]
            assert todo_store.read() == []
            assert todo_store.snapshot()["revision"] == 0

        assert middleware_calls == []
        assert agent._repair_tool_call("TodoTool_tool") is None
        assert legacy_agent._repair_tool_call("todoo_list") == "todo_list"
        legacy_result = runtime_helpers.invoke_tool(
            legacy_agent, "todo_list", {"todos": [{"id": "legacy", "content": "legacy dispatch remains"}]},
            "legacy-child",
        )
        assert '"id": "legacy"' in legacy_result
        assert legacy_todo_store.read()[0]["id"] == "legacy"
    finally:
        agent.close()
        legacy_agent.close()


def test_empty_child_mixed_protocol_and_refresh_compatibility(tmp_path, monkeypatch):
    from run_agent import AIAgent
    from tools.delegate_tool_validation import _parent_tool_grant

    requests = []

    class InferenceHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            tool_names = {item.get("function", {}).get("name") for item in payload.get("tools", [])}
            tool_messages = [message for message in payload.get("messages", []) if message.get("role") == "tool"]
            denied_followup = bool(tool_messages)
            if denied_followup:
                message = {"role": "assistant", "content": "LOCAL_CHILD_DONE"}
                finish_reason = "stop"
                response: dict[str, object] = {
                    "id": f"fixture-{len(requests)}",
                    "object": "chat.completion",
                    "created": 1,
                    "model": payload.get("model", "fixture-model"),
                    "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
                }
                self.send_response(200)
                self.send_header(
                    "Content-Type", "text/event-stream" if payload.get("stream") else "application/json"
                )
                self.end_headers()
                if payload.get("stream"):
                    chunks = [
                        {"id": response["id"], "object": "chat.completion.chunk", "created": 1,
                         "model": response["model"], "choices": [{"index": 0,
                             "delta": {"role": "assistant", "content": "LOCAL_CHILD_DONE"}, "finish_reason": None}]},
                        {"id": response["id"], "object": "chat.completion.chunk", "created": 1,
                         "model": response["model"], "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                         "usage": response["usage"]},
                    ]
                    for chunk in chunks:
                        self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                else:
                    self.send_header("Content-Length", str(len(json.dumps(response).encode())))
                    self.wfile.write(json.dumps(response).encode())
                return
            empty_tool_call = not tool_messages and any(
                "complete with no tools" in str(message.get("content", ""))
                for message in payload.get("messages", [])
            )
            mixed_protocol = not tool_messages and any(
                "mixed protocol" in str(message.get("content", ""))
                for message in payload.get("messages", [])
            )
            exact_tool_call = not tool_messages and tool_names == {"read_file"}
            tool_call_response = empty_tool_call or exact_tool_call or mixed_protocol
            calls = ([
                ("todo_list", {"todos": [{"id": "protocol-effect", "content": "must not be written"}]}),
                ("read_file", {"path": str(tmp_path / "exact-grant-readable.txt")}),
                ("read_file", {"path": str(tmp_path / "second-readable.txt")}),
                ("write_file", {"path": str(tmp_path / "mixed-written.txt"), "content": "written once"}),
            ] if mixed_protocol else [("todo_list", {"todos": [{"id": "protocol-effect", "content": "must not be written"}]})]
                if empty_tool_call else [("read_file", {"path": str(tmp_path / "exact-grant-readable.txt")})])
            message = (
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": f"protocol-call-{i}", "type": "function",
                     "function": {"name": name, "arguments": json.dumps(args)}}
                    for i, (name, args) in enumerate(calls)
                ]}
                if tool_call_response else {"role": "assistant", "content": "LOCAL_CHILD_DONE"}
            )
            finish_reason = "tool_calls" if tool_call_response else "stop"
            response: dict[str, object] = {
                "id": f"fixture-{len(requests)}",
                "object": "chat.completion",
                "created": 1,
                "model": payload.get("model", "fixture-model"),
                "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            }
            self.send_response(200)
            self.send_header(
                "Content-Type", "text/event-stream" if payload.get("stream") else "application/json"
            )
            self.end_headers()
            if payload.get("stream"):
                delta = {"role": "assistant"}
                if tool_call_response:
                    delta["tool_calls"] = [
                        {"index": i, "id": f"protocol-call-{i}", "type": "function",
                         "function": {"name": name, "arguments": json.dumps(args)}}
                        for i, (name, args) in enumerate(calls)
                    ]
                else:
                    delta["content"] = "LOCAL_CHILD_DONE"
                chunks = [
                    {"id": response["id"], "object": "chat.completion.chunk", "created": 1,
                     "model": response["model"], "choices": [{"index": 0,
                         "delta": delta, "finish_reason": None}]},
                    {"id": response["id"], "object": "chat.completion.chunk", "created": 1,
                     "model": response["model"], "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                     "usage": response["usage"]},
                ]
                for chunk in chunks:
                    self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                self.send_header("Content-Length", str(len(json.dumps(response).encode())))
                self.wfile.write(json.dumps(response).encode())

        def log_message(self, format, *args):
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), InferenceHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    parent = None
    try:
        home = tmp_path / ".hermes"
        home.mkdir()
        (tmp_path / "exact-grant-readable.txt").write_text("delegate startup read proof")
        (tmp_path / "second-readable.txt").write_text("second mixed read result")
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_HOME", str(home))
        endpoint = f"http://127.0.0.1:{server.server_port}/v1"
        parent = AIAgent(
            api_key="fixture-key", base_url=endpoint, provider="custom", api_mode="chat_completions",
            model="fixture-model", enabled_toolsets=["delegation", "file"], quiet_mode=True,
            skip_context_files=True, skip_memory=True, save_trajectories=False,
            ephemeral_system_prompt="PARENT_PROMPT_MARKER", max_iterations=2,
        )
        from hermes_state import SessionDB
        parent._session_db = SessionDB(db_path=tmp_path / "state.db")
        parent._ensure_db_session()
        owner_grant = _parent_tool_grant(parent)
        assert owner_grant is not None and {"read_file", "write_file"}.issubset(owner_grant)
        parent_tools_before = json.dumps(parent.tools, sort_keys=True, separators=(",", ":")).encode()
        assert parent.tools and "delegate_task" in parent.valid_tool_names

        first_parent = parent.run_conversation("parent before child startup")
        assert first_parent["final_response"] == "LOCAL_CHILD_DONE"
        parent_prompt_before = requests[-1]["messages"][0]["content"]

        from agent import agent_init
        real_inject_context_tools = agent_init._inject_context_engine_tools
        real_snapshot_primary_runtime = agent_init._snapshot_primary_runtime
        child_startups = []
        child_agents = []
        protocol_tool_results = []

        def inject_context_tools(agent):
            real_inject_context_tools(agent)
            policy = getattr(agent, "_delegation_launch_policy", None)
            if getattr(agent, "platform", None) != "subagent":
                return
            if getattr(policy, "tools", None) == ():
                # Simulate a late startup injection; production init must remove it.
                agent.valid_tool_names.add("todo_list")
                agent.tools.append({"type": "function", "function": {"name": "todo_list"}})
        def snapshot_primary_runtime(agent):
            real_snapshot_primary_runtime(agent)
            policy = getattr(agent, "_delegation_launch_policy", None)
            if getattr(agent, "platform", None) == "subagent":
                child_startups.append((getattr(policy, "tools", None), list(agent.tools),
                                       set(getattr(agent, "_exact_tool_name_ceiling", ())),
                                       {item["function"]["name"] for item in agent.tools
                                        if "function" in item and "name" in item["function"]}))
                child_agents.append(agent)

        monkeypatch.setattr(agent_init, "_inject_context_engine_tools", inject_context_tools)
        monkeypatch.setattr(agent_init, "_snapshot_primary_runtime", snapshot_primary_runtime)
        from agent import tool_executor
        from agent import delegation_launch
        original_enforce = delegation_launch.enforce_tool_ceiling

        def enforce_after_real_startup(agent):
            original_enforce(agent)
            if getattr(agent, "platform", None) == "subagent":
                policy = getattr(agent, "_delegation_launch_policy", None)
                child_startups.append((getattr(policy, "tools", None), list(agent.tools), set(agent.valid_tool_names),
                                       {item["function"]["name"] for item in agent.tools}))

        monkeypatch.setattr(delegation_launch, "enforce_tool_ceiling", enforce_after_real_startup)
        real_run = AIAgent.run_conversation
        def mixed_run(agent, *args, **kwargs):
            prompt = args[0] if args else kwargs.get("user_message", "")
            if getattr(agent, "platform", None) == "subagent" and prompt == "complete with exact read + mixed protocol":
                args = ("mixed protocol with exact read", *args[1:]) if args else args
                kwargs["user_message"] = "mixed protocol with exact read"
                flush = agent._flush_messages_to_session_db
                def persist_calls(messages, history=None):
                    staged = any(
                        row.get("role") == "assistant" and len(row.get("tool_calls", [])) == 4
                        for row in messages
                    )
                    result = flush(messages, history)
                    if staged:
                        staged_before_effect.append(result is not False)
                    return result
                agent._flush_messages_to_session_db = persist_calls
            return real_run(agent, *args, **kwargs)
        monkeypatch.setattr(AIAgent, "run_conversation", mixed_run)
        original_commit = tool_executor._commit_tool_result
        staged_before_effect = []

        def capture_protocol_result(agent, messages, ref, function_result, **kwargs):
            if (getattr(agent, "_delegation_launch_policy", None)
                    and any(row.get("role") == "assistant" and len(row.get("tool_calls", [])) == 4
                            for row in messages)):
                assert staged_before_effect and all(staged_before_effect)
            committed = original_commit(agent, messages, ref, function_result, **kwargs)
            if ref.name == "read_file" and messages:
                tool_result = messages[-1].get("content", "")
                protocol_tool_results.append((getattr(agent, "session_id", None), tool_result))
            return committed

        monkeypatch.setattr(tool_executor, "_commit_tool_result", capture_protocol_result)
        from tools.delegate_tool import delegate_task

        output = json.loads(delegate_task(tasks=[
            {"goal": "complete with no tools", "tools": []},
            {"goal": "complete with exact read", "tools": ["read_file"]},
            {"goal": "complete with exact read + mixed protocol", "tools": ["read_file", "write_file"]},
            {"goal": "complete with inherited tools"},
        ], parent_agent=parent))

        assert [result["status"] for result in output["results"]] == ["completed"] * 4, output
        assert [result["summary"] for result in output["results"]] == ["LOCAL_CHILD_DONE"] * 4
        empty_child = next(agent for agent in child_agents
                           if getattr(getattr(agent, "_delegation_launch_policy", None), "tools", None) == ())
        empty_startup = next(item for item in child_startups if item[0] == ())
        assert not ({"todo_list", "read_file", "write_file", "patch", "search_files"} & empty_startup[3])
        assert empty_child.tools == []
        assert empty_child.valid_tool_names == set()
        assert empty_child._todo_store.read() == []
        assert empty_child._todo_store.snapshot()["revision"] == 0

        read_child = next(agent for agent in child_agents
                          if getattr(getattr(agent, "_delegation_launch_policy", None), "tools", None) == ("read_file",))
        assert read_child.valid_tool_names == {"read_file"}
        assert read_child._delegation_launch_policy.parent_tool_grant == owner_grant
        exact_wire = next(request for request in requests if {
            item.get("function", {}).get("name") for item in request.get("tools", [])
        } == {"read_file"})
        assert {item["function"]["name"] for item in exact_wire["tools"]} == {"read_file"}
        mixed_session = next(agent.session_id for agent in child_agents
                            if agent.session_id != read_child.session_id
                            and getattr(getattr(agent, "_delegation_launch_policy", None), "tools", None)
                            == ("read_file", "write_file"))
        mixed_results = [content for session_id, content in protocol_tool_results if session_id == mixed_session]
        assert any("delegate startup read proof" in content for content in mixed_results)
        assert any("second mixed read result" in content for content in mixed_results)
        assert (tmp_path / "exact-grant-readable.txt").read_text() == "delegate startup read proof"
        assert not (tmp_path / "protocol-effect").exists()
        import agent.agent_runtime_helpers as runtime_helpers
        startup_read = runtime_helpers.invoke_tool(
            read_child, "read_file", {"path": str(tmp_path / "exact-grant-readable.txt")}, "exact-child",
        )
        assert "delegate startup read proof" in startup_read
        startup_write = runtime_helpers.invoke_tool(
            read_child, "write_file", {"path": str(tmp_path / "protocol-effect"), "content": "forbidden"}, "exact-child",
        )
        assert "outside this child's exact tool grant" in startup_write
        assert not (tmp_path / "protocol-effect").exists()
        empty_wire = next(request for request in requests if not request.get("tools"))
        assert empty_wire.get("tools") in (None, [])
        empty_followup = next(
            request for request in requests
            if not request.get("tools") and any(message.get("role") == "tool" for message in request.get("messages", []))
        )
        refused_result = next(message for message in empty_followup["messages"] if message.get("role") == "tool")
        assert refused_result["content"] == json.dumps({
            "error": "Tool call refused: this child was launched with an empty tool grant; no Hermes tools are permitted."
        })
        assert "LOCAL_CHILD_DONE" not in refused_result["content"]

        next(
            [call for call in message.get("tool_calls", [])]
            for request in requests for message in request.get("messages", [])
            if message.get("role") == "assistant" and len(message.get("tool_calls", [])) == 4
        )
        mixed_turns = [
            request["messages"] for request in requests
            if any(message.get("role") == "assistant" and len(message.get("tool_calls", [])) == 4
                   for message in request.get("messages", []))
        ]
        staged = mixed_turns[0]
        assistant_row = next(message for message in staged
                             if message.get("role") == "assistant" and message.get("tool_calls"))
        paired = [message for message in staged if message.get("role") == "tool"]
        assert len(assistant_row["tool_calls"]) == 4
        assert len(paired) == 4 and {m["tool_call_id"] for m in paired} == {c["id"] for c in assistant_row["tool_calls"]}
        assert [m["tool_call_id"] for m in paired].count("protocol-call-0") == 1
        assert [m["tool_call_id"] for m in paired].count("protocol-call-1") == 1
        assert any("outside this child's exact tool grant" in m["content"] for m in paired)
        assert any("delegate startup read proof" in m["content"] for m in paired)
        assert any("second mixed read result" in m["content"] for m in paired)
        assert (tmp_path / "mixed-written.txt").read_text() == "written once"
        assert sum(m["tool_call_id"] == "protocol-call-3" for m in paired) == 1
        assert not (tmp_path / "protocol-effect").exists()
        assert staged_before_effect and all(staged_before_effect)

        from tools.mcp_tool_agent import refresh_agent_mcp_tools
        original_tools = list(read_child.tools)
        original_names = set(read_child.valid_tool_names)
        import model_tools
        monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **kwargs: [
            {"type": "function", "function": {"name": "read_file"}},
            {"type": "function", "function": {"name": "write_file"}},
        ])
        refresh_agent_mcp_tools(read_child, quiet_mode=True)
        assert {tool["function"]["name"] for tool in read_child.tools} == {"read_file"}
        assert read_child.valid_tool_names == original_names == {"read_file"}
        assert read_child.tools != original_tools  # Schema refresh accepted; exact names remain pinned.

        denied_agent = next(agent for agent in child_agents
                            if getattr(getattr(agent, "_delegation_launch_policy", None), "tools", None) == ())
        monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **kwargs: [
            {"type": "function", "function": {"name": "todo_list"}},
        ])
        refresh_agent_mcp_tools(denied_agent, quiet_mode=True)
        assert denied_agent.tools == [] and denied_agent.valid_tool_names == set()

        from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall
        from agent.turn_tool_validation import validate_tool_calls
        guard_agent = read_child
        call = lambda name, args: ChatCompletionMessageToolCall.model_validate({
            "id": "guard-" + name, "type": "function",
            "function": {"name": name, "arguments": args},
        })
        def validate_pair(valid_args, denied_args):
            message = ChatCompletionMessage(role="assistant", content=None, tool_calls=[
                call("read_file", valid_args), call("write_file", denied_args),
            ])
            return validate_tool_calls(guard_agent, message, "tool_calls", messages=[],
                                       conversation_history=[], api_call_count=1, effective_task_id="mixed")
        malformed = validate_pair('{"path":', '{"path": "denied"}')
        assert malformed.action == "return" and malformed.result is not None
        assert malformed.result["partial"] is True
        assert not (tmp_path / "denied").exists()
        truncated = validate_pair('{"path": "unterminated', '{"path": "denied"}')
        assert truncated.action == "return" and truncated.result is not None
        assert truncated.result["partial"] is True
        assert not (tmp_path / "denied").exists()

        unknown = ChatCompletionMessage(role="assistant", content=None, tool_calls=[
            call("made_up_tool", "{}"), call("read_file", '{"path": "unknown-control.txt"}'),
        ])
        guard_agent._invalid_tool_retries = 2
        unknown_verdict = validate_tool_calls(guard_agent, unknown, "tool_calls", messages=[],
                                              conversation_history=[], api_call_count=1,
                                              effective_task_id="unknown-control")
        assert unknown_verdict.action == "ok" and unknown_verdict.runnable_tool_calls == [unknown.tool_calls[1]]
        assert unknown_verdict.rejected_tool_results is not None
        assert len(unknown_verdict.rejected_tool_results) == 1
        assert "does not exist" in unknown_verdict.rejected_tool_results[0][1]

        empty_child = next(agent for agent in child_agents
                           if getattr(getattr(agent, "_delegation_launch_policy", None), "tools", None) == ())
        pure_denied = ChatCompletionMessage(role="assistant", content=None, tool_calls=[call("todo_list", "{}")])
        pure_verdict = validate_tool_calls(empty_child, pure_denied, "tool_calls", messages=[],
                                           conversation_history=[], api_call_count=1,
                                           effective_task_id="pure-deny-control")
        assert pure_verdict.action == "continue"
        both_denied = ChatCompletionMessage(role="assistant", content=None, tool_calls=[
            call("todo_list", "{}"), call("write_file", "{}"),
        ])
        both_verdict = validate_tool_calls(guard_agent, both_denied, "tool_calls", messages=[],
                                           conversation_history=[], api_call_count=1,
                                           effective_task_id="both-denied-control")
        assert both_verdict.action == "continue"


        # The configured child profile's real cli surface excludes file tools.
        # Public batch dispatch refuses the read_file task before inference while
        # its explicit empty-grant sibling is still admitted and completes.
        import hermes_yaml as yaml
        restricted = home / "profiles" / "restricted"
        restricted.mkdir(parents=True)
        (restricted / "config.yaml").write_text(yaml.safe_dump({
            "model": {"provider": "custom", "default": "fixture-model", "base_url": endpoint},
            "platform_toolsets": {"cli": ["web"]},
        }), encoding="utf-8")
        (restricted / "SOUL.md").write_text("LIMITED_PROFILE_IDENTITY", encoding="utf-8")
        requests_before_profile_batch = len(requests)
        profile_batch = json.loads(delegate_task(tasks=[
            {"goal": "must be refused before inference", "profile": "restricted", "tools": ["read_file"]},
            {"goal": "permitted empty-grant sibling", "profile": "restricted", "tools": []},
        ], parent_agent=parent))
        assert [result["status"] for result in profile_batch["results"]] == ["error", "completed"]
        assert profile_batch["results"][0]["api_calls"] == 0
        assert profile_batch["results"][1]["api_calls"] == 1
        profile_inference = [request for request in requests[requests_before_profile_batch:]
                             if request.get("messages") and request["messages"][0].get("role") == "system"]
        assert len(profile_inference) == 1
        profile_first_turns = [request for request in profile_inference
                               if not any(message.get("role") == "tool" for message in request["messages"])]
        assert len(profile_first_turns) == 1
        assert any("LIMITED_PROFILE_IDENTITY" in str(message.get("content", ""))
                   for message in profile_first_turns[0]["messages"])

        last_parent = parent.run_conversation("parent after child startup")
        assert last_parent["final_response"] == "LOCAL_CHILD_DONE"
        parent_requests = [request for request in requests
                           if request.get("messages")
                           and request["messages"][0].get("content") == parent_prompt_before]
        assert parent_requests[-1]["messages"][0]["content"] == parent_prompt_before
        assert json.dumps(parent.tools, sort_keys=True, separators=(",", ":")).encode() == parent_tools_before
    finally:
        if parent is not None:
            parent.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)



def test_exact_builtin_grant_survives_real_pin_reload_and_compression(tmp_path, monkeypatch):
    import json

    from agent.delegation_launch import DelegationLaunchPolicy
    from agent.agent_runtime_helpers import invoke_tool
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from tools.mcp_tool_agent import refresh_agent_mcp_tools, restore_agent_tool_prefix, tool_pin_version
    from tools.registry import registry

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    effects = []
    fixture_name = "write_file"
    fixture_schema = {
        "name": fixture_name,
        "description": "A late-registered builtin fixture handler outside the grant.",
        "parameters": {"type": "object", "properties": {}},
    }
    policy = DelegationLaunchPolicy(
        tools=("read_file", "terminal"),
        parent_tool_grant=("read_file", "terminal", "write_file"),
    )
    legacy_policy = DelegationLaunchPolicy(tools=None)
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("lifecycle-session", source="test")
    agents = []

    def build_agent():
        child = AIAgent(
            api_key="fixture-key", base_url="http://127.0.0.1:9/v1", provider="custom",
            api_mode="chat_completions", model="fixture-model", enabled_toolsets=["file", "terminal"],
            quiet_mode=True, skip_context_files=True, skip_memory=True, save_trajectories=False,
            platform="subagent", session_id="lifecycle-session", delegation_launch_policy=policy,
        )
        child._session_db = db
        agents.append(child)
        return child

    def build_legacy_agent():
        child = AIAgent(
            api_key="fixture-key", base_url="http://127.0.0.1:9/v1", provider="custom",
            api_mode="chat_completions", model="fixture-model", enabled_toolsets=["file"],
            quiet_mode=True, skip_context_files=True, skip_memory=True, save_trajectories=False,
            platform="subagent", delegation_launch_policy=legacy_policy,
        )
        agents.append(child)
        return child

    previous_write = registry.snapshot_registration(fixture_name)
    previous_read = registry.snapshot_registration("read_file")
    assert previous_write is not None and previous_read is not None
    try:
        original = build_agent()
        assert {item["function"]["name"] for item in original.tools} == {"read_file", "terminal"}
        original_read_file_schema = json.dumps(next(
            item for item in original.tools if item["function"]["name"] == "read_file"
        ), sort_keys=True, separators=(",", ":"))
        original_terminal_schema = json.dumps(next(
            item for item in original.tools if item["function"]["name"] == "terminal"
        ), sort_keys=True, separators=(",", ":"))

        # Late registration replaces the builtin handler after the first startup; later rebuilds
        # must not add this newly available name outside the immutable grant.
        registry.register(
            name=fixture_name, toolset="file", schema=fixture_schema,
            handler=lambda args, **kwargs: effects.append("invoked") or "FIXTURE_INVOKED", override=True,
        )

        # Store a real SessionDB pin containing an out-of-grant registered schema, as from an older
        # or wider session surface. The same-code and updated-code restore branches both consume it.
        for pin_version in (tool_pin_version(), "updated-code"):
            pin = {"version": pin_version, "tools": [
                next(item for item in original.tools if item["function"]["name"] == "read_file"),
                {"type": "function", "function": fixture_schema},
                next(item for item in original.tools if item["function"]["name"] == "terminal"),
            ]}
            db.update_session_tool_names("lifecycle-session", pin)
            stored_pin = json.loads(db.get_session("lifecycle-session")["tool_names"])
            rebuilt = build_agent()
            rebuilt._context_engine_tool_names = {fixture_name}
            restore_agent_tool_prefix(rebuilt, stored_pin)
            assert [item["function"]["name"] for item in rebuilt.tools] == ["read_file", "terminal"]
            assert rebuilt.valid_tool_names == {"read_file", "terminal"}
            assert rebuilt._context_engine_tool_names == set()
            assert json.dumps(rebuilt.tools[0], sort_keys=True, separators=(",", ":")) == original_read_file_schema
            assert json.dumps(rebuilt.tools[1], sort_keys=True, separators=(",", ":")) == original_terminal_schema
            assert "outside this child's exact tool grant" in invoke_tool(
                rebuilt, fixture_name, {}, "lifecycle-session",
            )
            assert effects == []

        refreshed = agents[-1]
        assert refresh_agent_mcp_tools(refreshed, quiet_mode=True, preserve_prefix=True) == set()
        assert {item["function"]["name"] for item in refreshed.tools} == {"read_file", "terminal"}
        assert json.dumps(refreshed.tools[0], sort_keys=True, separators=(",", ":")) == original_read_file_schema
        assert json.dumps(refreshed.tools[1], sort_keys=True, separators=(",", ":")) == original_terminal_schema
        assert "outside this child's exact tool grant" in invoke_tool(
            refreshed, fixture_name, {}, "lifecycle-session",
        )
        assert effects == []

        # Ordinary reload accepts the changed allowed schema while keeping the exact granted name set.
        updated_read_schema = {**previous_read.schema, "description": "Read schema updated by explicit reload."}
        registry.register(
            name="read_file", toolset="file", schema=updated_read_schema,
            handler=previous_read.handler, override=True,
        )
        assert refresh_agent_mcp_tools(refreshed, quiet_mode=True) == set()
        assert {item["function"]["name"] for item in refreshed.tools} == {"read_file", "terminal"}
        assert refreshed.tools[0]["function"]["description"] == "Read schema updated by explicit reload."
        assert refreshed.valid_tool_names == {"read_file", "terminal"}
        assert effects == []

        # A post-build context-engine schema is freshly injected by the actual refresh path,
        # then the same exact filter keeps its routing name and schema outside the grant.
        context_name = "lcm_fixture_lifecycle_tool"
        context_schema = {"name": context_name, "description": "Fixture context action.",
                          "parameters": {"type": "object", "properties": {}}}
        refreshed.enabled_toolsets = ["file", "terminal", "context_engine"]
        refreshed.disabled_toolsets = None
        refreshed.context_compressor = type(
            "ContextFixture", (), {"get_tool_schemas": lambda self: [context_schema]}
        )()
        refresh_agent_mcp_tools(refreshed, quiet_mode=True, content_aware=True)
        assert {item["function"]["name"] for item in refreshed.tools} == {"read_file", "terminal"}
        assert refreshed._context_engine_tool_names == set()

        legacy = build_legacy_agent()
        assert "write_file" in legacy.valid_tool_names
        assert invoke_tool(legacy, fixture_name, {}, "legacy-session") == "FIXTURE_INVOKED"
        assert effects == ["invoked"]

        from agent.conversation_compression import _refresh_agent_tool_definitions
        assert _refresh_agent_tool_definitions(refreshed) is False
        assert {item["function"]["name"] for item in refreshed.tools} == {"read_file", "terminal"}
        assert refreshed.valid_tool_names == {"read_file", "terminal"}
        assert refreshed._context_engine_tool_names == set()
        assert effects == ["invoked"]
    finally:
        current_write = registry.snapshot_registration(fixture_name)
        current_read = registry.snapshot_registration("read_file")
        if current_write is not None:
            registry.restore_registration(fixture_name, current_write, previous_write)
        if current_read is not None:
            registry.restore_registration("read_file", current_read, previous_read)
        for child in agents:
            child.close()
        db.close()



def test_nonempty_child_tool_grant_is_exact_at_startup_and_runtime(tmp_path, monkeypatch):
    from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall

    from agent.delegation_launch import DelegationLaunchPolicy
    import agent.agent_runtime_helpers as runtime_helpers
    from agent.tool_executor import execute_tool_calls_concurrent, execute_tool_calls_sequential
    from run_agent import AIAgent

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    source = tmp_path / "readable.txt"
    source.write_text("exact grant can read this")
    target = tmp_path / "must-not-write.txt"
    agent = AIAgent(
        api_key="fixture-key", base_url="http://127.0.0.1:9/v1", provider="custom",
        api_mode="chat_completions", model="fixture-model", enabled_toolsets=["file"], disabled_toolsets=["terminal"],
        quiet_mode=True, skip_context_files=True, skip_memory=True, save_trajectories=False,
        platform="subagent",
        delegation_launch_policy=DelegationLaunchPolicy(
            tools=("read_file",), parent_tool_grant=("read_file", "write_file", "terminal", "todo_list"),
        ),
    )
    try:
        tool_names = {tool["function"]["name"] for tool in agent.tools}
        assert tool_names == {"read_file"}
        assert agent.valid_tool_names == {"read_file"}
        assert agent._delegation_launch_policy.tools == ("read_file",)
        assert agent._repair_tool_call("ReadFile") is None

        allowed = runtime_helpers.invoke_tool(agent, "read_file", {"path": str(source)}, "strict-child")
        assert "exact grant can read this" in allowed

        write_args = {"path": str(target), "content": "forbidden"}
        refused = runtime_helpers.invoke_tool(agent, "write_file", write_args, "strict-child")
        assert "outside this child's exact tool grant" in refused
        assert not target.exists()

        def call(name, call_id, arguments):
            return ChatCompletionMessage(role="assistant", content=None, tool_calls=[
                ChatCompletionMessageToolCall.model_validate({
                    "id": call_id, "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                })
            ])

        for executor in (execute_tool_calls_sequential, execute_tool_calls_concurrent):
            messages = []
            executor(
                agent, call("terminal", f"{executor.__name__}-terminal", {"command": "touch forbidden"}),
                messages, "strict-child", finalize=False,
            )
            assert len(messages) == 1
            assert "outside this child's exact tool grant" in messages[0]["content"]
            assert not target.exists()
            assert not (tmp_path / "forbidden").exists()
    finally:
        agent.close()


def test_explicit_child_tool_grant_rejects_unknown_and_parent_escalation():
    from tools.delegate_tool_options import TaskRouteOptions
    from tools.delegate_tool_validation import _exact_builtin_tool_grant_error

    parent = type("Parent", (), {"platform": "cli"})()
    route = type("Route", (), {"profile_home": None})()
    cases = (
        ({"tools": ["write_file"]}, ("read_file",), "exceed the owning parent's tools"),
        ({"tools": ["read_file"]}, ("write_file",), "exceed the owning parent's tools"),
        ({"tools": ["read_file"]}, None, "could not be verified"),
        ({"tools": ["made_up"]}, ("made_up",), "unsupported or non-builtin"),
    )
    for raw, parent_grant, expected in cases:
        option, error = TaskRouteOptions.parse(raw, 2)
        assert error is None and option is not None
        message = _exact_builtin_tool_grant_error(2, option, parent_grant, route, {}, parent)
        assert message is not None and expected in message

    option, error = TaskRouteOptions.parse({"tools": ["read_file"]}, 2)
    assert error is None and option is not None
    selected_profile_route = type("SelectedRoute", (), {"profile_home": "/selected-profile"})()
    disabled_profile = {"platform_toolsets": {"cli": ["web"]}}
    message = _exact_builtin_tool_grant_error(
        2, option, ("read_file",), selected_profile_route, disabled_profile, parent,
    )
    assert message and "disabled by the selected profile's cli toolset selection" in message


def test_nonempty_exact_grant_refuses_when_actual_child_surface_is_unavailable(tmp_path, monkeypatch):
    from agent.delegation_launch import DelegationLaunchPolicy
    from agent import agent_init
    from run_agent import AIAgent

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    seen = []
    original = agent_init._load_tools

    def capture(agent, enabled, disabled):
        original(agent, enabled, disabled)
        seen.append((enabled, disabled, set(agent.valid_tool_names)))

    monkeypatch.setattr(agent_init, "_load_tools", capture)
    try:
        AIAgent(
            api_key="fixture-key", base_url="http://127.0.0.1:9/v1", provider="custom",
            api_mode="chat_completions", model="fixture-model", enabled_toolsets=["file"],
            disabled_toolsets=["file"], quiet_mode=True, skip_context_files=True, skip_memory=True,
            save_trajectories=False, platform="subagent",
            delegation_launch_policy=DelegationLaunchPolicy(
                tools=("read_file",), parent_tool_grant=("read_file",),
            ),
        )
    except ValueError as exc:
        assert "unavailable" in str(exc) or "not admitted" in str(exc)
    else:
        raise AssertionError(f"Unavailable exact grant initialized successfully: {seen!r}")


def test_composite_profile_tool_selection_is_checked_by_effective_tool_membership():
    from tools.delegate_tool_options import TaskRouteOptions
    from tools.delegate_tool_validation import _exact_builtin_tool_grant_error

    option, error = TaskRouteOptions.parse({"tools": ["read_file"]}, 0)
    assert error is None and option is not None
    from pathlib import Path
    route = type("Route", (), {"profile_home": Path("/selected-profile")})()
    parent = type("Parent", (), {"platform": "cli"})()
    config = {"platform_toolsets": {"cli": ["hermes-cli", "web"]},
              "agent": {"disabled_toolsets": ["terminal"]}}
    message = _exact_builtin_tool_grant_error(0, option, ("read_file",), route, config, parent)
    assert message is None


def test_composite_profile_without_builtin_surface_refuses_exact_grant():
    from pathlib import Path
    from tools.delegate_tool_options import TaskRouteOptions
    from tools.delegate_tool_validation import _exact_builtin_tool_grant_error

    option, error = TaskRouteOptions.parse({"tools": ["read_file"]}, 0)
    assert error is None and option is not None
    route = type("Route", (), {"profile_home": Path("/selected-profile")})()
    parent = type("Parent", (), {"platform": "cli"})()
    config = {"platform_toolsets": {"cli": ["hermes-cli", "web"]},
              "agent": {"disabled_toolsets": ["file"]}}
    message = _exact_builtin_tool_grant_error(0, option, ("read_file",), route, config, parent)
    assert message and "disabled by the selected profile" in message


def test_unchanged_exact_tool_refresh_reconciles_context_routing_without_legacy_mutation(tmp_path, monkeypatch):
    import json

    from agent.agent_runtime_helpers import invoke_tool
    from agent.delegation_launch import DelegationLaunchPolicy
    from run_agent import AIAgent
    from tools.mcp_tool_agent import refresh_agent_mcp_tools
    from tools.registry import registry

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    agents = []

    def build_agent(tools):
        agent = AIAgent(
            api_key="fixture-key", base_url="http://127.0.0.1:9/v1", provider="custom",
            api_mode="chat_completions", model="fixture-model", enabled_toolsets=["file"],
            quiet_mode=True, skip_context_files=True, skip_memory=True, save_trajectories=False,
            platform="subagent", delegation_launch_policy=DelegationLaunchPolicy(
                tools=tools, parent_tool_grant=("read_file", "write_file"),
            ),
        )
        agents.append(agent)
        return agent

    try:
        read_only = build_agent(("read_file",))
        read_schema = json.dumps(read_only.tools, sort_keys=True, separators=(",", ":"))
        read_names = set(read_only.valid_tool_names)
        assert read_names == {"read_file"}
        read_only._context_engine_tool_names = {"write_file"}
        assert refresh_agent_mcp_tools(read_only, quiet_mode=True, preserve_prefix=True) == set()
        assert json.dumps(read_only.tools, sort_keys=True, separators=(",", ":")) == read_schema
        assert read_only.valid_tool_names == read_names
        assert read_only._context_engine_tool_names <= read_names
        forbidden_path = tmp_path / "routing-forbidden.txt"
        assert "outside this child's exact tool grant" in invoke_tool(
            read_only, "write_file", {"path": str(forbidden_path), "content": "forbidden"}, "routing-session",
        )
        assert not forbidden_path.exists()

        empty = build_agent(())
        empty_schema = json.dumps(empty.tools, sort_keys=True, separators=(",", ":"))
        empty._context_engine_tool_names = {"read_file"}
        assert refresh_agent_mcp_tools(empty, quiet_mode=True, preserve_prefix=True) == set()
        assert empty.tools == [] and empty.valid_tool_names == set()
        assert json.dumps(empty.tools, sort_keys=True, separators=(",", ":")) == empty_schema
        assert empty._context_engine_tool_names == set()

        # An older refresh cannot reconcile routing against a snapshot that already won.
        read_only._context_engine_tool_names = {"write_file"}
        read_only._tool_snapshot_generation = registry._generation + 1
        assert refresh_agent_mcp_tools(read_only, quiet_mode=True, preserve_prefix=True) == set()
        assert read_only._context_engine_tool_names == {"write_file"}
        assert json.dumps(read_only.tools, sort_keys=True, separators=(",", ":")) == read_schema

        legacy = build_agent(None)
        legacy._context_engine_tool_names = {"write_file"}
        legacy_schema = json.dumps(legacy.tools, sort_keys=True, separators=(",", ":"))
        assert refresh_agent_mcp_tools(legacy, quiet_mode=True, preserve_prefix=True) == set()
        assert json.dumps(legacy.tools, sort_keys=True, separators=(",", ":")) == legacy_schema
        assert "write_file" in legacy._context_engine_tool_names
    finally:
        for agent in agents:
            agent.close()


def test_nested_grant_boundaries_use_real_public_dispatch(tmp_path, monkeypatch):
    """Strict grants survive nesting, narrowing, empty grants, and the real depth ceiling."""
    import json

    import hermes_yaml as yaml
    from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall
    from agent.delegation_launch import DelegationLaunchPolicy, effective_tool_ceiling
    from run_agent import AIAgent
    from tools.delegate_tool import delegate_task

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump({
        "delegation": {"orchestrator_enabled": True, "max_spawn_depth": 2, "max_iterations": 5},
        "platform_toolsets": {"subagent": ["file", "delegation"]},
    }), encoding="utf-8")
    readable = tmp_path / "nested-readable.txt"
    readable.write_text("nested grandchild read succeeded", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))

    def respond(record):
        body = record["body"]
        messages = body.get("messages", [])
        user_text = " ".join(str(message.get("content", "")) for message in messages
                             if message.get("role") == "user")
        has_tool_result = any(message.get("role") == "tool" for message in messages)
        if "parent prompt before nested grants" in user_text:
            return Text("PARENT_BEFORE_DONE")
        if "explicit read-only child" in user_text:
            return Text("NARROW_CHILD_DONE")
        if "narrower-profile omitted grant" in user_text:
            return Text("PROFILE_CHILD_DONE")
        if "empty child boundary" in user_text:
            return Text("EMPTY_CHILD_DONE")
        if "strict parent cannot delegate" in user_text:
            if has_tool_result:
                return Text("STRICT_PARENT_DONE")
            return ToolCall("delegate_task", {"tasks": [{"goal": "strict parent must not spawn child"}]})
        if "parent prompt after nested grants" in user_text:
            return Text("PARENT_AFTER_DONE")
        if "nested child should attempt escalation" in user_text:
            if has_tool_result:
                return Text("NESTED_CHILD_DONE")
            return ToolCall("delegate_task", {"tasks": [
                {"goal": "explicit escalation must refuse", "tools": ["write_file"]},
                {"goal": "grandchild should read fixture"},
            ]})
        if "grandchild should read fixture" in user_text:
            if has_tool_result:
                return Text("GRANDCHILD_DONE")
            return ToolCall("read_file", {"path": str(readable)})
        if "depth-boundary child should delegate" in user_text:
            if has_tool_result:
                return Text("BOUNDARY_CHILD_DONE")
            return ToolCall("delegate_task", {"tasks": [
                {"goal": "maximum-depth grandchild attempts forbidden delegation"},
            ]})
        if "maximum-depth grandchild attempts forbidden delegation" in user_text:
            if has_tool_result:
                return Text("MAX_DEPTH_DONE")
            return ToolCall("delegate_task", {"tasks": [
                {"goal": "must not spawn beyond the depth ceiling"},
            ]})
        if has_tool_result:
            return Text("CHILD_DONE")
        return Text("PARENT_AFTER_DONE")

    with FakeLLMServer(respond, aux=lambda _request: Text("EMPTY_CHILD_DONE")) as server:
        strict_parent = AIAgent(
            api_key="fixture-key", base_url=server.base_url, provider="custom", api_mode="chat_completions",
            model="fake-model", enabled_toolsets=["file", "delegation"], quiet_mode=True,
            skip_context_files=True, skip_memory=True, save_trajectories=False, platform="subagent",
            ephemeral_system_prompt="STRICT_PARENT_PROMPT_MARKER", max_iterations=3,
            delegation_launch_policy=DelegationLaunchPolicy(
                tools=("read_file",), parent_tool_grant=("read_file",), admitted_tool_names=("read_file",),
            ),
        )
        parent = AIAgent(
            api_key="fixture-key", base_url=server.base_url, provider="custom", api_mode="chat_completions",
            model="fake-model", enabled_toolsets=["file", "delegation"], quiet_mode=True,
            skip_context_files=True, skip_memory=True, save_trajectories=False, platform="subagent",
            ephemeral_system_prompt="NESTED_PARENT_PROMPT_MARKER", max_iterations=2,
            delegation_launch_policy=DelegationLaunchPolicy(
                tools=("read_file", "delegate_task"),
                parent_tool_grant=("read_file", "write_file", "delegate_task"),
                admitted_tool_names=("read_file", "delegate_task"),
            ),
        )
        try:
            strict_parent._delegate_depth = 0
            strict_parent_result = strict_parent.run_conversation("strict parent cannot delegate")
            assert strict_parent_result["final_response"] == "STRICT_PARENT_DONE"
            assert {item["function"]["name"] for item in strict_parent.tools} == {"read_file"}
            assert not (tmp_path / "strict-parent-must-not-write").exists()
            strict_parent_requests = [request for request in server.main_requests()
                                      if any(message.get("role") == "user"
                                             and "strict parent cannot delegate" in str(message.get("content"))
                                             for message in request.get("messages", []))]
            assert len(strict_parent_requests) == 2
            assert {item["function"]["name"] for item in strict_parent_requests[0].get("tools", [])} == {"read_file"}
            refusal = next(message for message in strict_parent_requests[1]["messages"]
                           if message.get("role") == "tool")
            assert "outside this child's exact tool grant" in refusal["content"]
        finally:
            strict_parent.close()

        restricted = home / "profiles" / "restricted"
        restricted.mkdir(parents=True)
        (restricted / "config.yaml").write_text(yaml.safe_dump({
            "model": {"provider": "custom", "default": "fake-model", "base_url": server.base_url},
            "platform_toolsets": {"subagent": ["file"]},
        }), encoding="utf-8")

        try:
            parent._delegate_depth = 0
            assert effective_tool_ceiling(parent) == frozenset({"read_file", "delegate_task"})
            parent_tools_before = json.dumps(parent.tools, sort_keys=True, separators=(",", ":")).encode()
            before = parent.run_conversation("parent prompt before nested grants")
            assert before["final_response"] == "PARENT_BEFORE_DONE"
            requests = server.main_requests()
            parent_before = next(request for request in requests if any(
                message.get("role") == "user" and "parent prompt before nested grants" in str(message.get("content"))
                for message in request["messages"]
            ))
            parent_prompt_before = next(message["content"] for message in parent_before["messages"]
                                        if message.get("role") == "system")
            import model_tools
            parent_names_before = ["PARENT_LEGACY_DISPATCH_MARKER"]
            model_tools._last_resolved_tool_names = list(parent_names_before)
            from tools.delegate_tool_validation import _parent_tool_grant
            assert _parent_tool_grant(parent) is not None
            assert model_tools._last_resolved_tool_names == parent_names_before

            output = json.loads(delegate_task(tasks=[
                {"goal": "explicit read-only child", "tools": ["read_file"]},
                {"goal": "narrower-profile omitted grant", "profile": "restricted"},
                {"goal": "empty child boundary", "tools": []},
                {"goal": "nested child should attempt escalation"},
                {"goal": "depth-boundary child should delegate"},
            ], parent_agent=parent))
            assert [entry["status"] for entry in output["results"]] == ["completed"] * 5, output
            assert [entry["summary"] for entry in output["results"]] == [
                "NARROW_CHILD_DONE", "PROFILE_CHILD_DONE", "EMPTY_CHILD_DONE",
                "NESTED_CHILD_DONE", "BOUNDARY_CHILD_DONE",
            ]

            bodies = [record["body"] for record in server.requests]
            narrow = next(body for body in bodies if any(
                message.get("role") == "user" and "explicit read-only child" in str(message.get("content"))
                for message in body.get("messages", [])
            ))
            assert {item["function"]["name"] for item in narrow.get("tools", [])} == {"read_file"}
            profile_child = next(body for body in bodies if any(
                message.get("role") == "user" and "narrower-profile omitted grant" in str(message.get("content"))
                for message in body.get("messages", [])
            ))
            assert {item["function"]["name"] for item in profile_child.get("tools", [])} == {"read_file"}
            assert "delegate_task" not in {item["function"]["name"] for item in profile_child.get("tools", [])}

            empty = next(body for body in bodies if any(
                message.get("role") == "user" and "empty child boundary" in str(message.get("content"))
                for message in body.get("messages", [])
            ))
            assert not empty.get("tools")

            omitted = next(body for body in bodies if any(
                message.get("role") == "user" and "nested child should attempt escalation" in str(message.get("content"))
                for message in body.get("messages", [])
            ))
            assert {item["function"]["name"] for item in omitted.get("tools", [])} == {
                "read_file", "delegate_task",
            }

            nested_child = omitted
            assert {item["function"]["name"] for item in nested_child.get("tools", [])} == {
                "read_file", "delegate_task",
            }
            grandchild = next(body for body in bodies if any(
                message.get("role") == "user" and "grandchild should read fixture" in str(message.get("content"))
                for message in body.get("messages", [])
            ))
            # At max_spawn_depth the inherited grant is intersected with the leaf role.
            assert {item["function"]["name"] for item in grandchild.get("tools", [])} == {"read_file"}
            grandchild_read_call = next(call for body in bodies for message in body.get("messages", [])
                                        if message.get("role") == "assistant"
                                        for call in message.get("tool_calls", [])
                                        if call.get("function", {}).get("name") == "read_file"
                                        and readable.as_posix() in call.get("function", {}).get("arguments", ""))
            escalation_call = next(call for body in bodies for message in body.get("messages", [])
                                   if message.get("role") == "assistant" for call in message.get("tool_calls", [])
                                   if call.get("function", {}).get("name") == "delegate_task"
                                   and "explicit escalation must refuse" in call.get("function", {}).get("arguments", ""))
            escalation_result = next(message for body in bodies for message in body.get("messages", [])
                                     if message.get("role") == "tool"
                                     and message.get("tool_call_id") == escalation_call["id"])
            nested_result = json.loads(escalation_result["content"])
            assert [entry["status"] for entry in nested_result["results"]] == ["error", "completed"]
            assert nested_result["results"][0]["api_calls"] == 0
            assert "exceed the owning parent's tools" in nested_result["results"][0]["error"]
            assert nested_result["results"][1]["summary"] == "GRANDCHILD_DONE"
            read_result = next(message for body in bodies for message in body.get("messages", [])
                               if message.get("role") == "tool"
                               and message.get("tool_call_id") == grandchild_read_call["id"])
            assert "nested grandchild read succeeded" in read_result["content"]
            assert all("write_file" not in {item.get("function", {}).get("name") for item in body.get("tools", [])}
                       for body in bodies)

            boundary_child = next(body for body in bodies if any(
                message.get("role") == "user" and "depth-boundary child should delegate" in str(message.get("content"))
                for message in body.get("messages", [])
            ))
            assert {item["function"]["name"] for item in boundary_child.get("tools", [])} == {
                "read_file", "delegate_task",
            }
            max_depth_child = next(body for body in bodies if any(
                message.get("role") == "user" and "maximum-depth grandchild attempts forbidden delegation"
                in str(message.get("content"))
                for message in body.get("messages", [])
            ))
            assert {item["function"]["name"] for item in max_depth_child.get("tools", [])} == {"read_file"}
            forbidden_depth_call = next(call for body in bodies for message in body.get("messages", [])
                                        if message.get("role") == "assistant"
                                        for call in message.get("tool_calls", [])
                                        if call.get("function", {}).get("name") == "delegate_task"
                                        and "must not spawn beyond the depth ceiling"
                                        in call.get("function", {}).get("arguments", ""))
            forbidden_depth_result = next(message for body in bodies for message in body.get("messages", [])
                                          if message.get("role") == "tool"
                                          and message.get("tool_call_id") == forbidden_depth_call["id"])
            assert "outside this child's exact tool grant" in forbidden_depth_result["content"]
            assert not any("must not spawn beyond the depth ceiling" in str(message.get("content", ""))
                           for body in bodies for message in body.get("messages", []))
            after = parent.run_conversation("parent prompt after nested grants")
            assert after["final_response"] == "PARENT_AFTER_DONE"
            assert model_tools._last_resolved_tool_names == parent_names_before
            requests = server.main_requests()
            parent_after = next(request for request in requests if any(
                message.get("role") == "user" and "parent prompt after nested grants" in str(message.get("content"))
                for message in request["messages"]
            ))
            parent_prompt_after = next(message["content"] for message in parent_after["messages"]
                                       if message.get("role") == "system")
            assert parent_prompt_after == parent_prompt_before
            assert json.dumps(parent.tools, sort_keys=True, separators=(",", ":")).encode() == parent_tools_before
            assert effective_tool_ceiling(parent) == frozenset({"read_file", "delegate_task"})
            strict_parent.close()
        finally:
            parent.close()



def test_combined_exact_tool_child_refuses_ungranted_runtime_effect_and_keeps_parent_scope(tmp_path, monkeypatch):
    """A real delegated child receives its combined route/persona/skill/tool policy; denied exact tool never writes."""
    import json
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall
    from tools.delegate_tool import delegate_task
    from tools.registry import registry

    target = tmp_path / "forbidden-combined-write.txt"
    effect = []
    schema = {"type": "object", "properties": {
        "path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"]}
    name = "combined_grant_write_fixture"
    previous = registry.snapshot_registration(name)
    assert previous is None
    with FakeLLMServer([
        ToolCall(name, {"path": str(target), "content": "must never persist"}),
        Text("combined grant check complete"),
    ], api_key="combined-parent-key") as server:
        home = tmp_path / ".hermes"
        home.mkdir()
        skill_dir = home / "skills" / "research" / "combined-exact-skill"
        skill_dir.mkdir(parents=True)
        skill_body = "---\nname: combined-exact-skill\ndescription: fixture\n---\nCOMBINED EXACT SKILL BODY"
        (skill_dir / "SKILL.md").write_text(skill_body, encoding="utf-8")
        (home / "config.yaml").write_text(
            "model:\n  provider: custom\n  default: fixture-model\n"
            f"  base_url: {server.base_url}\n  api_key: combined-parent-key\n"
            "agent:\n  api_max_retries: 1\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HOME", str(tmp_path / "host-home"))
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
        import hermes_state
        monkeypatch.setattr(hermes_state, "_STATE_DB_GUARD_EXTRA_DENY_ROOTS", ())
        db = SessionDB(db_path=home / "state.db")
        parent = AIAgent(
            provider="custom", model="fixture-model", base_url=server.base_url,
            api_key="combined-parent-key", api_mode="chat_completions", enabled_toolsets=["delegation", "file"],
            quiet_mode=True, skip_context_files=True, skip_memory=True, save_trajectories=False,
            platform="cli", session_id="combined-exact-parent", session_db=db,
        )
        parent._ensure_db_session()
        parent_names = tuple(parent.valid_tool_names)
        # Add the fixture tool only after the parent startup, so its model-visible and execution
        # grants remain distinct: the task below explicitly permits read_file only.
        registry.register(name=name, toolset="file", schema={
            "name": name, "description": "fixture side effect", "parameters": schema,
        }, handler=lambda args, **kwargs: effect.append(dict(args)) or json.dumps({"written": True}))
        protected = {p: p.read_bytes() for p in (home / "config.yaml", skill_dir / "SKILL.md")}
        try:
            from tools.delegate_tool import DELEGATE_TASK_SCHEMA as delegate_schema
            from tools.delegate_tool import oneshot_delegate_function
            parent.tools = [{"type": "function", "function": oneshot_delegate_function(delegate_schema)}]
            parent._delegation_inline = True
            result = json.loads(delegate_task(tasks=[{
                "goal": "Verify exact write permission stays denied",
                "model": "fixture-model", "system_prompt": "COMBINED EXACT PERSONA",
                "skills": ["combined-exact-skill"], "tools": ["read_file"],
            }], parent_agent=parent))
            entry = result["results"][0]
            assert entry["status"] == "completed", entry
            assert entry["summary"] == "combined grant check complete"
            assert len(server.main_requests()) == 2
            request, tool_followup = server.main_requests()
            assert "COMBINED EXACT PERSONA" in request["messages"][0]["content"]
            assert skill_body in request["messages"][0]["content"]
            assert [tool["function"]["name"] for tool in request["tools"]] == ["read_file"]
            assert f"Tool '{name}' does not exist. Available tools: read_file" in str(tool_followup["messages"])
            assert not target.exists() and effect == []
            from tools.delegate_tool_routes import delegated_profile_scope
            with delegated_profile_scope(home):
                child_rows = db.list_sessions_rich(source="subagent", include_subagents=True)
            assert len(child_rows) == 1
            child = db.get_messages_as_conversation(child_rows[0]["id"])
            assert any(msg["role"] == "user" for msg in child)
            assert db.get_messages_as_conversation(parent.session_id) == []
            assert tuple(parent.valid_tool_names) == parent_names
            assert {p: p.read_bytes() for p in protected} == protected
        finally:
            parent.close()
            db.close()
            registry.deregister(name)
def test_public_delegate_child_runs_readonly_execute_code_end_to_end(tmp_path, monkeypatch):
    import json
    from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall
    requests = []
    root = tmp_path / ".hermes"
    root.mkdir()
    test_config = ("platform_toolsets:\n  subagent: [hermes-cli, code_execution, delegation]\n"
                   "delegation:\n  worktree_isolation: false\n"
                   "code_execution:\n  mode: strict\n  timeout: 15\n")
    (root / "config.yaml").write_text(test_config, encoding="utf-8")
    (tmp_path / "config.yaml").write_text(test_config, encoding="utf-8")
    source = tmp_path / "delegate-execute-source.txt"
    source.write_text("loopback child readback fixture\n", encoding="utf-8")
    observed_toolsets = {}
    def answer(request):
        requests.append(request)
        messages = request["body"].get("messages", [])
        prompt = " ".join(str(message.get("content", "")) for message in messages
                          if message.get("role") == "user")
        tools = {item["function"]["name"]: item for item in request["body"].get("tools", [])}
        if any(message.get("role") == "tool" for message in messages):
            return Text("CHILD_EXECUTION_COMPLETED")
        if "execute-only computation" in prompt:
            observed_toolsets["execute-only"] = set(tools)
            if "execute_code" not in tools:
                return Text("execute_code missing")
            return ToolCall("execute_code", {"code": "print(6 * 7)"})
        if "execute code to read one file" in prompt:
            observed_toolsets["read-only"] = set(tools)
            if "execute_code" not in tools:
                return Text("execute_code missing")
            return ToolCall("execute_code", {
                "code": f"import hermes_tools; print(hermes_tools.read_file({str(source)!r})['content'])"
            })
        return Text("Unexpected loopback prompt")
    with FakeLLMServer(answer, aux=lambda _request: Text("")) as server:
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_HOME", str(root))
        from run_agent import AIAgent
        from tools.delegate_tool import delegate_task
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "loopback-sessions.db")
        parent = AIAgent(
            api_key="loopback-test-key", base_url=server.base_url, provider="custom",
            api_mode="chat_completions", model="loopback-model",
            enabled_toolsets=["hermes-cli", "code_execution"], quiet_mode=True,
            skip_context_files=True, skip_memory=True, save_trajectories=False,
            platform="subagent", session_id="rpc-loopback-parent", session_db=db, cwd=str(tmp_path),
        )
        db.create_session(parent.session_id, "test-parent")
        try:
            read_result = json.loads(delegate_task(
                tasks=[{"goal": "execute code to read one file", "tools": ["execute_code", "read_file"]}],
                parent_agent=parent,
            ))
            execute_only_result = json.loads(delegate_task(
                tasks=[{"goal": "execute-only computation", "tools": ["execute_code"]}],
                parent_agent=parent,
            ))
            entries = read_result["results"] + execute_only_result["results"]
            assert len(entries) == 2 and all(item["status"] == "completed" for item in entries), entries
            assert all(item["summary"] == "CHILD_EXECUTION_COMPLETED" for item in entries)
            child_requests = [request for request in requests if any(
                "execute code to read one file" in str(message.get("content", ""))
                for message in request["body"].get("messages", [])
            )]
            assert child_requests
            body = child_requests[0]["body"]
            assert {item["function"]["name"] for item in body["tools"]} >= {
                "execute_code", "read_file",
            }
            assert observed_toolsets["execute-only"] == {"execute_code"}
            assert observed_toolsets["read-only"] >= {"execute_code", "read_file"}
            result_messages = [message["content"] for request in requests
                               for message in request["body"].get("messages", [])
                               if message.get("role") == "tool"]
            assert any("loopback child readback fixture" in str(content) for content in result_messages)
            assert any('"output": "42\\n"' in str(content) for content in result_messages)
        finally:
            parent.close()
            db.close()
def test_delegated_child_teardown_retires_local_kernel_authority():
    from unittest.mock import patch
    from agent.delegation_context import delegated_child_context
    from tools.code_execution_tool import execute_code
    from tools.code_kernel import _KERNELS, shutdown_all_kernels, shutdown_kernels_for_delegated_child
    shutdown_all_kernels()
    try:
        with patch("tools.code_execution_tool._load_config", return_value={"mode": "strict", "timeout": 20}):
            with delegated_child_context("teardown-fixture"):
                result = json.loads(execute_code("retained = True", task_id="teardown-fixture"))
            assert result["status"] == "success", result
        owner_kernels = [kernel for kernel in _KERNELS.values()
                         if kernel.owner.endswith("::child::teardown-fixture")]
        assert len(owner_kernels) == 1
        kernel = owner_kernels[0]
        retired_authority = kernel.cell_authority
        assert retired_authority is not None and not retired_authority.active
        shutdown_kernels_for_delegated_child("teardown-fixture")
        assert not any(item.owner.endswith("::child::teardown-fixture") for item in _KERNELS.values())
        kernel.proc.wait(timeout=10)
        assert not kernel.alive()
        assert not retired_authority.active
    finally:
        shutdown_all_kernels()
def test_execute_code_helper_grant_keeps_empty_explicit_and_schema_exact():
    from tools.code_execution_tool import (
        SANDBOX_ALLOWED_TOOLS, _sandbox_tools_for, generate_hermes_tools_module,
    )
    from tools.thread_context import exact_tool_scope
    import model_tools
    base = {"type": "function", "function": {"name": "execute_code", "description": ""}}
    with exact_tool_scope({"execute_code"}):
        assert _sandbox_tools_for(None) == frozenset()
        empty_module = generate_hermes_tools_module(sorted(_sandbox_tools_for(None)))
        rewritten = model_tools._rewrite_execute_code(base, set(SANDBOX_ALLOWED_TOOLS))
        assert rewritten is not None
        description = rewritten["function"]["description"]
        assert "read_file(" not in description
        assert "write_file(" not in description
        assert "terminal(" not in description
    with exact_tool_scope({"execute_code", "read_file"}):
        assert _sandbox_tools_for(None) == frozenset({"read_file"})
        module = generate_hermes_tools_module(sorted(_sandbox_tools_for(None)))
        rewritten = model_tools._rewrite_execute_code(base, set(SANDBOX_ALLOWED_TOOLS))
        assert rewritten is not None
        description = rewritten["function"]["description"]
    assert "def read_file(" in module
    assert all(f"def {name}(" not in module for name in ("write_file", "patch", "terminal"))
    assert "read_file(" in description
    assert "write_file(" not in description
    assert "terminal(" not in description
    assert "def read_file(" not in empty_module
def test_execute_only_exact_scope_runs_real_local_kernel_computation(tmp_path):
    from tools.code_execution_env import _resolve_child_cwd, _resolve_child_python
    from tools.code_kernel import _KERNELS, execute_in_session_kernel, shutdown_all_kernels
    from tools.thread_context import exact_tool_scope
    shutdown_all_kernels()
    try:
        with exact_tool_scope({"execute_code"}):
            result = json.loads(execute_in_session_kernel(
                "print(6 * 7)", task_id="execute-only-real-kernel", mode="strict",
                child_python=_resolve_child_python("strict"),
                child_cwd=_resolve_child_cwd("strict", "", task_id="execute-only-real-kernel"),
                sandbox_tools=frozenset(), timeout=15, max_tool_calls=5,
                reset=False, is_interrupted=lambda: False,
            ))
        assert result["status"] == "success", result
        assert result["output"].strip() == "42"
        assert len(_KERNELS) == 1
        assert next(iter(_KERNELS))[-1] == ()
    finally:
        shutdown_all_kernels()
def test_file_rpc_exact_scope_uses_real_script_transport_and_real_tool_dispatch(tmp_path, monkeypatch):
    import os
    import subprocess
    import threading
    from pm.shell import bash
    from tools.code_execution_rpc import _rpc_poll_loop
    from tools.code_execution_tool import generate_hermes_tools_module
    from tools.code_kernel import CellAuthority
    from tools.thread_context import exact_tool_scope, propagate_context_to_thread
    shell = bash()
    assert shell
    rpc_dir = tmp_path / "rpc"
    rpc_dir.mkdir()
    source = tmp_path / "source.txt"
    source.write_text("genuine file transport read\n", encoding="utf-8")
    forbidden_write = tmp_path / "forbidden.txt"
    forbidden_terminal = tmp_path / "terminal-ran.txt"
    module = tmp_path / "hermes_tools.py"
    module.write_text(generate_hermes_tools_module(
        ["read_file"], transport="file"), encoding="utf-8")
    token = "fixture-rpc-token"
    cell_token = "fixture-cell-token"
    monkeypatch.setenv("HERMES_RPC_DIR", str(rpc_dir))
    monkeypatch.setenv("HERMES_RPC_TOKEN", token)
    monkeypatch.setenv("HERMES_RPC_CELL_TOKEN", cell_token)
    class Shell:
        def execute(self, command, cwd=None, timeout=None, stdin_data=None):
            result = subprocess.run([shell, "-c", command], cwd=cwd, timeout=timeout,
                                    env=dict(os.environ), input=stdin_data or "", capture_output=True,
                                    text=True, check=False)
            return {"output": result.stdout, "returncode": result.returncode}
    authority = None
    stop = threading.Event()
    log, counter = [], [0]
    grant = frozenset({"read_file"})
    with exact_tool_scope(grant):
        authority = CellAuthority("file-rpc-exact", grant, cell_token)
        poller = threading.Thread(target=propagate_context_to_thread(_rpc_poll_loop), daemon=True,
                                  args=(Shell(), str(rpc_dir), "file-rpc-exact", log, counter, 8,
                                        frozenset({"read_file", "write_file", "terminal"}), stop, token),
                                  kwargs={"cell_authority": authority})
        poller.start()
        try:
            script = f'''\
import hermes_tools, json
read = hermes_tools.read_file({str(source)!r})
write = hermes_tools._call("write_file", {{"path": {str(forbidden_write)!r}, "content": "denied"}})
terminal = hermes_tools._call("terminal", {{"command": "touch {forbidden_terminal}", "timeout": 2}})
print(json.dumps({{"read": read["content"], "write": write, "terminal": terminal}}))
'''
            env = dict(os.environ, PYTHONPATH=str(tmp_path))
            result = subprocess.run([os.sys.executable, "-c", script], cwd=tmp_path, env=env,
                                    capture_output=True, text=True, timeout=10, check=False)
            assert result.returncode == 0, result.stderr
            payload = json.loads(result.stdout.strip())
            assert payload["read"].endswith("genuine file transport read")
            assert "exact tool grant" in payload["write"]["error"]
            assert "exact tool grant" in payload["terminal"]["error"]
            assert not forbidden_write.exists()
            assert not forbidden_terminal.exists()
            assert counter == [1]
            assert [item["tool"] for item in log] == ["read_file"]
        finally:
            stop.set()
            poller.join(timeout=10)
            authority.retire()
            assert not poller.is_alive()
def test_local_kernel_entry_intersects_helper_grant_before_spawn(tmp_path):
    from tools.code_execution_env import _resolve_child_cwd, _resolve_child_python
    from tools.code_execution_tool import SANDBOX_ALLOWED_TOOLS
    from tools.code_kernel import _KERNELS, execute_in_session_kernel, shutdown_all_kernels
    from tools.thread_context import exact_tool_scope
    shutdown_all_kernels()
    source = tmp_path / "direct-local-read.txt"
    source.write_text("direct local helper intersection\n", encoding="utf-8")
    script = f'''\
import hermes_tools, json
read = hermes_tools.read_file({str(source)!r})
helpers = sorted(name for name in {sorted(SANDBOX_ALLOWED_TOOLS)!r} if hasattr(hermes_tools, name))
print(json.dumps({{"read": read["content"], "helpers": helpers}}))
'''
    try:
        with exact_tool_scope({"execute_code", "read_file"}):
            result = json.loads(execute_in_session_kernel(
                script, task_id="direct-local-helper-scope", mode="strict",
                child_python=_resolve_child_python("strict"),
                child_cwd=_resolve_child_cwd("strict", "", task_id="direct-local-helper-scope"),
                sandbox_tools=frozenset(SANDBOX_ALLOWED_TOOLS), timeout=15,
                max_tool_calls=5, reset=False, is_interrupted=lambda: False,
            ))
        assert result["status"] == "success", result
        assert "direct local helper intersection" in result["output"]
        assert '"helpers": ["read_file"]' in result["output"]
        assert len(_KERNELS) == 1
        assert next(iter(_KERNELS))[-1] == ("read_file",)
    finally:
        shutdown_all_kernels()

def test_local_kernel_freezes_and_retires_per_cell_rpc_authority(tmp_path):
    from unittest.mock import patch

    from tools.code_execution_tool import execute_code
    from tools.code_kernel import _KERNELS, shutdown_all_kernels
    from tools.thread_context import exact_tool_scope

    shutdown_all_kernels()
    target = tmp_path / "late-write.txt"
    code = f'''\
import contextvars, json, pathlib, threading, hermes_tools
event = threading.Event()
late_result = {{}}
old_context = contextvars.copy_context()
def late_call():
    event.wait()
    late_result.update(old_context.run(hermes_tools.write_file, {str(target)!r}, "stale-cell"))
thread = threading.Thread(target=late_call)
thread.start()
'''
    next_code = f'''\
event.set()
thread.join(timeout=3)
print(json.dumps({{"exists": pathlib.Path({str(target)!r}).exists(), "late": late_result}}))
'''
    try:
        with patch("tools.code_execution_tool._load_config", return_value={"mode": "strict", "timeout": 20}):
            with exact_tool_scope({"execute_code", "write_file"}):
                first = json.loads(execute_code(code, task_id="cell-retirement", enabled_tools=None))
                second = json.loads(execute_code(next_code, task_id="cell-retirement", enabled_tools=None))
        assert first["status"] == "success", first
        assert second["status"] == "success", second
        assert '"exists": false' in second["output"]
        assert "retired" in second["output"]
        assert not target.exists()
        kernel = next(iter(_KERNELS.values()))
        assert kernel.cell_authority is not None and not kernel.cell_authority.active
        denied = json.loads(kernel.cell_authority.dispatch("write_file", {
            "path": str(target), "content": "after-teardown",
        }))
        assert "retired" in denied["error"]
        assert not target.exists()
    finally:
        shutdown_all_kernels()


def test_remote_cell_authority_intersects_helper_allowlist_and_retires(tmp_path):
    from tools.code_execution_rpc import _handle_rpc_request
    from tools.code_kernel import CellAuthority
    from tools.thread_context import exact_tool_scope

    effects = []
    with exact_tool_scope({"execute_code", "read_file"}):
        authority = CellAuthority("remote-cell", frozenset({"read_file", "write_file"}), "cell-secret")
    assert authority.allowed_tools == frozenset({"read_file"})
    counter, log = [0], []
    def dispatch(name, args):
        effects.append((name, args))
        return json.dumps({"ok": True})
    request = {"tool": "write_file", "args": {"path": str(tmp_path / "never"), "content": "x"},
               "cell_token": authority.token}
    denied = json.loads(_handle_rpc_request(
        request, allowed_tools=frozenset({"read_file", "write_file"}), tool_call_counter=counter,
        max_tool_calls=5, dispatch=dispatch, tool_call_log=log, call_start=0, where="test",
        cell_authority=authority))
    assert "exact tool grant" in denied["error"]
    assert effects == [] and counter == [0]
    authority.retire()
    stale = json.loads(_handle_rpc_request(
        {"tool": "read_file", "args": {}, "cell_token": authority.token},
        allowed_tools=frozenset({"read_file"}), tool_call_counter=counter, max_tool_calls=5,
        dispatch=dispatch, tool_call_log=log, call_start=0, where="test", cell_authority=authority))
    assert "retired" in stale["error"]
    assert effects == [] and counter == [0]


def test_remote_concurrent_cells_keep_one_authority_window(tmp_path):
    import json
    import os
    import subprocess
    import threading
    from pathlib import Path
    from pm.shell import bash
    from tools.code_kernel_remote import (
        _REMOTE_KERNELS, execute_in_remote_kernel, shutdown_remote_kernels_for_owner,
    )
    from tools.thread_context import exact_tool_scope

    shell = bash()
    assert shell
    first_result_waiting, release_first_result = threading.Event(), threading.Event()
    second_request_published = threading.Event()

    class LocalFileEnvironment:
        def __init__(self):
            self.lock = threading.Lock()
            self.first_result_paused = False

        def get_temp_dir(self):
            return str(tmp_path)

        def execute(self, command, cwd="/", timeout=20, stdin_data=None):
            if "cell_req_000002" in command:
                second_request_published.set()
            if "cat" in command and "cell_res_000001.json" in command:
                with self.lock:
                    pause = not self.first_result_paused
                    self.first_result_paused = True
                if pause:
                    first_result_waiting.set()
                    assert release_first_result.wait(timeout=10)
            result = subprocess.run([shell, "-c", command], cwd=cwd, timeout=timeout,
                                    env=dict(os.environ), input=stdin_data or "", capture_output=True,
                                    text=True, check=False)
            return {"output": result.stdout + result.stderr, "returncode": result.returncode}

    env = LocalFileEnvironment()
    source_a, source_b = tmp_path / "parallel-a.txt", tmp_path / "parallel-b.txt"
    source_a.write_text("first remote cell", encoding="utf-8")
    source_b.write_text("second remote cell", encoding="utf-8")
    results, failures = {}, []
    second_call_entered = threading.Event()

    def run_cell(label, source):
        try:
            if label == "second":
                second_call_entered.set()
            with exact_tool_scope({"execute_code", "read_file"}):
                results[label] = execute_in_remote_kernel(
                    f"import hermes_tools; print(hermes_tools.read_file({str(source)!r})['content'])",
                    env=env, env_type="local-file-concurrency", task_env_id="remote-cell-window",
                    sandbox_tools=frozenset({"read_file"}), timeout=15,
                    max_tool_calls=5, reset=False, idle_exit=60,
                )
        except BaseException as exc:
            failures.append(exc)

    first = threading.Thread(target=run_cell, args=("first", source_a))
    second = threading.Thread(target=run_cell, args=("second", source_b))
    first.start()
    try:
        assert first_result_waiting.wait(timeout=10)
        kernel = next(kernel for kernel in _REMOTE_KERNELS.values() if kernel.owner == "remote-cell-window")
        first_authority = kernel.cell_authority
        assert first_authority is not None and first_authority.active
        second.start()
        # The first cell is held after execution but before its result is read. A
        # second same-key request must remain outside the RPC/cell authority window.
        assert second_call_entered.wait(timeout=3)
        assert not second_request_published.wait(timeout=0.5)
        assert kernel.cell_authority is first_authority
    finally:
        release_first_result.set()
        first.join(timeout=10)
        if second.ident is not None:
            second.join(timeout=10)
    assert not first.is_alive() and not second.is_alive()
    assert failures == [], failures
    assert results["first"] is not None and results["first"]["status"] == "success", results
    assert results["second"] is not None and results["second"]["status"] == "success", results
    assert "first remote cell" in results["first"]["stdout"]
    assert "second remote cell" in results["second"]["stdout"]
    assert second_request_published.is_set()
    kernel = next(kernel for kernel in _REMOTE_KERNELS.values() if kernel.owner == "remote-cell-window")
    assert kernel.attached == 0 and kernel.cell_authority is None
    shutdown_remote_kernels_for_owner("remote-cell-window")
    assert not any(kernel.owner == "remote-cell-window" for kernel in _REMOTE_KERNELS.values())


def test_remote_persistent_kernel_runs_real_cell_and_rpc_over_file_transport(tmp_path):
    import os
    import subprocess
    from pm.shell import bash
    from tools.code_execution_tool import SANDBOX_ALLOWED_TOOLS
    from tools.code_kernel_remote import (
        _REMOTE_KERNELS, execute_in_remote_kernel, shutdown_remote_kernels_for_owner,
    )
    from tools.thread_context import exact_tool_scope

    shell = bash()
    assert shell
    class LocalFileEnvironment:
        def get_temp_dir(self): return str(tmp_path)
        def execute(self, command, cwd="/", timeout=20, stdin_data=None):
            result = subprocess.run([shell, "-c", command], cwd=cwd, timeout=timeout,
                                    env=dict(os.environ), input=stdin_data or "", capture_output=True,
                                    text=True, check=False)
            return {"output": result.stdout + result.stderr, "returncode": result.returncode}
    env = LocalFileEnvironment()
    source = tmp_path / "persistent-source.txt"
    source.write_text("remote persistent file fixture\n", encoding="utf-8")
    forbidden = tmp_path / "persistent-forbidden.txt"
    code = f'''\
import hermes_tools, json
read = hermes_tools.read_file({str(source)!r})
denied = hermes_tools._call("write_file", {{"path": {str(forbidden)!r}, "content": "forbidden"}})
print(json.dumps({{"read": read["content"], "denied": denied,
                  "has_write_stub": hasattr(hermes_tools, "write_file")}}))
'''
    with exact_tool_scope({"execute_code", "read_file"}):
        first = execute_in_remote_kernel(
            code, env=env, env_type="local-file-fixture", task_env_id="persistent-file-rpc",
            sandbox_tools=frozenset(SANDBOX_ALLOWED_TOOLS), timeout=8,
            max_tool_calls=5, reset=False, idle_exit=60,
        )
    assert first is not None and first["status"] == "success", first
    assert "remote persistent file fixture" in first["stdout"]
    assert "outside this execute_code cell's exact tool grant" in first["stdout"]
    assert '"has_write_stub": false' in first["stdout"]
    assert not forbidden.exists()
    kernels = [kernel for kernel in _REMOTE_KERNELS.values() if kernel.owner == "persistent-file-rpc"]
    assert len(kernels) == 1
    shutdown_remote_kernels_for_owner("persistent-file-rpc")
    assert not any(kernel.owner == "persistent-file-rpc" for kernel in _REMOTE_KERNELS.values())
    assert not kernels[0].is_alive()
