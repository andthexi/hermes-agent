"""Regression tests for exact parent-surface permission boundaries."""

import json


def test_public_exact_grant_cannot_admit_tool_appearing_after_parent_surface_freezes(tmp_path, monkeypatch):
    """A real parent's frozen tool schema stays the permission boundary across availability changes."""
    import hermes_yaml as yaml

    from hermes_state import SessionDB
    from run_agent import AIAgent
    from tests.fakes.fake_llm_provider import FakeLLMServer, Text
    from tools.delegate_tool import delegate_task
    from tools.registry import invalidate_check_fn_cache, registry

    available = {"web_search": False, "web_extract": True}
    prior = {name: registry.snapshot_registration(name) for name in available}
    for name in available:
        registry.register(
            name=name, toolset="web", schema={
                "name": name, "description": "Local surface availability fixture",
                "parameters": {"type": "object", "properties": {}},
            }, handler=lambda *_args, **_kwargs: "unused",
            check_fn=lambda _name=name: available[_name], override=True,
        )
    parent = None
    db = None
    try:
        home = tmp_path / ".hermes"
        home.mkdir()
        (home / "config.yaml").write_text(yaml.safe_dump({
            "platform_toolsets": {"cli": ["web", "delegation"]},
            "tools": {"tool_search": {"enabled": "off", "defer": []}},
        }), encoding="utf-8")
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_HOME", str(home))

        with FakeLLMServer(Text("fixture child completed")) as server:
            db = SessionDB(db_path=tmp_path / "sessions.db")
            parent = AIAgent(
                api_key="fixture-key", base_url=server.base_url, provider="custom", api_mode="chat_completions",
                model="fixture-model", enabled_toolsets=["web", "delegation"],
                quiet_mode=True, skip_context_files=True, skip_memory=True, save_trajectories=False,
                platform="cli", session_id="frozen-surface-parent", session_db=db,
            )
            parent._ensure_db_session()
            assert "web_search" not in parent.valid_tool_names
            assert "web_extract" in parent.valid_tool_names

            available["web_search"] = True
            invalidate_check_fn_cache()
            import model_tools
            refreshed = model_tools.get_tool_definitions(
                enabled_toolsets=parent.enabled_toolsets, disabled_toolsets=parent.disabled_toolsets,
                quiet_mode=True, skip_tool_search_assembly=True, update_last_resolved_tool_names=False,
            )
            assert {"web_search", "web_extract"}.issubset(
                {item["function"]["name"] for item in refreshed})

            output = json.loads(delegate_task(tasks=[
                {"goal": "must not get the newly available web tool", "tools": ["web_search"]},
                {"goal": "use the parent's visible web tool", "tools": ["web_extract"]},
            ], parent_agent=parent))
            requests = server.main_requests()

        assert [item["status"] for item in output["results"]] == ["error", "completed"]
        refused, sibling = output["results"]
        assert refused["api_calls"] == 0
        assert "exceed the owning parent's tools" in refused["error"]
        assert sibling["status"] == "completed"
        assert sibling["api_calls"] == 1
        assert len(requests) == 1
        assert {item["function"]["name"] for item in requests[0]["tools"]} == {"web_extract"}
    finally:
        if parent is not None:
            parent.close()
        if db is not None:
            db.close()
        invalidate_check_fn_cache()
        for name, previous in prior.items():
            current = registry.snapshot_registration(name)
            if current is not None:
                registry.restore_registration(name, current, previous)


def test_unrestricted_parent_can_delegate_visible_plugin_with_search_off(tmp_path, monkeypatch):
    """None selects available plugin toolsets; [] remains an empty permission surface."""
    import hermes_yaml as yaml
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from tests.fakes.fake_llm_provider import FakeLLMServer, Text
    from tools.delegate_tool import delegate_task
    from tools.registry import invalidate_check_fn_cache, registry

    visible = "parent_surface_plugin"
    unavailable = "parent_surface_unavailable"
    registrations = {name: registry.snapshot_registration(name) for name in (visible, unavailable)}
    home = tmp_path / "plugin-profile"
    home.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(yaml.safe_dump({
        "platform_toolsets": {"cli": ["delegation", "parent-surface-plugin"]},
        "tools": {"tool_search": {"enabled": "off", "defer": []}},
        "delegation": {"fallback_providers": [], "worktree_isolation": False},
    }), encoding="utf-8")
    for name in (visible, unavailable):
        registry.register(
            name=name, toolset="parent-surface-plugin",
            schema={"name": name, "description": "Local permission fixture",
                    "parameters": {"type": "object", "properties": {}}},
            handler=lambda *_args, **_kwargs: "unused", check_fn=lambda _name=name: _name == visible,
        )
    parents = []
    db = SessionDB(db_path=home / "sessions.db")
    try:
        with FakeLLMServer([Text("permitted plugin child completed")]) as server:
            for selection in (None, []):
                parent = AIAgent(
                    api_key="fixture-key", base_url=server.base_url, provider="custom",
                    api_mode="chat_completions", model="fixture-model", enabled_toolsets=selection,
                    platform="cli", quiet_mode=True, skip_context_files=True, skip_memory=True,
                    save_trajectories=False, session_id=f"plugin-parent-{selection}", session_db=db,
                )
                parents.append(parent)
                parent._ensure_db_session()
                if selection is None:
                    assert visible in parent.valid_tool_names
                    assert unavailable not in parent.valid_tool_names
                    result = json.loads(delegate_task(tasks=[
                        {"goal": "Use the actually visible plugin tool", "tools": [visible]},
                        {"goal": "Refuse the unavailable plugin sibling", "tools": [unavailable]},
                    ], parent_agent=parent))
                    assert "results" in result, result
                    allowed, denied = result["results"]
                    assert allowed["status"] == "completed" and allowed["api_calls"] == 1
                    assert denied["status"] == "error" and denied["api_calls"] == 0
                else:
                    assert parent.valid_tool_names == set()
                    result = json.loads(delegate_task(tasks=[{
                        "goal": "An empty parent cannot grant the plugin", "tools": [visible],
                    }], parent_agent=parent))
                    assert "error" in result and "owning parent's tools" in result["error"]
            requests = server.main_requests()
            assert len(requests) == 1
            assert {tool["function"]["name"] for tool in requests[0]["tools"]} == {visible}
    finally:
        for parent in parents:
            parent.close()
        db.close()
        for name, prior in registrations.items():
            current = registry.snapshot_registration(name)
            if current is not None:
                registry.restore_registration(name, current, prior)
        invalidate_check_fn_cache()
