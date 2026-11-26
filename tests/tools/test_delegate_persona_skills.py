"""Behavior contract for first-persona child prompt composition and wire delivery."""

import json
from hermes_constants import get_hermes_home
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import hermes_yaml as yaml

from agent.delegation_launch import DelegationLaunchPolicy
from tools.delegate_tool_options import TaskRouteOptions


def _agent(home, policy=None):
    return SimpleNamespace(
        load_soul_identity=True,
        skip_context_files=True,
        _session_db=SimpleNamespace(db_path=home / "state.db"),
        _memory_store=None,
        _memory_manager=None,
        _plugin_system_prompt_sections_snapshot=(),
        _cached_system_prompt=None,
        _cached_system_prompt_static=None,
        context_compressor=None,
        valid_tool_names=set(),
        _task_completion_guidance=False,
        _parallel_tool_call_guidance=False,
        _tool_use_enforcement=False,
        _execution_guidance=False,
        provider="custom",
        model="loopback-model",
        platform="subagent",
        session_id="20261125_120000_persona",
        pass_session_id=False,
        api_mode="chat_completions",
        _memory_enabled=False,
        _user_profile_enabled=False,
        _environment_probe=False,
        _bot_mode_protocol=False,
        _delegation_launch_policy=policy,
        ephemeral_system_prompt=None,
        _auto_load_skills_resolved=True,
        _auto_load_skills_result=("", [], []),
    )


def test_task_persona_parser_preserves_supplied_bytes_and_defaults_to_replace():
    text = "\n  persona text\nliteral `x`  \n"
    option, error = TaskRouteOptions.parse({"goal": "task", "system_prompt": text}, 0)
    assert error is None and option is not None
    assert option.system_prompt == text
    assert option.system_prompt_mode == "replace"

    append, error = TaskRouteOptions.parse({"goal": "task", "system_prompt": "\n raw \n",
                                           "system_prompt_mode": "append"}, 0)
    assert error is None and append is not None
    assert append.system_prompt == "\n raw \n" and append.system_prompt_mode == "append"

    empty, error = TaskRouteOptions.parse({"goal": "task", "system_prompt": ""}, 0)
    assert empty is None and error and "non-empty string" in error
    whitespace_text = "  \npersona	 "
    whitespace, error = TaskRouteOptions.parse({"goal": "task", "system_prompt": whitespace_text}, 0)
    assert error is None and whitespace is not None and whitespace.system_prompt == whitespace_text
    invalid_mode, error = TaskRouteOptions.parse(
        {"goal": "task", "system_prompt": "x", "system_prompt_mode": "replace-all"}, 0)
    assert invalid_mode is None and "must be 'replace' or 'append'" in error


def test_persona_transport_preflight_uses_the_selected_route_transport():
    from tools.delegate_tool_routes import TaskRoute
    from tools.delegate_tool_validation import _resolve_task_reasoning

    option, error = TaskRouteOptions.parse(
        {"goal": "task", "system_prompt": "persona", "system_prompt_mode": "replace"}, 0)
    assert error is None and option is not None
    import pytest

    for final_creds, stale_route_creds, expected_error in (
        ({"model": "m", "provider": "openai-codex", "api_mode": "codex_app_server"},
         {"api_mode": "chat_completions"}, "Codex app-server"),
        ({"model": "m", "provider": "custom", "command": "acp"},
         {"command": None}, "external command transport"),
    ):
        route = TaskRoute(stale_route_creds, {})
        with pytest.raises(ValueError, match=expected_error):
            _resolve_task_reasoning(option, "m", final_creds["provider"], final_creds, route, None)


def test_codex_app_server_refuses_exact_grant_and_inherited_ceiling_before_startup():
    from types import SimpleNamespace
    from agent.delegation_launch import DelegationLaunchPolicy
    from tools.delegate_tool_options import TaskRouteOptions
    from tools.delegate_tool_routes import TaskRoute
    from tools.delegate_tool_validation import _apply_task_route_options

    app_server_route = TaskRoute({
        "model": "gpt-6-luna", "provider": "openai-codex", "api_mode": "codex_app_server",
    }, {"fallback_providers": []})
    parent = SimpleNamespace(
        _delegate_depth=0, valid_tool_names={"web_search"},
        _session_db=None, platform="cli", enabled_toolsets=["search"],
    )
    tasks = [
        {"goal": "empty exact grant", "tools": []},
        {"goal": "narrow exact grant", "tools": ["web_search"]},
    ]
    for index, task in enumerate(tasks):
        option, error = TaskRouteOptions.parse(task, index)
        assert error is None and option is not None
        route = _apply_task_route_options(index, task, option, app_server_route, {}, parent)
        assert route.error and "exact tools override" in route.error
        assert route.launch_policy is not None
        assert dict(route.launch_policy.diagnostics)["validation"] == "refused"

    inherited_parent = SimpleNamespace(
        **parent.__dict__,
        _delegation_launch_policy=DelegationLaunchPolicy(admitted_tool_names=("web_search",)),
    )
    legacy, error = TaskRouteOptions.parse({"goal": "inherited exact ceiling"}, 2)
    assert error is None and legacy is not None
    route = _apply_task_route_options(2, {"goal": "inherited exact ceiling"}, legacy,
                                     app_server_route, {}, inherited_parent)
    assert route.error and "inherited exact tool ceiling" in route.error
    assert route.launch_policy is not None
    assert dict(route.launch_policy.diagnostics)["validation"] == "refused"

    # Ordinary app-server work with no exact controls retains legacy behavior.
    route = _apply_task_route_options(3, {"goal": "legacy"}, legacy, app_server_route, {}, parent)
    assert route.error is None
    assert route.launch_policy is None


def test_identity_composition_replace_append_and_default_keep_guidance(tmp_path, monkeypatch):
    root = tmp_path / "hermes-test-home"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    (root / "SOUL.md").write_text("selected profile persona", encoding="utf-8")
    from agent.system_prompt import build_system_prompt_parts
    import agent.system_prompt as prompt

    monkeypatch.setattr(prompt, "_post_workspace_parts", lambda agent: [])
    monkeypatch.setattr(prompt, "_context_files_part", lambda *args: [])
    monkeypatch.setattr(prompt, "_coding_parts", lambda agent: ([], [], []))
    monkeypatch.setattr(prompt, "_pb", SimpleNamespace(build_environment_hints=lambda: "",
                                                       load_soul_md=lambda *args, **kwargs: "selected profile persona"))
    monkeypatch.setattr(prompt, "_skills_prompt", lambda agent: "")
    monkeypatch.setattr(prompt, "_guidance_parts", lambda agent: ["completion-guidance", "tool-guidance"])
    monkeypatch.setattr(prompt, "_alibaba_identity_part", lambda agent: [])
    monkeypatch.setattr(prompt, "_auto_load_parts", lambda agent: [])
    monkeypatch.setattr(prompt, "_memory_parts", lambda agent: [])
    monkeypatch.setattr(prompt, "_active_profile_line", lambda agent: "profile-line")
    monkeypatch.setattr(prompt, "_timestamp_line", lambda agent: "timestamp")

    appended = "\n append verbatim \n"
    append_policy = DelegationLaunchPolicy(system_prompt=appended, system_prompt_mode="append",
                                          identity_snapshot="selected profile persona",
                                          identity_soul_loaded=True)
    append_prompt = build_system_prompt_parts(_agent(root, append_policy))["stable"]
    assert "selected profile persona" in append_prompt
    assert append_prompt.count(appended) == 1
    assert "append verbatim" in append_prompt
    assert "completion-guidance" in append_prompt and "tool-guidance" in append_prompt

    replaced = "  replacement\n\n"
    replace_policy = DelegationLaunchPolicy(system_prompt=replaced, system_prompt_mode="replace")
    replace_prompt = build_system_prompt_parts(_agent(root, replace_policy))["stable"]
    assert replaced in replace_prompt
    assert "selected profile persona" not in replace_prompt
    assert "completion-guidance" in replace_prompt and "tool-guidance" in replace_prompt
    default_prompt = build_system_prompt_parts(_agent(root))["stable"]
    assert default_prompt.startswith("selected profile persona\n\n")


def test_actual_delegated_child_sends_exact_persona_and_workspace_on_loopback(tmp_path, monkeypatch):
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"data": [{"id": "persona-model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            body = ("data: " + json.dumps({"id": "wire", "object": "chat.completion.chunk", "created": 1,
                                           "model": payload.get("model"), "choices": [{"index": 0,
                                           "delta": {"role": "assistant", "content": "ok"},
                                           "finish_reason": None}]}) + "\n\n" +
                    "data: " + json.dumps({"id": "wire", "object": "chat.completion.chunk", "created": 1,
                                           "model": payload.get("model"), "choices": [{"index": 0,
                                           "delta": {}, "finish_reason": "stop"}]}) + "\n\n" +
                    "data: [DONE]\n\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            del fmt, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    agents, db = [], None
    try:
        from hermes_state import SessionDB
        from run_agent import AIAgent
        home = tmp_path / "hermes"
        home.mkdir()
        (home / "SOUL.md").write_text("selected profile persona", encoding="utf-8")
        (home / "config.yaml").write_text(yaml.safe_dump({
            "model": {"provider": "custom", "default": "persona-model",
                     "base_url": f"http://127.0.0.1:{server.server_port}/v1"},
            "delegation": {},
        }), encoding="utf-8")
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "AGENTS.md").write_text("workspace-marker-9037", encoding="utf-8")
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("TERMINAL_CWD", str(workspace))
        monkeypatch.setattr("tools.delegate_tool_config._load_config", dict)
        import tools.delegate_tool_config as delegation_config
        monkeypatch.setattr(delegation_config, "_load_config", dict)
        db = SessionDB(db_path=get_hermes_home() / "state.db")
        endpoint = f"http://127.0.0.1:{server.server_port}/v1"
        parent = AIAgent(api_key="parent-secret", base_url=endpoint, provider="custom",
                         api_mode="chat_completions", model="persona-model", platform="cli",
                         quiet_mode=True, skip_context_files=True, skip_memory=True,
                         save_trajectories=False, enabled_toolsets=["delegation"],
                         session_id="parent-persona", session_db=db, cwd=str(workspace))
        agents.append(parent)
        parent._ensure_db_session()

        from tools.delegate_tool import _build_child_agent
        from tools.delegate_tool_options import TaskRouteOptions
        from agent.delegation_launch import DelegationLaunchPolicy
        supplied = " \nexact raw persona\n\nwith code `x`  \n"
        option, error = TaskRouteOptions.parse({"goal": "first user goal", "system_prompt": supplied,
                                                "system_prompt_mode": "append"}, 0)
        assert error is None and option is not None
        profile_home = home / "profiles" / "coder"
        profile_home.mkdir(parents=True)
        (profile_home / "SOUL.md").write_text("selected profile persona", encoding="utf-8")
        policy = DelegationLaunchPolicy.from_task({"system_prompt": supplied, "system_prompt_mode": "append"}, option)
        policy = __import__("dataclasses").replace(policy, profile_home=str(profile_home))
        child = _build_child_agent(
            task_index=0, goal="first user goal", context="separate context marker", toolsets=None,
            model="persona-model", max_iterations=1, task_count=1, parent_agent=parent,
            override_provider="custom", override_base_url=endpoint, override_api_key="child-secret",
            override_api_mode="chat_completions", routing_cfg={}, profile_home=profile_home, launch_policy=policy,
        )
        agents.append(child)
        first_policy = child.__dict__["_delegation_launch_policy"]
        assert first_policy.system_prompt == supplied
        assert first_policy.system_prompt_mode == "append"
        assert first_policy.identity_snapshot == "selected profile persona"
        result = child.run_conversation(user_message="first user goal", task_id="persona-wire")
        assert result["final_response"] == "ok"
        assert child.__dict__["_delegation_launch_policy"] == first_policy
        request = requests[-1]
        system_prompt = request["messages"][0]["content"]
        assert supplied in system_prompt
        assert "Additional task-specific persona instructions:" in system_prompt
        assert system_prompt.count("selected profile persona") == 1
        assert "Use this exact path for local repository/workdir operations" in system_prompt
        assert "Complete this task using the tools available to you" in system_prompt
        assert "Important workspace rule" in system_prompt
        assert "workspace-marker-9037" in system_prompt
        assert system_prompt.count("first user goal") == 0
        assert system_prompt.count("exact raw persona") == 1
        assert request["messages"][1] == {"role": "user", "content": "first user goal"}
        assert workspace.as_posix() in child.__dict__["_cached_system_prompt"]
        assert "workspace-marker-9037" not in child.__dict__["_cached_system_prompt"]
        assert "first user goal" not in child.__dict__["_cached_system_prompt"]

        from agent.system_prompt import invalidate_system_prompt
        (home / "SOUL.md").write_text("edited profile persona", encoding="utf-8")
        invalidate_system_prompt(child)
        rebuilt = child._build_system_prompt()
        child._cached_system_prompt = rebuilt
        child.run_conversation(user_message="follow-up", task_id="persona-wire")
        assert supplied in requests[-1]["messages"][0]["content"]
        assert requests[-1]["messages"][0]["content"].count("selected profile persona") == 1
        assert "edited profile persona" not in requests[-1]["messages"][0]["content"]

        route_profile = profile_home
        (route_profile / "SOUL.md").write_text("selected profile persona", encoding="utf-8")
        skill_dir = route_profile / "skills" / "research" / "profile-preload"
        (skill_dir / "references").mkdir(parents=True)
        skill_body = "---\nname: profile-preload\ndescription: profile skill\n---\nFULL PROFILE SKILL BODY 5901\n"
        (skill_dir / "SKILL.md").write_text(skill_body, encoding="utf-8")
        (skill_dir / "references" / "notes.md").write_text("SUPPORT FILE BODY MUST NOT PRELOAD", encoding="utf-8")
        parent_skill_dir = home / "skills" / "research" / "profile-preload"
        parent_skill_dir.mkdir(parents=True)
        (parent_skill_dir / "SKILL.md").write_text(
            "---\nname: profile-preload\ndescription: parent copy\n---\nPARENT SKILL MUST NOT LEAK\n", encoding="utf-8")
        (route_profile / "config.yaml").write_text(yaml.safe_dump({
            "model": {"provider": "custom", "default": "persona-model",
                     "base_url": f"http://127.0.0.1:{server.server_port}/v1"},
        }), encoding="utf-8")
        public_supplied = " \nappend persona verbatim\nwith tabs\t \n"
        replace_supplied = " \nreplace raw persona\n\nwith code `y`  \n"
        route_tasks = [{"goal": "append user goal", "profile": "coder", "context": "separate context marker",
                        "system_prompt": public_supplied, "system_prompt_mode": "append"},
                       {"goal": "missing skill sibling", "profile": "coder", "skills": ["absent-skill"]},
                       {"goal": "replace user goal", "system_prompt": replace_supplied, "context": "replace context"},
                       {"goal": "default user goal", "context": "default context"},
                       {"goal": "profile skill preload user goal", "profile": "coder",
                        "skills": ["profile-preload"], "context": "skill preload context"}]
        from tools.delegate_tool import delegate_task
        parent_prompt_bytes = parent._build_system_prompt().encode("utf-8")
        parent_tools_bytes = json.dumps(parent.tools, sort_keys=True, separators=(",", ":")).encode("utf-8")
        route_output = json.loads(delegate_task(tasks=route_tasks, parent_agent=parent))
        assert parent._build_system_prompt().encode("utf-8") == parent_prompt_bytes
        assert json.dumps(parent.tools, sort_keys=True, separators=(",", ":")).encode("utf-8") == parent_tools_bytes
        assert [entry["status"] for entry in route_output["results"]] == [
            "completed", "error", "completed", "completed", "completed"]
        assert route_output["results"][1]["api_calls"] == 0
        assert "absent-skill" in route_output["results"][1]["error"]
        from tools.delegate_tool_progress import child_task_message
        route_requests = {
            task["goal"]: request
            for task in route_tasks for request in requests
            if len(request.get("messages", [])) > 1
            and request["messages"][1].get("role") == "user"
            and request["messages"][1].get("content") == child_task_message(task["goal"], task.get("context"))
        }
        expected_inference_goals = {"append user goal", "replace user goal", "default user goal",
                                    "profile skill preload user goal"}
        assert set(route_requests) == expected_inference_goals
        skill_request = route_requests["profile skill preload user goal"]
        skill_system = skill_request["messages"][0]["content"]
        assert "FULL PROFILE SKILL BODY 5901" in skill_system
        assert "references/notes.md" in skill_system
        assert "SUPPORT FILE BODY MUST NOT PRELOAD" not in skill_system
        assert "FULL PROFILE SKILL BODY 5901" not in skill_request["messages"][1]["content"]
        assert "skill_view" not in json.dumps(skill_request.get("tools", []))
        for task in route_tasks:
            if task["goal"] == "missing skill sibling":
                continue
            wire_system = route_requests[task["goal"]]["messages"][0]["content"]
            assert "Complete this task using the tools available to you" in wire_system
            assert "Use this exact path for local repository/workdir operations" in wire_system
            assert "Important workspace rule" in wire_system
            assert workspace.as_posix() in wire_system
            assert "workspace-marker-9037" in wire_system
            assert task["context"] not in wire_system
            assert task["context"] in route_requests[task["goal"]]["messages"][1]["content"]
            assert "first user goal" not in wire_system and task["goal"] not in wire_system
            expected_personas = {
                "append user goal": lambda: (
                    "selected profile persona" in wire_system
                    and "Additional task-specific persona instructions:" in wire_system
                    and wire_system.index("selected profile persona") < wire_system.index(
                        "Additional task-specific persona instructions:")
                    and public_supplied in wire_system
                    and wire_system.count("append persona verbatim") == 1),
                "replace user goal": lambda: (
                    replace_supplied in wire_system and wire_system.count("replace raw persona") == 1
                    and "Task-specific persona instructions (replacing identity):" in wire_system
                    and "selected profile persona" not in wire_system),
                "default user goal": lambda: "You are Hermes Agent" in wire_system and task["context"] not in wire_system,
                "profile skill preload user goal": lambda: (
                    "FULL PROFILE SKILL BODY 5901" in wire_system and "PARENT SKILL MUST NOT LEAK" not in wire_system),
            }
            assert expected_personas[task["goal"]]()
    finally:
        for agent in reversed(agents):
            agent.close()
        if db is not None:
            db.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_non_activating_skill_reader_returns_full_raw_body_without_setup(tmp_path, monkeypatch):
    import json
    from tools import skills_tool

    root = tmp_path / "hermes-test-home"
    skill = root / "skills" / "research" / "raw-preload"
    (skill / "references").mkdir(parents=True)
    (skill / "references" / "guide.md").write_text("reference metadata only", encoding="utf-8")
    body = "\nfull body starts\n!`touch SHOULD_NOT_RUN`\n${HERMES_SKILL_DIR}\nfull body ends\n"
    raw = ("---\nname: raw-preload\ndescription: raw\ndeps: [must-not-install]\n"
           "required_environment_variables: [NEVER_PROMPT_ME]\n---\n" + body)
    (skill / "SKILL.md").write_text(raw, encoding="utf-8")
    root.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    (root / "config.yaml").write_text("skills:\n  disabled: []\n", encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("non-activating read crossed an activation boundary")

    monkeypatch.setattr(skills_tool, "_skill_readiness", forbidden)
    import pm
    monkeypatch.setattr(pm, "ensure", forbidden)
    from agent.delegation_launch import preload_skill
    result = preload_skill("raw-preload", None, home_override=root)
    assert result["success"] is True
    assert result["content"] == raw
    assert result["readiness_status"] == "setup_needed"
    assert result["linked_files"] == {"references": ["references/guide.md"]}
    assert not (tmp_path / "SHOULD_NOT_RUN").exists()
    from agent.delegation_launch import (
        DelegationLaunchPolicy, build_preloaded_prompt_from_payloads, freeze_preloaded_skills,
    )
    from tools.delegate_tool_options import TaskRouteOptions
    task = {"goal": "frozen-body", "skills": ["raw-preload"]}
    option, error = TaskRouteOptions.parse(task, 0)
    assert error is None and option is not None
    policy = __import__("dataclasses").replace(
        DelegationLaunchPolicy.from_task(task, option), profile_home=str(root), workspace=str(tmp_path))
    child = SimpleNamespace(session_id="preload-test", _session_db=SimpleNamespace(db_path=root / "state.db"))
    freeze_preloaded_skills(child, policy)
    snapshot = child.__dict__["_delegation_preloaded_skills"]
    assert len(snapshot) == 1
    import pytest
    with pytest.raises(TypeError):
        snapshot[0][1]["content"] = "mutation"
    with pytest.raises(TypeError):
        snapshot[0][1]["linked_files"]["references"] = ()
    with pytest.raises(TypeError):
        snapshot[0][1]["required_environment_variables"][0]["name"] = "changed"
    frozen_prompt = child.__dict__["_delegation_preloaded_prompt"]
    assert raw in frozen_prompt
    from agent.delegation_launch import skill_preflight_fingerprint
    fingerprint = skill_preflight_fingerprint("raw-preload", result)
    (skill / "SKILL.md").write_text(raw.replace("full body starts", "edited body"), encoding="utf-8")
    rebuilt, names, missing = build_preloaded_prompt_from_payloads(policy.skills, snapshot)
    assert names == ("raw-preload",) and not missing
    assert rebuilt == frozen_prompt
    assert "edited body" not in rebuilt

    checked_policy = __import__("dataclasses").replace(
        policy, skill_preflight_snapshot=(("raw-preload", fingerprint),))
    changed_child = SimpleNamespace(session_id="preload-changed",
                                    _session_db=SimpleNamespace(db_path=root / "state.db"))
    import pytest
    with pytest.raises(ValueError, match="changed after preflight"):
        freeze_preloaded_skills(changed_child, checked_policy)
    assert not hasattr(changed_child, "_delegation_preloaded_prompt")


def test_preload_preserves_canonical_errors_and_valid_siblings(tmp_path, monkeypatch):
    from agent.delegation_launch import preload_skill
    from tools.delegate_tool_options import TaskRouteOptions
    from tools.delegate_tool_validation import _preflight_selected_skills
    root = tmp_path / "profile"
    skills = root / "skills"
    first = skills / "research" / "alpha"
    second = skills / "research" / "beta"
    sibling = skills / "research" / "gamma"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    sibling.mkdir(parents=True)
    (first / "SKILL.md").write_text("---\nname: alpha\ndescription: alpha\n---\nAlpha body", encoding="utf-8")
    (second / "SKILL.md").write_text("---\nname: beta\ndescription: beta\n---\nBeta body", encoding="utf-8")
    (sibling / "SKILL.md").write_text("---\nname: gamma\ndescription: gamma\n---\nGamma body", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    (root / "config.yaml").write_text("skills:\n  disabled: []\n", encoding="utf-8")

    absent = preload_skill("exact-absent-id", None, home_override=root)
    assert absent["success"] is False
    assert absent["identifier"] == "exact-absent-id"
    assert "Skill 'exact-absent-id' not found." in absent["error"]

    option, error = TaskRouteOptions.parse(
        {"goal": "task", "skills": ["exact-absent-id", "gamma"]}, 0)
    assert error is None and option is not None
    refusal, missing, snapshot = _preflight_selected_skills(option, root)
    assert refusal and missing == ("exact-absent-id",) and not snapshot
    assert "Skill 'exact-absent-id' not found." in refusal
    assert "gamma" not in refusal

    valid_option, error = TaskRouteOptions.parse({"goal": "task", "skills": ["gamma"]}, 0)
    assert error is None and valid_option is not None
    refusal, missing, snapshot = _preflight_selected_skills(valid_option, root)
    assert refusal is None and not missing
    assert len(snapshot) == 1 and snapshot[0][0] == "gamma"

    (root / "config.yaml").write_text("skills:\n  disabled: [beta]\n", encoding="utf-8")
    option, error = TaskRouteOptions.parse({"goal": "task", "skills": ["beta"]}, 0)
    assert error is None and option is not None
    refusal, missing, snapshot = _preflight_selected_skills(option, root)
    assert refusal and missing == ("beta",) and "disabled" in refusal and not snapshot
    (root / "config.yaml").write_text("skills:\n  disabled: []\n", encoding="utf-8")

    # Ambiguous and unreadable canonical failures retain their original reason; a valid sibling
    # still does not get listed among failures.
    (skills / "other" / "alpha").mkdir(parents=True)
    (skills / "other" / "alpha" / "SKILL.md").write_text(
        "---\nname: alpha\n---\nOther alpha", encoding="utf-8")
    from tools import skills_tool
    read_skill = skills_tool._read_skill_text
    def fail_beta(path):
        if path == second / "SKILL.md":
            raise PermissionError("token=TOP_SECRET_FILE_TOKEN_937")
        return read_skill(path)
    monkeypatch.setattr(skills_tool, "_read_skill_text", fail_beta)
    option, error = TaskRouteOptions.parse({"goal": "task", "skills": ["alpha", "beta", "gamma"]}, 0)
    assert error is None and option is not None
    refusal, missing, snapshot = _preflight_selected_skills(option, root)
    assert refusal and missing == ("alpha", "beta") and not snapshot
    assert "refusing to guess" in refusal
    assert "Failed to read skill 'beta' (PermissionError)." in refusal
    assert "TOP_SECRET_FILE_TOKEN_937" not in refusal
    assert "beta: Failed to read" in refusal


def test_registered_plugin_skill_reads_without_discovery_or_activation(tmp_path, monkeypatch):
    import json
    from hermes_cli import plugins
    from tools import skills_tool

    root = tmp_path / "profile"
    plugin_skill = root / "plugins" / "already-loaded" / "skills" / "guide"
    (plugin_skill / "references").mkdir(parents=True)
    body = "---\nname: guide\ndescription: registered plugin skill\n---\nPLUGIN BODY `literal`"
    skill_md = plugin_skill / "SKILL.md"
    skill_md.write_text(body, encoding="utf-8")
    (plugin_skill / "references" / "ref.md").write_text("reference body", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    (root / "config.yaml").write_text("plugins:\n  enabled: [sample-plugin]\n  disabled: []\nskills:\n  disabled: []\n", encoding="utf-8")

    manager = plugins.PluginManager(scope_key=root)
    monkeypatch.setitem(plugins._plugin_managers_by_home, root.resolve(), manager)
    manager._plugin_skills["sample-plugin:guide"] = {
        "path": skill_md, "plugin": "sample-plugin", "plugin_key": "sample-plugin",
        "plugin_name": "sample-plugin", "portable": False, "bare_name": "guide",
        "description": "registered plugin skill", "frontmatter": {"name": "guide"},
    }
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    def forbidden(*args, **kwargs):
        raise AssertionError("preload attempted plugin discovery/activation")
    monkeypatch.setattr(plugins, "discover_plugins", forbidden)
    monkeypatch.setattr(manager, "discover_and_load", forbidden)
    monkeypatch.setattr(skills_tool, "_serve_plugin_skill", forbidden)

    from agent.delegation_launch import preload_skill
    payload = preload_skill("sample-plugin:guide", None, home_override=root)
    assert payload["success"] is True, payload
    assert payload["name"] == "sample-plugin:guide"
    assert payload["content"] == body
    assert payload["linked_files"] == {"references": ["references/ref.md"]}

    # A qualified namespace already backed by registered metadata cannot alias a local category.
    local = root / "skills" / "sample-plugin" / "guide"
    local.mkdir(parents=True)
    (local / "SKILL.md").write_text("---\nname: guide\n---\nLOCAL ALIAS MUST NOT WIN", encoding="utf-8")
    payload = preload_skill("sample-plugin:missing", None, home_override=root)
    assert payload["success"] is False
    assert "registered plugin 'sample-plugin'" in payload["error"]
    assert "LOCAL ALIAS MUST NOT WIN" not in json.dumps(payload)


def test_registered_plugin_disabled_and_unreadable_reads_fail_closed(tmp_path, monkeypatch):
    from hermes_cli import plugins
    from tools import skills_tool
    import pytest

    root = tmp_path / "profile"
    plugin_skill = root / "plugin-skill"
    plugin_skill.mkdir(parents=True)
    skill_md = plugin_skill / "SKILL.md"
    skill_md.write_text("---\nname: guide\n---\nbody", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    (root / "config.yaml").write_text("plugins:\n  enabled: [sample-plugin]\n  disabled: []\nskills:\n  disabled: [sample-plugin:guide]\n", encoding="utf-8")
    manager = plugins.PluginManager(scope_key=root)
    monkeypatch.setitem(plugins._plugin_managers_by_home, root.resolve(), manager)
    manager._plugin_skills["sample-plugin:guide"] = {
        "path": skill_md, "plugin": "sample-plugin", "plugin_key": "sample-plugin",
        "plugin_name": "sample-plugin", "portable": False, "bare_name": "guide",
        "description": "", "frontmatter": {"name": "guide"},
    }
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    monkeypatch.setattr(plugins, "discover_plugins", lambda *a, **k: pytest.fail("discovery"))
    from agent.delegation_launch import preload_skill
    disabled = preload_skill("sample-plugin:guide", None, home_override=root)
    assert disabled["success"] is False and "disabled" in disabled["error"]

    (root / "config.yaml").write_text("plugins:\n  enabled: [sample-plugin]\n  disabled: []\nskills:\n  disabled: []\n", encoding="utf-8")
    def unreadable(path):
        if path == skill_md:
            raise PermissionError("fixture secret token=abc123456789012345")
        return Path.read_text(path, encoding="utf-8-sig", errors="replace")
    monkeypatch.setattr(skills_tool, "_read_skill_text", unreadable)
    failed = preload_skill("sample-plugin:guide", None, home_override=root)
    assert failed["success"] is False
    assert "Failed to read skill 'sample-plugin:guide'" in failed["error"]
    assert "abc123456789012345" not in failed["error"]


def test_preloaded_skill_init_failure_stops_before_first_model_call(tmp_path, monkeypatch):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from agent.delegation_launch import DelegationLaunchPolicy
    from tools.delegate_tool_options import TaskRouteOptions
    import pytest

    requests = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"data": [{"id": "preload-model"}]}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        def do_POST(self):
            requests.append(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_error(500)
        def log_message(self, fmt, *args):
            del fmt, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    db = None; agent = None
    try:
        home = tmp_path / "profile"; home.mkdir()
        monkeypatch.setenv("HOME", str(tmp_path)); monkeypatch.setenv("HERMES_HOME", str(home))
        (home / "config.yaml").write_text(yaml.safe_dump({"model": {
            "provider": "custom", "default": "preload-model",
            "base_url": f"http://127.0.0.1:{server.server_port}/v1"}}), encoding="utf-8")
        (home / "skills" / "research" / "needed").mkdir(parents=True)
        (home / "skills" / "research" / "needed" / "SKILL.md").write_text(
            "---\nname: needed\n---\nrequired-body", encoding="utf-8")
        db = SessionDB(db_path=get_hermes_home() / "state.db")
        option, error = TaskRouteOptions.parse({"goal": "must-not-infer", "skills": ["needed"]}, 0)
        assert error is None and option is not None
        policy = __import__("dataclasses").replace(
            DelegationLaunchPolicy.from_task({"goal": "must-not-infer", "skills": ["needed"]}, option),
            profile_home=str(home), workspace=str(tmp_path))
        from tools import skills_tool
        def fail_read(*args, **kwargs):
            raise PermissionError("canonical preload disk read failed")
        monkeypatch.setattr(skills_tool, "_read_skill_text", fail_read)
        with pytest.raises(ValueError, match="needed.*PermissionError"):
            agent = AIAgent(api_key="local-test", base_url=f"http://127.0.0.1:{server.server_port}/v1",
                            provider="custom", api_mode="chat_completions", model="preload-model",
                            platform="subagent", quiet_mode=True, skip_context_files=True, skip_memory=True,
                            save_trajectories=False, session_id="preload-init", session_db=db,
                            delegation_launch_policy=policy)
        assert requests == []
    finally:
        if agent is not None: agent.close()
        if db is not None: db.close()
        server.shutdown(); server.server_close(); thread.join(timeout=5)


def test_explicit_empty_skill_preload_does_not_require_skill_tool(tmp_path, monkeypatch):
    import json
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from agent.delegation_launch import DelegationLaunchPolicy
    from tools.delegate_tool_options import TaskRouteOptions

    requests = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"data": [{"id": "empty-skills-model"}]}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = ("data: " + json.dumps({"id": "empty-skills", "object": "chat.completion.chunk",
                    "created": 1, "model": "empty-skills-model", "choices": [{"index": 0,
                    "delta": {"role": "assistant", "content": "ok"}, "finish_reason": None}]}) + "\n\n" +
                    "data: " + json.dumps({"id": "empty-skills", "object": "chat.completion.chunk",
                    "created": 1, "model": "empty-skills-model", "choices": [{"index": 0,
                    "delta": {}, "finish_reason": "stop"}]}) + "\n\n" + "data: [DONE]\n\n").encode()
            self.send_response(200); self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        def log_message(self, fmt, *args): del fmt, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    db = None; agent = None
    try:
        home = tmp_path / "profile"; home.mkdir()
        monkeypatch.setenv("HOME", str(tmp_path)); monkeypatch.setenv("HERMES_HOME", str(home))
        endpoint = f"http://127.0.0.1:{server.server_port}/v1"
        (home / "config.yaml").write_text(yaml.safe_dump({"model": {"provider": "custom",
            "default": "empty-skills-model", "base_url": endpoint}}), encoding="utf-8")
        db = SessionDB(db_path=get_hermes_home() / "state.db")
        task = {"goal": "empty-skills-goal", "skills": []}
        option, error = TaskRouteOptions.parse(task, 0)
        assert error is None and option is not None
        policy = __import__("dataclasses").replace(
            DelegationLaunchPolicy.from_task(task, option), profile_home=str(home), workspace=str(tmp_path))
        agent = AIAgent(api_key="local-test", base_url=endpoint, provider="custom",
                        api_mode="chat_completions", model="empty-skills-model", platform="subagent",
                        quiet_mode=True, skip_context_files=True, skip_memory=True, save_trajectories=False,
                        enabled_toolsets=[], session_id="empty-skills", session_db=db,
                        delegation_launch_policy=policy)
        from agent.system_prompt import invalidate_system_prompt
        invalidate_system_prompt(agent)
        agent._build_system_prompt()
        result = agent.run_conversation(user_message=task["goal"], task_id="empty-skills")
        assert result["final_response"] == "ok"
        assert requests and requests[-1]["messages"][1]["content"] == task["goal"]
        assert agent.__dict__["_delegation_preloaded_skills"] == ()
        assert agent.__dict__["_delegation_preloaded_missing"] == ()
    finally:
        if agent is not None: agent.close()
        if db is not None: db.close()
        server.shutdown(); server.server_close(); thread.join(timeout=5)


def test_actual_child_freezes_preloaded_body_across_prompt_rebuild_without_skill_tools(tmp_path, monkeypatch):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from agent.delegation_launch import DelegationLaunchPolicy
    from tools.delegate_tool_options import TaskRouteOptions

    requests = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"data": [{"id": "frozen-skills-model"}]}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            body = ("data: " + json.dumps({"id": "frozen-skills", "object": "chat.completion.chunk",
                    "created": 1, "model": "frozen-skills-model", "choices": [{"index": 0,
                    "delta": {"role": "assistant", "content": "ok"}, "finish_reason": None}]}) + "\n\n" +
                    "data: " + json.dumps({"id": "frozen-skills", "object": "chat.completion.chunk",
                    "created": 1, "model": "frozen-skills-model", "choices": [{"index": 0,
                    "delta": {}, "finish_reason": "stop"}]}) + "\n\n" + "data: [DONE]\n\n").encode()
            self.send_response(200); self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        def log_message(self, fmt, *args): del fmt, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    db = None; agent = None
    try:
        home = tmp_path / "profile"; home.mkdir()
        monkeypatch.setenv("HOME", str(tmp_path)); monkeypatch.setenv("HERMES_HOME", str(home))
        endpoint = f"http://127.0.0.1:{server.server_port}/v1"
        (home / "config.yaml").write_text(yaml.safe_dump({"model": {"provider": "custom",
            "default": "frozen-skills-model", "base_url": endpoint}}), encoding="utf-8")
        skill_dir = home / "skills" / "research" / "frozen-guide"
        (skill_dir / "references").mkdir(parents=True)
        raw_body = "---\nname: frozen-guide\ndescription: frozen\n---\nEXACT ORIGINAL FROZEN BODY\n!`touch NO_SIDE_EFFECT`\n"
        skill_file = skill_dir / "SKILL.md"; skill_file.write_text(raw_body, encoding="utf-8")
        (skill_dir / "references" / "quick.md").write_text("reference bytes stay metadata", encoding="utf-8")
        db = SessionDB(db_path=get_hermes_home() / "state.db")
        task = {"goal": "frozen-skill-goal", "skills": ["frozen-guide"]}
        option, error = TaskRouteOptions.parse(task, 0)
        assert error is None and option is not None
        policy = __import__("dataclasses").replace(
            DelegationLaunchPolicy.from_task(task, option), profile_home=str(home), workspace=str(tmp_path))
        agent = AIAgent(api_key="local-test", base_url=endpoint, provider="custom",
                        api_mode="chat_completions", model="frozen-skills-model", platform="subagent",
                        quiet_mode=True, skip_context_files=True, skip_memory=True, save_trajectories=False,
                        enabled_toolsets=[], session_id="frozen-skills", session_db=db,
                        delegation_launch_policy=policy)
        assert "EXACT ORIGINAL FROZEN BODY" in agent._build_system_prompt()
        assert "references/quick.md" in agent._build_system_prompt()
        assert "reference bytes stay metadata" not in agent._build_system_prompt()
        assert "skill_view" not in agent.valid_tool_names
        agent.run_conversation(user_message=task["goal"], task_id="frozen-skills")
        assert "EXACT ORIGINAL FROZEN BODY" in requests[-1]["messages"][0]["content"]

        skill_file.write_text(raw_body.replace("EXACT ORIGINAL FROZEN BODY", "EDITED SOURCE BODY"), encoding="utf-8")
        from agent.system_prompt import invalidate_system_prompt
        invalidate_system_prompt(agent)
        rebuilt = agent._build_system_prompt()
        assert "EXACT ORIGINAL FROZEN BODY" in rebuilt
        assert "EDITED SOURCE BODY" not in rebuilt
        agent.run_conversation(user_message="follow-up", task_id="frozen-skills")
        assert "EXACT ORIGINAL FROZEN BODY" in requests[-1]["messages"][0]["content"]
        assert "EDITED SOURCE BODY" not in requests[-1]["messages"][0]["content"]
        conversation_requests = [request for request in requests if "messages" in request]
        assert len(conversation_requests) == 2
    finally:
        if agent is not None: agent.close()
        if db is not None: db.close()
        server.shutdown(); server.server_close(); thread.join(timeout=5)
