"""Delegated launch policy is immutable and keeps explicit choices distinct."""

from dataclasses import FrozenInstanceError

import pytest


def test_launch_policy_captures_task_overrides_without_mutable_aliases():
    from agent.delegation_launch import DelegationLaunchPolicy

    tools = ["read_file", "read_file"]
    policy = DelegationLaunchPolicy.from_task({"system_prompt": "persona", "model": "exact-id",
                                               "reasoning_effort": "high", "tools": tools,
                                               "skills": ["search"]})
    tools.append("terminal")
    assert policy.model == "exact-id"
    assert policy.reasoning_effort == "high"
    assert policy.system_prompt == "persona" and policy.system_prompt_mode == "replace"
    assert policy.tools == ("read_file",)
    assert policy.skills == ("search",)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        policy.model = "other"


def test_task_options_distinguish_omitted_empty_and_deduplicated_lists():
    from tools.delegate_tool_options import TaskRouteOptions

    omitted, error = TaskRouteOptions.parse({"goal": "x"}, 0)
    empty, empty_error = TaskRouteOptions.parse({"goal": "x", "tools": [], "skills": []}, 0)
    repeated, repeated_error = TaskRouteOptions.parse({"goal": "x", "tools": ["file_read", "file_read"],
                                                        "skills": ["search", "search"]}, 0)
    whitespace, whitespace_error = TaskRouteOptions.parse({"goal": "x", "tools": [" file_read"]}, 0)
    assert whitespace is None and whitespace_error and "surrounding whitespace" in whitespace_error
    assert error is empty_error is repeated_error is None
    assert omitted is not None and omitted.tools is None and omitted.skills is None and not omitted.any_set
    prompt_only, prompt_error = TaskRouteOptions.parse({"goal": "x", "system_prompt": "persona"}, 0)
    assert prompt_error is None and prompt_only is not None
    assert prompt_only.system_prompt == "persona" and prompt_only.system_prompt_mode == "replace"
    assert prompt_only.system_prompt_mode_set is False
    assert empty is not None and empty.tools == () and empty.skills == () and empty.any_set
    assert repeated is not None and repeated.tools == ("file_read",) and repeated.skills == ("search",)


@pytest.mark.parametrize("task", [
    {"goal": "x", "system_prompt_mode": "replace"},
    {"goal": "x", "system_prompt": "persona", "system_prompt_mode": "unsupported"},
    {"goal": "x", "tools": [""]},
    {"goal": "x", "tools": [" read_file"]},
    {"goal": "x", "skills": ["search "]},
    {"goal": "x", "skills": ["  "]},
    {"goal": "x", "tools": "file_read"},
])
def test_invalid_launch_inputs_are_refused_before_child_construction(task):
    from tools.delegate_tool_options import TaskRouteOptions

    policy, error = TaskRouteOptions.parse(task, 3)
    assert policy is None
    assert error and "Task 3" in error
