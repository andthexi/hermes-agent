"""Per-task route options, profile resolution and selected-route validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

TASK_OPTION_FIELDS = frozenset({"profile", "model", "provider", "reasoning_effort"})
LAUNCH_POLICY_FIELDS = frozenset({"system_prompt", "system_prompt_mode", "tools", "skills"})
_TASK_INPUT_FIELDS = TASK_OPTION_FIELDS | LAUNCH_POLICY_FIELDS


def _names(value: Any, field: str, index: int) -> tuple[str, ...] | None:
    if not isinstance(value, list) or any(
        not isinstance(name, str) or not name.strip() or name != name.strip() for name in value
    ):
        raise ValueError(f"Task {index}: {field!r} must be an array of non-empty strings without surrounding whitespace.")
    return tuple(dict.fromkeys(value))


@dataclass(frozen=True, slots=True)
class TaskRouteOptions:
    """Exact per-task route and launch inputs; set flags distinguish omission from invalid values."""

    profile: str | None = None
    model: str | None = None
    provider: str | None = None
    reasoning_effort: str | None = None
    system_prompt: str | None = None
    system_prompt_mode: str | None = None
    tools: tuple[str, ...] | None = None
    skills: tuple[str, ...] | None = None
    profile_set: bool = False
    model_set: bool = False
    provider_set: bool = False
    reasoning_effort_set: bool = False
    system_prompt_set: bool = False
    system_prompt_mode_set: bool = False
    tools_set: bool = False
    skills_set: bool = False

    @property
    def route_pinned(self) -> bool:
        return self.profile_set or self.model_set or self.provider_set

    @property
    def any_set(self) -> bool:
        return any((self.route_pinned, self.reasoning_effort_set, self.system_prompt_set,
                    self.system_prompt_mode_set, self.tools_set, self.skills_set))

    @classmethod
    def parse(cls, task: Mapping[str, Any], index: int) -> tuple[TaskRouteOptions | None, str | None]:
        values = {}
        explicit_fields = set()
        for name in _TASK_INPUT_FIELDS:
            if name not in task:
                continue
            value = task[name]
            explicit_fields.add(name)
            if name in {"tools", "skills"}:
                try:
                    values[name] = _names(value, name, index)
                except ValueError as exc:
                    return None, str(exc)
                continue
            if not isinstance(value, str) or not value.strip():
                return None, f"Task {index}: {name!r} must be a non-empty string."
            if name == "reasoning_effort":
                from hermes_constants import VALID_REASONING_EFFORTS
                if value != "none" and value not in VALID_REASONING_EFFORTS:
                    levels = ", ".join(VALID_REASONING_EFFORTS)
                    return None, f"Task {index}: unknown reasoning_effort {value!r}. Use none or one of: {levels}."
            if name == "system_prompt_mode" and value not in {"replace", "append"}:
                return None, f"Task {index}: 'system_prompt_mode' must be 'replace' or 'append'."
            values[name] = value
        if "system_prompt_mode" in task and "system_prompt" not in task:
            return None, f"Task {index}: 'system_prompt_mode' requires 'system_prompt'."
        if "system_prompt" in values and "system_prompt_mode" not in values:
            values["system_prompt_mode"] = "replace"
        return cls(
            profile=values.get("profile"), model=values.get("model"), provider=values.get("provider"),
            reasoning_effort=values.get("reasoning_effort"), system_prompt=values.get("system_prompt"),
            system_prompt_mode=values.get("system_prompt_mode"), tools=values.get("tools"), skills=values.get("skills"),
            profile_set="profile" in values, model_set="model" in values, provider_set="provider" in values,
            reasoning_effort_set="reasoning_effort" in values, system_prompt_set="system_prompt" in values,
            system_prompt_mode_set="system_prompt_mode" in explicit_fields, tools_set="tools" in explicit_fields,
            skills_set="skills" in explicit_fields,
        ), None


def launch_policy(task: Mapping[str, Any], route: TaskRouteOptions) -> Any:
    from agent.delegation_launch import DelegationLaunchPolicy
    return DelegationLaunchPolicy.from_task(task, route)


def _provider_credentials(
    provider: str, model: str, config: Mapping[str, Any], *,
    base_url: str | None = None, api_key: str | None = None,
) -> dict[str, Any]:
    """Resolve provider endpoint, API mode and credentials through the profile-scoped runtime owner."""
    from hermes_cli.runtime_provider import resolve_runtime_provider

    runtime = resolve_runtime_provider(
        requested=provider, target_model=model, explicit_base_url=base_url, explicit_api_key=api_key,
    )
    key = runtime.get("api_key")
    if not key:
        raise ValueError(f"Selected provider {provider!r} has no configured credential.")
    runtime_provider = str(runtime.get("provider") or provider)
    return {
        "model": model,
        "provider": provider if runtime_provider.lower() == "custom" else runtime_provider,
        "base_url": runtime.get("base_url"),
        "api_key": key,
        "api_mode": runtime.get("api_mode"),
        "request_overrides": dict(runtime.get("request_overrides") or {}),
        "command": runtime.get("command"),
        "args": list(runtime.get("args") or []),
    }


def _profile_default_model(config: Mapping[str, Any], provider: str) -> dict[str, Any] | None:
    """Resolve a saved profile default to a concrete model before provider replacement."""
    from hermes_cli.model_switch import resolve_startup_model_route
    from tools.delegate_tool_validation import model_alias_route

    model_block = config.get("model")
    saved = (model_block.get("default") or model_block.get("model")) if isinstance(model_block, Mapping) else model_block
    if not isinstance(saved, str) or not saved.strip():
        return None
    alias = model_alias_route(saved, config)
    if alias:
        return alias
    startup = resolve_startup_model_route(
        saved, current_provider=str(model_block.get("provider") or "") if isinstance(model_block, Mapping) else "",
        user_providers=config.get("providers"), custom_providers=config.get("custom_providers"), alias_config=dict(config),
    )
    return {"model": startup.model if startup else saved, "provider": provider}
