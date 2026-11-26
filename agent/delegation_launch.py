"""Immutable child launch policy shared by construction and runtime enforcement."""

from __future__ import annotations

from dataclasses import dataclass, field, replace as dataclass_replace
from pathlib import Path
from typing import Any, Mapping
import json
import re

from tools.delegate_tool_options import TaskRouteOptions


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class DelegationLaunchPolicy:
    """Launch-local options with frozen model-facing input containers."""

    route: TaskRouteOptions = field(default_factory=TaskRouteOptions)
    profile: str | None = None
    model: str | None = None
    provider: str | None = None
    reasoning_effort: str | None = None
    system_prompt: str | None = None
    system_prompt_mode: str | None = None
    tools: tuple[str, ...] | None = None
    skills: tuple[str, ...] | None = None
    strict_reasoning: bool = False
    diagnostics: tuple[tuple[str, Any], ...] = ()
    reason: str | None = None
    profile_home: str | None = None
    workspace: str | None = None
    parent_tool_grant: tuple[str, ...] | None = None
    admitted_tool_names: tuple[str, ...] | None = None
    identity_snapshot: str | None = None
    identity_soul_loaded: bool = False
    skill_preflight_snapshot: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.tools is not None:
            object.__setattr__(self, "tools", tuple(self.tools))
        if self.skills is not None:
            object.__setattr__(self, "skills", tuple(self.skills))
        if self.parent_tool_grant is not None:
            object.__setattr__(self, "parent_tool_grant", tuple(self.parent_tool_grant))
        if self.admitted_tool_names is None and self.tools is not None:
            object.__setattr__(self, "admitted_tool_names", tuple(self.tools))
        elif self.admitted_tool_names is not None:
            object.__setattr__(self, "admitted_tool_names", tuple(self.admitted_tool_names))
        object.__setattr__(self, "diagnostics", tuple((str(key), _freeze(value)) for key, value in self.diagnostics))
        object.__setattr__(self, "skill_preflight_snapshot", tuple(
            (str(identifier), str(fingerprint)) for identifier, fingerprint in self.skill_preflight_snapshot))

    @staticmethod
    def freeze_diagnostics(diagnostics: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
        return tuple((str(key), _freeze(value)) for key, value in sorted(diagnostics.items()))

    @classmethod
    def from_task(cls, task: Mapping[str, Any], route: TaskRouteOptions | None = None) -> DelegationLaunchPolicy:
        """Freeze parsed launch inputs for the child lifetime."""
        if route is None:
            route, error = TaskRouteOptions.parse(task, 0)
            if error:
                raise ValueError(error)
        selected_route = route or TaskRouteOptions()
        return cls(
            route=selected_route,
            profile=selected_route.profile, model=selected_route.model, provider=selected_route.provider,
            reasoning_effort=selected_route.reasoning_effort,
            system_prompt=selected_route.system_prompt, system_prompt_mode=selected_route.system_prompt_mode,
            tools=selected_route.tools, skills=selected_route.skills,
            strict_reasoning=selected_route.reasoning_effort_set,
        )


def with_diagnostics(policy: DelegationLaunchPolicy, diagnostics: Mapping[str, Any], reason: str | None = None) -> DelegationLaunchPolicy:
    """Return a policy whose diagnostics are recursively immutable."""
    frozen = tuple((str(key), _freeze(value)) for key, value in sorted(diagnostics.items()))
    return dataclass_replace(policy, diagnostics=frozen, reason=reason)


def effective_tool_ceiling(agent: Any) -> frozenset[str] | None:
    """Exact immutable allowed-name ceiling, or None for ordinary sessions."""
    policy = getattr(agent, "_delegation_launch_policy", None)
    if not isinstance(policy, DelegationLaunchPolicy):
        return None
    grant = policy.admitted_tool_names if policy.admitted_tool_names is not None else policy.tools
    return frozenset(grant) if grant is not None else None


def task_tool_ceiling(agent: Any) -> frozenset[str] | None:
    """Exact concrete-tool ceiling; explicit task [] stays empty while legacy None stays open."""
    return effective_tool_ceiling(agent)


def exact_permission_names(agent: Any) -> frozenset[str] | None:
    """Exact literal permissions, including bridge names only when explicitly granted."""
    ceiling = task_tool_ceiling(agent)
    return ceiling



def _repair_tool_name_ordinary(tool_name: str, valid_tool_names: set[str] | frozenset[str]) -> str | None:
    """Repair casing, separators, CamelCase, doubled ``_tool`` suffixes and fuzzy matches."""
    from difflib import get_close_matches
    if not tool_name:
        return None
    for _xml_sep in ('"', "'", "<", ">"):
        _idx = tool_name.find(_xml_sep)
        if _idx > 0:
            tool_name = tool_name[:_idx]
    if not tool_name:
        return None
    _norm = lambda s: s.lower().replace("-", "_").replace(" ", "_")
    _camel_snake = lambda s: re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()

    def _strip_tool_suffix(s: str) -> str | None:
        lc = s.lower()
        return next((s[: -len(sfx)].rstrip("_-") for sfx in ("_tool", "-tool", "tool") if lc.endswith(sfx)), None)
    # Cheap fast-paths first.
    lowered = tool_name.lower()
    if lowered in valid_tool_names:
        return lowered
    normalized = _norm(tool_name)
    if normalized in valid_tool_names:
        return normalized
    cands: set[str] = {tool_name, lowered, normalized, _camel_snake(tool_name)}
    for _ in range(2):  # strip trailing tool-suffix up to twice (TodoTool_tool needs it)
        extra: set[str] = set()
        for c in cands:
            stripped = _strip_tool_suffix(c)
            if stripped:
                extra.update((stripped, _norm(stripped), _camel_snake(stripped)))
        cands |= extra
    for c in cands:
        if c and c in valid_tool_names:
            return c
    matches = get_close_matches(lowered, valid_tool_names, n=1, cutoff=0.7)
    return matches[0] if matches else None


def repair_tool_name(agent: Any, tool_name: str) -> str | None:
    """Repair names only from the child grant."""
    if not tool_name:
        return None
    ceiling = task_tool_ceiling(agent)
    if ceiling is not None and tool_name not in ceiling:
        return None
    accepted = agent.valid_tool_names if ceiling is None else agent.valid_tool_names & ceiling
    return _repair_tool_name_ordinary(tool_name, accepted)


def provider_default_routes(provider: str) -> set[str]:
    """Return known exact default routes for a canonical provider id."""
    from contextlib import suppress
    from hermes_cli.route_identity import normalize_route_base_url

    routes: set[str] = set()

    def add(value):
        route = normalize_route_base_url(value)
        if route:
            routes.add(route)

    with suppress(Exception):
        from hermes_cli.providers import HERMES_OVERLAYS, get_provider
        overlay = HERMES_OVERLAYS.get(provider)
        provider_def = get_provider(provider, allow_network=False)
        add(getattr(overlay, "base_url_override", ""))
        add(getattr(provider_def, "base_url", ""))
    with suppress(Exception):
        from providers import get_provider_profile
        add(getattr(get_provider_profile(provider), "base_url", ""))
    with suppress(Exception):
        from hermes_cli.auth import PROVIDER_REGISTRY
        from hermes_cli.models import normalize_provider as normalize_model_provider
        from hermes_cli.providers import normalize_provider as normalize_registry_provider
        for provider_id, config in PROVIDER_REGISTRY.items():
            if normalize_registry_provider(normalize_model_provider(provider_id)) == provider:
                add(getattr(config, "inference_base_url", ""))
    if provider == "gemini":
        routes.update(f"{route.rstrip('/')}/openai" for route in list(routes))
    return routes

def tool_grant_refusal(agent: Any, tool_name: str) -> str | None:
    """Return one canonical launch-grant refusal across dispatch surfaces."""
    exact_names = exact_permission_names(agent)
    if exact_names is None or tool_name in exact_names:
        return None
    if not exact_names:
        return "Tool call refused: this child was launched with an empty tool grant; no Hermes tools are permitted."
    return f"Tool call refused: {tool_name!r} is outside this child's exact tool grant."


def tool_grant_result(agent: Any, tool_name: str, fallback: str | None = None) -> str | None:
    """Return canonical JSON denial for an exact grant or a caller-supplied block."""
    refusal = tool_grant_refusal(agent, tool_name) or fallback
    return json.dumps({"error": refusal}, ensure_ascii=False) if refusal is not None else None


def exact_tool_surface(agent: Any, raw_definitions: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """Build the child surface from direct literals plus only explicitly granted bridges."""
    ceiling = effective_tool_ceiling(agent)
    if ceiling is None:
        return None
    from tools.tool_search_catalog import BRIDGE_TOOL_NAMES
    direct = []
    for item in raw_definitions:
        name = item.get("function", {}).get("name")
        if name not in ceiling - BRIDGE_TOOL_NAMES:
            continue
        if name == "execute_code":
            from tools.code_execution_tool import SANDBOX_ALLOWED_TOOLS, build_execute_code_schema
            helpers = set(ceiling & SANDBOX_ALLOWED_TOOLS)
            item = {**item, "function": build_execute_code_schema(helpers)}
        direct.append(item)
    if not ceiling & BRIDGE_TOOL_NAMES:
        return direct
    from tools.tool_search import assemble_tool_defs, load_config_readonly
    config = load_config_readonly()
    if config.enabled == "off":
        return direct
    bridge_defs = assemble_tool_defs(raw_definitions, config=config).tool_defs
    names = {item.get("function", {}).get("name") for item in direct}
    direct.extend(item for item in bridge_defs
                  if item.get("function", {}).get("name") in ceiling & BRIDGE_TOOL_NAMES
                  and item.get("function", {}).get("name") not in names)
    return direct


def tool_name_in_grant(agent: Any, tool_name: str) -> bool:
    names = exact_permission_names(agent)
    return names is None or tool_name in names


def blocked_tool_result(agent: Any, tool_name: str, args: dict[str, Any], task_id: str,
                        call_id: str | None, trace: list[dict[str, Any]]) -> str:
    """Emit the terminal blocked event and return one denial result without dispatch."""
    refusal = tool_grant_refusal(agent, tool_name) or f"Tool call refused: {tool_name!r} is outside this child's exact tool grant."
    result = json.dumps({"error": refusal}, ensure_ascii=False)
    from agent.inline_tool_executors import emit_terminal_post_tool_call
    emit_terminal_post_tool_call(
        agent, function_name=tool_name, function_args=args, result=result,
        effective_task_id=task_id, tool_call_id=call_id, status="blocked",
        error_type="tool_grant_refused", error_message=refusal, middleware_trace=trace,
    )
    return result


def narrow_tool_surface(agent: Any, tool_defs: list | None = None) -> set[str] | None:
    """Narrow live or staged schemas to the exact launch ceiling without admitting names."""
    ceiling = effective_tool_ceiling(agent)
    if ceiling is None:
        return None
    target = ((getattr(agent, "tools", None) or []) if tool_defs is None else tool_defs)
    target[:] = [
        item for item in target
        if isinstance(item, Mapping) and isinstance(item.get("function"), Mapping)
        and item["function"].get("name") in ceiling
    ]
    names = {item["function"]["name"] for item in target}
    if tool_defs is None:
        agent.tools = target
        agent.valid_tool_names = names
        agent._exact_tool_name_ceiling = ceiling
    return names


def initialize_agent_tool_surface(agent: Any, enabled_toolsets: list[str] | None,
                                  disabled_toolsets: list[str] | None) -> None:
    """Publish the permission-aware startup surface for strict children."""
    import model_tools
    from toolsets import agent_tool_surface

    ceiling = effective_tool_ceiling(agent)
    if ceiling is None:
        agent.tools = agent_tool_surface(agent, model_tools.get_tool_definitions(
            enabled_toolsets=enabled_toolsets, disabled_toolsets=disabled_toolsets,
            quiet_mode=agent.quiet_mode) or [])
        return
    args = {"enabled_toolsets": enabled_toolsets, "disabled_toolsets": disabled_toolsets,
            "quiet_mode": True, "skip_tool_search_assembly": True,
            "update_last_resolved_tool_names": False}
    agent._tool_exact_startup_names = set(ceiling)
    raw = model_tools.get_tool_definitions(**args) or []
    surface = exact_tool_surface(agent, raw) or []
    if ceiling & {"tool_search", "tool_describe", "tool_call"}:
        from tools.tool_search import ToolSearchConfig, assemble_tool_defs
        raw_grant = [item for item in raw if item.get("function", {}).get("name") in ceiling
                     and item.get("function", {}).get("name") not in {"tool_search", "tool_describe", "tool_call"}]
        bridge_defs = assemble_tool_defs(raw_grant,
                                         config=ToolSearchConfig.from_raw({"enabled": "on"})).tool_defs
        names = {item.get("function", {}).get("name") for item in surface}
        surface.extend(item for item in bridge_defs
                       if item.get("function", {}).get("name") in ceiling
                       and item.get("function", {}).get("name") not in names)
    agent.tools = agent_tool_surface(agent, surface)
    agent._tool_exact_direct_names = set(ceiling) - {"tool_search", "tool_describe", "tool_call"}
    agent._tool_exact_live_names = set(getattr(agent, "_tool_exact_direct_names", ceiling)) & {
        item.get("function", {}).get("name") for item in raw
    }
    agent._tool_exact_live_names.update(ceiling & {"tool_search", "tool_describe", "tool_call"})


def enforce_tool_ceiling(agent: Any) -> None:
    """Reject unavailable exact grants, then narrow startup tools and accepted names."""
    ceiling = effective_tool_ceiling(agent)
    if ceiling is None:
        return
    tools = getattr(agent, "tools", None) or []
    present = {name for item in tools if (name := item.get("function", {}).get("name"))}
    available = getattr(agent, "_tool_exact_live_names", present)
    missing = ceiling - available
    if missing:
        raise ValueError(
            "Exact child tool grant unavailable in the final startup surface: "
            f"{', '.join(sorted(missing))}."
        )
    names = narrow_tool_surface(agent)
    context_names = getattr(agent, "_context_engine_tool_names", None)
    if isinstance(context_names, set) and names is not None:
        context_names.intersection_update(names)


def _canonical_skill_error(payload: Mapping[str, Any], identifier: str) -> str:
    error = payload.get("error")
    if isinstance(error, str) and error.strip():
        return _safe_skill_error(error)
    return _safe_skill_error(f"Skill '{identifier}' is unavailable.")


def _safe_skill_error(error: object) -> str:
    """Sanitize canonical reader errors before retaining them in task refusal diagnostics."""
    from agent.redact import redact_sensitive_text

    return redact_sensitive_text(str(error or "Skill read failed"), force=True, redact_url_credentials=True)


def preload_skill(identifier: str, task_id: str | None, *, home_override: Any = None) -> dict[str, Any]:
    """Return canonical skill payload or a sanitized, identifier-preserving refusal record."""
    import json
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from agent.skill_utils import normalize_skill_lookup_name
    from tools.skills_tool import skill_view

    home_token = set_hermes_home_override(str(home_override)) if home_override is not None else None
    try:
        try:
            raw = skill_view(normalize_skill_lookup_name(identifier), task_id=task_id,
                             preprocess=False, non_activating=True)
            payload = json.loads(raw)
        except (TypeError, ValueError, OSError, UnicodeError) as exc:
            return {"success": False, "identifier": identifier,
                    "error": _safe_skill_error(f"Skill '{identifier}' read failed ({type(exc).__name__}).")}
        if not isinstance(payload, dict):
            return {"success": False, "identifier": identifier,
                    "error": _safe_skill_error(f"Skill '{identifier}' returned invalid reader metadata.")}
    finally:
        if home_token is not None:
            reset_hermes_home_override(home_token)
    if isinstance(payload, dict) and payload.get("success") is True and isinstance(payload.get("content"), str):
        return payload
    error = payload.get("error") if isinstance(payload, dict) else None
    return {"success": False, "identifier": identifier,
            "error": _safe_skill_error(error or f"Skill '{identifier}' returned invalid content.")}




def _freeze_snapshot(value: Any) -> Any:
    """Recursively make retained skill metadata immutable without transforming literal text."""
    from types import MappingProxyType

    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_snapshot(item) for key, item in value.items()})
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(_freeze_snapshot(item) for item in value)
    return value


def skill_preflight_fingerprint(identifier: str, payload: Mapping[str, Any]) -> str:
    """Stable fingerprint for the full canonical payload verified before child construction."""
    import hashlib
    import json

    encoded = json.dumps({"identifier": identifier, "payload": payload}, sort_keys=True,
                         separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def freeze_preloaded_skills(agent: Any, policy: DelegationLaunchPolicy) -> None:
    """Read required skills once, freezing body/support metadata and refusing init on any failure."""
    home = policy.profile_home
    if home is None:
        db_path = getattr(getattr(agent, "_session_db", None), "db_path", None)
        home = str(Path(db_path).parent) if isinstance(db_path, (str, Path)) else None
    loaded, missing, payloads = [], [], []
    expected_snapshot = dict(policy.skill_preflight_snapshot)
    if policy.skills is not None:
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        from agent.runtime_cwd import reset_session_cwd, set_session_cwd
        home_token = set_hermes_home_override(home) if home else None
        cwd_token = set_session_cwd(policy.workspace) if policy.workspace else None
        from tools.skills_tool import _is_skill_disabled
        try:
            for identifier in policy.skills:
                payload = preload_skill(identifier, getattr(agent, "session_id", None), home_override=home)
                if payload.get("success") is not True:
                    missing.append((identifier, _canonical_skill_error(payload, identifier)))
                    continue
                fingerprint = skill_preflight_fingerprint(identifier, payload)
                if (identifier in expected_snapshot and expected_snapshot[identifier] != fingerprint):
                    missing.append((identifier, _safe_skill_error(
                        f"Skill '{identifier}' changed after preflight; refusing to build child.")))
                    continue
                skill_name = str(payload.get("name") or identifier)
                if payload.get("readiness_status") in {"unsupported", "unverified"} or _is_skill_disabled(
                        skill_name, non_activating=True):
                    missing.append((identifier, _safe_skill_error(
                        payload.get("error") or f"Skill '{identifier}' is disabled or unsupported.")))
                    continue
                loaded.append(skill_name)
                frozen = _freeze_snapshot({
                    **payload, "content": str(payload["content"]),
                    "linked_files": {
                        str(category): tuple(str(path) for path in paths)
                        for category, paths in (payload.get("linked_files") or {}).items()
                        if isinstance(paths, list)
                    },
                })
                payloads.append((identifier, frozen))
        except (OSError, UnicodeError, ValueError, TypeError) as exc:
            remaining = [identifier for identifier in policy.skills if identifier not in loaded]
            missing.extend((identifier, _safe_skill_error(exc)) for identifier in remaining)
        finally:
            if cwd_token is not None:
                reset_session_cwd(cwd_token)
            if home_token is not None:
                reset_hermes_home_override(home_token)
    agent.__dict__["_delegation_preloaded_skill_names"] = tuple(loaded)
    agent.__dict__["_delegation_preloaded_missing"] = tuple(missing)
    agent.__dict__["_delegation_preloaded_skills"] = tuple(payloads)
    if missing:
        details = "; ".join(f"{identifier}: {error}" for identifier, error in missing)
        raise ValueError(f"Selected profile skill preload failed: {details}")
    agent.__dict__["_delegation_preloaded_prompt"] = _format_preloaded_payloads(
        policy.skills or (), payloads)


def _format_preloaded_payloads(identifiers: tuple[str, ...], payloads: list[tuple[str, dict[str, Any]]] | tuple
                              ) -> str:
    """Format the immutable snapshot body and support paths; never reload support files."""
    payload_map = dict(payloads)
    blocks = []
    for identifier in identifiers:
        payload = payload_map.get(identifier)
        if payload is None:
            continue
        name = str(payload.get("name") or identifier)
        lines = [f'[IMPORTANT: The "{name}" skill was preloaded for this delegated task. Treat its instructions as active guidance.]',
                 "", str(payload["content"])]
        if payload.get("skill_dir"):
            lines.extend(["", f"[Skill directory: {payload['skill_dir']}]"])
        files = tuple(path for paths in (payload.get("linked_files") or {}).values() for path in paths)
        if files:
            lines.extend(["", "[Supporting files (path metadata only; not preloaded):]"])
            lines.extend(f"- {path}" for path in files)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def build_preloaded_prompt_from_payloads(identifiers: tuple[str, ...],
                                         payloads: list[tuple[str, dict[str, Any]]] | tuple
                                         ) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Render only stored preload data; retain lookup compatibility without source rereads."""
    payload_map = dict(payloads)
    loaded = tuple(str(payload_map[key].get("name") or key) for key in identifiers if key in payload_map)
    missing = tuple(key for key in identifiers if key not in payload_map)
    return _format_preloaded_payloads(identifiers, payloads), loaded, missing


def initialize_launch_policy(
    agent: Any,
    strict_reasoning_effort: bool,
    reasoning_config: dict[str, Any] | None,
    delegation_launch_policy: Any,
) -> None:
    """Apply launch policy before API-mode, client, and tool initialization."""
    agent._strict_reasoning_effort = bool(strict_reasoning_effort)
    agent._strict_requested_reasoning_effort = (
        reasoning_config.get("effort") if strict_reasoning_effort and isinstance(reasoning_config, dict) else None
    )
    if delegation_launch_policy is None:
        return
    if not isinstance(delegation_launch_policy, DelegationLaunchPolicy):
        raise TypeError("delegation_launch_policy must be a DelegationLaunchPolicy.")
    agent._delegation_launch_policy = delegation_launch_policy
    if delegation_launch_policy.admitted_tool_names is not None:
        ceiling = set(delegation_launch_policy.admitted_tool_names)
        parent_grant = delegation_launch_policy.parent_tool_grant
        if ceiling and parent_grant is None:
            raise ValueError("Exact child tools refused because the owning parent's tool grant could not be verified.")
        invalid_parent = ceiling - set(parent_grant or ())
        if invalid_parent:
            raise ValueError(f"Exact child tools exceed the owning parent's tools: {', '.join(sorted(invalid_parent))}.")
    if (delegation_launch_policy.identity_snapshot is None
            and (delegation_launch_policy.system_prompt_mode == "append"
                 or delegation_launch_policy.profile_home is not None)):
        from pathlib import Path
        from agent.prompt_builder import DEFAULT_AGENT_IDENTITY, load_soul_md
        identity_home = Path(delegation_launch_policy.profile_home) if delegation_launch_policy.profile_home is not None else None
        identity_soul = load_soul_md(home_override=identity_home)
        delegation_launch_policy = dataclass_replace(
            delegation_launch_policy,
            identity_snapshot=identity_soul or DEFAULT_AGENT_IDENTITY,
            identity_soul_loaded=bool(identity_soul),
        )
        agent._delegation_launch_policy = delegation_launch_policy
    freeze_preloaded_skills(agent, delegation_launch_policy)
    agent._strict_reasoning_effort = delegation_launch_policy.strict_reasoning
    agent._strict_requested_reasoning_effort = delegation_launch_policy.reasoning_effort
    agent._delegate_route_diagnostics = dict(delegation_launch_policy.diagnostics)
