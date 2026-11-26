"""Task-local option validation and application over real backend route objects."""

import logging
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from hermes_cli.auth_constants import AuthError
from tools.delegate_tool_options import TaskRouteOptions, launch_policy

logger = logging.getLogger("tools.delegate_tool")


def resolve_task_options(tasks: list[dict[str, Any]]) -> tuple[list[TaskRouteOptions], list[str | None]]:
    options, errors = [], []
    for index, task in enumerate(tasks):
        parsed, error = TaskRouteOptions.parse(task, index)
        options.append(parsed or TaskRouteOptions())
        errors.append(error)
    return options, errors


def model_alias_route(model: str, config: Mapping[str, Any], provider: str | None = None) -> dict[str, Any] | None:
    """Resolve configured aliases exactly; explicit provider takes precedence over alias routing."""
    from hermes_cli.model_switch import _load_direct_aliases
    alias = _load_direct_aliases(dict(config)).get(model.lower())
    if alias is None:
        return None
    if provider:
        return {"model": alias.model, "provider": provider}
    return {"model": alias.model, "provider": alias.provider or "custom", "base_url": alias.base_url or None,
            "api_key": alias.api_key or None, "key_env": alias.key_env or None}


def _catalog_verdict(model: str, provider: str, route: Mapping[str, Any],
                     config: Mapping[str, Any] | None = None) -> tuple[bool, str, bool]:
    """Authoritative rejection refuses; inconclusive evidence stays unverified."""
    from hermes_cli.models_validate import static_model_provider_conflict, validate_requested_model
    from hermes_cli.model_switch import _extra_headers_from_config

    conflict = static_model_provider_conflict(model, provider)
    if conflict:
        return True, conflict["message"], False
    headers = None
    providers_cfg = (config or {}).get("providers")
    if isinstance(providers_cfg, Mapping) and provider in providers_cfg:
        headers = _extra_headers_from_config(providers_cfg[provider])
    try:
        verdict = validate_requested_model(
            model, provider, base_url=route.get("base_url"), api_key=route.get("api_key"),
            api_mode=route.get("api_mode"), headers=headers,
        )
    except Exception as exc:  # health: allow BLE001 -- plugin/catalog failures are inconclusive; attempt only pinned route
        exception_type = type(exc).__name__
        logger.debug("Delegation model catalog validation inconclusive (%s)", exception_type)
        return False, f"Model/provider validation was inconclusive ({exception_type}); exact route will be attempted.", False
    message = str(verdict.get("message") or "")
    if verdict.get("accepted") is False:
        return True, message or f"Model {model!r} was rejected by the {provider!r} catalog.", False
    return False, message, verdict.get("recognized") is True


def _reasoning_profile_capabilities(model: str, provider: str, route: Mapping[str, Any]):
    """Return selected profile capability ownership; failed hooks safely become unknown."""
    from providers import get_provider_profile, profile_for_host
    from providers.base import ProviderProfile

    provider_name = provider.lower()
    api_mode = route.get("api_mode")
    boundary_failure = None
    try:
        profile = get_provider_profile(provider)
        endpoint_profile = profile_for_host(str(route.get("base_url") or "")) if route.get("base_url") else None
        profile = endpoint_profile or profile
        provider_owner = profile.name if profile else provider_name
        declared = profile.supported_reasoning_efforts(model) if profile else None
        profile_owns_capabilities = profile is not None and (
            type(profile).supported_reasoning_efforts is not ProviderProfile.supported_reasoning_efforts
        )
    except Exception as exc:  # health: allow BLE001 -- arbitrary profile hooks; failed evidence stays unknown
        profile, provider_owner, declared, profile_owns_capabilities = None, provider_name, None, False
        boundary_failure = type(exc).__name__
        logger.debug("Delegation provider capability lookup failed (%s)", boundary_failure)

    # Without an explicit resolved mode, only the profile's own default wire is known.
    if not api_mode and profile is not None:
        api_mode = profile.api_mode
    if profile is not None and api_mode:
        profile_mode = profile.api_mode
        mode_hook_failed = False
        try:
            profile_mode = profile.api_mode_for_model(model) or profile_mode
        except Exception as exc:  # health: allow BLE001 -- per-model mode hook failure invalidates capability proof
            # A failed mode lookup cannot validate capabilities against the static mode.
            api_mode = None
            declared = None
            profile_owns_capabilities = False
            mode_hook_failed = True
            logger.debug("Delegation provider api_mode_for_model hook failed (%s)", type(exc).__name__)
            boundary_failure = type(exc).__name__
        if not mode_hook_failed and profile_mode != api_mode:
            # A vocabulary declared for one protocol does not prove another protocol's levels.
            declared = None
            profile_owns_capabilities = False
    return provider_name, api_mode, provider_owner, declared, profile_owns_capabilities, boundary_failure


def _declared_reasoning_verdict(model: str, provider: str, provider_name: str, provider_owner: str,
                                api_mode: str | None, effort: str, declared: tuple[str, ...] | None,
                                profile_owns_capabilities: bool) -> tuple[bool, tuple[str, str] | None]:
    """Decide explicit declared capabilities; False means defer to non-profile route rules."""
    from agent.reasoning_effort import codex_supported_efforts, route_supported_efforts

    if effort == "none" and provider_owner == "custom" and profile_owns_capabilities and api_mode == "chat_completions":
        return True, None

    if declared is not None:
        if effort in declared:
            return True, None
        return True, ("refused", f"Reasoning effort {effort!r} is unsupported by {provider} for {model!r}.")

    # OpenAI's Responses route has a model-specific ladder, not the generic chat wire levels.
    if provider_name == "openai-codex" and api_mode in ("codex_responses", "codex_app_server"):
        supported = (route_supported_efforts(provider, model, api_mode)
                     if api_mode == "codex_app_server" else codex_supported_efforts(model))
        if effort in supported:
            return True, None
        return True, ("refused", f"Reasoning effort {effort!r} is unsupported by {provider} for {model!r}.")

    # These providers expose route-specific wire vocabularies through shared transport helpers.
    if provider_owner == "kimi-coding" and api_mode == "chat_completions":
        if effort == "none":
            return True, None  # Kimi's existing thinking toggle carries explicit disable.
        from agent.reasoning_effort import kimi_supported_efforts
        if effort in kimi_supported_efforts(model):
            return True, None
        return True, ("refused", f"Reasoning effort {effort!r} is unsupported by {provider} for {model!r}.")
    if provider_owner == "tencent-tokenhub" and api_mode == "chat_completions":
        from agent.reasoning_effort import TOKENHUB_EFFORTS
        if effort == "none":
            return True, ("refused", "Explicit reasoning effort 'none' cannot be carried exactly by the final TokenHub request.")
        if effort in TOKENHUB_EFFORTS:
            return True, None
        return True, ("refused", f"Reasoning effort {effort!r} is unsupported by {provider} for {model!r}.")
    return False, None


def _reasoning_catalog_verdict(model: str, provider: str, provider_name: str, api_mode: str | None,
                               effort: str, declared: tuple[str, ...] | None,
                               provider_owner: str, profile_owns_capabilities: bool) -> tuple[str, str] | None:
    """Use cache-only model metadata to reject known lack, not infer exact effort levels."""
    from agent.reasoning_effort import codex_supported_efforts

    # A native model catalog may say a model has no reasoning at all, but cannot prove exact levels.
    cache_failure = None
    try:
        from agent.models_dev import get_model_capabilities
        caps = get_model_capabilities(provider, model, allow_network=False) if provider_name not in {"custom", "auto"} else None
    except Exception as exc:  # health: allow BLE001 -- cache lookup failure means unknown; never infer support
        caps = None
        cache_failure = type(exc).__name__
        logger.debug("Delegation reasoning capability cache lookup failed (%s)", cache_failure)

    failure_note = f" Capability lookup failed ({cache_failure})." if cache_failure else ""
    if caps is not None and caps.supports_reasoning is False and effort != "none":
        return "refused", f"Model {model!r} on provider {provider!r} does not support reasoning effort {effort!r}."
    if effort == "none":
        if api_mode in {"codex_responses", "codex_app_server"} and effort not in codex_supported_efforts(model):
            return "refused", f"Reasoning effort none is unsupported by {provider} for {model!r}."
        if declared == () or (provider_owner == "custom" and profile_owns_capabilities):
            return "refused", f"Reasoning effort 'none' is unsupported by {provider} for {model!r}."
        return "unverified", f"Reasoning disable support is unknown for {provider}/{model}; preserving the requested effort exactly.{failure_note}"

    if api_mode in {"anthropic_messages", "bedrock_converse"}:
        return "unverified", f"Reasoning support is unknown for {provider}/{model} on {api_mode}; preserving the requested effort exactly.{failure_note}"
    if api_mode in {"codex_responses", "codex_app_server"}:
        return "unverified", f"Reasoning support is unknown for {provider}/{model} on {api_mode}; preserving the requested effort exactly.{failure_note}"

    # A generic OpenAI-compatible effort name is syntax, not route evidence. Only declared route
    # owners above may turn it into a verified capability claim.
    return "unverified", f"Reasoning support is unknown for {provider}/{model}; preserving the requested effort exactly.{failure_note}"


def _reasoning_verdict(model: str, provider: str, route: Mapping[str, Any], effort: str) -> tuple[str, str] | None:
    """Reject known unsupported efforts; mark unknown capability instead of guessing."""
    provider_name, api_mode, provider_owner, declared, profile_owns_capabilities, boundary_failure = (
        _reasoning_profile_capabilities(model, provider, route)
    )
    handled, verdict = _declared_reasoning_verdict(
        model, provider, provider_name, provider_owner, api_mode, effort, declared, profile_owns_capabilities,
    )
    if handled:
        return verdict
    result = _reasoning_catalog_verdict(
        model, provider, provider_name, api_mode, effort, declared, provider_owner, profile_owns_capabilities,
    )
    if boundary_failure and result and result[0] == "unverified":
        return result[0], f"{result[1]} Provider capability hook failed ({boundary_failure})."
    return result


def _workspace_for_parent(parent_agent: Any) -> str | None:
    from tools.delegate_tool_progress import _resolve_workspace_hint
    return _resolve_workspace_hint(parent_agent)


def _parent_profile_home(parent_agent: Any) -> Path:
    """Read owner identity from typed persistence before any ambient selected-child scope."""
    from hermes_constants import get_hermes_home

    for owner in (getattr(parent_agent, "persistence", None), getattr(parent_agent, "_session_db", None)):
        home = getattr(getattr(owner, "scope", None), "profile_dir", None)
        if isinstance(home, (str, Path)):
            return Path(home)
    db_path = getattr(getattr(parent_agent, "_session_db", None), "db_path", None)
    return Path(db_path).parent if isinstance(db_path, (str, Path)) else get_hermes_home()


def _parent_tool_grant(parent_agent: Any, requested_names: tuple[str, ...] | list[str] = (),
                       *, include_bridges: bool = False) -> tuple[str, ...] | None:
    """Concrete tools visible to the owning parent before deferred tools are bridged."""
    if parent_agent is None:
        return None
    owner_home = _parent_profile_home(parent_agent)
    try:
        from tools.delegate_tool_routes import delegated_profile_scope
        import model_tools
        from tools import tool_search
        from toolsets import agent_tool_surface, session_disabled_toolsets
        from agent.oneshot_footprint import agent_is_oneshot

        # Route resolution can be running inside a selected child profile. Rebind explicitly to
        # the parent SessionDB owner before reading its config, toolsets, and deferred registry.
        with delegated_profile_scope(owner_home):
            enabled_toolsets = getattr(parent_agent, "enabled_toolsets", None)
            configured_disabled = getattr(parent_agent, "_configured_disabled_toolsets",
                                          getattr(parent_agent, "disabled_toolsets", None))
            disabled_toolsets = session_disabled_toolsets(
                configured_disabled, getattr(parent_agent, "platform", None),
                oneshot=agent_is_oneshot(parent_agent))
            definitions = model_tools.get_tool_definitions(
                enabled_toolsets=enabled_toolsets,
                disabled_toolsets=disabled_toolsets,
                quiet_mode=True,
                skip_tool_search_assembly=True,
                update_last_resolved_tool_names=False,
            ) or []
            search_config = tool_search.load_config_readonly()
            deferred_enabled = search_config.effective_defer_tools
            deferred_bridge_enabled = search_config.enabled != "off"
            # Inactive search assembly exposes its incoming definitions directly.
            deferred_names = (set(tool_search.scoped_deferrable_names(definitions))
                              if deferred_bridge_enabled else set())
            eager_definitions = [item for item in definitions
                                 if item.get("function", {}).get("name") not in deferred_names]
            assembled_parent_defs = agent_tool_surface(SimpleNamespace(
                platform=getattr(parent_agent, "platform", None),
                _tool_snapshot_selection=getattr(parent_agent, "_tool_snapshot_selection", None),
            ), eager_definitions)
            assembled_names = {
                item["function"]["name"] for item in assembled_parent_defs
                if isinstance(item, Mapping) and isinstance(item.get("function"), Mapping)
                and isinstance(item["function"].get("name"), str)
            }
            concrete_names = {
                function["name"] for item in definitions
                if isinstance(item, Mapping) and isinstance((function := item.get("function")), Mapping)
                and isinstance(function.get("name"), str)
            }
            live_names = getattr(parent_agent, "valid_tool_names", None)
            if live_names is None:
                live_tools = getattr(parent_agent, "tools", None)
                parent_live_surface_known = live_tools is not None
                parent_live_names = {
                    function["name"] for item in (live_tools or [])
                    if isinstance(item, Mapping) and isinstance((function := item.get("function")), Mapping)
                    and isinstance(function.get("name"), str)
                }
            else:
                parent_live_surface_known = True
                parent_live_names = set(live_names)
            from tools.connectors.gateway.config import connectors_available
            connector_portal_available = bool(connectors_available())
        from tools.registry import registry
        from tools.connectors.gateway.names import format_connector_name
        from agent.delegation_launch import effective_tool_ceiling
        # Definitions already apply the canonical None/[] selection and session gates.
        selected_names = concrete_names
        registered_entries = registry.get_all_entries()
        registered_names = {entry.name for entry in registered_entries}
        registered_toolsets = {entry.name: entry.toolset for entry in registered_entries}
        classes = {
            name: ("deferred" if name in deferred_names else "eager")
            for name in selected_names & registered_names
            if ((name in assembled_names and
                 (name in parent_live_names or not parent_live_surface_known)) if name not in deferred_names else (
                (name in deferred_enabled or
                 (registered_toolsets.get(name, "").startswith("mcp-") and deferred_bridge_enabled))
                and name in concrete_names))
        }
        from tools.tool_search_catalog import BRIDGE_TOOL_NAMES
        if include_bridges:
            classes.update({name: "bridge" for name in BRIDGE_TOOL_NAMES & parent_live_names
                            if name in requested_names})
        connector_names = [name for name in requested_names if name.startswith("connectors__")]
        if connector_names and include_bridges:
            ancestor_names = set(getattr(parent_agent, "_exact_tool_name_ceiling", ()))
            inherited = bool(ancestor_names) and {"tool_call", "manage_connections"}.issubset(ancestor_names)
            if not inherited and not {"tool_call", "manage_connections"}.issubset(parent_live_names):
                return None
            from tools.connectors.gateway.bridge import connector_describe
            from tools.connectors.gateway.config import connectors_available
            if connectors_available() or inherited:
                leg = connector_describe(connector_names)
                tools = (leg.payload or {}).get("tools") if not leg.failure else None
                if isinstance(tools, Mapping):
                    classes.update({name: "connector" for name in connector_names if name in tools})
        if connector_portal_available and "manage_connections" in assembled_names:
            from tools.connectors.gateway.bridge import connector_search_hits
            availability = connector_search_hits([{"use_case": ""}])
            schemas = (availability.payload or {}).get("schemas") if not availability.failure else None
            if isinstance(schemas, Mapping):
                if "manage_connections" not in selected_names:
                    schemas = {}
                for slug, schema in schemas.items():
                    if isinstance(schema, Mapping) and schema.get("connector"):
                        classes[format_connector_name(str(schema["connector"]).lower(), str(slug))] = "connector"
        ceiling = effective_tool_ceiling(parent_agent)
        if ceiling is not None:
            classes = {name: kind for name, kind in classes.items() if name in ceiling}
        return tuple(sorted(classes))
    except Exception as exc:  # health: allow BLE001 -- parent-grant plugin/connector boundary fails closed; log class only
        exception_type = type(exc).__name__
        logger.warning("Unable to snapshot owning-parent concrete tool grant (%s)", exception_type)
        return None


def _strict_inherited_tool_ceiling(parent_agent: Any, parent_tool_grant: tuple[str, ...] | None) -> tuple[str, ...] | None:
    """Compose a strict ancestor grant with this child's depth-derived delegation role."""
    from agent.delegation_launch import effective_tool_ceiling
    ceiling = effective_tool_ceiling(parent_agent)
    if ceiling is None:
        return None
    if parent_tool_grant is None:
        raise ValueError("The owning parent's exact tool grant could not be verified.")
    from tools.delegate_tool_toolsets import DELEGATE_BLOCKED_TOOLS
    blocked = set(DELEGATE_BLOCKED_TOOLS)
    child_depth = int(getattr(parent_agent, "_delegate_depth", 0) or 0) + 1
    from tools.delegate_tool_config import _get_max_spawn_depth, _get_orchestrator_enabled
    if _get_orchestrator_enabled() and child_depth < _get_max_spawn_depth():
        blocked.discard("delegate_task")
    return tuple(sorted((set(parent_tool_grant) & set(ceiling)) - blocked))


def _exact_builtin_tool_grant_error(index: int, option: TaskRouteOptions,
                                    parent_tool_grant: tuple[str, ...] | None, route: Any,
                                    profile_config: Mapping[str, Any], parent_agent: Any) -> str | None:
    """Refuse exact selections unless each literal is admitted and exists in the profile."""
    if not option.tools_set:
        return None
    if option.tools is None:
        return f"Task {index}: exact tools are missing their immutable grant list."
    if not option.tools:
        return None
    from tools.tool_search_catalog import BRIDGE_TOOL_NAMES
    requested = set(option.tools)
    connectors = {name for name in requested if name.startswith("connectors__")}
    connector_error = _exact_connector_error(index, connectors, requested, BRIDGE_TOOL_NAMES)
    if connector_error:
        return connector_error
    bridge_error = _exact_bridge_error(index, requested, BRIDGE_TOOL_NAMES)
    if bridge_error:
        return bridge_error
    verified_parent_grant = parent_tool_grant
    if verified_parent_grant is None:
        if not connectors:
            return f"Task {index}: exact tool grant could not be verified against the owning parent's tools."
        verified_parent_grant = _parent_tool_grant(parent_agent, tuple(requested), include_bridges=True)
    if verified_parent_grant is None:
        return f"Task {index}: exact tool grant could not be verified against the owning parent's tools."
    inherited = _strict_inherited_tool_ceiling(parent_agent, verified_parent_grant)
    allowed_parent = set(inherited if inherited is not None else verified_parent_grant)
    missing = sorted(requested - allowed_parent)
    if missing:
        return f"Task {index}: exact tools {missing!r} exceed the owning parent's tools."
    from tools.registry import registry
    from tools.tool_search_catalog import BRIDGE_TOOL_NAMES
    unsupported = sorted(name for name in requested
                         if name not in BRIDGE_TOOL_NAMES and not name.startswith("connectors__")
                         and registry.get_entry(name) is None)
    if unsupported:
        return f"Task {index}: exact tools {unsupported!r} are unsupported or non-builtin."
    return _validate_exact_profile_surface(index, option, route, profile_config, parent_agent)


def _exact_connector_error(index: int, connectors: set[str], requested: set[str], bridges: frozenset[str]) -> str | None:
    """Require registered, well-formed connector literals plus all necessary bridge grants."""
    if not connectors:
        return None
    from tools.connectors.gateway.names import parse_connector_name
    if any(parse_connector_name(name) is None for name in connectors):
        return f"Task {index}: malformed connector tool names: {sorted(connectors)!r}."
    missing = _missing_connector_grants(requested, bridges)
    return (f"Task {index}: exact connector tools require explicit bridge and manage_connections grants."
            if missing else None)


def _missing_connector_grants(requested: set[str], bridges: frozenset[str]) -> frozenset[str]:
    return frozenset({"tool_call", "manage_connections"}) - requested


def _exact_bridge_error(index: int, requested: set[str], bridges: frozenset[str]) -> str | None:
    """Bridge names are explicit grants and unavailable when tool_search is disabled."""
    return _bridge_configuration_error(index, requested, bridges)


def _bridge_configuration_error(index: int, requested: set[str], bridges: frozenset[str]) -> str | None:
    if not requested & bridges:
        return None
    from tools.tool_search import load_config_readonly
    return (f"Task {index}: exact bridge tools are disabled by tool_search configuration."
            if load_config_readonly().enabled == "off" else None)


def _validate_exact_profile_surface(index: int, option: TaskRouteOptions, route: Any,
                                    profile_config: Mapping[str, Any], parent_agent: Any) -> str | None:
    """Prove every literal resolves in the selected profile's concrete registry surface."""
    from tools.tool_search_catalog import BRIDGE_TOOL_NAMES
    from tools import tool_search
    parent_live_names: set[str] = set(getattr(parent_agent, "valid_tool_names", ()) or ())
    from tools.delegate_tool_toolsets import DELEGATE_BLOCKED_TOOLS
    blocked = set(DELEGATE_BLOCKED_TOOLS)
    child_depth = int(getattr(parent_agent, "_delegate_depth", 0) or 0) + 1
    from tools.delegate_tool_config import _get_max_spawn_depth, _get_orchestrator_enabled
    if _get_orchestrator_enabled() and child_depth < _get_max_spawn_depth():
        blocked.discard("delegate_task")
    selected_home = getattr(route, "profile_home", None)
    config = dict(profile_config)
    if selected_home is None and option.profile_set:
        try:
            from tools.delegate_tool_routes import _profile_home, delegated_profile_scope
            from hermes_cli.config import load_config_readonly
            selected_home = _profile_home(option.profile)
            with delegated_profile_scope(selected_home):
                config = dict(load_config_readonly() or {})
        except (ValueError, TypeError, KeyError, OSError) as exc:
            return f"Task {index}: exact builtin tool grant could not inspect the selected profile ({type(exc).__name__})."
    try:
        from hermes_cli.tools_config import _get_platform_tools
        from tools.delegate_tool_routes import delegated_profile_scope
        from tools.registry import registry
        from toolsets import session_disabled_toolsets
        import model_tools
        platform = str(getattr(parent_agent, "platform", None) or "cli")
        parent_home = _parent_profile_home(parent_agent)
        parent_config = dict(profile_config)
        with delegated_profile_scope(parent_home):
            from hermes_cli.config import load_config_readonly
            parent_config = dict(load_config_readonly() or {})
        if selected_home is None:
            selected_home = parent_home
        with delegated_profile_scope(selected_home):
            enabled = sorted(_get_platform_tools(config, platform))
            disabled = session_disabled_toolsets((config.get("agent") or {}).get("disabled_toolsets"), platform)
            definitions = model_tools.get_tool_definitions(enabled_toolsets=enabled,
                                                           skip_tool_search_assembly=True,
                                                           disabled_toolsets=disabled, quiet_mode=True,
                                                           update_last_resolved_tool_names=False) or []
            bridge_enabled = tool_search.load_config_readonly().enabled != "off"
            registered_names = {entry.name for entry in registry.get_all_entries()}
            available = _exact_surface_names(definitions, registered_names,
                                             bridge_enabled, parent_live_names)
        if route.profile_home is None and not option.profile_set:
            platform = str(getattr(parent_agent, "platform", None) or "cli")
            with delegated_profile_scope(parent_home):
                enabled = sorted(_get_platform_tools(parent_config, platform))
                disabled = session_disabled_toolsets((parent_config.get("agent") or {}).get("disabled_toolsets"), platform)
                definitions = model_tools.get_tool_definitions(
                    enabled_toolsets=enabled, disabled_toolsets=disabled, quiet_mode=True,
                    skip_tool_search_assembly=True,
                    update_last_resolved_tool_names=False,
                ) or []
                bridge_enabled = tool_search.load_config_readonly().enabled != "off"
                registered_names = {entry.name for entry in registry.get_all_entries()}
                available = _exact_surface_names(definitions, registered_names,
                                                 bridge_enabled, parent_live_names)
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
        logger.debug("Exact built-in grant validation inconclusive (%s)", type(exc).__name__)
        return f"Task {index}: exact builtin tool grant could not be verified against the selected profile's available tools."
    unavailable = [name for name in option.tools if name not in available or name in blocked]
    if unavailable:
        if getattr(route, "profile_home", None) is not None or option.profile_set:
            return (f"Task {index}: exact tools {unavailable!r} are disabled by the selected profile's "
                    f"{platform} toolset selection or child capability policy.")
        return f"Task {index}: exact tools {unavailable!r} are unavailable in the effective child tool surface."
    return None


def _exact_surface_names(definitions: Any, registered: set[str],
                         bridges_enabled: bool, parent_live: set[str]) -> set[str]:
    """Names available to a strict child in the selected session surface."""
    from tools.tool_search_catalog import BRIDGE_TOOL_NAMES

    available: set[str] = set()
    for item in definitions:
        function = item.get("function")
        if not isinstance(function, Mapping):
            continue
        name = function.get("name")
        if not isinstance(name, str):
            continue
        if name in BRIDGE_TOOL_NAMES:
            if bridges_enabled and name in parent_live:
                available.add(name)
        elif name in registered:
            available.add(name)
    return available


def _launch_policy(task: Mapping[str, Any], option: TaskRouteOptions, diagnostics: Mapping[str, Any], reason: str | None = None,
                   *, profile_home: Any = None, workspace: str | None = None,
                   parent_tool_grant: tuple[str, ...] | None = None):
    from agent.delegation_launch import with_diagnostics
    policy = with_diagnostics(launch_policy(task, option), diagnostics, reason)
    return replace(policy, profile_home=str(profile_home) if profile_home is not None else None,
                   workspace=workspace, parent_tool_grant=parent_tool_grant)


def _baseline_default_model(config: Mapping[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    """Resolve the profile's saved default and alias before an explicit provider switch."""
    from tools.delegate_tool_options import _profile_default_model

    model_cfg = config.get("model")
    saved = (model_cfg.get("default") or model_cfg.get("model")) if isinstance(model_cfg, Mapping) else model_cfg
    if not isinstance(saved, str) or not saved.strip():
        return None, None
    route = _profile_default_model(config, str(model_cfg.get("provider") or "") if isinstance(model_cfg, Mapping) else "")
    if not route:
        return saved, None
    return str(route.get("model") or saved), route


def _selected_profile_config(profile_home: Any) -> dict[str, Any]:
    from tools.delegate_tool_routes import delegated_profile_scope
    from hermes_cli.config import load_config_readonly
    with delegated_profile_scope(profile_home):
        return dict(load_config_readonly() or {})


def _task_baseline(task: Mapping[str, Any], config: Mapping[str, Any], route: Any,
                   option: TaskRouteOptions) -> tuple[Any, Mapping[str, Any]]:
    from tools.delegate_tool_routes import _profile_home, _profile_route

    if task.get("profile"):
        home = _profile_home(task["profile"])
        if not option.provider_set:
            selected = _profile_route(home)
            return selected, _selected_profile_config(selected.profile_home)
        # An explicit provider replaces the profile's provider, so do not resolve its credentials.
        from tools.delegate_tool_routes import _refuse_under_worker_snapshot
        _refuse_under_worker_snapshot(home)
        profile_config = _selected_profile_config(home)
        from hermes_constants import resolve_reasoning_config
        from tools.delegate_tool_config import ResolvedReasoning
        model_block = profile_config.get("model")
        baseline_provider = str(model_block.get("provider") or "custom") if isinstance(model_block, Mapping) else "custom"
        default_route = None
        if not option.model_set:
            from tools.delegate_tool_options import _profile_default_model
            default_route = _profile_default_model(profile_config, baseline_provider)
        reasoning_model = str((default_route or {}).get("model") or option.model or "")
        selected = replace(
            route,
            creds={"model": None, "provider": None, "base_url": None, "api_key": None,
                   "api_mode": None, "request_overrides": {}},
            reasoning_effort=ResolvedReasoning(resolve_reasoning_config(profile_config, reasoning_model)),
            routing_cfg={"fallback_providers": []}, profile_home=home,
        )
        return selected, profile_config
    if route.profile_home is not None:
        return route, _selected_profile_config(route.profile_home)
    from hermes_cli.config import load_config_readonly
    return route, load_config_readonly() or {}


def _task_model_provider(task: Mapping[str, Any], option: TaskRouteOptions, creds: dict[str, Any],
                         profile_config: Mapping[str, Any], index: int, parent_agent: Any) -> tuple[dict[str, Any], str | None]:
    model, provider = creds.get("model"), creds.get("provider")
    if option.provider_set and not option.model_set:
        # Preserve the selected profile's saved model/alias, but don't resolve old-provider auth.
        from tools.delegate_tool_options import _profile_default_model
        default_route = _profile_default_model(profile_config, str(provider or "custom"))
        if not default_route:
            raise ValueError(f"Task {index}: provider {option.provider!r} has no saved profile default model; supply model explicitly.")
        model = default_route.get("model")
    alias = None
    if option.model_set:
        alias = model_alias_route(str(option.model), profile_config, option.provider if option.provider_set else None)
        if alias:
            model = alias["model"]
            if not isinstance(model, str) or not model.strip():
                raise ValueError(f"Task {index}: selected route has no model identifier.")
            requested_alias_provider = option.provider if option.provider_set else alias.get("provider") or "custom"
            if option.provider_set:
                from tools.delegate_tool_options import _provider_credentials
                creds = _provider_credentials(str(option.provider), model, profile_config)
            else:
                from hermes_cli.model_switch import _load_direct_aliases, direct_alias_runtime_request
                from tools.delegate_tool_options import _provider_credentials
                direct_alias = _load_direct_aliases(dict(profile_config)).get(str(option.model).lower())
                if direct_alias is None:
                    raise ValueError(f"Configured model alias {option.model!r} could not be resolved.")
                alias_provider, alias_key = direct_alias_runtime_request(direct_alias)
                creds = _provider_credentials(alias_provider, model, profile_config,
                                              base_url=direct_alias.base_url or None, api_key=alias_key)
            creds["provider"] = requested_alias_provider
            provider = requested_alias_provider
        else:
            model = option.model
            if not isinstance(model, str) or not model.strip():
                raise ValueError(f"Task {index}: selected route has no model identifier.")
            from tools.delegate_tool_options import _provider_credentials
            provider = str(provider or getattr(parent_agent, "provider", "") or "")
            if provider and not option.provider_set:
                base_url = creds.get("base_url") or getattr(parent_agent, "base_url", None)
                api_key = creds.get("api_key") or getattr(parent_agent, "api_key", None)
                creds = {**creds, **_provider_credentials(provider, model, profile_config,
                                                          base_url=base_url, api_key=api_key)}
    if option.provider_set:
        provider = option.provider
        # Alias handling above already resolves an explicitly selected provider exactly once.
        if not option.model_set or not alias:
            from tools.delegate_tool_options import _provider_credentials
            creds = {**creds, **_provider_credentials(str(provider), str(model), profile_config)}
    return {**creds, "model": model, "provider": provider}, str(model) if model is not None else None


def _preflight_static_pair(model: str, provider: str, index: int) -> None:
    """Refuse curated model/provider conflicts before provider credential resolution."""
    from hermes_cli.models_validate import static_model_provider_conflict

    conflict = static_model_provider_conflict(model, provider)
    if conflict:
        raise ValueError(f"Task {index}: {conflict['message']}")


def _preflight_route_pair(option: TaskRouteOptions, credentials: Mapping[str, Any],
                          profile_config: Mapping[str, Any], route: Any, parent_agent: Any, index: int) -> None:
    """Resolve only the selected pair identity, without touching its provider credentials."""
    model = str(option.model or credentials.get("model") or "") or None
    provider = str(option.provider or credentials.get("provider") or getattr(parent_agent, "provider", "") or "") or None
    if option.provider_set and not option.model_set:
        from tools.delegate_tool_options import _profile_default_model
        default_route = _profile_default_model(profile_config, str(option.provider))
        model = str((default_route or {}).get("model") or "") or None
    if option.model_set:
        alias = model_alias_route(str(option.model), profile_config, option.provider if option.provider_set else None)
        if alias:
            model = str(alias.get("model") or model or "") or None
            if not option.provider_set:
                provider = str(alias.get("provider") or "custom")
    if not model and not (option.provider_set and not option.model_set) and route.profile_home is None:
        model = str(getattr(parent_agent, "model", "") or "") or None
    if model and provider:
        _preflight_static_pair(model, provider, index)


def _validate_task_route(model: str, provider: str, creds: dict[str, Any], profile_config: Mapping[str, Any],
                         option: TaskRouteOptions, routing: dict[str, Any], diagnostics: dict[str, Any], index: int) -> None:
    if not option.route_pinned:
        return
    mismatch, message, recognized = _catalog_verdict(model, provider, creds, profile_config)
    if mismatch:
        raise ValueError(f"Task {index}: {message}")
    if not recognized:
        diagnostics.update(validation="unverified", warning=(
            f"Task {index}: pair {model!r} / {provider!r} is unverified; attempting the exact selection. " + message))
    routing["fallback_providers"] = []


def _resolve_task_reasoning(option: TaskRouteOptions, model: str | None, provider: str | None,
                            creds: dict[str, Any], route: Any, parent_agent: Any):
    effort = option.reasoning_effort if option.reasoning_effort_set else route.reasoning_effort
    effective_api_mode = creds.get("api_mode")
    if (not effective_api_mode and not creds.get("provider") and not creds.get("command")
            and parent_agent is not None):
        effective_api_mode = getattr(parent_agent, "api_mode", None)
    if option.reasoning_effort_set:
        if not isinstance(effort, str):
            raise ValueError("Explicit reasoning effort is missing from the child reasoning config.")
        if not model or not provider:
            raise ValueError("Task route has no model/provider for reasoning validation.")
        reasoning_route = creds if creds.get("api_mode") == effective_api_mode else {
            **creds, "api_mode": effective_api_mode,
        }
        verdict = _reasoning_verdict(model, provider, reasoning_route, effort)
        if verdict and verdict[0] == "refused":
            raise ValueError(verdict[1])
        reasoning_diagnostics = ({"reasoning_validation": "unverified", "reasoning_warning": verdict[1]}
                                 if verdict and verdict[0] == "unverified" else {})
    else:
        reasoning_diagnostics = {}
    if option.tools_set and effective_api_mode == "codex_app_server":
        raise ValueError("An exact tools override cannot be enforced by the Codex app-server transport.")
    if option.system_prompt is not None and effective_api_mode == "codex_app_server":
        raise ValueError("A persona override cannot be enforced by the Codex app-server transport.")
    if option.system_prompt is not None and creds.get("command"):
        raise ValueError("A persona override cannot be enforced by an external command transport.")
    if creds.get("command"):
        if option.tools_set:
            raise ValueError("An exact tools override cannot be enforced by an external command transport.")
        if option.skills_set:
            raise ValueError("A preloaded skills override cannot be enforced by an external command transport.")
        if option.reasoning_effort_set:
            raise ValueError("An explicit reasoning effort cannot be enforced by an external command transport.")
    policy = replace(launch_policy({}, option), strict_reasoning=option.reasoning_effort_set)
    if reasoning_diagnostics:
        policy = _launch_policy({}, option, reasoning_diagnostics)
    from agent.delegation_launch import with_diagnostics
    parent_grant = _parent_tool_grant(parent_agent)
    if parent_grant is not None:
        policy = replace(policy, parent_tool_grant=parent_grant)
    inherited_ceiling = _strict_inherited_tool_ceiling(parent_agent, parent_grant)
    if inherited_ceiling is not None:
        policy = replace(policy, admitted_tool_names=inherited_ceiling)
    policy = replace(policy, system_prompt=option.system_prompt, system_prompt_mode=option.system_prompt_mode,
                     tools=option.tools, skills=option.skills,
                     profile_home=(str(route.profile_home) if route.profile_home is not None else
                                   (str(_parent_home(parent_agent)) if option.skills is not None else None)),
                     workspace=_workspace_for_parent(parent_agent) if parent_agent is not None else None)
    if option.tools_set:
        policy = replace(policy, admitted_tool_names=option.tools)
    if route.diagnostics:
        policy = with_diagnostics(policy, route.diagnostics)
    return effort, policy, reasoning_diagnostics


def _preflight_selected_skills(option: TaskRouteOptions, profile_home: Any,
                               parent_agent: Any = None) -> tuple[str | None, tuple[str, ...], tuple[tuple[str, str], ...]]:
    """Validate requested skill names against the selected profile before child inference."""
    if not option.skills_set or option.skills is None:
        return None, (), ()
    from agent.delegation_launch import preload_skill, skill_preflight_fingerprint
    missing = []
    seen_names = set()
    snapshots = []
    for identifier in option.skills:
        selected_home = profile_home or _parent_home(parent_agent)
        from contextlib import nullcontext
        from agent.runtime_cwd import reset_session_cwd, set_session_cwd
        from tools.delegate_tool_routes import delegated_profile_scope
        profile_context = delegated_profile_scope(selected_home) if selected_home else nullcontext()
        workspace = _workspace_for_parent(parent_agent)
        cwd_token = set_session_cwd(workspace) if workspace else None
        try:
            with profile_context:
                payload = preload_skill(identifier, None, home_override=selected_home)
        finally:
            if cwd_token is not None:
                reset_session_cwd(cwd_token)
        if payload.get("success") is not True:
            from agent.delegation_launch import _canonical_skill_error
            missing.append((identifier, _canonical_skill_error(payload, identifier)))
            continue
        skill_name = str(payload.get("name") or identifier)
        if skill_name in seen_names:
            missing.append((identifier, f"Skill '{identifier}' resolves to a duplicate selected skill name '{skill_name}'."))
            continue
        seen_names.add(skill_name)
        snapshots.append((identifier, skill_preflight_fingerprint(identifier, payload)))
        if payload.get("readiness_status") in {"unsupported", "disabled", "unverified"}:
            missing.append((identifier, str(payload.get("error") or
                                           f"Skill '{identifier}' is disabled or unsupported.")))
    if missing:
        from agent.delegation_launch import _safe_skill_error
        details = "; ".join(f"{identifier}: {_safe_skill_error(error)}" for identifier, error in missing)
        return (f"Selected profile skill preload refused: {details}",
                tuple(identifier for identifier, _ in missing), ())
    return None, (), tuple(snapshots)


def _parent_home(parent_agent: Any) -> Any:
    if parent_agent is None:
        return None
    from hermes_constants import get_hermes_home
    return get_hermes_home()


def _replace_refused(route: Any, task: Mapping[str, Any], option: TaskRouteOptions, error: Exception,
                     parent_agent: Any = None) -> Any:
    message = str(error)
    diagnostics = {"validation": "refused", "warning": message}
    policy = _launch_policy(
        task, option, diagnostics, message, profile_home=route.profile_home,
        workspace=_workspace_for_parent(parent_agent) if parent_agent is not None else None,
        parent_tool_grant=_parent_tool_grant(parent_agent),
    )
    return replace(route, error=message, diagnostics=diagnostics, launch_policy=policy)


def _compose_parent_tool_policy(policy: Any, parent_agent: Any, option: TaskRouteOptions):
    """Attach verified parent tools and intersect a strict ancestor ceiling."""
    from agent.delegation_launch import effective_tool_ceiling

    parent_grant = _parent_tool_grant(parent_agent)
    ceiling = effective_tool_ceiling(parent_agent)
    if ceiling is None:
        return replace(policy, parent_tool_grant=parent_grant) if parent_grant is not None else policy
    if parent_grant is None:
        raise ValueError("The owning parent's exact tool grant could not be verified.")
    inherited = _strict_inherited_tool_ceiling(parent_agent, parent_grant)
    admitted = option.tools if option.tools_set else inherited
    return replace(policy, parent_tool_grant=parent_grant, admitted_tool_names=admitted)


def _legacy_route_policy(task: Mapping[str, Any], route: Any, parent_agent: Any):
    """Build a no-override policy while leaving the ordinary legacy surface untouched."""
    policy = launch_policy(task, TaskRouteOptions())
    if route.diagnostics:
        from agent.delegation_launch import with_diagnostics
        policy = with_diagnostics(policy, route.diagnostics)
    from agent.delegation_launch import effective_tool_ceiling
    if effective_tool_ceiling(parent_agent) is None:
        return policy
    parent_grant = _parent_tool_grant(parent_agent)
    if parent_grant is None:
        raise ValueError("The owning parent's exact tool grant could not be verified.")
    inherited = _strict_inherited_tool_ceiling(parent_agent, parent_grant)
    return replace(policy, parent_tool_grant=parent_grant, admitted_tool_names=inherited)


def _inherited_external_tool_error(route: Any, parent_agent: Any) -> str | None:
    """Refuse transports that cannot enforce an ancestor's exact tool ceiling."""
    from agent.delegation_launch import effective_tool_ceiling
    if effective_tool_ceiling(parent_agent) is None:
        return None
    credentials = route.creds
    if credentials.get("command"):
        return "An inherited exact tool ceiling cannot be enforced by an external command transport."
    api_mode = credentials.get("api_mode")
    if api_mode is None and not credentials.get("provider"):
        api_mode = getattr(parent_agent, "api_mode", None)
    if api_mode == "codex_app_server":
        return "An inherited exact tool ceiling cannot be enforced by the Codex app-server transport."
    return None


def _apply_legacy_route_policy(task: Mapping[str, Any], option: TaskRouteOptions, route: Any,
                               parent_agent: Any) -> Any:
    error = _inherited_external_tool_error(route, parent_agent)
    if error:
        return _refuse_inherited_external_route(route, task, option, error, parent_agent)
    try:
        route_policy = _legacy_route_policy(task, route, parent_agent)
    except ValueError as exc:
        return _replace_refused(route, task, option, exc, parent_agent)
    if not route.diagnostics and getattr(route_policy, "admitted_tool_names", None) is None:
        return route
    return replace(route, launch_policy=route_policy)


def _refuse_inherited_external_route(route: Any, task: Mapping[str, Any], option: TaskRouteOptions,
                                     error: str, parent_agent: Any) -> Any:
    refused = _replace_refused(route, task, option, ValueError(error), parent_agent)
    return replace(refused, creds={**route.creds, **refused.creds})


def _apply_task_route_options(index: int, task: Mapping[str, Any], option: TaskRouteOptions, route: Any,
                              config: Mapping[str, Any], parent_agent: Any) -> Any:
    """Refuse unrouteable tasks and unsupported inherited exact grants."""
    if not option.any_set:
        return _apply_legacy_route_policy(task, option, route, parent_agent)
    if route.error:
        return _replace_refused(route, task, option, ValueError(route.error), parent_agent)

    def _grant_error_for(option: TaskRouteOptions, selected_route: Any, config: Mapping[str, Any]) -> str | None:
        if not option.tools_set:
            return None
        from tools.tool_search_catalog import BRIDGE_TOOL_NAMES
        requested = option.tools or ()
        return _exact_builtin_tool_grant_error(
            index, option, _parent_tool_grant(
                parent_agent, requested_names=requested,
                include_bridges=bool(set(requested) & BRIDGE_TOOL_NAMES or
                                     any(name.startswith("connectors__") for name in requested)),
            ),
            selected_route, config, parent_agent,
        )

    try:
        tool_grant = getattr(parent_agent, "valid_tool_names", None)
        if tool_grant is None and hasattr(parent_agent, "tools"):
            tool_grant = tuple(
                function.get("name") for item in (parent_agent.tools or ())
                if isinstance(item, Mapping) and isinstance((function := item.get("function")), Mapping)
                and isinstance(function.get("name"), str)
            )
        selected_route, loaded_config = _task_baseline(task, config, route, option)
        if option.profile_set and selected_route.profile_home is None:
            from tools.delegate_tool_routes import _profile_home, _profile_route
            selected_route = _profile_route(_profile_home(option.profile))
            loaded_config = _selected_profile_config(selected_route.profile_home)
        if policy_error := _inherited_external_tool_error(selected_route, parent_agent):
            return _refuse_inherited_external_route(selected_route, task, option, policy_error, parent_agent)
        profile_config = dict(loaded_config)
        if option.tools_set:
            grant_error = _grant_error_for(option, selected_route, profile_config)
            if grant_error:
                raise ValueError(grant_error)
        from tools.delegate_tool_routes import delegated_profile_scope
        with delegated_profile_scope(selected_route.profile_home):
            credentials = dict(selected_route.creds)
            routing = dict(selected_route.routing_cfg)
            if option.route_pinned:
                _preflight_route_pair(option, credentials, profile_config, selected_route, parent_agent, index)
                credentials, resolved_model = _task_model_provider(task, option, credentials, profile_config, index, parent_agent)
                if resolved_model is None:
                    raise ValueError(f"Task {index}: resolved route has no model.")
                model = resolved_model
                provider = str(credentials.get("provider") or getattr(parent_agent, "provider", "") or "")
                if not provider:
                    raise ValueError(f"Task {index}: resolved route has no provider.")
                diagnostics = dict(selected_route.diagnostics)
                _validate_task_route(model, provider, credentials, profile_config, option, routing, diagnostics, index)
            else:
                model = str(credentials.get("model") or getattr(parent_agent, "model", "") or "")
                provider = str(credentials.get("provider") or getattr(parent_agent, "provider", "") or "")
                diagnostics = dict(selected_route.diagnostics)

            effort, policy, reasoning_diagnostics = _resolve_task_reasoning(
                option, model, provider, credentials, selected_route, parent_agent)
            policy = _compose_parent_tool_policy(policy, parent_agent, option)
            policy = replace(
                policy, system_prompt=option.system_prompt, system_prompt_mode=option.system_prompt_mode,
                tools=option.tools, skills=option.skills,
                profile_home=(str(selected_route.profile_home) if selected_route.profile_home is not None else
                              (str(_parent_home(parent_agent)) if option.skills is not None else None)),
                workspace=_workspace_for_parent(parent_agent) if parent_agent is not None else None,
            )
            if option.tools_set:
                policy = replace(policy, admitted_tool_names=option.tools)
            skill_error, _missing_skills, skill_snapshot = _preflight_selected_skills(
                option, selected_route.profile_home, parent_agent)
            if skill_error:
                logger.warning("Delegated skill preflight refused: %s", skill_error)
                raise ValueError(skill_error)
            if option.skills is not None:
                policy = replace(policy, skill_preflight_snapshot=skill_snapshot)
            if reasoning_diagnostics:
                diagnostics.update(reasoning_diagnostics)
                diagnostics["validation"] = "unverified"
                warnings = [str(diagnostics[key]) for key in ("warning", "reasoning_warning") if diagnostics.get(key)]
                diagnostics["warning"] = " ".join(warnings)
            from agent.delegation_launch import with_diagnostics
            if diagnostics:
                policy = with_diagnostics(policy, diagnostics)
            report = dict(diagnostics) if diagnostics else {}
            if option.route_pinned:
                report.update(requested_provider=option.provider, requested_model=option.model,
                              requested_profile=option.profile, resolved_provider=provider,
                              resolved_model=model, validation=diagnostics.get("validation", "verified"))
                diagnostics = report
            if report:
                policy = with_diagnostics(policy, report)
        return replace(selected_route, creds={**credentials, **({"model": model} if option.route_pinned else {})},
                       routing_cfg=routing, reasoning_effort=effort, diagnostics=diagnostics, launch_policy=policy)
    except (ValueError, KeyError, TypeError, AuthError) as exc:
        return _replace_refused(route, task, option, exc, parent_agent)


def resolve_task_routes(task_list: list[dict[str, Any]], config: Mapping[str, Any], parent_agent: Any,
                        routes: list[Any], options: list[TaskRouteOptions], errors: list[str | None],
                        base_route: Any = None) -> list[Any]:
    """Apply options after backend selection without overwriting backend refusals."""
    resolved = []
    for index, (task, option, route, error) in enumerate(zip(task_list, options, routes, errors)):
        if error:
            route = replace(route, error=error, diagnostics={"validation": "refused", "warning": error})
            resolved.append(_replace_refused(route, task, option, ValueError(error), parent_agent))
            continue
        resolved.append(_apply_task_route_options(index, task, option, route, config, parent_agent))
    return resolved
