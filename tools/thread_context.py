"""Propagate agent-turn context into worker threads that dispatch Hermes tools.

A bare ``threading.Thread`` / ``ThreadPoolExecutor`` worker starts with an empty
``contextvars.Context`` and no thread-local approval/sudo callbacks, so tool dispatch inside it
silently loses the approval ContextVars (gateway sessions then auto-approve dangerous commands)
and the CLI callbacks (``prompt_dangerous_approval`` cannot reach the user, GHSA-qg5c-hvr5-hjgr).
Call :func:`propagate_context_to_thread` **on the parent thread** (it snapshots at call time) and
use the result as the worker target; callbacks are installed for the worker's lifetime and
always cleared on exit.
"""

from __future__ import annotations

import contextvars
import functools
import json
import logging
from contextlib import contextmanager
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)

# None means ordinary legacy execution. An empty immutable set means explicitly deny every
# concrete tool. This ContextVar is execution-owned rather than agent-global so a reused worker
# cannot retain one task's grant for its next call.
_EXACT_TOOL_SCOPE: contextvars.ContextVar[frozenset[str] | None] = contextvars.ContextVar(
    "hermes_exact_tool_scope", default=None,
)


def current_exact_tool_scope() -> frozenset[str] | None:
    """Return the immutable task grant, or None for legacy unrestricted callers."""
    return _EXACT_TOOL_SCOPE.get()


@contextmanager
def exact_tool_scope(allowed_names: Iterable[str] | None):
    """Bind an immutable exact-name grant for a dispatch lifetime; nested grants intersect."""
    frozen = None if allowed_names is None else frozenset(allowed_names)
    ancestor = current_exact_tool_scope()
    if ancestor is not None:
        frozen = ancestor if frozen is None else ancestor & frozen
    token = _EXACT_TOOL_SCOPE.set(frozen)
    try:
        yield frozen
    finally:
        _EXACT_TOOL_SCOPE.reset(token)


def exact_tool_name_block(agent: Any, raw_name: str) -> str | None:
    """Refuse an ungranted literal before alias or parser normalization."""
    scope = current_exact_tool_scope()
    policy = getattr(agent, "_delegation_launch_policy", None)
    if scope is None and policy is not None:
        scope = getattr(policy, "admitted_tool_names", None)
        if scope is None:
            scope = getattr(policy, "tools", None)
    if scope is None or raw_name in scope:
        return None
    if not scope:
        return "Tool call refused: this child was launched with an empty tool grant; no Hermes tools are permitted."
    return f"Tool call refused: {raw_name!r} is outside this child's exact tool grant."


def canonical_tool_name(function_name: str) -> str:
    """Apply a compatibility alias only when the original literal is granted."""
    from model_tools import _LEGACY_TOOL_ALIASES
    scope = current_exact_tool_scope()
    if scope is not None and function_name not in scope:
        return function_name
    return _LEGACY_TOOL_ALIASES.get(function_name, function_name)


def exact_tool_scope_for_agent(agent: Any):
    """Decorate a real executor entry to bind that agent's launch grant for its lifetime."""
    @functools.wraps(agent)
    def scoped(agent_arg: Any, *args: Any, **kwargs: Any):
        policy = getattr(agent_arg, "_delegation_launch_policy", None)
        allowed = None
        if policy is not None:
            allowed = getattr(policy, "admitted_tool_names", None)
            if allowed is None:
                allowed = getattr(policy, "tools", None)
        with exact_tool_scope(allowed):
            if len(args) >= 3 and isinstance(args[0], str) and isinstance(args[1], dict):
                name, arguments, task_id = args[:3]
                if refusal := exact_tool_refusal(name):
                    call_id = kwargs.get("tool_call_id")
                    trace = kwargs.get("tool_request_middleware_trace") or []
                    from agent.inline_tool_executors import emit_terminal_post_tool_call
                    result = json.dumps({"error": refusal}, ensure_ascii=False)
                    emit_terminal_post_tool_call(
                        agent_arg, function_name=name, function_args=arguments, result=result,
                        effective_task_id=task_id, tool_call_id=call_id, status="blocked",
                        error_type="tool_grant_refused", error_message=refusal, middleware_trace=trace,
                    )
                    return result
            return agent(agent_arg, *args, **kwargs)
    return scoped


def exact_tool_refusal(name: str) -> str | None:
    """Refuse an exact out-of-grant concrete registered name without alias repair."""
    scope = current_exact_tool_scope()
    if scope is not None and name.startswith("connectors__"):
        if not scope:
            return "Tool call refused: this task has an empty exact tool grant; no tools are permitted."
        if name in scope and {"tool_call", "manage_connections"}.issubset(scope):
            return None
        return f"Connector entry {name!r} is only callable through an explicitly granted tool_call batch."
    if scope is None or name in scope:
        return None
    if not scope:
        return "Tool call refused: this child was launched with an empty tool grant; no Hermes tools are permitted."
    return f"Tool call refused: {name!r} is outside this child's exact tool grant."


def propagate_context_to_thread(target: Callable) -> Callable:
    """Wrap *target* to run with the *current* thread's ContextVars and per-thread prompt callbacks
    (approval, sudo, password-manager unlock).

    Fail-closed: if callback installation raises they stay ``None`` — dangerous commands are then
    denied by ``prompt_dangerous_approval`` and the gateway approval queue blocks.
    """
    ctx = contextvars.copy_context()
    # (setter, parent callback) pairs; None when the callback API could not be captured.
    installs = None
    try:
        installs = tuple((setter, getter()) for getter, setter in _callback_api())
    except Exception:
        logger.debug("Could not capture parent approval/sudo callbacks", exc_info=True)

    def _runner(*args, **kwargs):
        def _inner():
            if installs is None:
                return target(*args, **kwargs)
            try:
                for setter, cb in installs:
                    if cb is not None:
                        setter(cb)
            except Exception:
                logger.debug("Failed to install propagated approval/sudo callbacks; "
                             "dangerous-command approval will fail closed", exc_info=True)
            try:
                return target(*args, **kwargs)
            finally:
                try:
                    for setter, _cb in installs:
                        setter(None)
                except Exception:
                    logger.debug("Failed to clear propagated approval/sudo callbacks",
                                 exc_info=True)

        return ctx.run(_inner)

    return _runner


def _callback_api():
    """(getter, setter) pairs for every thread-local prompt callback a tool may need mid-dispatch
    (lazy: terminal_tool imports tools.approval at load, so a top-level import risks a cycle).
    Add a new per-thread prompt here — a callback missing from this table is silently absent on
    every parallel/timeout worker, so the tool believes nobody can answer."""
    from agent.vault_backends import unlock as vault_unlock
    from tools import terminal_tool as tt

    return ((tt._get_approval_callback, tt.set_approval_callback),
            (tt._get_sudo_password_callback, tt.set_sudo_password_callback),
            (vault_unlock.get_unlock_prompt_callback, vault_unlock.set_unlock_prompt_callback),
            (vault_unlock.get_save_login_prompt_callback, vault_unlock.set_save_login_prompt_callback),
            (vault_unlock.get_code_prompt_callback, vault_unlock.set_code_prompt_callback),
            # The unlock owner: a manager unlocked on a worker must still be released with its session.
            (vault_unlock.get_current_session_id, vault_unlock.set_current_session_id))
