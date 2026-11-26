"""Tool-call validation for the conversation turn loop: unknown tool names (with
auto-repair and the 3-strike partial exit) and malformed JSON arguments (retry, then
recovery tool results).

Role alternation is preserved on every path: an invalid batch is answered with tool-role
error results (never a user message), and the exits close any open tool-result tail
(#48879). Nothing here imports ``agent.conversation_loop`` at module level (cycle).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent.message_metadata import append_message
from agent.message_sanitization import close_interrupted_tool_sequence, normalize_provider_tool_call_ids
from session_rules.turns import coalesce_tool_call_id
from agent.turn_failure_copy import site_copy, stamp_failure
from hermes_constants import FINISH_REASON_LENGTH

logger = logging.getLogger("agent.conversation_loop")


@dataclass
class ToolValidationVerdict:
    """Outcome of ``validate_tool_calls``.

    ``action``: ``"ok"`` (dispatch the calls), ``"continue"`` (re-issue the API call —
    error results / retry state were recorded) or ``"return"`` (terminal partial
    result in ``result``). ``mixed_invalid_batch`` is True when the batch contains BOTH
    valid and unknown tool names: only the invalid calls get error results, the valid
    ones run."""

    action: str
    result: Optional[dict[str, Any]]
    mixed_invalid_batch: bool
    rejected_tool_results: Optional[list[tuple[Any, str]]] = None
    runnable_tool_calls: Optional[list[Any]] = None


def _preview_name(name: str) -> str:
    return name[:80] + "..." if len(name) > 80 else name


def _append_tool_error_results(messages, tool_calls, content_for) -> None:
    """One tool-role result per call so every tool_call keeps a matching result."""
    for tc in tool_calls:
        append_message(messages, {
            "role": "tool",
            "name": tc.function.name,
            "tool_call_id": coalesce_tool_call_id(tc),
            "content": content_for(tc),
        })


def _mixed_batch_results(tool_calls, accepted_names, refused_ids, agent, tool_grant_refusal,
                         invalid_tool_name_error) -> tuple[list[tuple[Any, str]], list[Any]]:
    """Separate denied/unknown result slots from allowed calls without editing durable messages."""
    rejected = {}
    for tc in tool_calls:
        if id(tc) in refused_ids:
            rejected[id(tc)] = json.dumps(
                {"error": tool_grant_refusal(agent, tc.function.name)}, ensure_ascii=False)
        elif tc.function.name not in accepted_names:
            rejected[id(tc)] = invalid_tool_name_error(tc.function.name, accepted_names)
    return (
        [(tc, rejected[id(tc)]) for tc in tool_calls if id(tc) in rejected],
        [tc for tc in tool_calls if id(tc) not in rejected],
    )


def _route_deferred_call(agent, tc) -> bool:
    """Rewrite a call to a deferred tool by its real name into the ``tool_call`` bridge call, so
    the executor's scope check, schema probe and hooks run on it as on a model-written one."""
    from agent.tool_executor import _canonical_tool_name, _tool_search_scoped_names
    from tools.tool_search_catalog import TOOL_CALL_NAME

    # A legacy alias (``process``) names a deferred tool too, and hints still spell it that way.
    name = _canonical_tool_name(tc.function.name)
    if TOOL_CALL_NAME not in agent.valid_tool_names or name not in _tool_search_scoped_names(agent):
        return False
    args = tc.function.arguments or "{}"
    if isinstance(args, str):
        try:
            args = json.loads(args.strip() or "{}")
        except json.JSONDecodeError:
            return False
    if not isinstance(args, dict):
        return False
    tc.function.name = TOOL_CALL_NAME
    tc.function.arguments = json.dumps({"calls": [{"name": name, "arguments": args}]}, ensure_ascii=False)
    agent._vprint(f"{agent.log_prefix}🔧 Routed deferred tool '{name}' through '{TOOL_CALL_NAME}'",
                  force=True, diagnostic=True)
    return True


def _partial_exit(agent, messages, conversation_history, api_call_count, final_response: str) -> dict[str, Any]:
    """Terminal partial result. Prior retries or an earlier tool batch leave a tool-result
    tail; close it as interrupt aborts do so the next turn is not tool→user (#48879).
    This path never reaches finalize_turn, so persist here."""
    close_interrupted_tool_sequence(messages, final_response)
    agent._persist_session(messages, conversation_history)
    return stamp_failure({
        "final_response": final_response,
        "messages": messages,
        "api_calls": api_call_count,
        "completed": False,
        "partial": True,
        "error": final_response,
    }, "truncated", True)


def validate_tool_calls(
    agent: Any, assistant_message: Any, finish_reason: str, *, messages: list[dict[str, Any]],
    conversation_history: Any, api_call_count: int, effective_task_id: Any,
) -> ToolValidationVerdict:
    """Validate ``assistant_message.tool_calls`` in place (ids uniquified, names
    repaired, dict/empty args normalized to JSON strings). Strikes for invalid names
    advance only when a turn has NO valid call, so a degenerate model still halts at
    3; args cut off mid-stream (routers rewrite ``length`` → ``tool_calls``) are refused
    outright rather than retried."""
    from agent.conversation_loop import _invalid_tool_name_error_content

    _mixed_invalid_batch = False
    tool_calls = assistant_message.tool_calls
    from agent.delegation_launch import exact_permission_names, tool_grant_refusal
    from toolsets import BUILTIN_TOOL_NAMES
    permission_names = exact_permission_names(agent)
    accepted_names = (set(agent.valid_tool_names) | set(permission_names or ())
                      if permission_names is not None else agent.valid_tool_names)
    valid_names = accepted_names

    def _verdict(action: str, result: Optional[dict[str, Any]] = None, *,
                 rejected_tool_results=None, runnable_tool_calls=None) -> ToolValidationVerdict:
        return ToolValidationVerdict(action=action, result=result, mixed_invalid_batch=_mixed_invalid_batch,
                                     rejected_tool_results=rejected_tool_results,
                                     runnable_tool_calls=runnable_tool_calls)

    # Uniquify duplicate tool-call ids BEFORE any downstream consumer: the
    # pre-API sanitizer keeps only the first call/result per id, and the desktop
    # binds each tool card to its call id. Some providers name every call
    # "call_0", so an id already used earlier in the session counts as taken.
    taken = {
        coalesce_tool_call_id(tc)
        for msg in messages if msg.get("role") == "assistant"
        for tc in msg.get("tool_calls") or ()
    }
    agent._uniquify_tool_call_ids(tool_calls, taken=taken)
    normalize_provider_tool_call_ids(tool_calls)

    # Repair mismatched tool names before validating (model hallucinations).
    repaired_ids = set()
    refused_ids = set()
    for tc in tool_calls:
        if permission_names is not None and tc.function.name in BUILTIN_TOOL_NAMES and tc.function.name not in permission_names:
            refused_ids.add(id(tc))
            continue
        if permission_names is not None and tc.function.name in {"tool_search", "tool_describe", "tool_call"} \
                and tc.function.name not in permission_names:
            refused_ids.add(id(tc))
            continue
        if tc.function.name not in accepted_names:
            if permission_names is None and _route_deferred_call(agent, tc):
                repaired_ids.add(id(tc))
                continue
            repaired = agent._repair_tool_call(tc.function.name)
            if repaired:
                agent._vprint(f"{agent.log_prefix}🔧 Auto-repaired tool name: '{tc.function.name}' -> '{repaired}'",
                              force=True, diagnostic=True)
                tc.function.name = repaired
                repaired_ids.add(id(tc))
    # Signed-thinking turns replay the native sidecar blocks, not tool_calls: every rename must reach them.
    if repaired_ids and (provider_data := getattr(assistant_message, "provider_data", None)):
        from agent.tool_call_batches import replace_native_tool_blocks
        replace_native_tool_blocks(provider_data, {coalesce_tool_call_id(tc): [tc] for tc in tool_calls
                                                   if id(tc) in repaired_ids})
    # Counted here, before any exit or normalization, so every emitted call is seen once as the model sent it.
    from hermes_cli.observability.shared_metrics_model import record_tool_call_quality
    record_tool_call_quality(agent, tool_calls, repaired_ids)
    invalid_tool_calls = [tc.function.name for tc in tool_calls if tc.function.name not in accepted_names]
    _mixed_invalid_batch = (
        bool(invalid_tool_calls) and any(tc.function.name in accepted_names for tc in tool_calls)
    ) or (
        bool(refused_ids) and any(
            id(tc) not in refused_ids and tc.function.name in accepted_names for tc in tool_calls
        )
    )
    if invalid_tool_calls and not refused_ids and not any(
        tc.function.name in accepted_names for tc in tool_calls
    ):
        agent._invalid_tool_retries += 1
        # Return helpful error to model — model can agent-correct next turn
        invalid_preview = _preview_name(invalid_tool_calls[0])
        agent._buffer_vprint(f"⚠️  Unknown tool '{invalid_preview}' — sending error to model for agent-correction ({agent._invalid_tool_retries}/3)")

        if agent._invalid_tool_retries >= 3:
            agent._flush_status_buffer()
            agent._vprint(f"{agent.log_prefix}❌ Max retries (3) for invalid tool calls exceeded. Stopping as partial.", force=True, diagnostic=True)
            agent._invalid_tool_retries = 0
            return _verdict("return", _partial_exit(
                agent, messages, conversation_history, api_call_count,
                f"Model generated invalid tool call: {invalid_preview}",
            ))

        append_message(messages, agent._build_assistant_message(assistant_message, finish_reason))
        # See _invalid_tool_name_error_content for the blank-name anti-priming rationale (#47967).
        _append_tool_error_results(
            messages, tool_calls,
            lambda tc: _invalid_tool_name_error_content(tc.function.name, valid_names),
        )
        return _verdict("continue")
    if refused_ids and not _mixed_invalid_batch:
        agent._invalid_tool_retries = 0
        denied = [tc for tc in tool_calls if id(tc) in refused_ids]
        assistant_message.tool_calls = denied
        append_message(messages, agent._build_assistant_message(assistant_message, finish_reason))
        _append_tool_error_results(messages, denied, lambda tc: json.dumps(
            {"error": tool_grant_refusal(agent, tc.function.name)}, ensure_ascii=False))
        assistant_message.tool_calls = [tc for tc in tool_calls if id(tc) not in refused_ids]
        return _verdict("continue")
    # Validate tool call arguments are valid JSON; empty strings become empty
    # objects (common model quirk).
    validation_calls = ([tc for tc in tool_calls if id(tc) not in refused_ids and tc.function.name in valid_names]
                        if _mixed_invalid_batch else tool_calls)
    invalid_json_args = []
    for tc in validation_calls:
        args = tc.function.arguments
        if isinstance(args, (dict, list)):
            tc.function.arguments = json.dumps(args)
            continue
        if args is not None and not isinstance(args, str):
            tc.function.arguments = args = str(args)
        if not args or not args.strip():
            tc.function.arguments = "{}"
            continue
        try:
            json.loads(args)
        except json.JSONDecodeError as e:
            # A mixed-batch invalid-name call never executes (error result later);
            # don't let its broken args trigger the whole-turn JSON retry.
            if not (_mixed_invalid_batch and tc.function.name not in valid_names):
                invalid_json_args.append((tc.function.name, str(e)))

    if invalid_json_args:
        invalid_names = {n for n, _ in invalid_json_args}
        # Routers may rewrite finish_reason "length" → "tool_calls", hiding
        # truncation; args not ending in } or ] (stripped) were cut off
        # mid-stream.
        _truncated = any(
            not (tc.function.arguments or "").rstrip().endswith(("}", "]"))
            for tc in validation_calls if tc.function.name in invalid_names
        )
        if _truncated:
            agent._vprint(
                f"{agent.log_prefix}⚠️  Truncated tool call arguments detected "
                f"(finish_reason={finish_reason!r}) — refusing to execute.",
                force=True, diagnostic=True,
            )
            agent._invalid_json_retries = 0
            agent._cleanup_task_resources(effective_task_id)
            # Blame the output cap only when the model reported one; otherwise the args
            # were cut by a stream break or a router rewriting finish_reason (#91717).
            _copy = (
                site_copy("truncated") if finish_reason == FINISH_REASON_LENGTH
                else site_copy("truncated_unreported")
            )
            return _verdict("return", _partial_exit(
                agent, messages, conversation_history, api_call_count, _copy,
            ))

        agent._invalid_json_retries += 1
        tool_name, error_msg = invalid_json_args[0]
        agent._buffer_vprint(f"⚠️  Invalid JSON in tool call arguments for '{tool_name}': {error_msg}")

        if agent._invalid_json_retries < 3:
            agent._buffer_vprint(f"🔄 Retrying API call ({agent._invalid_json_retries}/3)...")
            # Don't add anything to messages, just retry the API call
            return _verdict("continue")
        # Instead of returning partial, inject tool error results so the model can recover.
        # Using tool results (not user messages) preserves role alternation.
        agent._buffer_vprint("⚠️  Injecting recovery tool results for invalid JSON...")
        agent._invalid_json_retries = 0  # Reset for next attempt
        # Append the assistant message with its (broken) tool_calls, then one
        # error result per call.
        append_message(messages, agent._build_assistant_message(assistant_message, finish_reason))

        def _json_error_result(tc) -> str:
            if tc.function.name not in invalid_names:
                return "Skipped: other tool call in this response had invalid JSON."
            err = next(e for n, e in invalid_json_args if n == tc.function.name)
            return (
                f"Error: Invalid JSON arguments. {err}. "
                f"For tools with no required parameters, use an empty object: {{}}. "
                f"Please retry with valid JSON."
            )

        _append_tool_error_results(messages, validation_calls, _json_error_result)
        return _verdict("continue")

    if _mixed_invalid_batch:
        rejected, valid = _mixed_batch_results(
            tool_calls, accepted_names, refused_ids, agent, tool_grant_refusal,
            _invalid_tool_name_error_content)
        agent._invalid_tool_retries = 0
        return _verdict("ok", rejected_tool_results=rejected, runnable_tool_calls=valid)

    # Reset retry counter on successful JSON validation
    agent._invalid_json_retries = 0
    return _verdict("ok")
