"""Route connector calls through the normal dispatch policy pipeline."""

import json
from contextlib import nullcontext
from dataclasses import asdict, replace

from tools.registry import tool_error
from tools.connectors.gateway.config import MAX_CALLS_PER_DISPATCH
from tools.connectors.gateway.merge import assemble_results, fill_remote_failure, partition_calls


def dispatch_connector_call(name, arguments, tool_call_id, turn_id=None, task_id=None):
    from tools.thread_context import exact_tool_refusal
    if refusal := exact_tool_refusal(name):
        return tool_error(refusal)
    from tools.connectors.data_use import record_data_use
    from tools.connectors.gateway.bridge import run_remote

    partition = partition_calls([{"name": name, "arguments": arguments}])
    entries = run_remote(partition.remote, tool_call_id, availability=None, client_factory=None, turn_id=turn_id,
                         task_id=task_id)
    entry = entries[0]
    record_data_use(entry.get("data_use") or ())
    return json.dumps({key: value for key, value in entry.items() if key in {"response", "error", "account"}},
                      ensure_ascii=False)


def dispatch_connector_batch(calls, ids, *, user_task, enabled_tools,
                             middleware_trace, enabled_toolsets, disabled_toolsets):
    from model_tools import handle_function_call, suppress_post_tool_call_hook
    from tools.interrupt import is_interrupted

    if len(calls) > MAX_CALLS_PER_DISPATCH:
        return tool_error(f"too many calls: {len(calls)} > max {MAX_CALLS_PER_DISPATCH}. "
                          "Retry with fewer calls per batch.")
    partition = partition_calls(calls)
    from tools.thread_context import current_exact_tool_scope
    exact_scope = current_exact_tool_scope()
    if exact_scope is not None:
        bridge_granted = "tool_call" in exact_scope
        connector_granted = "manage_connections" in exact_scope
        allowed_names = exact_scope if bridge_granted and connector_granted else frozenset()
        denied = [plan for plan in partition.remote if plan.name not in allowed_names]
        allowed = [plan for plan in partition.remote if plan.name in allowed_names]
        entries = list(partition.errors)
        for plan in denied:
            entries.append({"index": plan.position, "name": plan.name, "error": {
                "code": "TOOL_FORBIDDEN",
                "message": "Connector entry is outside this task's exact tool grant.",
            }})
        partition = replace(partition, remote=tuple(allowed), errors=tuple(entries))
    if partition.local:
        from tools.tool_search_validation import local_batch_error
        return tool_error(local_batch_error(calls))
    entries = list(partition.errors)
    for offset, plan in enumerate(partition.remote):
        if is_interrupted():
            # Check before every entry so /stop prevents unstarted remote side effects.
            entries.extend(fill_remote_failure(
                partition.remote[offset:], "Stopped by the user before this call was made.",
                code="INTERRUPTED"))
            break
        # Each entry is its own call: its own policy, middleware, hooks and call id (``<id>#<position>``),
        # so observers never fold an entry into the wrapper or into a sibling entry.
        entry_ids = replace(ids, tool_call_id=f"{ids.tool_call_id}#{plan.position}") if ids.tool_call_id else ids
        from tools.thread_context import exact_tool_scope
        with exact_tool_scope(exact_scope) if exact_scope is not None else nullcontext():
            with suppress_post_tool_call_hook(False):
                payload = handle_function_call(
                    plan.name, plan.arguments, **asdict(entry_ids), user_task=user_task,
                    enabled_tools=enabled_tools, tool_request_middleware_trace=list(middleware_trace),
                    skip_pre_tool_call_hook=False, skip_tool_request_middleware=False,
                    skip_tool_execution_middleware=False,
                    enabled_toolsets=enabled_toolsets, disabled_toolsets=disabled_toolsets,
                )
        try:
            value = json.loads(payload) if isinstance(payload, str) else payload
        except ValueError:
            value = payload
        entry = {"index": plan.position, "name": plan.name}
        if isinstance(value, dict) and "error" in value:
            error = value["error"]
            entry["error"] = error if isinstance(error, dict) else {"code": "TOOL_ERROR", "message": str(error)}
        else:
            entry["response"] = value.get("response", value) if isinstance(value, dict) else value
        if isinstance(value, dict) and value.get("account"):
            entry["account"] = value["account"]
        entries.append(entry)
    return json.dumps(assemble_results(len(calls), entries), ensure_ascii=False)
