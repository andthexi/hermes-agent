"""Direct child metadata for the dashboard tree after the typed-store migration."""

from typing import Any

from persistence.sqlite._holder import StoreHolder
from persistence.sqlite._state.common import _shape_preview
from persistence.sqlite.conversation._store.core import _session_row_dict
from persistence.sqlite.conversation._store.portability import _compact_session_cols
from persistence.sqlite.conversation._store.sql_common import _PREVIEW_RAW_SELECT, _sql_session_last_active
from persistence.conversation.sessions import parse_model_config


def classify_session_child(child: dict[str, Any], parent: dict[str, Any]) -> str:
    """Explicit branch/delegate markers precede legacy branch and compression fallback."""
    config = parse_model_config(child.get("model_config"))
    parent_id = str(parent.get("id") or "")
    if str(config.get("_delegate_from") or "") == parent_id:
        return "subagent"
    if str(config.get("_branched_from") or "") == parent_id:
        return "branch"
    started, ended = child.get("started_at"), parent.get("ended_at")
    if parent.get("end_reason") == "branched" and started is not None and ended is not None and started >= ended:
        return "branch"
    if parent.get("end_reason") == "compression" and child.get("source") != "tool":
        return "compression"
    return "other"


def list_session_children(
    holder: StoreHolder, parent_session_id: str, *, include_archived: bool = True,
) -> list[dict[str, Any]]:
    from persistence.sqlite.conversation._store.sessions import get_session

    parent = get_session(holder, parent_session_id)
    if not parent:
        return []
    where = "s.parent_session_id = ?" + ("" if include_archived else " AND s.archived = 0")
    query = f"""
        SELECT {_compact_session_cols(holder.db_path.parent)},
            COALESCE(
                (SELECT {_PREVIEW_RAW_SELECT}
                 FROM messages m
                 WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL
                 ORDER BY m.timestamp, m.id LIMIT 1), '') AS _preview_raw,
            {_sql_session_last_active('s')} AS last_active,
            (SELECT COUNT(*) FROM sessions child WHERE child.parent_session_id = s.id) AS child_count
        FROM sessions s WHERE {where} ORDER BY s.started_at ASC, s.id ASC
    """
    result = []
    for raw in holder._read_all(query, [parent_session_id]):
        child = _session_row_dict(raw)
        child["preview"] = _shape_preview(child.pop("_preview_raw", ""))
        child["kind"] = classify_session_child(child, parent)
        child["child_count"] = int(child.get("child_count") or 0)
        child["has_children"] = child["child_count"] > 0
        result.append(child)
    return result
