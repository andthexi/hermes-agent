"""Fork session-tree behavior survives the upstream typed-persistence migration."""

import pytest

from persistence.conversation.sessions import SessionCountQuery, SessionQuery, SessionWindow
from persistence.sqlite.conversation._store.messages import append_message
from persistence.sqlite.conversation._store.sessions import create_session, end_session


@pytest.fixture
def store(backend, scope_a):
    return backend.open(scope_a)


def test_tree_roots_counts_and_children_use_typed_repository(store):
    db = store.holder()
    create_session(db, "root", "cli")
    append_message(db, "root", "user", "root question")
    end_session(db, "root", "compression")
    create_session(db, "continued", "cli", parent_session_id="root")
    create_session(db, "branch", "cli", parent_session_id="root",
                   model_config={"_branched_from": "root"})
    create_session(db, "child", "tool", parent_session_id="root",
                   model_config={"_delegate_from": "root"})
    create_session(db, "grandchild", "tool", parent_session_id="child",
                   model_config={"_delegate_from": "child"})
    roots = store.sessions.list_sessions(
        SessionQuery(top_level_only=True, include_children=True,
                     include_child_counts=True, project_compression_tips=False),
        SessionWindow(limit=20), compact_rows=True)
    assert [entry.session.session_id for entry in roots] == ["root"]
    assert roots[0].as_list_row()["child_count"] == 3
    assert store.sessions.count_sessions(SessionCountQuery(top_level_only=True)) == 1
    children = store.sessions.list_children("root")
    rows = {entry.session.session_id: entry.as_list_row() for entry in children}
    assert set(rows) == {"continued", "branch", "child"}
    assert rows["continued"]["kind"] == "compression"
    assert rows["branch"]["kind"] == "branch"
    assert rows["child"]["kind"] == "subagent"
    assert rows["child"]["child_count"] == 1
    assert rows["child"]["has_children"] is True


def test_tree_count_and_listing_share_archived_filters(store):
    db = store.holder()
    create_session(db, "visible", "cli")
    create_session(db, "archived", "cli")
    store.sessions.set_archived("archived", True)
    query = SessionQuery(top_level_only=True, project_compression_tips=False)
    assert [row.session.session_id for row in store.sessions.list_sessions(query)] == ["visible"]
    assert store.sessions.count_sessions(SessionCountQuery(top_level_only=True)) == 1
