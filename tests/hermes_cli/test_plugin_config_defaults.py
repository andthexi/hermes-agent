"""Behavior contracts for declarative plugin config defaults."""

import copy

import pytest

from hermes_cli.config_plugin_defaults import (
    PluginDefaultsError,
    compose_user_layer,
    parse_default_declaration,
    resolve_default_merge_modes,
)


def declaration(values, merge=None):
    return parse_default_declaration("fixture", {
        "version": 1,
        "values": values,
        "merge": merge or {},
    })


@pytest.mark.parametrize("value", [None, False, 0, "", "high"])
def test_explicit_user_leaf_wins_by_presence(value):
    raw = {"agent": {"reasoning_effort": value}}
    before = copy.deepcopy(raw)
    result = compose_user_layer(raw, [declaration({"agent": {"reasoning_effort": "high"}})])
    assert result["agent"]["reasoning_effort"] == value
    assert raw == before


def test_default_fills_missing_leaf_without_mutating_inputs():
    raw = {"agent": {"max_turns": 8}}
    defaults = declaration({"agent": {"reasoning_effort": "high"}})
    before = copy.deepcopy((raw, defaults))
    result = compose_user_layer(raw, [defaults])
    assert result == {"agent": {"max_turns": 8, "reasoning_effort": "high"}}
    assert (raw, defaults) == before


@pytest.mark.parametrize(
    ("raw_list", "expected"),
    [(None, None), ([], ["shared"]), (["local", "shared"], ["local", "shared"])],
)
def test_append_unique_distinguishes_null_from_empty(raw_list, expected):
    defaults = declaration(
        {"skills": {"external_dirs": ["shared"]}},
        {"skills.external_dirs": "append_unique"},
    )
    result = compose_user_layer({"skills": {"external_dirs": raw_list}}, [defaults])
    assert result["skills"]["external_dirs"] == expected


def test_null_ancestor_blocks_default_descendants():
    result = compose_user_layer(
        {"agent": None}, [declaration({"agent": {"reasoning_effort": "high"}})]
    )
    assert result["agent"] is None


def test_scalar_first_contributor_wins_in_declared_order():
    first = parse_default_declaration("first", {"version": 1, "values": {"agent": {"reasoning_effort": "high"}}})
    second = parse_default_declaration("second", {"version": 1, "values": {"agent": {"reasoning_effort": "low"}}})
    result = compose_user_layer({}, [first, second])
    assert result["agent"]["reasoning_effort"] == "high"


def test_earliest_merge_mode_wins_and_reports_conflicts(caplog):
    first = parse_default_declaration(
        "first", {"version": 1, "values": {"skills": {"external_dirs": ["one"]}},
                  "merge": {"skills.external_dirs": "append_unique"}},
    )
    incompatible = declaration({"skills": {"external_dirs": ["two"]}})
    result = compose_user_layer({}, [first, incompatible])
    assert result["skills"]["external_dirs"] == ["one"]
    assert "merge mode conflict for skills.external_dirs" in caplog.text
    assert "first" in caplog.text and "fixture" in caplog.text


def test_resolved_merge_metadata_uses_first_list_contributor_mode():
    first = parse_default_declaration(
        "first", {"version": 1, "values": {"skills": {"external_dirs": ["one"]}}}
    )
    later = parse_default_declaration(
        "later", {"version": 1, "values": {"skills": {"external_dirs": ["two"]}},
                   "merge": {"skills.external_dirs": "append_unique"}},
    )

    modes = resolve_default_merge_modes([first, later])

    assert modes[("skills", "external_dirs")] == ("first", "replace", True)


def test_append_unique_is_stable_and_type_sensitive():
    first = parse_default_declaration(
        "first",
        {"version": 1, "values": {"skills": {"external_dirs": [False, 0, "a", "a"]}},
         "merge": {"skills.external_dirs": "append_unique"}},
    )
    second = parse_default_declaration(
        "second",
        {"version": 1, "values": {"skills": {"external_dirs": [0, True, "b"]}},
         "merge": {"skills.external_dirs": "append_unique"}},
    )
    result = compose_user_layer({"skills": {"external_dirs": [False, "user"]}}, [first, second])
    assert result["skills"]["external_dirs"] == [False, "user", 0, "a", True, "b"]


def test_known_token_budget_settings_are_ordinary_values():
    parsed = parse_default_declaration("fixture", {
        "version": 1,
        "values": {
            "compression": {"proactive_prune_min_reclaim_tokens": 2048},
            "tools": {"tool_search": {"listing_max_tokens": 6000}},
        },
    })
    assert parsed.values["compression"]["proactive_prune_min_reclaim_tokens"] == 2048
    assert parsed.values["tools"]["tool_search"]["listing_max_tokens"] == 6000


@pytest.mark.parametrize("values", [
    {"agent": {"nonexistent_fixture_setting": "high"}},
    {"model": {"default": []}},
    {"agent": {"max_turns": "many"}},
    {"tools": {"tool_search": {"listing_max_tokens": "a lot"}}},
])
def test_unknown_or_wrong_shaped_known_settings_are_rejected(values):
    with pytest.raises(PluginDefaultsError):
        parse_default_declaration("fixture", {"version": 1, "values": values})


@pytest.mark.parametrize("values", [
    {"agent": {"max_turns": {}}},
    {"skills": {"external_dirs": {}}},
])
def test_empty_mapping_cannot_replace_scalar_or_list_config_setting(values):
    with pytest.raises(PluginDefaultsError):
        parse_default_declaration("fixture", {"version": 1, "values": values})


def test_credential_fields_inside_list_objects_are_rejected():
    with pytest.raises(PluginDefaultsError, match="credential-value"):
        parse_default_declaration("fixture", {
            "version": 1,
            "values": {"fallback_providers": [{
                "provider": "fixture", "model": "fixture-model", "api_key": "SYNTHETIC-NONSECRET",
            }]},
        })


def test_harmless_list_of_config_objects_remains_valid():
    parsed = parse_default_declaration("fixture", {
        "version": 1,
        "values": {"fallback_providers": [{"provider": "openrouter", "model": "vendor/model"}]},
    })
    assert parsed.values["fallback_providers"][0]["model"] == "vendor/model"


def test_unknown_roots_require_an_enabled_plugin_config_key_declaration():
    block = {"version": 1, "values": {"future_section": {"setting": "value"}}}
    with pytest.raises(PluginDefaultsError, match="unknown Hermes config path"):
        parse_default_declaration("fixture", block)
    parsed = parse_default_declaration(
        "fixture", block, known_config_paths={"future_section.setting"}
    )
    assert parsed.values["future_section"]["setting"] == "value"


@pytest.mark.parametrize(
    "block",
    [
        {"version": 2, "values": {}},
        {"version": 1, "values": {"plugins": {"enabled": ["x"]}}},
        {"version": 1, "values": {"model": {"api_key": "not-a-real-secret"}}},
        {"version": 1, "values": {"agent": {"bad": object()}}},
        {"version": 1, "values": {"skills": {"external_dirs": "bad"}},
         "merge": {"skills.external_dirs": "append_unique"}},
    ],
)
def test_invalid_declarations_raise_value_free_diagnostics(block):
    with pytest.raises(PluginDefaultsError) as caught:
        parse_default_declaration("fixture", block)
    assert "fixture" in str(caught.value)
    assert "not-a-real-secret" not in str(caught.value)
