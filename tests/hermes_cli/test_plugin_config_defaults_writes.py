"""Raw/effective config API write-intent contracts for plugin defaults."""

from pathlib import Path

import hermes_yaml as yaml
import pytest
from starlette.testclient import TestClient


def _plugin(home: Path, *, scalar: str = "high") -> Path:
    directory = home / "plugins" / "fixture-defaults"
    directory.mkdir(parents=True)
    (directory / "plugin.yaml").write_text(chr(10).join([
        "name: fixture-defaults", "version: 0.1.0", "config_defaults:", "  version: 1", "  values:",
        "    agent:", f"      reasoning_effort: {scalar}", "    skills:", "      external_dirs:",
        "        - shared-skills", "  merge:", "    skills.external_dirs: append_unique",
    ]) + chr(10), encoding="utf-8")
    return directory


@pytest.fixture
def config_api(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    _plugin(home)
    config_path = home / "config.yaml"
    config_path.write_text("plugins:\n  enabled: [fixture-defaults]\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli import config
    config._LOAD_CONFIG_CACHE.clear()
    config._RAW_CONFIG_CACHE.clear()
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN
    client = TestClient(app, raise_server_exceptions=False)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return client, home, config_path


def test_legacy_raw_get_stays_web_normalized_while_new_raw_source_view_is_exact(config_api):
    client, _home, config_path = config_api
    config_path.write_text(chr(10).join([
        "model:", "  provider: fixture-provider", "  default: fixture-model", "  context_length: 123456",
    ]) + chr(10), encoding="utf-8")

    legacy = client.get("/api/config?include_defaults=false")
    exact = client.get("/api/config?raw_source=true")

    assert legacy.status_code == 200 and exact.status_code == 200
    assert legacy.json()["model"] == "fixture-model"
    assert legacy.json()["model_context_length"] == 123456
    assert exact.json()["model"] == {
        "provider": "fixture-provider", "default": "fixture-model", "context_length": 123456,
    }
    assert "model_context_length" not in exact.json()


def test_effective_read_followed_by_legacy_full_document_put_refuses_atomically(config_api):
    client, _home, config_path = config_api
    raw_before = config_path.read_bytes()

    effective = client.get("/api/config")
    assert effective.status_code == 200
    assert effective.json()["agent"]["reasoning_effort"] == "high"
    raw_read = client.get("/api/config?include_defaults=false")
    assert raw_read.status_code == 200
    assert "agent" not in raw_read.json()

    response = client.put("/api/config", json={"config": effective.json()})

    assert response.status_code == 409
    assert "write_mode='explicit_patch'" in response.json()["detail"]
    assert config_path.read_bytes() == raw_before


def test_schema_merge_metadata_is_profile_scoped_and_uses_the_first_mode(config_api):
    client, home, config_path = config_api
    replacement = home / "plugins" / "a-replace"
    append = home / "plugins" / "z-append"
    replacement.mkdir()
    append.mkdir()
    (replacement / "plugin.yaml").write_text(chr(10).join([
        "name: a-replace", "version: 0.1.0", "config_defaults:", "  version: 1", "  values:",
        "    skills:", "      external_dirs: [replacement]",
    ]) + chr(10), encoding="utf-8")
    (replacement / "__init__.py").write_text("def register(ctx):" + chr(10) + "    pass" + chr(10), encoding="utf-8")
    (append / "plugin.yaml").write_text(chr(10).join([
        "name: z-append", "version: 0.1.0", "config_defaults:", "  version: 1", "  values:",
        "    skills:", "      external_dirs: [append]", "  merge:",
        "    skills.external_dirs: append_unique",
    ]) + chr(10), encoding="utf-8")
    (append / "__init__.py").write_text("def register(ctx):" + chr(10) + "    pass" + chr(10), encoding="utf-8")
    config_path.write_text(chr(10).join(["plugins:", "  enabled: [a-replace, z-append]"]) + chr(10), encoding="utf-8")
    from hermes_cli import config
    config._LOAD_CONFIG_CACHE.clear()
    config._RAW_CONFIG_CACHE.clear()

    first = client.get("/api/config/schema")
    assert first.status_code == 200, first.text
    assert first.json()["fields"]["skills.external_dirs"]["merge_mode"] == "replace"
    from hermes_cli.config_effective import load_user_config_effective
    assert load_user_config_effective()["skills"]["external_dirs"] == ["replacement"]

    config_path.write_text(chr(10).join(["plugins:", "  enabled: []"]) + chr(10), encoding="utf-8")
    config._LOAD_CONFIG_CACHE.clear()
    config._RAW_CONFIG_CACHE.clear()
    second = client.get("/api/config/schema")
    assert second.status_code == 200, second.text
    assert "plugin_default" not in second.json()["fields"]["skills.external_dirs"]


def test_schema_reports_list_merge_mode_without_exposing_default_values(config_api):
    client, _home, _config_path = config_api

    response = client.get("/api/config/schema")

    assert response.status_code == 200, response.text
    field = response.json()["fields"]["skills.external_dirs"]
    assert field["merge_mode"] == "append_unique"
    assert "profile entries" in field["merge_description"]
    assert "shared-skills" not in field["merge_description"]


def test_explicit_sparse_patch_does_not_materialize_inherited_default(config_api):
    client, _home, config_path = config_api

    response = client.put("/api/config", json={
        "write_mode": "explicit_patch", "config": {"display": {"compact": True}},
    })

    assert response.status_code == 200, response.text
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert raw["display"]["compact"] is True
    assert "agent" not in raw and "skills" not in raw


def test_explicit_value_equal_to_default_is_persisted_and_survives_manifest_change(config_api):
    client, home, config_path = config_api
    response = client.put("/api/config", json={
        "write_mode": "explicit_patch", "config": {"agent": {"reasoning_effort": "high"}},
    })
    assert response.status_code == 200, response.text
    assert yaml.safe_load(config_path.read_text(encoding="utf-8"))["agent"]["reasoning_effort"] == "high"

    manifest = home / "plugins" / "fixture-defaults" / "plugin.yaml"
    manifest.write_text(manifest.read_text(encoding="utf-8").replace("reasoning_effort: high", "reasoning_effort: low"),
                        encoding="utf-8")
    from hermes_cli.config import load_config
    assert load_config()["agent"]["reasoning_effort"] == "high"


def test_save_config_rebases_effective_reads_onto_raw_without_plugin_materialization(config_api):
    _client, _home, config_path = config_api
    from hermes_cli.config import load_config, save_config
    config = load_config()
    config["display"]["compact"] = True

    save_config(config)

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert raw["display"]["compact"] is True
    assert "agent" not in raw and "skills" not in raw


def test_dashboard_aux_assignment_persists_only_selected_fields(config_api):
    _client, _home, config_path = config_api
    from hermes_cli.web_server_config import _apply_model_assignment_sync

    _apply_model_assignment_sync("auxiliary", "nous", "hermes-4", "compression", "")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert raw["auxiliary"]["compression"]["provider"] == "nous"
    assert raw["auxiliary"]["compression"]["model"] == "hermes-4"
    assert "agent" not in raw and "skills" not in raw


def test_save_config_does_not_resurrect_explicitly_deleted_plugin_override(config_api):
    _client, _home, config_path = config_api
    config_path.write_text(chr(10).join([
        "plugins:", "  enabled: [fixture-defaults]", "agent:", "  reasoning_effort: max",
    ]) + chr(10), encoding="utf-8")
    from hermes_cli import config
    config._LOAD_CONFIG_CACHE.clear()
    config._RAW_CONFIG_CACHE.clear()
    effective = config.load_config()
    del effective["agent"]
    config.save_config(effective)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert "agent" not in raw
    from hermes_cli.config_effective import load_user_config_effective
    assert load_user_config_effective()["agent"]["reasoning_effort"] == "high"


def test_raw_source_removal_of_null_ancestor_reveals_default(config_api):
    _client, _home, config_path = config_api
    config_path.write_text(chr(10).join(["plugins:", "  enabled: [fixture-defaults]", "agent: null"]) + chr(10),
                           encoding="utf-8")
    from hermes_cli.config import read_raw_config, save_config
    raw = read_raw_config()
    raw.pop("agent")

    save_config(raw)

    assert "agent" not in yaml.safe_load(config_path.read_text(encoding="utf-8"))
    from hermes_cli.config_effective import load_user_config_effective
    assert load_user_config_effective()["agent"]["reasoning_effort"] == "high"


def test_rest_unset_of_null_ancestor_reveals_default(config_api):
    client, _home, config_path = config_api
    config_path.write_text(chr(10).join(["plugins:", "  enabled: [fixture-defaults]", "agent: null"]) + chr(10),
                           encoding="utf-8")
    response = client.put("/api/config", json={
        "write_mode": "explicit_patch", "config": {}, "unset_paths": ["agent"],
    })

    assert response.status_code == 200, response.text
    assert "agent" not in yaml.safe_load(config_path.read_text(encoding="utf-8"))
    from hermes_cli.config_effective import load_user_config_effective
    assert load_user_config_effective()["agent"]["reasoning_effort"] == "high"


def test_save_config_preserves_scalar_ancestor_while_unrelated_setting_changes(config_api):
    _client, _home, config_path = config_api
    config_path.write_text(chr(10).join(["plugins:", "  enabled: [fixture-defaults]", "agent: malformed-section"]) + chr(10),
                           encoding="utf-8")
    from hermes_cli import config
    config._LOAD_CONFIG_CACHE.clear()
    config._RAW_CONFIG_CACHE.clear()
    effective = config.load_config()
    effective["display"]["compact"] = True

    config.save_config(effective)

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert raw["agent"] == "malformed-section"
    assert raw["display"]["compact"] is True


def test_rest_unset_of_scalar_ancestor_reveals_plugin_default(config_api):
    client, _home, config_path = config_api
    config_path.write_text(chr(10).join(["plugins:", "  enabled: [fixture-defaults]", "agent: malformed-section"]) + chr(10),
                           encoding="utf-8")
    response = client.put("/api/config", json={
        "write_mode": "explicit_patch", "config": {}, "unset_paths": ["agent"],
    })

    assert response.status_code == 200, response.text
    assert "agent" not in yaml.safe_load(config_path.read_text(encoding="utf-8"))
    from hermes_cli.config_effective import load_user_config_effective
    assert load_user_config_effective()["agent"]["reasoning_effort"] == "high"


def test_save_config_preserves_null_ancestor_while_unrelated_setting_changes(config_api):
    _client, _home, config_path = config_api
    config_path.write_text(chr(10).join(["plugins:", "  enabled: [fixture-defaults]", "agent: null"]) + chr(10),
                           encoding="utf-8")
    from hermes_cli import config
    config._LOAD_CONFIG_CACHE.clear()
    config._RAW_CONFIG_CACHE.clear()
    effective = config.load_config()
    effective["display"]["compact"] = True

    config.save_config(effective)

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert raw["agent"] is None
    assert raw["display"]["compact"] is True


def test_save_config_preserves_an_explicit_equal_plugin_default_path(config_api):
    _client, _home, config_path = config_api
    from hermes_cli.config import load_config, save_config
    config = load_config()
    config["agent"]["reasoning_effort"] = "high"

    save_config(config, preserve_keys={("agent", "reasoning_effort")})

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert raw["agent"]["reasoning_effort"] == "high"
    assert "shared-skills" not in str(raw.get("skills"))


def test_explicit_list_patch_persists_only_user_contribution(config_api):
    client, _home, config_path = config_api

    response = client.put("/api/config", json={
        "write_mode": "explicit_patch", "config": {"skills": {"external_dirs": ["local-skills"]}},
    })

    assert response.status_code == 200, response.text
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert raw["skills"]["external_dirs"] == ["local-skills"]
    from hermes_cli.config_effective import load_user_config_effective
    assert load_user_config_effective()["skills"]["external_dirs"] == ["local-skills", "shared-skills"]


def test_null_patch_blocks_plugin_list_and_persists_null(config_api):
    client, _home, config_path = config_api

    response = client.put("/api/config", json={
        "write_mode": "explicit_patch", "config": {"skills": {"external_dirs": None}},
    })

    assert response.status_code == 200, response.text
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert raw["skills"]["external_dirs"] is None
    from hermes_cli.config_effective import load_user_config_effective
    assert load_user_config_effective()["skills"]["external_dirs"] is None


def test_reset_unset_path_reveals_the_current_plugin_default(config_api):
    client, _home, config_path = config_api
    response = client.put("/api/config", json={
        "write_mode": "explicit_patch", "config": {"agent": {"reasoning_effort": "low"}},
    })
    assert response.status_code == 200, response.text

    reset = client.put("/api/config", json={
        "write_mode": "explicit_patch", "config": {}, "unset_paths": ["agent.reasoning_effort"],
    })

    assert reset.status_code == 200, reset.text
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert "agent" not in raw
    from hermes_cli.config_effective import load_user_config_effective
    assert load_user_config_effective()["agent"]["reasoning_effort"] == "high"


def test_legacy_put_without_defaults_keeps_its_existing_behavior(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    config_path = home / "config.yaml"
    config_path.write_text("plugins:\n  enabled: []\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli import config
    config._LOAD_CONFIG_CACHE.clear()
    config._RAW_CONFIG_CACHE.clear()
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN
    client = TestClient(app, raise_server_exceptions=False)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN

    response = client.put("/api/config", json={"config": {"display": {"compact": True}}})

    assert response.status_code == 200, response.text
    assert yaml.safe_load(config_path.read_text(encoding="utf-8"))["display"]["compact"] is True
