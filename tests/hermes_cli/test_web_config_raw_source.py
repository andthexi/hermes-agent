"""Strict raw-source GET behavior for profile config editors and exports."""

from __future__ import annotations

import hermes_yaml as yaml
import pytest
from starlette.testclient import TestClient


@pytest.fixture
def config_api(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    config_path = home / "config.yaml"
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli import config
    config._LOAD_CONFIG_CACHE.clear()
    config._RAW_CONFIG_CACHE.clear()
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    client = TestClient(app, raise_server_exceptions=False)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return client, home, config_path, config


def test_raw_source_returns_exact_mapping_and_keeps_other_get_views(config_api):
    client, _home, config_path, _config = config_api
    config_path.write_text(
        "model:\n  provider: fixture-provider\n  default: fixture-model\n  context_length: 123456\n"
        "display:\n  compact: false\n",
        encoding="utf-8",
    )

    raw = client.get("/api/config?raw_source=true")
    legacy = client.get("/api/config?include_defaults=false")
    effective = client.get("/api/config")

    assert raw.status_code == 200 and raw.json() == yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert legacy.status_code == 200
    assert legacy.json()["model"] == "fixture-model"
    assert legacy.json()["model_context_length"] == 123456
    assert effective.status_code == 200
    assert effective.json()["model"] == "fixture-model"
    assert effective.json()["display"]["compact"] is False


@pytest.mark.parametrize("content", [None, ""], ids=["absent", "empty"])
def test_absent_or_empty_raw_source_remains_an_empty_mapping(config_api, content):
    client, _home, config_path, _config = config_api
    if content is not None:
        config_path.write_text(content, encoding="utf-8")

    response = client.get("/api/config?raw_source=true")

    assert response.status_code == 200
    assert response.json() == {}


def test_malformed_raw_source_is_a_value_free_client_error(config_api):
    client, home, config_path, _config = config_api
    config_path.write_text("private_marker: [unterminated\n", encoding="utf-8")
    original = config_path.read_bytes()

    response = client.get("/api/config?raw_source=true")

    assert response.status_code == 422
    assert response.json()["detail"]
    assert "private_marker" not in response.text
    assert "unterminated" not in response.text
    assert config_path.read_bytes() == original
    assert not (home / "backups" / "config").exists()


@pytest.mark.platforms("posix")
def test_unreadable_raw_source_is_not_reported_as_empty(config_api):
    client, home, config_path, _config = config_api
    config_path.write_text("display:\n  compact: true\n", encoding="utf-8")
    original = config_path.read_bytes()
    config_path.chmod(0)
    try:
        response = client.get("/api/config?raw_source=true")
    finally:
        config_path.chmod(0o600)

    assert response.status_code == 403
    assert response.json()["detail"]
    assert "compact" not in response.text
    assert config_path.read_bytes() == original
    assert not (home / "backups" / "config").exists()


def test_raw_source_repair_read_ignores_invalid_enabled_plugin_defaults(config_api):
    client, home, config_path, _config = config_api
    plugin = home / "plugins" / "fixture-invalid-defaults"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(
        "name: fixture-invalid-defaults\nconfig_defaults: [malformed\n", encoding="utf-8"
    )
    config_path.write_text(
        "plugins:\n  enabled: [fixture-invalid-defaults]\ndisplay:\n  compact: true\n",
        encoding="utf-8",
    )

    response = client.get("/api/config?raw_source=true")

    assert response.status_code == 200, response.text
    assert response.json() == yaml.safe_load(config_path.read_text(encoding="utf-8"))


def test_config_backend_outage_is_an_actionable_value_free_unavailable_response(config_api, monkeypatch):
    client, _home, _config_path, _config = config_api
    from hermes_cli import config_backend

    def unavailable(_path):
        raise config_backend.ConfigBackendUnavailable("plane-secret-marker")

    monkeypatch.setattr(config_backend, "read_config_doc_readonly", unavailable)

    response = client.get("/api/config?raw_source=true")

    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"].lower()
    assert "plane-secret-marker" not in response.text
