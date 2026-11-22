"""Contract tests for the native, read-only dotenv-file secret source."""

from __future__ import annotations

import os
import socket
import subprocess
from pathlib import Path

import pytest

from agent.secret_sources.base import ErrorKind
from agent.secret_sources.file import FileSource, MAX_FILE_BYTES
from tests.secret_sources.conformance import SecretSourceConformance

pytestmark = pytest.mark.platforms("posix")


def _private_file(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o600)
    return path


@pytest.fixture(autouse=True)
def _reset_source_state():
    from agent.secret_sources import registry
    from hermes_cli import env_loader

    registry._reset_registry_for_tests()
    env_loader._SECRET_SOURCES.clear()
    env_loader._SECRET_SOURCE_VALUES_BY_HOME.clear()
    env_loader._SECRET_SOURCE_WRITES_BY_HOME.clear()
    env_loader._SOURCE_SUPPLIED_NAMES.clear()
    env_loader.reset_secret_source_cache()
    yield
    env_loader.reset_secret_source_cache()
    env_loader._SECRET_SOURCES.clear()
    env_loader._SECRET_SOURCE_VALUES_BY_HOME.clear()
    env_loader._SECRET_SOURCE_WRITES_BY_HOME.clear()
    env_loader._SOURCE_SUPPLIED_NAMES.clear()
    registry._reset_registry_for_tests()


def test_source_schema_is_recognized_by_hermes_config():
    from hermes_cli.web_server_config import CONFIG_SCHEMA

    schema = FileSource().config_schema()
    assert set(schema) == {"enabled", "path", "override_existing"}
    assert {f"secrets.file.{name}" for name in schema} <= set(CONFIG_SCHEMA)


def test_reads_existing_dotenv_syntax_without_evaluating_it(tmp_path):
    dotenv = _private_file(tmp_path / "shared.env", (
        "# shared values\n"
        "export FILE_SOURCE_KEY=plain-value\n"
        "QUOTED_KEY=\"two words\" # comment\n"
        "SINGLE_KEY='literal $HOME'\n"
    ))

    result = FileSource().fetch({"enabled": True, "path": str(dotenv)}, tmp_path)

    assert result.ok
    assert result.secrets == {
        "FILE_SOURCE_KEY": "plain-value",
        "QUOTED_KEY": "two words",
        "SINGLE_KEY": "literal $HOME",
    }


def test_relative_path_is_profile_home_relative(tmp_path):
    home = tmp_path / "profile"
    dotenv = _private_file(home / "secrets.env", "PROFILE_FILE_KEY=from-profile\n")

    result = FileSource().fetch({"enabled": True, "path": "secrets.env"}, home)

    assert result.ok and result.secrets == {"PROFILE_FILE_KEY": "from-profile"}
    assert dotenv.is_file()


def test_tilde_path_uses_the_current_users_home(monkeypatch, tmp_path):
    user_home = tmp_path / "account"
    dotenv = _private_file(
        user_home / ".config" / "hermes-shared" / "secrets.env", "SHARED_FILE_KEY=from-home\n"
    )
    monkeypatch.setenv("HOME", str(user_home))

    result = FileSource().fetch(
        {"enabled": True, "path": "~/.config/hermes-shared/secrets.env"}, tmp_path / "profile"
    )

    assert result.ok and result.secrets == {"SHARED_FILE_KEY": "from-home"}
    assert dotenv.is_file()


def test_missing_path_is_a_value_free_not_configured_failure(tmp_path):
    result = FileSource().fetch({"enabled": True, "path": str(tmp_path / "absent.env")}, tmp_path)

    assert not result.ok and result.error_kind is ErrorKind.NOT_CONFIGURED
    assert not result.secrets
    assert "absent.env" not in result.error


def test_malformed_path_and_unreadable_input_never_raise(tmp_path):
    source = FileSource()
    for value in (None, [], {}, ""):
        result = source.fetch({"enabled": True, "path": value}, tmp_path)
        assert isinstance(result.error, str) and result.error_kind is not None
        assert not result.secrets

    dotenv = _private_file(tmp_path / "unreadable.env", "SYNTHETIC_PRIVATE_VALUE\n")
    dotenv.chmod(0o644)
    rejected = source.fetch({"enabled": True, "path": str(dotenv)}, tmp_path)
    assert not rejected.ok and rejected.error_kind is ErrorKind.INTERNAL
    assert "SYNTHETIC_PRIVATE_VALUE" not in rejected.error


def test_rejects_symlink_and_non_regular_file_before_open(tmp_path, monkeypatch):
    from agent.secret_sources import file as file_source

    target = _private_file(tmp_path / "target.env", "SYMLINK_PRIVATE_VALUE=secret\n")
    link = tmp_path / "link.env"
    link.symlink_to(target)
    source = FileSource()

    linked = source.fetch({"enabled": True, "path": str(link)}, tmp_path)
    assert not linked.ok and not linked.secrets
    assert "SYMLINK_PRIVATE_VALUE" not in linked.error

    fifo = tmp_path / "pipe.env"
    os.mkfifo(fifo)
    opened = []
    original_open = os.open

    def track_fifo_open(path, *args, **kwargs):
        if isinstance(path, (str, bytes, os.PathLike)) and Path(path) == fifo:
            opened.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(file_source.os, "open", track_fifo_open)
    piped = source.fetch({"enabled": True, "path": str(fifo)}, tmp_path)
    assert not piped.ok and not piped.secrets
    assert not opened, "the source must reject a FIFO from lstat before opening it"


def test_rejects_oversized_file(tmp_path):
    dotenv = _private_file(tmp_path / "large.env", "X" * (MAX_FILE_BYTES + 1))

    result = FileSource().fetch({"enabled": True, "path": str(dotenv)}, tmp_path)

    assert not result.ok and not result.secrets
    assert "SYNTHETIC_PRIVATE_VALUE" not in result.error


def test_file_read_does_not_spawn_or_open_a_network_connection(tmp_path, monkeypatch):
    dotenv = _private_file(tmp_path / "local.env", "LOCAL_FILE_KEY=local\n")
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: pytest.fail("subprocess used"))
    monkeypatch.setattr(socket, "create_connection", lambda *_args, **_kwargs: pytest.fail("network used"))

    result = FileSource().fetch({"enabled": True, "path": str(dotenv)}, tmp_path)

    assert result.ok and result.secrets == {"LOCAL_FILE_KEY": "local"}


def test_builtin_source_lookup_does_not_fetch(monkeypatch):
    from agent.secret_sources import registry

    monkeypatch.setattr(FileSource, "fetch", lambda *_args, **_kwargs: pytest.fail("lookup fetched secrets"))
    registry._reset_registry_for_tests()
    try:
        source = registry.get_source("file")
        assert isinstance(source, FileSource)
    finally:
        registry._reset_registry_for_tests()


def test_hydration_keeps_profile_dotenv_and_adds_shared_file(tmp_path, monkeypatch):
    from agent.secret_sources import registry
    from hermes_cli import env_loader

    home = tmp_path / "home"
    home.mkdir()
    shared = _private_file(tmp_path / "shared.env", "FILE_SHARED_KEY=common\nFILE_ADDITIVE_KEY=shared\n")
    (home / ".env").write_text("FILE_SHARED_KEY=profile\nPROFILE_ONLY_KEY=local\n", encoding="utf-8")
    (home / "config.yaml").write_text(
        f"secrets:\n  file:\n    enabled: true\n    path: {shared}\n    override_existing: false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    for name in ("FILE_SHARED_KEY", "FILE_ADDITIVE_KEY", "PROFILE_ONLY_KEY"):
        monkeypatch.delenv(name, raising=False)
    registry._reset_registry_for_tests()
    env_loader.reset_secret_source_cache(home)

    env_loader.load_hermes_dotenv(hermes_home=home)

    assert os.environ["FILE_SHARED_KEY"] == "profile"
    assert os.environ["FILE_ADDITIVE_KEY"] == "shared"
    assert os.environ["PROFILE_ONLY_KEY"] == "local"
    assert env_loader.get_secret_source("FILE_ADDITIVE_KEY") == "file"
    assert "common" not in (home / "config.yaml").read_text(encoding="utf-8")


def test_reset_source_cache_reads_file_rotation(tmp_path, monkeypatch):
    from agent.secret_sources import registry
    from hermes_cli import env_loader

    home = tmp_path / "home"
    home.mkdir()
    source_file = _private_file(tmp_path / "rotating.env", "FILE_ROTATION_KEY=old-value\n")
    (home / ".env").write_text("# source owns FILE_ROTATION_KEY\n", encoding="utf-8")
    (home / "config.yaml").write_text(
        f"secrets:\n  file:\n    enabled: true\n    path: {source_file}\n    override_existing: false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("FILE_ROTATION_KEY", raising=False)
    registry._reset_registry_for_tests()
    env_loader.reset_secret_source_cache(home)
    env_loader.load_hermes_dotenv(hermes_home=home)
    assert os.environ["FILE_ROTATION_KEY"] == "old-value"

    source_file.write_text("FILE_ROTATION_KEY=new-value\n", encoding="utf-8")
    source_file.chmod(0o600)
    env_loader.reset_secret_source_cache(home)
    env_loader.load_hermes_dotenv(hermes_home=home)

    assert os.environ["FILE_ROTATION_KEY"] == "new-value"


def test_file_source_cannot_reselect_the_config_backend(tmp_path, monkeypatch, capsys):
    from agent.secret_sources import registry
    from hermes_cli import env_loader

    home = tmp_path / "home"
    home.mkdir()
    source_file = _private_file(tmp_path / "bootstrap.env", (
        "HERMES_CONFIG_BACKEND=remote\nFILE_BOOTSTRAP_KEY=available\n"
    ))
    (home / "config.yaml").write_text(
        f"secrets:\n  file:\n    enabled: true\n    path: {source_file}\n    override_existing: true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_CONFIG_BACKEND", "file")
    monkeypatch.delenv("FILE_BOOTSTRAP_KEY", raising=False)
    registry._reset_registry_for_tests()
    env_loader.reset_secret_source_cache(home)

    env_loader._apply_external_secret_sources(home)

    assert os.environ["HERMES_CONFIG_BACKEND"] == "file"
    assert os.environ["FILE_BOOTSTRAP_KEY"] == "available"
    assert env_loader.get_secret_source("HERMES_CONFIG_BACKEND") is None
    assert env_loader.get_secret_source("FILE_BOOTSTRAP_KEY") == "file"
    assert "ignored HERMES_CONFIG_BACKEND" in capsys.readouterr().err


class TestFileSourceConformance(SecretSourceConformance):
    @pytest.fixture
    def source(self) -> FileSource:
        return FileSource()
