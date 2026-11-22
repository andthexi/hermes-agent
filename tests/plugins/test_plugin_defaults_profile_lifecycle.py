"""Profile-bound malformed-default recovery and established-agent stability."""

from pathlib import Path

import pytest

from hermes_cli.config_plugin_defaults import PluginDefaultsError
from hermes_cli import plugins_config_defaults


def _write_plugin(home: Path, *, malformed: bool = False) -> Path:
    plugin_dir = home / "plugins" / "fixture-defaults"
    plugin_dir.mkdir(parents=True)
    text = (
        "name: fixture-defaults\nversion: 0.1.0\nconfig_defaults:\n  version: 1\n"
        "  values:\n    agent:\n      reasoning_effort: high\n"
    )
    if malformed:
        text = "name: fixture-defaults\nconfig_defaults: [unterminated\n"
    (plugin_dir / "plugin.yaml").write_text(text, encoding="utf-8")
    (home / "config.yaml").write_text("plugins:\n  enabled: [fixture-defaults]\n", encoding="utf-8")
    return plugin_dir


def _avoid_real_payload_manifest(monkeypatch) -> None:
    """The checkout parent may contain a live PM manifest; tests never read it."""
    from pm import environments

    monkeypatch.setattr(environments, "_payload_manifest", lambda _root: None)


def test_new_effective_read_rejects_invalid_defaults_without_using_gateway_last_good(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    plugin_dir = _write_plugin(home)
    monkeypatch.setattr(plugins_config_defaults, "_BUNDLED_PLUGINS_DIR", tmp_path / "bundled")
    (tmp_path / "bundled").mkdir()
    _avoid_real_payload_manifest(monkeypatch)
    from gateway import run as gateway_run

    assert gateway_run._load_gateway_config(home / "config.yaml")["agent"]["reasoning_effort"] == "high"
    (plugin_dir / "plugin.yaml").write_text("name: fixture-defaults\nconfig_defaults: [unterminated\n", encoding="utf-8")

    with pytest.raises(PluginDefaultsError, match="fixture-defaults"):
        gateway_run._load_gateway_config(home / "config.yaml")


def test_secondary_profile_initialization_surfaces_invalid_defaults_at_owning_boundary(tmp_path, monkeypatch):
    import asyncio
    broken = tmp_path / "broken"
    broken.mkdir()
    _write_plugin(broken, malformed=True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "launch-home"))
    monkeypatch.setattr(plugins_config_defaults, "_BUNDLED_PLUGINS_DIR", tmp_path / "bundled")
    (tmp_path / "bundled").mkdir()
    _avoid_real_payload_manifest(monkeypatch)

    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)

    with pytest.raises(PluginDefaultsError, match="fixture-defaults"):
        asyncio.run(runner._load_secondary_profile_config("broken", broken))


def test_multiplex_startup_skips_only_profile_with_invalid_defaults(tmp_path, monkeypatch, caplog):
    import asyncio
    broken = tmp_path / "broken"
    healthy = tmp_path / "healthy"
    broken.mkdir()
    healthy.mkdir()
    _write_plugin(broken, malformed=True)
    _write_plugin(healthy)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "launch-home"))
    monkeypatch.setattr(plugins_config_defaults, "_BUNDLED_PLUGINS_DIR", tmp_path / "bundled")
    (tmp_path / "bundled").mkdir()
    _avoid_real_payload_manifest(monkeypatch)

    from gateway import run as gateway_run
    from gateway import run_profile_reconcile
    from gateway.run import GatewayRunner
    from hermes_cli import profiles

    monkeypatch.setattr(gateway_run, "_multiplex_profile_homes", lambda _cfg: [("broken", broken), ("healthy", healthy)])
    monkeypatch.setattr(run_profile_reconcile, "profile_serve_signature", lambda _home: "fixture-signature")
    monkeypatch.setattr(profiles, "profiles_to_serve", lambda *_a, **_kw: [("default", tmp_path / "launch-home")])
    monkeypatch.setattr(profiles, "profile_is_parked", lambda _home: False)
    runner = object.__new__(GatewayRunner)
    runner.config = {}
    runner.adapters = {}
    runner._failed_platforms = {}
    runner._multiplex_on = lambda: True
    runner._primary_profile_name = "default"
    runner._primary_resource_claims = lambda _active: {}
    runner._record_served_profiles = lambda *_args: None
    runner._restore_secondary_completion_ledgers = lambda *_args: None

    async def start_profile(name, profile_home, _claimed):
        if name == "broken":
            await runner._load_secondary_profile_config(name, profile_home)
        return 1

    runner._start_one_profile_adapters = start_profile
    connected = asyncio.run(runner._start_secondary_profile_adapters())

    assert connected == 1
    assert runner._served_profile_signatures == {"healthy": "fixture-signature"}
    assert "broken" in caplog.text


def test_raw_disable_recovers_after_an_invalid_enabled_manifest(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    _write_plugin(home, malformed=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(plugins_config_defaults, "_BUNDLED_PLUGINS_DIR", tmp_path / "bundled")
    (tmp_path / "bundled").mkdir()

    from hermes_cli.config import read_raw_config, save_config
    from hermes_cli.config_effective import load_user_config_effective
    with pytest.raises(PluginDefaultsError):
        load_user_config_effective(home / "config.yaml")
    raw = read_raw_config()
    raw["plugins"]["disabled"] = ["fixture-defaults"]
    save_config(raw)

    effective = load_user_config_effective(home / "config.yaml")
    assert "agent" not in effective
    assert effective["plugins"]["disabled"] == ["fixture-defaults"]


def test_invalid_profile_does_not_leak_or_block_an_unrelated_profile(tmp_path, monkeypatch):
    broken = tmp_path / "broken"
    healthy = tmp_path / "healthy"
    broken.mkdir()
    healthy.mkdir()
    _write_plugin(broken, malformed=True)
    _write_plugin(healthy)
    monkeypatch.setattr(plugins_config_defaults, "_BUNDLED_PLUGINS_DIR", tmp_path / "bundled")
    (tmp_path / "bundled").mkdir()
    from hermes_cli.config_effective import load_user_config_effective

    with pytest.raises(PluginDefaultsError):
        load_user_config_effective(broken / "config.yaml")
    assert load_user_config_effective(healthy / "config.yaml")["agent"]["reasoning_effort"] == "high"


def test_frozen_worker_uses_owner_snapshot_without_manifest_discovery(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    _write_plugin(home, malformed=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(plugins_config_defaults, "_BUNDLED_PLUGINS_DIR", tmp_path / "bundled")
    (tmp_path / "bundled").mkdir()
    from agent import safe_worker_policy

    snapshot = {"agent": {"reasoning_effort": "low"}}
    monkeypatch.setattr(safe_worker_policy, "worker_config_snapshot", lambda: snapshot)
    monkeypatch.setattr(
        plugins_config_defaults, "collect_config_default_layers",
        lambda *_args, **_kwargs: pytest.fail("frozen worker rediscovered plugin manifests"),
    )

    from hermes_cli.config import load_config
    from hermes_cli.config_effective import load_user_config_effective

    assert load_config()["agent"]["reasoning_effort"] == "low"
    assert load_user_config_effective(home / "config.yaml")["agent"]["reasoning_effort"] == "low"


def test_invalid_default_reload_does_not_mutate_established_agent_prefix(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    plugin_dir = _write_plugin(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(plugins_config_defaults, "_BUNDLED_PLUGINS_DIR", tmp_path / "bundled")
    (tmp_path / "bundled").mkdir()
    _avoid_real_payload_manifest(monkeypatch)
    from run_agent import AIAgent

    agent = AIAgent(
        model="fixture-model", provider="custom", base_url="http://127.0.0.1:1/v1",
        api_key="synthetic-not-a-real-key", enabled_toolsets=[], quiet_mode=True,
        skip_context_files=True, skip_memory=True, skip_background_review=True,
    )
    agent._cached_system_prompt = agent._build_system_prompt()
    before = (agent._cached_system_prompt, agent._cached_system_prompt_static)
    assert before[0]
    try:
        (plugin_dir / "plugin.yaml").write_text(
            "name: fixture-defaults\nconfig_defaults: [unterminated\n", encoding="utf-8"
        )
        with pytest.raises(PluginDefaultsError):
            from gateway import run as gateway_run
            gateway_run._load_gateway_config(home / "config.yaml")
        assert (agent._cached_system_prompt, agent._cached_system_prompt_static) == before
    finally:
        agent.close()
