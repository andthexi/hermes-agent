"""Bootstrap-safe, profile-bound manifest defaults discovery contracts."""

from pathlib import Path

import pytest

from hermes_cli import plugins_config_defaults
from hermes_cli.config_plugin_defaults import PluginDefaultsError
from hermes_cli.plugins_config_defaults import collect_config_default_layers


def _write_plugin(root: Path, key: str, *, name: str | None = None, depends=None, effort="high") -> Path:
    plugin_dir = root / key
    plugin_dir.mkdir(parents=True)
    display_name = name or key.rsplit("/", 1)[-1]
    dependencies = "" if not depends else "requires_plugins:\n" + "".join(f"  - {item}\n" for item in depends)
    (plugin_dir / "plugin.yaml").write_text(
        "name: " + display_name + "\n"
        "kind: model-provider\n"
        "version: 0.1.0\n"
        + dependencies
        + "config_defaults:\n"
        "  version: 1\n"
        "  values:\n"
        f"    agent:\n      reasoning_effort: {effort}\n",
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text("raise RuntimeError('manifest scan imported plugin code')\n", encoding="utf-8")
    return plugin_dir


def test_effective_loader_composes_manifest_defaults_under_raw_user_values(tmp_path, monkeypatch):
    bundled = tmp_path / "bundled"
    home = tmp_path / "home"
    bundled.mkdir()
    home.mkdir()
    _write_plugin(home / "plugins", "fixture")
    config_path = home / "config.yaml"
    config_path.write_text(
        "plugins:\n  enabled: [fixture]\nagent:\n  max_turns: 17\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(plugins_config_defaults, "_BUNDLED_PLUGINS_DIR", bundled)

    from hermes_cli.config_effective import load_user_config_effective
    effective = load_user_config_effective(config_path)

    assert effective["agent"] == {"max_turns": 17, "reasoning_effort": "high"}
    assert "reasoning_effort" not in config_path.read_text(encoding="utf-8")


def test_full_and_readonly_loaders_merge_plugin_defaults_before_user_values(tmp_path, monkeypatch):
    bundled = tmp_path / "bundled"
    home = tmp_path / "home"
    bundled.mkdir()
    home.mkdir()
    _write_plugin(home / "plugins", "fixture")
    config_path = home / "config.yaml"
    config_path.write_text("plugins:\n  enabled: [fixture]\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(plugins_config_defaults, "_BUNDLED_PLUGINS_DIR", bundled)
    from hermes_cli import config
    config._LOAD_CONFIG_CACHE.clear()
    config._RAW_CONFIG_CACHE.clear()

    full = config.load_config()
    readonly = config.load_config_readonly()

    assert full["agent"]["reasoning_effort"] == "high"
    assert readonly["agent"]["reasoning_effort"] == "high"
    assert "reasoning_effort" not in config_path.read_text(encoding="utf-8")


def test_cli_loader_uses_plugin_defaults_but_ignore_user_config_does_not(tmp_path, monkeypatch):
    bundled = tmp_path / "bundled"
    home = tmp_path / "home"
    bundled.mkdir()
    home.mkdir()
    _write_plugin(home / "plugins", "fixture")
    (home / "config.yaml").write_text("plugins:\n  enabled: [fixture]\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(plugins_config_defaults, "_BUNDLED_PLUGINS_DIR", bundled)
    from pm import environments
    monkeypatch.setattr(environments, "_payload_manifest", lambda _root: None)
    import cli
    monkeypatch.setattr(cli, "_hermes_home", home)
    from hermes_cli.cli_config_load import load_cli_config

    assert load_cli_config()["agent"]["reasoning_effort"] == "high"
    monkeypatch.setenv("HERMES_IGNORE_USER_CONFIG", "1")
    assert load_cli_config()["agent"]["reasoning_effort"] != "high"


def test_secret_source_bootstrap_reads_enabled_manifest_defaults_without_materializing(tmp_path, monkeypatch):
    bundled = tmp_path / "bundled"
    home = tmp_path / "home"
    bundled.mkdir()
    home.mkdir()
    plugin_dir = _write_plugin(home / "plugins", "fixture")
    (plugin_dir / "plugin.yaml").write_text(chr(10).join([
        "name: fixture", "version: 0.1.0", "config_defaults:", "  version: 1", "  values:",
        "    secrets:", "      file:", "        enabled: true",
        "        path: ~/.config/hermes-shared/secrets.env",
        "        override_existing: false",
    ]) + chr(10), encoding="utf-8")
    config_path = home / "config.yaml"
    config_path.write_text(chr(10).join(["plugins:", "  enabled: [fixture]"]) + chr(10), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(plugins_config_defaults, "_BUNDLED_PLUGINS_DIR", bundled)

    from hermes_cli.env_loader import _load_secrets_config
    secrets = _load_secrets_config(home)

    assert secrets["file"]["enabled"] is True
    assert secrets["file"]["path"] == "~/.config/hermes-shared/secrets.env"
    assert secrets["file"]["override_existing"] is False
    assert "secrets:" not in config_path.read_text(encoding="utf-8")


def test_enabled_syntax_broken_manifest_fails_with_value_free_diagnostic(tmp_path, monkeypatch):
    bundled = tmp_path / "bundled"
    home = tmp_path / "home"
    bundled.mkdir()
    home.mkdir()
    plugin_dir = home / "plugins" / "fixture"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        "name: fixture\\nversion: [broken\\nconfig_defaults:\\n", encoding="utf-8"
    )
    monkeypatch.setattr(plugins_config_defaults, "_BUNDLED_PLUGINS_DIR", bundled)

    with pytest.raises(PluginDefaultsError) as caught:
        collect_config_default_layers(home, {"plugins": {"enabled": ["fixture"]}}, {})

    assert "fixture" in str(caught.value)
    assert "broken" not in str(caught.value)


def test_disabled_syntax_broken_manifest_does_not_block_profile(tmp_path, monkeypatch):
    bundled = tmp_path / "bundled"
    home = tmp_path / "home"
    bundled.mkdir()
    home.mkdir()
    plugin_dir = home / "plugins" / "fixture"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text("name: fixture\\nconfig_defaults: [broken\\n", encoding="utf-8")
    monkeypatch.setattr(plugins_config_defaults, "_BUNDLED_PLUGINS_DIR", bundled)

    layers, _signature = collect_config_default_layers(
        home, {"plugins": {"enabled": ["fixture"], "disabled": ["fixture"]}}, {}
    )

    assert layers == []


def test_valid_user_source_shadows_broken_bundled_manifest(tmp_path, monkeypatch):
    bundled = tmp_path / "bundled"
    home = tmp_path / "home"
    bundled.mkdir()
    home.mkdir()
    broken = bundled / "fixture"
    broken.mkdir()
    (broken / "plugin.yaml").write_text("name: fixture\\nconfig_defaults: [broken\\n", encoding="utf-8")
    _write_plugin(home / "plugins", "fixture", effort="high")
    monkeypatch.setattr(plugins_config_defaults, "_BUNDLED_PLUGINS_DIR", bundled)

    layers, _signature = collect_config_default_layers(home, {"plugins": {"enabled": ["fixture"]}}, {})

    assert len(layers) == 1
    assert layers[0].values["agent"]["reasoning_effort"] == "high"


def test_collect_reads_enabled_model_provider_manifest_without_importing_plugin(tmp_path, monkeypatch):
    bundled = tmp_path / "bundled"
    home = tmp_path / "home"
    bundled.mkdir()
    home.mkdir()
    plugin_dir = _write_plugin(home / "plugins" / "model-providers", "fixture")
    monkeypatch.setattr(plugins_config_defaults, "_BUNDLED_PLUGINS_DIR", bundled)

    layers, signature = collect_config_default_layers(
        home, {"plugins": {"enabled": ["fixture"]}}, {}
    )

    assert [layer.plugin_key for layer in layers] == ["model-providers/fixture"]
    assert layers[0].values["agent"]["reasoning_effort"] == "high"
    assert signature
    assert "fixture" not in __import__("sys").modules
    assert plugin_dir.joinpath("__init__.py").exists()


def test_disabled_list_wins_and_model_provider_requires_explicit_enablement(tmp_path, monkeypatch):
    bundled = tmp_path / "bundled"
    home = tmp_path / "home"
    bundled.mkdir()
    home.mkdir()
    _write_plugin(home / "plugins" / "model-providers", "fixture")
    monkeypatch.setattr(plugins_config_defaults, "_BUNDLED_PLUGINS_DIR", bundled)

    assert collect_config_default_layers(home, {"plugins": {}}, {})[0] == []
    assert collect_config_default_layers(
        home, {"plugins": {"enabled": ["fixture"], "disabled": ["fixture"]}}, {}
    )[0] == []


def test_managed_bootstrap_enablement_wins_over_user_disable(tmp_path, monkeypatch):
    bundled = tmp_path / "bundled"
    home = tmp_path / "home"
    bundled.mkdir()
    home.mkdir()
    _write_plugin(home / "plugins", "fixture")
    monkeypatch.setattr(plugins_config_defaults, "_BUNDLED_PLUGINS_DIR", bundled)

    layers, _ = collect_config_default_layers(
        home,
        {"plugins": {"enabled": [], "disabled": ["fixture"]}},
        {"plugins": {"enabled": ["fixture"], "disabled": []}},
    )

    assert [layer.plugin_key for layer in layers] == ["fixture"]


def test_same_name_user_manifest_replaces_bundled_before_defaults_are_read(tmp_path, monkeypatch):
    bundled = tmp_path / "bundled"
    home = tmp_path / "home"
    bundled.mkdir()
    home.mkdir()
    _write_plugin(bundled, "fixture", effort="low")
    _write_plugin(home / "plugins", "fixture", effort="high")
    monkeypatch.setattr(plugins_config_defaults, "_BUNDLED_PLUGINS_DIR", bundled)

    layers, _ = collect_config_default_layers(home, {"plugins": {"enabled": ["fixture"]}}, {})

    assert len(layers) == 1
    assert layers[0].values["agent"]["reasoning_effort"] == "high"


def test_enabled_invalid_declaration_fails_without_echoing_value(tmp_path, monkeypatch):
    bundled = tmp_path / "bundled"
    home = tmp_path / "home"
    bundled.mkdir()
    home.mkdir()
    plugin_dir = _write_plugin(home / "plugins", "fixture")
    manifest_path = plugin_dir / "plugin.yaml"
    manifest_path.write_text(
        "name: fixture\nversion: 0.1.0\nconfig_defaults: [sensitive-fake-marker]\n", encoding="utf-8"
    )
    monkeypatch.setattr(plugins_config_defaults, "_BUNDLED_PLUGINS_DIR", bundled)

    try:
        collect_config_default_layers(home, {"plugins": {"enabled": ["fixture"]}}, {})
    except PluginDefaultsError as exc:
        assert "fixture" in str(exc)
        assert "sensitive-fake-marker" not in str(exc)
    else:
        raise AssertionError("invalid enabled declaration was ignored")


def test_unchanged_enabled_manifest_cache_skips_yaml_discovery(tmp_path, monkeypatch):
    bundled = tmp_path / "bundled"
    home = tmp_path / "home"
    bundled.mkdir()
    home.mkdir()
    _write_plugin(home / "plugins", "fixture")
    monkeypatch.setattr(plugins_config_defaults, "_BUNDLED_PLUGINS_DIR", bundled)
    from hermes_cli import plugins_discovery
    original_scan = plugins_discovery.scan_directory
    calls = []

    def count_scans(*args, **kwargs):
        calls.append(args[0])
        return original_scan(*args, **kwargs)

    monkeypatch.setattr(plugins_discovery, "scan_directory", count_scans)
    enabled = {"plugins": {"enabled": ["fixture"]}}
    first, first_signature = collect_config_default_layers(home, enabled, {})
    first_scan_count = len(calls)

    second, second_signature = collect_config_default_layers(home, enabled, {})

    assert first[0].values == second[0].values
    assert first_signature == second_signature
    assert first_scan_count > 0
    assert len(calls) == first_scan_count


def test_manifest_removal_invalidates_a_contributor_cache(tmp_path, monkeypatch):
    bundled = tmp_path / "bundled"
    home = tmp_path / "home"
    bundled.mkdir()
    home.mkdir()
    plugin_dir = _write_plugin(home / "plugins", "fixture")
    monkeypatch.setattr(plugins_config_defaults, "_BUNDLED_PLUGINS_DIR", bundled)
    enabled = {"plugins": {"enabled": ["fixture"]}}

    first, first_signature = collect_config_default_layers(home, enabled, {})
    (plugin_dir / "plugin.yaml").unlink()
    second, second_signature = collect_config_default_layers(home, enabled, {})

    assert first and not second
    assert first_signature != second_signature


def test_enabled_manifest_cache_isolated_between_homes(tmp_path, monkeypatch):
    bundled = tmp_path / "bundled"
    home_a = tmp_path / "home-a"
    home_b = tmp_path / "home-b"
    bundled.mkdir()
    home_a.mkdir()
    home_b.mkdir()
    _write_plugin(home_a / "plugins", "fixture", effort="high")
    _write_plugin(home_b / "plugins", "fixture", effort="low")
    monkeypatch.setattr(plugins_config_defaults, "_BUNDLED_PLUGINS_DIR", bundled)
    enabled = {"plugins": {"enabled": ["fixture"]}}

    layers_a, _ = collect_config_default_layers(home_a, enabled, {})
    layers_b, _ = collect_config_default_layers(home_b, enabled, {})
    layers_a_again, _ = collect_config_default_layers(home_a, enabled, {})

    assert layers_a[0].values["agent"]["reasoning_effort"] == "high"
    assert layers_b[0].values["agent"]["reasoning_effort"] == "low"
    assert layers_a_again[0].values["agent"]["reasoning_effort"] == "high"


def test_manifest_content_change_changes_cache_signature_even_with_same_metadata(tmp_path, monkeypatch):
    bundled = tmp_path / "bundled"
    home = tmp_path / "home"
    bundled.mkdir()
    home.mkdir()
    plugin_dir = _write_plugin(home / "plugins", "fixture")
    monkeypatch.setattr(plugins_config_defaults, "_BUNDLED_PLUGINS_DIR", bundled)
    enabled = {"plugins": {"enabled": ["fixture"]}}

    first_layers, first_sig = collect_config_default_layers(home, enabled, {})
    path = plugin_dir / "plugin.yaml"
    old_stat = path.stat()
    text = path.read_text(encoding="utf-8").replace("reasoning_effort: high", "reasoning_effort: low")
    path.write_text(text, encoding="utf-8")
    path.touch()
    import os
    os.utime(path, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))

    second_layers, second_sig = collect_config_default_layers(home, enabled, {})

    assert first_layers[0].values["agent"]["reasoning_effort"] == "high"
    assert second_layers[0].values["agent"]["reasoning_effort"] == "low"
    assert first_sig != second_sig


def test_dependency_order_traverses_enabled_intermediate_without_defaults(tmp_path, monkeypatch):
    bundled = tmp_path / "bundled"
    home = tmp_path / "home"
    bundled.mkdir()
    home.mkdir()
    plugins = home / "plugins"
    _write_plugin(plugins, "a-dependent", depends=["b-middle"], effort="low")
    middle = plugins / "b-middle"
    middle.mkdir()
    (middle / "plugin.yaml").write_text(chr(10).join([
        "name: b-middle", "kind: model-provider", "version: 0.1.0", "requires_plugins:", "  - z-base",
    ]) + chr(10), encoding="utf-8")
    (middle / "__init__.py").write_text("raise RuntimeError('must not import')\\n", encoding="utf-8")
    _write_plugin(plugins, "z-base", effort="high")
    monkeypatch.setattr(plugins_config_defaults, "_BUNDLED_PLUGINS_DIR", bundled)

    layers, _signature = collect_config_default_layers(
        home, {"plugins": {"enabled": ["a-dependent", "b-middle", "z-base"]}}, {}
    )

    assert [layer.plugin_key for layer in layers] == ["z-base", "a-dependent"]
    from hermes_cli.config_plugin_defaults import compose_user_layer
    assert compose_user_layer({}, layers)["agent"]["reasoning_effort"] == "high"


def test_dependency_order_then_alphabetical_first_wins(tmp_path, monkeypatch):
    bundled = tmp_path / "bundled"
    home = tmp_path / "home"
    bundled.mkdir()
    home.mkdir()
    plugins = home / "plugins"
    _write_plugin(plugins, "z-dependent", depends=["a-base"], effort="low")
    _write_plugin(plugins, "a-base", effort="high")
    _write_plugin(plugins, "b-middle", effort="medium")
    monkeypatch.setattr(plugins_config_defaults, "_BUNDLED_PLUGINS_DIR", bundled)

    layers, _ = collect_config_default_layers(
        home, {"plugins": {"enabled": ["z-dependent", "a-base", "b-middle"]}}, {}
    )

    assert [layer.plugin_key for layer in layers] == ["a-base", "b-middle", "z-dependent"]
