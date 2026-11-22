"""Manifest defaults participate in existing bare-name skill discovery without materialization."""

from pathlib import Path


def _enabled_skill_plugin(home: Path) -> None:
    plugin = home / "plugins" / "fixture-defaults"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(chr(10).join([
        "name: fixture-defaults", "version: 0.1.0", "config_defaults:", "  version: 1", "  values:",
        "    skills:", "      external_dirs:", "        - ~/.agents/skills", "  merge:",
        "    skills.external_dirs: append_unique",
    ]) + chr(10), encoding="utf-8")
    (home / "config.yaml").write_text(chr(10).join([
        "plugins:", "  enabled:", "    - fixture-defaults",
    ]) + chr(10), encoding="utf-8")


def test_bare_skill_name_is_discovered_from_plugin_default_root(tmp_path, monkeypatch):
    os_home = tmp_path / "os-home"
    hermes_home = os_home / ".hermes"
    hermes_home.mkdir(parents=True)
    external = os_home / ".agents" / "skills" / "fixture"
    external.mkdir(parents=True)
    (external / "SKILL.md").write_text(
        "---\nname: fixture\ndescription: Harmless fixture skill\n---\n\nFixture content.\n", encoding="utf-8"
    )
    _enabled_skill_plugin(hermes_home)
    monkeypatch.setattr(Path, "home", lambda: os_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from agent.skill_utils import get_external_skills_dirs
    from tools import skills_tool

    monkeypatch.setenv("HOME", str(os_home))
    skills_tool.clear_skills_cache()
    assert get_external_skills_dirs() == [os_home / ".agents" / "skills"]
    assert any(skill["name"] == "fixture" for skill in skills_tool.find_all_skills())
    assert "Fixture content." in skills_tool.skill_view("fixture")
    assert "skills:" not in (hermes_home / "config.yaml").read_text(encoding="utf-8")


def test_explicit_empty_external_list_inherits_but_null_blocks_plugin_skills(tmp_path, monkeypatch):
    os_home = tmp_path / "os-home"
    hermes_home = os_home / ".hermes"
    hermes_home.mkdir(parents=True)
    (os_home / ".agents" / "skills").mkdir(parents=True)
    _enabled_skill_plugin(hermes_home)
    monkeypatch.setattr(Path, "home", lambda: os_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from agent.skill_utils import _external_dirs_cache_clear, get_external_skills_dirs

    monkeypatch.setenv("HOME", str(os_home))
    assert get_external_skills_dirs() == [os_home / ".agents" / "skills"]
    config_path = hermes_home / "config.yaml"
    config_path.write_text(
        "plugins:\n  enabled: [fixture-defaults]\nskills:\n  external_dirs: null\n", encoding="utf-8"
    )
    _external_dirs_cache_clear()
    assert get_external_skills_dirs() == []
