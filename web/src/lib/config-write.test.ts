import { describe, expect, it } from "vitest";

import {
  buildExplicitConfigPatch,
  explicitConfigPaths,
  mergeConfigPatchPreview,
} from "./config-write";

describe("explicit config write intent", () => {
  it("persists an intentionally edited value equal to its effective default only when its path is dirty", () => {
    const effective = {
      agent: { reasoning_effort: "high" },
      display: { compact: false },
    };

    expect(buildExplicitConfigPatch(effective, ["agent.reasoning_effort"])).toEqual({
      agent: { reasoning_effort: "high" },
    });
    expect(buildExplicitConfigPatch(effective, [])).toEqual({});
  });

  it("keeps user list contributions and explicit null as submitted raw values", () => {
    expect(buildExplicitConfigPatch({ skills: { external_dirs: ["local"] } }, ["skills.external_dirs"])).toEqual({
      skills: { external_dirs: ["local"] },
    });
    expect(buildExplicitConfigPatch({ skills: { external_dirs: null } }, ["skills.external_dirs"])).toEqual({
      skills: { external_dirs: null },
    });
  });

  it("overlays imported raw values without replacing unrelated raw or effective sections", () => {
    expect(mergeConfigPatchPreview(
      { agent: { reasoning_effort: "high" }, display: { compact: false } },
      { display: { compact: true } },
    )).toEqual({ agent: { reasoning_effort: "high" }, display: { compact: true } });
  });

  it("treats every supplied import leaf as intentional", () => {
    expect(explicitConfigPaths({
      agent: { reasoning_effort: "high" },
      skills: { external_dirs: ["local"] },
      display: { compact: false },
    })).toEqual(["agent.reasoning_effort", "skills.external_dirs", "display.compact"]);
  });

  it("round-trips literal dots in imported config key segments", () => {
    const imported = JSON.parse(
      '{"providers":{"stranerd-foundry":{"models":{"gpt-6.1-sol":{"supports_vision":true}}}}}',
    ) as Record<string, unknown>;
    const paths = explicitConfigPaths(imported);

    expect(paths).toEqual(["providers.stranerd-foundry.models.gpt-6\\.1-sol.supports_vision"]);
    expect(buildExplicitConfigPatch(imported, paths)).toEqual(imported);
    expect(mergeConfigPatchPreview({}, imported)).toEqual(imported);
  });

  it("rejects prototype-sensitive imported keys without changing Object.prototype", () => {
    const marker = "hermes_config_import_proto_probe";
    const imported = JSON.parse(`{"__proto__":{"${marker}":true}}`) as Record<string, unknown>;
    try {
      expect(() => explicitConfigPaths(imported)).toThrow();
      expect(() => buildExplicitConfigPatch(imported, ["__proto__." + marker])).toThrow();
      expect(() => mergeConfigPatchPreview({}, imported)).toThrow();
      expect(Object.prototype.hasOwnProperty.call(Object.prototype, marker)).toBe(false);
    } finally {
      delete (Object.prototype as Record<string, unknown>)[marker];
    }
  });
});
