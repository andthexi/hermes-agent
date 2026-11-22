// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from "vitest";

import { api } from "./api";

describe("config write request contract", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("uses the additive exact-raw query without redefining legacy include_defaults=false", async () => {
    const fetchMock = vi.fn<typeof fetch>(async () => new Response(JSON.stringify({ model: { default: "raw-model" } }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    await api.getConfigSource("worker");

    const requestUrl = new URL(String(fetchMock.mock.calls[0][0]), window.location.href);
    expect(requestUrl.searchParams.get("raw_source")).toBe("true");
    expect(requestUrl.searchParams.get("profile")).toBe("worker");
    expect(requestUrl.searchParams.has("include_defaults")).toBe(false);
  });

  it("tags sparse updates and reset paths as explicit write intent", async () => {
    const fetchMock = vi.fn<typeof fetch>(async () => new Response(JSON.stringify({ ok: true }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    await api.saveConfig({ agent: { reasoning_effort: "high" } }, "worker", ["skills.external_dirs"]);

    const init = fetchMock.mock.calls[0][1];
    expect(JSON.parse(String(init?.body))).toEqual({
      config: { agent: { reasoning_effort: "high" } },
      write_mode: "explicit_patch",
      unset_paths: ["skills.external_dirs"],
    });
    expect(String(fetchMock.mock.calls[0][0])).toContain("profile=worker");
  });
});
