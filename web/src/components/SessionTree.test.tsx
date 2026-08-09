// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { getSessionChildren } = vi.hoisted(() => ({
  getSessionChildren: vi.fn(),
}));

vi.mock("@/lib/api", () => ({
  api: { getSessionChildren },
}));
vi.mock("@/lib/utils", () => ({
  cn: (...parts: Array<string | false | null | undefined>) =>
    parts.filter(Boolean).join(" "),
}));

import { SessionTree } from "@/components/SessionTree";
import type { SessionInfo } from "@/lib/api";

const makeSession = (id: string, extra: Partial<SessionInfo> = {}): SessionInfo => ({
  id,
  source: "desktop",
  model: null,
  title: id,
  started_at: 1,
  ended_at: null,
  last_active: 1,
  is_active: false,
  message_count: 1,
  tool_call_count: 0,
  input_tokens: 0,
  output_tokens: 0,
  preview: null,
  ...extra,
});

describe("SessionTree", () => {
  let root: Root | null = null;
  let host: HTMLDivElement;

  beforeEach(() => {
    host = document.createElement("div");
    document.body.appendChild(host);
    root = createRoot(host);
    getSessionChildren.mockReset();
  });

  afterEach(() => {
    act(() => root?.unmount());
    host.remove();
  });

  it("loads direct children lazily and keeps nested node IDs distinct", async () => {
    const parent = makeSession("parent", { child_count: 1, has_children: true });
    const child = makeSession("child");
    getSessionChildren.mockResolvedValue({
      parent_session_id: "parent",
      children: [child],
    });

    await act(async () => {
      root?.render(
        <SessionTree
          roots={[parent]}
          renderNode={({ session, hasChildren, onToggleChildren }) => (
            <div data-session-id={session.id}>
              <button
                type="button"
                data-expand={session.id}
                disabled={!hasChildren}
                onClick={onToggleChildren}
              >
                expand
              </button>
              <button type="button" data-pick={session.id}>
                {session.id}
              </button>
            </div>
          )}
        />,
      );
    });

    expect(host.querySelector('[data-session-id="child"]')).toBeNull();
    await act(async () => {
      host.querySelector<HTMLButtonElement>('[data-expand="parent"]')?.click();
    });

    expect(getSessionChildren).toHaveBeenCalledTimes(1);
    expect(getSessionChildren).toHaveBeenCalledWith("parent", undefined);
    expect(host.querySelector('[data-session-id="parent"]')).not.toBeNull();
    expect(host.querySelector('[data-session-id="child"]')).not.toBeNull();
  });

  it("does not refetch a cached child list after collapse and re-expand", async () => {
    const parent = makeSession("cached-parent", { child_count: 1 });
    getSessionChildren.mockResolvedValue({
      parent_session_id: "cached-parent",
      children: [makeSession("cached-child")],
    });

    await act(async () => {
      root?.render(
        <SessionTree
          roots={[parent]}
          renderNode={({ session, onToggleChildren }) => (
            <div data-session-id={session.id}>
              <button type="button" data-expand={session.id} onClick={onToggleChildren}>
                expand
              </button>
            </div>
          )}
        />,
      );
    });

    const toggle = () =>
      host.querySelector<HTMLButtonElement>('[data-expand="cached-parent"]')?.click();
    await act(async () => toggle());
    await act(async () => toggle());
    await act(async () => toggle());

    expect(getSessionChildren).toHaveBeenCalledTimes(1);
  });
});
