// @vitest-environment jsdom
import { act, useState, type ChangeEvent, type ReactNode, type SetStateAction } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter } from "react-router";
import { ProfileContext } from "@/contexts/profile-context";
import { createConfigPageActions } from "./config-page-actions";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const apiMocks = vi.hoisted(() => ({
  getConfig: vi.fn(),
  getConfigSource: vi.fn(),
  getSchema: vi.fn(),
  getDefaults: vi.fn(),
  getConfigRaw: vi.fn(),
  getStatus: vi.fn(),
  saveConfig: vi.fn(),
}));

vi.mock("@/lib/api", () => ({ api: apiMocks }));
vi.mock("@/plugins", () => ({ PluginSlot: () => null }));
vi.mock("@nous-research/ui/hooks/use-toast", () => ({
  useToast: () => ({ toast: null, showToast: vi.fn() }),
}));
vi.mock("@nous-research/ui/ui/components/toast", () => ({ Toast: () => null }));

let container: HTMLDivElement;
let root: Root;
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

async function flush() {
  await act(async () => {
    for (let index = 0; index < 8; index += 1) await Promise.resolve();
  });
}

async function waitFor(assertion: () => void) {
  let lastError: unknown;
  for (let index = 0; index < 8; index += 1) {
    await flush();
    try {
      assertion();
      return;
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError;
}

function mutableState<T>(initial: T) {
  let value = initial;
  return {
    get: () => value,
    set: (action: SetStateAction<T>) => {
      value = typeof action === "function" ? (action as (previous: T) => T)(value) : action;
    },
  };
}

function makeImportActions({
  config,
  rawConfig,
  editedPaths,
  unsetPaths,
}: {
  config: Record<string, unknown>;
  rawConfig: Record<string, unknown>;
  editedPaths: string[];
  unsetPaths: string[];
}) {
  const configState = mutableState<Record<string, unknown> | null>(config);
  const rawConfigState = mutableState<Record<string, unknown> | null>(rawConfig);
  const editedPathsState = mutableState(editedPaths);
  const unsetPathsState = mutableState(unsetPaths);
  const confirmResetState = mutableState(false);
  const showToast = vi.fn();
  const actions = createConfigPageActions({
    activeCategory: "general",
    activeFields: [],
    config: configState.get(),
    defaults: {},
    isSearching: false,
    rawConfig: rawConfigState.get(),
    rawSourceReady: true,
    configImported: "Imported",
    invalidJson: "Invalid config",
    resetScopeToast: "Reset {scope}",
    searchMatchedFields: [],
    schema: {},
    searchResults: "",
    setConfig: configState.set,
    setConfirmReset: confirmResetState.set,
    setEditedPaths: editedPathsState.set,
    setRawConfig: rawConfigState.set,
    setUnsetPaths: unsetPathsState.set,
    showToast,
    prettyCategoryName: (category) => category,
  });
  return { actions, configState, rawConfigState, editedPathsState, unsetPathsState, showToast };
}

function importConfig(actions: ReturnType<typeof makeImportActions>["actions"], document: Record<string, unknown>) {
  class ImmediateFileReader {
    result: string | ArrayBuffer | null = null;
    onload: ((this: FileReader, event: ProgressEvent<FileReader>) => unknown) | null = null;

    readAsText() {
      this.result = JSON.stringify(document);
      this.onload?.call(this as unknown as FileReader, {} as ProgressEvent<FileReader>);
    }
  }
  vi.stubGlobal("FileReader", ImmediateFileReader);
  try {
    actions.handleImport({
      target: { files: [new File(["{}"], "config.json")] as unknown as FileList, value: "" },
    } as unknown as ChangeEvent<HTMLInputElement>);
  } finally {
    vi.unstubAllGlobals();
  }
}

function changeValue(element: HTMLInputElement | HTMLTextAreaElement, value: string) {
  const prototype = element instanceof HTMLInputElement ? HTMLInputElement.prototype : HTMLTextAreaElement.prototype;
  Object.getOwnPropertyDescriptor(prototype, "value")!.set!.call(element, value);
  element.dispatchEvent(new Event("input", { bubbles: true }));
}

function ProfileHarness({
  children,
  initialProfile,
  showSwitch,
}: {
  children: ReactNode;
  initialProfile: string;
  showSwitch: boolean;
}) {
  const [profile, setProfile] = useState(initialProfile);

  return (
    <ProfileContext.Provider value={{ currentProfile: "default", profile, profiles: [], setProfile }}>
      {showSwitch && <button onClick={() => setProfile("beta")} type="button">Switch profile</button>}
      {children}
    </ProfileContext.Provider>
  );
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, reject, resolve };
}

async function renderPage({ initialProfile = "", showSwitch = false } = {}) {
  const [{ default: ConfigPage }, { I18nProvider }, { PageHeaderProvider }] = await Promise.all([
    import("./ConfigPage"),
    import("@/i18n"),
    import("@/contexts/PageHeaderProvider"),
  ]);
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => root.render(
    <I18nProvider>
      <MemoryRouter>
        <ProfileHarness initialProfile={initialProfile} showSwitch={showSwitch}>
          <PageHeaderProvider pluginTabs={[]}>
            <ConfigPage />
          </PageHeaderProvider>
        </ProfileHarness>
      </MemoryRouter>
    </I18nProvider>,
  ));
  await flush();
}

beforeEach(() => {
  Object.values(apiMocks).forEach((mock) => mock.mockReset());
  apiMocks.getSchema.mockResolvedValue({ fields: {}, category_order: [] });
  apiMocks.getDefaults.mockResolvedValue({});
  apiMocks.getConfigRaw.mockResolvedValue({ yaml: "", path: "temp/config.yaml" });
  apiMocks.getStatus.mockResolvedValue({ config_path: "temp/config.yaml" });
  apiMocks.saveConfig.mockResolvedValue({ ok: true });
  apiMocks.getConfigSource.mockResolvedValue({});
});

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
});

describe("ConfigPage explicit config writes", () => {
  it("preserves dirty edits and pending resets when importing an unrelated path", () => {
    const state = makeImportActions({
      config: { agent: { reasoning_effort: "high" }, skills: { disabled: ["local"] } },
      rawConfig: { agent: { reasoning_effort: "low" }, skills: { disabled: ["local"] } },
      editedPaths: ["agent.reasoning_effort"],
      unsetPaths: ["skills.disabled"],
    });

    importConfig(state.actions, { display: { compact: true } });

    expect(state.configState.get()).toEqual({
      agent: { reasoning_effort: "high" }, skills: { disabled: ["local"] }, display: { compact: true },
    });
    expect(state.rawConfigState.get()).toEqual({
      agent: { reasoning_effort: "low" }, skills: { disabled: ["local"] }, display: { compact: true },
    });
    expect(state.editedPathsState.get()).toEqual(["agent.reasoning_effort", "display.compact"]);
    expect(state.unsetPathsState.get()).toEqual(["skills.disabled"]);
  });

  it("lets imported paths supersede matching edits and resets only", () => {
    const state = makeImportActions({
      config: { agent: { reasoning_effort: "high" }, display: { compact: false }, skills: { disabled: ["local"] } },
      rawConfig: { agent: { reasoning_effort: "high" }, display: { compact: false }, skills: { disabled: ["local"] } },
      editedPaths: ["display.compact", "agent.reasoning_effort"],
      unsetPaths: ["display.compact", "skills.disabled"],
    });

    importConfig(state.actions, { display: { compact: true } });

    expect(state.configState.get()).toEqual({
      agent: { reasoning_effort: "high" }, display: { compact: true }, skills: { disabled: ["local"] },
    });
    expect(state.rawConfigState.get()).toEqual({
      agent: { reasoning_effort: "high" }, display: { compact: true }, skills: { disabled: ["local"] },
    });
    expect(state.editedPathsState.get()).toEqual(["display.compact", "agent.reasoning_effort"]);
    expect(state.unsetPathsState.get()).toEqual(["skills.disabled"]);
  });

  it("cancels an ancestor model reset when a raw model.default import overlaps it", () => {
    const state = makeImportActions({
      config: { model: "old" },
      rawConfig: { model: { provider: "openrouter", default: "old" } },
      editedPaths: [],
      unsetPaths: ["model"],
    });

    importConfig(state.actions, { model: { default: "new" } });

    expect(state.configState.get()).toEqual({ model: { default: "new" } });
    expect(state.rawConfigState.get()).toEqual({ model: { provider: "openrouter", default: "new" } });
    expect(state.editedPathsState.get()).toEqual(["model.default"]);
    expect(state.unsetPathsState.get()).toEqual([]);
  });

  it("cancels a raw model.default reset when a normalized model import overlaps it", () => {
    const state = makeImportActions({
      config: { model: { default: "old" } },
      rawConfig: { model: { default: "old" } },
      editedPaths: [],
      unsetPaths: ["model.default"],
    });

    importConfig(state.actions, { model: "new" });

    expect(state.configState.get()).toEqual({ model: "new" });
    expect(state.rawConfigState.get()).toEqual({ model: "new" });
    expect(state.editedPathsState.get()).toEqual(["model"]);
    expect(state.unsetPathsState.get()).toEqual([]);
  });

  it("rejects prototype-sensitive imports before changing either draft", () => {
    const marker = "hermes_config_import_proto_probe";
    const imported = JSON.parse(`{"__proto__":{"${marker}":true}}`) as Record<string, unknown>;
    const state = makeImportActions({
      config: { display: { compact: false } },
      rawConfig: { display: { compact: false } },
      editedPaths: [],
      unsetPaths: [],
    });
    delete (Object.prototype as Record<string, unknown>)[marker];
    try {
      importConfig(state.actions, imported);
      expect(state.configState.get()).toEqual({ display: { compact: false } });
      expect(state.rawConfigState.get()).toEqual({ display: { compact: false } });
      expect(state.showToast).toHaveBeenCalledWith("Invalid config", "error");
      expect(Object.prototype.hasOwnProperty.call(Object.prototype, marker)).toBe(false);
    } finally {
      delete (Object.prototype as Record<string, unknown>)[marker];
    }
  });

  it("keeps list writes unavailable through raw-source failure until retry loads the user's entries", async () => {
    const effective = { skills: { external_dirs: ["local-skills", "shared-skills"] } };
    const firstSource = deferred<Record<string, unknown>>();
    const retriedSource = deferred<Record<string, unknown>>();
    apiMocks.getConfig.mockResolvedValue(effective);
    apiMocks.getConfigSource
      .mockReturnValueOnce(firstSource.promise)
      .mockReturnValueOnce(retriedSource.promise)
      .mockResolvedValue({ skills: { external_dirs: ["local-skills"] } });
    apiMocks.getSchema.mockResolvedValue({
      fields: { "skills.external_dirs": { type: "list", category: "skills", plugin_default: true, merge_mode: "append_unique" } },
      category_order: ["skills"],
    });
    await renderPage();

    const field = container.querySelector<HTMLInputElement>('input[placeholder="Loading..."]');
    if (!field) throw new Error("pending raw list state missing");
    expect(field.disabled).toBe(true);
    changeValue(field, "invented-skills");
    expect(apiMocks.saveConfig).not.toHaveBeenCalled();

    await act(async () => firstSource.reject(new Error("raw source unavailable")));
    await flush();
    expect(container.querySelector('[role="alert"]')).not.toBeNull();
    expect(field.disabled).toBe(true);
    for (const icon of ["download", "upload", "rotate-ccw"]) {
      const button = container.querySelector<HTMLElement>(`.lucide-${icon}`)?.closest("button");
      if (!button) throw new Error(`${icon} action missing`);
      expect((button as HTMLButtonElement).disabled).toBe(true);
    }

    const retry = Array.from(container.querySelectorAll("button")).find((button) => button.textContent?.trim() === "Refresh");
    if (!retry) throw new Error("raw-source retry missing");
    await act(async () => {
      retry.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await flush();
    });
    await act(async () => retriedSource.resolve({ skills: { external_dirs: ["local-skills"] } }));
    await waitFor(() => expect(field.value).toBe("local-skills"));
    expect(field.disabled).toBe(false);

    await act(async () => {
      changeValue(field, "local-skills, another-local");
      field.dispatchEvent(new Event("blur", { bubbles: true }));
    });
    const save = Array.from(container.querySelectorAll("button")).find((button) => button.textContent?.trim() === "Save");
    if (!save) throw new Error("save button missing");
    await act(async () => {
      save.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await flush();
    });
    expect(apiMocks.saveConfig).toHaveBeenCalledWith(
      { skills: { external_dirs: ["local-skills", "another-local"] } }, undefined, [],
    );
  });

  it("does not publish an old profile raw read after the dashboard scope changes", async () => {
    const alphaSource = deferred<Record<string, unknown>>();
    const betaSource = deferred<Record<string, unknown>>();
    apiMocks.getConfig.mockImplementation((profile) => Promise.resolve({
      skills: { external_dirs: [`${profile}-local`, `${profile}-shared`] },
    }));
    apiMocks.getConfigSource.mockImplementation((profile) => profile === "alpha" ? alphaSource.promise : betaSource.promise);
    apiMocks.getSchema.mockResolvedValue({
      fields: { "skills.external_dirs": { type: "list", category: "skills", plugin_default: true, merge_mode: "append_unique" } },
      category_order: ["skills"],
    });
    await renderPage({ initialProfile: "alpha", showSwitch: true });

    let field = container.querySelector<HTMLInputElement>('input[placeholder="Loading..."]');
    if (!field) throw new Error("alpha raw list pending state missing");
    expect(field.disabled).toBe(true);
    const switchProfile = Array.from(container.querySelectorAll("button")).find((button) => button.textContent?.trim() === "Switch profile");
    if (!switchProfile) throw new Error("profile switch control missing");
    await act(async () => {
      switchProfile.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await flush();
    });
    expect(apiMocks.getConfigSource).toHaveBeenCalledWith("beta");

    await act(async () => alphaSource.resolve({ skills: { external_dirs: ["alpha-local"] } }));
    field = container.querySelector<HTMLInputElement>('input[placeholder="Loading..."]');
    if (!field) throw new Error("beta raw list pending state missing");
    expect(field.disabled).toBe(true);
    await act(async () => betaSource.resolve({ skills: { external_dirs: ["beta-local"] } }));
    await waitFor(() => {
      const loaded = container.querySelector<HTMLInputElement>('input[placeholder="comma-separated values"]');
      expect(loaded?.value).toBe("beta-local");
    });
  });

  it("saves an intentionally touched equal value as a sparse explicit patch", async () => {
    apiMocks.getConfig.mockResolvedValue({ agent: { reasoning_effort: "high" } });
    apiMocks.getConfigSource.mockResolvedValue({ plugins: { enabled: ["fixture"] } });
    apiMocks.getSchema.mockResolvedValue({
      fields: { "agent.reasoning_effort": { type: "text", category: "agent" } },
      category_order: ["agent"],
    });
    await renderPage();

    const field = container.querySelector("textarea");
    if (!field) throw new Error("reasoning field missing");
    await act(async () => {
      changeValue(field, "");
      changeValue(field, "high");
    });
    const save = Array.from(container.querySelectorAll("button")).find((button) => button.textContent?.trim() === "Save");
    if (!save) throw new Error("save button missing");
    await act(async () => {
      save.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await flush();
    });

    expect(apiMocks.saveConfig).toHaveBeenCalledWith(
      { agent: { reasoning_effort: "high" } }, undefined, [],
    );
  });

  it("edits an append_unique list as raw user entries, not effective inherited entries", async () => {
    apiMocks.getConfig.mockResolvedValue({ skills: { external_dirs: ["local-skills", "shared-skills"] } });
    apiMocks.getConfigSource.mockResolvedValue({
      plugins: { enabled: ["fixture"] }, skills: { external_dirs: ["local-skills"] },
    });
    apiMocks.getSchema.mockResolvedValue({
      fields: { "skills.external_dirs": { type: "list", category: "skills", plugin_default: true, merge_mode: "append_unique" } },
      category_order: ["skills"],
    });
    await renderPage();

    const field = container.querySelector<HTMLInputElement>('input[placeholder="comma-separated values"]');
    if (!field) throw new Error("external skills field missing");
    expect(field.value).toBe("local-skills");
    await act(async () => changeValue(field, "local-skills, another-local"));
    const save = Array.from(container.querySelectorAll("button")).find((button) => button.textContent?.trim() === "Save");
    if (!save) throw new Error("save button missing");
    await act(async () => {
      save.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await flush();
    });

    expect(apiMocks.saveConfig).toHaveBeenCalledWith(
      { skills: { external_dirs: ["local-skills", "another-local"] } }, undefined, [],
    );
  });
});
