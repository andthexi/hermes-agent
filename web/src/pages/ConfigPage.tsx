import { useEffect, useLayoutEffect, useRef, useState, useMemo } from "react";
import {
  Code,
  Download,
  FormInput,
  RotateCcw,
  Search,
  Upload,
  X,
  Settings2,
  Settings,
  Bot,
  Monitor,
  Palette,
  Users,
  Brain,
  Package,
  Lock,
  Globe,
  Mic,
  Volume2,
  Ear,
  ClipboardList,
  MessageCircle,
  Wrench,
  FileQuestion,
  Cloud,
  Sparkles,
  LayoutDashboard,
  BookOpen,
  Route,
  History,
  Shield,
  FileOutput,
  RefreshCw,
} from "lucide-react";
import { api } from "@/lib/api";
import { getNestedValue, setNestedValue } from "@/lib/nested";
import { useRawConfigSource } from "@/lib/use-raw-config-source";
import { useConfigYamlSource } from "@/lib/use-config-yaml-source";
import { buildExplicitConfigPatch } from "@/lib/config-write";
import { createConfigPageActions } from "./config-page-actions";
import { useToast } from "@nous-research/ui/hooks/use-toast";
import { Toast } from "@nous-research/ui/ui/components/toast";
import { AutoField } from "@/components/AutoField";
import { Button } from "@nous-research/ui/ui/components/button";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { ConfirmDialog } from "@nous-research/ui/ui/components/confirm-dialog";
import { Input } from "@nous-research/ui/ui/components/input";
import { useI18n } from "@/i18n";
import { usePageHeader } from "@/contexts/usePageHeader";
import { useProfileScope } from "@/contexts/useProfileScope";
import { PluginSlot } from "@/plugins";
import { errorMessage } from "@/lib/api-error";
import { ConfigPageBody } from "./config-page-body";

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

const CATEGORY_ICONS: Record<
  string,
  React.ComponentType<{ className?: string }>
> = {
  general: Settings,
  agent: Bot,
  terminal: Monitor,
  display: Palette,
  delegation: Users,
  memory: Brain,
  compression: Package,
  security: Lock,
  browser: Globe,
  voice: Mic,
  tts: Volume2,
  stt: Ear,
  logging: ClipboardList,
  discord: MessageCircle,
  auxiliary: Wrench,
  bedrock: Cloud,
  curator: Sparkles,
  kanban: LayoutDashboard,
  model_catalog: BookOpen,
  openrouter: Route,
  sessions: History,
  tool_loop_guardrails: Shield,
  tool_output: FileOutput,
  updates: RefreshCw,
};

function CategoryIcon({
  category,
  className,
}: {
  category: string;
  className?: string;
}) {
  const Icon = CATEGORY_ICONS[category] ?? FileQuestion;
  return <Icon className={className ?? "h-4 w-4"} />;
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export default function ConfigPage() {
  const { profile } = useProfileScope();
  return <ConfigPageInner key={profile} profile={profile} />;
}

function ConfigPageInner({ profile }: { profile: string }) {
  const [config, setConfig] = useState<Record<string, unknown> | null>(null);
  const { source: rawSource, status: rawSourceStatus, retry: retryRawConfigSource } = useRawConfigSource(profile);
  const [rawConfigDraft, setRawConfigDraft] = useState<Record<string, unknown> | null>(null);
  const rawConfig = rawConfigDraft ?? rawSource;
  const [editedPaths, setEditedPaths] = useState<string[]>([]);
  const [unsetPaths, setUnsetPaths] = useState<string[]>([]);
  const [schema, setSchema] = useState<Record<
    string,
    Record<string, unknown>
  > | null>(null);
  const [categoryOrder, setCategoryOrder] = useState<string[]>([]);
  const [defaults, setDefaults] = useState<Record<string, unknown> | null>(
    null,
  );
  const [saving, setSaving] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [yamlMode, setYamlMode] = useState(false);
  const { loading: yamlLoading, retry: retryYamlSource, setText: setYamlText, status: yamlStatus, text: yamlText } = useConfigYamlSource(profile, yamlMode);
  const [yamlSaving, setYamlSaving] = useState(false);
  const [configPath, setConfigPath] = useState<string | null>(null);
  const [activeCategory, setActiveCategory] = useState<string>("");
  const [confirmReset, setConfirmReset] = useState(false);
  const { toast, showToast } = useToast();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { t } = useI18n();
  const { setEnd } = usePageHeader();

  useLayoutEffect(() => {
    if (!config || !schema) {
      setEnd(null);
      return;
    }
    setEnd(
      <div className="relative w-full min-w-0 sm:max-w-xs">
        <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
        <Input
          className="h-8 pl-8 pr-7 text-xs"
          placeholder={t.common.search}
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />
        {searchQuery && (
          <Button
            ghost
            size="xs"
            className="absolute right-1.5 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
            onClick={() => setSearchQuery("")}
            aria-label={t.common.clear}
          >
            <X />
          </Button>
        )}
      </div>,
    );
    return () => setEnd(null);
  }, [config, schema, searchQuery, setEnd, t.common.clear, t.common.search]);

  function prettyCategoryName(cat: string): string {
    const key = cat as keyof typeof t.config.categories;
    if (t.config.categories[key]) return t.config.categories[key];
    return cat.charAt(0).toUpperCase() + cat.slice(1);
  }

  useEffect(() => {
    let cancelled = false;
    api
      .getConfig(profile || undefined)
      .then((next) => {
        if (!cancelled) setConfig(next);
      })
      .catch(() => {});
    api
      .getSchema(profile || undefined)
      .then((resp) => {
        if (cancelled) return;
        // memory.provider has a dedicated management UI on the Plugins page
        // (provider cards + guided setup/switch flow). Hide it from the
        // generic config form so the two surfaces don't fight; the schema
        // keeps the field for other consumers (Desktop settings).
        const fields = { ...resp.fields } as Record<
          string,
          Record<string, unknown>
        >;
        delete fields["memory.provider"];
        setSchema(fields);
        setCategoryOrder(resp.category_order ?? []);
      })
      .catch(() => {});
    api
      .getDefaults()
      .then(setDefaults)
      .catch(() => {});
    // getConfigRaw is profile-scoped (fetchJSON appends ?profile=), so its
    // `path` reflects the switched profile's config.yaml. /api/status's
    // config_path is machine-global (the dashboard's own profile) — wrong
    // header under the global profile switcher, so it's only a fallback.
    api
      .getConfigRaw(profile || undefined)
      .then((resp) => {
        if (!cancelled && resp.path) setConfigPath(resp.path);
      })
      .catch(() => {});
    api
      .getStatus()
      .then((resp) => {
        if (!cancelled) setConfigPath((prev) => prev ?? resp.config_path);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [profile]);

  // Set active category when categories load
  useEffect(() => {
    if (categoryOrder.length > 0 && !activeCategory) {
      setActiveCategory(categoryOrder[0]);
    }
  }, [categoryOrder, activeCategory]);

  /* ---- Categories ---- */
  const categories = useMemo(() => {
    if (!schema) return [];
    const allCats = [
      ...new Set(
        Object.values(schema).map((s) => String(s.category ?? "general")),
      ),
    ];
    const ordered = categoryOrder.filter((c) => allCats.includes(c));
    const extra = allCats.filter((c) => !categoryOrder.includes(c)).sort();
    return [...ordered, ...extra];
  }, [schema, categoryOrder]);

  /* ---- Category field counts ---- */
  const categoryCounts = useMemo(() => {
    if (!schema) return {};
    const counts: Record<string, number> = {};
    for (const s of Object.values(schema)) {
      const cat = String(s.category ?? "general");
      counts[cat] = (counts[cat] || 0) + 1;
    }
    return counts;
  }, [schema]);

  /* ---- Search ---- */
  const isSearching = searchQuery.trim().length > 0;
  const lowerSearch = searchQuery.toLowerCase();

  const searchMatchedFields = useMemo(() => {
    if (!isSearching || !schema) return [];
    return Object.entries(schema).filter(([key, s]) => {
      const label = key.split(".").pop() ?? key;
      const humanLabel = label.replace(/_/g, " ");
      return (
        key.toLowerCase().includes(lowerSearch) ||
        humanLabel.toLowerCase().includes(lowerSearch) ||
        String(s.category ?? "")
          .toLowerCase()
          .includes(lowerSearch) ||
        String(s.description ?? "")
          .toLowerCase()
          .includes(lowerSearch)
      );
    });
  }, [isSearching, lowerSearch, schema]);

  /* ---- Active tab fields ---- */
  const activeFields = useMemo(() => {
    if (!schema || isSearching) return [];
    return Object.entries(schema).filter(
      ([, s]) => String(s.category ?? "general") === activeCategory,
    );
  }, [schema, activeCategory, isSearching]);

  /* ---- Handlers ---- */
  const hasRawSourceDependentEdits = editedPaths.some((path) => schema?.[path]?.plugin_default);
  const retryRawSource = () => {
    setRawConfigDraft(null);
    retryRawConfigSource();
  };

  const handleSave = async () => {
    if (!config || (hasRawSourceDependentEdits && rawSourceStatus !== "ready")) return;
    setSaving(true);
    try {
      const patch = buildExplicitConfigPatch(config, editedPaths);
      await api.saveConfig(patch, profile || undefined, unsetPaths);
      retryRawSource();
      const effective = await api.getConfig(profile || undefined);
      setConfig(effective);
      setEditedPaths([]);
      setUnsetPaths([]);
      showToast(t.config.configSaved, "success");
    } catch (e) {
      showToast(`${t.config.failedToSave}: ${errorMessage(e)}`, "error");
    } finally {
      setSaving(false);
    }
  };

  const handleYamlSave = async () => {
    if (yamlStatus !== "ready" || yamlLoading) return;
    setYamlSaving(true);
    try {
      await api.saveConfigRaw(yamlText, profile || undefined);
      showToast(t.config.yamlConfigSaved, "success");
      api.getConfig(profile || undefined).then(setConfig).catch(() => {});
      retryRawSource();
      setEditedPaths([]);
      setUnsetPaths([]);
    } catch (e) {
      showToast(`${t.config.failedToSaveYaml}: ${errorMessage(e)}`, "error");
    } finally {
      setYamlSaving(false);
    }
  };

  const { executeReset, handleExport, handleImport, handleReset } = createConfigPageActions({
    activeCategory,
    activeFields,
    config,
    configImported: t.config.configImported,
    defaults,
    invalidJson: t.config.invalidJson,
    isSearching,
    prettyCategoryName,
    rawConfig,
    rawSourceReady: rawSourceStatus === "ready",
    resetScopeToast: t.config.resetScopeToast,
    schema,
    searchMatchedFields,
    searchResults: t.config.searchResults,
    setConfig,
    setConfirmReset,
    setEditedPaths,
    setRawConfig: setRawConfigDraft,
    setUnsetPaths,
    showToast,
  });

  /* ---- Loading ---- */
  if (!config || !schema) {
    return (
      <div className="flex items-center justify-center py-24">
        <Spinner className="text-2xl text-primary" />
      </div>
    );
  }

  /* ---- Render field list (shared between search & normal) ---- */
  const renderFields = (
    fields: [string, Record<string, unknown>][],
    showCategory = false,
  ) => {
    let lastSection = "";
    let lastCat = "";
    return fields.map(([key, s]) => {
      const parts = key.split(".");
      const section = parts.length > 1 ? parts[0] : "";
      const cat = String(s.category ?? "general");
      const showCatBadge = showCategory && cat !== lastCat;
      const showSection =
        !showCategory &&
        section &&
        section !== lastSection &&
        section !== activeCategory;
      lastSection = section;
      lastCat = cat;

      return (
        <div key={key}>
          {showCatBadge && (
            <div className="flex items-center gap-2 pt-4 pb-2 first:pt-0">
              <CategoryIcon
                category={cat}
                className="h-4 w-4 text-muted-foreground"
              />
              <span className="font-mondwest text-display text-xs font-semibold tracking-wider text-muted-foreground">
                {prettyCategoryName(cat)}
              </span>
              <div className="flex-1 border-t border-border" />
            </div>
          )}
          {showSection && (
            <div className="flex items-center gap-2 pt-4 pb-2 first:pt-0">
              <span className="font-mondwest text-display text-xs font-semibold tracking-wider text-muted-foreground">
                {section.replace(/_/g, " ")}
              </span>
              <div className="flex-1 border-t border-border" />
            </div>
          )}
          <div className="py-1">
            <AutoField
              schemaKey={key}
              schema={s}
              value={s.plugin_default
                ? rawConfig ? getNestedValue(rawConfig, key) : undefined
                : getNestedValue(config, key)}
              disabled={Boolean(s.plugin_default && rawSourceStatus !== "ready")}
              placeholder={s.plugin_default && rawSourceStatus !== "ready"
                ? rawSourceStatus === "loading" ? t.common.loading : t.config.failedToLoadRaw
                : undefined}
              onChange={(v) => {
                if (s.plugin_default && rawSourceStatus !== "ready") return;
                setConfig(setNestedValue(config, key, v));
                if (s.plugin_default) {
                  setRawConfigDraft((previous) => {
                    const source = previous ?? rawConfig;
                    return source ? setNestedValue(source, key, v) : previous;
                  });
                }
                setEditedPaths((previous) => previous.includes(key) ? previous : [...previous, key]);
                setUnsetPaths((previous) => previous.filter((path) => path !== key));
              }}
            />
          </div>
        </div>
      );
    });
  };

  return (
    <div className="flex flex-col gap-4">
      <PluginSlot name="config:top" />
      <Toast toast={toast} />
      {rawSourceStatus === "error" && (
        <div className="flex items-center justify-between gap-3 rounded-lg border border-destructive/40 bg-destructive/5 p-3" role="alert">
          <span className="text-sm">{t.config.failedToLoadRaw}</span>
          <Button ghost size="sm" onClick={retryRawSource}>{t.common.refresh}</Button>
        </div>
      )}

      <div className="flex min-w-0 flex-col gap-3 sm:flex-row sm:items-center sm:justify-between sm:gap-4">
        <div className="flex min-w-0 items-center gap-2 sm:flex-1">
          <Settings2 className="h-4 w-4 shrink-0 text-muted-foreground" />
          <code className="min-w-0 flex-1 break-words text-xs text-muted-foreground bg-muted/50 px-2 py-0.5">
            {configPath ?? t.config.configPath}
          </code>
        </div>
        <div className="flex flex-wrap items-center gap-1.5 sm:shrink-0">
          <Button
            ghost
            size="icon"
            onClick={handleExport}
            disabled={rawSourceStatus !== "ready"}
            title={t.config.exportConfig}
            aria-label={t.config.exportConfig}
          >
            <Download />
          </Button>
          <Button
            ghost
            size="icon"
            onClick={() => fileInputRef.current?.click()}
            disabled={rawSourceStatus !== "ready"}
            title={t.config.importConfig}
            aria-label={t.config.importConfig}
          >
            <Upload />
          </Button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".json"
            className="hidden"
            disabled={rawSourceStatus !== "ready"}
            onChange={handleImport}
          />
          {!yamlMode &&
            (() => {
              const resetScopeLabel = isSearching
                ? t.config.searchResults
                : prettyCategoryName(activeCategory);
              const resetTitle = t.config.resetScopeTooltip.replace(
                "{scope}",
                resetScopeLabel,
              );
              return (
                <Button
                  ghost
                  size="icon"
                  onClick={handleReset}
                  disabled={rawSourceStatus !== "ready"}
                  title={resetTitle}
                  aria-label={resetTitle}
                >
                  <RotateCcw />
                </Button>
              );
            })()}

          <div className="w-px h-5 bg-border mx-1" />

          <Button
            size="sm"
            outlined={!yamlMode}
            onClick={() => setYamlMode(!yamlMode)}
            prefix={yamlMode ? <FormInput /> : <Code />}
          >
            {yamlMode ? t.common.form : "YAML"}
          </Button>

          {yamlMode ? (
            <Button
              size="sm"
              className="uppercase"
              onClick={handleYamlSave}
              disabled={yamlSaving || yamlStatus !== "ready" || yamlLoading}
            >
              {yamlSaving ? t.common.saving : t.common.save}
            </Button>
          ) : (
            <Button
              size="sm"
              className="uppercase"
              onClick={handleSave}
              disabled={saving || (hasRawSourceDependentEdits && rawSourceStatus !== "ready")}
            >
              {saving ? t.common.saving : t.common.save}
            </Button>
          )}
        </div>
      </div>

      <ConfigPageBody
        CategoryIcon={CategoryIcon}
        activeCategory={activeCategory}
        activeFields={activeFields}
        categories={categories}
        categoryCounts={categoryCounts}
        isSearching={isSearching}
        onRetryYamlSource={retryYamlSource}
        onSelectCategory={(category) => {
          setSearchQuery("");
          setActiveCategory(category);
        }}
        onYamlChange={setYamlText}
        prettyCategoryName={prettyCategoryName}
        renderFields={renderFields}
        searchMatchedFields={searchMatchedFields}
        searchQuery={searchQuery}
        yamlLoading={yamlLoading}
        yamlMode={yamlMode}
        yamlStatus={yamlStatus}
        yamlText={yamlText}
      />
      <PluginSlot name="config:bottom" />
      <ConfirmDialog
        open={confirmReset}
        onCancel={() => setConfirmReset(false)}
        onConfirm={executeReset}
        title={t.config.confirmResetScope.replace(
          "{scope}",
          isSearching
            ? t.config.searchResults
            : prettyCategoryName(activeCategory),
        )}
        description={`This will reset ${
          (isSearching ? searchMatchedFields : activeFields).length
        } field(s) to their default values.`}
        destructive
        confirmLabel={t.config.resetDefaults}
      />
    </div>
  );
}
