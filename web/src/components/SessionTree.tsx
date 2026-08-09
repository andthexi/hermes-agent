import { ChevronDown, ChevronRight, RefreshCw } from "lucide-react";
import { Fragment, type ReactNode, useCallback, useRef, useState } from "react";

import { api, type SessionInfo } from "@/lib/api";
import { cn } from "@/lib/utils";

export interface SessionTreeNodeProps {
  session: SessionInfo;
  depth: number;
  hasChildren: boolean;
  expanded: boolean;
  loading: boolean;
  error: string | null;
  onToggleChildren: () => void;
}

interface SessionTreeProps {
  roots: SessionInfo[];
  profile?: string;
  className?: string;
  renderNode: (props: SessionTreeNodeProps) => ReactNode;
}

/**
 * Lazy session-lineage tree shared by the history page and chat switcher.
 * Only direct children of an expanded node are fetched, and the metadata
 * cache is deliberately separate from transcript/message expansion.
 */
export function SessionTree({
  roots,
  profile,
  className,
  renderNode,
}: SessionTreeProps) {
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [childrenByParent, setChildrenByParent] = useState<
    Record<string, SessionInfo[]>
  >({});
  const [loadingParents, setLoadingParents] = useState<Record<string, boolean>>(
    {},
  );
  const [errorsByParent, setErrorsByParent] = useState<Record<string, string>>(
    {},
  );
  const inFlightRef = useRef(new Set<string>());

  const loadChildren = useCallback(
    async (parentId: string) => {
      if (inFlightRef.current.has(parentId)) return;
      inFlightRef.current.add(parentId);
      setLoadingParents((prev) => ({ ...prev, [parentId]: true }));
      setErrorsByParent((prev) => {
        const next = { ...prev };
        delete next[parentId];
        return next;
      });
      try {
        const response = await api.getSessionChildren(parentId, profile);
        setChildrenByParent((prev) => ({ ...prev, [parentId]: response.children }));
      } catch (error) {
        setErrorsByParent((prev) => ({
          ...prev,
          [parentId]: error instanceof Error ? error.message : "Failed to load child sessions",
        }));
      } finally {
        inFlightRef.current.delete(parentId);
        setLoadingParents((prev) => ({ ...prev, [parentId]: false }));
      }
    },
    [profile],
  );

  const toggleChildren = useCallback(
    (session: SessionInfo) => {
      const id = session.id;
      if (expanded[id]) {
        setExpanded((prev) => ({ ...prev, [id]: false }));
        return;
      }
      setExpanded((prev) => ({ ...prev, [id]: true }));
      if (!childrenByParent[id]) void loadChildren(id);
    },
    [childrenByParent, expanded, loadChildren],
  );

  const renderBranch = (session: SessionInfo, depth: number): ReactNode => {
    const hasChildren = Boolean(
      session.has_children || (session.child_count ?? 0) > 0,
    );
    const isExpanded = Boolean(expanded[session.id]);
    const children = childrenByParent[session.id] ?? [];
    const loading = Boolean(loadingParents[session.id]);
    const error = errorsByParent[session.id] ?? null;

    return (
      <Fragment key={session.id}>
        {renderNode({
          session,
          depth,
          hasChildren,
          expanded: isExpanded,
          loading,
          error,
          onToggleChildren: () => toggleChildren(session),
        })}
        {isExpanded && (
          <div
            className={cn(
              "ml-3 border-l border-border/60 pl-2",
              depth === 0 && "ml-4",
            )}
            data-session-tree-children={session.id}
          >
            {loading && (
              <div className="flex items-center gap-1.5 px-2 py-2 text-xs text-text-tertiary">
                <RefreshCw className="h-3 w-3 animate-spin" />
                Loading child sessions…
              </div>
            )}
            {error && !loading && (
              <div className="flex items-center gap-2 px-2 py-2 text-xs text-destructive" role="alert">
                <span className="min-w-0 flex-1 truncate">{error}</span>
                <button
                  type="button"
                  className="shrink-0 underline underline-offset-2 hover:text-foreground"
                  onClick={() => void loadChildren(session.id)}
                >
                  Retry
                </button>
              </div>
            )}
            {!loading && !error && children.length === 0 && (
              <div className="px-2 py-2 text-xs text-text-tertiary">
                No child sessions
              </div>
            )}
            {!loading && !error && children.map((child) => renderBranch(child, depth + 1))}
          </div>
        )}
      </Fragment>
    );
  };

  return (
    <div className={cn("flex min-w-0 flex-col gap-1.5", className)}>
      {roots.map((root) => renderBranch(root, 0))}
    </div>
  );
}

/** Compact chevron control used by row renderers without duplicating tree state. */
export function SessionTreeToggle({
  hasChildren,
  expanded,
  onClick,
  label,
}: {
  hasChildren: boolean;
  expanded: boolean;
  onClick: () => void;
  label: string;
}) {
  if (!hasChildren) return <span className="h-6 w-6 shrink-0" aria-hidden />;
  return (
    <button
      type="button"
      className="flex h-6 w-6 shrink-0 items-center justify-center rounded text-text-tertiary hover:bg-midground/10 hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary"
      aria-label={label}
      aria-expanded={expanded}
      onClick={(event) => {
        event.stopPropagation();
        onClick();
      }}
    >
      {expanded ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
    </button>
  );
}
