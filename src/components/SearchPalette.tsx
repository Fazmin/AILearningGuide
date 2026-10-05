import {
  BookOpen,
  Bookmark,
  CircleHelp,
  FlaskConical,
  Home,
  ListTodo,
  Map,
  Search,
  Settings,
  SlidersHorizontal,
  X,
} from "lucide-react";
import { useEffect, useId, useMemo, useRef, useState } from "react";
import type { CSSProperties } from "react";
import { querySearchIndex } from "@app/search/client";
import { navigateToSearchHit } from "@app/search/navigate";
import type { SearchHit, SearchKind } from "@app/search/types";
import { useAppStore } from "@app/store/app-store";
import { IconButton } from "./IconButton";

const kindMeta: Record<SearchKind, { label: string; Icon: typeof Search }> = {
  module: { label: "Lab", Icon: FlaskConical },
  group: { label: "Path", Icon: Map },
  learn: { label: "Learn", Icon: BookOpen },
  glossary: { label: "Glossary", Icon: Bookmark },
  checkpoint: { label: "Check", Icon: CircleHelp },
  step: { label: "Step", Icon: ListTodo },
  card: { label: "Instrument", Icon: SlidersHorizontal },
  screen: { label: "App", Icon: Settings },
};

function shortcutLabel() {
  return /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent) ? "⌘K" : "Ctrl K";
}

function HighlightedText({ value }: { value: string }) {
  const parts = value.split(/(«[^»]*»)/g).filter(Boolean);
  return (
    <>
      {parts.map((part, index) =>
        part.startsWith("«") && part.endsWith("»") ? (
          <mark key={`${part}-${index}`}>{part.slice(1, -1)}</mark>
        ) : (
          <span key={`${part}-${index}`}>{part}</span>
        ),
      )}
    </>
  );
}

function ResultIcon({ hit }: { hit: SearchHit }) {
  if (hit.kind === "screen" && hit.screen === "home") return <Home />;
  const Icon = kindMeta[hit.kind].Icon;
  return <Icon />;
}

export function SearchPalette() {
  const searchOpen = useAppStore((state) => state.searchOpen);
  const setSearchOpen = useAppStore((state) => state.setSearchOpen);
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<SearchHit[]>([]);
  const [active, setActive] = useState(0);
  const [searching, setSearching] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const listId = useId();
  const shortcut = useMemo(() => shortcutLabel(), []);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (!(event.metaKey || event.ctrlKey) || event.key.toLowerCase() !== "k") return;
      event.preventDefault();
      const current = useAppStore.getState();
      current.setSearchOpen(!current.searchOpen);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    if (!searchOpen) {
      setQuery("");
      setHits([]);
      setActive(0);
      return;
    }
    const frame = window.requestAnimationFrame(() => inputRef.current?.focus());
    return () => window.cancelAnimationFrame(frame);
  }, [searchOpen]);

  useEffect(() => {
    if (!searchOpen) return;
    if (!query.trim()) {
      setHits([]);
      setActive(0);
      setSearching(false);
      return;
    }
    let cancelled = false;
    setSearching(true);
    void querySearchIndex(query).then((results) => {
      if (cancelled) return;
      setHits(results);
      setActive(0);
      setSearching(false);
    });
    return () => {
      cancelled = true;
    };
  }, [query, searchOpen]);

  const jump = (hit: SearchHit) => {
    navigateToSearchHit(useAppStore.getState(), hit);
  };

  if (!searchOpen) return null;

  return (
    <div
      className="search-overlay"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) setSearchOpen(false);
      }}
    >
      <div
        className="search-palette"
        role="dialog"
        aria-modal="true"
        aria-label="Search the field guide"
        onKeyDown={(event) => {
          if (event.key === "Escape") {
            event.stopPropagation();
            setSearchOpen(false);
            return;
          }
          if (event.key === "ArrowDown") {
            event.preventDefault();
            setActive((value) => Math.min(hits.length - 1, value + 1));
            return;
          }
          if (event.key === "ArrowUp") {
            event.preventDefault();
            setActive((value) => Math.max(0, value - 1));
            return;
          }
          if (event.key === "Enter" && hits[active]) {
            event.preventDefault();
            jump(hits[active]);
          }
        }}
      >
        <div className="search-palette__grid" aria-hidden="true" />
        <label className="search-palette__query">
          <Search />
          <input
            ref={inputRef}
            value={query}
            placeholder="Search labs, glossary, explanations…"
            aria-label="Search the field guide"
            aria-controls={listId}
            aria-activedescendant={hits[active] ? `${listId}-${hits[active].id}` : undefined}
            autoComplete="off"
            spellCheck={false}
            onChange={(event) => setQuery(event.target.value)}
          />
          <kbd>{shortcut}</kbd>
          <IconButton label="Close search" onClick={() => setSearchOpen(false)}>
            <X />
          </IconButton>
        </label>

        {query.trim() ? (
          <div className="search-palette__meta">
            <span>Ranked matches</span>
            <small>{searching ? "Looking…" : `${String(hits.length).padStart(2, "0")} results`}</small>
          </div>
        ) : null}

        <ul className="search-palette__results" id={listId} role="listbox" aria-label="Search results" hidden={!query.trim()}>
          {query.trim() && hits.length === 0 && !searching && (
            <li className="search-palette__empty">
              Nothing in the packaged index matches that. Try a lab name, a glossary term, or a phrase from Learn.
            </li>
          )}
          {hits.map((hit, index) => (
            <li key={hit.id} role="presentation">
              <button
                type="button"
                id={`${listId}-${hit.id}`}
                role="option"
                aria-selected={index === active}
                className={index === active ? "is-active" : ""}
                style={
                  {
                    "--result-accent":
                      hit.kind === "module" || hit.kind === "card"
                        ? "var(--forward)"
                        : hit.kind === "glossary"
                          ? "var(--attention)"
                          : hit.kind === "step"
                            ? "var(--positive)"
                            : hit.kind === "checkpoint"
                              ? "var(--loss)"
                              : "var(--muted)",
                  } as CSSProperties
                }
                onMouseEnter={() => setActive(index)}
                onClick={() => jump(hit)}
              >
                <b>{String(index + 1).padStart(2, "0")}</b>
                <i>
                  <ResultIcon hit={hit} />
                </i>
                <span>
                  <strong>{hit.title}</strong>
                  <em>
                    {kindMeta[hit.kind].label}
                    {hit.moduleTitle ? ` · ${hit.moduleTitle}` : hit.subtitle ? ` · ${hit.subtitle}` : ""}
                  </em>
                  {hit.snippet ? (
                    <small>
                      <HighlightedText value={hit.snippet} />
                    </small>
                  ) : null}
                </span>
              </button>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
