export type SearchKind =
  | "module"
  | "group"
  | "learn"
  | "glossary"
  | "checkpoint"
  | "step"
  | "card"
  | "screen";

export type ExplanationMode = "plain" | "standard";

export interface SearchDocument {
  id: string;
  kind: SearchKind;
  title: string;
  subtitle: string;
  body: string;
  moduleId: string | null;
  moduleTitle: string | null;
  moduleSlug: string | null;
  groupId: string | null;
  mode: ExplanationMode | null;
  section: string | null;
  stepIndex: number | null;
  term: string | null;
  cardLabel: string | null;
  screen: "home" | "settings" | null;
  settingsTab: string | null;
  rankBoost: number;
}

export interface SearchHit extends SearchDocument {
  score: number;
  snippet: string;
}

export interface SearchIndexFile {
  version: number;
  schema: string;
  fingerprint: string;
  generatedAt: string;
  documentCount: number;
  documents: SearchDocument[];
}

export type SearchFocus =
  | { kind: "module"; moduleId: string }
  | { kind: "learn"; moduleId: string; mode: ExplanationMode; section: string }
  | { kind: "glossary"; moduleId: string; term: string }
  | { kind: "checkpoint"; moduleId: string }
  | { kind: "step"; moduleId: string; stepIndex: number }
  | { kind: "card"; moduleId: string; label: string }
  | { kind: "screen"; screen: "home" | "settings"; settingsTab?: string; groupId?: string };

export const SEARCH_MARK_START = "«";
export const SEARCH_MARK_END = "»";

export function focusFromDocument(document: SearchDocument): SearchFocus | null {
  if (document.kind === "screen" || document.kind === "group") {
    return {
      kind: "screen",
      screen: document.screen ?? "home",
      settingsTab: document.settingsTab ?? undefined,
      groupId: document.groupId ?? undefined,
    };
  }
  if (!document.moduleId) return null;
  if (document.kind === "learn" && document.section && document.mode) {
    return { kind: "learn", moduleId: document.moduleId, mode: document.mode, section: document.section };
  }
  if (document.kind === "glossary" && document.term) {
    return { kind: "glossary", moduleId: document.moduleId, term: document.term };
  }
  if (document.kind === "checkpoint") {
    return { kind: "checkpoint", moduleId: document.moduleId };
  }
  if (document.kind === "step" && document.stepIndex !== null) {
    return { kind: "step", moduleId: document.moduleId, stepIndex: document.stepIndex };
  }
  if (document.kind === "card" && document.cardLabel) {
    return { kind: "card", moduleId: document.moduleId, label: document.cardLabel };
  }
  return { kind: "module", moduleId: document.moduleId };
}
