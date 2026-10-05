import type { ModuleState } from "@app/module-sdk";

export const BASE_VOCAB = 69;
export const FULL_VOCAB = 512;

export const DEFAULT_TEXT = "The king and the queen read the attention paper";

export const initialState: ModuleState = {
  text: DEFAULT_TEXT,
  vocabSize: FULL_VOCAB,
  focus: 1,
  word: "queen",
  analogyA: "king",
  analogyB: "man",
  analogyC: "woman",
  exclude: true,
  projection: "umap",
  mapShow: "neighbours",
  mapFrame: "fit",
};

/** Version 1 stored a 1–5 "vocabulary" level for a length-based splitter. */
const LEGACY_VOCAB = [69, 180, 290, 400, 512];

const clampInt = (value: unknown, fallback: number, low: number, high: number) => {
  const numeric = typeof value === "number" && Number.isFinite(value) ? Math.round(value) : fallback;
  return Math.min(high, Math.max(low, numeric));
};

const asWord = (value: unknown, fallback: string) =>
  typeof value === "string" && /^[a-z]{1,40}$/.test(value.trim().toLowerCase())
    ? value.trim().toLowerCase()
    : fallback;

const asMember = (value: unknown, allowed: readonly string[], fallback: string) =>
  typeof value === "string" && allowed.includes(value) ? value : fallback;

export function hydrateTokensState(value: string): ModuleState {
  let parsed: Record<string, unknown>;
  try {
    const raw = JSON.parse(value) as unknown;
    parsed = raw && typeof raw === "object" && !Array.isArray(raw) ? (raw as Record<string, unknown>) : {};
  } catch {
    return { ...initialState };
  }

  let vocabSize = parsed.vocabSize;
  if (vocabSize === undefined && typeof parsed.vocabulary === "number") {
    vocabSize = LEGACY_VOCAB[clampInt(parsed.vocabulary, 5, 1, 5) - 1];
  }

  return {
    text: typeof parsed.text === "string" ? parsed.text.slice(0, 200) : (initialState.text as string),
    vocabSize: clampInt(vocabSize, FULL_VOCAB, BASE_VOCAB, FULL_VOCAB),
    focus: clampInt(parsed.focus, 1, 0, 400),
    word: asWord(parsed.word, "queen"),
    analogyA: asWord(parsed.analogyA, "king"),
    analogyB: asWord(parsed.analogyB, "man"),
    analogyC: asWord(parsed.analogyC, "woman"),
    exclude: typeof parsed.exclude === "boolean" ? parsed.exclude : true,
    projection: asMember(parsed.projection, ["umap", "pca"], "umap"),
    mapShow: asMember(parsed.mapShow, ["neighbours", "analogy"], "neighbours"),
    mapFrame: asMember(parsed.mapFrame, ["fit", "all"], "fit"),
  };
}
