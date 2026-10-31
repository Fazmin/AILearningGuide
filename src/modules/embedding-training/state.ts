import type { ModuleState } from "@app/module-sdk";
import { DEFAULT_QUERY_ID, HELD_OUT_QUERIES } from "./data";
import { WIDTHS } from "./encoder";
import { BATCH_SIZES, NEGATIVE_MODES, SNAPSHOT_EVERY, STEPS, type NegativeMode } from "./train";

/**
 * The lab's controls. Training runs synchronously inside a useMemo, so every number a shared link can
 * carry is clamped or snapped to a value the controls can produce: a step on the 10-step grid, a batch
 * size and width from their short lists, a temperature on the slider's 0.01 grid, a seed from 1 to 20.
 * Version 1 is the first released shape, so nothing older needs translating.
 */
export const TEMPERATURE_MIN = 0.02;
export const TEMPERATURE_MAX = 1;
export const SEED_MIN = 1;
export const SEED_MAX = 20;

export interface TrainingState {
  step: number;
  batch: number;
  temperature: number;
  negatives: NegativeMode;
  skipDuplicates: boolean;
  dim: number;
  seed: number;
  query: string;
  rerank: boolean;
}

export const defaultState: TrainingState = {
  step: STEPS,
  batch: 8,
  temperature: 0.1,
  negatives: "in-batch",
  skipDuplicates: false,
  dim: 16,
  seed: 1,
  query: DEFAULT_QUERY_ID,
  rerank: false,
};

export const initialState: ModuleState = { ...defaultState };

const finite = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);

/** The allowed value nearest to a finite number, or the fallback for anything else. */
export function nearestOf(value: unknown, allowed: readonly number[], fallback: number): number {
  if (!finite(value)) return fallback;
  return allowed.reduce((best, option) => (Math.abs(option - value) < Math.abs(best - value) ? option : best), allowed[0]);
}

const bounded = (value: unknown, low: number, high: number, fallback: number) =>
  finite(value) ? Math.min(high, Math.max(low, value)) : fallback;

/** Rebuilds a typed state from untrusted input: unknown keys dropped, every field validated, clamped or snapped. */
export function readState(value: unknown): TrainingState {
  if (!value || typeof value !== "object" || Array.isArray(value)) return { ...defaultState };
  const source = value as Record<string, unknown>;
  return {
    step: Math.round(bounded(source.step, 0, STEPS, defaultState.step) / SNAPSHOT_EVERY) * SNAPSHOT_EVERY,
    batch: nearestOf(source.batch, BATCH_SIZES, defaultState.batch),
    temperature: Math.round(bounded(source.temperature, TEMPERATURE_MIN, TEMPERATURE_MAX, defaultState.temperature) * 100) / 100,
    negatives:
      typeof source.negatives === "string" && (NEGATIVE_MODES as readonly string[]).includes(source.negatives)
        ? (source.negatives as NegativeMode)
        : defaultState.negatives,
    skipDuplicates: typeof source.skipDuplicates === "boolean" ? source.skipDuplicates : defaultState.skipDuplicates,
    dim: nearestOf(source.dim, WIDTHS, defaultState.dim),
    seed: Math.round(bounded(source.seed, SEED_MIN, SEED_MAX, defaultState.seed)),
    query:
      typeof source.query === "string" && HELD_OUT_QUERIES.some((query) => query.id === source.query)
        ? source.query
        : defaultState.query,
    rerank: typeof source.rerank === "boolean" ? source.rerank : defaultState.rerank,
  };
}

export function hydrateEmbeddingTrainingState(value: string): ModuleState {
  try {
    return { ...readState(JSON.parse(value)) };
  } catch {
    return { ...initialState };
  }
}
