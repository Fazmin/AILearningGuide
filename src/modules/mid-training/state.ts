import type { ModuleState } from "@app/module-sdk";
import { LATE_PHASES, SCHEDULES, WEIGHT_MAX, WEIGHT_MIN, type LatePhase, type ScheduleKind } from "./mixture";
import {
  BUDGET_MAX,
  BUDGET_MIN,
  BUDGET_STEP,
  DEFAULT_BUDGET,
  DEFAULT_UNIQUE,
  UNIQUE_MAX,
  UNIQUE_MIN,
} from "./repetition";

/**
 * Bounds are the slider ranges in Explore.tsx, so every state the interface can produce passes through
 * readState unchanged. Every card trains inside the render, so each number is hard-capped: the four
 * weights at 0 to 8, the repeated-data budget at 16,000 pairs. Version 1 is the first released shape.
 */
export interface MidTrainingState {
  harborWeight: number;
  recipesWeight: number;
  proverbsWeight: number;
  arithmeticWeight: number;
  latePhase: LatePhase;
  schedule: ScheduleKind;
  uniqueSentences: number;
  repeatBudget: number;
}

export const WEIGHT_KEYS = ["harborWeight", "recipesWeight", "proverbsWeight", "arithmeticWeight"] as const;

/** The default mixture is General-heavy: three parts each of the general domains, one of Arithmetic. */
export const defaultState: MidTrainingState = {
  harborWeight: 3,
  recipesWeight: 3,
  proverbsWeight: 3,
  arithmeticWeight: 1,
  latePhase: "same",
  schedule: "wsd",
  uniqueSentences: DEFAULT_UNIQUE,
  repeatBudget: DEFAULT_BUDGET,
};

export const initialState: ModuleState = { ...defaultState };

const bounded = (value: unknown, low: number, high: number, fallback: number) =>
  typeof value === "number" && Number.isFinite(value) ? Math.min(high, Math.max(low, value)) : fallback;

const oneOf = <T extends string>(value: unknown, allowed: ReadonlyArray<{ value: T }>, fallback: T): T =>
  typeof value === "string" && allowed.some((entry) => entry.value === value) ? (value as T) : fallback;

/** Rebuilds a typed state from untrusted input: unknown keys dropped, every field validated or clamped. */
export function readState(value: unknown): MidTrainingState {
  if (!value || typeof value !== "object" || Array.isArray(value)) return { ...defaultState };
  const source = value as Record<string, unknown>;
  const weights = WEIGHT_KEYS.map((key) =>
    Math.round(bounded(source[key], WEIGHT_MIN, WEIGHT_MAX, defaultState[key])),
  );
  // A mixture with every weight at zero has nothing to draw from; it falls back to Uniform.
  const [harborWeight, recipesWeight, proverbsWeight, arithmeticWeight] = weights.every((weight) => weight === 0)
    ? [1, 1, 1, 1]
    : weights;
  return {
    harborWeight,
    recipesWeight,
    proverbsWeight,
    arithmeticWeight,
    latePhase: oneOf(source.latePhase, LATE_PHASES, defaultState.latePhase),
    schedule: oneOf(source.schedule, SCHEDULES, defaultState.schedule),
    uniqueSentences: Math.round(bounded(source.uniqueSentences, UNIQUE_MIN, UNIQUE_MAX, defaultState.uniqueSentences)),
    repeatBudget:
      Math.round(bounded(source.repeatBudget, BUDGET_MIN, BUDGET_MAX, defaultState.repeatBudget) / BUDGET_STEP) *
      BUDGET_STEP,
  };
}

export function hydrateMidTrainingState(value: string): ModuleState {
  try {
    return { ...readState(JSON.parse(value)) };
  } catch {
    return { ...initialState };
  }
}

/** The four mixture weights in domain order: Harbor, Recipes, Proverbs, Arithmetic. */
export const weightsOf = (state: MidTrainingState): [number, number, number, number] => [
  state.harborWeight,
  state.recipesWeight,
  state.proverbsWeight,
  state.arithmeticWeight,
];
