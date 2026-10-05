import type { ModuleState } from "@app/module-sdk";
import { GROUP_RANGE, ITERATIONS, K_RANGE, RATE_RANGE, SEED_RANGE } from "./grpo";

/**
 * Bounds for every control, matching the slider ranges in Explore.tsx. The
 * group-relative training run happens synchronously inside a useMemo, so a
 * shared link must never be able to request an unbounded one: every field is
 * validated or clamped here and `sanitizeState` is the only way state is read.
 */
export const PROBLEMS = ["sunday", "add", "ferry"] as const;
export const SPREADS = ["same", "three", "unique"] as const;
export const BUDGET_MAX = 6;
export const SAMPLES_MAX = 32;

export const initialState: ModuleState = {
  problem: "sunday",
  budget: 3,
  samples: 8,
  accuracy: 0.4,
  spread: "three",
  grpoK: 3,
  grpoGroup: 16,
  grpoRate: 0.3,
  grpoCost: false,
  grpoSeed: 1,
  grpoView: ITERATIONS,
};

const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));

/** A finite number clamped to [low, high], or the fallback for anything else. */
export function boundedNumber(value: unknown, low: number, high: number, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? clamp(value, low, high) : fallback;
}

const boundedInteger = (value: unknown, low: number, high: number, fallback: number) =>
  Math.round(boundedNumber(value, low, high, fallback));

function oneOf<T extends string>(value: unknown, allowed: readonly T[], fallback: T): T {
  return typeof value === "string" && (allowed as readonly string[]).includes(value) ? (value as T) : fallback;
}

/**
 * Rebuilds a state from untrusted input: unknown keys are dropped and every
 * field is validated or clamped to the control's range. Payloads from before
 * the training card existed (stateVersion 2 and earlier) simply lack the
 * `grpo*` keys and receive their defaults.
 */
export function sanitizeState(value: unknown): ModuleState {
  if (!value || typeof value !== "object" || Array.isArray(value)) return { ...initialState };
  const source = value as Record<string, unknown>;
  return {
    problem: oneOf(source.problem, PROBLEMS, initialState.problem as (typeof PROBLEMS)[number]),
    budget: boundedInteger(source.budget, 0, BUDGET_MAX, initialState.budget as number),
    samples: boundedInteger(source.samples, 1, SAMPLES_MAX, initialState.samples as number),
    accuracy: Math.round(boundedNumber(source.accuracy, 0.05, 0.95, initialState.accuracy as number) * 20) / 20,
    spread: oneOf(source.spread, SPREADS, initialState.spread as (typeof SPREADS)[number]),
    grpoK: boundedInteger(source.grpoK, K_RANGE.min, K_RANGE.max, initialState.grpoK as number),
    grpoGroup: boundedInteger(source.grpoGroup, GROUP_RANGE.min, GROUP_RANGE.max, initialState.grpoGroup as number),
    grpoRate:
      Math.round(boundedNumber(source.grpoRate, RATE_RANGE.min, RATE_RANGE.max, initialState.grpoRate as number) * 10) /
      10,
    grpoCost: typeof source.grpoCost === "boolean" ? source.grpoCost : (initialState.grpoCost as boolean),
    grpoSeed: boundedInteger(source.grpoSeed, SEED_RANGE.min, SEED_RANGE.max, initialState.grpoSeed as number),
    grpoView: boundedInteger(source.grpoView, 0, ITERATIONS, initialState.grpoView as number),
  };
}
