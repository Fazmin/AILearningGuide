import type { ModuleState } from "@app/module-sdk";
import type { Checker, RewardDesign, TicketSet } from "./env";
import { COST_RANGE, GROUP_RANGE, KL_RANGE, SEED_RANGE, UPDATES, type AdvantageScale } from "./train";

/**
 * Bounds for every control, matching the controls in Explore.tsx. The training run and the
 * 10-seed strip happen synchronously inside useMemo, so a shared link must never be able to
 * request an unbounded one: every field is validated, clamped, or snapped to its slider's grid
 * here, unknown keys are dropped, and Explore reads state only through `readState`.
 *
 * Version 1 is the first release, so there is nothing older to translate.
 */
export const REWARDS = ["outcome", "partial", "judge"] as const satisfies readonly RewardDesign[];
export const CHECKERS = ["strict", "sandbox"] as const satisfies readonly Checker[];
export const TICKET_SETS = ["one", "three", "six"] as const satisfies readonly TicketSet[];
export const SCALES = ["std", "none"] as const satisfies readonly AdvantageScale[];
export const EPISODE_RANGE = { min: 1, max: 20 } as const;

export interface RlState {
  reward: RewardDesign;
  checker: Checker;
  tickets: TicketSet;
  group: number;
  kl: number;
  stepCost: number;
  scale: AdvantageScale;
  seed: number;
  view: number;
  episode: number;
}

const DEFAULTS: RlState = {
  reward: "outcome",
  checker: "strict",
  tickets: "six",
  group: 8,
  kl: 0,
  stepCost: 0,
  scale: "std",
  seed: 1,
  view: UPDATES,
  episode: 1,
};

export const initialState: ModuleState = { ...DEFAULTS };

const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));

/** A finite number clamped to [low, high], or the fallback for anything else. */
export function boundedNumber(value: unknown, low: number, high: number, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? clamp(value, low, high) : fallback;
}

const boundedInteger = (value: unknown, low: number, high: number, fallback: number) =>
  Math.round(boundedNumber(value, low, high, fallback));

/** Clamps, then snaps to a slider grid of `1 / perUnit`, so 0.15 stays 0.15 and 0.149 becomes 0.15. */
const snapped = (value: unknown, low: number, high: number, fallback: number, perUnit: number) =>
  Math.round(boundedNumber(value, low, high, fallback) * perUnit) / perUnit;

function oneOf<T extends string>(value: unknown, allowed: readonly T[], fallback: T): T {
  return typeof value === "string" && (allowed as readonly string[]).includes(value) ? (value as T) : fallback;
}

/** Rebuilds a typed state from untrusted input. */
export function readState(value: unknown): RlState {
  if (!value || typeof value !== "object" || Array.isArray(value)) return { ...DEFAULTS };
  const source = value as Record<string, unknown>;
  return {
    reward: oneOf(source.reward, REWARDS, DEFAULTS.reward),
    checker: oneOf(source.checker, CHECKERS, DEFAULTS.checker),
    tickets: oneOf(source.tickets, TICKET_SETS, DEFAULTS.tickets),
    group: boundedInteger(source.group, GROUP_RANGE.min, GROUP_RANGE.max, DEFAULTS.group),
    kl: snapped(source.kl, KL_RANGE.min, KL_RANGE.max, DEFAULTS.kl, 20),
    stepCost: snapped(source.stepCost, COST_RANGE.min, COST_RANGE.max, DEFAULTS.stepCost, 100),
    scale: oneOf(source.scale, SCALES, DEFAULTS.scale),
    seed: boundedInteger(source.seed, SEED_RANGE.min, SEED_RANGE.max, DEFAULTS.seed),
    view: boundedInteger(source.view, 0, UPDATES, DEFAULTS.view),
    episode: boundedInteger(source.episode, EPISODE_RANGE.min, EPISODE_RANGE.max, DEFAULTS.episode),
  };
}

export const sanitizeState = (value: unknown): ModuleState => ({ ...readState(value) });

export function hydrateRlEnvironmentsState(value: string): ModuleState {
  try {
    return sanitizeState(JSON.parse(value));
  } catch {
    return { ...initialState };
  }
}
