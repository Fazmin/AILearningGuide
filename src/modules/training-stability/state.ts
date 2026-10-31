import type { ModuleState } from "@app/module-sdk";
import {
  ADAM_RATES,
  CLIP_MAX,
  CLIP_STEP,
  DEFAULT_ADAM_RATE,
  DEFAULT_SCALE,
  FAULT_IDS,
  SCALE_MAX,
  SCALE_MIN,
  SCALING_IDS,
  SCHEDULE_IDS,
  Z_LOSSES,
  type FaultSettings,
} from "./faults";

/**
 * Every control of the lab, validated. The dashboard trains up to three runs synchronously inside a
 * useMemo, so a value from a share link or a hand-edited snapshot must never reach the trainer
 * unchecked: `readState` rebuilds the state from scratch, keeping only known keys, whitelisting the
 * choices, and clamping or snapping every number to a value a control can produce. Explore reads its
 * state only through it.
 *
 * Version 1 is the first released shape, so there is nothing older to translate.
 */
export const defaultState: FaultSettings = {
  fault: "none",
  clipNorm: 0,
  schedule: "constant",
  lossScale: DEFAULT_SCALE,
  scaling: "static",
  rollback: false,
  zLoss: 0,
  adamRate: DEFAULT_ADAM_RATE,
};

export const initialState: ModuleState = { ...defaultState };

const pick = <T extends string>(value: unknown, allowed: ReadonlyArray<T>, fallback: T): T =>
  typeof value === "string" && (allowed as ReadonlyArray<string>).includes(value) ? (value as T) : fallback;

const finite = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);

/** The rung nearest a positive value on a log scale; zero and below snap to the lowest rung. */
export function nearestRung(value: unknown, ladder: ReadonlyArray<number>, fallback: number): number {
  if (!finite(value)) return fallback;
  const positive = ladder.filter((rung) => rung > 0);
  if (ladder.includes(0) && value <= 0) return 0;
  if (value <= 0) return positive[0] ?? fallback;
  const target = Math.log(value);
  let best = positive[0] ?? fallback;
  for (const rung of positive) {
    if (Math.abs(Math.log(rung) - target) < Math.abs(Math.log(best) - target)) best = rung;
  }
  // A z-loss far below the smallest rung reads as off rather than as the smallest rung.
  if (ladder.includes(0) && value < (positive[0] ?? 0) / 10) return 0;
  return best;
}

/** Rebuilds a typed state from untrusted input: unknown keys dropped, every field validated, clamped or snapped. */
export function readState(value: unknown): FaultSettings {
  const source = value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
  const clip = finite(source.clipNorm) ? Math.min(CLIP_MAX, Math.max(0, source.clipNorm)) : defaultState.clipNorm;
  const scale = finite(source.lossScale) ? Math.min(SCALE_MAX, Math.max(SCALE_MIN, source.lossScale)) : defaultState.lossScale;
  return {
    fault: pick(source.fault, FAULT_IDS, defaultState.fault),
    clipNorm: Number((Math.round(clip / CLIP_STEP) * CLIP_STEP).toFixed(2)),
    schedule: pick(source.schedule, SCHEDULE_IDS, defaultState.schedule),
    lossScale: Math.round(scale),
    scaling: pick(source.scaling, SCALING_IDS, defaultState.scaling),
    rollback: source.rollback === true,
    zLoss: nearestRung(source.zLoss, Z_LOSSES, defaultState.zLoss),
    adamRate: nearestRung(source.adamRate, ADAM_RATES, defaultState.adamRate),
  };
}

/** The state exactly as the lab stores it. */
export function sanitizeState(value: unknown): ModuleState {
  return { ...readState(value) };
}

export function hydrateStabilityState(value: string): ModuleState {
  try {
    return sanitizeState(JSON.parse(value));
  } catch {
    return { ...initialState };
  }
}
