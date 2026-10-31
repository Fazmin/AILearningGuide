import type { ModuleState } from "@app/module-sdk";
import { MAX_EXPERTS, MAX_TOP_K, type Balancing } from "./moe";

/**
 * Every control of the lab, with the bounds its sliders use. The lab trains up to three runs inside a
 * useMemo on every change, so a value from a share link or a hand-edited snapshot must never be able
 * to ask for an unbounded run: `readState` rebuilds the state from scratch, clamping or snapping every
 * field and dropping every key it does not know. Explore reads its state only through it.
 */
export const BALANCING_MODES: ReadonlyArray<Balancing> = ["none", "aux", "bias"];
export const SPECIAL_VIEWS = ["chars", "corpus"] as const;
export type SpecialView = (typeof SPECIAL_VIEWS)[number];

/** Aux weight notches, spaced the way a real sweep is: Switch Transformer used 0.01. */
export const AUX_LADDER = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1] as const;
/** Bias step notches: DeepSeek-V3 used 0.001. */
export const BIAS_LADDER = [0.001, 0.003, 0.01, 0.03, 0.1] as const;
/** Capacity factor notches; 0 switches the cap off. */
export const CAPACITY_LADDER = [0, 1, 1.25, 1.5, 2] as const;

export const EXPERTS_MIN = 1;
export const EXPERTS_MAX = MAX_EXPERTS;
export const TOP_K_MAX = MAX_TOP_K;
export const EPOCHS_MIN = 1;
export const EPOCHS_MAX = 40;
export const SEED_MIN = 1;
export const SEED_MAX = 6;

export interface MoeLabState {
  balancing: Balancing;
  auxWeight: number;
  biasStep: number;
  experts: number;
  topK: number;
  /** Capacity factor; 0 means no cap. */
  capacity: number;
  epochs: number;
  seed: number;
  specialView: SpecialView;
}

export const defaultState: MoeLabState = {
  balancing: "none",
  auxWeight: 0.01,
  biasStep: 0.001,
  experts: 4,
  topK: 1,
  capacity: 0,
  epochs: 30,
  seed: 1,
  specialView: "chars",
};

export const initialState: ModuleState = { ...defaultState };

const finite = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);

/** A whole number clamped to [low, high], or the fallback for anything that is not a finite number. */
const wholeIn = (value: unknown, low: number, high: number, fallback: number) =>
  finite(value) ? Math.min(high, Math.max(low, Math.round(value))) : fallback;

/** The ladder rung closest to the value on a log scale; the fallback for anything that is not a finite number. */
export function nearestLogRung(value: unknown, ladder: ReadonlyArray<number>, fallback: number): number {
  if (!finite(value)) return fallback;
  const target = Math.log(Math.max(1e-9, value));
  let best = ladder[0];
  for (const rung of ladder) {
    if (Math.abs(Math.log(rung) - target) < Math.abs(Math.log(best) - target)) best = rung;
  }
  return best;
}

/** The ladder rung closest to the value; the fallback for anything that is not a finite number. */
export function nearestRung(value: unknown, ladder: ReadonlyArray<number>, fallback: number): number {
  if (!finite(value)) return fallback;
  let best = ladder[0];
  for (const rung of ladder) if (Math.abs(rung - value) < Math.abs(best - value)) best = rung;
  return best;
}

const pick = <T extends string>(value: unknown, allowed: ReadonlyArray<T>, fallback: T): T =>
  typeof value === "string" && (allowed as ReadonlyArray<string>).includes(value) ? (value as T) : fallback;

/** Rebuilds a typed state from untrusted input: unknown keys dropped, every field validated, clamped or snapped. */
export function readState(value: unknown): MoeLabState {
  const source = value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
  const experts = wholeIn(source.experts, EXPERTS_MIN, EXPERTS_MAX, defaultState.experts);
  return {
    balancing: pick(source.balancing, BALANCING_MODES, defaultState.balancing),
    auxWeight: nearestLogRung(source.auxWeight, AUX_LADDER, defaultState.auxWeight),
    biasStep: nearestLogRung(source.biasStep, BIAS_LADDER, defaultState.biasStep),
    experts,
    topK: wholeIn(source.topK, 1, Math.min(TOP_K_MAX, experts), Math.min(defaultState.topK, experts)),
    capacity: nearestRung(source.capacity, CAPACITY_LADDER, defaultState.capacity),
    epochs: wholeIn(source.epochs, EPOCHS_MIN, EPOCHS_MAX, defaultState.epochs),
    seed: wholeIn(source.seed, SEED_MIN, SEED_MAX, defaultState.seed),
    specialView: pick(source.specialView, SPECIAL_VIEWS, defaultState.specialView),
  };
}

/** Version 1 is the only version, so hydration is validation: anything malformed falls back to the defaults. */
export function hydrateMoeState(value: string): ModuleState {
  try {
    return { ...readState(JSON.parse(value)) };
  } catch {
    return { ...initialState };
  }
}
