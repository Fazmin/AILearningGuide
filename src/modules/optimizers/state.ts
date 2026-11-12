import type { ModuleState } from "@app/module-sdk";
import {
  DECAY_LADDER,
  NS_MAX,
  OPTIMIZER_IDS,
  rateLadder,
  snapToLadder,
  WARMUP_MAX,
  WARMUP_STEP,
  type CaptureId,
  type DecayMode,
  type OptimizerId,
} from "./race";
import { RAVINE_STEPS, type RavineRates, type RavineTurn } from "./ravine";

/**
 * The lab's controls. Every number is a notch on a slider, so `readState` snaps it to the nearest notch
 * rather than only clamping it: a run is cached by its settings, and a value between notches would
 * train a run that no control can reach. Version 1.
 */
export interface OptimizersState {
  /** Which optimizer Step size per row shows, and which curve the race draws thick. */
  optimizer: OptimizerId;
  curves: "train" | "held";
  sgdRate: number;
  momentumRate: number;
  adamRate: number;
  muonRate: number;
  biasCorrection: "on" | "off";
  /** Share of the AdamW run spent warming up, 0 to 0.3. */
  warmup: number;
  weightDecay: number;
  decayMode: DecayMode;
  capture: CaptureId;
  nsSteps: number;
  /** A ravine in two weights: which way it runs, the rates the three walks use, and how far they have walked. */
  ravineTurn: RavineTurn;
  ravineRates: RavineRates;
  ravineStep: number;
}

export const defaultState: OptimizersState = {
  optimizer: "sgd",
  curves: "train",
  sgdRate: 1,
  momentumRate: 0.1,
  adamRate: 0.03,
  muonRate: 0.05,
  biasCorrection: "on",
  warmup: 0,
  weightDecay: 0,
  decayMode: "decoupled",
  capture: "middle",
  nsSteps: 5,
  ravineTurn: "axis",
  ravineRates: "default",
  ravineStep: 0,
};

export const initialState: ModuleState = { ...defaultState };

const pick = <T extends string>(value: unknown, allowed: ReadonlyArray<T>, fallback: T): T =>
  typeof value === "string" && (allowed as ReadonlyArray<string>).includes(value) ? (value as T) : fallback;

const bounded = (value: unknown, low: number, high: number, fallback: number) =>
  typeof value === "number" && Number.isFinite(value) ? Math.min(high, Math.max(low, value)) : fallback;

/** Warmup moves in 5% notches from 0 to 30% of the run. */
const warmupNotch = (value: unknown) =>
  Number((Math.round(bounded(value, 0, WARMUP_MAX, defaultState.warmup) / WARMUP_STEP) * WARMUP_STEP).toFixed(2));

/** Rebuilds a typed state from untrusted input: unknown keys dropped, every field validated or snapped. */
export function readState(value: unknown): OptimizersState {
  if (!value || typeof value !== "object" || Array.isArray(value)) return { ...defaultState };
  const source = value as Record<string, unknown>;
  return {
    optimizer: pick(source.optimizer, OPTIMIZER_IDS, defaultState.optimizer),
    curves: pick(source.curves, ["train", "held"] as const, defaultState.curves),
    sgdRate: snapToLadder(source.sgdRate, rateLadder("sgd"), defaultState.sgdRate),
    momentumRate: snapToLadder(source.momentumRate, rateLadder("momentum"), defaultState.momentumRate),
    adamRate: snapToLadder(source.adamRate, rateLadder("adamw"), defaultState.adamRate),
    muonRate: snapToLadder(source.muonRate, rateLadder("muon"), defaultState.muonRate),
    biasCorrection: pick(source.biasCorrection, ["on", "off"] as const, defaultState.biasCorrection),
    warmup: warmupNotch(source.warmup),
    weightDecay: snapToLadder(source.weightDecay, DECAY_LADDER, defaultState.weightDecay),
    decayMode: pick(source.decayMode, ["decoupled", "l2"] as const, defaultState.decayMode),
    capture: pick(source.capture, ["first", "middle"] as const, defaultState.capture),
    nsSteps: Math.round(bounded(source.nsSteps, 0, NS_MAX, defaultState.nsSteps)),
    ravineTurn: pick(source.ravineTurn, ["axis", "diagonal"] as const, defaultState.ravineTurn),
    ravineRates: pick(source.ravineRates, ["half", "default", "double"] as const, defaultState.ravineRates),
    ravineStep: Math.round(bounded(source.ravineStep, 0, RAVINE_STEPS, defaultState.ravineStep)),
  };
}

export function hydrateOptimizersState(value: string): ModuleState {
  try {
    return { ...readState(JSON.parse(value)) };
  } catch {
    return { ...initialState };
  }
}
