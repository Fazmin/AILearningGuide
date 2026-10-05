import type { ModuleState } from "@app/module-sdk";
import {
  DEPTH_CLIP_MAX,
  DEPTH_CLIP_MIN,
  DEPTH_GAIN_MAX,
  DEPTH_GAIN_MIN,
  DEPTH_LAYERS_MAX,
  DEPTH_LAYERS_MIN,
} from "./depth";
import { EPOCHS, type SeedsMode } from "./seeds";

/**
 * Bounds for the controls; they are the slider ranges in Explore.tsx. All four schedules (and, with
 * five seeds on, twenty runs) train inside a useMemo, so every one of these is hard-capped.
 *
 * Version 1 had no inspected epoch. Version 2 added it. Version 3 adds the seed switch and the three
 * depth-card controls. Every earlier key keeps its meaning, so an older payload is clamped into range
 * and the new keys take their defaults.
 */
export const SCHEDULES = ["constant", "cosine", "warmup-decay", "one-cycle"] as const;
export const SEED_MODES: ReadonlyArray<SeedsMode> = ["one", "five"];

export interface HyperState {
  schedule: string;
  peakLearningRate: number;
  warmup: number;
  clipNorm: number;
  weightDecay: number;
  batchSize: number;
  inspect: number;
  seeds: SeedsMode;
  depthLayers: number;
  depthGain: number;
  depthClip: number;
}

export const defaultState: HyperState = {
  schedule: "cosine",
  peakLearningRate: 0.6,
  warmup: 0.15,
  clipNorm: 0,
  weightDecay: 0,
  batchSize: 16,
  inspect: 12,
  seeds: "one",
  depthLayers: 24,
  depthGain: 1.2,
  depthClip: 0,
};

export const initialState: ModuleState = { ...defaultState };

const bounded = (value: unknown, low: number, high: number, fallback: number) =>
  typeof value === "number" && Number.isFinite(value) ? Math.min(high, Math.max(low, value)) : fallback;

/** Rebuilds a typed state from untrusted input: unknown keys dropped, every field validated or clamped. */
export function readState(value: unknown): HyperState {
  if (!value || typeof value !== "object" || Array.isArray(value)) return { ...defaultState };
  const source = value as Record<string, unknown>;
  return {
    schedule:
      typeof source.schedule === "string" && (SCHEDULES as ReadonlyArray<string>).includes(source.schedule)
        ? source.schedule
        : defaultState.schedule,
    peakLearningRate: bounded(source.peakLearningRate, 0.1, 60, defaultState.peakLearningRate),
    warmup: bounded(source.warmup, 0.02, 0.6, defaultState.warmup),
    clipNorm: bounded(source.clipNorm, 0, 1.5, defaultState.clipNorm),
    weightDecay: bounded(source.weightDecay, 0, 0.08, defaultState.weightDecay),
    batchSize: Math.round(bounded(source.batchSize, 2, 64, defaultState.batchSize)),
    inspect: Math.round(bounded(source.inspect, 0, EPOCHS, defaultState.inspect)),
    seeds: source.seeds === "five" ? "five" : "one",
    depthLayers: Math.round(bounded(source.depthLayers, DEPTH_LAYERS_MIN, DEPTH_LAYERS_MAX, defaultState.depthLayers)),
    depthGain: Math.round(bounded(source.depthGain, DEPTH_GAIN_MIN, DEPTH_GAIN_MAX, defaultState.depthGain) * 100) / 100,
    depthClip: Math.round(bounded(source.depthClip, DEPTH_CLIP_MIN, DEPTH_CLIP_MAX, defaultState.depthClip) * 2) / 2,
  };
}

export function hydrateHyperState(value: string): ModuleState {
  try {
    return { ...readState(JSON.parse(value)) };
  } catch {
    return { ...initialState };
  }
}
