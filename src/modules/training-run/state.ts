import type { ModuleState } from "@app/module-sdk";
import {
  DEFAULT_MODEL_SIZE,
  DEFAULT_OPTIMIZER,
  DEFAULT_PRECISION,
  MODEL_SIZES,
  OPTIMIZERS,
  PRECISIONS,
  type OptimizerId,
  type PrecisionId,
} from "./memory";

/**
 * Bounds for the controls. They are the slider ranges in Explore.tsx. The loop trains inside a
 * useMemo, so an unbounded epoch count from a shared link would freeze the lab.
 *
 * Version 1 stored { stage, epochs, batchSize, checkpoint }. Version 2 adds the number of single
 * batch steps taken from the selected checkpoint and the three memory-card choices. Every version 1
 * key keeps its meaning, so an older payload is clamped and the new keys take their defaults.
 */
export const STAGE_COUNT = 7;
export const EPOCHS_MIN = 1;
export const EPOCHS_MAX = 60;
export const BATCH_MIN = 2;
export const BATCH_MAX = 64;
/** The checkpoint slider sets a value this large to mean "the last snapshot"; Explore clamps it again. */
export const CHECKPOINT_MAX = 99;
/** How many single steps the learner can chain from one checkpoint before returning to it. */
export const STEPS_TAKEN_MAX = 24;

export interface TrainingRunState {
  stage: number;
  epochs: number;
  batchSize: number;
  checkpoint: number;
  stepsTaken: number;
  memModel: string;
  memPrecision: PrecisionId;
  memOptimizer: OptimizerId;
}

export const defaultState: TrainingRunState = {
  stage: 0,
  epochs: 12,
  batchSize: 16,
  checkpoint: 8,
  stepsTaken: 0,
  memModel: DEFAULT_MODEL_SIZE,
  memPrecision: DEFAULT_PRECISION,
  memOptimizer: DEFAULT_OPTIMIZER,
};

export const initialState: ModuleState = { ...defaultState };

/** A finite number rounded to a whole number and clamped, or the fallback for anything else. */
export function boundedInteger(value: unknown, low: number, high: number, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value)
    ? Math.min(high, Math.max(low, Math.round(value)))
    : fallback;
}

/** One of a closed list of ids, or the fallback. */
export function oneOf<T extends string>(value: unknown, allowed: ReadonlyArray<T>, fallback: T): T {
  return typeof value === "string" && (allowed as ReadonlyArray<string>).includes(value) ? (value as T) : fallback;
}

/** Rebuilds a typed state from untrusted input: unknown keys dropped, every field validated or clamped. */
export function readState(value: unknown): TrainingRunState {
  if (!value || typeof value !== "object" || Array.isArray(value)) return { ...defaultState };
  const source = value as Record<string, unknown>;
  return {
    stage: boundedInteger(source.stage, 0, STAGE_COUNT - 1, defaultState.stage),
    epochs: boundedInteger(source.epochs, EPOCHS_MIN, EPOCHS_MAX, defaultState.epochs),
    batchSize: boundedInteger(source.batchSize, BATCH_MIN, BATCH_MAX, defaultState.batchSize),
    checkpoint: boundedInteger(source.checkpoint, 0, CHECKPOINT_MAX, defaultState.checkpoint),
    stepsTaken: boundedInteger(source.stepsTaken, 0, STEPS_TAKEN_MAX, defaultState.stepsTaken),
    memModel: oneOf(source.memModel, MODEL_SIZES.map((entry) => entry.id), defaultState.memModel),
    memPrecision: oneOf(source.memPrecision, PRECISIONS.map((entry) => entry.id), defaultState.memPrecision),
    memOptimizer: oneOf(source.memOptimizer, OPTIMIZERS.map((entry) => entry.id), defaultState.memOptimizer),
  };
}

export function hydrateTrainingRunState(value: string): ModuleState {
  try {
    return { ...readState(JSON.parse(value)) };
  } catch {
    return { ...initialState };
  }
}
