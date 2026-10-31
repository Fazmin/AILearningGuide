/**
 * The Repeated data card: train on Arithmetic alone for a fixed budget of pairs, drawing from only the
 * first N of its 120 training sentences, and score the 40 held-out arithmetic sentences. The same
 * budget spread over fewer unique sentences means more passes over each one, so the card separates
 * "more data" from "more passes over the same data". The recipe matches the mixture runs: plain SGD at
 * rate 3, batch 16, warmup-stable-decay, seed 7.
 */
import { trainTinyExtended } from "@app/module-sdk";
import { ARITHMETIC, joinSentences, pairCount } from "./corpora";
import { BATCH_SIZE, DECAY_FRACTION, LEARNING_RATE, SEED, WARMUP_FRACTION } from "./mixture";

export const UNIQUE_MIN = 2;
export const UNIQUE_MAX = ARITHMETIC.train.length;
export const DEFAULT_UNIQUE = UNIQUE_MAX;

/** Budgets are whole multiples of 2,000 pairs, so every budget is a whole number of 16-pair steps. */
export const BUDGET_MIN = 2000;
export const BUDGET_MAX = 16000;
export const BUDGET_STEP = 2000;
export const DEFAULT_BUDGET = 8000;

/** The bars: each halves the unique sentences of the one before, so each roughly doubles the epochs. */
export const SWEEP_UNIQUE: ReadonlyArray<number> = [120, 60, 30, 15, 8, 4, 2];

export const REPEAT_CHECKPOINTS = 21;

export interface RepetitionPoint {
  pairs: number;
  epochs: number;
  held: number;
}

export interface RepetitionRun {
  unique: number;
  budget: number;
  steps: number;
  /** Training pairs in the first `unique` sentences. */
  uniquePairs: number;
  /** budget / uniquePairs: passes over the unique sentences. */
  epochs: number;
  /** Held-out arithmetic loss at the end of the run. */
  held: number;
  curve: RepetitionPoint[];
  /** The lowest held-out loss any checkpoint reached, and where. */
  lowest: RepetitionPoint;
}

export const uniqueText = (unique: number) => joinSentences(ARITHMETIC.train.slice(0, unique));

const runs = new Map<string, RepetitionRun>();
const CACHE_LIMIT = 80;

function train(unique: number, budget: number): RepetitionRun {
  const text = uniqueText(unique);
  const uniquePairs = pairCount(text);
  const steps = Math.round(budget / BATCH_SIZE);
  const run = trainTinyExtended({
    text: ARITHMETIC.heldOutText,
    sources: [{ id: "arithmetic", text }],
    steps,
    batchSize: BATCH_SIZE,
    learningRate: LEARNING_RATE,
    schedule: "wsd",
    warmupFraction: WARMUP_FRACTION,
    decayFraction: DECAY_FRACTION,
    seed: SEED,
    checkpoints: REPEAT_CHECKPOINTS,
  });
  const curve = run.checkpoints.map((point) => {
    const pairs = point.step * BATCH_SIZE;
    return { pairs, epochs: pairs / uniquePairs, held: point.loss };
  });
  const lowest = curve.reduce((best, point) => (point.held < best.held ? point : best), curve[0]);
  return {
    unique,
    budget,
    steps: run.steps,
    uniquePairs,
    epochs: run.sourceSeen[0] / run.sourcePairs[0],
    held: run.finalLoss,
    curve,
    lowest,
  };
}

/** One arithmetic-only run, cached by its two settings. Out-of-range settings are clamped; non-finite ones take the defaults. */
export function runRepetition(unique: number, budget: number): RepetitionRun {
  const finiteUnique = Number.isFinite(unique) ? unique : DEFAULT_UNIQUE;
  const finiteBudget = Number.isFinite(budget) ? budget : DEFAULT_BUDGET;
  const safeUnique = Math.round(Math.min(UNIQUE_MAX, Math.max(UNIQUE_MIN, finiteUnique)));
  const safeBudget = Math.round(Math.min(BUDGET_MAX, Math.max(BUDGET_MIN, finiteBudget)) / BUDGET_STEP) * BUDGET_STEP;
  const key = `${safeUnique}:${safeBudget}`;
  const hit = runs.get(key);
  if (hit) {
    runs.delete(key);
    runs.set(key, hit);
    return hit;
  }
  const value = train(safeUnique, safeBudget);
  runs.set(key, value);
  if (runs.size > CACHE_LIMIT) runs.delete(runs.keys().next().value as string);
  return value;
}

/** The bars at one budget. */
export function repetitionSweep(budget: number): RepetitionRun[] {
  return SWEEP_UNIQUE.map((unique) => runRepetition(unique, budget));
}
