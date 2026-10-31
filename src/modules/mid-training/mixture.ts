/**
 * The mixture runs behind the Data mixture, Arithmetic weight sweep and Late phase cards. Every run
 * trains the full 30 × 30 table from zeros with plain SGD for a fixed budget of steps, drawing each
 * batch from the four domains in the proportions the weights set, on a warmup-stable-decay schedule
 * (or a constant rate). The late phase can switch the proportions when the decay begins, which is the
 * shape of mid-training. Results are cached by their settings, so moving one slider retrains only
 * what that slider changes.
 */
import {
  extendedScheduleFactor,
  tinyCrossEntropy,
  trainTinyExtended,
  type TinyMixturePhase,
} from "@app/module-sdk";
import { ARITHMETIC, DOMAINS, type DomainId } from "./corpora";

export const STEPS = 600;
export const BATCH_SIZE = 16;
/** Character pairs drawn over the run: the token budget the weights divide. */
export const BUDGET_PAIRS = STEPS * BATCH_SIZE;
export const LEARNING_RATE = 3;
export const WARMUP_FRACTION = 0.05;
export const DECAY_FRACTION = 0.2;
export const SEED = 7;
/** The initial table plus one checkpoint every 30 steps. */
export const CHECKPOINTS = 21;
/** The progress at which the decay, and any late mixture, begins. */
export const SWITCH_PROGRESS = 1 - DECAY_FRACTION;
/** The first 0-based step the trainer counts as late: its progress, step / (STEPS − 1), reaches SWITCH_PROGRESS. */
export const SWITCH_STEP = Math.ceil(SWITCH_PROGRESS * (STEPS - 1));

export const WEIGHT_MIN = 0;
export const WEIGHT_MAX = 8;
/** Arithmetic's share of every late batch under Target mix; the rest is the general domains in their own proportions. */
export const TARGET_MIX_SHARE = 0.5;

export type LatePhase = "same" | "mix" | "only";
export type ScheduleKind = "wsd" | "constant";

export const LATE_PHASES: ReadonlyArray<{ value: LatePhase; label: string }> = [
  { value: "same", label: "Same mix" },
  { value: "mix", label: "Target mix" },
  { value: "only", label: "Target only" },
];

export const SCHEDULES: ReadonlyArray<{ value: ScheduleKind; label: string }> = [
  { value: "wsd", label: "Warmup-stable-decay" },
  { value: "constant", label: "Constant rate" },
];

export const PRESETS: ReadonlyArray<{ id: string; label: string; weights: readonly [number, number, number, number] }> = [
  { id: "uniform", label: "Uniform", weights: [1, 1, 1, 1] },
  { id: "general", label: "General-heavy", weights: [3, 3, 3, 1] },
];

const finiteWeight = (value: number) => (Number.isFinite(value) && value > 0 ? value : 0);

/** Weights as shares of every batch. All zero falls back to equal shares, as the trainer does. */
export function sharesOf(weights: ReadonlyArray<number>): number[] {
  const clean = weights.map(finiteWeight);
  const total = clean.reduce((sum, value) => sum + value, 0);
  return total > 0 ? clean.map((value) => value / total) : clean.map(() => 1 / Math.max(1, clean.length));
}

/** The shares every late batch is drawn in. */
export function lateShares(stable: ReadonlyArray<number>, late: LatePhase): number[] {
  if (late === "same") return sharesOf(stable);
  if (late === "only") return [0, 0, 0, 1];
  const general = stable.slice(0, 3).map(finiteWeight);
  const generalTotal = general.reduce((sum, value) => sum + value, 0);
  if (generalTotal === 0) return [0, 0, 0, 1];
  return [...general.map((value) => (value / generalTotal) * (1 - TARGET_MIX_SHARE)), TARGET_MIX_SHARE];
}

/**
 * One constant mixture that draws, on average, the same pairs from each domain as the switched run:
 * the stable shares for 80% of the steps plus the late shares for 20%.
 */
export function evenShares(stable: ReadonlyArray<number>, late: LatePhase): number[] {
  const early = sharesOf(stable);
  const after = lateShares(stable, late);
  return early.map((share, index) => SWITCH_PROGRESS * share + DECAY_FRACTION * after[index]);
}

export function mixturePhases(stable: ReadonlyArray<number>, late: LatePhase): TinyMixturePhase[] {
  const first = { from: 0, weights: sharesOf(stable) };
  return late === "same" ? [first] : [first, { from: SWITCH_PROGRESS, weights: lateShares(stable, late) }];
}

export interface DomainResult {
  id: DomainId;
  /** Pairs the run drew from this domain. */
  seen: number;
  /** Training pairs the domain holds. */
  pairs: number;
  /** seen / pairs: how many passes over the domain's training text the budget bought. */
  epochs: number;
  /** Cross-entropy on the domain's held-out sentences, nats per character. */
  held: number;
}

export interface CurvePoint {
  step: number;
  held: number[];
  general: number;
}

export interface MixtureRun {
  domains: DomainResult[];
  /** The mean held-out loss of Harbor, Recipes and Proverbs, each counted once. */
  general: number;
  arithmetic: number;
  /** Per-domain held-out loss at every checkpoint, from the zero table to the end. */
  curve: CurvePoint[];
  /** The checkpoint at SWITCH_STEP: the table just before any late batch. */
  atSwitch: CurvePoint;
  /** General at the end minus General at the switch: positive when the late phase made it worse. */
  forgetting: number;
  steps: number;
  weights: Float32Array;
}

/** Mean held-out loss of the three general domains. */
export const generalOf = (held: ReadonlyArray<number>) => (held[0] + held[1] + held[2]) / 3;

const heldOutOf = (weights: Float32Array, arithmetic?: number) =>
  DOMAINS.map((domain, index) =>
    index === 3 && arithmetic !== undefined ? arithmetic : tinyCrossEntropy(weights, domain.heldOutText),
  );

function train(phases: TinyMixturePhase[], schedule: ScheduleKind, seed: number): MixtureRun {
  const run = trainTinyExtended({
    // In step mode this text is only what the checkpoints and the final loss read.
    text: ARITHMETIC.heldOutText,
    sources: DOMAINS.map((domain) => ({ id: domain.id, text: domain.trainText })),
    mixture: phases,
    steps: STEPS,
    batchSize: BATCH_SIZE,
    learningRate: LEARNING_RATE,
    schedule,
    warmupFraction: WARMUP_FRACTION,
    decayFraction: DECAY_FRACTION,
    seed,
    checkpoints: CHECKPOINTS,
  });
  const curve = run.checkpoints.map((point) => {
    const held = heldOutOf(point.weights, point.loss);
    return { step: point.step, held, general: generalOf(held) };
  });
  const held = curve[curve.length - 1].held;
  const atSwitch = curve.find((point) => point.step === SWITCH_STEP) ?? curve[0];
  const domains = DOMAINS.map((domain, index) => ({
    id: domain.id,
    seen: run.sourceSeen[index],
    pairs: run.sourcePairs[index],
    epochs: run.sourcePairs[index] > 0 ? run.sourceSeen[index] / run.sourcePairs[index] : 0,
    held: held[index],
  }));
  const general = generalOf(held);
  return {
    domains,
    general,
    arithmetic: held[3],
    curve,
    atSwitch,
    forgetting: general - atSwitch.general,
    steps: run.steps,
    weights: run.weights,
  };
}

/** A small least-recently-used cache: the sweep and the cards share runs with equal settings. */
const CACHE_LIMIT = 120;
const runs = new Map<string, MixtureRun>();

function remember(key: string, compute: () => MixtureRun): MixtureRun {
  const hit = runs.get(key);
  if (hit) {
    runs.delete(key);
    runs.set(key, hit);
    return hit;
  }
  const value = compute();
  runs.set(key, value);
  if (runs.size > CACHE_LIMIT) runs.delete(runs.keys().next().value as string);
  return value;
}

function runPhases(phases: TinyMixturePhase[], schedule: ScheduleKind, seed: number): MixtureRun {
  return remember(JSON.stringify({ phases, schedule, seed }), () => train(phases, schedule, seed));
}

/**
 * The run every card describes: the stable weights, then the late phase from the decay onward. The lab
 * always uses SEED; other seeds exist so tests can measure how much the batch order alone moves a result.
 */
export function runMixture(
  stable: ReadonlyArray<number>,
  late: LatePhase,
  schedule: ScheduleKind,
  seed = SEED,
): MixtureRun {
  return runPhases(mixturePhases(stable, late), schedule, seed);
}

/** The same expected pairs per domain as the switched run, spread evenly over all the steps. */
export function runEvenSpread(
  stable: ReadonlyArray<number>,
  late: LatePhase,
  schedule: ScheduleKind,
  seed = SEED,
): MixtureRun {
  return runPhases([{ from: 0, weights: evenShares(stable, late) }], schedule, seed);
}

export interface SweepPoint {
  weight: number;
  /** Null when all four weights would be zero, a mixture the lab never trains. */
  run: MixtureRun | null;
}

/** One run per Arithmetic weight from 0 to 8, with the three general weights and the late phase as set. */
export function arithmeticSweep(stable: ReadonlyArray<number>, late: LatePhase, schedule: ScheduleKind): SweepPoint[] {
  return Array.from({ length: WEIGHT_MAX - WEIGHT_MIN + 1 }, (_, offset) => {
    const weight = WEIGHT_MIN + offset;
    const weights = [stable[0], stable[1], stable[2], weight];
    const empty = weights.every((value) => finiteWeight(value) === 0);
    return { weight, run: empty ? null : runMixture(weights, late, schedule) };
  });
}

/**
 * Arithmetic's held-out gain per extra epoch between neighbouring weights: the fall in its loss
 * divided by the rise in its epochs. Null for the first weight, or where either run is missing.
 */
export function gainPerEpoch(sweep: ReadonlyArray<SweepPoint>): Array<number | null> {
  return sweep.map((point, index) => {
    const previous = sweep[index - 1]?.run;
    const current = point.run;
    if (!previous || !current) return null;
    const extraEpochs = current.domains[3].epochs - previous.domains[3].epochs;
    if (!(extraEpochs > 0)) return null;
    return (previous.arithmetic - current.arithmetic) / extraEpochs;
  });
}

/** The learning-rate multiplier across the run, as the trainer applies it. */
export function scheduleCurve(schedule: ScheduleKind, points = 81): Array<{ x: number; y: number }> {
  return Array.from({ length: points }, (_, index) => {
    const x = index / (points - 1);
    return { x, y: extendedScheduleFactor(schedule, x, WARMUP_FRACTION, DECAY_FRACTION) };
  });
}

/** Arithmetic's share of each batch across the run: the stable share, then the late share. */
export function arithmeticShareCurve(stable: ReadonlyArray<number>, late: LatePhase): Array<{ x: number; y: number }> {
  const early = sharesOf(stable)[3];
  const after = lateShares(stable, late)[3];
  const edge = 0.002;
  return [
    { x: 0, y: early },
    { x: SWITCH_PROGRESS - edge, y: early },
    { x: SWITCH_PROGRESS + edge, y: after },
    { x: 1, y: after },
  ];
}
