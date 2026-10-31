/**
 * The optimizer race and everything the other cards read from it.
 *
 * Every run trains the whole 30 × 30 bigram table with `trainTinyExtended`: the first six of Harbor's
 * eight sentences (378 training pairs), batches of 16 for 25 epochs (600 steps), shuffle seed 1, from
 * zeroed weights. The last two sentences (118 pairs) are never trained on; they are the held-out text.
 * Only the optimizer and its settings change between runs, so every difference on the page is the
 * optimizer's doing. Runs are cached at module scope, so moving one optimizer's slider retrains only
 * that optimizer.
 */
import {
  newtonSchulz,
  teachingMemory,
  tinyCrossEntropy,
  TINY_CORPORA,
  TINY_VOCAB,
  TINY_VOCAB_SIZE,
  trainTinyExtended,
  type TinyOptimizerKind,
} from "@app/module-sdk";
import { singularValues, summarizeSpectrum, type SpectrumSummary } from "./spectrum";

const V = TINY_VOCAB_SIZE;

const SENTENCES = TINY_CORPORA.harbor.text
  .split(".")
  .map((part) => part.trim())
  .filter(Boolean);
/** The last quarter of Harbor (two of eight sentences) is held out, as in Ship it. */
export const HELD_OUT_SENTENCES = Math.max(1, Math.round(SENTENCES.length * 0.25));
export const TRAIN_TEXT = `${SENTENCES.slice(0, SENTENCES.length - HELD_OUT_SENTENCES).join(". ")}.`;
export const HELD_OUT_TEXT = `${SENTENCES.slice(SENTENCES.length - HELD_OUT_SENTENCES).join(". ")}.`;

export const EPOCHS = 25;
export const BATCH_SIZE = 16;
export const SEED = 1;
/** 378 training pairs in batches of 16: 23 full batches and one of 10. */
export const STEPS_PER_EPOCH = 24;
export const TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH;
/** "Early" means the first tenth of the run, which is also the longest warmup the lesson uses. */
export const EARLY_STEPS = 60;

export type OptimizerId = "sgd" | "momentum" | "adamw" | "muon";
export type DecayMode = "decoupled" | "l2";
export type CaptureId = "first" | "middle";
export type RateKey = "sgdRate" | "momentumRate" | "adamRate" | "muonRate";

export interface OptimizerChoice {
  id: OptimizerId;
  label: string;
  /** The state key holding this optimizer's learning rate, and the slider's label. */
  rateKey: RateKey;
  rateLabel: string;
  defaultRate: number;
  /** fp32 buffers the optimizer keeps per weight between steps. */
  buffers: number;
  /** The teachingMemory row that counts the same buffers: Muon keeps one, like momentum. */
  memoryRow: teachingMemory.OptimizerId;
  detail: string;
}

export const OPTIMIZERS: ReadonlyArray<OptimizerChoice> = [
  {
    id: "sgd",
    label: "SGD",
    rateKey: "sgdRate",
    rateLabel: "SGD rate",
    defaultRate: 1,
    buffers: 0,
    memoryRow: "sgd",
    detail: "Moves each weight by rate × gradient and keeps nothing between steps.",
  },
  {
    id: "momentum",
    label: "Momentum",
    rateKey: "momentumRate",
    rateLabel: "Momentum rate",
    defaultRate: 0.1,
    buffers: 1,
    memoryRow: "momentum",
    detail: "Keeps a velocity, 0.9 × the last one plus the new gradient, and moves by rate × velocity.",
  },
  {
    id: "adamw",
    label: "AdamW",
    rateKey: "adamRate",
    rateLabel: "AdamW rate",
    defaultRate: 0.03,
    buffers: 2,
    memoryRow: "adam",
    detail: "Keeps running averages of each weight's gradient and squared gradient, and divides one by the root of the other.",
  },
  {
    id: "muon",
    label: "Muon",
    rateKey: "muonRate",
    rateLabel: "Muon rate",
    defaultRate: 0.05,
    buffers: 1,
    memoryRow: "momentum",
    detail: "Keeps one momentum matrix and orthogonalizes it with five Newton–Schulz iterations before each step.",
  },
];

export const OPTIMIZER_IDS: ReadonlyArray<OptimizerId> = OPTIMIZERS.map((entry) => entry.id);

export const optimizerChoice = (id: OptimizerId): OptimizerChoice =>
  OPTIMIZERS.find((entry) => entry.id === id) ?? OPTIMIZERS[0];

const significant = (value: number) => Number(value.toPrecision(6));

/** Each rate slider has nine notches, from a tenth of the optimizer's default to ten times it. */
export const RATE_FACTORS = [0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10] as const;
/** Tune each trains every optimizer at five of those notches, about half a decade apart. */
export const SWEEP_FACTORS = [0.1, 0.3, 1, 3, 10] as const;

export const rateLadder = (id: OptimizerId): number[] =>
  RATE_FACTORS.map((factor) => significant(optimizerChoice(id).defaultRate * factor));
export const sweepRates = (id: OptimizerId): number[] =>
  SWEEP_FACTORS.map((factor) => significant(optimizerChoice(id).defaultRate * factor));

/** Weight decay notches; 0 is off. */
export const DECAY_LADDER = [0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3] as const;
export const WARMUP_MAX = 0.3;
export const WARMUP_STEP = 0.05;
export const NS_MAX = 8;

export const CAPTURES: Record<CaptureId, { label: string; step: number }> = {
  first: { label: "Step 1", step: 0 },
  middle: { label: "Step 300", step: 299 },
};

/** The nearest notch in log space; non-numbers and non-finite values fall back. */
export function snapToLadder(value: unknown, ladder: ReadonlyArray<number>, fallback: number): number {
  if (typeof value !== "number" || !Number.isFinite(value)) return fallback;
  const positive = ladder.filter((rung) => rung > 0);
  if (ladder[0] === 0 && value <= positive[0] / 2) return 0;
  if (value <= positive[0]) return positive[0];
  if (value >= positive[positive.length - 1]) return positive[positive.length - 1];
  let best = positive[0];
  let distance = Number.POSITIVE_INFINITY;
  for (const rung of positive) {
    const gap = Math.abs(Math.log(rung) - Math.log(value));
    if (gap < distance) {
      best = rung;
      distance = gap;
    }
  }
  return best;
}

/** Index of a value on its ladder, for the sliders. */
export const ladderIndex = (value: number, ladder: ReadonlyArray<number>) =>
  Math.max(0, ladder.findIndex((rung) => rung === snapToLadder(value, ladder, ladder[0])));

/** Up to three significant figures, no trailing zeros: 0.009, 0.015, 1, 10. */
export function formatRate(value: number): string {
  if (!Number.isFinite(value)) return "—";
  if (value === 0) return "0";
  return String(Number(value.toPrecision(3)));
}

const SUPERSCRIPT: Record<string, string> = {
  "-": "⁻",
  "0": "⁰",
  "1": "¹",
  "2": "²",
  "3": "³",
  "4": "⁴",
  "5": "⁵",
  "6": "⁶",
  "7": "⁷",
  "8": "⁸",
  "9": "⁹",
};

/** Two significant figures, in scientific form below 0.001: 0.030, 0.0030, 3.0 × 10⁻⁶. */
export function formatSmall(value: number): string {
  if (!Number.isFinite(value)) return "—";
  if (value === 0) return "0";
  if (Math.abs(value) < 0.001) {
    const [mantissa, exponent] = value.toExponential(1).split("e");
    return `${mantissa} × 10${String(Number(exponent)).replace(/./g, (character) => SUPERSCRIPT[character] ?? character)}`;
  }
  return value.toPrecision(2);
}

/* ------------------------------------------------------------------------------------------------ */
/* Runs                                                                                              */
/* ------------------------------------------------------------------------------------------------ */

export interface RunSpec {
  optimizer: OptimizerId;
  rate: number;
  /** AdamW only: divide each moment by 1 − βᵗ. */
  biasCorrection: boolean;
  /** AdamW only: share of the run spent ramping up, then constant. */
  warmup: number;
  /** AdamW only. */
  weightDecay: number;
  /** AdamW only: `l2` adds decay × weight to the gradient (the trainer's "adam"); `decoupled` is AdamW. */
  decayMode: DecayMode;
  epochs: number;
  /** 0-based step whose Muon direction and update are kept, or −1. */
  captureStep: number;
  seed: number;
}

export interface AdamSettings {
  rate: number;
  biasCorrection: boolean;
  warmup: number;
  weightDecay: number;
  decayMode: DecayMode;
}

/** A run at the lab's defaults for every setting except the optimizer and its rate. */
export const baseSpec = (optimizer: OptimizerId, rate: number): RunSpec => ({
  optimizer,
  rate,
  biasCorrection: true,
  warmup: 0,
  weightDecay: 0,
  decayMode: "decoupled",
  epochs: EPOCHS,
  captureStep: -1,
  seed: SEED,
});

export const adamSpec = (settings: AdamSettings): RunSpec => ({
  ...baseSpec("adamw", settings.rate),
  biasCorrection: settings.biasCorrection,
  warmup: settings.warmup,
  weightDecay: settings.weightDecay,
  decayMode: settings.decayMode,
});

/** Settings that do not apply to an optimizer are reset, so equal runs share one cache entry. */
function normalize(spec: RunSpec): RunSpec {
  if (spec.optimizer !== "adamw") {
    return { ...spec, biasCorrection: true, warmup: 0, weightDecay: 0, decayMode: "decoupled" };
  }
  // With no decay the two modes are the same arithmetic.
  return spec.weightDecay === 0 ? { ...spec, decayMode: "decoupled" } : spec;
}

/** The trainer's optimizer for a run: Adam with L2 is "adam", decoupled decay is "adamw". */
export const trainerKind = (spec: RunSpec): TinyOptimizerKind =>
  spec.optimizer === "adamw" ? (spec.decayMode === "l2" && spec.weightDecay > 0 ? "adam" : "adamw") : spec.optimizer;

export interface RaceRun {
  spec: RunSpec;
  steps: number;
  trainLoss: number;
  heldOut: number;
  /** Whole-split losses at each epoch boundary, epoch 0 to `epochs`. */
  trainCurve: number[];
  heldCurve: number[];
  /** AdamW only: each step's batch loss, measured before its update. */
  batchLoss: number[];
  /** AdamW only: the largest change any one weight made on each step. */
  maxUpdate: number[];
  firstStepMove: number;
  earlyPeakLoss: number;
  weightNorm: number;
  rowNorms: number[];
  /** Total length of every step's change to each row, and the batches that contained the row. */
  rowMovement: number[];
  rowVisits: number[];
  /** Floats the trainer kept between steps: 0, 900 or 1,800. */
  stateFloats: number;
  captured: { direction: Float32Array; update: Float32Array; rows: number; cols: number } | null;
  diverged: boolean;
}

const CACHE_LIMIT = 64;
const cache = new Map<string, RaceRun>();

export function trainRun(input: RunSpec): RaceRun {
  const spec = normalize(input);
  const key = JSON.stringify(spec);
  const hit = cache.get(key);
  if (hit) {
    cache.delete(key);
    cache.set(key, hit);
    return hit;
  }
  const traced = spec.optimizer === "adamw";
  const run = trainTinyExtended({
    text: TRAIN_TEXT,
    epochs: spec.epochs,
    batchSize: BATCH_SIZE,
    learningRate: spec.rate,
    schedule: spec.warmup > 0 ? "wsd" : "constant",
    warmupFraction: spec.warmup,
    decayFraction: 0,
    weightDecay: spec.weightDecay,
    seed: spec.seed,
    optimizer: { kind: trainerKind(spec), biasCorrection: spec.biasCorrection },
    checkpoints: spec.epochs + 1,
    historyPoints: 1,
    trace: traced,
    captureStep: spec.captureStep >= 0 ? spec.captureStep : undefined,
  });
  const weights = run.weights;
  const rowNorms = Array.from({ length: V }, (_, row) => {
    let sum = 0;
    for (let k = 0; k < V; k += 1) sum += weights[row * V + k] ** 2;
    return Math.sqrt(sum);
  });
  const batchLoss = run.trace.map((point) => point.loss);
  const result: RaceRun = {
    spec,
    steps: run.steps,
    trainLoss: run.finalLoss,
    heldOut: run.diverged ? Number.NaN : tinyCrossEntropy(weights, HELD_OUT_TEXT),
    trainCurve: run.checkpoints.map((point) => point.loss),
    heldCurve: run.checkpoints.map((point) =>
      Number.isFinite(point.loss) ? tinyCrossEntropy(point.weights, HELD_OUT_TEXT) : Number.NaN,
    ),
    batchLoss,
    maxUpdate: run.trace.map((point) => point.maxUpdate),
    firstStepMove: run.trace[0]?.maxUpdate ?? Number.NaN,
    earlyPeakLoss: batchLoss.length ? Math.max(...batchLoss.slice(0, EARLY_STEPS)) : Number.NaN,
    weightNorm: Math.sqrt(rowNorms.reduce((sum, norm) => sum + norm * norm, 0)),
    rowNorms,
    rowMovement: Array.from(run.rowMovement),
    rowVisits: Array.from(run.rowVisits),
    stateFloats: run.stateFloats,
    captured: run.captured
      ? { direction: run.captured.gradient, update: run.captured.update, rows: run.captured.rows, cols: run.captured.cols }
      : null,
    diverged: run.diverged,
  };
  cache.set(key, result);
  while (cache.size > CACHE_LIMIT) cache.delete(cache.keys().next().value as string);
  return result;
}

/* ------------------------------------------------------------------------------------------------ */
/* Per-row step sizes                                                                                */
/* ------------------------------------------------------------------------------------------------ */

/** The five busiest input characters and the three rarest that appear in the training text. */
export const ROW_CHARACTERS = [" ", "e", "t", "a", "o", "k", "q", "z"] as const;
export const RARE_CHARACTERS = ["k", "q", "z"] as const;

export interface RowStep {
  character: string;
  label: string;
  visits: number;
  /** Distance the row travelled over the run, per batch that contained it. */
  perVisit: number;
}

export const rowLabel = (character: string) => (character === " " ? "space" : character);

export function rowSteps(run: RaceRun): RowStep[] {
  return ROW_CHARACTERS.map((character) => {
    const row = TINY_VOCAB.indexOf(character);
    const visits = run.rowVisits[row] ?? 0;
    return {
      character,
      label: rowLabel(character),
      visits,
      perVisit: visits > 0 ? run.rowMovement[row] / visits : 0,
    };
  });
}

/* ------------------------------------------------------------------------------------------------ */
/* Weight decay                                                                                      */
/* ------------------------------------------------------------------------------------------------ */

export const KEPT_CHARACTERS = [" ", "k", "q"] as const;

/** Each row's size after training with decay, as a share of the same run with no decay. */
export function rowsKept(run: RaceRun, reference: RaceRun) {
  return KEPT_CHARACTERS.map((character) => {
    const row = TINY_VOCAB.indexOf(character);
    const without = reference.rowNorms[row];
    return {
      character,
      label: rowLabel(character),
      visits: run.rowVisits[row] ?? 0,
      kept: without > 0 ? run.rowNorms[row] / without : 1,
    };
  });
}

export const IDLE_STEPS = 10;

/**
 * A closed-form illustration, not a measurement: one weight starting at 1.0 that gets no gradient of
 * its own, stepped with the same Adam arithmetic as the trainer for `IDLE_STEPS` steps. Returns its
 * average drop per step. Decoupled decay takes rate × decay of it each step. L2 decay reaches it as a
 * gradient of decay × weight, which Adam then divides by that same gradient's running size, so the
 * step is about the full rate however small the decay is.
 */
export function idleShrink(settings: Pick<AdamSettings, "rate" | "weightDecay" | "decayMode" | "biasCorrection">): number {
  const { rate, weightDecay, decayMode, biasCorrection } = settings;
  if (weightDecay === 0) return 0;
  const beta1 = 0.9;
  const beta2 = 0.999;
  let weight = 1;
  let first = 0;
  let second = 0;
  for (let step = 1; step <= IDLE_STEPS; step += 1) {
    let gradient = 0;
    if (decayMode === "l2") gradient += weightDecay * weight;
    else weight *= 1 - rate * weightDecay;
    first = beta1 * first + (1 - beta1) * gradient;
    second = beta2 * second + (1 - beta2) * gradient * gradient;
    const firstHat = biasCorrection ? first / (1 - beta1 ** step) : first;
    const secondHat = biasCorrection ? second / (1 - beta2 ** step) : second;
    weight -= rate * (firstHat / (Math.sqrt(secondHat) + 1e-8));
  }
  return (1 - weight) / IDLE_STEPS;
}

/* ------------------------------------------------------------------------------------------------ */
/* Muon's orthogonalized update                                                                      */
/* ------------------------------------------------------------------------------------------------ */

/**
 * The Muon run up to the captured step. With a constant rate and no warmup, a run that stops early
 * takes exactly the same steps as the full one, so only the epochs up to the capture are trained.
 */
export function captureMuon(rate: number, capture: CaptureId) {
  const step = CAPTURES[capture].step;
  const epochs = Math.min(EPOCHS, Math.floor(step / STEPS_PER_EPOCH) + 1);
  return trainRun({ ...baseSpec("muon", rate), epochs, captureStep: step }).captured;
}

export interface MuonSpectra {
  /** The momentum direction divided by its Frobenius norm: zero Newton–Schulz iterations. */
  before: SpectrumSummary;
  after: SpectrumSummary;
  iterations: number;
}

export function muonSpectra(rate: number, capture: CaptureId, iterations: number): MuonSpectra | null {
  const captured = captureMuon(rate, capture);
  if (!captured) return null;
  const { direction, rows, cols } = captured;
  return {
    before: summarizeSpectrum(singularValues(newtonSchulz(direction, rows, cols, 0), rows, cols)),
    after: summarizeSpectrum(singularValues(newtonSchulz(direction, rows, cols, iterations), rows, cols)),
    iterations,
  };
}

/* ------------------------------------------------------------------------------------------------ */
/* Tune each: a learning-rate sweep per optimizer, run one training at a time                        */
/* ------------------------------------------------------------------------------------------------ */

export interface SweepPoint {
  optimizer: OptimizerId;
  rate: number;
  factor: number;
  trainLoss: number;
  heldOut: number;
}

export interface SweepRow {
  optimizer: OptimizerId;
  points: SweepPoint[];
  /** Lowest held-out loss on the grid. */
  best: SweepPoint;
  /** The grid point at the default rate. */
  atDefault: SweepPoint;
}

const sweepStore = new Map<string, SweepPoint>();
const sweepKey = (optimizer: OptimizerId, rate: number) => `${optimizer}:${rate}`;
let sweepTimer: ReturnType<typeof setTimeout> | null = null;
const sweepListeners = new Set<() => void>();

export const sweepTasks = () =>
  OPTIMIZERS.flatMap((choice) =>
    SWEEP_FACTORS.map((factor) => ({ optimizer: choice.id, rate: significant(choice.defaultRate * factor), factor })),
  );

export function sweepProgress() {
  const tasks = sweepTasks();
  const done = tasks.filter((task) => sweepStore.has(sweepKey(task.optimizer, task.rate))).length;
  return { done, total: tasks.length, running: sweepTimer !== null, complete: done === tasks.length };
}

/** Trains the next run the sweep is missing. Returns whether any are left. */
export function runSweepStep(): boolean {
  const next = sweepTasks().find((task) => !sweepStore.has(sweepKey(task.optimizer, task.rate)));
  if (!next) return false;
  const run = trainRun(baseSpec(next.optimizer, next.rate));
  sweepStore.set(sweepKey(next.optimizer, next.rate), { ...next, trainLoss: run.trainLoss, heldOut: run.heldOut });
  return sweepTasks().some((task) => !sweepStore.has(sweepKey(task.optimizer, task.rate)));
}

/** The finished table, or null until every run is in. */
export function sweepTable(): SweepRow[] | null {
  if (!sweepProgress().complete) return null;
  return OPTIMIZERS.map((choice) => {
    const points = SWEEP_FACTORS.map(
      (factor) => sweepStore.get(sweepKey(choice.id, significant(choice.defaultRate * factor)))!,
    );
    const finite = points.filter((point) => Number.isFinite(point.heldOut));
    const best = finite.reduce((winner, point) => (point.heldOut < winner.heldOut ? point : winner), finite[0] ?? points[0]);
    return { optimizer: choice.id, points, best, atDefault: points[SWEEP_FACTORS.indexOf(1)] };
  });
}

/** Runs the whole sweep at once. Tests use it; the lab spreads the runs out with `startSweep`. */
export function runSweepNow(): SweepRow[] {
  while (runSweepStep());
  return sweepTable()!;
}

export function subscribeSweep(listener: () => void): () => void {
  sweepListeners.add(listener);
  return () => {
    sweepListeners.delete(listener);
  };
}

/** One training per macrotask, so the page repaints between runs; a Muon run is the slowest of the four. */
export function startSweep() {
  if (sweepTimer !== null || sweepProgress().complete) return;
  const tick = () => {
    const more = runSweepStep();
    sweepTimer = more ? setTimeout(tick, 0) : null;
    sweepListeners.forEach((listener) => listener());
  };
  sweepTimer = setTimeout(tick, 0);
  sweepListeners.forEach((listener) => listener());
}

/* ------------------------------------------------------------------------------------------------ */
/* Optimizer state                                                                                   */
/* ------------------------------------------------------------------------------------------------ */

export const LAB_PARAMETERS = V * V;
export const LARGE_PARAMETERS =
  teachingMemory.MODEL_SIZES.find((entry) => entry.id === "7.5b")?.parameters ?? 7_500_000_000;

export interface StateRow {
  optimizer: OptimizerId;
  buffers: number;
  /** Bytes of the trainer's own buffers: its measured float count × 4. */
  labBytes: number;
  /** fp32 buffers only, at 7.5 billion weights (teachingMemory's fp32 row). */
  largeBytes: number;
  /** Weights, gradients, master copy and buffers in mixed precision, at 7.5 billion (ZeRO's accounting). */
  largeMixedTotal: number;
}

export function stateRows(stateFloats: Record<OptimizerId, number>): StateRow[] {
  return OPTIMIZERS.map((choice) => ({
    optimizer: choice.id,
    buffers: choice.buffers,
    labBytes: stateFloats[choice.id] * teachingMemory.STATE_BYTES,
    largeBytes: teachingMemory.memoryBreakdown(LARGE_PARAMETERS, "fp32", choice.memoryRow).optimizerState,
    largeMixedTotal: teachingMemory.memoryBreakdown(LARGE_PARAMETERS, "mixed", choice.memoryRow).total,
  }));
}
