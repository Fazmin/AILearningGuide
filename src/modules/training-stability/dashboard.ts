/**
 * Turns a traced run into what the dashboard draws: one point per optimizer step with every value
 * that is not a finite number replaced by null, so NaN and Infinity never reach the page; log-scale
 * domains and ticks; normal-range bands from a reference run; and the readouts the run report
 * prints. Step numbers are 1-based, as the dashboard prints them.
 */
import type { TinyCheckpoint, TinyExtendedRun } from "@app/module-sdk";

export interface StepPoint {
  /** 1-based optimizer step. */
  step: number;
  /** Batch cross-entropy before the update. Null on a skipped or halted step. */
  loss: number | null;
  /** Global gradient norm before clipping. Null on an overflow or the halted step. */
  norm: number | null;
  maxLogit: number | null;
  logPartition: number | null;
  /** ‖Δw‖ ÷ ‖w‖. Null when no update was applied. */
  ratio: number | null;
  clipped: boolean;
  skipped: boolean;
  overflow: boolean;
  faulty: boolean;
  lossScale: number;
}

const finiteOrNull = (value: number) => (Number.isFinite(value) ? value : null);

/**
 * A skipped step's batch was read but never trained on, and the halted step's numbers are not
 * numbers, so both are drawn as gaps; the dashboard marks them instead.
 */
export function stepPoints(run: TinyExtendedRun): StepPoint[] {
  return run.trace.map((point) => {
    const step = point.step + 1;
    const halted = run.haltedAt === step;
    const applied = !point.skipped && !halted;
    return {
      step,
      loss: applied ? finiteOrNull(point.loss) : null,
      norm: applied && !point.overflow ? finiteOrNull(point.gradientNorm) : null,
      maxLogit: applied ? finiteOrNull(point.maxLogit) : null,
      logPartition: applied ? finiteOrNull(point.logPartition) : null,
      ratio: applied ? finiteOrNull(point.updateRatio) : null,
      clipped: point.clipped,
      skipped: point.skipped,
      overflow: point.overflow,
      faulty: point.faulty,
      lossScale: point.lossScale,
    };
  });
}

/** Whole-corpus loss at each checkpoint that is still a number. */
export function corpusPoints(checkpoints: ReadonlyArray<TinyCheckpoint>): Array<{ step: number; loss: number }> {
  return checkpoints.filter((point) => Number.isFinite(point.loss)).map((point) => ({ step: point.step, loss: point.loss }));
}

export type Signal = "loss" | "norm" | "maxLogit" | "ratio";

export const values = (points: ReadonlyArray<StepPoint>, signal: Signal) =>
  points.map((point) => point[signal]).filter((value): value is number => value !== null);

/** The q-quantile of finite values (nearest rank, as the lesson's percentiles are), or null for none. */
export function quantile(list: ReadonlyArray<number>, q: number): number | null {
  const finite = list.filter(Number.isFinite);
  if (finite.length === 0) return null;
  const sorted = [...finite].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.floor(Math.min(1, Math.max(0, q)) * (sorted.length - 1))));
  return sorted[index];
}

/** The normal range of a signal: its 5th to 95th percentile over a clean run. */
export function normalBand(points: ReadonlyArray<StepPoint>, signal: Signal) {
  const list = values(points, signal);
  const low = quantile(list, 0.05);
  const high = quantile(list, 0.95);
  return low === null || high === null ? null : { low, high };
}

/* -------------------------------------------------------------------------- */
/* Log scales                                                                  */
/* -------------------------------------------------------------------------- */

export interface LogDomain {
  /** Powers of ten at the bottom and top of the axis. */
  low: number;
  high: number;
}

/**
 * Whole decades that cover the positive values, at least `minDecades` tall and never wider than
 * 10^floor to 10^ceiling. Values below the floor are pinned to the bottom edge.
 */
export function logDomain(
  list: ReadonlyArray<number>,
  { floor = -4, ceiling = 40, minDecades = 2 }: { floor?: number; ceiling?: number; minDecades?: number } = {},
): LogDomain {
  const positive = list.filter((value) => Number.isFinite(value) && value > 0);
  if (positive.length === 0) return { low: floor, high: floor + minDecades };
  const lowest = Math.floor(Math.log10(Math.min(...positive)));
  const highest = Math.ceil(Math.log10(Math.max(...positive)));
  let low = Math.min(Math.max(floor, lowest), ceiling - minDecades);
  let high = Math.max(Math.min(ceiling, highest), low + 1);
  while (high - low < minDecades) {
    if (high < ceiling) high += 1;
    else low -= 1;
  }
  // `|| 0` turns a negative zero from Math.ceil into a plain zero.
  return { low: low || 0, high: high || 0 };
}

/** Position from 0 (bottom) to 1 (top) on a log axis, pinned to the edges; null values stay null. */
export function logPosition(value: number | null, domain: LogDomain): number | null {
  if (value === null || !Number.isFinite(value)) return null;
  if (value <= 0) return 0;
  const exponent = Math.log10(value);
  return Math.min(1, Math.max(0, (exponent - domain.low) / Math.max(1e-9, domain.high - domain.low)));
}

/**
 * Decade exponents to label: every one when there are few, otherwise evenly thinned to about
 * `maxTicks`. The bottom and top of the axis are always labelled.
 */
export function decadeTicks(domain: LogDomain, maxTicks = 5): number[] {
  const span = domain.high - domain.low;
  const stride = Math.max(1, Math.ceil(span / Math.max(1, maxTicks - 1)));
  const ticks: number[] = [];
  for (let exponent = domain.low; exponent < domain.high; exponent += stride) ticks.push(exponent);
  // Drop an inner tick that would crowd the top label.
  if (ticks.length > 1 && domain.high - ticks[ticks.length - 1] < stride / 2) ticks.pop();
  ticks.push(domain.high);
  return ticks;
}

/* -------------------------------------------------------------------------- */
/* Formatting                                                                  */
/* -------------------------------------------------------------------------- */

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

export const superscript = (value: number) =>
  String(Math.round(value))
    .split("")
    .map((character) => SUPERSCRIPT[character] ?? character)
    .join("");

/** A power of ten as an axis label: 10⁻², 1, 10, 10⁴. */
export const powerOfTen = (exponent: number) => (exponent === 0 ? "1" : exponent === 1 ? "10" : `10${superscript(exponent)}`);

/** A power of two as the loss scale control prints it: 2²⁰. */
export const powerOfTwo = (exponent: number) => `2${superscript(exponent)}`;

/**
 * A readable number: plain decimals between 0.001 and 100,000, scientific with a superscript
 * exponent outside that, and a dash for anything that is not a finite number.
 */
export function formatNumber(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  const size = Math.abs(value);
  if (size === 0) return "0";
  if (size >= 1e5 || size < 1e-3) {
    const [mantissa, exponent] = value.toExponential(1).split("e");
    return `${mantissa} × 10${superscript(Number(exponent))}`;
  }
  if (size >= 1000) return Math.round(value).toLocaleString("en-US");
  if (size >= 100) return value.toFixed(0);
  if (size >= 10) return value.toFixed(Math.min(digits, 1));
  if (size >= 0.1) return value.toFixed(digits);
  return value.toPrecision(2);
}

/** A loss in nats, always to three decimals, or a dash when it is not a number. */
export const formatLoss = (value: number | null | undefined) =>
  value === null || value === undefined || !Number.isFinite(value) ? "—" : value.toFixed(3);

/** z-loss coefficients and Adam rates as the controls print them. */
export const formatCoefficient = (value: number) => (value === 0 ? "off" : `${value}`);

/* -------------------------------------------------------------------------- */
/* Run summary                                                                 */
/* -------------------------------------------------------------------------- */

export interface RunSummary {
  steps: number;
  /** Whole-corpus loss of the final weights, or null when the run halted. */
  finalLoss: number | null;
  /** 1-based step on which a weight or the loss stopped being a number. */
  haltedAt: number | null;
  peakNorm: number | null;
  peakLogit: number | null;
  medianRatio: number | null;
  clipped: number;
  skipped: number;
  overflow: number;
  /** Steps whose batch was garbage. */
  faulty: number;
  firstOverflow: number | null;
  /** Steps skipped by dynamic loss scaling after an overflow, as opposed to a rollback. */
  overflowSkips: number[];
  /** The loss scale after the last step: dynamic scaling halves it on an overflow and doubles it after clean stretches. */
  endScale: number;
}

export function summarizeRun(run: TinyExtendedRun, points: ReadonlyArray<StepPoint> = stepPoints(run)): RunSummary {
  const norms = values(points, "norm");
  const logits = values(points, "maxLogit");
  const overflowSteps = points.filter((point) => point.overflow).map((point) => point.step);
  return {
    steps: run.steps,
    finalLoss: run.diverged || !Number.isFinite(run.finalLoss) ? null : run.finalLoss,
    haltedAt: run.haltedAt,
    peakNorm: norms.length ? Math.max(...norms) : null,
    peakLogit: logits.length ? Math.max(...logits) : null,
    medianRatio: quantile(values(points, "ratio"), 0.5),
    clipped: run.clippedSteps,
    skipped: run.skippedSteps,
    overflow: run.overflowSteps,
    faulty: points.filter((point) => point.faulty).length,
    firstOverflow: overflowSteps[0] ?? null,
    overflowSkips: points.filter((point) => point.overflow && point.skipped).map((point) => point.step),
    endScale: run.trace.at(-1)?.lossScale ?? 1,
  };
}

/** Contiguous runs of steps that match, as inclusive [first, last] pairs. */
export function stepRanges(points: ReadonlyArray<StepPoint>, match: (point: StepPoint) => boolean): Array<[number, number]> {
  const ranges: Array<[number, number]> = [];
  for (const point of points) {
    if (!match(point)) continue;
    const last = ranges.at(-1);
    if (last && last[1] === point.step - 1) last[1] = point.step;
    else ranges.push([point.step, point.step]);
  }
  return ranges;
}
