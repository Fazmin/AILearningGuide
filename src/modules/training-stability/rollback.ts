/**
 * Roll back and skip, the response PaLM describes to a loss spike: reload a checkpoint saved before
 * the spike, then train on, never applying the batches that were in flight when it happened.
 *
 * In this lab the replay is a second run with `skipSteps` over the garbage steps. A skipped batch
 * is read but never applied, so the data order after it is unchanged, and every step before the
 * first skipped one is bit-identical to the original run: the steps between the checkpoint and the
 * garbage replay exactly. Step numbers here are 1-based, as the dashboard prints them; the trace
 * and `skipSteps` count from 0.
 */
import type { TinyCheckpoint, TinyTracePoint } from "@app/module-sdk";

/** How often the run saves a checkpoint it could restart from. */
export const CHECKPOINT_EVERY = 50;

/** The last checkpoint saved before `step` ran: the weights after that many updates (0 is the start). */
export function lastCheckpointBefore(step: number, every = CHECKPOINT_EVERY): number {
  const safeEvery = Math.max(1, Math.round(every));
  return Math.max(0, Math.floor((Math.max(1, Math.round(step)) - 1) / safeEvery) * safeEvery);
}

export interface RollbackPlan {
  /** The checkpoint the run restarts from. */
  checkpointStep: number;
  /** First and last replayed step that is identical to the original. Empty when replayFrom > replayTo. */
  replayFrom: number;
  replayTo: number;
  /** First and last skipped step, inclusive. */
  skipFrom: number;
  skipTo: number;
  /** The same range as `trainTinyExtended` takes it: 0-based and half-open. */
  skipSteps: Array<[number, number]>;
}

/**
 * Restart from the last checkpoint at least one full checkpoint interval before the first bad step,
 * as PaLM restarted well before its spikes, and skip exactly the bad steps.
 */
export function rollbackPlan(firstBadStep: number, count: number, every = CHECKPOINT_EVERY): RollbackPlan {
  const first = Math.max(1, Math.round(firstBadStep));
  const length = Math.max(1, Math.round(count));
  const checkpointStep = lastCheckpointBefore(first, every);
  return {
    checkpointStep,
    replayFrom: checkpointStep + 1,
    replayTo: first - 1,
    skipFrom: first,
    skipTo: first + length - 1,
    skipSteps: [[first - 1, first - 1 + length]],
  };
}

/** The 1-based step where two traces first differ, or null when they agree over their common length. */
export function firstDifference(left: ReadonlyArray<TinyTracePoint>, right: ReadonlyArray<TinyTracePoint>): number | null {
  const length = Math.min(left.length, right.length);
  for (let index = 0; index < length; index += 1) {
    const a = left[index];
    const b = right[index];
    if (
      !Object.is(a.loss, b.loss) ||
      !Object.is(a.gradientNorm, b.gradientNorm) ||
      !Object.is(a.updateRatio, b.updateRatio) ||
      !Object.is(a.maxLogit, b.maxLogit) ||
      a.skipped !== b.skipped
    ) {
      return index + 1;
    }
  }
  return null;
}

export interface SpikeReport {
  /** Whole-corpus loss at the last checkpoint before the first bad step. */
  before: number;
  /** Highest whole-corpus loss from the first bad step on, and where it happened. */
  peak: number;
  peakStep: number;
  /** peak − before, in nats. */
  rise: number;
}

/**
 * Measures a loss spike on the whole-corpus checkpoints: the loss just before the first bad step,
 * against the highest loss from that step to the end. Null when the run left no finite checkpoint
 * on either side.
 */
export function spikeReport(checkpoints: ReadonlyArray<TinyCheckpoint>, firstBadStep: number): SpikeReport | null {
  const finite = checkpoints.filter((point) => Number.isFinite(point.loss));
  const before = [...finite].reverse().find((point) => point.step < firstBadStep);
  const after = finite.filter((point) => point.step >= firstBadStep);
  if (!before || after.length === 0) return null;
  const peak = after.reduce((best, point) => (point.loss > best.loss ? point : best));
  return { before: before.loss, peak: peak.loss, peakStep: peak.step, rise: peak.loss - before.loss };
}
