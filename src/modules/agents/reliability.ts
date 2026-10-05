/**
 * How often does an agent complete every step of a run?
 *
 * Closed form, no model involved. Assumptions, all stated on the card: every step succeeds
 * independently with the same probability p; a wrong step is invisible to the agent; an approval
 * gate after step k lets a reviewer inspect the work so far, and the reviewer catches a wrong prefix
 * with probability `catchRate`. A caught prefix is redone once, and the redo is not checked again.
 *
 *   no gate:   P = p^n
 *   gate at k: s = p^k;  P = (s + (1 − s)·catchRate·s) · p^(n − k)
 *
 * With catchRate = 1 the first factor is 1 − (1 − s)², "run the prefix twice, keep the better".
 * A gate after step 1 is a retry of step 1.
 */

export const STEP_SUCCESS_RANGE = { min: 50, max: 100 } as const;
export const RUN_STEPS_RANGE = { min: 1, max: 30 } as const;
/** 0 means "no gate". A gate placed after the run's last step sits at the end of the run. */
export const GATE_RANGE = { min: 0, max: 30 } as const;
export const CATCH_RANGE = { min: 0, max: 100 } as const;

export const DEFAULT_STEP_SUCCESS = 95;
export const DEFAULT_RUN_STEPS = 20;
export const DEFAULT_GATE = 0;
export const DEFAULT_CATCH = 100;

/** Step counts shown in the table, besides the current run length. */
export const TABLE_STEPS = [1, 5, 10, 20, 30] as const;

export interface Gate {
  /** 1-based step after which the reviewer looks; clamped to the run length. */
  after: number;
  /** Probability, 0 to 1, that the reviewer notices a wrong prefix. */
  catchRate: number;
}

const unit = (value: number) => (Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : 0);
const whole = (value: number, low: number, high: number) =>
  Number.isFinite(value) ? Math.min(high, Math.max(low, Math.round(value))) : low;

/** The step the gate sits after, or 0 when there is no gate. */
export function gatePosition(steps: number, gate: Gate | null): number {
  if (!gate || gate.after < 1) return 0;
  return Math.min(whole(steps, 1, RUN_STEPS_RANGE.max), whole(gate.after, 1, GATE_RANGE.max));
}

/** Probability that a run of `steps` steps finishes with every step right. */
export function completionProbability(stepSuccess: number, steps: number, gate: Gate | null = null): number {
  const p = unit(stepSuccess);
  const n = whole(steps, 1, RUN_STEPS_RANGE.max);
  const k = gatePosition(n, gate);
  if (k === 0 || !gate) return p ** n;
  const prefix = p ** k;
  const afterGate = prefix + (1 - prefix) * unit(gate.catchRate) * prefix;
  return afterGate * p ** (n - k);
}

/** Steps executed on average: the run itself, plus the redo of a prefix the reviewer sent back. */
export function expectedStepsRun(stepSuccess: number, steps: number, gate: Gate | null = null): number {
  const p = unit(stepSuccess);
  const n = whole(steps, 1, RUN_STEPS_RANGE.max);
  const k = gatePosition(n, gate);
  if (k === 0 || !gate) return n;
  return n + k * (1 - p ** k) * unit(gate.catchRate);
}

/**
 * The longest run that still completes at least half the time without a gate, or null when a
 * step never fails (so no run length ever drops below half).
 */
export function halfLifeSteps(stepSuccess: number): number | null {
  const p = unit(stepSuccess);
  if (p >= 1) return null;
  if (p <= 0) return 0;
  let steps = 0;
  while (p ** (steps + 1) >= 0.5 && steps < 100000) steps += 1;
  return steps;
}

export interface CurvePoint {
  steps: number;
  plain: number;
  gated: number | null;
}

/** One point per run length from 1 to `maxSteps`, with and without the gate. */
export function completionCurve(stepSuccess: number, gate: Gate | null, maxSteps: number = RUN_STEPS_RANGE.max): CurvePoint[] {
  return Array.from({ length: whole(maxSteps, 1, RUN_STEPS_RANGE.max) }, (_, index) => {
    const steps = index + 1;
    return {
      steps,
      plain: completionProbability(stepSuccess, steps),
      gated: gate && gate.after >= 1 ? completionProbability(stepSuccess, steps, gate) : null,
    };
  });
}

/**
 * The same process played out run by run with a seeded random source, to check the closed form.
 * Returns the fraction of simulated runs that finished with every step right.
 */
export function simulateCompletion(
  stepSuccess: number,
  steps: number,
  gate: Gate | null,
  runs: number,
  random: () => number,
): number {
  const p = unit(stepSuccess);
  const n = whole(steps, 1, RUN_STEPS_RANGE.max);
  const k = gatePosition(n, gate);
  const catchRate = gate ? unit(gate.catchRate) : 0;
  const allRight = (count: number) => {
    let right = true;
    for (let step = 0; step < count; step += 1) if (random() >= p) right = false;
    return right;
  };
  let completed = 0;
  for (let run = 0; run < runs; run += 1) {
    let prefixRight = allRight(k);
    if (k > 0 && !prefixRight && random() < catchRate) prefixRight = allRight(k);
    const suffixRight = allRight(n - k);
    if (prefixRight && suffixRight) completed += 1;
  }
  return completed / runs;
}

export const percent = (value: number, digits = 1) => `${(unit(value) * 100).toFixed(digits)}%`;
