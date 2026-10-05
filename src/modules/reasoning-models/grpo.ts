import { tinyRandom } from "@app/module-sdk";

/**
 * A group-relative policy-gradient (GRPO-style) toy.
 *
 * What is trained: ONE number, mu, the typical trace length. The policy is a
 * softmax over the eight lengths L = 1..MAX_LENGTH whose logits are
 * -(L - mu)^2 / (2 sigma^2): a bump of fixed width centred at mu. It is a
 * one-parameter family over trace LENGTH, not a language model.
 *
 * What the verifier does: a hand-written rule, not a learned reward model. Each
 * step of a trace lands a useful move with probability STEP_SUCCESS and slips
 * otherwise; the final answer is correct exactly when at least k of the L steps
 * landed. So a trace shorter than k can never pass, a trace of exactly k passes
 * only when no step slips, and every extra step is spare capacity that makes a
 * pass more likely, with diminishing returns.
 *
 * What one update does, once per iteration:
 *   1. sample a group of G lengths from the current policy and run the checker
 *      on each,
 *   2. reward_i = pass_i - (cost ? LENGTH_COST * L_i : 0),
 *   3. advantage_i = (reward_i - group mean) / (group std + epsilon), so a group
 *      whose rewards are all equal produces advantage 0 and no learning signal,
 *   4. mu += rate * (1/G) * sum_i advantage_i * dlog pi(L_i)/dmu, where
 *      dlog pi(L)/dmu = (L - E[L]) / sigma^2.
 *
 * Left out on purpose: a token-level policy, the KL penalty to a reference
 * model, PPO-style clipped ratios (there is one gradient step per sampled group,
 * so the ratio is always 1), and many prompts per batch.
 *
 * Why one parameter and not eight free logits: with a free logit per length the
 * softmax collapses onto whichever length first earns reward and stays there
 * (rare lengths are almost never sampled, so their logits barely move). That
 * hides both the drift and the trimming this card exists to show, and it is not
 * what a language model's shared weights do either.
 */

export const MAX_LENGTH = 8;
/** Chance that one trace step lands a useful move. Fixed, not a control. */
export const STEP_SUCCESS = 0.8;
/** Reward subtracted per step when the length cost is on. */
export const LENGTH_COST = 0.05;
/** Policy-gradient iterations in one run. Fixed so a run is a few milliseconds. */
export const ITERATIONS = 100;
export const STD_EPSILON = 1e-6;
/** Width of the length bump. */
export const SIGMA = 1;
/** The starting policy: it likes short answers, centred between 1 and 2 steps. */
export const INITIAL_CENTRE = 1.5;
export const CENTRE_LIMITS = { min: 0, max: 10 } as const;

export const K_RANGE = { min: 1, max: 4 } as const;
export const GROUP_RANGE = { min: 2, max: 32 } as const;
export const RATE_RANGE = { min: 0.1, max: 1, step: 0.1 } as const;
export const SEED_RANGE = { min: 1, max: 20 } as const;

export interface GrpoOptions {
  /** Dependent steps the task needs. */
  k: number;
  /** Traces sampled per iteration. */
  group: number;
  rate: number;
  /** Whether each step costs LENGTH_COST reward. */
  cost: boolean;
  seed: number;
  iterations?: number;
}

export interface GrpoPoint {
  iteration: number;
  /** The one trained parameter after this many updates. */
  centre: number;
  /** Exact policy distribution over lengths 1..MAX_LENGTH after this many updates. */
  probabilities: number[];
  /** Exact expectation under the policy: chance the checker passes a sampled trace. */
  passRate: number;
  /** Exact expectation under the policy: mean trace length. */
  meanLength: number;
  /** Exact expectation of pass - cost, the quantity the update climbs. */
  meanReward: number;
  /** Exact expectation of steps beyond k, max(0, L - k). */
  extraSteps: number;
  /** Sampled groups so far whose rewards were all equal, so produced no update. */
  flatSoFar: number;
}

export interface GrpoRun {
  history: GrpoPoint[];
  /** Iterations whose sampled group had identical rewards, so produced no update. */
  flatGroups: number;
}

/** How the lab prints a probability, so tests can pin exactly what the tiles show. */
export const percent = (value: number) => `${Math.round(value * 100)}%`;

export const LENGTHS: ReadonlyArray<number> = Array.from({ length: MAX_LENGTH }, (_, index) => index + 1);

/** Logits over lengths 1..MAX_LENGTH for a bump of width SIGMA centred at `centre`. */
export function lengthLogits(centre: number): number[] {
  return LENGTHS.map((length) => -((length - centre) ** 2) / (2 * SIGMA ** 2));
}

export function softmax(logits: ReadonlyArray<number>): number[] {
  const peak = Math.max(...logits);
  const exps = logits.map((value) => Math.exp(value - peak));
  const total = exps.reduce((sum, value) => sum + value, 0);
  return exps.map((value) => value / total);
}

export function lengthPolicy(centre: number): number[] {
  return softmax(lengthLogits(centre));
}

function choose(n: number, r: number): number {
  let value = 1;
  for (let i = 1; i <= r; i += 1) value = (value * (n - r + i)) / i;
  return value;
}

/** P(Binomial(length, q) >= k): the checker's pass probability for a trace of this length. */
export function passProbability(length: number, k: number, q: number = STEP_SUCCESS): number {
  if (length < k) return 0;
  let total = 0;
  for (let landed = k; landed <= length; landed += 1) {
    total += choose(length, landed) * q ** landed * (1 - q) ** (length - landed);
  }
  return Math.min(1, total);
}

/** Expected reward of a trace of this length: pass probability minus the length cost, if on. */
export function expectedReward(length: number, k: number, cost: boolean): number {
  return passProbability(length, k) - (cost ? LENGTH_COST * length : 0);
}

/** Exact expectations of the policy that has its bump centred at `centre`. */
export function summarize(iteration: number, centre: number, k: number, cost: boolean): GrpoPoint {
  const probabilities = lengthPolicy(centre);
  let passRate = 0;
  let meanLength = 0;
  let meanReward = 0;
  let extraSteps = 0;
  probabilities.forEach((probability, index) => {
    const length = index + 1;
    passRate += probability * passProbability(length, k);
    meanLength += probability * length;
    meanReward += probability * expectedReward(length, k, cost);
    extraSteps += probability * Math.max(0, length - k);
  });
  return { iteration, centre, probabilities, passRate, meanLength, meanReward, extraSteps, flatSoFar: 0 };
}

/** Group-relative advantages: (reward - mean) / (std + epsilon), population std. */
export function groupAdvantages(rewards: ReadonlyArray<number>): number[] {
  const count = rewards.length;
  const mean = rewards.reduce((sum, value) => sum + value, 0) / count;
  const variance = rewards.reduce((sum, value) => sum + (value - mean) ** 2, 0) / count;
  const std = Math.sqrt(variance);
  return rewards.map((value) => (value - mean) / (std + STD_EPSILON));
}

/**
 * The centre that maximises expected reward over the whole one-parameter family,
 * found by scanning. This is where training should settle if it had unlimited
 * samples; the sampled run only approaches it.
 */
export function bestCentre(k: number, cost: boolean): { centre: number; point: GrpoPoint } {
  let best = { centre: CENTRE_LIMITS.min as number, point: summarize(0, CENTRE_LIMITS.min, k, cost) };
  for (let centre = CENTRE_LIMITS.min; centre <= CENTRE_LIMITS.max + 1e-9; centre += 0.05) {
    const point = summarize(0, centre, k, cost);
    if (point.meanReward > best.point.meanReward + 1e-12) best = { centre, point };
  }
  return best;
}

/** Deterministic in `seed`. Costs ITERATIONS x group x MAX_LENGTH operations: milliseconds. */
export function trainGroupRelative(options: GrpoOptions): GrpoRun {
  const { k, group, rate, cost, seed } = options;
  const iterations = options.iterations ?? ITERATIONS;
  const random = tinyRandom(seed);
  let centre = INITIAL_CENTRE;
  const history: GrpoPoint[] = [];
  let flatGroups = 0;

  for (let iteration = 0; iteration <= iterations; iteration += 1) {
    const point = { ...summarize(iteration, centre, k, cost), flatSoFar: flatGroups };
    history.push(point);
    if (iteration === iterations) break;

    const lengths: number[] = [];
    const rewards: number[] = [];
    for (let sample = 0; sample < group; sample += 1) {
      let draw = random();
      let length = MAX_LENGTH;
      for (let index = 0; index < MAX_LENGTH; index += 1) {
        draw -= point.probabilities[index];
        if (draw <= 0) {
          length = index + 1;
          break;
        }
      }
      const passed = random() < passProbability(length, k) ? 1 : 0;
      lengths.push(length);
      rewards.push(passed - (cost ? LENGTH_COST * length : 0));
    }

    const advantages = groupAdvantages(rewards);
    if (advantages.every((value) => Math.abs(value) < 1e-9)) flatGroups += 1;
    let gradient = 0;
    for (let sample = 0; sample < group; sample += 1) {
      gradient += advantages[sample] * ((lengths[sample] - point.meanLength) / SIGMA ** 2);
    }
    centre = Math.min(CENTRE_LIMITS.max, Math.max(CENTRE_LIMITS.min, centre + (rate / group) * gradient));
  }

  return { history, flatGroups };
}
