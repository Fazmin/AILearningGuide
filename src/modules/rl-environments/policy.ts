import { FEATURE_COUNT, TOOL_COUNT } from "./env";

/**
 * The agent's policy: a linear softmax over the six tools.
 *
 * Each tool has a bias plus one weight per feature, so the policy is 6 × (1 + 7) = 48 numbers.
 * The logit of tool a in a state with features x is w[a][0] + Σ_f w[a][f + 1] · x[f], and the
 * policy is the softmax of the six logits. It chooses which tool to call and nothing else: no
 * arguments, no text. It is a stand-in for a language model, not one.
 */

export const WEIGHTS_PER_TOOL = 1 + FEATURE_COUNT;
export const WEIGHT_COUNT = TOOL_COUNT * WEIGHTS_PER_TOOL;

export type Weights = Float64Array;

/** The input vector: a constant 1 for the bias, then the seven features. */
export const inputOf = (features: ReadonlyArray<number>) => [1, ...features];

/**
 * The reference policy, written by hand to stand in for a model after supervised fine-tuning. It
 * usually reads first and writes once it has read, but it is sloppy: it often closes without
 * writing, rarely looks a missing value up (so it writes a guess), seldom checks its write, and
 * closes after a failed check more often than it retries. It solves about one held-out ticket in
 * five. Its edit_check logit is a constant −2.5, so it rarely touches the test. Rows are tools,
 * columns are [bias, read, value missing, value known, updated, last check passed, last check
 * failed, test edited].
 */
export const REFERENCE_ROWS: ReadonlyArray<ReadonlyArray<number>> = [
  // read_ticket: the first thing it usually does, and almost never twice.
  [1.5, -4, 0, 0, 0, 0, 0, 0],
  // look_up: a weak pull when the value is missing, less once it is known or written.
  [-1, 0, 0.2, -1.5, -0.5, 0, 0, 0],
  // update_record: likelier after reading, a little more once the value is known, again after a failed check.
  [-0.5, 0.8, 0, 0.5, -1.5, 0, 1, 0],
  // run_check: mostly after a write, rarely twice in a row.
  [-1.5, 0, 0, 0, 0.8, -2, -1, 0],
  // edit_check: the loophole, a constant −2.5.
  [-2.5, 0, 0, 0, 0, 0, 0, 0],
  // close_ticket: after reading, more after a write, and almost always after a passing check.
  [-0.5, 0.7, 0, 0, 1.2, 2.5, 0, 0.5],
];

export const EDIT_BIAS = REFERENCE_ROWS[4][0];

export function referenceWeights(): Weights {
  return Float64Array.from(REFERENCE_ROWS.flat());
}

/** The six logits for one input vector. */
export function logits(weights: Weights, input: ReadonlyArray<number>): number[] {
  const result: number[] = [];
  for (let tool = 0; tool < TOOL_COUNT; tool += 1) {
    let total = 0;
    const offset = tool * WEIGHTS_PER_TOOL;
    for (let index = 0; index < WEIGHTS_PER_TOOL; index += 1) total += weights[offset + index] * input[index];
    result.push(total);
  }
  return result;
}

export function softmax(values: ReadonlyArray<number>): number[] {
  const peak = Math.max(...values);
  const exps = values.map((value) => Math.exp(value - peak));
  const total = exps.reduce((sum, value) => sum + value, 0);
  return exps.map((value) => value / total);
}

/** π(· | features): the chance of each tool in a state with these features. */
export const toolProbabilities = (weights: Weights, features: ReadonlyArray<number>) =>
  softmax(logits(weights, inputOf(features)));

/**
 * Adds scale × ∇_w log π(tool | features) into `into`. For a softmax over linear logits the
 * gradient for tool b's weights is (1[b = tool] − π_b) · x.
 */
export function addLogProbGradient(
  into: Float64Array,
  probabilities: ReadonlyArray<number>,
  input: ReadonlyArray<number>,
  tool: number,
  scale: number,
) {
  for (let b = 0; b < TOOL_COUNT; b += 1) {
    const factor = scale * ((b === tool ? 1 : 0) - probabilities[b]);
    if (factor === 0) continue;
    const offset = b * WEIGHTS_PER_TOOL;
    for (let index = 0; index < WEIGHTS_PER_TOOL; index += 1) into[offset + index] += factor * input[index];
  }
}

/** KL(π ‖ ρ) = Σ_b π_b log(π_b / ρ_b), in nats, for two distributions over the six tools. */
export function klDivergence(policy: ReadonlyArray<number>, reference: ReadonlyArray<number>) {
  let total = 0;
  for (let b = 0; b < policy.length; b += 1) {
    if (policy[b] > 0) total += policy[b] * Math.log(policy[b] / reference[b]);
  }
  return Math.max(0, total);
}

/**
 * Adds scale × ∇_w KL(π(·|x) ‖ ρ(·|x)) into `into`, exactly, for one state. With z the logits,
 * ∂KL/∂z_b = π_b (log π_b − log ρ_b − KL), and ∂z_b/∂w[b] = x.
 */
export function addKlGradient(
  into: Float64Array,
  probabilities: ReadonlyArray<number>,
  reference: ReadonlyArray<number>,
  input: ReadonlyArray<number>,
  scale: number,
) {
  const kl = klDivergence(probabilities, reference);
  for (let b = 0; b < TOOL_COUNT; b += 1) {
    const p = probabilities[b];
    if (p <= 0) continue;
    const factor = scale * p * (Math.log(p) - Math.log(reference[b]) - kl);
    const offset = b * WEIGHTS_PER_TOOL;
    for (let index = 0; index < WEIGHTS_PER_TOOL; index += 1) into[offset + index] += factor * input[index];
  }
}
