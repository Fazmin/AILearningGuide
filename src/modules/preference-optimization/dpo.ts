/**
 * Helpers around the SDK's DPO trainer: parsing authored comparisons, the
 * per-pair log-ratio margin, and the loss curve −log σ(β·m) the optimizer
 * descends. The trainer itself is `trainTinyDpo` in the SDK.
 */

import {
  encodeTinyText,
  TINY_VOCAB_SIZE,
  tinySequenceLogProb,
  type TinyPreferencePair,
} from "@app/module-sdk";

/**
 * Optimizer time grows with the number of pairs and the length of each completion, and the
 * labs retrain inside the render. These limits hold for the editor and for any shared link.
 */
export const MAX_PAIRS = 5;
export const MAX_FIELD_LENGTH = 50;
export const MAX_LINE_LENGTH = 3 * MAX_FIELD_LENGTH + 8;

/** Cuts raw editor text to the line count and line length the editor itself allows. */
export function sanitizePairsText(text: string): string {
  return text
    .split("\n")
    .slice(0, MAX_PAIRS)
    .map((line) => line.slice(0, MAX_LINE_LENGTH))
    .join("\n");
}

/**
 * Parses `prompt | preferred | rejected` lines. The prompt keeps one trailing
 * space so "the | fog …" is scored as "the fog …" rather than "thefog …". At most
 * `MAX_PAIRS` pairs are kept and each part is cut to `MAX_FIELD_LENGTH` characters.
 */
export function parsePairs(text: string): TinyPreferencePair[] {
  return text
    .split("\n")
    .map((line) => line.split("|").map((part) => part.trim().toLowerCase().slice(0, MAX_FIELD_LENGTH)))
    .filter((parts) => parts.length >= 3 && parts[1] && parts[2])
    .slice(0, MAX_PAIRS)
    .map((parts) => ({
      prompt: parts[0] ? `${parts[0]} ` : "",
      chosen: parts[1],
      rejected: parts[2],
    }));
}

/**
 * Three comparisons of the same kind as the defaults, a corpus sentence preferred over a loop, that
 * the optimizer never sees. They were fixed before any run and are scored by whatever policy
 * the training pairs produce.
 */
export const HELD_OUT_PAIRS: TinyPreferencePair[] = parsePairs(
  [
    "the | clouds break apart and the sun warms the streets. | clouds clouds clouds clouds clouds.",
    "stir the | onions until they turn soft and sweet. | onions onions onions onions.",
    "boil the | pasta in salted water until it is tender. | pasta and the pasta and the pasta.",
  ].join("\n"),
);

export type ReferenceChoice = "base" | "sft";

export const isReferenceChoice = (value: unknown): value is ReferenceChoice =>
  value === "base" || value === "sft";

/** Full-batch steps and step size of the SFT-style reference. */
export const SFT_STEPS = 60;
export const SFT_LEARNING_RATE = 0.35;

interface Transition {
  row: number;
  target: number;
}

/**
 * The completion's own transitions, with the prompt as context only. The prompt's last
 * character conditions the first completion character, and no prompt character is a target:
 * the same response-only loss that Instruction tuning & templates trains with.
 */
function responseTransitions(prompt: string, completion: string): Transition[] {
  const promptIds = encodeTinyText(prompt);
  let previous = promptIds.length ? promptIds[promptIds.length - 1] : 0;
  return encodeTinyText(completion).map((target) => {
    const transition = { row: previous, target };
    previous = target;
    return transition;
  });
}

function softmaxInto(weights: Float32Array, row: number, into: Float64Array) {
  const offset = row * TINY_VOCAB_SIZE;
  let maximum = -Infinity;
  for (let k = 0; k < TINY_VOCAB_SIZE; k += 1) maximum = Math.max(maximum, weights[offset + k]);
  let total = 0;
  for (let k = 0; k < TINY_VOCAB_SIZE; k += 1) {
    into[k] = Math.exp(weights[offset + k] - maximum);
    total += into[k];
  }
  for (let k = 0; k < TINY_VOCAB_SIZE; k += 1) into[k] /= total;
}

/** Mean cross-entropy per response character over the preferred completions, in nats. */
export function sftLoss(weights: Float32Array, pairs: ReadonlyArray<TinyPreferencePair>): number {
  const probabilities = new Float64Array(TINY_VOCAB_SIZE);
  let total = 0;
  let count = 0;
  for (const pair of pairs) {
    for (const { row, target } of responseTransitions(pair.prompt, pair.chosen)) {
      softmaxInto(weights, row, probabilities);
      total -= Math.log(Math.max(1e-12, probabilities[target]));
      count += 1;
    }
  }
  return count ? total / count : Math.log(TINY_VOCAB_SIZE);
}

/**
 * An SFT-style checkpoint: the base table fine-tuned with next-character loss on the preferred
 * completions only, the prompt read as context and never graded. Full-batch gradient descent on the
 * sum over each completion's characters, averaged over pairs. The base is read, never written.
 */
export function trainSftReference(
  base: Float32Array,
  pairs: ReadonlyArray<TinyPreferencePair>,
  steps = SFT_STEPS,
  learningRate = SFT_LEARNING_RATE,
): Float32Array {
  const weights = Float32Array.from(base);
  if (pairs.length === 0) return weights;
  const sequences = pairs.map((pair) => responseTransitions(pair.prompt, pair.chosen));
  const probabilities = new Float64Array(TINY_VOCAB_SIZE);
  const gradient = new Float64Array(weights.length);
  for (let step = 0; step < steps; step += 1) {
    gradient.fill(0);
    for (const sequence of sequences) {
      for (const { row, target } of sequence) {
        softmaxInto(weights, row, probabilities);
        for (let k = 0; k < TINY_VOCAB_SIZE; k += 1) {
          gradient[row * TINY_VOCAB_SIZE + k] += (probabilities[k] - (k === target ? 1 : 0)) / pairs.length;
        }
      }
    }
    for (let index = 0; index < weights.length; index += 1) weights[index] -= learningRate * gradient[index];
  }
  return weights;
}

export interface PairScore {
  /** log πθ(y_w|x) − log πref(y_w|x) */
  chosenShift: number;
  /** log πθ(y_l|x) − log πref(y_l|x) */
  rejectedShift: number;
  /** chosenShift − rejectedShift: the log-ratio margin m, before β. */
  margin: number;
  referenceChosen: number;
  referenceRejected: number;
}

export function scorePair(
  policy: Float32Array,
  reference: Float32Array,
  pair: TinyPreferencePair,
): PairScore {
  const referenceChosen = tinySequenceLogProb(reference, pair.prompt, pair.chosen);
  const referenceRejected = tinySequenceLogProb(reference, pair.prompt, pair.rejected);
  const chosenShift = tinySequenceLogProb(policy, pair.prompt, pair.chosen) - referenceChosen;
  const rejectedShift = tinySequenceLogProb(policy, pair.prompt, pair.rejected) - referenceRejected;
  return {
    chosenShift,
    rejectedShift,
    margin: chosenShift - rejectedShift,
    referenceChosen,
    referenceRejected,
  };
}

/** −log σ(z), computed without overflow for large |z|. */
export function negLogSigmoid(z: number): number {
  return z >= 0 ? Math.log1p(Math.exp(-z)) : -z + Math.log1p(Math.exp(z));
}

/** DPO loss for one pair at log-ratio margin m. */
export function dpoLoss(beta: number, margin: number): number {
  return negLogSigmoid(beta * margin);
}

/**
 * How strongly a pair still pulls on the weights: the gradient of the loss with
 * respect to m is −β·σ(−β·m). This returns σ(−β·m), the share of full strength.
 */
export function gradientWeight(beta: number, margin: number): number {
  return 1 / (1 + Math.exp(beta * margin));
}

/** The margin at which the loss falls to a target value: m = −ln(e^L − 1) / β. */
export function marginForLoss(beta: number, loss: number): number {
  return -Math.log(Math.expm1(loss)) / beta;
}

/* -------------------------------------------------------------------------- */
/* Held-out margins along a run                                                */
/* -------------------------------------------------------------------------- */

/**
 * Full-batch DPO with the same gradient as the SDK's `trainTinyDpo`, written to read a set of
 * held-out pairs at chosen step counts in one pass. The SDK run records training diagnostics at
 * every step, which would make five separate runs several times dearer than the lab's own sweep.
 * Returns, for each step in `marks`, the log-ratio margin (before beta) of every pair in `heldOut`.
 */
export function heldOutGapTrace({
  reference,
  pairs,
  heldOut,
  beta,
  learningRate,
  marks,
}: {
  reference: Float32Array;
  pairs: ReadonlyArray<TinyPreferencePair>;
  heldOut: ReadonlyArray<TinyPreferencePair>;
  beta: number;
  learningRate: number;
  /** Ascending step counts, each at least 0. */
  marks: ReadonlyArray<number>;
}): number[][] {
  const weights = Float32Array.from(reference);
  const trained = pairs.map((pair) => ({
    chosen: responseTransitions(pair.prompt, pair.chosen),
    rejected: responseTransitions(pair.prompt, pair.rejected),
  }));
  const referenceMargins = pairs.map(
    (pair) =>
      tinySequenceLogProb(reference, pair.prompt, pair.chosen) -
      tinySequenceLogProb(reference, pair.prompt, pair.rejected),
  );
  const gaps = () => heldOut.map((pair) => scorePair(weights, reference, pair).margin);
  const probabilities = new Float64Array(TINY_VOCAB_SIZE);
  const gradient = new Float64Array(weights.length);
  const logProb = (transitions: ReadonlyArray<Transition>) => {
    let total = 0;
    for (const { row, target } of transitions) {
      softmaxInto(weights, row, probabilities);
      total += Math.log(Math.max(1e-12, probabilities[target]));
    }
    return total;
  };
  const accumulate = (transitions: ReadonlyArray<Transition>, factor: number) => {
    for (const { row, target } of transitions) {
      softmaxInto(weights, row, probabilities);
      for (let k = 0; k < TINY_VOCAB_SIZE; k += 1) {
        gradient[row * TINY_VOCAB_SIZE + k] += factor * ((k === target ? 1 : 0) - probabilities[k]);
      }
    }
  };

  const out: number[][] = [];
  let step = 0;
  for (const mark of marks) {
    while (step < mark && pairs.length > 0) {
      gradient.fill(0);
      trained.forEach((pair, index) => {
        const policyMargin = logProb(pair.chosen) - logProb(pair.rejected);
        const logit = beta * (policyMargin - referenceMargins[index]);
        const factor = (-1 / (1 + Math.exp(logit)) * beta) / pairs.length;
        accumulate(pair.chosen, factor);
        accumulate(pair.rejected, -factor);
      });
      for (let index = 0; index < weights.length; index += 1) weights[index] -= learningRate * gradient[index];
      step += 1;
    }
    out.push(gaps());
  }
  return out;
}
