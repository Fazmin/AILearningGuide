/**
 * One real minibatch update, replayed from a saved checkpoint.
 *
 * `trainTinyModel` shuffles the training pairs once per epoch with a seeded generator, cuts the
 * shuffle into batches, and subtracts `learningRate × batch gradient` from every row the batch
 * touched. A checkpoint stores the weights after step `s`, so the update the run actually took
 * next is batch `s mod batchesPerEpoch` of epoch `floor(s / batchesPerEpoch)`. This file rebuilds
 * that batch and applies the same update, so the rows it shows changing are the rows the run
 * changed. `step.test.ts` pins the equivalence against the trainer's own per-step checkpoints.
 */
import { bigramPairs, tinyRandom, TINY_VOCAB, TINY_VOCAB_SIZE } from "@app/module-sdk";

const V = TINY_VOCAB_SIZE;

/** Learning rate of the run this module trains. Constant, so no schedule factor applies. */
export const STEP_LEARNING_RATE = 0.6;

export interface StepBatch {
  /** Global step index of the update this batch feeds (0 is the first update of the run). */
  step: number;
  epoch: number;
  /** Position of the batch inside its epoch. */
  batchInEpoch: number;
  inputs: number[];
  targets: number[];
}

/** The same Fisher-Yates shuffle the trainer uses, drawn from the same generator stream. */
function shuffledOrder(count: number, random: () => number): Int32Array {
  const order = new Int32Array(count);
  for (let index = 0; index < count; index += 1) order[index] = index;
  for (let index = count - 1; index > 0; index -= 1) {
    const swap = Math.floor(random() * (index + 1));
    const held = order[index];
    order[index] = order[swap];
    order[swap] = held;
  }
  return order;
}

/**
 * The minibatch the run used for global step `step`. Steps beyond the end of the run keep drawing
 * further epochs from the same generator, so "one step past the final checkpoint" is defined too.
 */
export function stepBatch(text: string, batchSize: number, seed: number, step: number): StepBatch {
  const { inputs, targets } = bigramPairs(text);
  const pairCount = inputs.length;
  const size = Math.max(1, Math.min(batchSize, Math.max(1, pairCount)));
  const batchesPerEpoch = Math.max(1, Math.ceil(pairCount / size));
  const epoch = Math.floor(step / batchesPerEpoch);
  const batchInEpoch = step % batchesPerEpoch;
  const random = tinyRandom(seed);
  let order = shuffledOrder(pairCount, random);
  for (let pass = 0; pass < epoch; pass += 1) order = shuffledOrder(pairCount, random);
  const start = batchInEpoch * size;
  const end = Math.min(pairCount, start + size);
  const chosen = Array.from(order.subarray(start, end));
  return {
    step,
    epoch,
    batchInEpoch,
    inputs: chosen.map((pair) => inputs[pair]),
    targets: chosen.map((pair) => targets[pair]),
  };
}

function rowProbabilities(weights: Float32Array, row: number, into: Float32Array) {
  const offset = row * V;
  let maximum = -Infinity;
  for (let k = 0; k < V; k += 1) if (weights[offset + k] > maximum) maximum = weights[offset + k];
  let total = 0;
  for (let k = 0; k < V; k += 1) {
    const value = Math.exp(weights[offset + k] - maximum);
    into[k] = value;
    total += value;
  }
  for (let k = 0; k < V; k += 1) into[k] /= total;
}

/** Mean cross-entropy of a batch under the given weights, in nats per token. */
export function batchLoss(weights: Float32Array, batch: Pick<StepBatch, "inputs" | "targets">): number {
  const probabilities = new Float32Array(V);
  let total = 0;
  for (let index = 0; index < batch.inputs.length; index += 1) {
    rowProbabilities(weights, batch.inputs[index], probabilities);
    total -= Math.log(Math.max(1e-12, probabilities[batch.targets[index]]));
  }
  return total / Math.max(1, batch.inputs.length);
}

export interface RowChange {
  /** Context row: the previous character. */
  row: number;
  /** How many pairs in the batch start from this row. */
  pairs: number;
  /** Length of this row's slice of the batch gradient. */
  gradientNorm: number;
  /** The next character this row was asked for most often in the batch (ties: lowest id). */
  target: number;
  logitBefore: number;
  logitAfter: number;
  probabilityBefore: number;
  probabilityAfter: number;
}

export interface SingleStep {
  batch: StepBatch;
  before: Float32Array;
  after: Float32Array;
  lossBefore: number;
  lossAfter: number;
  /** Length of the whole batch gradient, the quantity the trainer logs and clips. */
  gradientNorm: number;
  rows: RowChange[];
  weightsChanged: number;
}

/** Applies the batch's gradient once: `weights -= learningRate × mean over pairs of (p − onehot)`. */
export function applyBatchStep(
  weights: Float32Array,
  batch: StepBatch,
  learningRate = STEP_LEARNING_RATE,
): SingleStep {
  const before = Float32Array.from(weights);
  const after = Float32Array.from(weights);
  const gradient = new Float32Array(V * V);
  const probabilities = new Float32Array(V);
  const inverseSize = 1 / Math.max(1, batch.inputs.length);
  const touched = new Set<number>();
  const targetCounts = new Map<number, Map<number, number>>();
  const pairsPerRow = new Map<number, number>();

  for (let index = 0; index < batch.inputs.length; index += 1) {
    const row = batch.inputs[index];
    const target = batch.targets[index];
    rowProbabilities(before, row, probabilities);
    const offset = row * V;
    for (let k = 0; k < V; k += 1) gradient[offset + k] += (probabilities[k] - (k === target ? 1 : 0)) * inverseSize;
    touched.add(row);
    pairsPerRow.set(row, (pairsPerRow.get(row) ?? 0) + 1);
    const counts = targetCounts.get(row) ?? new Map<number, number>();
    counts.set(target, (counts.get(target) ?? 0) + 1);
    targetCounts.set(row, counts);
  }

  let squaredNorm = 0;
  const rowNorms = new Map<number, number>();
  for (const row of touched) {
    let rowSquared = 0;
    for (let k = 0; k < V; k += 1) rowSquared += gradient[row * V + k] ** 2;
    rowNorms.set(row, Math.sqrt(rowSquared));
    squaredNorm += rowSquared;
    for (let k = 0; k < V; k += 1) after[row * V + k] -= learningRate * gradient[row * V + k];
  }

  const afterProbabilities = new Float32Array(V);
  const rows: RowChange[] = Array.from(touched, (row) => {
    const counts = [...(targetCounts.get(row) ?? [])].sort((left, right) => right[1] - left[1] || left[0] - right[0]);
    const target = counts[0][0];
    rowProbabilities(before, row, probabilities);
    rowProbabilities(after, row, afterProbabilities);
    return {
      row,
      pairs: pairsPerRow.get(row) ?? 0,
      gradientNorm: rowNorms.get(row) ?? 0,
      target,
      logitBefore: before[row * V + target],
      logitAfter: after[row * V + target],
      probabilityBefore: probabilities[target],
      probabilityAfter: afterProbabilities[target],
    };
  }).sort((left, right) => right.gradientNorm - left.gradientNorm || left.row - right.row);

  let weightsChanged = 0;
  for (let index = 0; index < after.length; index += 1) if (after[index] !== before[index]) weightsChanged += 1;

  return {
    batch,
    before,
    after,
    lossBefore: batchLoss(before, batch),
    lossAfter: batchLoss(after, batch),
    gradientNorm: Math.sqrt(squaredNorm),
    rows,
    weightsChanged,
  };
}

/**
 * Starting from the checkpoint at `startStep`, takes `count` real steps in a row and returns the
 * last one, with `before` being the weights after `count - 1` steps. Returns null for a count of 0.
 */
export function stepFromCheckpoint(
  checkpointWeights: Float32Array,
  startStep: number,
  count: number,
  text: string,
  batchSize: number,
  seed: number,
  learningRate = STEP_LEARNING_RATE,
): SingleStep | null {
  if (count <= 0) return null;
  let weights = checkpointWeights;
  let last: SingleStep | null = null;
  for (let taken = 0; taken < count; taken += 1) {
    last = applyBatchStep(weights, stepBatch(text, batchSize, seed, startStep + taken), learningRate);
    weights = last.after;
  }
  return last;
}

/** A pair as the lab prints it: previous character, arrow, next character, with a visible space. */
export const pairLabel = (input: number, target: number) => {
  const glyph = (id: number) => (TINY_VOCAB[id] === " " ? "␣" : TINY_VOCAB[id]);
  return `${glyph(input)}→${glyph(target)}`;
};
