/**
 * The instrument behind the lesson's underflow claim. `trainTinyExtended` reports overflows but not
 * underflows, so this replays the factored model's plain-SGD run step by step, with the same
 * arithmetic in the same order and the same emulated fp16 cast, and counts every nonzero gradient
 * entry the cast rounds to zero. Its final weights are checked bit for bit against the SDK's in
 * fp16-audit.test.ts, so the counts describe exactly the runs the dashboard draws. The lab itself
 * never calls it.
 */
import { bigramPairs, TINY_VOCAB_SIZE, tinyRandom, toHalf } from "@app/module-sdk";

const V = TINY_VOCAB_SIZE;

export interface HalfAuditOptions {
  text: string;
  rank: number;
  epochs: number;
  batchSize: number;
  learningRate: number;
  seed: number;
  /** Static loss scale 2^exponent; null replays the 32-bit run with no cast at all. */
  scaleExponent: number | null;
}

export interface HalfAudit {
  /** U·V after the last step, comparable with a run's `weights`. */
  weights: Float32Array;
  steps: number;
  /** Nonzero gradient entries over every step, before the cast. */
  nonzero: number;
  /** Nonzero entries the cast rounded to zero. */
  flushed: number;
  /** Nonzero entries the cast left in fp16's subnormal range, below 2^-14 after scaling. */
  subnormal: number;
  /** Smallest magnitude of any nonzero gradient entry, before scaling. */
  smallest: number;
  /** 1-based step on which a weight stopped being finite, or null. */
  haltedAt: number | null;
}

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

export function auditHalfGradients(options: HalfAuditOptions): HalfAudit {
  const { text, rank, epochs, batchSize, learningRate, seed, scaleExponent } = options;
  const scale = scaleExponent === null ? null : 2 ** scaleExponent;
  const random = tinyRandom(seed);
  const left = new Float32Array(V * rank);
  const right = new Float32Array(rank * V);
  for (let index = 0; index < left.length; index += 1) left[index] = (random() - 0.5) * 0.3;
  for (let index = 0; index < right.length; index += 1) right[index] = (random() - 0.5) * 0.3;
  const leftGradient = new Float32Array(left.length);
  const rightGradient = new Float32Array(right.length);
  const { inputs, targets } = bigramPairs(text);
  const pairs = inputs.length;
  const perEpoch = Math.ceil(pairs / batchSize);
  const total = perEpoch * epochs;
  const logits = new Float32Array(V);
  const probabilities = new Float32Array(V);
  const outer = new Float32Array(V);
  let order: Int32Array = new Int32Array(0);
  let nonzero = 0;
  let flushed = 0;
  let subnormal = 0;
  let smallest = Infinity;
  let haltedAt: number | null = null;
  let step = 0;

  for (; step < total && haltedAt === null; step += 1) {
    const batch = step % perEpoch;
    if (batch === 0) order = shuffledOrder(pairs, random);
    const start = batch * batchSize;
    const size = Math.min(pairs, start + batchSize) - start;
    const inverseSize = 1 / size;
    for (let slot = 0; slot < size; slot += 1) {
      const pair = order[start + slot];
      const row = inputs[pair];
      const target = targets[pair];
      for (let k = 0; k < V; k += 1) {
        let sum = 0;
        for (let j = 0; j < rank; j += 1) sum += left[row * rank + j] * right[j * V + k];
        logits[k] = sum;
      }
      let maximum = -Infinity;
      for (let k = 0; k < V; k += 1) maximum = Math.max(maximum, logits[k] / 1);
      let normalizer = 0;
      for (let k = 0; k < V; k += 1) {
        probabilities[k] = Math.exp(logits[k] / 1 - maximum);
        normalizer += probabilities[k];
      }
      for (let k = 0; k < V; k += 1) probabilities[k] /= normalizer;
      for (let k = 0; k < V; k += 1) outer[k] = (probabilities[k] - (k === target ? 1 : 0)) * inverseSize;
      for (let j = 0; j < rank; j += 1) {
        let leftTerm = 0;
        for (let k = 0; k < V; k += 1) {
          leftTerm += outer[k] * right[j * V + k];
          rightGradient[j * V + k] += outer[k] * left[row * rank + j];
        }
        leftGradient[row * rank + j] += leftTerm;
      }
    }

    for (const gradient of [leftGradient, rightGradient]) {
      for (let index = 0; index < gradient.length; index += 1) {
        if (gradient[index] === 0) continue;
        nonzero += 1;
        smallest = Math.min(smallest, Math.abs(gradient[index]));
        if (scale === null) continue;
        const half = toHalf(gradient[index] * scale);
        if (half === 0) flushed += 1;
        else if (Number.isFinite(half) && Math.abs(half) < 2 ** -14) subnormal += 1;
        gradient[index] = half / scale;
      }
    }

    for (let index = 0; index < left.length; index += 1) left[index] -= learningRate * (leftGradient[index] * 1);
    for (let index = 0; index < right.length; index += 1) right[index] -= learningRate * (rightGradient[index] * 1);
    leftGradient.fill(0);
    rightGradient.fill(0);
    for (const matrix of [left, right]) {
      for (let index = 0; index < matrix.length; index += 1) {
        if (!Number.isFinite(matrix[index])) {
          haltedAt = step + 1;
          break;
        }
      }
      if (haltedAt !== null) break;
    }
  }

  const weights = new Float32Array(V * V);
  for (let row = 0; row < V; row += 1) {
    for (let k = 0; k < V; k += 1) {
      let sum = 0;
      for (let j = 0; j < rank; j += 1) sum += left[row * rank + j] * right[j * V + k];
      weights[row * V + k] = sum;
    }
  }
  return { weights, steps: step, nonzero, flushed, subnormal, smallest, haltedAt };
}
