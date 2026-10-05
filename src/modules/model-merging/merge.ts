/**
 * Merge algebra on task vectors. A task vector is `tuned − base`, one entry per
 * weight, so every function here is ordinary arithmetic on flat arrays. The
 * same code runs on the 900-entry vectors of the lab and on the hand-written
 * five-entry vectors in the tests.
 */

import { tinyRandom } from "@app/module-sdk";

export type MergeMethod = "linear" | "task" | "ties" | "dare" | "slerp";

export interface MergeParams {
  /** Blend position for linear and SLERP: 0 is all task A, 1 is all task B. */
  mix: number;
  /** Scaling coefficient applied to the merged task vector (task, TIES, DARE). */
  lambda: number;
  /** TIES: share of each task vector's entries kept, by magnitude. */
  density: number;
  /** DARE: probability that any one entry is dropped. */
  dropRate: number;
  /** DARE: seed for the two drop masks. */
  maskSeed: number;
}

export interface MergeResult {
  /** The combined task vector. Add it to the base to get the served weights. */
  vector: Float32Array;
  /** TIES: the trimmed copies of each input. DARE: the dropped-and-rescaled copies. */
  processed: [Float32Array, Float32Array];
  /** TIES: elected sign per entry (−1, 0, +1). Empty for other methods. */
  elected: Int8Array;
  /** SLERP: angle between the two task vectors in degrees. NaN otherwise. */
  angle: number;
}

export function taskVector(tuned: Float32Array, base: Float32Array): Float32Array {
  const out = new Float32Array(base.length);
  for (let index = 0; index < base.length; index += 1) out[index] = tuned[index] - base[index];
  return out;
}

export function addToBase(base: Float32Array, vector: Float32Array, scale = 1): Float32Array {
  const out = Float32Array.from(base);
  for (let index = 0; index < out.length; index += 1) out[index] += scale * vector[index];
  return out;
}

export function norm(vector: ArrayLike<number>): number {
  let total = 0;
  for (let index = 0; index < vector.length; index += 1) total += vector[index] ** 2;
  return Math.sqrt(total);
}

/** Cosine similarity, or NaN when either vector is all zeros. */
export function cosine(left: ArrayLike<number>, right: ArrayLike<number>): number {
  let dot = 0;
  for (let index = 0; index < left.length; index += 1) dot += left[index] * right[index];
  const denominator = norm(left) * norm(right);
  return denominator === 0 ? Number.NaN : dot / denominator;
}

export interface SignStats {
  /** Entries where both vectors are non-zero. */
  overlap: number;
  /** Overlapping entries whose signs disagree. */
  conflicts: number;
  onlyA: number;
  onlyB: number;
}

export function signStats(
  left: ArrayLike<number>,
  right: ArrayLike<number>,
  epsilon = 1e-6,
): SignStats {
  let overlap = 0;
  let conflicts = 0;
  let onlyA = 0;
  let onlyB = 0;
  for (let index = 0; index < left.length; index += 1) {
    const a = Math.abs(left[index]) > epsilon;
    const b = Math.abs(right[index]) > epsilon;
    if (a && b) {
      overlap += 1;
      if (Math.sign(left[index]) !== Math.sign(right[index])) conflicts += 1;
    } else if (a) onlyA += 1;
    else if (b) onlyB += 1;
  }
  return { overlap, conflicts, onlyA, onlyB };
}

/**
 * TIES step 1, trim: keep the `density` share of entries with the largest
 * magnitude and zero the rest. Ties at the threshold are broken by index so the
 * kept count is exact.
 */
export function trimTopK(vector: Float32Array, density: number): Float32Array {
  const share = Math.min(1, Math.max(0, density));
  const keep = Math.round(share * vector.length);
  const order = Array.from(vector, (_, index) => index).sort(
    (left, right) => Math.abs(vector[right]) - Math.abs(vector[left]) || left - right,
  );
  const out = new Float32Array(vector.length);
  for (let rank = 0; rank < keep; rank += 1) out[order[rank]] = vector[order[rank]];
  return out;
}

/**
 * TIES-Merging (Yadav et al., 2023): trim each task vector, elect one sign per
 * entry from the summed trimmed values (the sign with more total mass), then
 * average only the entries that agree with the elected sign ("disjoint mean").
 */
export function tiesMerge(
  vectors: ReadonlyArray<Float32Array>,
  density: number,
): { merged: Float32Array; trimmed: Float32Array[]; elected: Int8Array } {
  const length = vectors[0]?.length ?? 0;
  const trimmed = vectors.map((vector) => trimTopK(vector, density));
  const merged = new Float32Array(length);
  const elected = new Int8Array(length);
  for (let index = 0; index < length; index += 1) {
    let total = 0;
    for (const vector of trimmed) total += vector[index];
    const sign = total > 0 ? 1 : total < 0 ? -1 : 0;
    elected[index] = sign;
    if (sign === 0) continue;
    let sum = 0;
    let count = 0;
    for (const vector of trimmed) {
      const value = vector[index];
      if (value !== 0 && Math.sign(value) === sign) {
        sum += value;
        count += 1;
      }
    }
    merged[index] = count > 0 ? sum / count : 0;
  }
  return { merged, trimmed, elected };
}

/** 1 keeps an entry, 0 drops it. Each entry is dropped with probability p. */
export function dropMask(length: number, dropRate: number, seed: number): Uint8Array {
  const random = tinyRandom(seed);
  const mask = new Uint8Array(length);
  for (let index = 0; index < length; index += 1) mask[index] = random() >= dropRate ? 1 : 0;
  return mask;
}

/**
 * DARE (Yu et al., 2023): drop entries at random and rescale the survivors by
 * 1 / (1 − p), so every entry keeps its expected value.
 */
export function dareApply(vector: Float32Array, mask: Uint8Array, dropRate: number): Float32Array {
  const rate = Math.min(0.999, Math.max(0, dropRate));
  const scale = 1 / (1 - rate);
  const out = new Float32Array(vector.length);
  for (let index = 0; index < vector.length; index += 1) {
    out[index] = mask[index] ? vector[index] * scale : 0;
  }
  return out;
}

/**
 * Spherical interpolation between two vectors: the angle is measured on
 * normalized copies, and the weights sin((1−t)Ω)/sin Ω and sin(tΩ)/sin Ω are
 * applied to the originals, as mergekit does. Nearly parallel vectors fall back
 * to linear interpolation. Returns null when either vector is all zeros.
 */
export function slerp(
  left: Float32Array,
  right: Float32Array,
  t: number,
): { vector: Float32Array; angle: number } | null {
  const similarity = cosine(left, right);
  if (!Number.isFinite(similarity)) return null;
  const omega = Math.acos(Math.min(1, Math.max(-1, similarity)));
  const out = new Float32Array(left.length);
  if (omega < 1e-4) {
    for (let index = 0; index < out.length; index += 1) {
      out[index] = (1 - t) * left[index] + t * right[index];
    }
    return { vector: out, angle: 0 };
  }
  const sine = Math.sin(omega);
  const leftWeight = Math.sin((1 - t) * omega) / sine;
  const rightWeight = Math.sin(t * omega) / sine;
  for (let index = 0; index < out.length; index += 1) {
    out[index] = leftWeight * left[index] + rightWeight * right[index];
  }
  return { vector: out, angle: (omega * 180) / Math.PI };
}

/** One merge of two task vectors by the named method. Null only for SLERP on a zero vector. */
export function mergeTaskVectors(
  method: MergeMethod,
  taskA: Float32Array,
  taskB: Float32Array,
  params: MergeParams,
): MergeResult | null {
  const length = taskA.length;
  const empty = new Int8Array(0);
  const combine = (a: Float32Array, b: Float32Array, weightA: number, weightB: number) => {
    const out = new Float32Array(length);
    for (let index = 0; index < length; index += 1) {
      out[index] = weightA * a[index] + weightB * b[index];
    }
    return out;
  };

  switch (method) {
    case "linear":
      return {
        vector: combine(taskA, taskB, 1 - params.mix, params.mix),
        processed: [taskA, taskB],
        elected: empty,
        angle: Number.NaN,
      };
    case "task":
      return {
        vector: combine(taskA, taskB, params.lambda, params.lambda),
        processed: [taskA, taskB],
        elected: empty,
        angle: Number.NaN,
      };
    case "ties": {
      const { merged, trimmed, elected } = tiesMerge([taskA, taskB], params.density);
      const vector = new Float32Array(length);
      for (let index = 0; index < length; index += 1) vector[index] = params.lambda * merged[index];
      return { vector, processed: [trimmed[0], trimmed[1]], elected, angle: Number.NaN };
    }
    case "dare": {
      const dropA = dareApply(taskA, dropMask(length, params.dropRate, params.maskSeed * 2 + 1), params.dropRate);
      const dropB = dareApply(taskB, dropMask(length, params.dropRate, params.maskSeed * 2 + 2), params.dropRate);
      return {
        vector: combine(dropA, dropB, params.lambda, params.lambda),
        processed: [dropA, dropB],
        elected: empty,
        angle: Number.NaN,
      };
    }
    case "slerp": {
      const result = slerp(taskA, taskB, params.mix);
      if (!result) return null;
      return {
        vector: result.vector,
        processed: [taskA, taskB],
        elected: empty,
        angle: result.angle,
      };
    }
  }
}

/**
 * Picks coordinates worth drawing: the largest sign conflicts and the largest
 * agreements, ranked by |a| + |b|. Returned in that order, conflicts first.
 */
export function pickCoordinates(
  taskA: Float32Array,
  taskB: Float32Array,
  perKind: number,
): { index: number; conflict: boolean }[] {
  const ranked = Array.from(taskA, (_, index) => index)
    .filter((index) => Math.abs(taskA[index]) > 1e-6 && Math.abs(taskB[index]) > 1e-6)
    .sort(
      (left, right) =>
        Math.abs(taskA[right]) + Math.abs(taskB[right]) -
          (Math.abs(taskA[left]) + Math.abs(taskB[left])) || left - right,
    );
  const conflicts = ranked
    .filter((index) => Math.sign(taskA[index]) !== Math.sign(taskB[index]))
    .slice(0, perKind)
    .map((index) => ({ index, conflict: true }));
  const agreements = ranked
    .filter((index) => Math.sign(taskA[index]) === Math.sign(taskB[index]))
    .slice(0, perKind)
    .map((index) => ({ index, conflict: false }));
  return [...conflicts, ...agreements];
}
