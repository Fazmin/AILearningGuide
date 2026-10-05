/**
 * Small, exact linear algebra for the LoRA lab: singular values of a square
 * update matrix, row centering, and the unmerged forward pass used to check a
 * merge. Everything runs in float64 on at most a 30 × 30 matrix.
 */

import { bigramPairs } from "@app/module-sdk";

/**
 * Subtracts each row's mean. A softmax row ignores any constant added to all of
 * its logits, so this keeps exactly the part of an update that changes
 * predictions. Right-multiplying by a centering matrix cannot raise the rank.
 */
export function centerRows(matrix: ArrayLike<number>, size: number): Float64Array {
  const out = new Float64Array(size * size);
  for (let row = 0; row < size; row += 1) {
    let mean = 0;
    for (let column = 0; column < size; column += 1) mean += matrix[row * size + column];
    mean /= size;
    for (let column = 0; column < size; column += 1) {
      out[row * size + column] = matrix[row * size + column] - mean;
    }
  }
  return out;
}

/** Eigenvalues of a symmetric matrix by cyclic Jacobi rotations. */
export function symmetricEigenvalues(input: ReadonlyArray<ReadonlyArray<number>>): number[] {
  const n = input.length;
  const a = input.map((row) => row.slice());
  for (let sweep = 0; sweep < 100; sweep += 1) {
    let off = 0;
    for (let p = 0; p < n; p += 1) for (let q = p + 1; q < n; q += 1) off += a[p][q] ** 2;
    if (off < 1e-24) break;
    for (let p = 0; p < n; p += 1) {
      for (let q = p + 1; q < n; q += 1) {
        if (Math.abs(a[p][q]) < 1e-300) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k += 1) {
          const kp = a[k][p];
          const kq = a[k][q];
          a[k][p] = c * kp - s * kq;
          a[k][q] = s * kp + c * kq;
        }
        for (let k = 0; k < n; k += 1) {
          const pk = a[p][k];
          const qk = a[q][k];
          a[p][k] = c * pk - s * qk;
          a[q][k] = s * pk + c * qk;
        }
      }
    }
  }
  return a.map((row, index) => row[index]);
}

/** Singular values of a rows × columns matrix, largest first. */
export function singularValues(matrix: ArrayLike<number>, rows: number, columns: number): number[] {
  const gram: number[][] = Array.from({ length: columns }, () => new Array<number>(columns).fill(0));
  for (let i = 0; i < columns; i += 1) {
    for (let j = i; j < columns; j += 1) {
      let total = 0;
      for (let k = 0; k < rows; k += 1) total += matrix[k * columns + i] * matrix[k * columns + j];
      gram[i][j] = total;
      gram[j][i] = total;
    }
  }
  return symmetricEigenvalues(gram)
    .map((value) => Math.sqrt(Math.max(0, value)))
    .sort((left, right) => right - left);
}

/** Share of the squared Frobenius norm held by the largest `rank` singular values. */
export function energyCaptured(values: ReadonlyArray<number>, rank: number): number {
  const total = values.reduce((sum, value) => sum + value * value, 0);
  if (total === 0) return 0;
  return values.slice(0, rank).reduce((sum, value) => sum + value * value, 0) / total;
}

/** Numerical rank: singular values above a tolerance relative to the largest one. */
export function numericalRank(values: ReadonlyArray<number>, tolerance = 1e-5): number {
  const largest = values[0] ?? 0;
  if (largest === 0) return 0;
  return values.filter((value) => value > largest * tolerance).length;
}

/**
 * The unmerged forward pass, in float64: logits for context row `row` are
 * `W[row] + scale · B[row] · A`, computed without ever forming `B · A`.
 */
export function unmergedLogits(
  base: Float32Array,
  up: Float32Array,
  down: Float32Array,
  rank: number,
  scale: number,
  size: number,
  row: number,
): Float64Array {
  const logits = new Float64Array(size);
  for (let column = 0; column < size; column += 1) {
    let adaptation = 0;
    for (let j = 0; j < rank; j += 1) adaptation += up[row * rank + j] * down[j * size + column];
    logits[column] = base[row * size + column] + scale * adaptation;
  }
  return logits;
}

export interface MergeCheck {
  /** Largest |merged − unmerged| over every logit of every row. */
  maxLogitGap: number;
  /** Cross-entropy on a text through the unmerged path. */
  unmergedLoss: number;
}

/** Compares a merged table against the unmerged path, logit by logit and on a corpus. */
export function checkMerge(
  merged: Float32Array,
  base: Float32Array,
  up: Float32Array,
  down: Float32Array,
  rank: number,
  scale: number,
  size: number,
  text: string,
): MergeCheck {
  const rows: Float64Array[] = [];
  let maxLogitGap = 0;
  for (let row = 0; row < size; row += 1) {
    const logits = unmergedLogits(base, up, down, rank, scale, size, row);
    rows.push(logits);
    for (let column = 0; column < size; column += 1) {
      maxLogitGap = Math.max(maxLogitGap, Math.abs(merged[row * size + column] - logits[column]));
    }
  }
  const { inputs, targets } = bigramPairs(text);
  let total = 0;
  for (let index = 0; index < inputs.length; index += 1) {
    const logits = rows[inputs[index]];
    let maximum = -Infinity;
    for (const value of logits) maximum = Math.max(maximum, value);
    let partition = 0;
    for (const value of logits) partition += Math.exp(value - maximum);
    total -= logits[targets[index]] - maximum - Math.log(partition);
  }
  return { maxLogitGap, unmergedLoss: inputs.length ? total / inputs.length : 0 };
}
