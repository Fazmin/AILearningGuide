/**
 * Singular values of a small matrix, for the Orthogonalized update card.
 *
 * The singular values of M are the square roots of the eigenvalues of MᵀM, a symmetric matrix, so a
 * cyclic Jacobi eigen-solver is enough: each rotation zeroes one off-diagonal entry, and a few sweeps
 * drive the whole off-diagonal to zero. On the 30 × 30 table that takes well under a millisecond.
 * Going through MᵀM squares the matrix, so a singular value below about 1e-8 of the largest cannot be
 * told from zero; `ZERO_SINGULAR` is set well above that floor.
 */

/**
 * A singular value at or below this counts as a zero direction. Every matrix measured here is
 * normalized, so its largest singular value sits between about 0.2 and 1.2. An exact zero read
 * through MᵀM comes back as about 1e-8, and the weakest real direction the lab captures is about 2e-6.
 */
export const ZERO_SINGULAR = 1e-7;

/** Eigenvalues of a symmetric n × n matrix (row-major), largest first. */
export function symmetricEigenvalues(source: ArrayLike<number>, n: number, maxSweeps = 60): number[] {
  const a = Float64Array.from({ length: n * n }, (_, index) => source[index] ?? 0);
  let size = 0;
  for (let index = 0; index < n * n; index += 1) size += a[index] * a[index];
  // Stop once the off-diagonal mass is negligible against the whole matrix.
  const floor = 1e-30 * Math.max(size, Number.MIN_VALUE);
  for (let sweep = 0; sweep < maxSweeps; sweep += 1) {
    let off = 0;
    for (let p = 0; p < n; p += 1) {
      for (let q = p + 1; q < n; q += 1) off += a[p * n + q] * a[p * n + q];
    }
    if (off <= floor) break;
    for (let p = 0; p < n - 1; p += 1) {
      for (let q = p + 1; q < n; q += 1) {
        const apq = a[p * n + q];
        if (apq === 0) continue;
        // Rotate in the (p, q) plane by the angle that zeroes a[p][q].
        const theta = (a[q * n + q] - a[p * n + p]) / (2 * apq);
        const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k += 1) {
          const akp = a[k * n + p];
          const akq = a[k * n + q];
          a[k * n + p] = c * akp - s * akq;
          a[k * n + q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k += 1) {
          const apk = a[p * n + k];
          const aqk = a[q * n + k];
          a[p * n + k] = c * apk - s * aqk;
          a[q * n + k] = s * apk + c * aqk;
        }
      }
    }
  }
  return Array.from({ length: n }, (_, index) => a[index * n + index]).sort((left, right) => right - left);
}

/** Singular values of a rows × cols matrix (row-major), largest first: √eig(MᵀM). */
export function singularValues(matrix: ArrayLike<number>, rows: number, cols: number): number[] {
  const gram = new Float64Array(cols * cols);
  for (let i = 0; i < cols; i += 1) {
    for (let j = i; j < cols; j += 1) {
      let sum = 0;
      for (let r = 0; r < rows; r += 1) sum += (matrix[r * cols + i] ?? 0) * (matrix[r * cols + j] ?? 0);
      gram[i * cols + j] = sum;
      gram[j * cols + i] = sum;
    }
  }
  return symmetricEigenvalues(gram, cols).map((value) => Math.sqrt(Math.max(0, value)));
}

/** Muon's quintic, applied to one singular value: σ → aσ + bσ³ + cσ⁵ (Jordan et al., 2024). */
export const NEWTON_SCHULZ_COEFFICIENTS = [3.4445, -4.775, 2.0315] as const;

/**
 * What `steps` Newton–Schulz iterations do to one singular value of the Frobenius-normalized matrix.
 * Every iteration multiplies X by a polynomial in XXᵀ, so it keeps the singular vectors and maps each
 * singular value through the same scalar polynomial. Zero stays zero.
 */
export function newtonSchulzScalar(value: number, steps: number): number {
  const [a, b, c] = NEWTON_SCHULZ_COEFFICIENTS;
  let x = value;
  for (let step = 0; step < steps; step += 1) x = a * x + b * x ** 3 + c * x ** 5;
  return x;
}

export interface SpectrumSummary {
  /** Singular values, largest first. */
  values: number[];
  /** Directions with a singular value above `ZERO_SINGULAR`. */
  nonzero: number;
  /** Directions at zero: no row or column of the update points that way. */
  zeros: number;
  largest: number;
  /** Smallest singular value that is not a zero direction. */
  smallestNonzero: number;
  /** Nonzero singular values between `low` and `high`. */
  inBand: number;
}

export const BAND_LOW = 0.5;
export const BAND_HIGH = 1.5;

export function summarizeSpectrum(values: ReadonlyArray<number>): SpectrumSummary {
  const sorted = [...values].sort((left, right) => right - left);
  const nonzeroValues = sorted.filter((value) => value > ZERO_SINGULAR);
  return {
    values: sorted,
    nonzero: nonzeroValues.length,
    zeros: sorted.length - nonzeroValues.length,
    largest: sorted[0] ?? 0,
    smallestNonzero: nonzeroValues.length ? nonzeroValues[nonzeroValues.length - 1] : 0,
    inBand: nonzeroValues.filter((value) => value >= BAND_LOW && value <= BAND_HIGH).length,
  };
}
