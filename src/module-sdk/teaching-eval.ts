import { encodeTinyText, tinyNextDistribution, tinySequenceLogProb } from "./tiny-lm";

/**
 * Scoring and the statistics that keep a leaderboard honest.
 *
 * A candidate is scored as the continuation " candidate" after the context:
 * the space transition plus every candidate character, averaged over those
 * transitions. That is per-byte normalization of the continuation, the same
 * idea as lm-evaluation-harness's acc_norm.
 */

export interface BenchmarkItem {
  context: string;
  correct: string;
  distractor: string;
}

export function parseItems(text: string): BenchmarkItem[] {
  return text
    .split("\n")
    .map((line) => line.split("|").map((part) => part.trim().toLowerCase()))
    .filter((parts) => parts.length >= 3 && parts[0] && parts[1] && parts[2])
    .map((parts) => ({ context: parts[0], correct: parts[1], distractor: parts[2] }));
}

/** Total and mean log-probability of " candidate" after the context. */
export function scoreContinuation(weights: Float32Array, context: string, candidate: string) {
  const space = Math.log(Math.max(1e-12, tinyNextDistribution(weights, context)[0]));
  const body = tinySequenceLogProb(weights, `${context} `, candidate);
  const transitions = encodeTinyText(candidate).length + 1;
  return { total: space + body, transitions, mean: (space + body) / transitions };
}

const Z95 = 1.959964;

/** Standard error of a proportion, sqrt(p(1 − p) / n). */
export function proportionStandardError(p: number, n: number) {
  return n > 0 ? Math.sqrt((p * (1 - p)) / n) : Number.NaN;
}

/**
 * Wilson 95% interval for k successes in n trials. Unlike p ± 1.96·SE it does
 * not collapse to zero width at 0/n or n/n, which matters at six items.
 */
export function wilsonInterval(k: number, n: number, z = Z95): [number, number] {
  if (n <= 0) return [0, 1];
  const p = k / n;
  const z2 = z * z;
  const denominator = 1 + z2 / n;
  const centre = (p + z2 / (2 * n)) / denominator;
  const half = (z * Math.sqrt((p * (1 - p)) / n + z2 / (4 * n * n))) / denominator;
  return [Math.max(0, centre - half), Math.min(1, centre + half)];
}

/** Two-sided 97.5th percentile of Student's t for small samples. */
const T975 = [
  Number.NaN, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.16, 2.145, 2.131,
  2.12, 2.11, 2.101, 2.093, 2.086, 2.08, 2.074, 2.069, 2.064, 2.06, 2.056, 2.052, 2.048, 2.045, 2.042,
];

export function tCritical(degreesOfFreedom: number) {
  if (degreesOfFreedom < 1) return Number.NaN;
  return degreesOfFreedom < T975.length ? T975[degreesOfFreedom] : Z95;
}

/** Mean with a 95% t-interval. */
export function meanInterval(values: ReadonlyArray<number>) {
  const n = values.length;
  const mean = n ? values.reduce((sum, value) => sum + value, 0) / n : Number.NaN;
  if (n < 2) return { mean, low: Number.NaN, high: Number.NaN, standardError: Number.NaN };
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (n - 1);
  const standardError = Math.sqrt(variance / n);
  const half = tCritical(n - 1) * standardError;
  return { mean, low: mean - half, high: mean + half, standardError };
}

/** Paired difference a − b over the same items, with a 95% t-interval. */
export function pairedDifference(a: ReadonlyArray<number>, b: ReadonlyArray<number>) {
  return meanInterval(a.map((value, index) => value - (b[index] ?? 0)));
}

/** Items needed for a 95% interval of ± halfWidth at pass rate p (normal approximation). */
export function itemsForHalfWidth(p: number, halfWidth: number) {
  const variance = Math.max(p * (1 - p), 1e-9);
  return Math.ceil((Z95 * Z95 * variance) / (halfWidth * halfWidth));
}

/**
 * Unbiased pass@k from n samples with c correct: 1 − C(n − c, k) / C(n, k),
 * computed as a running product so large n does not overflow.
 */
export function passAtK(n: number, c: number, k: number) {
  if (k > n || n <= 0) return Number.NaN;
  if (n - c < k) return 1;
  let failAll = 1;
  for (let i = n - c + 1; i <= n; i += 1) failAll *= 1 - k / i;
  return 1 - failAll;
}
