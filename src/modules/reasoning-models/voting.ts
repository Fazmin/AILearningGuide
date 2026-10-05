/**
 * Exact accuracy of two test-time-compute strategies over n independent samples,
 * each correct with probability p.
 *
 * - Verifier picks: a perfect checker accepts any correct sample, so the
 *   strategy succeeds when at least one sample is correct: 1 − (1 − p)^n. This
 *   is pass@n, the ceiling every selection method is bounded by.
 * - Majority vote (self-consistency): the most common final answer wins, ties
 *   broken uniformly at random. Wrong samples are spread evenly over `wrong`
 *   distinct wrong answers, or are all different when `wrong` is "unique".
 */

export type WrongSpread = number | "unique";

export function binomialPmf(n: number, p: number): number[] {
  const pmf: number[] = [];
  let logChoose = 0;
  for (let k = 0; k <= n; k += 1) {
    if (k > 0) logChoose += Math.log(n - k + 1) - Math.log(k);
    const logP =
      logChoose + (k > 0 ? k * Math.log(p) : 0) + (n - k > 0 ? (n - k) * Math.log(1 - p) : 0);
    pmf.push(p <= 0 ? (k === 0 ? 1 : 0) : p >= 1 ? (k === n ? 1 : 0) : Math.exp(logP));
  }
  return pmf;
}

export function verifierAccuracy(n: number, p: number) {
  return 1 - (1 - p) ** n;
}

const logFactorial = (() => {
  const table = [0];
  for (let k = 1; k <= 256; k += 1) table.push(table[k - 1] + Math.log(k));
  return (k: number) => table[k];
})();

/**
 * Probability that the correct answer wins a plurality vote given it received
 * `correct` of `n` votes and the rest are spread evenly over `wrong` answers.
 */
export function winGivenCorrect(n: number, correct: number, wrong: WrongSpread): number {
  const rest = n - correct;
  if (rest === 0) return correct > 0 ? 1 : 0;
  if (wrong === "unique") {
    if (correct >= 2) return 1;
    if (correct === 1) return 1 / (rest + 1);
    return 0;
  }
  const m = Math.max(1, Math.round(wrong));
  // f[s][t]: sum over count assignments to the categories seen so far that use
  // s samples, keep every count ≤ correct, and have t categories tied at
  // `correct`, of the product of 1/k! (the multinomial weight without r!).
  let f = new Map<string, number>([["0,0", 1]]);
  for (let category = 0; category < m; category += 1) {
    const next = new Map<string, number>();
    for (const [key, weight] of f) {
      const [used, tied] = key.split(",").map(Number);
      for (let k = 0; k <= Math.min(correct, rest - used); k += 1) {
        const target = `${used + k},${tied + (k === correct ? 1 : 0)}`;
        next.set(target, (next.get(target) ?? 0) + weight * Math.exp(-logFactorial(k)));
      }
    }
    f = next;
  }
  const scale = Math.exp(logFactorial(rest) - rest * Math.log(m));
  let win = 0;
  for (const [key, weight] of f) {
    const [used, tied] = key.split(",").map(Number);
    if (used !== rest) continue;
    win += (weight * scale) / (tied + 1);
  }
  return Math.min(1, win);
}

export function majorityAccuracy(n: number, p: number, wrong: WrongSpread) {
  return binomialPmf(n, p).reduce((total, probability, correct) => total + probability * winGivenCorrect(n, correct, wrong), 0);
}

export function votingCurves(maxSamples: number, p: number, wrong: WrongSpread) {
  return Array.from({ length: maxSamples }, (_, index) => {
    const n = index + 1;
    return { n, single: p, majority: majorityAccuracy(n, p, wrong), verifier: verifierAccuracy(n, p) };
  });
}
