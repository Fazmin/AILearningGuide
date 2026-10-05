/** Closed forms behind the six refreshers. Every readout on the page comes from here. */

export type Surface = "convex" | "nonconvex";

export const convexLoss = (w: number) => (w - 0.8) ** 2 + 0.25;
export const convexGrad = (w: number) => 2 * (w - 0.8);
export const nonconvexLoss = (w: number) => (w * w - 1) ** 2 + 0.35 * w + 0.6;
export const nonconvexGrad = (w: number) => 4 * w * (w * w - 1) + 0.35;

export const lossOf = (surface: Surface, w: number) => (surface === "convex" ? convexLoss(w) : nonconvexLoss(w));
export const gradOf = (surface: Surface, w: number) => (surface === "convex" ? convexGrad(w) : nonconvexGrad(w));

/** One gradient-descent step, clipped to the slider range [−2, 2]. */
export const descend = (surface: Surface, w: number, eta: number) =>
  Math.min(2, Math.max(-2, w - eta * gradOf(surface, w)));

/** Roots of L'(w) on [−2, 2], found by sign changes and bisection, labelled by the second derivative. */
export const stationaryPoints = (surface: Surface) => {
  const second = (w: number) => (surface === "convex" ? 2 : 12 * w * w - 4);
  const found: { w: number; loss: number; kind: "minimum" | "maximum" }[] = [];
  const steps = 400;
  for (let i = 0; i < steps; i += 1) {
    let low = -2 + (4 * i) / steps;
    let high = -2 + (4 * (i + 1)) / steps;
    if (gradOf(surface, low) === 0) {
      found.push({ w: low, loss: lossOf(surface, low), kind: second(low) > 0 ? "minimum" : "maximum" });
      continue;
    }
    if (Math.sign(gradOf(surface, low)) === Math.sign(gradOf(surface, high))) continue;
    for (let j = 0; j < 60; j += 1) {
      const mid = (low + high) / 2;
      if (Math.sign(gradOf(surface, mid)) === Math.sign(gradOf(surface, low))) low = mid;
      else high = mid;
    }
    const w = (low + high) / 2;
    found.push({ w, loss: lossOf(surface, w), kind: second(w) > 0 ? "minimum" : "maximum" });
  }
  return found;
};

/** Slope of the chord from x to x + ε on f(x) = x². Algebraically 2x + ε. */
export const secantSlope = (x: number, eps: number) => ((x + eps) ** 2 - x * x) / eps;

export const softmax2 = (a: number, b: number) => {
  const shift = Math.max(a, b);
  const ea = Math.exp(a - shift);
  const eb = Math.exp(b - shift);
  return [ea / (ea + eb), eb / (ea + eb)] as const;
};

/** Bernoulli entropy, cross-entropy, and KL in nats (natural log). */
export const entropy = (p: number) => -p * Math.log(p) - (1 - p) * Math.log(1 - p);
export const crossEntropy = (p: number, q: number) => -p * Math.log(q) - (1 - p) * Math.log(1 - q);
export const klDivergence = (p: number, q: number) => crossEntropy(p, q) - entropy(p);

export const det2 = (m00: number, m01: number, m10: number, m11: number) => m00 * m11 - m01 * m10;

/** The logistic function, with the exponent clipped so a huge logit cannot overflow. */
export const sigmoid = (z: number) => 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, z))));

/** Natural log of the odds p/(1−p): the inverse of the sigmoid. */
export const logit = (p: number) => Math.log(p / (1 - p));

/**
 * The authored count table behind the probability refresher: eight rows, four per class. Rows are x = 0
 * and x = 1, columns are y = 0 and y = 1.
 */
export const COUNTS = { x0: { y0: 3, y1: 1 }, x1: { y0: 1, y1: 3 } } as const;

/** P(x = 1 | y = 1) and P(x = 1 | y = 0), read from COUNTS. These are the likelihoods in Bayes' rule. */
export const LIKELIHOOD_POSITIVE = COUNTS.x1.y1 / (COUNTS.x0.y1 + COUNTS.x1.y1);
export const LIKELIHOOD_NEGATIVE = COUNTS.x1.y0 / (COUNTS.x0.y0 + COUNTS.x1.y0);

/** Bayes' rule for one yes/no hypothesis and one observation: posterior = prior × likelihood / evidence. */
export const bayes = (prior: number, likelihoodPositive: number, likelihoodNegative: number) => {
  const jointPositive = prior * likelihoodPositive;
  const jointNegative = (1 - prior) * likelihoodNegative;
  const evidence = jointPositive + jointNegative;
  const posterior = jointPositive / evidence;
  return {
    jointPositive,
    jointNegative,
    evidence,
    posterior,
    priorOdds: prior / (1 - prior),
    likelihoodRatio: likelihoodPositive / likelihoodNegative,
    posteriorOdds: posterior / (1 - posterior),
  };
};

/** The four authored numbers the Gaussian refresher can fit. */
export const SAMPLE: readonly number[] = [-1, 0, 1, 2];

export const sampleMean = (values: readonly number[]) => values.reduce((sum, value) => sum + value, 0) / values.length;

/** Mean squared deviation from the mean (divide by n), the maximum-likelihood estimate naive Bayes uses. */
export const sampleVariance = (values: readonly number[]) => {
  const mean = sampleMean(values);
  return values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
};

/** Height of the Gaussian density N(mean, variance) at x. It is a density, not a probability. */
export const gaussianPdf = (x: number, mean: number, variance: number) =>
  Math.exp(-((x - mean) ** 2) / (2 * variance)) / Math.sqrt(2 * Math.PI * variance);

/** Area under the density between lo and hi, by Simpson's rule. */
export const gaussianMass = (mean: number, variance: number, lo: number, hi: number, steps = 400) => {
  const h = (hi - lo) / steps;
  let total = gaussianPdf(lo, mean, variance) + gaussianPdf(hi, mean, variance);
  for (let i = 1; i < steps; i += 1) {
    total += gaussianPdf(lo + i * h, mean, variance) * (i % 2 === 0 ? 2 : 4);
  }
  return (total * h) / 3;
};
