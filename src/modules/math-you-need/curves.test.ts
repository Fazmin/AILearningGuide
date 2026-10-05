import { describe, expect, it } from "vitest";
import {
  COUNTS,
  LIKELIHOOD_NEGATIVE,
  LIKELIHOOD_POSITIVE,
  SAMPLE,
  bayes,
  convexGrad,
  convexLoss,
  crossEntropy,
  descend,
  det2,
  entropy,
  gaussianMass,
  gaussianPdf,
  klDivergence,
  logit,
  nonconvexGrad,
  nonconvexLoss,
  sampleMean,
  sampleVariance,
  secantSlope,
  sigmoid,
  softmax2,
  stationaryPoints,
} from "./curves";

const numericSlope = (f: (w: number) => number, w: number) => (f(w + 1e-6) - f(w - 1e-6)) / 2e-6;

describe("loss surfaces", () => {
  it("prints the exact derivative of each surface", () => {
    for (const w of [-1.7, -0.4, 0.3, 1.2]) {
      expect(convexGrad(w)).toBeCloseTo(numericSlope(convexLoss, w), 5);
      expect(nonconvexGrad(w)).toBeCloseTo(numericSlope(nonconvexLoss, w), 5);
    }
  });

  it("finds one minimum on the bowl and a deeper left well plus a shallower right well on the tilted curve", () => {
    const bowl = stationaryPoints("convex");
    expect(bowl).toHaveLength(1);
    expect(bowl[0].w).toBeCloseTo(0.8, 9);
    const wells = stationaryPoints("nonconvex");
    const minima = wells.filter((point) => point.kind === "minimum");
    expect(minima).toHaveLength(2);
    expect(wells.filter((point) => point.kind === "maximum")).toHaveLength(1);
    const [left, right] = minima;
    expect(left.w).toBeLessThan(-1);
    expect(right.w).toBeGreaterThan(0.9);
    expect(left.loss).toBeLessThan(right.loss);
  });

  it("settles in the shallower well from w = 1.4 and in the deeper well from w = −1.4", () => {
    let fromRight = 1.4;
    let fromLeft = -1.4;
    for (let i = 0; i < 200; i += 1) {
      fromRight = descend("nonconvex", fromRight, 0.05);
      fromLeft = descend("nonconvex", fromLeft, 0.05);
    }
    expect(fromRight).toBeGreaterThan(0.9);
    expect(fromLeft).toBeLessThan(-1);
  });

  it("overshoots but still converges on the bowl when η is large", () => {
    let w = -1.4;
    const first = descend("convex", w, 0.8);
    expect(first).toBeGreaterThan(0.8);
    for (let i = 0; i < 60; i += 1) w = descend("convex", w, 0.8);
    expect(w).toBeCloseTo(0.8, 6);
  });
});

describe("derivatives, probabilities, and logs", () => {
  it("gives a secant slope of exactly 2x + ε on x²", () => {
    expect(secantSlope(1.2, 0.4)).toBeCloseTo(2.8, 12);
    expect(secantSlope(-0.5, 0.05)).toBeCloseTo(-0.95, 12);
  });

  it("turns two logits into a distribution", () => {
    const [a, b] = softmax2(2.4, 1.1);
    expect(a + b).toBeCloseTo(1, 12);
    expect(a).toBeCloseTo(1 / (1 + Math.exp(-1.3)), 12);
  });

  it("makes KL zero only when q = p, and positive otherwise", () => {
    expect(klDivergence(0.25, 0.25)).toBeCloseTo(0, 12);
    expect(klDivergence(0.25, 0.6)).toBeGreaterThan(0);
    expect(crossEntropy(0.25, 0.6)).toBeGreaterThan(entropy(0.25));
    expect(entropy(0.5)).toBeCloseTo(Math.LN2, 12);
  });

  it("scales area by |det W|", () => {
    expect(det2(1, 0.4, 0, 1)).toBeCloseTo(1, 12);
    expect(det2(2, 0, 0, 0.5)).toBeCloseTo(1, 12);
    expect(det2(1, 2, 0.5, 1)).toBeCloseTo(0, 12);
  });
});

describe("the step size the widget starts with", () => {
  it("settles in the shallower well from w = 1.4 and the deeper well from w = −1.4 at η = 0.15", () => {
    let fromRight = 1.4;
    let fromLeft = -1.4;
    for (let i = 0; i < 300; i += 1) {
      fromRight = descend("nonconvex", fromRight, 0.15);
      fromLeft = descend("nonconvex", fromLeft, 0.15);
    }
    expect(fromRight).toBeCloseTo(0.953, 3);
    expect(nonconvexLoss(fromRight)).toBeCloseTo(0.942, 3);
    expect(fromLeft).toBeCloseTo(-1.041, 3);
    expect(nonconvexLoss(fromLeft)).toBeCloseTo(0.243, 3);
    expect(nonconvexLoss(fromRight)).toBeGreaterThan(nonconvexLoss(fromLeft));
  });
});

describe("Bayes' rule on the count table", () => {
  it("reads the likelihoods from the counts", () => {
    expect(COUNTS).toEqual({ x0: { y0: 3, y1: 1 }, x1: { y0: 1, y1: 3 } });
    expect(LIKELIHOOD_POSITIVE).toBe(0.75);
    expect(LIKELIHOOD_NEGATIVE).toBe(0.25);
  });

  it("returns the table's own 3/4 at the table's own prior of 0.5", () => {
    const result = bayes(0.5, LIKELIHOOD_POSITIVE, LIKELIHOOD_NEGATIVE);
    expect(result.jointPositive).toBeCloseTo(0.375, 12);
    expect(result.jointNegative).toBeCloseTo(0.125, 12);
    expect(result.evidence).toBeCloseTo(0.5, 12);
    expect(result.posterior).toBeCloseTo(COUNTS.x1.y1 / (COUNTS.x1.y0 + COUNTS.x1.y1), 12);
    expect(result.posterior).toBeCloseTo(0.75, 12);
  });

  it("falls to 0.25 at a prior of 0.10 and obeys posterior odds = prior odds × likelihood ratio", () => {
    const rare = bayes(0.1, LIKELIHOOD_POSITIVE, LIKELIHOOD_NEGATIVE);
    expect(rare.posterior).toBeCloseTo(0.25, 12);
    expect(rare.evidence).toBeCloseTo(0.3, 12);
    expect(rare.priorOdds).toBeCloseTo(1 / 9, 12);
    expect(rare.posteriorOdds).toBeCloseTo(1 / 3, 12);
    for (const prior of [0.02, 0.1, 0.5, 0.9, 0.98]) {
      const result = bayes(prior, LIKELIHOOD_POSITIVE, LIKELIHOOD_NEGATIVE);
      expect(result.likelihoodRatio).toBeCloseTo(3, 12);
      expect(result.posteriorOdds).toBeCloseTo(result.priorOdds * result.likelihoodRatio, 9);
    }
  });

  it("rises monotonically with the prior", () => {
    let last = 0;
    for (let prior = 0.02; prior <= 0.98; prior += 0.04) {
      const { posterior } = bayes(prior, LIKELIHOOD_POSITIVE, LIKELIHOOD_NEGATIVE);
      expect(posterior).toBeGreaterThan(last);
      last = posterior;
    }
  });
});

describe("mean and variance of a Gaussian", () => {
  it("fits the authored sample to mean 0.50 and variance 1.25 by dividing by n", () => {
    expect(SAMPLE).toEqual([-1, 0, 1, 2]);
    expect(sampleMean(SAMPLE)).toBe(0.5);
    expect(sampleVariance(SAMPLE)).toBe(1.25);
    expect(Math.sqrt(sampleVariance(SAMPLE))).toBeCloseTo(1.118, 3);
  });

  it("gives a standard-normal height of 0.2420 at x = 1", () => {
    expect(gaussianPdf(1, 0, 1)).toBeCloseTo(0.24197, 5);
    expect(gaussianPdf(0, 0, 1)).toBeCloseTo(1 / Math.sqrt(2 * Math.PI), 12);
  });

  it("lets a density exceed 1 when the variance is small, while its area stays 1", () => {
    expect(gaussianPdf(0, 0, 0.1)).toBeCloseTo(1.2616, 4);
    expect(gaussianPdf(0, 0, 0.1)).toBeGreaterThan(gaussianPdf(0, 0, 2));
    for (const variance of [0.1, 1, 2]) {
      const sigma = Math.sqrt(variance);
      expect(gaussianMass(0.5, variance, 0.5 - 8 * sigma, 0.5 + 8 * sigma)).toBeCloseTo(1, 6);
    }
  });

  it("puts 68.3% of the area within one standard deviation for every mean and variance", () => {
    for (const [mean, variance] of [
      [0, 1],
      [0.5, 1.25],
      [-2, 0.1],
      [2, 2],
    ]) {
      const sigma = Math.sqrt(variance);
      expect(gaussianMass(mean, variance, mean - sigma, mean + sigma)).toBeCloseTo(0.6827, 4);
    }
  });
});

describe("from logit to probability", () => {
  it("makes two-class softmax the sigmoid of the logit gap", () => {
    for (const [a, b] of [
      [2.4, 1.1],
      [0.1, 6],
      [5.5, 5.4],
    ]) {
      expect(softmax2(a, b)[0]).toBeCloseTo(sigmoid(a - b), 12);
    }
    expect(softmax2(2.4, 1.1)[0]).toBeCloseTo(0.7858, 4);
  });

  it("is unchanged when both logits shift together", () => {
    expect(softmax2(4.4, 3.1)[0]).toBeCloseTo(softmax2(2.4, 1.1)[0], 12);
  });

  it("has logit as its inverse and sits at 0.5 for z = 0", () => {
    expect(sigmoid(0)).toBe(0.5);
    for (const p of [0.02, 0.3, 0.5, 0.9]) expect(sigmoid(logit(p))).toBeCloseTo(p, 12);
    expect(logit(softmax2(2.4, 1.1)[0])).toBeCloseTo(1.3, 12);
  });

  it("does not overflow for extreme logits", () => {
    expect(sigmoid(1e6)).toBeLessThanOrEqual(1);
    expect(sigmoid(-1e6)).toBeGreaterThanOrEqual(0);
  });
});
