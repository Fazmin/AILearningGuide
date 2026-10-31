import { describe, expect, it } from "vitest";
import { tinyRandom } from "@app/module-sdk";
import { FEATURE_COUNT, TOOL_COUNT } from "./env";
import {
  EDIT_BIAS,
  REFERENCE_ROWS,
  WEIGHT_COUNT,
  addKlGradient,
  addLogProbGradient,
  inputOf,
  klDivergence,
  referenceWeights,
  softmax,
  toolProbabilities,
  type Weights,
} from "./policy";

const randomWeights = (seed: number) => {
  const random = tinyRandom(seed);
  return Float64Array.from({ length: WEIGHT_COUNT }, () => (random() - 0.5) * 4);
};
const randomFeatures = (seed: number) => {
  const random = tinyRandom(seed);
  return Array.from({ length: FEATURE_COUNT }, () => (random() < 0.5 ? 1 : 0));
};
const nudged = (weights: Weights, index: number, delta: number) => {
  const copy = Float64Array.from(weights);
  copy[index] += delta;
  return copy;
};

describe("the linear softmax policy", () => {
  it("has 48 weights: six tools, each with a bias and seven feature weights", () => {
    expect(WEIGHT_COUNT).toBe(48);
    expect(TOOL_COUNT * (1 + FEATURE_COUNT)).toBe(48);
    expect(REFERENCE_ROWS).toHaveLength(TOOL_COUNT);
    for (const row of REFERENCE_ROWS) expect(row).toHaveLength(1 + FEATURE_COUNT);
  });

  it("gives edit_check a constant logit of −2.5 in the reference policy", () => {
    expect(EDIT_BIAS).toBe(-2.5);
    expect(REFERENCE_ROWS[4]).toEqual([-2.5, 0, 0, 0, 0, 0, 0, 0]);
  });

  it("builds a softmax that sums to one and never overflows", () => {
    const extreme = softmax([1000, -1000, 0, 5, 40, -40]);
    expect(extreme.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 12);
    expect(extreme.every((value) => Number.isFinite(value))).toBe(true);
    const policy = toolProbabilities(referenceWeights(), [0, 0, 0, 0, 0, 0, 0]);
    expect(policy.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 12);
  });
});

describe("gradients checked against finite differences", () => {
  it("computes ∇log π(tool | features) exactly", () => {
    for (const seed of [1, 2, 3]) {
      const weights = randomWeights(seed);
      const values = randomFeatures(seed + 10);
      const input = inputOf(values);
      for (let tool = 0; tool < TOOL_COUNT; tool += 1) {
        const analytic = new Float64Array(WEIGHT_COUNT);
        addLogProbGradient(analytic, toolProbabilities(weights, values), input, tool, 1);
        for (let index = 0; index < WEIGHT_COUNT; index += 1) {
          const h = 1e-6;
          const up = Math.log(toolProbabilities(nudged(weights, index, h), values)[tool]);
          const down = Math.log(toolProbabilities(nudged(weights, index, -h), values)[tool]);
          expect(analytic[index]).toBeCloseTo((up - down) / (2 * h), 6);
        }
      }
    }
  });

  it("computes ∇KL(π ‖ π_ref) exactly at a state", () => {
    const reference = referenceWeights();
    for (const seed of [4, 5, 6]) {
      const weights = randomWeights(seed);
      const values = randomFeatures(seed + 20);
      const referenceProbabilities = toolProbabilities(reference, values);
      const analytic = new Float64Array(WEIGHT_COUNT);
      addKlGradient(analytic, toolProbabilities(weights, values), referenceProbabilities, inputOf(values), 1);
      const kl = (candidate: Weights) => klDivergence(toolProbabilities(candidate, values), referenceProbabilities);
      for (let index = 0; index < WEIGHT_COUNT; index += 1) {
        const h = 1e-6;
        expect(analytic[index]).toBeCloseTo((kl(nudged(weights, index, h)) - kl(nudged(weights, index, -h))) / (2 * h), 6);
      }
    }
  });

  it("gives a KL of zero with a zero gradient at the reference, and a positive KL elsewhere", () => {
    const reference = referenceWeights();
    for (const seed of [7, 8, 9]) {
      const values = randomFeatures(seed);
      const probabilities = toolProbabilities(reference, values);
      expect(klDivergence(probabilities, probabilities)).toBe(0);
      const gradient = new Float64Array(WEIGHT_COUNT);
      addKlGradient(gradient, probabilities, probabilities, inputOf(values), 1);
      for (const value of gradient) expect(Math.abs(value)).toBeLessThan(1e-15);
      expect(klDivergence(toolProbabilities(randomWeights(seed), values), probabilities)).toBeGreaterThan(0);
    }
  });

  it("scales both gradients linearly, so the sign of an advantage or of −β flips the step", () => {
    const weights = randomWeights(11);
    const values = randomFeatures(12);
    const probabilities = toolProbabilities(weights, values);
    const once = new Float64Array(WEIGHT_COUNT);
    const negative = new Float64Array(WEIGHT_COUNT);
    addLogProbGradient(once, probabilities, inputOf(values), 2, 1);
    addLogProbGradient(negative, probabilities, inputOf(values), 2, -2);
    once.forEach((value, index) => expect(negative[index]).toBeCloseTo(-2 * value, 12));
  });
});
