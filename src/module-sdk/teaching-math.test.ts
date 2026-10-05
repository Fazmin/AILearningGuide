import { describe, expect, it } from "vitest";
import {
  backpropSnapshot,
  dot,
  mixVectors,
  numericWeightGradient,
  relativeGradientError,
  scaledDotProductAttention,
} from "./teaching-math";

describe("backpropagation teaching graph", () => {
  it.each([
    [0.2, -1.4, 0.1],
    [0.8, 0.6, 0.9],
    [1.3, 2.2, 0.4],
  ])(
    "matches a central-difference gradient for x=%s, w=%s, y=%s",
    (input, weight, target) => {
      const analytic = backpropSnapshot(input, weight, target).dLossWeight;
      const numeric = numericWeightGradient(input, weight, target);
      expect(relativeGradientError(analytic, numeric)).toBeLessThan(1e-8);
    },
  );

  it("updates the weight downhill for a small learning rate", () => {
    const before = backpropSnapshot(0.8, 0.6, 0.9);
    const nextWeight = before.weight - 0.2 * before.dLossWeight;
    const after = backpropSnapshot(0.8, nextWeight, 0.9);
    expect(after.loss).toBeLessThan(before.loss);
  });
});

describe("scaled dot-product attention", () => {
  const query = [1, 0, 0, 0];
  const keys = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [-1, 0, 0, 0],
  ];

  it("normalizes routes and preserves score order", () => {
    const result = scaledDotProductAttention(query, keys);
    expect(result.weights.reduce((sum, value) => sum + value, 0)).toBeCloseTo(
      1,
      12,
    );
    expect(result.weights[0]).toBeGreaterThan(result.weights[1]);
    expect(result.weights[1]).toBeGreaterThan(result.weights[2]);
  });

  it("recomputes live when a key is edited", () => {
    const before = scaledDotProductAttention(query, keys);
    const edited = scaledDotProductAttention(query, [
      keys[0],
      [3, 1, 0, 0],
      keys[2],
    ]);
    expect(edited.weights[1]).toBeGreaterThan(before.weights[1]);
  });

  it("mixes values with the normalized weights", () => {
    expect(
      mixVectors([0.25, 0.75], [
        [4, 0],
        [0, 8],
      ]),
    ).toEqual([1, 6]);
  });

  it("zeroes future keys when a causal index is set", () => {
    const result = scaledDotProductAttention(query, keys, { causalIndex: 0 });
    expect(result.weights[0]).toBeCloseTo(1, 12);
    expect(result.weights[1]).toBe(0);
    expect(result.weights[2]).toBe(0);
    expect(result.scores[1]).toBe(dot(query, keys[1]));
    expect(result.scaledScores[1]).toBe(Number.NEGATIVE_INFINITY);
  });
});
