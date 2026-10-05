import { describe, expect, it } from "vitest";
import { decodeState, encodeState, mse, sigmoid, softmax } from "./math";

describe("softmax", () => {
  it("returns a normalized, order-preserving distribution", () => {
    const distribution = softmax([-1, 0, 2]);
    expect(distribution.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 12);
    expect(distribution[2]).toBeGreaterThan(distribution[1]);
    expect(distribution[1]).toBeGreaterThan(distribution[0]);
  });

  it("becomes flatter as temperature rises", () => {
    const cold = softmax([0, 1, 2], 0.2);
    const warm = softmax([0, 1, 2], 1.8);
    expect(cold[2]).toBeGreaterThan(warm[2]);
  });
});

describe("backpropagation math", () => {
  it("matches a numeric gradient for the single-weight graph", () => {
    const input = 0.8;
    const target = 0.9;
    const weight = 0.6;
    const prediction = sigmoid(input * weight);
    const analytic = 2 * (prediction - target) * prediction * (1 - prediction) * input;
    const lossAt = (value: number) => mse(sigmoid(input * value), target);
    const epsilon = 1e-5;
    const numeric = (lossAt(weight + epsilon) - lossAt(weight - epsilon)) / (2 * epsilon);
    expect(analytic).toBeCloseTo(numeric, 6);
  });
});

describe("shared state encoding", () => {
  it("round-trips Unicode module state", () => {
    const state = { prompt: "Why does attention use √d?", head: 3 };
    expect(decodeState(encodeState(state))).toEqual(state);
  });
});
