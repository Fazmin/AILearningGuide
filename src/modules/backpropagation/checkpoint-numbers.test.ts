import { describe, expect, it } from "vitest";
import { backpropSnapshot, checkpointQuestions } from "@app/module-sdk";
import { FLOW_INITS, measureGradientFlow } from "./gradient-flow";
import definition from "./module";

/**
 * Pins the comparisons the checkpoint questions rely on. Defaults: x 0.8, w 0.6, target 0.9.
 */
const questions = checkpointQuestions(definition);
const correctOption = (index: number) => questions[index].options[questions[index].answer];

describe("backpropagation checkpoint questions", () => {
  it("question 1: at x 1.5, w 3, y 0.1 the loss is large and the sigmoid factor alone is tiny", () => {
    const saturated = backpropSnapshot(1.5, 3, 0.1);
    const defaults = backpropSnapshot(0.8, 0.6, 0.9);
    // The loss is several times the default loss, yet the gradient on w is smaller than at the default.
    expect(saturated.loss).toBeGreaterThan(5 * defaults.loss);
    expect(Math.abs(saturated.dLossWeight)).toBeLessThan(Math.abs(defaults.dLossWeight));
    // The loss node's factor is large (the prediction is far from the target), not small.
    expect(Math.abs(saturated.dLossPrediction)).toBeGreaterThan(1);
    expect(Math.abs(saturated.dLossPrediction)).toBeGreaterThan(Math.abs(defaults.dLossPrediction));
    // The multiply node's factor is x = 1.5, greater than one.
    expect(saturated.dProductWeight).toBe(1.5);
    // The sigmoid's slope is far below its peak of 0.25.
    expect(saturated.dPredictionProduct).toBeLessThan(0.02);
    expect(saturated.dPredictionProduct).toBeLessThan(0.25 / 10);
    // And the product of the three factors is the gradient the lab prints.
    expect(saturated.dLossPrediction * saturated.dPredictionProduct * saturated.dProductWeight).toBe(saturated.dLossWeight);
    expect(correctOption(0)).toMatch(/^The sigmoid's ŷ\(1 − ŷ\)/);
  });

  it("question 2: the Product → weight factor is x, and the factor toward x is w", () => {
    for (const [x, w] of [
      [0.8, 0.6],
      [1.1, -0.7],
      [0.1, 3],
    ]) {
      const snapshot = backpropSnapshot(x, w, 0.35);
      expect(snapshot.dProductWeight).toBe(x);
      expect(snapshot.dProductInput).toBe(w);
      expect(snapshot.dLossWeight).toBe(snapshot.dLossProduct * x);
    }
    expect(correctOption(1)).toMatch(/^x, the input value from the forward pass/);
  });

  it("question 4: the branch toward x is a real nonzero derivative that the chain rule continues through", () => {
    const snapshot = backpropSnapshot(0.8, 0.6, 0.9);
    expect(snapshot.dLossInput).not.toBe(0);
    expect(snapshot.dLossInput).toBe(snapshot.dLossProduct * snapshot.dProductInput);
    expect(correctOption(3)).toMatch(/^In a deeper network that same branch/);
  });

  it("question 5: at depth 12 with sigmoid and Glorot weights Layer 1's norm is millions of times smaller than Layer 12's", () => {
    for (const seed of [1, 2, 3, 4, 5, 6, 7]) {
      const result = measureGradientFlow({ depth: 12, activation: "sigmoid", init: "glorot", skip: false, seed });
      expect(result.firstOverLast).toBeLessThan(3e-7);
      expect(result.firstOverLast).toBeGreaterThan(9e-8);
    }
    expect(correctOption(4)).toMatch(/^It is many orders of magnitude smaller/);
  });

  it("question 6: a skip connection restores Layer 1's gradient but raises the loss at the start", () => {
    const plain = measureGradientFlow({ depth: 12, activation: "sigmoid", init: "glorot", skip: false });
    const skipped = measureGradientFlow({ depth: 12, activation: "sigmoid", init: "glorot", skip: true });
    expect(skipped.layers[0].gradNorm).toBeGreaterThan(1e6 * plain.layers[0].gradNorm);
    expect(skipped.firstOverLast).toBeGreaterThan(1e-2);
    expect(plain.loss.toFixed(2)).toBe("0.90");
    expect(skipped.loss.toFixed(2)).toBe("7.30");
    expect(Number(plain.layers[0].gradNorm.toExponential(1))).toBe(2.6e-8);
    expect(skipped.layers[0].gradNorm.toFixed(2)).toBe("0.24");
    expect(plain.firstOverLast.toExponential(1)).toBe("1.2e-7");
    expect(correctOption(5)).toMatch(/^Layer 1's norm rises by orders of magnitude/);
  });

  it("question 7: with ReLU the starting scale moves every layer's norm together, in the order small, Glorot, He, large", () => {
    const quoted = FLOW_INITS.map((init) =>
      Number(measureGradientFlow({ depth: 12, activation: "relu", init, skip: false }).layers[0].gradNorm.toExponential(1)),
    );
    expect(quoted).toEqual([2.4e-7, 2e-2, 1.3, 3.5e5]);
    for (const seed of [1, 2, 3, 4, 5, 6, 7]) {
      const norms = FLOW_INITS.map(
        (init) => measureGradientFlow({ depth: 12, activation: "relu", init, skip: false, seed }).layers[0].gradNorm,
      );
      expect([...norms].sort((a, b) => a - b)).toEqual(norms);
    }
    expect(correctOption(6)).toMatch(/^It rises at every step/);
  });
});
