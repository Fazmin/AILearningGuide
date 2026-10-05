import { describe, expect, it } from "vitest";
import { backpropSnapshot, numericWeightGradient, relativeGradientError } from "@app/module-sdk";

const central = (f: (h: number) => number, h = 1e-6) => (f(h) - f(-h)) / (2 * h);

describe("backpropagation lab numbers", () => {
  it("reproduces the default trace the lesson quotes", () => {
    const s = backpropSnapshot(0.8, 0.6, 0.9);
    expect(s.product).toBeCloseTo(0.48, 12);
    expect(s.prediction).toBeCloseTo(0.61775, 5);
    expect(s.loss).toBeCloseTo(0.07967, 5);
    expect(s.dLossPrediction).toBeCloseTo(-0.5645, 4);
    expect(s.dPredictionProduct).toBeCloseTo(0.23614, 5);
    expect(s.dLossProduct).toBeCloseTo(-0.1333, 4);
    expect(s.dLossWeight).toBeCloseTo(-0.10664, 5);
    expect(s.dLossInput).toBeCloseTo(-0.07998, 5);
  });

  it("multiplies exactly the three factors the chain-rule row prints", () => {
    const s = backpropSnapshot(1.1, -0.7, 0.35);
    expect(s.dLossPrediction * s.dPredictionProduct * s.dProductWeight).toBe(s.dLossWeight);
    expect(s.dLossProduct * s.dProductInput).toBe(s.dLossInput);
  });

  it("agrees with finite differences for both the weight and the input", () => {
    for (const x of [0.1, 0.8, 1.5]) {
      for (const w of [-3, -0.6, 0.6, 3]) {
        for (const y of [0, 0.3, 0.9, 1]) {
          const s = backpropSnapshot(x, w, y);
          const numericW = central((h) => backpropSnapshot(x, w + h, y).loss);
          const numericX = central((h) => backpropSnapshot(x + h, w, y).loss);
          expect(Math.abs(s.dLossWeight - numericW)).toBeLessThan(1e-8);
          expect(Math.abs(s.dLossInput - numericX)).toBeLessThan(1e-8);
        }
      }
    }
    const defaults = backpropSnapshot(0.8, 0.6, 0.9);
    expect(relativeGradientError(defaults.dLossWeight, numericWeightGradient(0.8, 0.6, 0.9))).toBeLessThan(1e-9);
  });

  it("shows saturation: a larger loss with a ten times smaller gradient", () => {
    const saturated = backpropSnapshot(1.5, 3, 0.1);
    const centred = backpropSnapshot(1.5, 0, 0.1);
    expect(saturated.loss).toBeCloseTo(0.790, 3);
    expect(saturated.dPredictionProduct).toBeCloseTo(0.0109, 4);
    expect(saturated.dLossWeight).toBeCloseTo(0.029, 3);
    expect(centred.loss).toBeCloseTo(0.16, 6);
    expect(centred.dLossWeight).toBeCloseTo(0.3, 6);
    expect(centred.dLossWeight / saturated.dLossWeight).toBeGreaterThan(10);
  });

  it("lowers the loss with one update at the default learning rate", () => {
    const before = backpropSnapshot(0.8, 0.6, 0.9);
    const next = 0.6 - 0.7 * before.dLossWeight;
    expect(next).toBeCloseTo(0.6746, 4);
    expect(backpropSnapshot(0.8, next, 0.9).loss).toBeCloseTo(0.07196, 5);
  });
});
