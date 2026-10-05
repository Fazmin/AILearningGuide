import { describe, expect, it } from "vitest";
import {
  activate,
  cornersCorrect,
  footOnBoundary,
  levelChord,
  signedDistance,
  slope,
  weightedSum,
  type CornerTask,
} from "./neuron";

const defaults = { w1: 1.2, w2: -0.8, bias: -0.1 };

describe("single neuron", () => {
  it("computes the weighted sum and the three activations at the default state", () => {
    const z = weightedSum(0.7, 0.35, defaults);
    expect(z).toBeCloseTo(0.46, 10);
    expect(activate(z, "sigmoid")).toBeCloseTo(0.613, 3);
    expect(activate(z, "relu")).toBeCloseTo(0.46, 10);
    expect(activate(z, "step")).toBe(1);
  });

  it("gives three different outputs for z = 0 at x = (0.75, 1)", () => {
    const z = weightedSum(0.75, 1, defaults);
    expect(z).toBeCloseTo(0, 10);
    expect(activate(0, "sigmoid")).toBe(0.5);
    expect(activate(0, "relu")).toBe(0);
    expect(activate(0, "step")).toBe(0);
  });

  it("has the textbook slopes, and none for step or ReLU at the kink", () => {
    expect(slope(0, "sigmoid")).toBeCloseTo(0.25, 10);
    expect(slope(5, "sigmoid")).toBeCloseTo(0.00665, 5);
    expect(slope(2, "relu")).toBe(1);
    expect(slope(-2, "relu")).toBe(0);
    expect(slope(1, "step")).toBe(0);
    expect(slope(0, "step")).toBeNaN();
  });

  it("reaches z = 6 at the corner of the control ranges, where sigmoid saturates", () => {
    const z = weightedSum(1, 1, { w1: 2, w2: 2, bias: 2 });
    expect(z).toBe(6);
    expect(activate(z, "sigmoid")).toBeCloseTo(0.9975, 4);
    expect(slope(z, "sigmoid")).toBeLessThan(0.003);
  });

  it("measures distance to the boundary as z / ‖w‖, with the foot on the line", () => {
    const d = signedDistance(0.7, 0.35, defaults);
    expect(d).toBeCloseTo(0.46 / Math.hypot(1.2, 0.8), 10);
    const foot = footOnBoundary(0.7, 0.35, defaults);
    expect(foot).not.toBeNull();
    expect(weightedSum(foot![0], foot![1], defaults)).toBeCloseTo(0, 10);
  });

  it("scaling w and b together leaves the boundary where it was", () => {
    const line = levelChord(defaults);
    const doubled = levelChord({ w1: 2.4, w2: -1.6, bias: -0.2 });
    expect(doubled).toEqual(line);
    const weightsOnly = levelChord({ w1: 2.4, w2: -1.6, bias: -0.1 });
    expect(weightsOnly).not.toEqual(line);
  });

  it("draws the vertical line x₁ = −b / w₁ when w₂ = 0, and nothing when w = 0", () => {
    const line = levelChord({ w1: 1.2, w2: 0, bias: -0.1 });
    expect(line?.[0][0]).toBeCloseTo(0.1 / 1.2, 10);
    expect(line?.[1][0]).toBeCloseTo(0.1 / 1.2, 10);
    expect(levelChord({ w1: 0, w2: 0, bias: -0.1 })).toBeNull();
  });

  // Brute force over the slider grid: AND and OR are separable, XOR never beats 3 of 4.
  it.each<[CornerTask, number]>([
    ["and", 4],
    ["or", 4],
    ["xor", 3],
  ])("the best setting of the sliders scores %s at %i of 4", (task, best) => {
    let max = 0;
    for (let w1 = -2; w1 <= 2.001; w1 += 0.1) {
      for (let w2 = -2; w2 <= 2.001; w2 += 0.1) {
        for (let bias = -2; bias <= 2.001; bias += 0.1) {
          max = Math.max(max, cornersCorrect(task, { w1, w2, bias }));
        }
      }
    }
    expect(max).toBe(best);
  });
});
