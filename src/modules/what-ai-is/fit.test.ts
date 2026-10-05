import { describe, expect, it } from "vitest";
import { contour, downhill, gradient, leastSquares, mse } from "./fit";

describe("three-point line fit", () => {
  it("reads L = 0.043 at the default line ŷ = x + 0.9", () => {
    expect(mse(1, 0.9)).toBeCloseTo(0.13 / 3, 12);
  });

  it("prints the analytic gradient of that same MSE", () => {
    for (const [w, b] of [
      [0, 0],
      [1, 0.9],
      [1.7, 0.3],
    ]) {
      const g = gradient(w, b);
      expect(g.w).toBeCloseTo((mse(w + 1e-6, b) - mse(w - 1e-6, b)) / 2e-6, 5);
      expect(g.b).toBeCloseTo((mse(w, b + 1e-6) - mse(w, b - 1e-6)) / 2e-6, 5);
    }
  });

  it("puts the bottom of the bowl at w = 0.9, b ≈ 0.883, inside the slider box", () => {
    const best = leastSquares();
    expect(best.w).toBeCloseTo(0.9, 12);
    expect(best.b).toBeCloseTo(2.2333333333 - 1.35, 8);
    expect(best.loss).toBeCloseTo(0.0266666667 / 3, 8);
    const g = gradient(best.w, best.b);
    expect(Math.hypot(g.w, g.b)).toBeLessThan(1e-9);
  });

  it("walks downhill from (0, 0): L falls every step and ∂L/∂w moves toward zero", () => {
    let point = { w: 0, b: 0 };
    let loss = mse(point.w, point.b);
    const firstSlope = Math.abs(gradient(0, 0).w);
    for (let step = 0; step < 6; step += 1) {
      point = downhill(point.w, point.b);
      const next = mse(point.w, point.b);
      expect(next).toBeLessThan(loss);
      loss = next;
    }
    expect(Math.abs(gradient(point.w, point.b).w)).toBeLessThan(firstSlope);
  });

  it("samples every contour at exactly its level", () => {
    for (const level of [0.05, 0.5, 2]) {
      for (const point of contour(level, 24)) expect(mse(point.w, point.b)).toBeCloseTo(level, 9);
    }
    expect(contour(0.001)).toHaveLength(0);
  });
});
