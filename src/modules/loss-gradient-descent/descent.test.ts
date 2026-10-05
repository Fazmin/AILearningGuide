import { describe, expect, it } from "vitest";
import {
  contourEllipse,
  GD_OVERSHOOT_LIMIT,
  GD_STABLE_LIMIT,
  gradient,
  HESSIAN,
  LEAST_SQUARES,
  MAX_STEPS,
  modeBehaviour,
  momentumStableLimit,
  mse,
  regime,
  runDescent,
  stepsToSettle,
} from "./descent";
import definition from "./module";

const START = { w0: -1, b0: 1.2 };

describe("loss-gradient-descent math", () => {
  it("matches a central finite difference for both partial derivatives", () => {
    for (const [w, b] of [
      [-1, 1.2],
      [0.3, -0.7],
      [3, 1.5],
    ]) {
      const h = 1e-6;
      const numericW = (mse(w + h, b) - mse(w - h, b)) / (2 * h);
      const numericB = (mse(w, b + h) - mse(w, b - h)) / (2 * h);
      const { gw, gb } = gradient(w, b);
      expect(gw).toBeCloseTo(numericW, 6);
      expect(gb).toBeCloseTo(numericB, 6);
    }
  });

  it("puts a zero gradient at the least-squares minimum", () => {
    const { gw, gb } = gradient(LEAST_SQUARES.w, LEAST_SQUARES.b);
    expect(Math.abs(gw)).toBeLessThan(1e-12);
    expect(Math.abs(gb)).toBeLessThan(1e-12);
    expect(LEAST_SQUARES.w).toBeCloseTo(1.443, 3);
    expect(LEAST_SQUARES.b).toBeCloseTo(-0.542, 3);
    expect(LEAST_SQUARES.loss).toBeCloseTo(0.0205, 4);
  });

  it("reports the curvatures that set the learning-rate limits", () => {
    expect(HESSIAN.steep).toBeCloseTo(4.363, 3);
    expect(HESSIAN.shallow).toBeCloseTo(0.307, 3);
    expect(GD_STABLE_LIMIT).toBeCloseTo(0.458, 3);
    expect(GD_OVERSHOOT_LIMIT).toBeCloseTo(0.229, 3);
    expect(HESSIAN.condition).toBeGreaterThan(14);
  });

  it("draws contours on which the loss really equals the level", () => {
    for (const level of [0.05, 0.5, 4]) {
      for (const point of contourEllipse(level, 24)) {
        expect(mse(point.w, point.b)).toBeCloseTo(level, 9);
      }
    }
    expect(contourEllipse(LEAST_SQUARES.loss / 2)).toEqual([]);
  });

  it("converges below the stability limit and diverges above it", () => {
    const stable = runDescent({ ...START, learningRate: 0.44, optimizer: "gd", beta: 0 });
    expect(stable.divergedAt).toBeNull();
    expect(stable.points[MAX_STEPS].loss).toBeLessThan(0.025);
    const unstable = runDescent({ ...START, learningRate: 0.5, optimizer: "gd", beta: 0 });
    expect(unstable.divergedAt).not.toBeNull();
    expect(regime(0.44, "gd", 0)).toBe("overshoots");
    expect(regime(0.5, "gd", 0)).toBe("diverges");
    expect(regime(0.2, "gd", 0)).toBe("one-sided");
  });

  it("lowers the loss on every plain step while η is under 1/λ_max", () => {
    const run = runDescent({ ...START, learningRate: 0.2, optimizer: "gd", beta: 0 });
    for (let step = 1; step <= MAX_STEPS; step += 1) {
      expect(run.points[step].loss).toBeLessThanOrEqual(run.points[step - 1].loss + 1e-12);
    }
  });

  it("makes momentum with β = 0 identical to plain descent", () => {
    const plain = runDescent({ ...START, learningRate: 0.1, optimizer: "gd", beta: 0.9 });
    const zero = runDescent({ ...START, learningRate: 0.1, optimizer: "momentum", beta: 0 });
    expect(zero.points.map((point) => point.loss)).toEqual(plain.points.map((point) => point.loss));
  });

  it("gives momentum's first step exactly the plain step", () => {
    const plain = runDescent({ ...START, learningRate: 0.05, optimizer: "gd", beta: 0 });
    const heavy = runDescent({ ...START, learningRate: 0.05, optimizer: "momentum", beta: 0.9 });
    expect(heavy.points[1].w).toBe(plain.points[1].w);
    expect(heavy.points[1].b).toBe(plain.points[1].b);
  });

  it("measures the lesson's quoted comparison: momentum crosses the shallow valley faster", () => {
    const plain = runDescent({ ...START, learningRate: 0.05, optimizer: "gd", beta: 0 });
    const heavy = runDescent({ ...START, learningRate: 0.05, optimizer: "momentum", beta: 0.9 });
    expect(stepsToSettle(plain)).toBeNull();
    expect(stepsToSettle(heavy)).not.toBeNull();
    expect(plain.points[MAX_STEPS].loss).toBeGreaterThan(0.1);
  });

  it("predicts the observed per-step shrink from the curvature", () => {
    const behaviour = modeBehaviour(0.1, HESSIAN.shallow, "gd", 0);
    expect(behaviour.rate).toBeCloseTo(1 - 0.1 * HESSIAN.shallow, 12);
    expect(momentumStableLimit(0.9)).toBeGreaterThan(0.5);
    expect(modeBehaviour(0.3, HESSIAN.steep, "momentum", 0.9).oscillates).toBe(true);
  });
});

describe("loss-gradient-descent state", () => {
  it("migrates the one-weight version 1 payload", () => {
    const old = JSON.stringify({
      weight: -1.2,
      learningRate: 0.18,
      momentum: 0.8,
      velocity: 0.3,
      optimizer: "sgd",
    });
    const state = definition.hydrateState(old);
    expect(state.weight).toBeUndefined();
    expect(state.velocity).toBeUndefined();
    expect(state.optimizer).toBe("gd");
    expect(state.momentum).toBe(0.8);
    expect(state.learningRate).toBe(definition.initialState.learningRate);
    expect(state.startW).toBe(definition.initialState.startW);
  });

  it("keeps a momentum choice from version 1", () => {
    const state = definition.hydrateState(JSON.stringify({ optimizer: "momentum", weight: 0 }));
    expect(state.optimizer).toBe("momentum");
  });
});
