import { describe, expect, it } from "vitest";
import { checkpointQuestions } from "@app/module-sdk";
import {
  GD_OVERSHOOT_LIMIT,
  GD_STABLE_LIMIT,
  gradient,
  HESSIAN,
  LEAST_SQUARES,
  modeBehaviour,
  mse,
  regime,
  runDescent,
  stepsToSettle,
} from "./descent";
import definition from "./module";

/**
 * Pins the figures and comparisons the checkpoint questions rely on. The lab's default
 * start is (w, b) = (−1, 1.2) with learning rate 0.1.
 */
const START = { w0: -1, b0: 1.2 };
const questions = checkpointQuestions(definition);

describe("loss-gradient-descent checkpoint questions", () => {
  it("question 2: both partials are negative at the start, one step raises both w and b, and the step does not aim at the minimum", () => {
    const { gw, gb } = gradient(START.w0, START.b0);
    expect(gw).toBeLessThan(0);
    expect(gb).toBeLessThan(0);
    const run = runDescent({ ...START, learningRate: 0.1, optimizer: "gd", beta: 0, steps: 1 });
    const [first, next] = run.points;
    expect(next.w).toBeGreaterThan(first.w);
    expect(next.b).toBeGreaterThan(first.b);
    expect(next.loss).toBeLessThan(first.loss);
    // The step is −η∇L exactly; the minimum lies in a visibly different direction.
    expect(next.w - first.w).toBeCloseTo(-0.1 * gw, 12);
    expect(next.b - first.b).toBeCloseTo(-0.1 * gb, 12);
    const toMin = [LEAST_SQUARES.w - START.w0, LEAST_SQUARES.b - START.b0];
    const cosine = (-gw * toMin[0] + -gb * toMin[1]) / (Math.hypot(gw, gb) * Math.hypot(toMin[0], toMin[1]));
    expect(cosine).toBeLessThan(0.7);
    expect(Math.hypot(next.w - LEAST_SQUARES.w, next.b - LEAST_SQUARES.b)).toBeGreaterThan(1);
    expect(questions[1].options[questions[1].answer]).toMatch(/^Both w and b increase/);
  });

  it("question 3: doubling the learning rate keeps the first step's direction and doubles its length", () => {
    const small = runDescent({ ...START, learningRate: 0.1, optimizer: "gd", beta: 0, steps: 1 }).points;
    const large = runDescent({ ...START, learningRate: 0.2, optimizer: "gd", beta: 0, steps: 1 }).points;
    const dw = small[1].w - small[0].w;
    const db = small[1].b - small[0].b;
    expect(large[1].w - large[0].w).toBeCloseTo(2 * dw, 12);
    expect(large[1].b - large[0].b).toBeCloseTo(2 * db, 12);
    expect(large[1].moved / small[1].moved).toBeCloseTo(2, 12);
  });

  it("question 4: at 0.48 the steep-axis factor is −1.09 and the run diverges, while any rate under 2/λ_max settles", () => {
    expect(1 - 0.48 * HESSIAN.steep).toBeCloseTo(-1.094, 3);
    expect(modeBehaviour(0.48, HESSIAN.steep, "gd", 0).rate).toBeGreaterThan(1.09);
    expect(modeBehaviour(0.48, HESSIAN.steep, "gd", 0).rate).toBeLessThan(1.1);
    expect(0.48).toBeGreaterThan(GD_STABLE_LIMIT);
    expect(GD_STABLE_LIMIT).toBeCloseTo(0.458, 3);
    expect(regime(0.48, "gd", 0)).toBe("diverges");
    const unstable = runDescent({ ...START, learningRate: 0.48, optimizer: "gd", beta: 0 });
    expect(unstable.divergedAt).not.toBeNull();
    // Starting nearer the minimum only delays it: the same rate still grows the loss every few steps.
    const near = runDescent({
      w0: LEAST_SQUARES.w + 0.05,
      b0: LEAST_SQUARES.b + 0.05,
      learningRate: 0.48,
      optimizer: "gd",
      beta: 0,
    });
    expect(near.points[80].loss).toBeGreaterThan(100 * near.points[0].loss);
    expect(near.points[80].loss).toBeGreaterThan(near.points[40].loss);
    // The zigzag regime sits between 1/λ_max and 2/λ_max and shrinks.
    expect(regime(0.4, "gd", 0)).toBe("overshoots");
    expect(0.4).toBeGreaterThan(GD_OVERSHOOT_LIMIT);
    expect(0.4).toBeLessThan(GD_STABLE_LIMIT);
    expect(questions[3].options[questions[3].answer]).toMatch(/^Lower the learning rate below 2 divided by the largest curvature/);
  });

  it("question 5: at 0.20 the steep axis has already collapsed while the shallow axis shrinks by only about 6 percent a step", () => {
    const steep = modeBehaviour(0.2, HESSIAN.steep, "gd", 0).rate;
    const shallow = modeBehaviour(0.2, HESSIAN.shallow, "gd", 0).rate;
    expect(steep).toBeLessThan(0.2);
    expect(shallow).toBeGreaterThan(0.9);
    expect(shallow).toBeLessThan(0.95);
    const run = runDescent({ ...START, learningRate: 0.2, optimizer: "gd", beta: 0 });
    expect(run.divergedAt).toBeNull();
    const settles = stepsToSettle(run);
    expect(settles).not.toBeNull();
    expect(settles!).toBeGreaterThanOrEqual(20);
    expect(settles!).toBeLessThan(80);
    // The bowl is a perfect quadratic with a single minimum: no saddle, and the gradient is nonzero away from it.
    expect(HESSIAN.shallow).toBeGreaterThan(0);
    const g = gradient(LEAST_SQUARES.w + 0.2, LEAST_SQUARES.b);
    expect(Math.hypot(g.gw, g.gb)).toBeGreaterThan(0);
    expect(mse(LEAST_SQUARES.w + 0.2, LEAST_SQUARES.b)).toBeGreaterThan(LEAST_SQUARES.loss);
  });
});
