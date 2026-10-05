import { describe, expect, it } from "vitest";
import { checkpointQuestions } from "@app/module-sdk";
import {
  DEEP_NARROW_NET,
  forward,
  LARGE_NET,
  NARROW_NET,
  SHALLOW_WIDE_NET,
  snapshotAt,
  train,
  trainRingSplit,
} from "./mlp";
import definition from "./module";

/**
 * Pins the comparisons the checkpoint questions rely on. The lab's own defaults are
 * XOR, tanh, one layer of 2, seed 2, epoch 150.
 */
const questions = checkpointQuestions(definition);
const correctOption = (index: number) => questions[index].options[questions[index].answer];

/** Worst-case miss of the best-fitting plane z ≈ a·x + b·y + c over a grid, by normal equations. */
function planeResidual(sample: (x: number, y: number) => number) {
  const points: Array<[number, number, number]> = [];
  for (let i = 0; i < 15; i += 1) {
    for (let j = 0; j < 15; j += 1) {
      const x = -1 + (2 * i) / 14;
      const y = -1 + (2 * j) / 14;
      points.push([x, y, sample(x, y)]);
    }
  }
  // Solve the 3×3 normal equations with Cramer's rule.
  let sxx = 0, sxy = 0, sx = 0, syy = 0, sy = 0, n = 0, sxz = 0, syz = 0, sz = 0;
  for (const [x, y, z] of points) {
    sxx += x * x; sxy += x * y; sx += x; syy += y * y; sy += y; n += 1; sxz += x * z; syz += y * z; sz += z;
  }
  const det = (m: number[][]) =>
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
    m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
    m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  const A = [[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, n]];
  const d = det(A);
  const column = (index: number, rhs: number[]) => A.map((row, r) => row.map((value, c) => (c === index ? rhs[r] : value)));
  const rhs = [sxz, syz, sz];
  const [a, b, c] = [0, 1, 2].map((index) => det(column(index, rhs)) / d);
  return Math.max(...points.map(([x, y, z]) => Math.abs(z - (a * x + b * y + c))));
}

describe("stacking-neurons checkpoint questions", () => {
  it("question 1: on Ring, two units stay below 90% for seeds 1 to 6 and three reach 100%", () => {
    for (let seed = 1; seed <= 6; seed += 1) {
      const narrow = train({ dataset: "ring", depth: 1, width: 2, activation: "tanh", seed });
      expect(Math.max(...narrow.accuracies)).toBeLessThan(0.9);
      const wide = train({ dataset: "ring", depth: 1, width: 3, activation: "tanh", seed });
      expect(wide.accuracies[300]).toBe(1);
    }
    expect(correctOption(0)).toMatch(/^Accuracy reaches 100%/);
  });

  it("question 2: a first-layer unit's pre-activation is exactly affine in the input, a second-layer unit's is not", () => {
    const history = train({ dataset: "xor", depth: 2, width: 3, activation: "tanh", seed: 2 });
    const network = snapshotAt(history, 150);
    const first = planeResidual((x, y) => forward(network, [x, y], "tanh").pre[0][0]);
    const second = planeResidual((x, y) => forward(network, [x, y], "tanh").pre[1][0]);
    expect(first).toBeLessThan(1e-9);
    expect(second).toBeGreaterThan(0.05);
    expect(correctOption(1)).toMatch(/^First-layer units read x₁ and x₂ directly/);
  });

  it("question 3: at the defaults both label-0 corners land near the same hidden point, far from the label-1 corners", () => {
    const network = snapshotAt(train({ dataset: "xor", depth: 1, width: 2, activation: "tanh", seed: 2 }), 150);
    const hidden = (x: number, y: number) => forward(network, [x, y], "tanh").post[1];
    const a = hidden(0.6, 0.6);
    const b = hidden(-0.6, -0.6);
    const distance = (u: number[], v: number[]) => Math.hypot(u[0] - v[0], u[1] - v[1]);
    expect(distance(a, b)).toBeLessThan(0.1);
    for (const corner of [hidden(-0.6, 0.6), hidden(0.6, -0.6)]) {
      expect(distance(a, corner)).toBeGreaterThan(1.5);
      expect(distance(b, corner)).toBeGreaterThan(1.5);
    }
    expect(correctOption(2)).toMatch(/^Both points land near the same values/);
  });

  it("question 4: with no nonlinearity, 3 layers of 6 stay at 50% accuracy and loss ln 2 for seeds 1 to 6", () => {
    for (let seed = 1; seed <= 6; seed += 1) {
      const run = train({ dataset: "xor", depth: 3, width: 6, activation: "linear", seed });
      expect(run.accuracies[300]).toBe(0.5);
      expect(run.losses[300]).toBeCloseTo(Math.LN2, 3);
    }
    expect(correctOption(3)).toMatch(/0\.693/);
  });

  it("question 5: on the checkerboard with tanh the two-layer shape fits and the one-layer shape of equal size does not", { timeout: 60000 }, () => {
    for (const seed of [1, 2, 3, 4, 5, 6]) {
      const wide = train({ dataset: "checker", ...SHALLOW_WIDE_NET, activation: "tanh", seed });
      const deep = train({ dataset: "checker", ...DEEP_NARROW_NET, activation: "tanh", seed });
      expect(wide.sizes.length).toBe(3);
      expect(deep.sizes.length).toBe(4);
      expect(deep.accuracies[300]).toBeGreaterThanOrEqual(0.8);
      expect(wide.accuracies[300]).toBeLessThanOrEqual(0.6);
    }
    expect(correctOption(4)).toMatch(/^Two layers of 7 fit most of the board/);
  });

  it("question 6: XOR and the ring do not separate the shapes, and ReLU turns the checkerboard around", { timeout: 60000 }, () => {
    for (const dataset of ["xor", "ring"] as const) {
      const wide = train({ dataset, ...SHALLOW_WIDE_NET, activation: "tanh", seed: 2 });
      const deep = train({ dataset, ...DEEP_NARROW_NET, activation: "tanh", seed: 2 });
      expect(wide.accuracies[300]).toBe(1);
      expect(deep.accuracies[300]).toBe(1);
    }
    for (const seed of [1, 3, 4, 5]) {
      const wide = train({ dataset: "checker", ...SHALLOW_WIDE_NET, activation: "relu", seed });
      const deep = train({ dataset: "checker", ...DEEP_NARROW_NET, activation: "relu", seed });
      expect(wide.accuracies[300]).toBeGreaterThan(deep.accuracies[300]);
    }
    expect(correctOption(5)).toMatch(/^With tanh and 300 epochs two layers beat one wide layer/);
  });

  it("question 7: with 20% flipped labels the large network's training accuracy outruns its held-out accuracy", { timeout: 60000 }, () => {
    for (const seed of [1, 2, 3, 4, 5, 6]) {
      const large = trainRingSplit({ ...LARGE_NET, seed, noisy: true });
      const narrow = trainRingSplit({ ...NARROW_NET, seed, noisy: true });
      const gap = (run: typeof large) => run.accuracies[300] - (run.heldAccuracies as number[])[300];
      expect(gap(large)).toBeGreaterThanOrEqual(0.15 - 1e-9);
      expect(gap(narrow)).toBeLessThanOrEqual(0.07);
    }
    expect(correctOption(6)).toMatch(/^It has fit some flipped labels/);
  });

  it("question 8: with clean labels the large network's gap is zero at every seed", { timeout: 60000 }, () => {
    for (const seed of [1, 2, 3, 4, 5, 6]) {
      const large = trainRingSplit({ ...LARGE_NET, seed, noisy: false });
      expect(large.accuracies[300]).toBe(1);
      expect(large.heldAccuracies?.[300]).toBe(1);
    }
    expect(correctOption(7)).toMatch(/^It nearly vanishes/);
  });

  it("the default run still reaches 100% on its training points while its loss is above zero", () => {
    const run = train({ dataset: "xor", depth: 1, width: 2, activation: "tanh", seed: 2 });
    expect(run.accuracies[150]).toBe(1);
    expect(run.losses[150]).toBeGreaterThan(0);
    expect(run.data).toHaveLength(48);
  });
});
