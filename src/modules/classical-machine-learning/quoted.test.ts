import { describe, expect, it } from "vitest";
import {
  POINTS,
  SVM_C,
  bestSplit,
  buildTree,
  classCorrelation,
  fitNaiveBayes,
  fitSvm,
  gini,
  marginWidthOf,
  naiveFactors,
  naivePosterior,
  score,
  scoreSplit,
  supportVectors,
} from "./fit";

/** Pins the numbers the lesson's three sub-mechanism sections and the probe readouts quote. */

describe("the decision-tree split criterion", () => {
  it("scores the whole set at Gini 4/9 and picks the cut x < 0.15 with gain 1/9", () => {
    expect(gini(POINTS)).toBeCloseTo(4 / 9, 12);
    const best = bestSplit(POINTS);
    expect(best.axis).toBe("x");
    expect(best.threshold).toBeCloseTo(0.15, 12);
    const detail = scoreSplit(POINTS, best.axis, best.threshold);
    expect(detail).toMatchObject({ leftCount: 2, rightCount: 16, leftPositives: 2, rightPositives: 4 });
    expect(detail.parent).toBeCloseTo(0.444, 3);
    expect(detail.leftGini).toBe(0);
    expect(detail.rightGini).toBeCloseTo(0.375, 12);
    expect(detail.weighted).toBeCloseTo(1 / 3, 12);
    expect(detail.gain).toBeCloseTo(1 / 9, 12);
    expect(detail.gain).toBeCloseTo(0.111, 3);
  });

  it("ties the root cut with y < 0.86, which peels off the top two class-1 points, and keeps the first", () => {
    const tie = scoreSplit(POINTS, "y", 0.86);
    expect(tie).toMatchObject({ leftCount: 16, rightCount: 2, rightPositives: 2 });
    expect(tie.gain).toBeCloseTo(1 / 9, 12);
    expect(buildTree(POINTS, 1)).toMatchObject({ kind: "split", axis: "x" });
  });

  it("finds no single cut with a larger gain than the one it keeps", () => {
    let largest = 0;
    for (const axis of ["x", "y"] as const) {
      const values = [...new Set(POINTS.map((point) => point[axis]))].sort((a, b) => a - b);
      for (let index = 0; index < values.length - 1; index += 1) {
        largest = Math.max(largest, scoreSplit(POINTS, axis, (values[index] + values[index + 1]) / 2).gain);
      }
    }
    expect(largest).toBeCloseTo(1 / 9, 12);
  });
});

describe("the SVM margin", () => {
  const weights = fitSvm(POINTS);
  const supports = supportVectors(weights, POINTS);
  const supportIndexes = supports.map((point) => POINTS.indexOf(point));
  const otherIndexes = POINTS.map((_, index) => index).filter((index) => !supportIndexes.includes(index));

  it("fits w = (−2.41, 2.80), b = −1.23, so the street is 2/‖w‖ = 0.54 wide", () => {
    expect(weights.w1).toBeCloseTo(-2.41, 2);
    expect(weights.w2).toBeCloseTo(2.8, 2);
    expect(weights.w0).toBeCloseTo(-1.23, 2);
    expect(Math.hypot(weights.w1, weights.w2)).toBeCloseTo(3.7, 2);
    expect(marginWidthOf(weights)).toBeCloseTo(0.54, 2);
    expect(SVM_C).toBe(8);
  });

  it("has 12 of 18 support vectors, and every other point is class 0 well outside the street", () => {
    expect(supports).toHaveLength(12);
    expect(otherIndexes).toEqual([1, 2, 4, 7, 9, 10]);
    for (const index of otherIndexes) {
      expect(POINTS[index].label).toBe(0);
      expect(-score(weights, POINTS[index].x, POINTS[index].y)).toBeGreaterThan(1.001);
    }
  });

  it("is rebuilt by its support vectors alone, to within 0.01 per weight", () => {
    const rebuilt = fitSvm(supports);
    expect(Math.abs(rebuilt.w0 - weights.w0)).toBeLessThan(0.01);
    expect(Math.abs(rebuilt.w1 - weights.w1)).toBeLessThan(0.01);
    expect(Math.abs(rebuilt.w2 - weights.w2)).toBeLessThan(0.01);
  });

  it("barely moves when a non-support point is deleted and moves the bias by a third when a support vector is", () => {
    const withoutOther = fitSvm(POINTS.filter((_, index) => index !== otherIndexes[0]));
    expect(Math.abs(withoutOther.w0 - weights.w0)).toBeLessThan(0.01);
    expect(Math.abs(withoutOther.w1 - weights.w1)).toBeLessThan(0.01);
    expect(Math.abs(withoutOther.w2 - weights.w2)).toBeLessThan(0.01);
    const withoutSupport = fitSvm(POINTS.filter((_, index) => index !== supportIndexes[0]));
    expect(Math.abs(withoutSupport.w0 - weights.w0)).toBeGreaterThan(0.3);
    expect(Math.abs(withoutSupport.w0 - weights.w0)).toBeLessThan(0.36);
  });

  it("charges the two bottom-right class-1 points about two thirds of the objective", () => {
    const hinge = (index: number) => Math.max(0, 1 - score(weights, POINTS[index].x, POINTS[index].y));
    expect(hinge(16)).toBeCloseTo(3.54, 2);
    expect(hinge(17)).toBeCloseTo(3.51, 2);
    const regularizer = 0.5 * (weights.w1 ** 2 + weights.w2 ** 2);
    const total = regularizer + SVM_C * POINTS.reduce((sum, point) => sum + Math.max(0, 1 - (point.label === 1 ? 1 : -1) * score(weights, point.x, point.y)), 0);
    expect((SVM_C * (hinge(16) + hinge(17))) / total).toBeGreaterThan(0.65);
    expect((SVM_C * (hinge(16) + hinge(17))) / total).toBeLessThan(0.68);
  });
});

describe("the naive-Bayes product", () => {
  const classes = fitNaiveBayes(POINTS);

  it("fits priors 2/3 and 1/3 and a mean and variance per coordinate per class", () => {
    expect(classes[0].prior).toBeCloseTo(0.667, 3);
    expect(classes[1].prior).toBeCloseTo(0.333, 3);
    expect([classes[0].mx, classes[0].vx, classes[0].my, classes[0].vy].map((value) => Number(value.toFixed(3)))).toEqual([
      0.513, 0.076, 0.51, 0.06,
    ]);
    expect([classes[1].mx, classes[1].vx, classes[1].my, classes[1].vy].map((value) => Number(value.toFixed(3)))).toEqual([
      0.407, 0.098, 0.633, 0.074,
    ]);
  });

  it("scores the default probe (0.50, 0.50) as 0.667 × 1.446 × 1.632 = 1.574 against 0.333 × 1.217 × 1.298 = 0.527", () => {
    const [zero, one] = naiveFactors(classes, 0.5, 0.5);
    expect(zero.densityX).toBeCloseTo(1.446, 3);
    expect(zero.densityY).toBeCloseTo(1.632, 3);
    expect(zero.score).toBeCloseTo(1.574, 3);
    expect(one.densityX).toBeCloseTo(1.217, 3);
    expect(one.densityY).toBeCloseTo(1.298, 3);
    expect(one.score).toBeCloseTo(0.527, 3);
    expect(one.score / (zero.score + one.score)).toBeCloseTo(0.251, 3);
    expect(naivePosterior(classes, 0.5, 0.5)).toBeCloseTo(one.score / (zero.score + one.score), 12);
  });

  it("assumes zero correlation inside each class, which the data contradicts: r = −0.92 and +0.91", () => {
    expect(classCorrelation(POINTS, 1)).toBeCloseTo(-0.92, 2);
    expect(classCorrelation(POINTS, 0)).toBeCloseTo(0.91, 2);
  });

  it("classifies 14 of 18 training points", () => {
    const hits = POINTS.filter((point) => (naivePosterior(classes, point.x, point.y) >= 0.5 ? 1 : 0) === point.label).length;
    expect(hits).toBe(14);
  });
});
