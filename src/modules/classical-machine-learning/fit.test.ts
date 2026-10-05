import { describe, expect, it } from "vitest";
import {
  POINTS,
  buildForest,
  buildTree,
  countLeaves,
  fitKMeans,
  fitLinear,
  fitLogistic,
  fitNaiveBayes,
  fitSvm,
  forestVotes,
  gini,
  knnVote,
  lineInSquare,
  logLoss,
  naivePosterior,
  score,
  sigmoid,
  svmObjective,
  treePredict,
  treeSegments,
  type Point,
} from "./fit";

const accuracy = (predict: (point: Point) => number) =>
  POINTS.filter((point) => predict(point) === point.label).length;
const missed = (predict: (point: Point) => number) =>
  POINTS.map((point, index) => (predict(point) === point.label ? -1 : index)).filter((index) => index >= 0);
const bottomRight = [16, 17];

describe("the authored set", () => {
  it("has twelve class-0 and six class-1 points in an uneven XOR", () => {
    expect(POINTS).toHaveLength(18);
    expect(POINTS.filter((point) => point.label === 1)).toHaveLength(6);
    expect(POINTS.filter((point) => point.label === 1 && point.x < 0.5 && point.y > 0.5)).toHaveLength(4);
    expect(POINTS.filter((point) => point.label === 1 && point.x > 0.5 && point.y < 0.5)).toHaveLength(2);
  });
});

describe("linear methods", () => {
  it("solves the least-squares normal equation: residuals are orthogonal to [1, x, y]", () => {
    const weights = fitLinear(POINTS);
    const residual = POINTS.map((point) => score(weights, point.x, point.y) - point.label);
    expect(residual.reduce((sum, value) => sum + value, 0)).toBeCloseTo(0, 9);
    expect(residual.reduce((sum, value, index) => sum + value * POINTS[index].x, 0)).toBeCloseTo(0, 9);
    expect(residual.reduce((sum, value, index) => sum + value * POINTS[index].y, 0)).toBeCloseTo(0, 9);
  });

  it("draws a line inside the square that takes the top-left corner and misses the bottom-right pair", () => {
    const linear = fitLinear(POINTS);
    const logistic = fitLogistic(POINTS);
    const svm = fitSvm(POINTS);
    expect(lineInSquare(linear, 0.5)).not.toBeNull();
    expect(lineInSquare(logistic, 0)).not.toBeNull();
    expect(lineInSquare(svm, 0)).not.toBeNull();
    expect(missed((point) => (score(linear, point.x, point.y) >= 0.5 ? 1 : 0))).toEqual(bottomRight);
    expect(missed((point) => (sigmoid(score(logistic, point.x, point.y)) >= 0.5 ? 1 : 0))).toEqual(bottomRight);
    expect(missed((point) => (score(svm, point.x, point.y) >= 0 ? 1 : 0))).toEqual(bottomRight);
  });

  it("lowers log-loss from ln 2 at w = 0", () => {
    const start = logLoss({ w0: 0, w1: 0, w2: 0 }, POINTS);
    expect(start).toBeCloseTo(Math.LN2, 9);
    expect(logLoss(fitLogistic(POINTS), POINTS)).toBeLessThan(0.6);
  });

  it("gets the SVM objective within half a percent of a much longer run", () => {
    const fitted = svmObjective(fitSvm(POINTS), POINTS);
    const reference = svmObjective(fitSvm(POINTS, 20000), POINTS);
    expect(fitted).toBeLessThan(reference * 1.005);
  });
});

describe("local and axis-aligned methods", () => {
  it("lets k ≤ 4 neighbours keep the two-point corner and k ≥ 5 vote it away", () => {
    for (const k of [1, 2, 3, 4]) expect(accuracy((point) => knnVote(point.x, point.y, k))).toBe(18);
    for (const k of [5, 6, 7]) {
      expect(missed((point) => knnVote(point.x, point.y, k))).toEqual(bottomRight);
    }
  });

  it("uses binary Gini 2p(1 − p)", () => {
    expect(gini(POINTS)).toBeCloseTo(2 * (6 / 18) * (12 / 18), 12);
  });

  it("goes from one cut at depth 1 to all 18 points at depth 5", () => {
    expect(accuracy((point) => treePredict(buildTree(POINTS, 1), point.x, point.y))).toBe(14);
    const deep = buildTree(POINTS, 5);
    expect(accuracy((point) => treePredict(deep, point.x, point.y))).toBe(18);
    expect(countLeaves(deep)).toBe(7);
  });

  it("bounds each split segment by its parent's region", () => {
    const segments = treeSegments(buildTree(POINTS, 3));
    const root = segments[0];
    expect(root.depth).toBe(0);
    expect(root.from).toBe(0);
    expect(root.to).toBe(1);
    const child = segments.find((segment) => segment.depth === 1);
    expect(child).toBeTruthy();
    expect(child!.to - child!.from).toBeLessThan(1);
  });

  it("builds a seven-tree forest whose vote is a count", () => {
    const forest = buildForest(3);
    expect(forest).toHaveLength(7);
    const votes = forestVotes(forest, 0.1, 0.85);
    expect(votes).toBeGreaterThanOrEqual(0);
    expect(votes).toBeLessThanOrEqual(7);
  });
});

describe("naive Bayes and k-means", () => {
  it("fires class 1 only in the far top-left corner", () => {
    const classes = fitNaiveBayes(POINTS);
    expect(classes[0].prior).toBeCloseTo(12 / 18, 12);
    expect(naivePosterior(classes, 0.5, 0.5)).toBeLessThan(0.5);
    expect(naivePosterior(classes, 0.1, 0.9)).toBeGreaterThan(0.5);
    expect(naivePosterior(classes, 0.85, 0.25)).toBeLessThan(0.5);
    expect(accuracy((point) => (naivePosterior(classes, point.x, point.y) >= 0.5 ? 1 : 0))).toBe(14);
  });

  it("never reads a label: flipping every label leaves the clusters unchanged", () => {
    const flipped = POINTS.map((point) => ({ ...point, label: (1 - point.label) as 0 | 1 }));
    expect(fitKMeans(4, flipped).centers).toEqual(fitKMeans(4).centers);
    expect(fitKMeans(4).inertia).toBeLessThan(fitKMeans(2).inertia);
  });
});
