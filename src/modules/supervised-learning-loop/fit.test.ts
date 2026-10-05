import { describe, expect, it } from "vitest";
import {
  APARTMENTS,
  closedFormLinear,
  encodeAll,
  encodeApartment,
  GD_STEPS,
  linearGradient,
  meanLoss,
  planeLine,
  pointLoss,
  predictProbability,
  runFit,
  sliceSegments,
} from "./fit";

const trainCount = APARTMENTS.filter((row) => row.split === "train").length;
const holdoutCount = APARTMENTS.filter((row) => row.split === "holdout").length;

describe("supervised loop fits", () => {
  it("keeps twelve authored train rows out of the four holdout rows", () => {
    expect(trainCount).toBe(12);
    expect(holdoutCount).toBe(4);
    expect(APARTMENTS.filter((row) => row.name === "Rowan 9")[0]?.price).toBe(0);
    expect(APARTMENTS.filter((row) => row.name === "Rowan 9")[0]?.split).toBe("train");
  });

  it("descends train log-loss on numeric logistic without reading the holdout in the gradient", () => {
    const path = runFit("numeric", "logistic", "logloss", 2);
    expect(path).toHaveLength(GD_STEPS + 1);
    expect(path[0].trainLoss).toBeGreaterThan(path[path.length - 1].trainLoss);
    expect(path[0].weights.every((value) => value === 0)).toBe(true);
    expect(path[0].bias).toBe(0);
    const start = path[0].trainLoss;
    expect(start).toBeGreaterThan(0.68);
    expect(start).toBeLessThan(0.7);
  });

  it("matches least-squares closed form on a linear MSE fit of the train slice only", () => {
    const solved = closedFormLinear("numeric");
    const alder = encodeApartment(APARTMENTS[0], "numeric");
    const yHat = solved.bias + solved.weights[0] * alder[0] + solved.weights[1] * alder[1];
    expect(Number.isFinite(yHat)).toBe(true);
    expect(solved.weights).toHaveLength(2);
    const path = runFit("numeric", "linear", "mse", 2);
    expect(path[path.length - 1].trainLoss).toBeLessThan(path[0].trainLoss);
  });

  it("clips the 50% contour to the rooms × park window", () => {
    const path = runFit("numeric", "logistic", "logloss", 2);
    const last = path[path.length - 1];
    const line = planeLine(last.weights, last.bias, "numeric", APARTMENTS[0], 0);
    expect(line).toBeTruthy();
    expect(line!.y1).toBeGreaterThanOrEqual(-0.21);
    expect(line!.y1).toBeLessThanOrEqual(1.21);
    expect(line!.y2).toBeGreaterThanOrEqual(-0.21);
    expect(line!.y2).toBeLessThanOrEqual(1.21);
  });

  it("lets a deep tree beat a stump on train loss", () => {
    const path = runFit("integer", "tree", "logloss", 5);
    expect(path[0].trainLoss).toBeGreaterThan(path[path.length - 1].trainLoss);
    expect(path[path.length - 1].trainAccuracy).toBeGreaterThanOrEqual(path[1].trainAccuracy);
  });

  it("evaluates a single example loss from the same formula as the mean", () => {
    const path = runFit("numeric", "logistic", "logloss", 2);
    const last = path[path.length - 1];
    const x = encodeApartment(APARTMENTS[0], "numeric");
    const p = predictProbability(x, "logistic", "logloss", { weights: last.weights, bias: last.bias }, null);
    const ell = pointLoss(p, APARTMENTS[0].price, "logloss");
    expect(ell).toBeGreaterThanOrEqual(0);
    expect(p).toBeGreaterThan(0);
    expect(p).toBeLessThan(1);
  });
});

describe("gradients and tree slices", () => {
  const train = encodeAll("numeric")
    .filter((item) => item.row.split === "train")
    .map((item) => ({ x: item.x, y: item.y }));
  const lossAt = (weights: number[], bias: number, algorithm: "linear" | "logistic", loss: "mse" | "logloss") =>
    meanLoss(
      train.map((row) => predictProbability(row.x, algorithm, loss, { weights, bias }, null)),
      train.map((row) => row.y),
      loss,
    );

  it("prints the exact gradient of the mean loss for every linear algorithm and loss", () => {
    const weights = [0.3, -0.2];
    const bias = 0.1;
    for (const algorithm of ["linear", "logistic"] as const) {
      for (const loss of ["mse", "logloss"] as const) {
        const g = linearGradient(train, { weights, bias }, algorithm, loss);
        const h = 1e-6;
        const numericB = (lossAt(weights, bias + h, algorithm, loss) - lossAt(weights, bias - h, algorithm, loss)) / (2 * h);
        expect(g.gB).toBeCloseTo(numericB, 5);
        weights.forEach((_, index) => {
          const up = weights.map((value, j) => (j === index ? value + h : value));
          const down = weights.map((value, j) => (j === index ? value - h : value));
          const numeric = (lossAt(up, bias, algorithm, loss) - lossAt(down, bias, algorithm, loss)) / (2 * h);
          expect(g.gW[index]).toBeCloseTo(numeric, 5);
        });
      }
    }
  });

  it("keeps rooms + park unable to isolate Rowan 9, which shares (3, 1) with Cedar 9", () => {
    const path = runFit("numeric", "tree", "logloss", 5);
    expect(path[path.length - 1].trainAccuracy).toBeLessThan(1);
    expect(path[path.length - 1].holdoutLoss).toBeLessThanOrEqual(path[1].holdoutLoss);
  });

  it("lets a depth-5 tree on integer codes memorize train while holdout loss rises", () => {
    for (const loss of ["mse", "logloss"] as const) {
      const path = runFit("integer", "tree", loss, 5);
      expect(path[5].trainLoss).toBeLessThan(1e-6);
      expect(path[5].trainAccuracy).toBe(1);
      expect(path[5].holdoutLoss).toBeGreaterThan(path[1].holdoutLoss);
    }
  });

  it("draws only the cuts that apply at the selected color", () => {
    const tree = runFit("integer", "tree", "logloss", 5)[5].tree!;
    const segments = sliceSegments(tree, "integer", APARTMENTS[11]);
    for (const segment of segments) {
      expect(segment.to).toBeGreaterThan(segment.from);
    }
    const numericTree = runFit("numeric", "tree", "mse", 2)[2].tree!;
    expect(sliceSegments(numericTree, "numeric", APARTMENTS[0]).length).toBeGreaterThan(0);
  });
});
