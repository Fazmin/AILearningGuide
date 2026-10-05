import { describe, expect, it } from "vitest";
import {
  COMMON_EXAMPLES,
  COMMON_POSITIVE_IDS,
  COMMON_PROCESS,
  EXAMPLES,
  FRESH_ROWS,
  bestEpoch,
  bestThreshold,
  countMatrix,
  curveValue,
  drawCommon,
  drawFresh,
  examplesFor,
  foldOf,
  leakReport,
  precisionAtPrevalence,
  rates,
  regressionErrors,
  roc,
  sweep,
  THRESHOLDS,
} from "./metrics";

const slice = (split: string) => EXAMPLES.filter((example) => split === "all" || example.split === split);

describe("the authored detector", () => {
  it("has one positive, in test, below one confident false alarm in train", () => {
    expect(EXAMPLES.filter((example) => example.label === 1).map((example) => example.id)).toEqual([99]);
    expect(EXAMPLES[99].split).toBe("test");
    expect(EXAMPLES[12].split).toBe("train");
    const others = EXAMPLES.filter((example) => example.id !== 12 && example.id !== 99);
    expect(Math.max(...others.map((example) => example.score))).toBeCloseTo(0.256, 9);
  });
});

describe("confusion matrix readings", () => {
  it("gives 99% accuracy and 0% recall for the never-fire rule on all 100", () => {
    const counted = countMatrix(slice("all"), "always-negative", 0.5);
    expect(counted).toMatchObject({ tp: 0, fp: 0, tn: 99, fn: 1 });
    expect(counted.accuracy).toBeCloseTo(0.99, 12);
    expect(counted.recall).toBe(0);
    expect(counted.precision).toBeNull();
    expect(counted.f1).toBe(0);
  });

  it("leaves recall undefined, not zero, on slices with no positives", () => {
    for (const split of ["train", "val"]) {
      const counted = countMatrix(slice(split), "always-negative", 0.5);
      expect(counted.accuracy).toBe(1);
      expect(counted.recall).toBeNull();
      expect(counted.f1).toBeNull();
    }
  });

  it("catches row 99 and row 12 at t = 0.42, only row 12 between 0.43 and 0.88, and nothing above 0.88", () => {
    expect(countMatrix(slice("all"), "learned", 0.42)).toMatchObject({ tp: 1, fp: 1, tn: 98, fn: 0 });
    expect(countMatrix(slice("all"), "learned", 0.6)).toMatchObject({ tp: 0, fp: 1, fn: 1 });
    expect(countMatrix(slice("all"), "learned", 0.88)).toMatchObject({ tp: 0, fp: 1 });
    expect(countMatrix(slice("all"), "learned", 0.89)).toMatchObject({ tp: 0, fp: 0 });
    const both = countMatrix(slice("all"), "learned", 0.42);
    expect(both.precision).toBeCloseTo(0.5, 12);
    expect(both.f1).toBeCloseTo((2 * 0.5 * 1) / 1.5, 12);
  });

  it("computes F1 as the harmonic mean of precision and recall", () => {
    for (const point of sweep(slice("all"))) {
      if (point.precision === null || point.recall === null || point.f1 === null) continue;
      const harmonic =
        point.precision + point.recall === 0 ? 0 : (2 * point.precision * point.recall) / (point.precision + point.recall);
      expect(point.f1).toBeCloseTo(harmonic, 12);
    }
  });
});

describe("choosing t", () => {
  it("maximizing recall on test picks t = 0.42 and leaks row 99", () => {
    const chosen = bestThreshold(slice("test"), "recall");
    expect(chosen.threshold).toBeCloseTo(0.42, 12);
    expect(chosen.score).toBe(1);
  });

  it("cannot tune recall on train or val: no positives, so the search falls back to t = 1", () => {
    for (const split of ["train", "val"]) {
      const chosen = bestThreshold(slice(split), "recall");
      expect(chosen.score).toBeNull();
      expect(chosen.threshold).toBe(1);
    }
  });

  it("maximizing accuracy on val prefers never firing, which scores 93.3% on test", () => {
    const chosen = bestThreshold(slice("val"), "accuracy");
    expect(chosen.threshold).toBe(1);
    expect(countMatrix(slice("test"), "learned", chosen.threshold).accuracy).toBeCloseTo(14 / 15, 12);
  });
});

describe("cross-validation and curves", () => {
  it("puts row 99 in fold 4 only and averages never-fire accuracy to 99%", () => {
    const folds = [0, 1, 2, 3, 4].map((fold) =>
      countMatrix(EXAMPLES.filter((example) => foldOf(example) === fold), "always-negative", 1),
    );
    expect(folds.map((fold) => fold.recall)).toEqual([null, null, null, null, 0]);
    expect(folds.reduce((sum, fold) => sum + fold.accuracy, 0) / 5).toBeCloseTo(0.99, 12);
  });

  it("starts train and val near the same loss and turns val up only for the overfit sketch", () => {
    for (const kind of ["underfit", "good", "overfit"] as const) {
      expect(Math.abs(curveValue(kind, "train", 0) - curveValue(kind, "val", 0))).toBeLessThan(0.06);
      expect(curveValue(kind, "train", 20)).toBeLessThan(curveValue(kind, "train", 0));
    }
    expect(bestEpoch("overfit")).toBeGreaterThan(5);
    expect(bestEpoch("overfit")).toBeLessThan(15);
    expect(curveValue("overfit", "val", 20)).toBeGreaterThan(curveValue("overfit", "val", bestEpoch("overfit")));
    expect(bestEpoch("good")).toBe(20);
  });

  it("puts the overfit sketch's lowest val loss at epoch 10, as the lesson says", () => {
    expect(bestEpoch("overfit")).toBe(10);
  });

  it("searches 101 thresholds, 0.00 to 1.00", () => {
    expect(THRESHOLDS).toHaveLength(101);
    expect([THRESHOLDS[0], THRESHOLDS[100]]).toEqual([0, 1]);
  });

  it("squares the outlier in MSE but not in MAE", () => {
    const errors = regressionErrors([0.2, 0.1, -0.1, 0.2, 8]);
    expect(errors.mae).toBeCloseTo(1.72, 12);
    expect(errors.mse).toBeCloseTo(12.82, 12);
  });
});

describe("ranking", () => {
  it("ranks row 99 above 98 of the 99 negatives: AUC 0.990 on all 100, 1.0 on test, none on train", () => {
    expect(roc(slice("all"))!.auc).toBeCloseTo(98 / 99, 12);
    expect(roc(slice("test"))!.auc).toBe(1);
    expect(roc(slice("train"))).toBeNull();
  });

  it("draws the ROC from (0, 0) to (1, 1), and its trapezoid area equals the AUC", () => {
    const curve = roc(slice("all"))!;
    expect(curve.points[0]).toMatchObject({ fpr: 0, tpr: 0 });
    expect(curve.points[curve.points.length - 1]).toMatchObject({ fpr: 1, tpr: 1 });
    let area = 0;
    for (let i = 1; i < curve.points.length; i += 1) {
      const a = curve.points[i - 1];
      const b = curve.points[i];
      area += (b.fpr - a.fpr) * (a.tpr + b.tpr) * 0.5;
    }
    expect(area).toBeCloseTo(curve.auc, 12);
  });
});

const common = (split: string) => COMMON_EXAMPLES.filter((example) => split === "all" || example.split === split);
const round = (value: number | null, digits = 3) => (value === null ? null : Number(value.toFixed(digits)));

describe("the 1-in-5 population", () => {
  it("is selected by examplesFor and leaves the rare population untouched", () => {
    expect(examplesFor("rare")).toBe(EXAMPLES);
    expect(examplesFor("common")).toBe(COMMON_EXAMPLES);
    expect(COMMON_EXAMPLES).toHaveLength(100);
  });

  it("holds twenty positives: exactly 20 percent in train, val, and test, and four in every fold", () => {
    expect(COMMON_POSITIVE_IDS).toHaveLength(20);
    const positivesIn = (rows: typeof COMMON_EXAMPLES) => rows.filter((example) => example.label === 1).length;
    expect(["train", "val", "test"].map((split) => positivesIn(common(split)))).toEqual([14, 3, 3]);
    expect(["train", "val", "test"].map((split) => positivesIn(common(split)) / common(split).length)).toEqual([0.2, 0.2, 0.2]);
    for (const fold of [0, 1, 2, 3, 4]) {
      expect(positivesIn(COMMON_EXAMPLES.filter((example) => foldOf(example) === fold))).toBe(4);
    }
  });

  it("gives the never-fire rule 80 percent accuracy and zero recall on every slice and fold", () => {
    for (const split of ["all", "train", "val", "test"]) {
      const counted = countMatrix(common(split), "always-negative", 0.5);
      expect(counted.accuracy).toBe(0.8);
      expect(counted.recall).toBe(0);
    }
    for (const fold of [0, 1, 2, 3, 4]) {
      const counted = countMatrix(COMMON_EXAMPLES.filter((example) => foldOf(example) === fold), "always-negative", 1);
      expect(counted).toMatchObject({ tp: 0, fn: 4, tn: 16, fp: 0 });
      expect(counted.recall).toBe(0);
    }
  });

  it("draws overlapping scores from the stated process, clipped and rounded", () => {
    const positives = COMMON_EXAMPLES.filter((example) => example.label === 1).map((example) => example.score);
    const negatives = COMMON_EXAMPLES.filter((example) => example.label === 0).map((example) => example.score);
    expect(COMMON_PROCESS).toMatchObject({ positiveMean: 0.64, negativeMean: 0.3 });
    expect(Math.min(...positives)).toBeLessThan(Math.max(...negatives));
    for (const score of [...positives, ...negatives]) {
      expect(score).toBeGreaterThanOrEqual(COMMON_PROCESS.low);
      expect(score).toBeLessThanOrEqual(COMMON_PROCESS.high);
      expect(Math.round(score * 1000) / 1000).toBe(score);
    }
    expect(roc(common("all"))!.auc).toBeCloseTo(0.94875, 9);
    expect(roc(FRESH_ROWS)!.auc).toBeCloseTo(0.9281640625, 9);
  });

  it("is reproducible: the same seed draws the same rows, and a different seed draws different ones", () => {
    expect(drawCommon(9)).toEqual(COMMON_EXAMPLES);
    expect(drawFresh(777001)).toEqual(FRESH_ROWS);
    expect(drawCommon(10)).not.toEqual(COMMON_EXAMPLES);
    expect(FRESH_ROWS.filter((example) => example.label === 1)).toHaveLength(80);
    expect(FRESH_ROWS).toHaveLength(400);
  });

  it("reads TP 16, FP 7, TN 73, FN 4 at t = 0.50, so precision is 16/23, recall 16/20, and F1 32/43", () => {
    const counted = countMatrix(common("all"), "learned", 0.5);
    expect(counted).toMatchObject({ tp: 16, fp: 7, tn: 73, fn: 4 });
    expect(counted.precision).toBeCloseTo(16 / 23, 12);
    expect(counted.recall).toBeCloseTo(0.8, 12);
    expect(counted.f1).toBeCloseTo(32 / 43, 12);
    expect(counted.accuracy).toBeCloseTo(0.89, 12);
    expect(round(counted.precision, 3)).toBe(0.696);
    expect(round(counted.f1, 3)).toBe(0.744);
  });

  it("trades precision for recall as t rises: false alarms leave first, real positives later", () => {
    const at = (t: number) => countMatrix(common("all"), "learned", t);
    expect(at(0.4)).toMatchObject({ tp: 19, fp: 22 });
    expect(at(0.6)).toMatchObject({ tp: 11, fp: 1 });
    expect([round(at(0.4).precision, 3), round(at(0.6).precision, 3)]).toEqual([0.463, 0.917]);
    expect([at(0.4).recall, at(0.6).recall]).toEqual([0.95, 0.55]);
    expect(at(0.7)).toMatchObject({ fp: 0, tp: 9 });
    const points = sweep(common("all"));
    for (let index = 1; index < points.length; index += 1) {
      expect(points[index].recall!).toBeLessThanOrEqual(points[index - 1].recall!);
    }
    expect(at(0.3).accuracy).toBeLessThan(at(0.6).accuracy);
    expect(at(0.9).accuracy).toBeLessThan(at(0.6).accuracy);
    expect(bestThreshold(common("all"), "accuracy").threshold).toBeGreaterThan(0.3);
    expect(bestThreshold(common("all"), "accuracy").threshold).toBeLessThan(0.9);
  });

  it("moves precision from 29.9 to 100 percent and recall from 100 to 45 percent between t = 0.30 and t = 0.70", () => {
    const at = (t: number) => countMatrix(common("all"), "learned", t);
    expect([round(at(0.3).precision, 3), at(0.7).precision]).toEqual([0.299, 1]);
    expect([at(0.3).recall, at(0.7).recall]).toEqual([1, 0.45]);
  });

  it("peaks F1 and accuracy where the lab says", () => {
    const f1 = bestThreshold(common("all"), "f1");
    const accuracy = bestThreshold(common("all"), "accuracy");
    expect(f1.threshold).toBe(0.54);
    expect(f1.score).toBeCloseTo(32 / 41, 12);
    expect(accuracy.threshold).toBe(0.62);
    expect(accuracy.score).toBeCloseTo(0.91, 12);
    const nudged = countMatrix(common("all"), "learned", 0.54);
    expect(nudged).toMatchObject({ tp: 16, fp: 5, fn: 4 });
    expect(nudged.precision).toBeCloseTo(16 / 21, 12);
    expect(round(nudged.precision, 3)).toBe(0.762);
  });
});

describe("base rate", () => {
  it("returns the rule's own precision at its own base rate, and a much lower one at 1 in 100", () => {
    const counted = countMatrix(common("all"), "learned", 0.5);
    expect(rates(counted).tpr).toBeCloseTo(0.8, 12);
    expect(rates(counted).fpr).toBeCloseTo(7 / 80, 12);
    expect(precisionAtPrevalence(counted, 0.2)).toBeCloseTo(16 / 23, 12);
    expect(round(precisionAtPrevalence(counted, 0.01), 3)).toBe(0.085);
    expect(precisionAtPrevalence(counted, 0.01)!).toBeLessThan(precisionAtPrevalence(counted, 0.2)! / 5);
  });

  it("recovers the rare population's own precision of 0.5, and the base rate itself for always-positive", () => {
    const rare = countMatrix(slice("all"), "learned", 0.42);
    expect(precisionAtPrevalence(rare, 0.01)).toBeCloseTo(0.5, 12);
    const everything = countMatrix(slice("all"), "always-positive", 0.5);
    expect(precisionAtPrevalence(everything, 0.2)).toBeCloseTo(0.2, 12);
    expect(precisionAtPrevalence(everything, 0.01)).toBeCloseTo(0.01, 12);
  });

  it("has no value when nothing would fire, or when a class is missing", () => {
    expect(precisionAtPrevalence(countMatrix(slice("all"), "always-negative", 0.5), 0.2)).toBeNull();
    expect(precisionAtPrevalence(countMatrix(slice("train"), "learned", 0.1), 0.2)).toBeNull();
  });
});

describe("leakage on the 1-in-5 population", () => {
  it("raises the test claim for the leaked path and then fails to keep it on fresh rows", () => {
    const f1 = leakReport(COMMON_EXAMPLES, "f1", FRESH_ROWS);
    expect([f1.valThreshold, f1.testThreshold]).toEqual([0.55, 0.46]);
    expect(round(f1.intactTest, 3)).toBe(0.667);
    expect(round(f1.leakedTest, 3)).toBe(0.857);
    expect(round(f1.inflation, 3)).toBe(0.19);
    expect(round(f1.intactFresh, 3)).toBe(0.779);
    expect(round(f1.leakedFresh, 3)).toBe(0.688);
    expect(f1.leakedTest!).toBeGreaterThan(f1.leakedFresh!);
    expect(f1.intactFresh!).toBeGreaterThan(f1.leakedFresh!);
  });

  it("inflates accuracy and recall too, and the leaked claim beats the fresh reading for every goal", () => {
    for (const goal of ["accuracy", "recall", "f1"] as const) {
      const report = leakReport(COMMON_EXAMPLES, goal, FRESH_ROWS);
      expect(report.inflation!, goal).toBeGreaterThan(0);
      expect(report.leakedTest!, goal).toBeGreaterThan(report.leakedFresh!);
    }
    const accuracy = leakReport(COMMON_EXAMPLES, "accuracy", FRESH_ROWS);
    expect([round(accuracy.intactTest, 3), round(accuracy.leakedTest, 3)]).toEqual([0.867, 0.933]);
  });

  it("also inflates the rare population's accuracy by one row, from 14/15 to 15/15, with no fresh batch", () => {
    const report = leakReport(EXAMPLES, "accuracy", null);
    expect(report.valThreshold).toBe(1);
    expect(report.testThreshold).toBeCloseTo(0.42, 12);
    expect(report.intactTest).toBeCloseTo(14 / 15, 12);
    expect(report.leakedTest).toBe(1);
    expect(report.inflation).toBeCloseTo(1 / 15, 12);
    expect(report.intactFresh).toBeNull();
  });

  it("is the usual direction, not a fluke of the seed: across 400 re-draws the leaked F1 claim exceeds the intact one", () => {
    const fresh = FRESH_ROWS;
    let draws = 0;
    let higher = 0;
    let gapSum = 0;
    let overclaimLeaked = 0;
    let overclaimIntact = 0;
    for (let seed = 1; seed <= 400; seed += 1) {
      const report = leakReport(drawCommon(seed), "f1", fresh);
      if (report.inflation === null || report.intactFresh === null || report.leakedFresh === null) continue;
      draws += 1;
      if (report.inflation > 0) higher += 1;
      gapSum += report.inflation;
      overclaimLeaked += report.leakedTest! - report.leakedFresh;
      overclaimIntact += report.intactTest! - report.intactFresh;
    }
    expect(draws).toBeGreaterThan(380);
    expect(higher / draws).toBeGreaterThan(0.5);
    expect(round(gapSum / draws, 2)).toBe(0.22);
    // The leaked claim overstates what fresh rows show by far more than the intact claim does.
    expect(overclaimLeaked / draws).toBeGreaterThan(0.1);
    expect(Math.abs(overclaimIntact / draws)).toBeLessThan(0.1);
  });
});
