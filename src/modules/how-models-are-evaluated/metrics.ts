/** Counts and readings of one confusion matrix. Every number on the page is computed here. */

export type Split = "train" | "val" | "test";
export type Baseline = "learned" | "always-negative" | "always-positive";
export type Example = { id: number; split: Split; label: 0 | 1; score: number };
/** Which scored population the lab runs on: the deliberate 1-in-100 imbalance, or about one positive in five. */
export type Population = "rare" | "common";
export const POPULATIONS = ["rare", "common"] as const;
/** What the threshold search tries to maximize on its slice. */
export type Goal = "accuracy" | "recall" | "f1";
export const GOALS = ["accuracy", "recall", "f1"] as const;

const splitOf = (id: number): Split => (id < 70 ? "train" : id < 85 ? "val" : "test");

/** 100 authored rows: 70 train, 15 val, 15 test. Row 99 is the only positive; row 12 is a high-scoring negative. */
export const EXAMPLES: Example[] = Array.from({ length: 100 }, (_, id) => {
  const label: 0 | 1 = id === 99 ? 1 : 0;
  const score = id === 99 ? 0.42 : id === 12 ? 0.88 : 0.04 + (id % 19) * 0.012;
  return { id, split: splitOf(id), label, score };
});

/** A small seeded generator (mulberry32), so the second population is the same on every machine. */
const seededRandom = (seed: number) => {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};

/**
 * The scoring process behind the 20% population. A detector's score is a noisy reading of the label:
 * positives centre on 0.64, negatives on 0.30, both with a spread, so the two overlap and no threshold
 * separates them cleanly. Scores are clipped to [0.02, 0.98] and rounded to three decimals.
 */
export const COMMON_PROCESS = {
  positiveMean: 0.64,
  positiveSd: 0.16,
  negativeMean: 0.3,
  negativeSd: 0.14,
  low: 0.02,
  high: 0.98,
} as const;

const drawScore = (label: 0 | 1, random: () => number) => {
  const u1 = random();
  const u2 = random();
  const z = Math.sqrt(-2 * Math.log(1 - u1)) * Math.cos(2 * Math.PI * u2);
  const mean = label === 1 ? COMMON_PROCESS.positiveMean : COMMON_PROCESS.negativeMean;
  const sd = label === 1 ? COMMON_PROCESS.positiveSd : COMMON_PROCESS.negativeSd;
  const raw = Math.min(COMMON_PROCESS.high, Math.max(COMMON_PROCESS.low, mean + sd * z));
  return Math.round(raw * 1000) / 1000;
};

/**
 * Twenty positives among 100 rows, four in every block of twenty (so every cross-validation fold holds some),
 * which leaves 14 in train, 3 in val, and 3 in test.
 */
export const COMMON_POSITIVE_IDS: readonly number[] = [
  2, 7, 11, 16, 23, 28, 33, 37, 42, 46, 51, 57, 62, 66, 73, 78, 83, 88, 93, 97,
];

/**
 * Both seeds are authored constants. COMMON_SEED = 9 is the lowest seed, counting up from 1, whose draw has an AUC
 * between 0.90 and 0.96 and shows the leak clearly: choosing t on the 15 test rows raises test F1 by at least 10
 * points and test accuracy by at least one row, while the t chosen on val scores at least 5 F1 points higher on the
 * fresh rows than the leaked t does. That direction is the usual one, not a fluke of the seed (metrics.test.ts re-draws the process 400 times).
 * FRESH_SEED is an unrelated draw.
 */
export const COMMON_SEED = 9;
export const FRESH_SEED = 777001;

/** One draw of the 100-row 20% population. The lab shows the draw for COMMON_SEED; tests re-draw it to check what is typical. */
export const drawCommon = (seed: number): Example[] => {
  const random = seededRandom(seed);
  return Array.from({ length: 100 }, (_, id) => {
    const label: 0 | 1 = COMMON_POSITIVE_IDS.includes(id) ? 1 : 0;
    return { id, split: splitOf(id), label, score: drawScore(label, random) };
  });
};

/** Four hundred rows from the same scoring process, every fifth row positive (exactly 20%). */
export const drawFresh = (seed: number): Example[] => {
  const random = seededRandom(seed);
  return Array.from({ length: 400 }, (_, id) => {
    const label: 0 | 1 = id % 5 === 2 ? 1 : 0;
    return { id, split: "test" as Split, label, score: drawScore(label, random) };
  });
};

/** The 20% population: same split ranges as the rare one, a realistic share of positives, overlapping scores. */
export const COMMON_EXAMPLES: Example[] = drawCommon(COMMON_SEED);

/**
 * Four hundred more rows from the same scoring process. They are never used to choose a threshold, so they show
 * what a threshold actually does on rows nobody has seen.
 */
export const FRESH_ROWS: Example[] = drawFresh(FRESH_SEED);

export const examplesFor = (population: Population): Example[] => (population === "common" ? COMMON_EXAMPLES : EXAMPLES);

export const predictRow = (example: Example, baseline: Baseline, threshold: number): 0 | 1 =>
  baseline === "always-negative" ? 0 : baseline === "always-positive" ? 1 : example.score >= threshold ? 1 : 0;

export type Matrix = {
  tp: number;
  fp: number;
  tn: number;
  fn: number;
  n: number;
  accuracy: number;
  /** TP/(TP+FP), or null when nothing was predicted positive (0/0). */
  precision: number | null;
  /** TP/(TP+FN), or null when the slice holds no positives (0/0). */
  recall: number | null;
  /** 2TP/(2TP+FP+FN), the harmonic mean of precision and recall, or null when all three counts are 0. */
  f1: number | null;
};

export const countMatrix = (rows: readonly Example[], baseline: Baseline, threshold: number): Matrix => {
  let tp = 0;
  let fp = 0;
  let tn = 0;
  let fn = 0;
  for (const example of rows) {
    const predicted = predictRow(example, baseline, threshold);
    if (example.label === 1 && predicted === 1) tp += 1;
    else if (example.label === 0 && predicted === 1) fp += 1;
    else if (example.label === 0 && predicted === 0) tn += 1;
    else fn += 1;
  }
  const n = rows.length;
  return {
    tp,
    fp,
    tn,
    fn,
    n,
    accuracy: n === 0 ? 0 : (tp + tn) / n,
    precision: tp + fp === 0 ? null : tp / (tp + fp),
    recall: tp + fn === 0 ? null : tp / (tp + fn),
    f1: 2 * tp + fp + fn === 0 ? null : (2 * tp) / (2 * tp + fp + fn),
  };
};

export const THRESHOLDS = Array.from({ length: 101 }, (_, step) => step / 100);

/** The reading of a matrix that a threshold search is trying to raise. */
export const readingOf = (counted: Matrix, goal: Goal) =>
  goal === "recall" ? counted.recall : goal === "f1" ? counted.f1 : counted.accuracy;

/**
 * Search t on one slice. Ties go to the highest t, so an all-zero or undefined search falls back to
 * t = 1.00 (never fire). A score of null means the goal is undefined on that slice.
 */
export const bestThreshold = (rows: readonly Example[], goal: Goal) => {
  let winner: { threshold: number; score: number | null } = { threshold: 1, score: null };
  for (const threshold of THRESHOLDS) {
    const counted = countMatrix(rows, "learned", threshold);
    const score = readingOf(counted, goal);
    if (score === null) continue;
    if (winner.score === null || score > winner.score + 1e-9 || (Math.abs(score - winner.score) < 1e-9 && threshold > winner.threshold)) {
      winner = { threshold, score };
    }
  }
  return winner;
};

/** Rows of one slice, in the order given. */
export const slicesOf = (rows: readonly Example[]) => ({
  val: rows.filter((example) => example.split === "val"),
  test: rows.filter((example) => example.split === "test"),
});

/**
 * The leakage comparison: pick t on val and score test (the intact path), or pick t on test and score test
 * (the leaked path), and score both thresholds on rows that were never used to choose either. A reading of
 * null means it is 0/0 on that slice.
 */
export const leakReport = (rows: readonly Example[], goal: Goal, fresh: readonly Example[] | null) => {
  const { val, test } = slicesOf(rows);
  const fromVal = bestThreshold(val, goal);
  const fromTest = bestThreshold(test, goal);
  const read = (slice: readonly Example[], threshold: number) => readingOf(countMatrix(slice, "learned", threshold), goal);
  const intactTest = read(test, fromVal.threshold);
  const leakedTest = read(test, fromTest.threshold);
  return {
    valThreshold: fromVal.threshold,
    testThreshold: fromTest.threshold,
    intactTest,
    leakedTest,
    /** Leaked minus intact test reading, in absolute points (0.12 = twelve percentage points). */
    inflation: intactTest === null || leakedTest === null ? null : leakedTest - intactTest,
    intactFresh: fresh ? read(fresh, fromVal.threshold) : null,
    leakedFresh: fresh ? read(fresh, fromTest.threshold) : null,
  };
};

/** True positive rate and false positive rate of a matrix: null when the slice holds no positives or no negatives. */
export const rates = (counted: Matrix) => ({
  tpr: counted.tp + counted.fn === 0 ? null : counted.tp / (counted.tp + counted.fn),
  fpr: counted.fp + counted.tn === 0 ? null : counted.fp / (counted.fp + counted.tn),
});

/**
 * Precision the same detector would reach if a share `prevalence` of the rows were positive, holding its true
 * positive rate and false positive rate fixed: TPR·π / (TPR·π + FPR·(1−π)). This is Bayes' rule on the
 * confusion matrix. Null when either rate is undefined or nothing would be flagged.
 */
export const precisionAtPrevalence = (counted: Matrix, prevalence: number) => {
  const { tpr, fpr } = rates(counted);
  if (tpr === null || fpr === null) return null;
  const hits = tpr * prevalence;
  const alarms = fpr * (1 - prevalence);
  return hits + alarms === 0 ? null : hits / (hits + alarms);
};

/** Accuracy, precision, recall, and F1 at every threshold on a slice. */
export const sweep = (rows: readonly Example[]) =>
  THRESHOLDS.map((threshold) => ({ threshold, ...countMatrix(rows, "learned", threshold) }));

/** Five contiguous folds of twenty rows. */
export const foldOf = (example: Example) => Math.floor(example.id / 20);

export type CurveKind = "underfit" | "good" | "overfit";

/**
 * Illustrative learning curves, not a training run. All three start near the same initial loss, as a
 * freshly initialized model would on both slices.
 */
export const curveValue = (kind: CurveKind, series: "train" | "val", epoch: number) => {
  const decay = (tau: number) => Math.exp(-epoch / tau);
  if (kind === "underfit") return series === "train" ? 0.45 + 0.55 * decay(3) : 0.49 + 0.55 * decay(3);
  if (kind === "good") return series === "train" ? 0.14 + 0.88 * decay(5) : 0.19 + 0.86 * decay(5);
  return series === "train" ? 0.05 + 0.9 * decay(4) : 0.1 + 0.88 * decay(4) + 0.0009 * epoch * epoch;
};

export const EPOCHS = Array.from({ length: 21 }, (_, epoch) => epoch);

export const curvePoints = (kind: CurveKind, series: "train" | "val") =>
  EPOCHS.map((epoch) => ({ x: epoch, y: curveValue(kind, series, epoch) }));

/** The epoch with the lowest validation loss — where early stopping would keep the weights. */
export const bestEpoch = (kind: CurveKind) =>
  EPOCHS.reduce((best, epoch) => (curveValue(kind, "val", epoch) < curveValue(kind, "val", best) ? epoch : best), 0);

export const regressionErrors = (residuals: readonly number[]) => {
  const n = residuals.length;
  const mae = residuals.reduce((sum, value) => sum + Math.abs(value), 0) / n;
  const mse = residuals.reduce((sum, value) => sum + value * value, 0) / n;
  return { mae, mse, rmse: Math.sqrt(mse) };
};

/**
 * ROC curve over every distinct score (plus the two ends), and AUC as the Mann–Whitney probability that a
 * random positive outscores a random negative, ties counting one half. Null when a class is missing.
 */
export const roc = (rows: readonly Example[]) => {
  const positives = rows.filter((example) => example.label === 1);
  const negatives = rows.filter((example) => example.label === 0);
  if (positives.length === 0 || negatives.length === 0) return null;
  const cuts = [Infinity, ...[...new Set(rows.map((example) => example.score))].sort((a, b) => b - a)];
  const points = cuts.map((cut) => ({
    threshold: cut,
    fpr: negatives.filter((example) => example.score >= cut).length / negatives.length,
    tpr: positives.filter((example) => example.score >= cut).length / positives.length,
  }));
  let wins = 0;
  for (const positive of positives) {
    for (const negative of negatives) {
      wins += positive.score > negative.score ? 1 : positive.score === negative.score ? 0.5 : 0;
    }
  }
  return { points, auc: wins / (positives.length * negatives.length) };
};
