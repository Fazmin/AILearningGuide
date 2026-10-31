import { describe, expect, it } from "vitest";
import {
  TINY_CORPORA,
  extendedScheduleFactor,
  newtonSchulz,
  scheduleFactor,
  tinyCrossEntropy,
  toHalf,
  trainTinyExtended,
  trainTinyFactored,
  trainTinyModel,
} from "./tiny-lm";

const harbor = TINY_CORPORA.harbor.text;
const recipes = TINY_CORPORA.recipes.text;
const proverbs = TINY_CORPORA.proverbs.text;

const sameRun = (a: ReturnType<typeof trainTinyModel>, b: ReturnType<typeof trainTinyModel>) => {
  expect(Array.from(a.weights)).toEqual(Array.from(b.weights));
  expect(a.history).toEqual(b.history);
  expect(a.checkpoints.map((point) => [point.step, point.progress, point.loss, ...point.weights])).toEqual(
    b.checkpoints.map((point) => [point.step, point.progress, point.loss, ...point.weights]),
  );
  expect([a.steps, a.tokens, a.updatedParameters, a.finalLoss, a.diverged, a.peakGradientNorm]).toEqual([
    b.steps,
    b.tokens,
    b.updatedParameters,
    b.finalLoss,
    b.diverged,
    b.peakGradientNorm,
  ]);
};

describe("plain SGD reproduces the original trainers bit for bit", () => {
  const start = trainTinyModel({ text: harbor, epochs: 12, seed: 2 }).weights;
  const cases = [
    { name: "defaults", options: { text: harbor } },
    {
      name: "cosine with warmup, clipping and decay",
      options: { text: harbor, epochs: 25, batchSize: 8, learningRate: 2, schedule: "cosine", warmupFraction: 0.1, clipNorm: 1, weightDecay: 0.01, seed: 3, checkpoints: 9 },
    },
    { name: "a very high rate at batch 4", options: { text: proverbs, epochs: 6, batchSize: 4, learningRate: 900, seed: 5 } },
    {
      name: "one-cycle with dense history",
      options: { text: recipes, epochs: 10, batchSize: 16, learningRate: 1.2, schedule: "one-cycle", seed: 7, historyPoints: Number.MAX_SAFE_INTEGER },
    },
    { name: "zero epochs", options: { text: harbor, epochs: 0 } },
  ] as const;

  it.each(cases.map((entry) => [entry.name, entry] as const))("on the table: %s", (_name, entry) => {
    sameRun(trainTinyExtended(entry.options), trainTinyModel(entry.options));
  });

  it("on the table from a starting table", () => {
    const options = { text: recipes, epochs: 15, learningRate: 0.8, schedule: "warmup-decay", seed: 11 } as const;
    sameRun(trainTinyExtended({ ...options, model: { kind: "table", init: start } }), trainTinyModel({ ...options, init: start }));
  });

  it("on the factored model, weights and final loss", () => {
    for (const options of [
      { text: harbor, epochs: 20, batchSize: 16, learningRate: 0.5, seed: 9 },
      { text: proverbs, epochs: 7, batchSize: 5, learningRate: 1.5, seed: 4, clipNorm: 0.5, weightDecay: 0.01, schedule: "cosine" as const },
    ]) {
      const rank = 8;
      const extended = trainTinyExtended({ ...options, model: { kind: "factored", rank } });
      const original = trainTinyFactored({ ...options, rank });
      expect(Array.from(extended.weights)).toEqual(Array.from(original.weights));
      expect(extended.finalLoss).toBe(original.finalLoss);
      expect(extended.clippedSteps).toBe(original.clippedSteps);
    }
  });
});

describe("optimizers", () => {
  const recipe = { text: harbor, epochs: 4, batchSize: 16, seed: 1, trace: true } as const;

  it("moves every weight by about the learning rate on Adam's first step with bias correction", () => {
    const run = trainTinyExtended({ ...recipe, learningRate: 0.3, optimizer: { kind: "adam" } });
    expect(run.trace[0].maxUpdate).toBeGreaterThan(0.3 * 0.999);
    expect(run.trace[0].maxUpdate).toBeLessThanOrEqual(0.3 + 1e-6);
  });

  it("moves Adam's first step about 3.16 times further without bias correction", () => {
    const corrected = trainTinyExtended({ ...recipe, learningRate: 0.3, optimizer: { kind: "adam" } });
    const raw = trainTinyExtended({ ...recipe, learningRate: 0.3, optimizer: { kind: "adam", biasCorrection: false } });
    // (1 − β1) / √(1 − β2) = 0.1 / √0.001 ≈ 3.16 for a weight whose first gradient dominates epsilon.
    expect(raw.trace[0].maxUpdate / corrected.trace[0].maxUpdate).toBeCloseTo(0.1 / Math.sqrt(0.001), 2);
  });

  it("decouples AdamW's decay from the gradient scaling", () => {
    const coupled = trainTinyExtended({ ...recipe, learningRate: 0.1, weightDecay: 0.1, optimizer: { kind: "adam" } });
    const decoupled = trainTinyExtended({ ...recipe, learningRate: 0.1, weightDecay: 0.1, optimizer: { kind: "adamw" } });
    const norm = (weights: Float32Array) => Math.hypot(...weights);
    expect(norm(decoupled.weights)).toBeGreaterThan(norm(coupled.weights));
  });

  it("counts optimizer state: none, one, two and one float per weight", () => {
    const state = (kind: "sgd" | "momentum" | "adamw" | "muon") =>
      trainTinyExtended({ ...recipe, epochs: 1, optimizer: { kind } }).stateFloats;
    expect([state("sgd"), state("momentum"), state("adamw"), state("muon")]).toEqual([0, 900, 1800, 900]);
  });

  it("trains with every optimizer to a loss well below chance", () => {
    for (const [kind, learningRate] of [["momentum", 0.3], ["adamw", 0.03], ["muon", 0.05]] as const) {
      const run = trainTinyExtended({ ...recipe, epochs: 12, learningRate, optimizer: { kind } });
      expect(run.diverged, kind).toBe(false);
      expect(run.finalLoss, kind).toBeLessThan(2.4);
    }
  });
});

describe("Newton–Schulz", () => {
  it("pushes singular values into a band around one and keeps a zero direction at zero", () => {
    // A diagonal matrix keeps its singular vectors on the axes, so the output's diagonal is its singular values.
    const rows = 6;
    const cols = 4;
    const singular = [5, 1, 0.2, 0];
    const source = new Float64Array(rows * cols);
    singular.forEach((value, index) => (source[index * cols + index] = value));
    const out = newtonSchulz(source, rows, cols, 5);
    for (let r = 0; r < rows; r += 1) {
      for (let c = 0; c < cols; c += 1) {
        const value = out[r * cols + c];
        if (r === c && c < 3) {
          expect(value).toBeGreaterThan(0.6);
          expect(value).toBeLessThan(1.2);
        } else {
          expect(Math.abs(value)).toBeLessThan(1e-12);
        }
      }
    }
  });

  it("returns the normalized input unchanged with zero iterations", () => {
    const out = newtonSchulz([3, 0, 0, 4], 2, 2, 0);
    expect(Array.from(out).map((value) => Number(value.toFixed(6)))).toEqual([0.6, 0, 0, 0.8]);
  });
});

describe("half precision and loss scaling", () => {
  it("rounds to half precision and overflows past 65504", () => {
    expect(toHalf(65504)).toBe(65504);
    expect(toHalf(65520)).toBe(Infinity);
    expect(toHalf(-1e6)).toBe(-Infinity);
    expect(toHalf(1 + 2 ** -12)).toBe(1);
    expect(toHalf(2 ** -26)).toBe(0);
  });

  it("halts on a static overflow and survives a dynamic one", () => {
    const options = { text: harbor, epochs: 4, learningRate: 0.6, seed: 1 } as const;
    const fixed = trainTinyExtended({ ...options, faults: [{ kind: "loss-scale", scale: 2 ** 20 }] });
    expect(fixed.diverged).toBe(true);
    expect(fixed.haltedAt).not.toBeNull();
    const dynamic = trainTinyExtended({ ...options, faults: [{ kind: "loss-scale", scale: 2 ** 20, dynamic: true }] });
    expect(dynamic.diverged).toBe(false);
    expect(dynamic.overflowSteps).toBeGreaterThan(0);
    expect(dynamic.skippedSteps).toBe(dynamic.overflowSteps);
    expect(Number.isFinite(dynamic.finalLoss)).toBe(true);
  });
});

describe("faults, guards and the trace", () => {
  const options = { text: harbor, epochs: 6, learningRate: 0.3, seed: 1, trace: true, optimizer: { kind: "adamw" } } as const;

  it("spikes the batch loss on a garbage batch", () => {
    const clean = trainTinyExtended(options);
    const faulty = trainTinyExtended({ ...options, faults: [{ kind: "bad-batch", atStep: 100, count: 3, text: "zqxj vkzq jxqz qzvx" }] });
    expect(faulty.trace[100].faulty).toBe(true);
    expect(faulty.trace[100].loss).toBeGreaterThan(clean.trace[100].loss + 1);
  });

  it("skips exactly the requested steps and leaves earlier steps untouched", () => {
    const clean = trainTinyExtended(options);
    const skipped = trainTinyExtended({ ...options, skipSteps: [[50, 53]] });
    expect(skipped.skippedSteps).toBe(3);
    expect(skipped.trace.slice(0, 50)).toEqual(clean.trace.slice(0, 50));
    expect(skipped.trace[51].skipped).toBe(true);
    expect(skipped.trace[51].maxUpdate).toBe(0);
  });

  it("pulls the log-partition toward zero with z-loss", () => {
    const plain = trainTinyExtended(options);
    const regularized = trainTinyExtended({ ...options, zLoss: 0.01 });
    const tail = (run: ReturnType<typeof trainTinyExtended>) =>
      run.trace.slice(-20).reduce((sum, point) => sum + point.logPartition, 0) / 20;
    expect(Math.abs(tail(regularized))).toBeLessThan(Math.abs(tail(plain)));
  });

  it("counts row visits and movement on the table", () => {
    const run = trainTinyExtended({ ...options, optimizer: { kind: "sgd" } });
    const space = 0;
    expect(run.rowVisits[space]).toBeGreaterThan(0);
    expect(run.rowMovement[space]).toBeGreaterThan(0);
  });
});

describe("schedules and mixtures", () => {
  it("leaves every existing schedule to scheduleFactor and shapes wsd", () => {
    for (const schedule of ["constant", "cosine", "warmup-decay", "one-cycle"] as const) {
      expect(extendedScheduleFactor(schedule, 0.4, 0.1)).toBe(scheduleFactor(schedule, 0.4, 0.1));
    }
    expect(extendedScheduleFactor("wsd", 0.05, 0.1, 0.2)).toBeCloseTo(0.5, 12);
    expect(extendedScheduleFactor("wsd", 0.5, 0.1, 0.2)).toBe(1);
    expect(extendedScheduleFactor("wsd", 0.9, 0.1, 0.2)).toBeCloseTo(0.5, 12);
    expect(extendedScheduleFactor("wsd", 1, 0.1, 0.2)).toBe(0);
  });

  it("draws batches in the mixture's proportions and switches phase on time", () => {
    const sources = [
      { id: "harbor", text: harbor },
      { id: "recipes", text: recipes },
    ];
    const even = trainTinyExtended({ text: harbor, sources, steps: 200, mixture: [{ from: 0, weights: [3, 1] }] });
    const share = even.sourceSeen[0] / (even.sourceSeen[0] + even.sourceSeen[1]);
    expect(share).toBeGreaterThan(0.7);
    expect(share).toBeLessThan(0.8);
    const switched = trainTinyExtended({
      text: harbor,
      sources,
      steps: 200,
      mixture: [
        { from: 0, weights: [1, 0] },
        { from: 0.5, weights: [0, 1] },
      ],
    });
    expect(switched.sourceSeen[0]).toBeGreaterThan(1500);
    expect(switched.sourceSeen[1]).toBeGreaterThan(1500);
    expect(switched.steps).toBe(200);
  });

  it("learns the mixed sources, measured on the text passed in", () => {
    const run = trainTinyExtended({
      text: recipes,
      sources: [{ id: "recipes", text: recipes }],
      steps: 300,
      learningRate: 1,
    });
    expect(run.finalLoss).toBeLessThan(tinyCrossEntropy(new Float32Array(900), recipes) - 0.5);
  });
});
