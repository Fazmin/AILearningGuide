import { describe, expect, it } from "vitest";
import { bigramPairs, encodeTinyText, toHalf, trainTinyExtended, trainTinyFactored, TINY_CORPORA } from "@app/module-sdk";
import {
  adamCleanOptions,
  describeRecipe,
  FAULTS,
  GARBAGE_COUNT,
  GARBAGE_FIRST_STEP,
  GARBAGE_TEXT,
  halfWindow,
  HALF_MAX,
  HALF_OVERFLOW,
  healthyOptions,
  NUMBER_FORMATS,
  originalOptions,
  PAIRS,
  PLANNED_STEPS,
  recipeOf,
  referenceLabel,
  referenceOptions,
  runOptions,
  TRAINING_TEXT,
  type FaultSettings,
} from "./faults";
import { defaultState } from "./state";

const settings = (patch: Partial<FaultSettings>): FaultSettings => ({ ...defaultState, ...patch });

describe("the shared model and data", () => {
  it("trains on the six Harbor sentences Train one from scratch trains on", () => {
    const sentences = TINY_CORPORA.harbor.text.split(".").map((part) => part.trim()).filter(Boolean);
    expect(sentences).toHaveLength(8);
    expect(TRAINING_TEXT).toBe(`${sentences.slice(0, 6).join(". ")}.`);
    expect(encodeTinyText(TRAINING_TEXT)).toHaveLength(379);
    expect(PAIRS).toBe(378);
    expect(bigramPairs(TRAINING_TEXT).inputs.length).toBe(378);
    // 24 batches of 16 per epoch, the last holding 10, for 20 epochs.
    expect(Math.ceil(PAIRS / 16)).toBe(24);
    expect(PAIRS - 23 * 16).toBe(10);
    expect(PLANNED_STEPS).toBe(480);
  });

  it("uses the rank-8 factored model with 480 weights for every fault", () => {
    for (const fault of FAULTS) {
      const options = runOptions(settings({ fault: fault.id }));
      expect(options.model).toEqual({ kind: "factored", rank: 8 });
      expect([options.epochs, options.batchSize, options.seed, options.trace]).toEqual([20, 16, 9, true]);
    }
    expect(trainTinyExtended({ ...healthyOptions(), epochs: 0 }).parameters).toBe(480);
  });

  it("reproduces Train one from scratch: plain SGD on this model matches trainTinyFactored bit for bit", () => {
    const shared = { text: TRAINING_TEXT, rank: 8, epochs: 20, batchSize: 16, seed: 9 };
    const healthy = trainTinyExtended(healthyOptions());
    const original = trainTinyFactored({ ...shared, learningRate: 0.5 });
    expect(Array.from(healthy.weights)).toEqual(Array.from(original.weights));
    const explosion = trainTinyFactored({ ...shared, learningRate: 8 });
    expect([explosion.diverged, explosion.steps]).toEqual([true, 154]);
    expect(trainTinyExtended(runOptions(settings({ fault: "rate" }))).haltedAt).toBe(154);
  });
});

describe("fault options", () => {
  it("builds the healthy run as plain SGD at 0.5 with a constant schedule", () => {
    const options = runOptions(settings({ fault: "none", clipNorm: 1, zLoss: 0.01 }));
    expect(options).toEqual(healthyOptions());
    expect(options.learningRate).toBe(0.5);
    expect(options.optimizer).toEqual({ kind: "sgd" });
    // Fixes for other faults never leak into this run.
    expect(options.clipNorm ?? 0).toBe(0);
    expect(options.zLoss ?? 0).toBe(0);
  });

  it("raises the rate to 8 and applies the clip and the one-cycle shape only to that fault", () => {
    const constant = runOptions(settings({ fault: "rate", clipNorm: 0.5 }));
    expect([constant.learningRate, constant.clipNorm, constant.schedule]).toEqual([8, 0.5, "constant"]);
    const warm = runOptions(settings({ fault: "rate", schedule: "warmup-cosine" }));
    expect([warm.schedule, warm.warmupFraction]).toEqual(["one-cycle", 0.15]);
    expect(runOptions(settings({ fault: "fp16", clipNorm: 0.5 })).clipNorm ?? 0).toBe(0);
    expect(runOptions(settings({ fault: "logits", clipNorm: 0.5 })).clipNorm ?? 0).toBe(0);
  });

  it("casts the healthy run's gradients through fp16 at the chosen scale", () => {
    const fixed = runOptions(settings({ fault: "fp16", lossScale: 20 }));
    expect(fixed.faults).toEqual([{ kind: "loss-scale", scale: 2 ** 20, dynamic: false, growthInterval: 200 }]);
    expect({ ...fixed, faults: undefined }).toEqual({ ...healthyOptions(), faults: undefined });
    const dynamic = runOptions(settings({ fault: "fp16", lossScale: 0, scaling: "dynamic" }));
    expect(dynamic.faults).toEqual([{ kind: "loss-scale", scale: 1, dynamic: true, growthInterval: 200 }]);
  });

  it("injects five garbage batches into the Adam run at steps 400 to 404, and skips exactly those after a rollback", () => {
    expect([GARBAGE_FIRST_STEP, GARBAGE_COUNT, GARBAGE_TEXT]).toEqual([400, 5, "qqqq zzzz xxxx jjjj"]);
    const faulty = runOptions(settings({ fault: "bad-batch", clipNorm: 1 }));
    expect(faulty.faults).toEqual([{ kind: "bad-batch", atStep: 399, count: 5, text: GARBAGE_TEXT }]);
    expect(faulty.skipSteps).toEqual([]);
    expect([faulty.learningRate, faulty.clipNorm, faulty.optimizer]).toEqual([0.1, 1, { kind: "adam" }]);
    const replay = runOptions(settings({ fault: "bad-batch", rollback: true }));
    expect(replay.skipSteps).toEqual([[399, 404]]);
    // The flag does nothing outside its own fault.
    expect(runOptions(settings({ fault: "rate", rollback: true }))).toEqual(runOptions(settings({ fault: "rate" })));
  });

  it("marks the garbage batches in the trace, and only those", () => {
    const run = trainTinyExtended(runOptions(settings({ fault: "bad-batch" })));
    const faulty = run.trace.filter((point) => point.faulty).map((point) => point.step + 1);
    expect(faulty).toEqual([400, 401, 402, 403, 404]);
  });

  it("runs Adam with no weight decay for Logit growth, at the chosen rate and z-loss", () => {
    const options = runOptions(settings({ fault: "logits", adamRate: 0.3, zLoss: 0.001 }));
    expect([options.learningRate, options.zLoss, options.optimizer, options.weightDecay ?? 0]).toEqual([0.3, 0.001, { kind: "adam" }, 0]);
  });

  it("names a clean reference run for every fault but None", () => {
    expect(referenceOptions(settings({ fault: "none" }))).toBeNull();
    expect(referenceOptions(settings({ fault: "rate", clipNorm: 0.5 }))).toEqual(healthyOptions());
    expect(referenceOptions(settings({ fault: "fp16" }))).toEqual(healthyOptions());
    expect(referenceOptions(settings({ fault: "bad-batch", clipNorm: 1 }))).toEqual(adamCleanOptions(1));
    expect(referenceOptions(settings({ fault: "logits", zLoss: 0.01 }))).toEqual(adamCleanOptions(0));
    expect(referenceLabel(settings({ fault: "logits" }))).toBe("Adam at 0.1 with no z-loss");
  });

  it("draws the original run behind the replay only after a rollback", () => {
    expect(originalOptions(settings({ fault: "bad-batch" }))).toBeNull();
    expect(originalOptions(settings({ fault: "logits", rollback: true }))).toBeNull();
    expect(originalOptions(settings({ fault: "bad-batch", rollback: true, clipNorm: 1 }))).toEqual(
      runOptions(settings({ fault: "bad-batch", clipNorm: 1 })),
    );
  });

  it("describes each recipe in words", () => {
    expect(describeRecipe(recipeOf(settings({ fault: "none" })))).toBe("SGD at 0.5 · constant");
    expect(describeRecipe(recipeOf(settings({ fault: "rate", clipNorm: 0.5, schedule: "warmup-cosine" })))).toBe(
      "SGD at 8 · warmup + cosine · clip 0.50",
    );
    expect(describeRecipe(recipeOf(settings({ fault: "fp16", scaling: "dynamic" })))).toBe(
      "SGD at 0.5 · constant · fp16 gradients, dynamic loss scale from 2²⁰",
    );
    expect(describeRecipe(recipeOf(settings({ fault: "logits", zLoss: 0.0001 })))).toBe("Adam at 1 · constant · z-loss 0.0001");
  });
});

describe("number formats", () => {
  it("matches the SDK's half-precision rounding at both ends", () => {
    expect(toHalf(HALF_MAX)).toBe(HALF_MAX);
    expect(toHalf(HALF_OVERFLOW - 0.001)).toBe(HALF_MAX);
    expect(toHalf(HALF_OVERFLOW)).toBe(Infinity);
    expect(toHalf(2 ** -25)).toBe(2 ** -24);
    expect(toHalf(2 ** -25 * 0.999)).toBe(0);
  });

  it("states each format's range from its bit layout", () => {
    const byId = Object.fromEntries(NUMBER_FORMATS.map((format) => [format.id, format]));
    expect(byId.fp16.largest).toBe(65504);
    expect(byId.fp16.smallestNormal).toBeCloseTo(6.1e-5, 6);
    expect(byId.fp16.smallestSubnormal).toBeCloseTo(5.96e-8, 9);
    expect(byId.fp32.largest).toBeCloseTo(3.4028e38, -34);
    expect(byId.bf16.largest).toBeCloseTo(3.3895e38, -34);
    expect(byId.bf16.smallestNormal).toBe(byId.fp32.smallestNormal);
    expect([byId.fp32.fractionBits, byId.bf16.fractionBits, byId.fp16.fractionBits]).toEqual([23, 7, 10]);
    expect([byId.fp32.exponentBits, byId.bf16.exponentBits, byId.fp16.exponentBits]).toEqual([8, 8, 5]);
  });

  it("moves the survival window down by the scale", () => {
    const at20 = halfWindow(20);
    expect(at20.scale).toBe(1048576);
    expect(at20.flushBelow).toBeCloseTo(2.842e-14, 16);
    expect(at20.overflowFrom).toBeCloseTo(0.0625, 3);
    expect(halfWindow(17).overflowFrom).toBeCloseTo(0.5, 3);
    expect(halfWindow(18).overflowFrom).toBeCloseTo(0.25, 3);
    expect(halfWindow(0).flushBelow).toBeCloseTo(2.98e-8, 9);
  });
});
