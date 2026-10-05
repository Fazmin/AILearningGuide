import { describe, expect, it } from "vitest";
import { TINY_SCHEDULES, UNIFORM_CROSS_ENTROPY } from "@app/module-sdk";
import { EPOCHS, SEED_SET, seedVerdict, summarizeSeeds, trainRecipe, type Recipe } from "./seeds";

/**
 * Pins every figure the Five seeds band, its card text, and the lesson quote. Seed 1 is the lab's own
 * seed, so the first column of every table here agrees with quoted.test.ts.
 */
const base: Recipe = { peakLearningRate: 0.6, warmup: 0.15, clipNorm: 0, weightDecay: 0, batchSize: 16 };
const summaries = (recipe: Recipe, schedule: (typeof TINY_SCHEDULES)[number]["value"]) =>
  summarizeSeeds(SEED_SET.map((seed) => trainRecipe(recipe, schedule, seed, false)));
const atPeak = (peakLearningRate: number) => ({ ...base, peakLearningRate });

describe("seed band", () => {
  it("uses the lab's seed for the first column and trains five distinct seeds", () => {
    expect(SEED_SET).toEqual([1, 2, 3, 4, 5]);
    const cosine = summaries(base, "cosine");
    expect(cosine.finals[0]).toBeCloseTo(trainRecipe(base, "cosine", 1, true).finalLoss, 12);
    expect(new Set(cosine.finals).size).toBe(5);
    expect(cosine.band).toHaveLength(EPOCHS + 1);
    // The band's last point is the lowest and highest final loss.
    expect(cosine.band[EPOCHS].low).toBeCloseTo(cosine.min, 12);
    expect(cosine.band[EPOCHS].high).toBeCloseTo(cosine.max, 12);
    // Every seed starts from the same zeroed weights, so epoch 0 has no spread.
    expect(cosine.band[0].low).toBe(cosine.band[0].high);
  });

  it("matches the default-peak table: three decaying schedules cannot be told apart, Constant is clearly ahead", () => {
    const [constant, cosine, warmupDecay, oneCycle] = TINY_SCHEDULES.map((entry) => summaries(base, entry.value));
    expect(constant.finals[0]).toBeCloseTo(2.075, 3);
    expect(constant.min).toBeCloseTo(2.075, 3);
    expect(constant.max).toBeCloseTo(2.077, 3);
    expect(constant.spread).toBeCloseTo(0.0024, 4);
    expect(cosine.mean).toBeCloseTo(2.2547, 4);
    expect(warmupDecay.mean).toBeCloseTo(2.2514, 4);
    expect(oneCycle.mean).toBeCloseTo(2.2545, 4);
    expect(cosine.spread).toBeCloseTo(0.0036, 4);
    expect(warmupDecay.spread).toBeCloseTo(0.0037, 4);
    expect(oneCycle.spread).toBeCloseTo(0.0037, 4);
    // The three decaying schedules' means sit within 0.004 of one another, as the lesson says.
    expect(Math.max(cosine.mean, warmupDecay.mean, oneCycle.mean) - Math.min(cosine.mean, warmupDecay.mean, oneCycle.mean)).toBeLessThan(0.004);

    const verdict = seedVerdict(TINY_SCHEDULES.map((entry, index) => ({ label: entry.label, summary: [constant, cosine, warmupDecay, oneCycle][index] })));
    expect(verdict.top).toMatchObject({ better: "Constant", other: "Warmup + decay", outsideBand: true });
    expect(verdict.top!.gap).toBeCloseTo(0.1753, 4);
    expect(verdict.top!.noise).toBeCloseTo(0.0037, 4);
    expect(verdict.top!.gap / verdict.top!.noise).toBeGreaterThan(40);
    expect(verdict.top!.gap / verdict.top!.noise).toBeLessThan(55);
    expect(verdict.closest).toMatchObject({ better: "One cycle", other: "Cosine decay", outsideBand: false });
    expect(verdict.closest!.gap).toBeCloseTo(0.0002, 4);
    expect(verdict.closest!.gap).toBeLessThan(verdict.closest!.noise);
  });

  it("keeps the order of Constant and Cosine decay at a peak of 6 for every seed, though Constant's band is wide", () => {
    const constant = summaries(atPeak(6), "constant");
    const cosine = summaries(atPeak(6), "cosine");
    expect(constant.finals[0]).toBeCloseTo(1.88, 3);
    expect(constant.min).toBeCloseTo(1.8805, 4);
    expect(constant.max).toBeCloseTo(2.0084, 4);
    expect(constant.spread).toBeCloseTo(0.128, 3);
    expect(cosine.max).toBeCloseTo(1.8475, 4);
    expect(constant.min).toBeGreaterThan(cosine.max);
  });

  it("shows Constant's band widening with the peak rate while the decaying schedules stay tight", () => {
    const constant12 = summaries(atPeak(12), "constant");
    expect(constant12.min).toBeCloseTo(2.0653, 4);
    expect(constant12.max).toBeCloseTo(2.5552, 4);
    expect(constant12.spread).toBeCloseTo(0.49, 2);
    const constant30 = summaries(atPeak(30), "constant");
    expect(constant30.finals[0]).toBeCloseTo(3.02, 3);
    expect(constant30.min).toBeCloseTo(2.3953, 4);
    expect(constant30.max).toBeCloseTo(4.6734, 4);
    expect(constant30.spread).toBeCloseTo(2.278, 3);
    expect(constant30.mean).toBeCloseTo(3.3021, 4);
  }, 30_000);

  it("keeps the decaying schedules under 0.011 wide at every peak the lesson quotes", () => {
    for (const peak of [0.6, 6, 12, 30]) {
      for (const schedule of ["cosine", "warmup-decay", "one-cycle"] as const) {
        expect(summaries(atPeak(peak), schedule).spread, `${schedule} at ${peak}`).toBeLessThan(0.011);
      }
    }
  }, 60_000);

  it("counts how many seeds end worse than chance as the peak rate rises", () => {
    const worse = (peak: number) => summaries(atPeak(peak), "constant").finals.filter((loss) => loss > UNIFORM_CROSS_ENTROPY).length;
    expect({ 30: worse(30), 35: worse(35), 38: worse(38), 40: worse(40), 60: worse(60) }).toEqual({ 30: 2, 35: 2, 38: 4, 40: 4, 60: 5 });
  }, 60_000);

  it("shows the batch-2 clip rescue is larger than the seed band", () => {
    const recipe = { ...base, peakLearningRate: 30, batchSize: 2 };
    const unclipped = summaries(recipe, "constant");
    const clipped = summaries({ ...recipe, clipNorm: 0.5 }, "constant");
    expect(unclipped.max).toBeCloseTo(12.325, 3);
    expect(unclipped.min).toBeCloseTo(9.854, 3);
    expect(clipped.max).toBeCloseTo(6.522, 3);
    expect(clipped.min).toBeCloseTo(6.168, 3);
    expect(clipped.max).toBeLessThan(unclipped.min);
  }, 30_000);

  it("is the clip's rescue at batch 16 that is not larger than its band", () => {
    const recipe = { ...base, peakLearningRate: 30, clipNorm: 0.5 };
    const clipped = summaries(recipe, "constant");
    const unclipped = summaries({ ...recipe, clipNorm: 0 }, "constant");
    expect(clipped.finals[0]).toBeCloseTo(2.884, 3);
    expect(clipped.min).toBeCloseTo(2.308, 3);
    expect(clipped.max).toBeCloseTo(3.606, 3);
    // The two ranges overlap, so seed 1's improvement alone does not show a rescue at this batch size.
    expect(clipped.max).toBeGreaterThan(unclipped.min);
  }, 30_000);

  it("treats a diverged run as unranked and handles an empty verdict", () => {
    const run = trainRecipe(base, "cosine", 1, false);
    const diverged = { ...run, finalLoss: Number.NaN, checkpoints: run.checkpoints.map((point) => ({ ...point, loss: Number.NaN })) };
    const summary = summarizeSeeds([diverged, run]);
    expect(summary.mean).toBe(Number.POSITIVE_INFINITY);
    expect(Number.isFinite(summary.band[1].low)).toBe(true);
    expect(seedVerdict([{ label: "Only", summary }])).toEqual({ top: null, closest: null });
  });

  it("stays fast enough to run inside a render: twenty runs at the slowest setting", () => {
    const started = performance.now();
    for (const entry of TINY_SCHEDULES) summaries({ ...base, batchSize: 2 }, entry.value);
    // About a second on a quiet machine; the ceiling is generous so a busy machine does not flake.
    expect(performance.now() - started).toBeLessThan(15_000);
  }, 30_000);
});
