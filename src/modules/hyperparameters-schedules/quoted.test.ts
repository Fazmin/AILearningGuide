import { describe, expect, it } from "vitest";
import { scheduleFactor, TINY_CORPORA, trainTinyModel, type TinySchedule } from "@app/module-sdk";

/** Pins the measured values the hyperparameters lesson and card text quote. */
const text = TINY_CORPORA.harbor.text;
const final = (schedule: TinySchedule, learningRate: number, extra: { batchSize?: number; clipNorm?: number; weightDecay?: number } = {}) =>
  trainTinyModel({ text, epochs: 25, batchSize: 16, learningRate, schedule, warmupFraction: 0.15, seed: 1, ...extra });

describe("hyperparameters quoted values", () => {
  it("flips the winner as the peak rises", () => {
    expect(final("constant", 0.6).finalLoss).toBeCloseTo(2.075, 3);
    expect(final("cosine", 0.6).finalLoss).toBeCloseTo(2.255, 3);
    expect(final("constant", 6).finalLoss).toBeCloseTo(1.88, 3);
    expect(final("cosine", 6).finalLoss).toBeCloseTo(1.846, 3);
    expect(final("constant", 30).finalLoss).toBeCloseTo(3.02, 3);
    expect(final("cosine", 30).finalLoss).toBeCloseTo(1.802, 3);
  });

  it("crosses the uniform baseline between a peak of 35 and 38", () => {
    expect(final("constant", 35).finalLoss).toBeCloseTo(3.332, 3);
    expect(final("constant", 38).finalLoss).toBeCloseTo(3.537, 3);
    expect(final("constant", 40).finalLoss).toBeCloseTo(3.687, 3);
    expect(final("constant", 40, { clipNorm: 0.4 }).finalLoss).toBeCloseTo(3.186, 3);
  });

  it("matches the batch-size, clipping, and decay figures", () => {
    const tiny = final("constant", 30, { batchSize: 2 });
    expect(tiny.finalLoss).toBeCloseTo(12.325, 3);
    expect(final("constant", 30, { batchSize: 2, clipNorm: 0.5 }).finalLoss).toBeCloseTo(6.522, 3);
    expect(final("constant", 30, { clipNorm: 0.5 }).finalLoss).toBeCloseTo(2.884, 3);
    expect(final("constant", 30, { batchSize: 64 }).peakGradientNorm).toBeCloseTo(0.18, 2);
    expect(final("constant", 0.6, { weightDecay: 0.02 }).finalLoss).toBeCloseTo(2.664, 3);
  });

  it("matches the inspected warmup point", () => {
    // Warmup + decay at the defaults: 800 steps, epoch 12 is step 384.
    expect(Math.round(0.15 * 799)).toBe(120);
    expect(0.6 * scheduleFactor("warmup-decay", 384 / 799, 0.15)).toBeCloseTo(0.378, 3);
  });

  it("matches the arithmetic in the Toy versus real callout", () => {
    // Llama 2: 2.0 trillion tokens at a global batch of 4 million tokens, 2,000 warmup steps (Touvron et al., Table 1).
    const llamaUpdates = 2.0e12 / 4e6;
    expect(llamaUpdates).toBe(500_000);
    expect(2000 / llamaUpdates).toBeCloseTo(0.004, 6);
    // GPT-3: linear warmup over the first 375 million of 300 billion tokens (Brown et al., Appendix B).
    expect(375e6 / 300e9).toBeCloseTo(0.00125, 8);
    // The lab's own default warmup: 15 percent of 800 steps.
    expect(final("warmup-decay", 0.6).steps).toBe(800);
    expect(Math.round(0.15 * 799)).toBe(120);
    expect(scheduleFactor("cosine", 1)).toBeCloseTo(0.03, 12);
  });
});
