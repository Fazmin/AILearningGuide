import { describe, expect, it } from "vitest";
import {
  TINY_CORPORA,
  TINY_VOCAB,
  tinyNextDistribution,
  tinyPerplexity,
  tinyTopTokens,
  trainTinyModel,
} from "@app/module-sdk";

/** Pins the measured values the training-run lesson and card text quote. */
const text = TINY_CORPORA.harbor.text;
const run = (batchSize: number, epochs = 12) =>
  trainTinyModel({ text, epochs, batchSize, learningRate: 0.6, seed: 1, checkpoints: 9 });

describe("training-run quoted values", () => {
  it("matches the default run", () => {
    const defaults = run(16);
    expect(defaults.steps).toBe(384);
    expect(defaults.finalLoss).toBeCloseTo(2.277, 3);
    expect(tinyPerplexity(defaults.weights, text)).toBeCloseTo(9.75, 2);
    expect(defaults.checkpoints.map((point) => Number(point.loss.toFixed(3)))).toEqual([
      3.401, 3.068, 2.816, 2.641, 2.53, 2.438, 2.374, 2.32, 2.277,
    ]);
    expect(defaults.updatedParameters / 30).toBe(26);
    expect(defaults.peakGradientNorm).toBeCloseTo(0.749, 3);
    const top = tinyTopTokens(defaults.weights, "the fo", 1)[0];
    expect(top.token).toBe("r");
    expect(top.probability).toBeCloseTo(0.235, 3);
  });

  it("matches the batch-size and epoch sweeps", () => {
    const small = run(2);
    expect(small.steps).toBe(2988);
    expect(small.finalLoss).toBeCloseTo(1.877, 3);
    expect(small.peakGradientNorm).toBeCloseTo(1.088, 3);
    const large = run(64);
    expect(large.steps).toBe(96);
    expect(large.finalLoss).toBeCloseTo(2.808, 3);
    expect(run(16, 1).finalLoss).toBeCloseTo(3.166, 3);
    expect(run(16, 60).finalLoss).toBeCloseTo(1.921, 3);
  });

  it("matches the Backward strip for the pair o → g", () => {
    const defaults = run(16);
    const g = TINY_VOCAB.indexOf("g");
    const start = tinyNextDistribution(defaults.checkpoints[0].weights, "the fo");
    const end = tinyNextDistribution(defaults.weights, "the fo");
    expect(start[g] - 1).toBeCloseTo(-0.967, 3);
    expect(end[g] - 1).toBeCloseTo(-0.957, 3);
  });

  it("matches the arithmetic in the Toy versus real callout", () => {
    // Llama 2 reports 2.0 trillion tokens at a global batch of 4 million tokens (Touvron et al., Table 1).
    expect(2.0e12 / 4e6).toBe(500_000);
    // The lab's own model: 900 weights at 4 bytes each is 3.6 KB.
    expect(30 * 30 * 4).toBe(3600);
    expect(defaultsSteps()).toBe(384);
  });
});

function defaultsSteps() {
  return run(16).steps;
}
