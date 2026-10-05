import { describe, expect, it } from "vitest";
import { TINY_CORPORA, tinyCrossEntropy, trainTinyLora, trainTinyModel } from "@app/module-sdk";

/**
 * Pins the claim in the second checkpoint question's explanation: the rank sweep at Alpha 8 mixes
 * two changes (rank and the alpha / rank scale), and once Set alpha = rank holds the scale at 1.00
 * the extra directions of rank 12 pay off, so rank 12 lands below rank 8 instead of above it.
 */
const base = trainTinyModel({ text: TINY_CORPORA.harbor.text, epochs: 60, seed: 1 });
const target = TINY_CORPORA.recipes;
const recipeLoss = (rank: number, alpha: number) =>
  tinyCrossEntropy(
    trainTinyLora({
      base: base.weights,
      text: target.text,
      rank,
      alpha,
      epochs: 24,
      batchSize: 16,
      learningRate: 0.6,
      seed: 5,
    }).merged,
    target.text,
  );

describe("lora-adapters checkpoint claims", () => {
  it("reverses the rank 8 versus rank 12 order once alpha equals rank", () => {
    // At Alpha 8 the scale is 1.00 at rank 8 and 0.67 at rank 12, and rank 12 fits worse.
    expect(recipeLoss(12, 8)).toBeGreaterThan(recipeLoss(8, 8));
    // At alpha = rank the scale is 1.00 at both, and rank 12 fits better.
    expect(recipeLoss(12, 12)).toBeLessThan(recipeLoss(8, 8));
  });
});
