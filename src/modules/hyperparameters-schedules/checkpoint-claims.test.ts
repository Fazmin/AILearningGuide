import { describe, expect, it } from "vitest";
import { TINY_CORPORA, trainTinyModel, type TinySchedule } from "@app/module-sdk";

/**
 * Pins the claim in the second checkpoint question: Warmup fraction changes only the Warmup + decay
 * and One cycle curves; Constant and Cosine decay ignore it. Settings are the lab's (25 epochs,
 * batch 16, seed 1, peak 0.6).
 */
const text = TINY_CORPORA.harbor.text;
const finalLoss = (schedule: TinySchedule, warmupFraction: number) =>
  trainTinyModel({ text, epochs: 25, batchSize: 16, learningRate: 0.6, schedule, warmupFraction, seed: 1 }).finalLoss;

describe("hyperparameters checkpoint claims", () => {
  it("moves only the two schedules that warm up when the warmup fraction changes", () => {
    for (const schedule of ["constant", "cosine"] as const) {
      expect(finalLoss(schedule, 0.1)).toBe(finalLoss(schedule, 0.5));
    }
    for (const schedule of ["warmup-decay", "one-cycle"] as const) {
      expect(finalLoss(schedule, 0.1)).not.toBe(finalLoss(schedule, 0.5));
    }
  });

  it("lets a small gradient clip rescue the very-high-rate, batch-2 constant run without removing the damage", () => {
    const run = (clipNorm: number) =>
      trainTinyModel({ text, epochs: 25, batchSize: 2, learningRate: 30, schedule: "constant", seed: 1, clipNorm }).finalLoss;
    const unclipped = run(0);
    // The run is far worse than knowing nothing (ln 30), and a small clip lowers the loss at both caps the lesson uses.
    expect(unclipped).toBeGreaterThan(Math.log(30));
    expect(run(0.1)).toBeLessThan(unclipped);
    expect(run(0.5)).toBeLessThan(unclipped);
  });
});
