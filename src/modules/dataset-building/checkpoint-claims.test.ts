import { describe, expect, it } from "vitest";
import { tinyCrossEntropy, trainTinyModel } from "@app/module-sdk";
import { RAW_LINES, runPipeline, VALIDATION_TEXT, type PipelineOptions } from "./clean";

/**
 * Pins the shuffle-seed claim the fourth checkpoint question and the lesson's "Where it breaks"
 * make: across seeds the raw-versus-cleaned gap stays clearly positive, while the gap between the
 * two best filter settings is smaller than a hundredth of a nat and changes sign. The lab itself
 * fixes seed 4, so these are the only places the other seeds are run.
 */
const defaults: PipelineOptions = {
  dropBoilerplate: true,
  dropExact: true,
  dropNear: false,
  decontaminate: false,
  leak: false,
  minLength: 0,
};
const kept = (options: PipelineOptions) =>
  runPipeline(options)
    .filter((entry) => entry.verdict === "kept")
    .map((entry) => entry.line)
    .join(" ");
const heldOut = (text: string, seed: number) =>
  tinyCrossEntropy(trainTinyModel({ text, epochs: 25, batchSize: 16, learningRate: 0.6, seed }).weights, VALIDATION_TEXT);

describe("dataset-building checkpoint claims", () => {
  const seeds = [1, 2, 3, 4, 5, 6, 7, 8];
  const rows = seeds.map((seed) => {
    const raw = heldOut(RAW_LINES.join(" "), seed);
    const cleaned = heldOut(kept(defaults), seed);
    const exactOnly = heldOut(kept({ ...defaults, dropBoilerplate: false }), seed);
    return { seed, gap: raw - cleaned, close: exactOnly - cleaned };
  });

  it("keeps the raw-to-cleaned gap positive at every seed from 1 to 8, between 0.065 and 0.085", () => {
    for (const row of rows) {
      expect(row.gap, `seed ${row.seed}`).toBeGreaterThanOrEqual(0.065 - 0.0005);
      expect(row.gap, `seed ${row.seed}`).toBeLessThanOrEqual(0.085 + 0.0005);
    }
  });

  it("makes the gap between the two best filter settings smaller than 0.01 nats, with a sign that flips", () => {
    for (const row of rows) expect(Math.abs(row.close), `seed ${row.seed}`).toBeLessThan(0.01);
    expect(rows.some((row) => row.close > 0)).toBe(true);
    expect(rows.some((row) => row.close < 0)).toBe(true);
    // The lab's own seed (4) is the one the lesson quotes: keeping only exact-duplicate removal looks best there.
    expect(rows.find((row) => row.seed === 4)!.close).toBeLessThan(0);
  });
});
