import { describe, expect, it } from "vitest";
import { TINY_CORPORA, tinyCrossEntropy, trainTinyFactored, trainTinyModel, type TinySchedule } from "@app/module-sdk";
import {
  BOILERPLATE_LINES,
  buildScrape,
  cleanScrape,
  encodedLength,
  scrapeText,
  splitCorpus as moduleSplit,
} from "./data";
import { RATE_LADDER } from "./state";

/** Pins the measured values the train-tiny-lm lesson and card text quote. Mirrors splitCorpus in Explore. */
function splitCorpus(text: string) {
  const sentences = text.split(".").map((part) => part.trim()).filter(Boolean);
  const held = Math.max(1, Math.round(sentences.length * 0.25));
  return {
    train: `${sentences.slice(0, sentences.length - held).join(". ")}.`,
    held: `${sentences.slice(sentences.length - held).join(". ")}.`,
  };
}
const recipe = { batchSize: 16, learningRate: 0.5, seed: 9 } as const;
const factored = (corpus: string, rank: number, epochs: number) => {
  const split = splitCorpus(TINY_CORPORA[corpus].text);
  const run = trainTinyFactored({ text: split.train, rank, epochs, ...recipe, checkpoints: epochs + 1 });
  return { run, split, held: run.checkpoints.map((point) => tinyCrossEntropy(point.weights, split.held)) };
};

describe("train-tiny-lm quoted values", () => {
  it("matches the rank sweep at 20 epochs", () => {
    const rows = [1, 4, 6, 8, 12].map((rank) => {
      const { run, held } = factored("harbor", rank, 20);
      return [rank, Number(run.finalLoss.toFixed(3)), Number(held.at(-1)!.toFixed(3)), Number((held.at(-1)! - run.finalLoss).toFixed(3))];
    });
    expect(rows).toEqual([
      [1, 2.925, 3.025, 0.1],
      [4, 2.208, 2.246, 0.038],
      [6, 2.128, 2.222, 0.095],
      [8, 2.039, 2.117, 0.077],
      [12, 2.038, 2.165, 0.128],
    ]);
  });

  it("matches the full table and the rank-1 plateau", () => {
    const split = splitCorpus(TINY_CORPORA.harbor.text);
    const full20 = trainTinyModel({ text: split.train, epochs: 20, ...recipe });
    const full30 = trainTinyModel({ text: split.train, epochs: 30, ...recipe });
    expect(full20.finalLoss).toBeCloseTo(2.283, 3);
    expect(tinyCrossEntropy(full20.weights, split.held)).toBeCloseTo(2.302, 3);
    const rankOne = factored("harbor", 1, 30).run;
    expect(rankOne.finalLoss).toBeCloseTo(2.852, 3);
    expect(rankOne.finalLoss - full30.finalLoss).toBeCloseTo(0.688, 3);
  });

  it("shows overfitting on Proverbs at rank 12 and 30 epochs", () => {
    const { run, held } = factored("proverbs", 12, 30);
    const best = Math.min(...held);
    expect(best).toBeCloseTo(2.668, 3);
    expect(held.indexOf(best)).toBe(18);
    expect(held.at(-1)).toBeCloseTo(2.818, 3);
    expect(run.finalLoss).toBeCloseTo(1.865, 3);
  });
});

/* -------------------------------------------------------------------------- */
/* The Training recipe card                                                    */
/* -------------------------------------------------------------------------- */

/** Mirrors the two runs Explore.tsx trains: the same options, with the recipe card's settings. */
const recipeRun = (
  text: string,
  options: { rate?: number; schedule?: TinySchedule; batch?: number; clip?: number; rank?: number; epochs?: number } = {},
) => {
  const { rate = 0.5, schedule = "constant", batch = 16, clip = 0, rank = 8, epochs = 20 } = options;
  const shared = { text, epochs, batchSize: batch, learningRate: rate, schedule, clipNorm: clip, seed: 9, checkpoints: epochs + 1 };
  return { factored: trainTinyFactored({ ...shared, rank }), full: trainTinyModel(shared) };
};
const harbor = moduleSplit(TINY_CORPORA.harbor.text);
const heldOut = (weights: Float32Array) => tinyCrossEntropy(weights, harbor.held);
const r3 = (value: number) => Number(value.toFixed(3));

describe("train-tiny-lm recipe card values", () => {
  it("keeps the default recipe identical to the lab before the recipe card existed", () => {
    const { factored, full } = recipeRun(harbor.train);
    expect(r3(factored.finalLoss)).toBe(2.039);
    expect(r3(heldOut(factored.weights))).toBe(2.117);
    expect(r3(full.finalLoss)).toBe(2.283);
    expect(factored.steps).toBe(480);
    expect(factored.peakGradientNorm.toFixed(2)).toBe("0.95");
    expect(full.peakGradientNorm.toFixed(2)).toBe("0.34");
  });

  it("matches the schedule comparison at a rate of 0.5 and at a rate of 4", () => {
    const held = (rate: number, schedule: TinySchedule) => r3(heldOut(recipeRun(harbor.train, { rate, schedule }).factored.weights));
    expect([held(0.5, "constant"), held(0.5, "cosine")]).toEqual([2.117, 2.251]);
    expect([held(4, "constant"), held(4, "cosine")]).toEqual([3.016, 2.054]);
  });

  it("matches the explosion at a rate of 8 and what clipping does to it", () => {
    const constant = recipeRun(harbor.train, { rate: 8 }).factored;
    expect(constant.diverged).toBe(true);
    expect(constant.steps).toBe(154);
    const clipped = recipeRun(harbor.train, { rate: 8, clip: 0.5 }).factored;
    expect(clipped.diverged).toBe(false);
    expect([clipped.clippedSteps, clipped.steps]).toEqual([459, 480]);
    expect(r3(heldOut(clipped.weights))).toBe(3.089);
    // The same rate is far from fine without the clip, even when the schedule decays.
    const cosine = recipeRun(harbor.train, { rate: 8, schedule: "cosine" }).factored;
    expect(r3(heldOut(cosine.weights))).toBe(22.245);
    expect(cosine.peakGradientNorm.toExponential(1)).toBe("1.2e+12");
    const cosineClipped = recipeRun(harbor.train, { rate: 8, schedule: "cosine", clip: 0.5 }).factored;
    expect(r3(heldOut(cosineClipped.weights))).toBe(2.065);
    // The full table never explodes: its batch gradient norm is bounded by the square root of 2.
    const full = recipeRun(harbor.train, { rate: 8 }).full;
    expect(full.diverged).toBe(false);
    expect(r3(heldOut(full.weights))).toBe(2.066);
    expect(full.peakGradientNorm.toFixed(2)).toBe("0.34");
  });

  it("bounds the full table's gradient norm by the square root of 2 at every rate, schedule and batch", () => {
    for (const rate of RATE_LADDER) {
      for (const schedule of ["constant", "cosine", "warmup-decay", "one-cycle"] as const) {
        for (const batch of [2, 16, 64]) {
          const { full } = recipeRun(harbor.train, { rate, schedule, batch, epochs: 3 });
          expect(full.peakGradientNorm, `rate ${rate} ${schedule} batch ${batch}`).toBeLessThanOrEqual(Math.SQRT2);
        }
      }
    }
  });

  it("matches the batch-size sweep: steps fall in proportion and the largest batch ends highest", () => {
    const rows = [8, 16, 64].map((batch) => {
      const { factored } = recipeRun(harbor.train, { batch });
      return [batch, factored.steps, r3(heldOut(factored.weights))];
    });
    expect(rows).toEqual([
      [8, 960, 2.089],
      [16, 480, 2.117],
      [64, 120, 2.618],
    ]);
  });

  it("builds the harbor scrape the card describes and cleans it back to the training sentences", () => {
    const scrape = buildScrape(harbor.sentences);
    const cleaning = cleanScrape(scrape);
    expect(scrape).toHaveLength(14);
    expect(encodedLength(scrapeText(scrape))).toBe(696);
    expect(cleaning.removed).toEqual({ boilerplate: 3, short: 2, duplicate: 2, near: 1 });
    expect(cleaning.kept).toHaveLength(6);
    expect(encodedLength(cleaning.text)).toBe(379);
    expect(cleaning.text).toBe(harbor.train);
  });

  it("shows the raw scrape adding steps and footer knowledge but not a clearly worse held-out loss", () => {
    const footers = BOILERPLATE_LINES.join(" ");
    const scrape = buildScrape(harbor.sentences);
    const raw = recipeRun(scrapeText(scrape)).factored;
    const cleaned = recipeRun(cleanScrape(scrape).text).factored;
    expect([raw.steps, cleaned.steps]).toEqual([880, 480]);
    expect([r3(heldOut(raw.weights)), r3(heldOut(cleaned.weights))]).toEqual([2.105, 2.117]);
    expect([r3(tinyCrossEntropy(raw.weights, footers)), r3(tinyCrossEntropy(cleaned.weights, footers))]).toEqual([2.509, 3.09]);
  });

  it("holds the raw-scrape claim on every corpus: footers drop by over half a nat while held-out barely moves", () => {
    const footers = BOILERPLATE_LINES.join(" ");
    for (const id of ["harbor", "recipes", "proverbs"]) {
      const split = moduleSplit(TINY_CORPORA[id].text);
      const scrape = buildScrape(split.sentences);
      const raw = recipeRun(scrapeText(scrape)).factored;
      const cleaned = recipeRun(cleanScrape(scrape).text).factored;
      const heldDifference = tinyCrossEntropy(raw.weights, split.held) - tinyCrossEntropy(cleaned.weights, split.held);
      const footerDrop = tinyCrossEntropy(cleaned.weights, footers) - tinyCrossEntropy(raw.weights, footers);
      expect(Math.abs(heldDifference), id).toBeLessThan(0.05);
      expect(footerDrop, id).toBeGreaterThan(0.5);
    }
  });

  it("changes the sign of the raw-minus-cleaned held-out difference from seed to seed on Harbor", () => {
    const scrape = buildScrape(harbor.sentences);
    const raw = scrapeText(scrape);
    const cleaned = cleanScrape(scrape).text;
    const differences = [1, 2, 3, 4].map((seed) => {
      const run = (text: string) =>
        trainTinyFactored({ text, rank: 8, epochs: 20, batchSize: 16, learningRate: 0.5, seed });
      return heldOut(run(raw).weights) - heldOut(run(cleaned).weights);
    });
    expect(differences.some((value) => value > 0)).toBe(true);
    expect(differences.some((value) => value < 0)).toBe(true);
  });

  it("never lets any learning rate or schedule, with or without a clip, close the rank-1 gap with the full table", { timeout: 60_000 }, () => {
    // The first checkpoint question offers a better rate as a way out of the rank-1 plateau.
    let smallest = Infinity;
    const settings = [
      ...(["constant", "cosine", "warmup-decay", "one-cycle"] as const).map((schedule) => ({ schedule, clip: 0 })),
      { schedule: "cosine" as const, clip: 0.5 },
    ];
    for (const rate of RATE_LADDER) {
      for (const { schedule, clip } of settings) {
        const { factored, full } = recipeRun(harbor.train, { rate, schedule, clip, rank: 1, epochs: 30 });
        if (!Number.isFinite(factored.finalLoss) || !Number.isFinite(full.finalLoss)) continue;
        smallest = Math.min(smallest, factored.finalLoss - full.finalLoss);
      }
    }
    // Under the same recipe the full table stays at least a third of a nat ahead of rank 1.
    expect(smallest).toBeGreaterThan(0.3);
  });
});
