import { describe, expect, it } from "vitest";
import {
  TINY_CORPORA,
  TINY_VOCAB,
  tinyCrossEntropy,
  tinyNextDistribution,
  trainTinyLora,
  trainTinyModel,
  TINY_VOCAB_SIZE,
} from "@app/module-sdk";
import { centerRows, checkMerge, energyCaptured, numericalRank, singularValues } from "./lowrank";

/**
 * Pins the measured values the lora-adapters lesson and card text quote
 * (module.ts, content/standard.mdx, content/plain.mdx, card-info.ts).
 *
 * `adapter` and `fullTune` mirror Explore.tsx exactly: the base is harbor, 60 epochs, seed 1; the adapter is
 * trainTinyLora on recipes (batch 16, learning rate 0.6, seed 5) and the full fine-tune is trainTinyModel from
 * the base weights with the same epochs, learning rate and seed 5. The defaults are rank 4, alpha 8, 24 epochs.
 */
const V = TINY_VOCAB_SIZE;
const source = TINY_CORPORA.harbor;
const target = TINY_CORPORA.recipes;
const base = trainTinyModel({ text: source.text, epochs: 60, seed: 1 });
const targetBase = tinyCrossEntropy(base.weights, target.text);
const sourceBase = tinyCrossEntropy(base.weights, source.text);
const round = (value: number, digits = 3) => Number(value.toFixed(digits));

const adapter = (rank: number, alpha: number, epochs: number) =>
  trainTinyLora({ base: base.weights, text: target.text, rank, alpha, epochs, batchSize: 16, learningRate: 0.6, seed: 5 });
const fullTune = (epochs: number) =>
  trainTinyModel({ text: target.text, epochs, learningRate: 0.6, init: base.weights, seed: 5 });

/** The centred full-fine-tune spectrum the energy chart draws. */
function fullSpectrum(epochs: number) {
  const run = fullTune(epochs);
  const raw = new Float64Array(V * V);
  for (let index = 0; index < raw.length; index += 1) raw[index] = run.weights[index] - base.weights[index];
  return { run, values: singularValues(centerRows(raw, V), V, V) };
}
const loraSpectrum = (run: ReturnType<typeof adapter>) => singularValues(centerRows(run.delta, V), V, V);
const share = (probability: number) => Number((probability * 100).toFixed(0));

const defaults = adapter(4, 8, 24);

describe("lora-adapters quoted values", () => {
  it("matches the base model and the defaults", () => {
    // standard.mdx:11-12, plain.mdx:11-12, card-info.ts:89 (1.921): base scores 1.921 on Harbor, 2.556 on Recipe steps.
    // standard.mdx:19-21 and plain.mdx:25-27 call this the identical 60-epoch, seed 1 Harbor table that Fine-tuning &
    // transfer starts from; fine-tuning-transfer/quoted.test.ts pins the same 1.921 and 2.556 for it.
    expect(round(sourceBase)).toBe(1.921);
    expect(round(targetBase)).toBe(2.556);
    // standard.mdx:39-41, plain.mdx:44-45: at the defaults (rank 4, alpha 8) the scale is 2.00
    expect(round(defaults.scale, 2)).toBe(2);
    // standard.mdx:68 and plain.mdx:70: "Set alpha = rank" gives a scale of 1.00 at every rank
    for (const rank of [1, 4, 12]) expect(adapter(rank, rank, 0).scale).toBe(1);
    expect(defaults.fullParameters).toBe(900);
    // standard.mdx:55-56, plain.mdx:58-59, card-info.ts:19: 60, 120, 240, 480 and 720 trainable weights (module.ts:27: "against the full 900")
    expect([1, 2, 4, 8, 12].map((rank) => adapter(rank, 8, 0).trainableParameters)).toEqual([60, 120, 240, 480, 720]);
    // standard.mdx:97 and card-info.ts:19: at rank 15, 60 x 15 would equal the full 900
    expect(adapter(15, 8, 0).trainableParameters).toBe(900);
    // card-info.ts:90: Serving weights reads 1140 at rank 4 unmerged (900 + 240), and 900 once merged or detached
    expect(defaults.fullParameters + defaults.trainableParameters).toBe(1140);
  });

  it("matches the headline: 240 trained weights reach 2.077, full fine-tuning of all 900 reaches 2.090", () => {
    // standard.mdx:49-51 and :72, plain.mdx:53-54 and :74-75, card-info.ts:27 ("2.077 against 2.090")
    expect(round(defaults.finalLoss)).toBe(2.077);
    expect(round(tinyCrossEntropy(defaults.merged, target.text))).toBe(2.077);
    expect(defaults.trainableParameters).toBe(240);
    const full = fullTune(24);
    expect(round(tinyCrossEntropy(full.weights, target.text))).toBe(2.09);
  });

  it("matches the untrained adapter: Target loss 2.556, B and the update blank, every probe pair matching", () => {
    const untrained = adapter(4, 8, 0);
    // standard.mdx:58-59, plain.mdx:61-63, card-info.ts:57,91: epochs 0 leaves the table exactly the base
    expect(round(tinyCrossEntropy(untrained.merged, target.text))).toBe(2.556);
    expect(untrained.up.every((value) => value === 0)).toBe(true);
    expect(untrained.delta.every((value) => value === 0)).toBe(true);
    expect(untrained.merged).toEqual(base.weights);
    for (const probe of [" ", "e", "s", "l", "a"]) {
      expect(tinyNextDistribution(untrained.merged, probe)).toEqual(tinyNextDistribution(base.weights, probe));
    }
    // card-info.ts:80,91: at 0 epochs the merge gap is exactly 0
    expect(checkMerge(untrained.merged, base.weights, untrained.up, untrained.down, 4, untrained.scale, V, target.text).maxLogitGap).toBe(0);
  });

  it("matches the rank sweep at 24 epochs and alpha 8: 2.462, 2.316, 2.077, 1.990, 2.059", () => {
    // standard.mdx:62-63 and :77, plain.mdx:78-79: recipe loss for ranks 1, 2, 4, 8 and 12
    const sweep = [1, 2, 4, 8, 12].map((rank) => adapter(rank, 8, 24));
    expect(sweep.map((run) => round(tinyCrossEntropy(run.merged, target.text)))).toEqual([2.462, 2.316, 2.077, 1.99, 2.059]);
    // standard.mdx:77-78 and plain.mdx:79-80: "rank 1 has scale 8.00 and rank 12 has 0.67"
    expect(round(sweep[0].scale, 2)).toBe(8);
    expect(round(sweep[4].scale, 2)).toBe(0.67);
    // standard.mdx:77-78: "Rank 12 lands at 2.059, worse than rank 8 at 1.990": more rank is not more fit
    expect(tinyCrossEntropy(sweep[4].merged, target.text)).toBeGreaterThan(tinyCrossEntropy(sweep[3].merged, target.text));
  });

  it("matches the serving metrics: Target 2.556 to 2.077, Source 1.921 to 2.235", () => {
    // standard.mdx:66-67 and plain.mdx:68-69
    expect(round(targetBase)).toBe(2.556);
    expect(round(tinyCrossEntropy(defaults.merged, target.text))).toBe(2.077);
    expect(round(sourceBase)).toBe(1.921);
    expect(round(tinyCrossEntropy(defaults.merged, source.text))).toBe(2.235);
    // standard.mdx:83, plain.mdx:81, card-info.ts:89: detaching restores the base exactly; the merge never wrote the base
    const before = Float32Array.from(base.weights);
    adapter(4, 8, 24);
    expect(base.weights).toEqual(before);
  });

  it("forgets more than the full fine-tune: Harbor 2.235 against 2.088", () => {
    // standard.mdx:79-80 and plain.mdx:96-97: base 1.921, full fine-tuning 2.088, rank-4 adapter 2.235
    const full = fullTune(24);
    expect(round(tinyCrossEntropy(full.weights, source.text))).toBe(2.088);
    expect(round(tinyCrossEntropy(defaults.merged, source.text))).toBe(2.235);
    expect(tinyCrossEntropy(defaults.merged, source.text)).toBeGreaterThan(tinyCrossEntropy(full.weights, source.text));
  });

  it("matches the spectra: the full update holds 63% in 4 directions and 86% in 8; LoRA reaches 100% at k = r", () => {
    const { values } = fullSpectrum(24);
    // standard.mdx:74-75, plain.mdx:76-77, card-info.ts:27
    expect(Math.round(energyCaptured(values, 4) * 100)).toBe(63);
    expect(Math.round(energyCaptured(values, 8) * 100)).toBe(86);
    // card-info.ts:23: "the dashed one needs 11 directions to pass 95%"
    const kFor95 = Array.from({ length: V }, (_, index) => energyCaptured(values, index + 1)).findIndex((value) => value > 0.95) + 1;
    expect(kFor95).toBe(11);
    // standard.mdx:27-28, plain.mdx:32-33, card-info.ts:23: "The solid curve always reaches 100% at k = r" for rank 1 to 12
    for (let rank = 1; rank <= 12; rank += 1) {
      const lora = loraSpectrum(adapter(rank, 8, 24));
      expect(numericalRank(lora)).toBe(rank);
      expect(Math.round(energyCaptured(lora, rank) * 100)).toBe(100);
    }
    // standard.mdx:74-75: the full fine-tune's update is spread out (not low rank), unlike the adapter's
    expect(numericalRank(values)).toBeGreaterThan(12);
  });

  it("matches the Next character probes after `s` and after the space", () => {
    // standard.mdx:63-64, plain.mdx:65-66, card-info.ts:54: after `s` the base gives `a` 5% and the adapter 29%
    const a = TINY_VOCAB.indexOf("a");
    expect(share(tinyNextDistribution(base.weights, "s")[a])).toBe(5);
    expect(share(tinyNextDistribution(defaults.merged, "s")[a])).toBe(29);
    // card-info.ts:58: the space row barely moves, `t` 22% to 18%
    const t = TINY_VOCAB.indexOf("t");
    expect(share(tinyNextDistribution(base.weights, " ")[t])).toBe(22);
    expect(share(tinyNextDistribution(defaults.merged, " ")[t])).toBe(18);
    // card-info.ts:58 ("such as `s`, `l` and `e`, move most"): each of them moves more than the space row.
    const movement = (probe: string) => {
      const before = tinyNextDistribution(base.weights, probe);
      const after = tinyNextDistribution(defaults.merged, probe);
      return before.reduce((sum, value, index) => sum + (value > 1e-9 ? value * Math.log(value / after[index]) : 0), 0);
    };
    for (const probe of ["s", "l", "e"]) expect(movement(probe)).toBeGreaterThan(movement(" "));
    // card-info.ts:58 now says "Of the five probe rows, `l` moves most, then `s` and `a`; `e` and the space row move
    // least": KL(base || adapted) is l 1.603, s 0.600, a 0.574, e 0.294, space 0.095.
    expect([round(movement("l")), round(movement("s")), round(movement("a")), round(movement("e")), round(movement(" "))]).toEqual([
      1.603, 0.6, 0.574, 0.294, 0.095,
    ]);
  });

  it("matches the merge check: about 4e-7 per logit at rank 4, same loss to four decimals", () => {
    const check = checkMerge(defaults.merged, base.weights, defaults.up, defaults.down, 4, defaults.scale, V, target.text);
    // standard.mdx:84-85, plain.mdx:82-83, card-info.ts:80: "about 4e-7", float32 rounding
    expect(check.maxLogitGap).toBeGreaterThan(3.5e-7);
    expect(check.maxLogitGap).toBeLessThan(4.5e-7);
    // "the recipe loss agrees to four decimals"
    expect(round(check.unmergedLoss, 4)).toBe(round(tinyCrossEntropy(defaults.merged, target.text), 4));
  });
});
