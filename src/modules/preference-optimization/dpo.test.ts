import { describe, expect, it } from "vitest";
import { TINY_CORPORA, trainTinyDpo, trainTinyModel } from "@app/module-sdk";
import { TINY_VOCAB, TINY_VOCAB_SIZE } from "@app/module-sdk";
import {
  dpoLoss,
  gradientWeight,
  HELD_OUT_PAIRS,
  heldOutGapTrace,
  marginForLoss,
  MAX_FIELD_LENGTH,
  MAX_LINE_LENGTH,
  MAX_PAIRS,
  negLogSigmoid,
  parsePairs,
  sanitizePairsText,
  scorePair,
  sftLoss,
  trainSftReference,
} from "./dpo";

const DEFAULTS = [
  "the | fog settles over the harbor. | fog fog fog fog fog.",
  "warm the | pan and add a spoon of oil. | pan and the pan and the pan.",
  "simmer the | sauce until it thickens. | sauce and simmer the sauce.",
].join("\n");

describe("parsing comparisons", () => {
  it("keeps a space between prompt and completion and drops malformed lines", () => {
    const pairs = parsePairs(`${DEFAULTS}\nno bars here\n | only | two`);
    expect(pairs).toHaveLength(4);
    expect(pairs[0]).toEqual({ prompt: "the ", chosen: "fog settles over the harbor.", rejected: "fog fog fog fog fog." });
    expect(pairs[3].prompt).toBe("");
  });
});

describe("the DPO loss curve", () => {
  it("starts at ln 2 with half-strength gradients", () => {
    expect(dpoLoss(0.4, 0)).toBeCloseTo(Math.LN2, 12);
    expect(gradientWeight(0.4, 0)).toBeCloseTo(0.5, 12);
  });

  it("is stable at extreme margins and invertible", () => {
    expect(negLogSigmoid(800)).toBeCloseTo(0, 12);
    expect(negLogSigmoid(-800)).toBeCloseTo(800, 9);
    const margin = marginForLoss(0.4, 0.1);
    expect(dpoLoss(0.4, margin)).toBeCloseTo(0.1, 10);
    // Doubling beta halves the margin needed for the same loss.
    expect(marginForLoss(0.8, 0.1)).toBeCloseTo(margin / 2, 10);
  });
});

describe("against the SDK trainer", () => {
  const reference = trainTinyModel({ text: `${TINY_CORPORA.harbor.text} ${TINY_CORPORA.recipes.text}`, epochs: 50, seed: 1 });
  const pairs = parsePairs(DEFAULTS);

  it("reproduces the trainer's averaged beta-scaled margin and loss", () => {
    const run = trainTinyDpo({ reference: reference.weights, pairs, beta: 0.4, steps: 40, learningRate: 0.35 });
    const scores = pairs.map((pair) => scorePair(run.weights, reference.weights, pair));
    const meanMargin = scores.reduce((sum, score) => sum + score.margin, 0) / scores.length;
    expect(0.4 * meanMargin).toBeCloseTo(run.finalMargin, 6);
    expect(scores.every((score) => score.margin > 0)).toBe(true);
    // The rejected side moves further than the chosen side at the defaults.
    for (const score of scores) expect(-score.rejectedShift).toBeGreaterThan(score.chosenShift);
  });

  it("shows beta's two roles: bigger early steps, smaller final movement", () => {
    const drift = (beta: number, steps: number) =>
      trainTinyDpo({ reference: reference.weights, pairs, beta, steps, learningRate: 0.35 }).finalDrift;
    // Early on the gradient is proportional to beta, so higher beta moves further.
    expect(drift(0.8, 4)).toBeGreaterThan(drift(0.2, 4));
    // Once the loss saturates, higher beta needs a smaller log-ratio gap and drifts less.
    expect(drift(0.8, 40)).toBeLessThan(drift(0.2, 40));
  });
});

describe("untrusted pair text", () => {
  it("keeps the default pairs exactly", () => {
    expect(sanitizePairsText(DEFAULTS)).toBe(DEFAULTS);
    expect(parsePairs(sanitizePairsText(DEFAULTS))).toHaveLength(3);
  });

  it("caps the number of pairs and the length of every part", () => {
    const huge = Array.from({ length: 5000 }, () => `${"p".repeat(500)} | ${"c".repeat(500)} | ${"r".repeat(500)}`).join("\n");
    const text = sanitizePairsText(huge);
    expect(text.split("\n")).toHaveLength(MAX_PAIRS);
    expect(Math.max(...text.split("\n").map((line) => line.length))).toBe(MAX_LINE_LENGTH);
    const pairs = parsePairs(huge);
    expect(pairs).toHaveLength(MAX_PAIRS);
    for (const pair of pairs) {
      expect(pair.chosen.length).toBeLessThanOrEqual(MAX_FIELD_LENGTH);
      expect(pair.rejected.length).toBeLessThanOrEqual(MAX_FIELD_LENGTH);
    }
  });
});

describe("held-out pairs", () => {
  it("are three well-formed comparisons that are not the default training pairs", () => {
    expect(HELD_OUT_PAIRS).toHaveLength(3);
    const training = parsePairs(DEFAULTS);
    for (const held of HELD_OUT_PAIRS) {
      expect(held.chosen).not.toBe(held.rejected);
      expect(training.some((pair) => pair.chosen === held.chosen || pair.rejected === held.rejected)).toBe(false);
    }
  });
});

describe("the SFT-style reference", () => {
  const base = trainTinyModel({ text: `${TINY_CORPORA.harbor.text} ${TINY_CORPORA.recipes.text}`, epochs: 50, seed: 1 }).weights;
  const pairs = parsePairs(DEFAULTS);

  it("lowers the response-only loss on the preferred completions and never writes the base", () => {
    const snapshot = Float32Array.from(base);
    const sft = trainSftReference(base, pairs);
    expect(base).toEqual(snapshot);
    expect(sftLoss(sft, pairs)).toBeLessThan(sftLoss(base, pairs));
  });

  it("returns an exact copy of the base with no pairs or no steps", () => {
    expect(trainSftReference(base, [])).toEqual(base);
    expect(trainSftReference(base, pairs, 0)).toEqual(base);
    expect(trainSftReference(base, [])).not.toBe(base);
  });

  it("grades only the completion: rows that appear only inside the prompt are untouched", () => {
    // The prompt's last character (a space) conditions the first reply character; "x", "y" and "z" appear only earlier in the prompt.
    const sft = trainSftReference(base, parsePairs("xyz | ab. | cd."));
    for (const letter of "xyz") {
      const row = TINY_VOCAB.indexOf(letter) * TINY_VOCAB_SIZE;
      expect(Array.from(sft.slice(row, row + TINY_VOCAB_SIZE))).toEqual(Array.from(base.slice(row, row + TINY_VOCAB_SIZE)));
    }
    // The space row and the completion's own rows did move.
    const space = 0;
    expect(Array.from(sft.slice(space, TINY_VOCAB_SIZE))).not.toEqual(Array.from(base.slice(space, TINY_VOCAB_SIZE)));
  });

  it("is a valid reference: DPO from it starts at ln 2 with every margin zero", () => {
    const sft = trainSftReference(base, pairs);
    const run = trainTinyDpo({ reference: sft, pairs, beta: 0.4, steps: 0 });
    expect(run.history[0].loss).toBeCloseTo(Math.LN2, 12);
    expect(run.history[0].margin).toBe(0);
  });
});

describe("held-out margins along one pass", () => {
  const reference = trainTinyModel({ text: `${TINY_CORPORA.harbor.text} ${TINY_CORPORA.recipes.text}`, epochs: 50, seed: 1 });
  const pairs = parsePairs(DEFAULTS);

  it("reproduces the SDK trainer's held-out gaps at every step count it is asked for", () => {
    const marks = [0, 4, 10, 40, 80];
    const trace = heldOutGapTrace({
      reference: reference.weights,
      pairs,
      heldOut: HELD_OUT_PAIRS,
      beta: 0.4,
      learningRate: 0.35,
      marks,
    });
    marks.forEach((steps, index) => {
      const run = trainTinyDpo({ reference: reference.weights, pairs, beta: 0.4, steps, learningRate: 0.35 });
      HELD_OUT_PAIRS.forEach((pair, held) => {
        expect(trace[index][held]).toBeCloseTo(scorePair(run.weights, reference.weights, pair).margin, 3);
      });
    });
    expect(trace[0]).toEqual([0, 0, 0]);
  });

  it("handles no pairs and no held-out pairs", () => {
    expect(heldOutGapTrace({ reference: reference.weights, pairs: [], heldOut: HELD_OUT_PAIRS, beta: 0.4, learningRate: 0.35, marks: [0, 5] })).toEqual([[0, 0, 0], [0, 0, 0]]);
    expect(heldOutGapTrace({ reference: reference.weights, pairs, heldOut: [], beta: 0.4, learningRate: 0.35, marks: [0, 5] })).toEqual([[], []]);
  });
});
