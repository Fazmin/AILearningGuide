import { describe, expect, it } from "vitest";
import { TINY_CORPORA, TINY_VOCAB_SIZE, tinySequenceLogProb, trainTinyDpo, trainTinyModel } from "@app/module-sdk";
import definition from "./module";
import { gradientWeight, HELD_OUT_PAIRS, heldOutGapTrace, marginForLoss, parsePairs, scorePair, trainSftReference } from "./dpo";

/**
 * Quote locations are given by file and section, not line number, because the lesson text is still being edited.
 * Pins the measured values the preference-optimization lesson (standard.mdx and plain.mdx) and card text
 * (card-info.ts) quote. Mirrors the Explore lab: the reference is 50 epochs on Harbor + Recipes at seed 1,
 * the pairs are the module's initial pairsText, and every DPO run starts from an exact copy of the reference.
 */
const referenceText = `${TINY_CORPORA.harbor.text} ${TINY_CORPORA.recipes.text}`;
const reference = trainTinyModel({ text: referenceText, epochs: 50, seed: 1 });
const pairs = parsePairs(definition.initialState.pairsText as string);
const { beta: BETA, steps: STEPS, learningRate: LEARNING_RATE } = definition.initialState as {
  beta: number;
  steps: number;
  learningRate: number;
};
const run = (beta = BETA, steps = STEPS) =>
  trainTinyDpo({ reference: reference.weights, pairs, beta, steps, learningRate: LEARNING_RATE });
const scored = (weights: Float32Array) => pairs.map((pair) => scorePair(weights, reference.weights, pair));
const round = (value: number, digits: number) => Number(value.toFixed(digits));

describe("preference-optimization quoted values", () => {
  it("uses the default lab state", () => {
    expect(pairs).toHaveLength(3);
    expect([BETA, STEPS, LEARNING_RATE]).toEqual([0.4, 40, 0.35]);
    // standard.mdx "Where it breaks" (900 logits, 30 context rows), card-info.ts "Drift from the reference model" (30 rows): a 30 x 30 table.
    expect(TINY_VOCAB_SIZE).toBe(30);
    expect(reference.weights).toHaveLength(900);
  });

  it("starts at ln 2 with every reward zero", () => {
    const untrained = run(BETA, 0);
    const [start] = untrained.history;
    // standard.mdx and plain.mdx "What it is", card-info.ts "Implicit reward margin during training" and "Preference pair editor", module.ts stepInstructions: 0.6931 = ln 2.
    expect(round(start.loss, 4)).toBe(0.6931);
    expect(start.loss).toBeCloseTo(Math.LN2, 12);
    expect(start.margin).toBe(0);
    expect(start.chosenShift).toBe(0);
    expect(start.rejectedShift).toBe(0);
    // card-info.ts "Implicit reward margin during training" notice: "At 0 steps the badge reads 0%".
    expect(untrained.accuracy).toBe(0);
  });

  it("matches the default run readouts", () => {
    const defaults = run();
    const last = defaults.history.at(-1)!;
    const scores = scored(defaults.weights);
    const meanGap = scores.reduce((sum, score) => sum + score.margin, 0) / scores.length;
    // standard.mdx and plain.mdx "How to play with it": the loss falls from 0.6931 to 0.0317.
    expect(last.loss).toBeCloseTo(0.0317, 4);
    // standard.mdx "How to play with it": beta-scaled margin 3.497 and Log-ratio gap 8.74 nats.
    expect(last.margin).toBeCloseTo(3.497, 3);
    expect(meanGap).toBeCloseTo(8.74, 2);
    // standard.mdx and plain.mdx "How to play with it", card-info.ts "Implicit reward margin during training" whatYouSee: all three pairs end correctly ordered (badge 100%).
    expect(defaults.accuracy).toBe(1);
    // standard.mdx and plain.mdx "How to play with it", card-info.ts "Implicit reward margin during training" controls: chosen reward +0.556, rejected -2.940.
    expect(last.chosenShift).toBeCloseTo(0.556, 3);
    expect(last.rejectedShift).toBeCloseTo(-2.94, 3);
    // standard.mdx and plain.mdx "What to notice", card-info.ts "Implicit reward margin during training" notice: the rejected side falls "about five times" further.
    expect(Math.round(-last.rejectedShift / last.chosenShift)).toBe(5);
  });

  it("matches where each pair sits on the loss", () => {
    const scores = scored(run().weights);
    // standard.mdx "How to play with it", card-info.ts "Where each pair sits on the loss" controls: the pairs sit at m = 9.50, 8.85 and 7.88.
    expect(scores.map((score) => round(score.margin, 2))).toEqual([9.5, 8.85, 7.88]);
    // card-info.ts "Where each pair sits on the loss" controls: pulls of 2.2%, 2.8% and 4.1%; standard.mdx and plain.mdx "How to play with it": "2 to 4%".
    const pulls = scores.map((score) => round(gradientWeight(BETA, score.margin) * 100, 1));
    expect(pulls).toEqual([2.2, 2.8, 4.1]);
    expect(Math.min(...pulls)).toBeGreaterThanOrEqual(1.5);
    expect(Math.max(...pulls)).toBeLessThan(4.5);
  });

  it("matches the per-pair rewards", () => {
    const rewards = scored(run().weights).map((score) => ({
      chosen: BETA * score.chosenShift,
      rejected: BETA * score.rejectedShift,
    }));
    // standard.mdx and plain.mdx "What to notice", card-info.ts "Per-pair rewards" controls: chosen rewards +0.18, +0.51 and +0.97.
    expect(rewards[0].chosen).toBeCloseTo(0.18, 2);
    // The lab computes 0.5146 (it displays +0.515). The lessons quote +0.51, not the double-rounded +0.52.
    expect(round(rewards[1].chosen, 2)).toBe(0.51);
    expect(rewards[2].chosen).toBeCloseTo(0.97, 2);
    // Same sources: rejected rewards -3.61, -3.03 and -2.18.
    expect(rewards.map((reward) => round(reward.rejected, 2))).toEqual([-3.61, -3.03, -2.18]);
    // standard.mdx "What to notice": "The rejected column moves further in every row".
    rewards.forEach((reward) => expect(Math.abs(reward.rejected)).toBeGreaterThan(Math.abs(reward.chosen)));
  });

  it("matches the margin each beta needs to reach a loss of 0.1", () => {
    // standard.mdx and plain.mdx "What to notice": m = 11.26 at beta 0.2, 5.63 at 0.4 and 2.82 at 0.8.
    expect([0.2, 0.4, 0.8].map((beta) => round(marginForLoss(beta, 0.1), 2))).toEqual([11.26, 5.63, 2.82]);
    // card-info.ts "Where each pair sits on the loss" howItWorks: "doubling beta halves the change needed to reach any given loss".
    expect(marginForLoss(0.8, 0.1) * 2).toBeCloseTo(marginForLoss(0.4, 0.1), 12);
  });

  it("matches the beta sweep at 40 steps", () => {
    const runs = [0.2, 0.4, 0.8].map((beta) => ({ beta, run: run(beta) }));
    // standard.mdx and plain.mdx "What to notice", card-info.ts "Drift from the reference model" controls: final KL 0.0133, 0.0102 and 0.0053.
    expect(runs.map((entry) => round(entry.run.finalDrift, 4))).toEqual([0.0133, 0.0102, 0.0053]);
    // standard.mdx "What to notice", card-info.ts "Drift from the reference model" notice: gaps of 10.2, 8.7 and 6.2 nats.
    expect(runs.map((entry) => round(entry.run.history.at(-1)!.margin / entry.beta, 1))).toEqual([10.2, 8.7, 6.2]);
    // standard.mdx "What to notice": the beta-scaled margin rises with beta, 2.045, 3.497 and 4.949.
    expect(runs.map((entry) => round(entry.run.finalMargin, 3))).toEqual([2.045, 3.497, 4.949]);
  });

  it("flips the drift order at 4 steps", () => {
    const drift = [0.2, 0.4, 0.8].map((beta) => run(beta, 4).finalDrift);
    // standard.mdx "What to notice", card-info.ts "Drift from the reference model" controls: 0.0005, 0.0013 and 0.0017, so higher beta drifts more.
    expect(drift.map((value) => round(value, 4))).toEqual([0.0005, 0.0013, 0.0017]);
    expect(drift[0]).toBeLessThan(drift[1]);
    expect(drift[1]).toBeLessThan(drift[2]);
  });

  it("matches the pair positions at 4 steps", () => {
    // card-info.ts "Where each pair sits on the loss" controls: "Set DPO steps to 4 and they sit near m = 3, still pulling at around 20%".
    const margins = scored(run(BETA, 4).weights).map((score) => score.margin);
    const pulls = margins.map((margin) => gradientWeight(BETA, margin));
    expect(Math.round(margins.reduce((sum, m) => sum + m, 0) / margins.length)).toBe(3);
    margins.forEach((margin) => expect(Math.abs(margin - 3)).toBeLessThan(1));
    expect(round(pulls.reduce((sum, p) => sum + p, 0) / pulls.length, 1)).toBe(0.2);
  });

  it("overshoots at beta 3.0", () => {
    // card-info.ts "Drift from the reference model" notice: "At beta 3.0 with these settings the steps are large enough to overshoot, and drift rises again."
    const drift = (beta: number) => run(beta).finalDrift;
    expect(drift(3)).toBeGreaterThan(drift(1.5));
    expect(drift(3)).toBeGreaterThan(drift(0.8));
  });

  it("matches the reference model's ordering of the default pairs", () => {
    const scores = scored(reference.weights);
    // standard.mdx "What it is" (comparisons), plain.mdx "What it is", card-info.ts "Per-pair rewards" notice: in all three the reference rates the loop as more likely.
    scores.forEach((score) => expect(score.referenceRejected).toBeGreaterThan(score.referenceChosen));
    // card-info.ts "Per-pair rewards" notice: "chosen completion is longer, 28 characters against 20".
    expect([pairs[0].chosen.length, pairs[0].rejected.length]).toEqual([28, 20]);
  });
});

/** module.ts checkpoint questions 1, 4 and 5 compare directions rather than quote numbers; these pin the directions. */
describe("preference-optimization checkpoint claims", () => {
  it("question 1: the loss starts at ln 2 although the base model prefers the loop in every default pair", () => {
    expect(run(BETA, 0).history[0].loss).toBeCloseTo(Math.LN2, 12);
    scored(reference.weights).forEach((score) => expect(score.referenceRejected).toBeGreaterThan(score.referenceChosen));
  });

  it("question 4: at 40 steps the scaled margin rises with beta while the log-ratio gap and the drift fall", () => {
    const runs = [0.2, 0.4, 0.8].map((beta) => run(beta));
    const scaled = runs.map((entry) => entry.finalMargin);
    const gaps = runs.map((entry, index) => entry.history.at(-1)!.margin / [0.2, 0.4, 0.8][index]);
    expect(scaled[0]).toBeLessThan(scaled[1]);
    expect(scaled[1]).toBeLessThan(scaled[2]);
    expect(gaps[0]).toBeGreaterThan(gaps[1]);
    expect(gaps[1]).toBeGreaterThan(gaps[2]);
    expect(runs[0].finalDrift).toBeGreaterThan(runs[2].finalDrift);
  });

  it("question 5: at 4 steps the order of drift is the reverse of the 40-step order", () => {
    const early = [0.2, 0.8].map((beta) => run(beta, 4).finalDrift);
    const late = [0.2, 0.8].map((beta) => run(beta, 40).finalDrift);
    expect(early[0]).toBeLessThan(early[1]);
    expect(late[0]).toBeGreaterThan(late[1]);
  });
});


/**
 * Held-out pairs and the SFT-style reference. Quote locations: standard.mdx and plain.mdx "How to play with it",
 * "What to notice" and "Going deeper"; card-info.ts "Implicit reward margin during training", "Per-pair rewards" and
 * "Preference pair editor"; module.ts checkpoint questions 6 and 7.
 */
describe("preference-optimization held-out pairs", () => {
  const heldGaps = (weights: Float32Array, referenceWeights = reference.weights) =>
    HELD_OUT_PAIRS.map((pair) => scorePair(weights, referenceWeights, pair).margin);

  it("orders every training pair but only one held-out pair at the defaults", () => {
    const defaults = run();
    expect(defaults.accuracy).toBe(1);
    const gaps = heldGaps(defaults.weights);
    // standard.mdx and plain.mdx "How to play with it": 1 of 3 held-out pairs ordered correctly.
    expect(gaps.filter((gap) => gap > 0)).toHaveLength(1);
    // standard.mdx "How to play with it" and "What to notice", card-info.ts "Per-pair rewards" notice: held-out margins -0.54, -1.12 and +2.90 as the rows show them (log-ratio gaps -1.35, -2.79 and +7.25 nats).
    expect(gaps.map((gap) => round(gap, 2))).toEqual([-1.35, -2.79, 7.25]);
    expect(gaps.map((gap) => round(BETA * gap, 2))).toEqual([-0.54, -1.12, 2.9]);
    // plain.mdx and standard.mdx: the training pairs average a beta-scaled margin of 3.497.
    expect(defaults.finalMargin).toBeCloseTo(3.497, 3);
    // card-info.ts "Implicit reward margin during training" controls: the held-out margin ends at +0.42 against +3.50.
    const heldOutMargin = (BETA * gaps.reduce((sum, gap) => sum + gap, 0)) / gaps.length;
    expect(round(heldOutMargin, 2)).toBe(0.42);
    expect(round(defaults.finalMargin, 2)).toBe(3.5);
  });

  it("gets worse on two held-out pairs, and better on one, as training runs longer", () => {
    // standard.mdx "What to notice", card-info.ts "Per-pair rewards" notice: more steps make the two reversed pairs worse.
    const byStep = [4, 10, 20, 40, 80, 120].map((steps) => heldGaps(run(BETA, steps).weights));
    for (let index = 1; index < byStep.length; index += 1) {
      expect(byStep[index][0]).toBeLessThan(byStep[index - 1][0]);
      expect(byStep[index][1]).toBeLessThan(byStep[index - 1][1]);
      expect(byStep[index][2]).toBeGreaterThan(byStep[index - 1][2]);
    }
    // At the default beta and learning rate, no even step count from 0 to 120 orders more than one of the three held-out pairs.
    const marks = Array.from({ length: 61 }, (_, index) => index * 2);
    const every = heldOutGapTrace({ reference: reference.weights, pairs, heldOut: HELD_OUT_PAIRS, beta: BETA, learningRate: LEARNING_RATE, marks });
    for (const gaps of every) expect(gaps.filter((gap) => gap > 0).length).toBeLessThanOrEqual(1);
  });

  it("carries the third held-out pair over from training pair 2 and barely from pair 1", () => {
    // standard.mdx "What to notice", card-info.ts "Per-pair rewards" notice: train on pair 2 alone and the third held-out margin is +2.25; on pair 1 alone, +0.28.
    const only = (index: number) =>
      trainTinyDpo({ reference: reference.weights, pairs: [pairs[index]], beta: BETA, steps: STEPS, learningRate: LEARNING_RATE });
    expect(round(BETA * heldGaps(only(1).weights)[2], 2)).toBe(2.25);
    expect(round(BETA * heldGaps(only(0).weights)[2], 2)).toBe(0.28);
    // The shared piece the lesson names: both loops contain " and the ".
    expect(pairs[1].rejected).toContain(" and the ");
    expect(HELD_OUT_PAIRS[2].rejected).toContain(" and the ");
  });

  it("starts at zero held-out margin and is not trained on the held-out pairs", () => {
    heldGaps(run(BETA, 0).weights).forEach((gap) => expect(gap).toBe(0));
    // The training pairs and the held-out pairs have no completion in common.
    const trained = new Set(pairs.flatMap((pair) => [pair.chosen, pair.rejected]));
    for (const held of HELD_OUT_PAIRS) {
      expect(trained.has(held.chosen)).toBe(false);
      expect(trained.has(held.rejected)).toBe(false);
    }
  });
});

describe("preference-optimization SFT-style reference", () => {
  const sft = trainSftReference(reference.weights, pairs);
  const prefers = (weights: Float32Array, list: ReadonlyArray<{ prompt: string; chosen: string; rejected: string }>) =>
    list.map((pair) => tinySequenceLogProb(weights, pair.prompt, pair.chosen) > tinySequenceLogProb(weights, pair.prompt, pair.rejected));

  it("makes the reference prefer the chosen completion on two of the three training pairs, where the base preferred none", () => {
    // standard.mdx and plain.mdx "How to play with it" and "Going deeper"; card-info.ts "Preference pair editor" controls; module.ts question 7.
    expect(prefers(reference.weights, pairs)).toEqual([false, false, false]);
    expect(prefers(sft, pairs)).toEqual([false, true, true]);
    // The first training pair keeps the shorter loop ahead: its chosen completion is 8 characters longer.
    expect(pairs[0].chosen.length - pairs[0].rejected.length).toBe(8);
  });

  it("still starts DPO at ln 2 and ends with every training pair ordered, while one held-out pair is", () => {
    const run40 = trainTinyDpo({ reference: sft, pairs, beta: BETA, steps: STEPS, learningRate: LEARNING_RATE });
    expect(run40.history[0].loss).toBeCloseTo(Math.LN2, 12);
    expect(run40.accuracy).toBe(1);
    // card-info.ts "Preference pair editor" controls: with the SFT reference the held-out margins are -1.48, -1.76 and +2.70.
    const gaps = HELD_OUT_PAIRS.map((pair) => scorePair(run40.weights, sft, pair).margin);
    expect(gaps.map((gap) => round(BETA * gap, 2))).toEqual([-1.48, -1.76, 2.7]);
    expect(gaps.filter((gap) => gap > 0)).toHaveLength(1);
    // ... and the KL from the SFT reference is larger than from the base at the same settings.
    expect(round(run40.finalDrift, 4)).toBe(0.0161);
    expect(run40.finalDrift).toBeGreaterThan(run().finalDrift);
  });
});
