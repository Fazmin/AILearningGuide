import { describe, expect, it } from "vitest";
import { checkpointQuestions, trainTinyExtended, UNIFORM_CROSS_ENTROPY } from "@app/module-sdk";
import { DOMAINS, pairCount } from "./corpora";
import definition from "./module";
import { BUDGET_PAIRS, runEvenSpread, runMixture, sharesOf, STEPS, SWITCH_STEP } from "./mixture";
import { answerProbe } from "./probe";
import { repetitionSweep, runRepetition, uniqueText } from "./repetition";

/**
 * Pins the claims the checkpoint explanations in module.ts make, and checks that each distractor's premise
 * is false in this lab where the lab can test it. quoted.test.ts pins the same figures where the lessons
 * quote them.
 */
const f3 = (value: number) => Number(value.toFixed(3));
const f2 = (value: number) => Number(value.toFixed(2));
const questions = checkpointQuestions(definition);

describe("mid-training checkpoint claims", () => {
  it("has six questions covering all five objectives", () => {
    expect(questions).toHaveLength(6);
    expect(new Set(questions.map((question) => question.objective))).toEqual(new Set([0, 1, 2, 3, 4]));
  });

  it("question 1: Proverbs is the smallest domain and already had the most passes", () => {
    const uniform = runMixture([1, 1, 1, 1], "same", "wsd");
    const heavy = runMixture([3, 3, 3, 1], "same", "wsd");
    const pairs = DOMAINS.map((domain) => pairCount(domain.trainText));
    expect(BUDGET_PAIRS).toBe(9600);
    expect(pairs[2]).toBe(Math.min(...pairs));
    expect(pairs[2]).toBe(195);
    expect(pairs[0]).toBe(378);
    expect(uniform.domains[2].epochs).toBe(Math.max(...uniform.domains.map((domain) => domain.epochs)));
    expect([f2(uniform.domains[2].epochs), f2(heavy.domains[2].epochs)]).toEqual([12.48, 14.72]);
    expect([f3(uniform.domains[2].held), f3(heavy.domains[2].held)]).toEqual([2.4, 2.408]);
    expect([f3(uniform.domains[0].held), f3(heavy.domains[0].held)]).toEqual([2.197, 2.097]);
    // Distractors: Proverbs' weight did change (1 to 3, share 25% to 30%); held-out sentences never join training.
    expect([sharesOf([1, 1, 1, 1])[2], sharesOf([3, 3, 3, 1])[2]]).toEqual([0.25, 0.3]);
    expect(DOMAINS[2].trainText.includes(DOMAINS[2].heldOut[0])).toBe(false);
  });

  it("question 2: two sentences fall, bottom out near 27 epochs, then climb", () => {
    const two = runRepetition(2, 8000);
    const all = runRepetition(120, 8000);
    expect(pairCount(uniqueText(2))).toBe(59);
    expect(Math.round(two.epochs)).toBe(136);
    expect([f3(two.lowest.held), Math.round(two.lowest.epochs), f3(two.held), f3(all.held)]).toEqual([2.217, 27, 2.542, 1.388]);
    // It falls first: the lowest point is well below the zero table and comes early.
    expect(two.lowest.held).toBeLessThan(UNIFORM_CROSS_ENTROPY - 1);
    expect(two.lowest.pairs).toBeLessThan(two.budget / 4);
    // Distractors: it ends worst of all the bars, not best, and the budget alone does not decide the loss.
    const bars = repetitionSweep(8000);
    expect(two.held).toBe(Math.max(...bars.map((run) => run.held)));
    expect(new Set(bars.map((run) => f3(run.held))).size).toBe(bars.length);
  });

  it("question 3: ending on the target mix wins on Arithmetic and costs a little General", () => {
    const switched = runMixture([3, 3, 3, 1], "mix", "wsd");
    const even = runEvenSpread([3, 3, 3, 1], "mix", "wsd");
    expect([switched.domains[3].seen, even.domains[3].seen]).toEqual([1721, 1707]);
    expect([f3(switched.arithmetic), f3(even.arithmetic)]).toEqual([1.911, 2.11]);
    expect([f3(switched.general), f3(even.general)]).toEqual([2.398, 2.339]);
    expect(switched.arithmetic).toBeLessThan(even.arithmetic);
    expect(switched.general).toBeGreaterThan(even.general);
    // "half Arithmetic": the late shares
    expect(runMixture([3, 3, 3, 1], "mix", "wsd")).toBe(switched);
  });

  it("question 4: Target only takes the same steps and schedule, and forgets far more", () => {
    const mix = runMixture([3, 3, 3, 1], "mix", "wsd");
    const only = runMixture([3, 3, 3, 1], "only", "wsd");
    expect([mix.steps, only.steps]).toEqual([STEPS, STEPS]);
    expect([f3(only.forgetting), f3(mix.forgetting)]).toEqual([0.353, 0.06]);
    // Before the switch the two runs are the same run, so the schedule did not differ.
    expect(only.atSwitch.held).toEqual(mix.atSwitch.held);
    // No general pair is drawn after the switch under Target only: the run draws exactly the general pairs that
    // its first 480 steps draw (the schedule never changes which batches are sampled, only how far they move).
    const firstPart = trainTinyExtended({
      text: "",
      sources: DOMAINS.map((domain) => ({ id: domain.id, text: domain.trainText })),
      mixture: [{ from: 0, weights: sharesOf([3, 3, 3, 1]) }],
      steps: SWITCH_STEP,
      batchSize: 16,
      learningRate: 3,
      schedule: "wsd",
      seed: 7,
    });
    expect(only.domains.slice(0, 3).map((domain) => domain.seen)).toEqual(firstPart.sourceSeen.slice(0, 3));
    expect(only.domains[3].seen).toBe(firstPart.sourceSeen[3] + (STEPS - SWITCH_STEP) * 16);
  });

  it("question 5: the Arithmetic loss falls while the probe stays at zero, giving ten every time", () => {
    const none = runMixture([3, 3, 3, 0], "same", "wsd");
    const eight = runMixture([3, 3, 3, 8], "same", "wsd");
    expect([f3(none.arithmetic), f3(eight.arithmetic)]).toEqual([2.832, 1.706]);
    const probe = answerProbe(eight.weights);
    expect([probe.correct, probe.total, probe.pick, probe.distinctPicks, probe.ranking.length]).toEqual([0, 40, "ten", 1, 21]);
    // Distractor: the held-out facts are not the problem; the trained facts fare no better.
    const trainedRight = DOMAINS[3].train.filter((sentence) => sentence.endsWith(` ${probe.pick}.`)).length;
    expect(trainedRight / DOMAINS[3].train.length).toBeLessThan(0.1);
  });

  it("question 6: at a constant rate the switch helps Arithmetic more and costs General more", () => {
    const decay = runMixture([3, 3, 3, 1], "mix", "wsd");
    const constant = runMixture([3, 3, 3, 1], "mix", "constant");
    expect([f3(constant.arithmetic), f3(decay.arithmetic)]).toEqual([1.785, 1.911]);
    expect([f3(constant.general), f3(decay.general)]).toEqual([2.462, 2.398]);
    expect([f3(decay.forgetting), f3(constant.forgetting)]).toEqual([0.06, 0.127]);
    // Distractors: the schedule does not change which batches are drawn, so Arithmetic examples are not doubled...
    expect(constant.domains.map((domain) => domain.seen)).toEqual(decay.domains.map((domain) => domain.seen));
    // ...and without a warmup the general domains still trained: General at the switch is within 0.01 of the decay run.
    expect(Math.abs(constant.atSwitch.general - decay.atSwitch.general)).toBeLessThan(0.01);
  });
});
