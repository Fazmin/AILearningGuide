import { describe, expect, it } from "vitest";
import definition from "./module";
import { LENGTH_COST, ITERATIONS, STEP_SUCCESS, passProbability, percent, trainGroupRelative } from "./grpo";
import { majorityAccuracy, verifierAccuracy, votingCurves, type WrongSpread } from "./voting";

/**
 * Quote locations are given by file and section, not line number, because the lesson text is still being edited.
 * Pins the measured values the reasoning-models lesson (standard.mdx and plain.mdx) and card text
 * (card-info.ts) quote. Mirrors the Explore lab: votingCurves(32, p, spread) read at the chosen sample count,
 * and the lab's own `pct` formatter so the strings match what the Metric tiles print. The trace step counts
 * (Useful / Padding / Unfaithful) are authored constants in Explore, not computed, so they are not pinned.
 */
const MAX_SAMPLES = 32;
const SPREADS: Record<string, WrongSpread> = { same: 1, three: 3, unique: "unique" };
const P = definition.initialState.accuracy as number;
const DEFAULT_SPREAD = SPREADS[definition.initialState.spread as string];
const pct = (value: number) =>
  value >= 0.99995 && value < 1 ? ">99.99%" : `${(value * 100).toFixed(value > 0.995 && value < 1 ? 2 : 1)}%`;
const at = (samples: number, spread: WrongSpread) => votingCurves(MAX_SAMPLES, P, spread)[samples - 1];

describe("reasoning-models quoted values", () => {
  it("uses the default lab state", () => {
    // standard.mdx "How to play with it": "leave p = 0.40 and Split over 3".
    expect(P).toBe(0.4);
    expect(DEFAULT_SPREAD).toBe(3);
  });

  it("matches the 16-sample readouts under Split over 3", () => {
    const point = at(16, DEFAULT_SPREAD);
    // standard.mdx and plain.mdx "How to play with it", card-info.ts "Sampling and voting" controls: Majority vote 70.8%, Verifier picks 99.97%.
    expect(pct(point.majority)).toBe("70.8%");
    expect(pct(point.verifier)).toBe("99.97%");
    // The single-sample tile reads p.
    expect(pct(point.single)).toBe("40.0%");
    expect(point.majority).toBeCloseTo(0.708, 3);
    expect(point.verifier).toBeCloseTo(1 - 0.6 ** 16, 12);
  });

  it("matches the 16-sample vote when the wrong answer is always the same", () => {
    const point = at(16, SPREADS.same);
    // standard.mdx and plain.mdx "How to play with it", card-info.ts "Sampling and voting" controls: the vote drops to 21.3%, below a single sample.
    expect(pct(point.majority)).toBe("21.3%");
    expect(point.majority).toBeLessThan(point.single);
    // module.ts stepInstructions: "watch Majority vote fall below One sample".
    // standard.mdx "What to notice", card-info.ts "Sampling and voting" notice: with one wrong answer likelier than the right one, more samples make it worse.
    expect(at(32, SPREADS.same).majority).toBeLessThan(point.majority);
  });

  it("matches pass@n at eight samples", () => {
    // standard.mdx "What to notice": "At p = 0.40 it passes 98% by eight samples".
    expect(Math.round(verifierAccuracy(8, P) * 100)).toBe(98);
    expect(at(8, DEFAULT_SPREAD).verifier).toBeCloseTo(verifierAccuracy(8, P), 12);
  });

  it("never beats one sample at two samples", () => {
    // card-info.ts "Sampling and voting" notice: "Two samples never beat one under voting: a split vote is a coin toss."
    for (const spread of Object.values(SPREADS)) expect(majorityAccuracy(2, P, spread)).toBeCloseTo(P, 12);
  });
});

/**
 * The "Train against a checker" figures in standard.mdx and plain.mdx ("How to play with it", "What to notice"),
 * card-info.ts "Train against a checker" (controls, notice, howItWorks) and module.ts (checkpoint questions 2 and 3,
 * step instructions 2 and 3). Strings are formatted exactly as the lab's Metric tiles format them.
 */
describe("reasoning-models quoted values: Train against a checker", () => {
  const DEFAULTS = { k: 3, group: 16, rate: 0.3, cost: false, seed: 1 };
  const free = trainGroupRelative(DEFAULTS).history;
  const priced = trainGroupRelative({ ...DEFAULTS, cost: true });
  const small = trainGroupRelative({ ...DEFAULTS, group: 8, cost: true }).history;
  const steps = (value: number) => value.toFixed(1);

  it("uses the default lab state and constants the text quotes", () => {
    expect(definition.initialState).toMatchObject({ grpoK: 3, grpoGroup: 16, grpoRate: 0.3, grpoCost: false, grpoSeed: 1 });
    expect(ITERATIONS).toBe(100);
    expect(STEP_SUCCESS).toBe(0.8);
    expect(LENGTH_COST).toBe(0.05);
  });

  it("starts at 1.8 steps and a 10% pass rate", () => {
    // "Mean length climbs from 1.8 steps to 7.1 ... pass rate goes from 10% to 99%"; howItWorks: "the mean length is 1.8 steps".
    expect(steps(free[0].meanLength)).toBe("1.8");
    expect(percent(free[0].passRate)).toBe("10%");
  });

  it("with the cost off ends at 7.1 steps, 99% pass, 4.1 steps past k, and 51 of 100 flat groups", () => {
    const end = free[ITERATIONS];
    expect(steps(end.meanLength)).toBe("7.1");
    expect(percent(end.passRate)).toBe("99%");
    // "What to notice": "Steps past k still reads 4.1 at update 100", "In 51 of the 100 updates every sampled trace had the same reward".
    expect(steps(end.extraSteps)).toBe("4.1");
    expect(end.flatSoFar).toBe(51);
    expect(`${end.flatSoFar} of ${ITERATIONS}`).toBe("51 of 100");
    expect(end.meanLength).toBeGreaterThan(definition.initialState.grpoK as number);
  });

  it("is already close to 100% at update 50 and still lengthening afterwards (checkpoint question 2)", () => {
    expect(free[50].passRate).toBeGreaterThan(0.97);
    expect(free[ITERATIONS].meanLength).toBeGreaterThan(free[50].meanLength);
  });

  it("with the cost on ends at 5.3 steps, 93% pass, and 2.3 steps past k", () => {
    const end = priced.history[ITERATIONS];
    expect(steps(end.meanLength)).toBe("5.3");
    expect(percent(end.passRate)).toBe("93%");
    expect(steps(end.extraSteps)).toBe("2.3");
    expect(end.meanLength).toBeLessThan(free[ITERATIONS].meanLength);
  });

  it("collapses to 1.2 steps and a 1% pass rate at Group size 8 with the cost on (step 3, checkpoint question 3)", () => {
    const end = small[ITERATIONS];
    expect(steps(end.meanLength)).toBe("1.2");
    expect(percent(end.passRate)).toBe("1%");
  });

  it("passes exactly k = 3 steps only 51% of the time, because every step must land", () => {
    expect(percent(passProbability(3, 3))).toBe("51%");
    expect(passProbability(3, 3)).toBeCloseTo(STEP_SUCCESS ** 3, 12);
  });

  it("keeps the cost smaller than a pass, so a trace never scores below zero for its cost alone (checkpoint question 3)", () => {
    expect(LENGTH_COST * 8).toBeLessThan(1);
  });
});
