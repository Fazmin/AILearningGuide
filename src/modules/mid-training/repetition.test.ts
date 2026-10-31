import { describe, expect, it } from "vitest";
import { trainTinyExtended, UNIFORM_CROSS_ENTROPY } from "@app/module-sdk";
import { ARITHMETIC, pairCount } from "./corpora";
import {
  BUDGET_MAX,
  BUDGET_MIN,
  BUDGET_STEP,
  DEFAULT_BUDGET,
  DEFAULT_UNIQUE,
  repetitionSweep,
  runRepetition,
  SWEEP_UNIQUE,
  UNIQUE_MAX,
  UNIQUE_MIN,
  uniqueText,
} from "./repetition";

describe("repeated data runs", () => {
  it("offers 2 to 120 unique sentences and budgets of whole 16-pair steps", () => {
    expect([UNIQUE_MIN, UNIQUE_MAX, DEFAULT_UNIQUE]).toEqual([2, 120, 120]);
    expect([BUDGET_MIN, BUDGET_MAX, BUDGET_STEP, DEFAULT_BUDGET]).toEqual([2000, 16000, 2000, 8000]);
    for (let budget = BUDGET_MIN; budget <= BUDGET_MAX; budget += BUDGET_STEP) expect(budget % 16).toBe(0);
    expect(SWEEP_UNIQUE).toEqual([120, 60, 30, 15, 8, 4, 2]);
  });

  it("trains on the first sentences only, and the held-out facts are none of them", () => {
    expect(uniqueText(2)).toBe(`${ARITHMETIC.train[0]} ${ARITHMETIC.train[1]}`);
    for (const sentence of ARITHMETIC.heldOut) expect(uniqueText(UNIQUE_MAX).includes(sentence)).toBe(false);
  });

  it("spends the whole budget on the chosen sentences", () => {
    for (const unique of [120, 8, 2]) {
      const run = runRepetition(unique, DEFAULT_BUDGET);
      expect(run.steps).toBe(500);
      expect(run.uniquePairs).toBe(pairCount(uniqueText(unique)));
      expect(run.epochs).toBeCloseTo(DEFAULT_BUDGET / run.uniquePairs, 12);
      expect(run.curve[0]).toEqual({ pairs: 0, epochs: 0, held: expect.closeTo(UNIFORM_CROSS_ENTROPY, 6) });
      expect(run.curve[run.curve.length - 1].pairs).toBe(DEFAULT_BUDGET);
      expect(run.curve[run.curve.length - 1].held).toBe(run.held);
      expect(run.lowest.held).toBe(Math.min(...run.curve.map((point) => point.held)));
    }
  });

  it("mirrors a direct trainTinyExtended call", () => {
    const run = runRepetition(8, 4000);
    const direct = trainTinyExtended({
      text: ARITHMETIC.heldOutText,
      sources: [{ id: "arithmetic", text: uniqueText(8) }],
      steps: 250,
      batchSize: 16,
      learningRate: 3,
      schedule: "wsd",
      warmupFraction: 0.05,
      decayFraction: 0.2,
      seed: 7,
      checkpoints: 21,
    });
    expect(run.held).toBe(direct.finalLoss);
    expect(run.curve.map((point) => point.held)).toEqual(direct.checkpoints.map((point) => point.loss));
  });

  it("clamps out-of-range settings and caches equal ones", () => {
    expect(runRepetition(1e9, 1e9)).toBe(runRepetition(UNIQUE_MAX, BUDGET_MAX));
    expect(runRepetition(-5, -5)).toBe(runRepetition(UNIQUE_MIN, BUDGET_MIN));
    expect(runRepetition(7.6, 7900)).toBe(runRepetition(8, 8000));
    expect(runRepetition(Number.NaN, Infinity)).toBe(runRepetition(DEFAULT_UNIQUE, DEFAULT_BUDGET));
  });

  it("returns one bar per sweep entry, in order", () => {
    const sweep = repetitionSweep(DEFAULT_BUDGET);
    expect(sweep.map((run) => run.unique)).toEqual([...SWEEP_UNIQUE]);
    for (let index = 1; index < sweep.length; index += 1) expect(sweep[index].epochs).toBeGreaterThan(sweep[index - 1].epochs);
  });

  it("finishes the largest sweep quickly", () => {
    const started = performance.now();
    repetitionSweep(BUDGET_MAX - BUDGET_STEP);
    expect(performance.now() - started).toBeLessThan(2000);
  });
});
