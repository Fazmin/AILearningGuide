import { describe, expect, it } from "vitest";
import { extendedScheduleFactor, tinyCrossEntropy, trainTinyExtended, UNIFORM_CROSS_ENTROPY } from "@app/module-sdk";
import { ARITHMETIC, DOMAINS } from "./corpora";
import {
  arithmeticShareCurve,
  arithmeticSweep,
  BATCH_SIZE,
  BUDGET_PAIRS,
  CHECKPOINTS,
  DECAY_FRACTION,
  evenShares,
  gainPerEpoch,
  lateShares,
  LEARNING_RATE,
  mixturePhases,
  runEvenSpread,
  runMixture,
  scheduleCurve,
  SEED,
  sharesOf,
  STEPS,
  SWITCH_PROGRESS,
  SWITCH_STEP,
  TARGET_MIX_SHARE,
  WARMUP_FRACTION,
} from "./mixture";

const sum = (values: ReadonlyArray<number>) => values.reduce((total, value) => total + value, 0);

describe("mixture shares", () => {
  it("turns weights into shares of every batch, and only the ratio counts", () => {
    expect(sharesOf([3, 3, 3, 1])).toEqual([0.3, 0.3, 0.3, 0.1]);
    expect(sharesOf([6, 6, 6, 2])).toEqual(sharesOf([3, 3, 3, 1]));
    expect(sharesOf([1, 1, 1, 1])).toEqual([0.25, 0.25, 0.25, 0.25]);
    // All zero, or garbage, falls back to equal shares, as the trainer does.
    expect(sharesOf([0, 0, 0, 0])).toEqual([0.25, 0.25, 0.25, 0.25]);
    expect(sharesOf([Number.NaN, -2, Infinity, 1])).toEqual([0, 0, 0, 1]);
  });

  it("builds the late shares: the same mix, half Arithmetic, or Arithmetic alone", () => {
    expect(lateShares([3, 3, 3, 1], "same")).toEqual(sharesOf([3, 3, 3, 1]));
    expect(lateShares([3, 3, 3, 1], "mix")).toEqual([1 / 6, 1 / 6, 1 / 6, TARGET_MIX_SHARE]);
    expect(lateShares([4, 2, 2, 7], "mix")).toEqual([0.25, 0.125, 0.125, 0.5]);
    expect(lateShares([3, 3, 3, 1], "only")).toEqual([0, 0, 0, 1]);
    // With no general weight there is no general text to keep, so Target mix is Arithmetic alone.
    expect(lateShares([0, 0, 0, 5], "mix")).toEqual([0, 0, 0, 1]);
  });

  it("spreads the switched run's expected pairs evenly over the whole run", () => {
    for (const late of ["same", "mix", "only"] as const) {
      const even = evenShares([3, 3, 3, 1], late);
      expect(sum(even)).toBeCloseTo(1, 12);
      const expected = sharesOf([3, 3, 3, 1]).map(
        (share, index) => SWITCH_PROGRESS * share + DECAY_FRACTION * lateShares([3, 3, 3, 1], late)[index],
      );
      even.forEach((share, index) => expect(share).toBeCloseTo(expected[index], 12));
    }
    expect(evenShares([3, 3, 3, 1], "mix")[3]).toBeCloseTo(0.8 * 0.1 + 0.2 * 0.5, 12);
  });

  it("switches at progress 0.8, which the trainer first reaches at step 480", () => {
    expect([STEPS, BATCH_SIZE, BUDGET_PAIRS]).toEqual([600, 16, 9600]);
    expect(SWITCH_PROGRESS).toBe(0.8);
    expect(SWITCH_STEP).toBe(480);
    expect((SWITCH_STEP - 1) / (STEPS - 1)).toBeLessThan(SWITCH_PROGRESS);
    expect(SWITCH_STEP / (STEPS - 1)).toBeGreaterThanOrEqual(SWITCH_PROGRESS);
    // The decay starts on the same step: the multiplier is 1 just before it and below 1 from it on.
    expect(extendedScheduleFactor("wsd", (SWITCH_STEP - 1) / (STEPS - 1), WARMUP_FRACTION, DECAY_FRACTION)).toBe(1);
    expect(extendedScheduleFactor("wsd", SWITCH_STEP / (STEPS - 1), WARMUP_FRACTION, DECAY_FRACTION)).toBeLessThan(1);
    expect(mixturePhases([3, 3, 3, 1], "same")).toEqual([{ from: 0, weights: [0.3, 0.3, 0.3, 0.1] }]);
    expect(mixturePhases([3, 3, 3, 1], "only")[1]).toEqual({ from: 0.8, weights: [0, 0, 0, 1] });
  });
});

describe("mixture runs", () => {
  it("draws batches in the proportions the weights set", () => {
    for (const weights of [
      [3, 3, 3, 1],
      [1, 1, 1, 1],
      [1, 0, 2, 5],
    ]) {
      const run = runMixture(weights, "same", "wsd");
      const drawn = run.domains.map((domain) => domain.seen);
      expect(sum(drawn)).toBe(BUDGET_PAIRS);
      const shares = sharesOf(weights);
      drawn.forEach((count, index) => {
        // 9,600 draws: a binomial share stays within about 0.025 of its target.
        expect(Math.abs(count / BUDGET_PAIRS - shares[index]), `${weights} domain ${index}`).toBeLessThan(0.025);
        if (shares[index] === 0) expect(count).toBe(0);
      });
    }
  });

  it("draws the late shares only from step 480 on", () => {
    const switched = runMixture([3, 3, 3, 1], "only", "wsd");
    const same = runMixture([3, 3, 3, 1], "same", "wsd");
    // The table at the switch is identical: the two runs only differ after it.
    expect(switched.atSwitch.step).toBe(SWITCH_STEP);
    expect(switched.atSwitch.held).toEqual(same.atSwitch.held);
    // 120 late steps of 16 pairs are all Arithmetic.
    const lateBatches = (STEPS - SWITCH_STEP) * BATCH_SIZE;
    expect(switched.domains[3].seen - lateBatches).toBeGreaterThan(0);
    expect(sum(switched.domains.slice(0, 3).map((domain) => domain.seen))).toBe(BUDGET_PAIRS - switched.domains[3].seen);
    expect(lateBatches).toBe(1920);
  });

  it("mirrors a direct trainTinyExtended call with the lab's recipe", () => {
    const run = runMixture([3, 3, 3, 1], "mix", "wsd");
    const direct = trainTinyExtended({
      text: ARITHMETIC.heldOutText,
      sources: DOMAINS.map((domain) => ({ id: domain.id, text: domain.trainText })),
      mixture: mixturePhases([3, 3, 3, 1], "mix"),
      steps: 600,
      batchSize: 16,
      learningRate: LEARNING_RATE,
      schedule: "wsd",
      warmupFraction: 0.05,
      decayFraction: 0.2,
      seed: SEED,
      checkpoints: CHECKPOINTS,
    });
    expect(Array.from(run.weights)).toEqual(Array.from(direct.weights));
    expect(run.domains.map((domain) => domain.seen)).toEqual(direct.sourceSeen);
    expect(run.domains.map((domain) => domain.held)).toEqual(DOMAINS.map((domain) => tinyCrossEntropy(direct.weights, domain.heldOutText)));
    expect(run.arithmetic).toBe(direct.finalLoss);
    expect([LEARNING_RATE, SEED, CHECKPOINTS]).toEqual([3, 7, 21]);
  });

  it("scores every checkpoint, every 30 steps, starting from the zero table", () => {
    const run = runMixture([3, 3, 3, 1], "same", "wsd");
    expect(run.curve.map((point) => point.step)).toEqual(Array.from({ length: 21 }, (_, index) => index * 30));
    for (const value of run.curve[0].held) expect(value).toBeCloseTo(UNIFORM_CROSS_ENTROPY, 6);
    expect(run.general).toBeCloseTo((run.domains[0].held + run.domains[1].held + run.domains[2].held) / 3, 12);
    expect(run.forgetting).toBeCloseTo(run.general - run.atSwitch.general, 12);
    for (const domain of run.domains) expect(domain.epochs).toBeCloseTo(domain.seen / domain.pairs, 12);
  });

  it("keeps the even spread's expected pairs equal to the switched run's", () => {
    for (const late of ["mix", "only"] as const) {
      const switched = runMixture([3, 3, 3, 1], late, "wsd");
      const even = runEvenSpread([3, 3, 3, 1], late, "wsd");
      // Both are random draws of the same expected counts, so they agree to within a few percent.
      expect(Math.abs(switched.domains[3].seen - even.domains[3].seen) / switched.domains[3].seen).toBeLessThan(0.03);
      expect(sum(even.domains.map((domain) => domain.seen))).toBe(BUDGET_PAIRS);
    }
  });

  it("returns the cached run for equal settings, and a different run for a different seed", () => {
    const first = runMixture([3, 3, 3, 1], "same", "wsd");
    expect(runMixture([6, 6, 6, 2], "same", "wsd")).toBe(first);
    expect(runMixture([3, 3, 3, 1], "same", "wsd", 7)).toBe(first);
    expect(runMixture([3, 3, 3, 1], "same", "wsd", 8)).not.toBe(first);
    expect(runMixture([3, 3, 3, 1], "same", "constant")).not.toBe(first);
  });
});

describe("the arithmetic weight sweep", () => {
  it("runs weights 0 to 8 with the general weights held fixed", () => {
    const sweep = arithmeticSweep([3, 3, 3, 5], "same", "wsd");
    expect(sweep.map((point) => point.weight)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8]);
    // The fourth weight passed in is ignored: each row sets its own.
    expect(sweep[1].run).toBe(runMixture([3, 3, 3, 1], "same", "wsd"));
    expect(sweep[0].run!.domains[3].seen).toBe(0);
    for (let index = 1; index < sweep.length; index += 1) {
      expect(sweep[index].run!.domains[3].epochs).toBeGreaterThan(sweep[index - 1].run!.domains[3].epochs);
    }
  });

  it("skips the one empty mixture and computes the gain per extra epoch between neighbours", () => {
    const empty = arithmeticSweep([0, 0, 0, 0], "same", "wsd");
    expect(empty[0].run).toBeNull();
    expect(empty[1].run).not.toBeNull();
    const gains = gainPerEpoch(empty);
    expect(gains[0]).toBeNull();
    expect(gains[1]).toBeNull();

    const sweep = arithmeticSweep([3, 3, 3, 1], "same", "wsd");
    const values = gainPerEpoch(sweep);
    expect(values[0]).toBeNull();
    for (let index = 1; index < sweep.length; index += 1) {
      const before = sweep[index - 1].run!;
      const after = sweep[index].run!;
      expect(values[index]).toBeCloseTo(
        (before.arithmetic - after.arithmetic) / (after.domains[3].epochs - before.domains[3].epochs),
        12,
      );
    }
  });
});

describe("the late-phase chart", () => {
  it("draws the trainer's own multiplier", () => {
    const wsd = scheduleCurve("wsd");
    expect(wsd[0].y).toBeCloseTo(0.02, 12);
    expect(wsd.find((point) => point.x === 0.5)!.y).toBe(1);
    expect(wsd[wsd.length - 1].y).toBe(0);
    for (const point of wsd) expect(point.y).toBe(extendedScheduleFactor("wsd", point.x, 0.05, 0.2));
    for (const point of scheduleCurve("constant")) expect(point.y).toBe(1);
  });

  it("steps the Arithmetic share at the switch", () => {
    expect(arithmeticShareCurve([3, 3, 3, 1], "mix").map((point) => point.y)).toEqual([0.1, 0.1, 0.5, 0.5]);
    expect(arithmeticShareCurve([3, 3, 3, 1], "same").map((point) => point.y)).toEqual([0.1, 0.1, 0.1, 0.1]);
    expect(arithmeticShareCurve([3, 3, 3, 1], "only").map((point) => point.y)).toEqual([0.1, 0.1, 1, 1]);
  });
});
