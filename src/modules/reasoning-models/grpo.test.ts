import { describe, expect, it } from "vitest";
import {
  bestCentre,
  expectedReward,
  groupAdvantages,
  GROUP_RANGE,
  INITIAL_CENTRE,
  ITERATIONS,
  K_RANGE,
  lengthPolicy,
  LENGTH_COST,
  MAX_LENGTH,
  passProbability,
  RATE_RANGE,
  trainGroupRelative,
  type GrpoOptions,
} from "./grpo";

const DEFAULTS: GrpoOptions = { k: 3, group: 16, rate: 0.3, cost: false, seed: 1 };
const final = (options: Partial<GrpoOptions>) => {
  const history = trainGroupRelative({ ...DEFAULTS, ...options }).history;
  return history[history.length - 1];
};

describe("group-relative advantages", () => {
  it("centres on the group mean and scales to unit population standard deviation", () => {
    const advantages = groupAdvantages([1, 0, 0, 0, 0, 0, 0, 0]);
    expect(advantages.reduce((sum, value) => sum + value, 0)).toBeCloseTo(0, 12);
    const variance = advantages.reduce((sum, value) => sum + value ** 2, 0) / advantages.length;
    expect(variance).toBeCloseTo(1, 4);
    // One pass in eight: the lone pass gets the large positive advantage.
    expect(advantages[0]).toBeGreaterThan(2);
    expect(advantages[1]).toBeLessThan(0);
  });

  it("gives a group of identical rewards zero advantage, so it teaches nothing", () => {
    expect(groupAdvantages([1, 1, 1, 1]).every((value) => value === 0)).toBe(true);
    expect(groupAdvantages([0, 0, 0, 0]).every((value) => value === 0)).toBe(true);
  });
});

describe("the hand-written checker", () => {
  it("never passes a trace shorter than k, and needs every step to land at exactly k", () => {
    for (let k = K_RANGE.min; k <= K_RANGE.max; k += 1) {
      for (let length = 1; length < k; length += 1) expect(passProbability(length, k)).toBe(0);
      expect(passProbability(k, k)).toBeCloseTo(0.8 ** k, 12);
    }
  });

  it("passes more often with every spare step, with shrinking gains", () => {
    for (let k = K_RANGE.min; k <= K_RANGE.max; k += 1) {
      let previousGain = Number.POSITIVE_INFINITY;
      for (let length = k; length < MAX_LENGTH; length += 1) {
        const gain = passProbability(length + 1, k) - passProbability(length, k);
        expect(gain).toBeGreaterThan(0);
        if (length > k) expect(gain).toBeLessThan(previousGain);
        previousGain = gain;
      }
    }
  });

  it("makes every step cost reward only when the length cost is on", () => {
    expect(expectedReward(5, 3, false)).toBe(passProbability(5, 3));
    expect(expectedReward(5, 3, true)).toBeCloseTo(passProbability(5, 3) - LENGTH_COST * 5, 12);
  });
});

describe("training is deterministic in the seed", () => {
  it("returns the identical history for the same controls and seed", () => {
    const a = trainGroupRelative({ ...DEFAULTS, seed: 7 });
    const b = trainGroupRelative({ ...DEFAULTS, seed: 7 });
    expect(b).toEqual(a);
  });

  it("takes a different path for a different seed", () => {
    const a = trainGroupRelative({ ...DEFAULTS, seed: 1 });
    const b = trainGroupRelative({ ...DEFAULTS, seed: 2 });
    expect(b.history[50].centre).not.toBe(a.history[50].centre);
  });

  it("records one point per update plus the starting policy, with exact distributions", () => {
    const run = trainGroupRelative(DEFAULTS);
    expect(run.history).toHaveLength(ITERATIONS + 1);
    expect(run.history[0].centre).toBe(INITIAL_CENTRE);
    expect(run.history[0].probabilities).toEqual(lengthPolicy(INITIAL_CENTRE));
    for (const point of run.history) {
      expect(point.probabilities.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 12);
      expect(point.passRate).toBeGreaterThanOrEqual(0);
      expect(point.passRate).toBeLessThanOrEqual(1);
    }
    expect(run.history[ITERATIONS].flatSoFar).toBe(run.flatGroups);
  });

  it("finishes the largest run the controls allow in a few milliseconds", () => {
    const start = performance.now();
    trainGroupRelative({ k: K_RANGE.max, group: GROUP_RANGE.max, rate: RATE_RANGE.max, cost: true, seed: 20 });
    expect(performance.now() - start).toBeLessThan(250);
  });
});

describe("length emerges when the task needs more than one step", () => {
  it("grows past the starting length for every k above 1, for every seed tried, with the cost off", () => {
    for (const k of [2, 3, 4]) {
      for (const seed of [1, 2, 3, 4, 5, 6]) {
        const run = trainGroupRelative({ ...DEFAULTS, k, seed }).history;
        expect(run[ITERATIONS].meanLength, `k ${k} seed ${seed}`).toBeGreaterThan(run[0].meanLength + 1);
        expect(run[ITERATIONS].meanLength, `k ${k} seed ${seed}`).toBeGreaterThan(k);
        expect(run[ITERATIONS].passRate, `k ${k} seed ${seed}`).toBeGreaterThan(run[0].passRate);
      }
    }
  });
});

describe("with no length cost, length keeps drifting past k; the cost trims it", () => {
  it("keeps rising after the pass rate has saturated", () => {
    const run = trainGroupRelative(DEFAULTS).history;
    expect(run[50].passRate).toBeGreaterThan(0.97);
    expect(run[ITERATIONS].meanLength).toBeGreaterThan(run[50].meanLength);
    expect(run[ITERATIONS].meanLength).toBeGreaterThan(DEFAULTS.k + 3);
  });

  it("ends shorter, with fewer steps past k, once each step costs reward, for every seed tried", () => {
    for (const seed of [1, 2, 3, 4, 5, 6, 7, 8]) {
      const free = final({ seed });
      const priced = final({ seed, cost: true });
      expect(priced.meanLength, `seed ${seed}`).toBeLessThan(free.meanLength - 1);
      expect(priced.extraSteps, `seed ${seed}`).toBeLessThan(free.extraSteps);
      expect(priced.passRate, `seed ${seed}`).toBeGreaterThan(0.8);
    }
  });

  it("does not trim all the way back to k, because spare steps still insure against a slip", () => {
    const priced = final({ cost: true });
    expect(priced.meanLength).toBeGreaterThan(DEFAULTS.k + 1);
    // The best reachable policy in this family is also well past k.
    expect(bestCentre(DEFAULTS.k, true).point.meanLength).toBeGreaterThan(DEFAULTS.k + 1);
    expect(bestCentre(DEFAULTS.k, false).point.meanLength).toBeGreaterThan(bestCentre(DEFAULTS.k, true).point.meanLength);
  });
});

describe("where the toy breaks", () => {
  it("collapses to one-step answers when the cost is on and no early group contains a pass", () => {
    const collapsed = final({ group: 8, cost: true });
    expect(collapsed.meanLength).toBeLessThan(1.5);
    expect(collapsed.passRate).toBeLessThan(0.05);
    // The same small group learns fine when steps are free, and a larger group learns fine with the cost.
    expect(final({ group: 8, cost: false }).passRate).toBeGreaterThan(0.95);
    expect(final({ group: 16, cost: true }).passRate).toBeGreaterThan(0.9);
  });

  it("collapses on the hard task (k = 4) with the cost on, and never finds a pass", () => {
    const hard = final({ k: 4, cost: true });
    expect(hard.meanLength).toBeLessThan(1.5);
    expect(hard.passRate).toBeLessThan(0.01);
    expect(final({ k: 4, cost: false }).passRate).toBeGreaterThan(0.9);
  });
});
