import { describe, expect, it } from "vitest";
import definition, { migrateRlState } from "./module";
import {
  ARM_MEANS,
  BANDIT_PULLS,
  BEST_ARM,
  logPolicyGradient,
  policyGradientAverages,
  runPolicyGradient,
  softmax,
} from "./mdp";

describe("policy gradient on the bandit", () => {
  it("builds a softmax that sums to one and does not overflow", () => {
    const policy = softmax([0, 0, 0, 0]);
    expect(policy).toEqual([0.25, 0.25, 0.25, 0.25]);
    const extreme = softmax([1000, 0, -1000, 5]);
    expect(extreme.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 12);
    expect(extreme.every((value) => Number.isFinite(value))).toBe(true);
  });

  it("uses the exact gradient of ln π: 1 for the pulled arm minus each probability, checked by finite differences", () => {
    const preferences = [0.4, -0.7, 1.1, 0.2];
    const policy = softmax(preferences);
    for (let arm = 0; arm < 4; arm += 1) {
      const analytic = logPolicyGradient(policy, arm);
      for (let index = 0; index < 4; index += 1) {
        const h = 1e-6;
        const up = [...preferences];
        const down = [...preferences];
        up[index] += h;
        down[index] -= h;
        const numeric = (Math.log(softmax(up)[arm]) - Math.log(softmax(down)[arm])) / (2 * h);
        expect(analytic[index]).toBeCloseTo(numeric, 6);
      }
    }
  });

  it("makes the expected update the gradient of expected reward, whatever constant baseline is subtracted", () => {
    const preferences = [0.3, -0.2, 0.5, 0.1];
    const policy = softmax(preferences);
    const expectedReward = (theta: number[]) => softmax(theta).reduce((sum, p, arm) => sum + p * ARM_MEANS[arm], 0);
    // E over arm and reward of (r − b)·∂ ln π(arm)/∂θ_i.
    const expectedUpdate = (baseline: number) =>
      preferences.map((_, index) =>
        policy.reduce((sum, p, arm) => sum + p * (ARM_MEANS[arm] - baseline) * logPolicyGradient(policy, arm)[index], 0),
      );
    const withoutBaseline = expectedUpdate(0);
    const withBaseline = expectedUpdate(0.4);
    withoutBaseline.forEach((value, index) => {
      expect(withBaseline[index]).toBeCloseTo(value, 12);
      const up = [...preferences];
      const down = [...preferences];
      up[index] += 1e-6;
      down[index] -= 1e-6;
      expect(value).toBeCloseTo((expectedReward(up) - expectedReward(down)) / 2e-6, 6);
    });
  });

  it("starts every arm at 25%, ends on a valid policy, and is deterministic", () => {
    const run = runPolicyGradient(0.2, true, 3);
    expect(run.bestProbability).toHaveLength(BANDIT_PULLS);
    expect(run.bestProbability[0]).toBe(0.25);
    expect(run.probabilities.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 12);
    expect(run.rewards).toHaveLength(BANDIT_PULLS);
    expect(runPolicyGradient(0.2, true, 3)).toEqual(run);
    expect(runPolicyGradient(0.2, false, 3)).not.toEqual(run);
  });

  it("with the baseline off, a reward of 0 moves nothing, so a run of unlucky pulls leaves the policy where it was", () => {
    const run = runPolicyGradient(0.5, false, 5, 1);
    // One pull: if it paid 0 the policy is untouched, if it paid 1 the pulled arm gains.
    if (run.rewards[0] === 0) expect(run.probabilities).toEqual([0.25, 0.25, 0.25, 0.25]);
    else expect(run.probabilities[run.choices[0]]).toBeGreaterThan(0.25);
  });

  it("quotes the 300-run averages at α 0.20: 92.1% with the baseline against 84.9% without, and 2 against 29 stuck runs", () => {
    const withBaseline = policyGradientAverages(0.2, true);
    const without = policyGradientAverages(0.2, false);
    expect(withBaseline.runs).toBe(300);
    expect(withBaseline.finalBestProbability).toBeCloseTo(0.921, 3);
    expect(without.finalBestProbability).toBeCloseTo(0.849, 3);
    expect(withBaseline.stuckRuns).toBe(2);
    expect(without.stuckRuns).toBe(29);
    expect(withBaseline.bestProbability[49]).toBeCloseTo(0.401, 3);
  });

  it("quotes the 300-run averages at α 1.00 with the baseline: committed sooner, 33 stuck runs, 88.3% at the end", () => {
    const fast = policyGradientAverages(1, true);
    expect(fast.bestProbability[49]).toBeCloseTo(0.651, 3);
    expect(fast.bestProbability[49]).toBeGreaterThan(policyGradientAverages(0.2, true).bestProbability[49]);
    expect(fast.stuckRuns).toBe(33);
    expect(fast.finalBestProbability).toBeCloseTo(0.883, 3);
    expect(fast.finalBestProbability).toBeLessThan(policyGradientAverages(0.2, true).finalBestProbability);
  });

  it("the baseline helps at every learning rate from 0.10 to 1.00 and changes nothing at 0.02, where nothing is learned", () => {
    for (const rate of [0.1, 0.2, 0.5, 1]) {
      const withBaseline = policyGradientAverages(rate, true);
      const without = policyGradientAverages(rate, false);
      expect(withBaseline.finalBestProbability).toBeGreaterThan(without.finalBestProbability);
      expect(withBaseline.stuckRuns).toBeLessThan(without.stuckRuns);
    }
    const slow = policyGradientAverages(0.02, true);
    expect(slow.finalBestProbability).toBeCloseTo(0.37, 2);
    expect(policyGradientAverages(0.02, false).finalBestProbability).toBeCloseTo(0.372, 3);
    expect(Math.abs(slow.finalBestProbability - policyGradientAverages(0.02, false).finalBestProbability)).toBeLessThan(0.01);
  });

  it("at the default seed 3 the baseline run ends on the best arm and the run without it on arm 2", () => {
    const withBaseline = runPolicyGradient(0.2, true, 3);
    const without = runPolicyGradient(0.2, false, 3);
    expect(withBaseline.probabilities.indexOf(Math.max(...withBaseline.probabilities))).toBe(BEST_ARM);
    expect(withBaseline.probabilities[BEST_ARM]).toBeCloseTo(0.83, 2);
    expect(without.probabilities.indexOf(Math.max(...without.probabilities))).toBe(1);
    expect(without.probabilities[1]).toBeCloseTo(0.92, 2);
  });
});

describe("rl-foundations state, version 3", () => {
  it("reads a version 2 payload and gives the policy-gradient keys their defaults", () => {
    const version2 = {
      gamma: 0.9,
      sweeps: 12,
      slip: 0.1,
      hack: "on",
      step: 4,
      seed: 6,
      cell: 7,
      epsilon: 0.2,
      banditSeed: 9,
    };
    expect(migrateRlState(version2)).toEqual({ ...version2, pgRate: 0.2, pgBaseline: "on", pgSeed: 3 });
    expect(definition.stateVersion).toBe(3);
  });

  it("clamps the new numbers and rejects an unknown baseline", () => {
    const hydrated = definition.hydrateState(JSON.stringify({ pgRate: 9e9, pgSeed: -50, pgBaseline: "sometimes" }));
    expect(hydrated.pgRate).toBe(1);
    expect(hydrated.pgSeed).toBe(1);
    expect(hydrated.pgBaseline).toBe("on");
    expect(definition.hydrateState(JSON.stringify({ pgRate: -3, pgBaseline: "off" })).pgRate).toBe(0.02);
    expect(definition.hydrateState(JSON.stringify({ pgBaseline: "off" })).pgBaseline).toBe("off");
  });
});
