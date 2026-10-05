import { describe, expect, it } from "vitest";
import { checkpointQuestions } from "@app/module-sdk";
import definition from "./module";
import {
  START,
  banditAverages,
  discountedReturn,
  policyGradientAverages,
  rollout,
  runBandit,
  valueIteration,
} from "./mdp";

const questions = checkpointQuestions(definition);
const correctOption = (index: number) => questions[index].options[questions[index].answer];

const plain = (gamma: number) => ({ gamma, slip: 0, bonus: false });

/** Facts the checkpoint questions rely on, so a change to the lab cannot silently invalidate them. */
describe("rl-foundations checkpoint claims", () => {
  it("shrinks the full return as the discount falls while the path still ends at G", () => {
    const far = rollout(valueIteration(plain(0.95))[30].values, plain(0.95), 3);
    const near = rollout(valueIteration(plain(0.5))[30].values, plain(0.5), 3);
    expect(far.outcome).toBe("goal");
    expect(near.outcome).toBe("goal");
    expect(near.cells).toEqual(far.cells);
    const g0 = (episode: typeof far, gamma: number) =>
      discountedReturn(episode.steps.map((step) => step.reward), gamma);
    expect(g0(near, 0.5)).toBeLessThan(g0(far, 0.95));
    expect(valueIteration(plain(0.5))[30].values[START]).toBeCloseTo(-0.046, 3);
  });

  it("keeps V(S) negative through sweep 5 and positive at sweep 6", () => {
    const history = valueIteration(plain(0.95));
    for (let sweep = 0; sweep <= 5; sweep += 1) expect(history[sweep].values[START]).toBeLessThanOrEqual(0);
    expect(history[6].values[START]).toBeGreaterThan(0);
  });

  it("explores too little at epsilon 0 and too much at 0.5", () => {
    for (const seed of [1, 2, 3]) {
      const run = runBandit(0, seed);
      expect(run.pulls.filter((count) => count > 0).length).toBeLessThanOrEqual(2);
    }
    expect(banditAverages(0.1).finalBestShare).toBeGreaterThan(banditAverages(0).finalBestShare);
    expect(banditAverages(0.1).finalBestShare).toBeGreaterThan(banditAverages(0.5).finalBestShare);
    // No exploration is not free: it also earns less on average than a little exploration.
    const last = (values: number[]) => values[values.length - 1];
    expect(last(banditAverages(0.1).meanRewardSoFar)).toBeGreaterThan(last(banditAverages(0).meanRewardSoFar));
    expect(banditAverages(0).finalBestShare).toBeCloseTo(0.326, 2);
    expect(banditAverages(0.1).finalBestShare).toBeCloseTo(0.717, 2);
  });

  it("question 6: switching the baseline off leaves the policy less sure of the best arm and strands more runs", () => {
    const withBaseline = policyGradientAverages(0.2, true);
    const without = policyGradientAverages(0.2, false);
    expect(withBaseline.finalBestProbability).toBeGreaterThan(without.finalBestProbability);
    expect(withBaseline.stuckRuns).toBeLessThan(without.stuckRuns);
    expect((withBaseline.finalBestProbability * 100).toFixed(1)).toBe("92.1");
    expect((without.finalBestProbability * 100).toFixed(1)).toBe("84.9");
    expect([withBaseline.stuckRuns, without.stuckRuns]).toEqual([2, 29]);
    expect(correctOption(5)).toMatch(/^The policy ends less sure of the best arm/);
  });

  it("question 7: a larger learning rate commits sooner but strands more runs", () => {
    const slow = policyGradientAverages(0.2, true);
    const fast = policyGradientAverages(1, true);
    expect((slow.bestProbability[49] * 100).toFixed(1)).toBe("40.1");
    expect((fast.bestProbability[49] * 100).toFixed(1)).toBe("65.1");
    expect(fast.stuckRuns).toBeGreaterThan(slow.stuckRuns);
    expect(fast.finalBestProbability).toBeLessThan(slow.finalBestProbability);
    expect((fast.finalBestProbability * 100).toFixed(1)).toBe("88.3");
    expect(correctOption(6)).toMatch(/^The policy commits sooner/);
  });
});
