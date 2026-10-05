import { describe, expect, it } from "vitest";
import definition, { migratePostTrainingState } from "./module";
import {
  CANDIDATES,
  DEFAULT_LABELS,
  betaSweep,
  comparisonsFrom,
  expectation,
  fitRewardModel,
  goldAgreement,
  goldLabels,
  klDivergence,
  parseLabels,
  rewardOf,
  sftLogits,
  sigmoid,
  softmax,
  tiltedPolicy,
  wordCount,
  type Label,
} from "./prefs";

const gold = CANDIDATES.map((candidate) => candidate.gold);
const reference = softmax(sftLogits(3).logits);
const rewardsFor = (labels: ReadonlyArray<Label>) => {
  const fit = fitRewardModel(comparisonsFrom(labels));
  return { fit, rewards: CANDIDATES.map((_, index) => rewardOf(fit.weights, index)) };
};

describe("post-training toy pipeline", () => {
  it("measures word counts from the text the lesson quotes", () => {
    expect(CANDIDATES.map((candidate) => wordCount(candidate.text))).toEqual([17, 25, 52, 9, 59, 39]);
  });

  it("moves probability toward the demonstrations with each SFT step", () => {
    const base = softmax(CANDIDATES.map((candidate) => candidate.baseLogit));
    expect(base[0]).toBeCloseTo(0.502, 3);
    const run = sftLogits(8);
    for (let step = 1; step < run.losses.length; step += 1) {
      expect(run.losses[step]).toBeLessThan(run.losses[step - 1]);
    }
    const after = softmax(run.logits);
    expect(after[1]).toBeCloseTo(0.379, 3);
    expect(after[5]).toBeCloseTo(0.364, 3);
  });

  it("starts the Bradley–Terry loss at ln 2 and fits the default labels", () => {
    const { fit, rewards } = rewardsFor(DEFAULT_LABELS);
    expect(fit.history[0].loss).toBeCloseTo(Math.log(2), 12);
    expect(fit.weights.map((value) => Number(value.toFixed(3)))).toEqual([1.255, 0.471, 0.142]);
    for (const c of comparisonsFrom(DEFAULT_LABELS)) expect(rewards[c.winner]).toBeGreaterThan(rewards[c.loser]);
    expect(goldAgreement(rewards).agree).toBe(9);
    expect(sigmoid(rewards[4] - rewards[1])).toBeCloseTo(0.98, 2);
  });

  it("drops the length weight once the confound is broken", () => {
    const labels = parseLabels("aaaaba");
    const { fit, rewards } = rewardsFor(labels);
    expect(fit.weights[0]).toBeCloseTo(0.456, 3);
    expect(fit.weights[1]).toBeCloseTo(1.857, 3);
    expect(goldAgreement(rewards).agree).toBe(12);
    expect(goldAgreement(rewardsFor(goldLabels()).rewards).agree).toBe(13);
  });

  it("uses the exact KL-regularized optimum", () => {
    const { rewards } = rewardsFor(DEFAULT_LABELS);
    const beta = 0.7;
    const optimum = tiltedPolicy(reference, rewards, beta);
    const objective = (p: number[]) => expectation(p, rewards) - beta * klDivergence(p, reference);
    // Any perturbation of the optimum along the simplex lowers the objective.
    for (let i = 0; i < 6; i += 1) {
      for (let j = 0; j < 6; j += 1) {
        if (i === j) continue;
        const shifted = [...optimum];
        const delta = Math.min(0.01, shifted[j]);
        shifted[i] += delta;
        shifted[j] -= delta;
        expect(objective(shifted)).toBeLessThanOrEqual(objective(optimum) + 1e-12);
      }
    }
  });

  it("shows the proxy rising while gold peaks and falls", () => {
    const { rewards } = rewardsFor(DEFAULT_LABELS);
    const sweep = betaSweep(reference, rewards);
    for (let k = 1; k < sweep.length; k += 1) {
      expect(sweep[k].reward).toBeGreaterThanOrEqual(sweep[k - 1].reward - 1e-12);
      expect(sweep[k].kl).toBeGreaterThanOrEqual(sweep[k - 1].kl - 1e-12);
    }
    const peak = sweep.reduce((best, point) => (point.gold > best.gold ? point : best));
    expect(expectation(reference, gold)).toBeCloseTo(0.487, 3);
    expect(peak.gold).toBeCloseTo(0.557, 3);
    expect(peak.kl).toBeCloseTo(0.2, 2);
    expect(sweep[sweep.length - 1].gold).toBeCloseTo(0.1, 2);
    const atTwo = tiltedPolicy(reference, rewards, 2);
    expect(expectation(atTwo, gold)).toBeCloseTo(0.53, 2);
    expect(atTwo[4]).toBeCloseTo(0.39, 2);
  });

  it("keeps gold high once pair 5 is labeled", () => {
    const { rewards } = rewardsFor(parseLabels("aaaaba"));
    const sweep = betaSweep(reference, rewards);
    const peak = sweep.reduce((best, point) => (point.gold > best.gold ? point : best));
    expect(peak.gold).toBeCloseTo(0.91, 2);
    expect(sweep[sweep.length - 1].gold).toBeCloseTo(0.85, 2);
  });
});

describe("post-training state", () => {
  it("drops the version 1 keys and keeps defaults", () => {
    const migrated = migratePostTrainingState({ preference: "a", strictness: 80, round: 4 });
    expect(migrated).toEqual(definition.initialState);
    expect(definition.hydrateState(JSON.stringify({ labels: "bbxx", beta: 999, sftSteps: -3 }))).toEqual({
      labels: "bbnnnn",
      beta: 20,
      sftSteps: 0,
    });
  });
});
