import { describe, expect, it } from "vitest";
import definition, { migrateRlState } from "./module";
import {
  BONUS,
  GOAL,
  START,
  TRAP,
  banditAverages,
  discountedReturn,
  greedyActions,
  qValue,
  rollout,
  runBandit,
  transitions,
  valueIteration,
} from "./mdp";

const plain = { gamma: 0.95, slip: 0, bonus: false };

describe("gridworld MDP", () => {
  it("keeps transition probabilities normalized, including wall bumps", () => {
    for (const slip of [0, 0.1, 0.3]) {
      for (const cell of [0, 4, 12, 15]) {
        const total = transitions(cell, "up", slip).reduce((sum, item) => sum + item.probability, 0);
        expect(total).toBeCloseTo(1, 12);
      }
    }
  });

  it("propagates value from G back to S one sweep per move", () => {
    const history = valueIteration(plain);
    // S is six moves from G around the trap, so V(S) only turns positive on sweep 6.
    expect(history[5].values[START]).toBeLessThan(0);
    expect(history[6].values[START]).toBeCloseTo(-0.04 * (1 + 0.95 + 0.95 ** 2 + 0.95 ** 3 + 0.95 ** 4) + 0.95 ** 5, 10);
    expect(history[30].residual).toBe(0);
  });

  it("satisfies the Bellman optimality equation at convergence", () => {
    const config = { gamma: 0.9, slip: 0.2, bonus: false };
    const values = valueIteration(config, 400)[400].values;
    for (let cell = 0; cell < 20; cell += 1) {
      if (cell === GOAL || cell === TRAP) continue;
      const best = Math.max(...(["up", "right", "down", "left"] as const).map((a) => qValue(values, cell, a, config)));
      expect(best).toBeCloseTo(values[cell], 9);
    }
  });

  it("obeys the contraction bound on the residual", () => {
    for (const config of [plain, { gamma: 0.9, slip: 0.3, bonus: true }]) {
      const history = valueIteration(config);
      for (let k = 2; k < history.length; k += 1) {
        expect(history[k].residual).toBeLessThanOrEqual(config.gamma * history[k - 1].residual + 1e-12);
      }
    }
  });

  it("matches the sampled return to V(S) with no slip and a converged table", () => {
    const values = valueIteration(plain)[30].values;
    const episode = rollout(values, plain, 3);
    expect(episode.outcome).toBe("goal");
    expect(episode.cells).toEqual([15, 10, 11, 12, 13, 14, 19]);
    expect(discountedReturn(episode.steps.map((s) => s.reward), 0.95)).toBeCloseTo(values[START], 12);
    expect(values[START]).toBeCloseTo(0.5928, 4);
  });

  it("turns away from the trap when moves can slip", () => {
    const calm = valueIteration(plain)[30].values;
    const slippery = { gamma: 0.95, slip: 0.2, bonus: false };
    const slipValues = valueIteration(slippery)[30].values;
    expect(greedyActions(calm, 16, plain)).toEqual(["up"]);
    expect(greedyActions(slipValues, 16, slippery)).toEqual(["left"]);
    expect(slipValues[START]).toBeCloseTo(0.314, 3);
    const rough = { gamma: 0.95, slip: 0.3, bonus: false };
    expect(greedyActions(calm, 12, plain)).toEqual(["right"]);
    expect(greedyActions(valueIteration(rough)[30].values, 12, rough)).toEqual(["up"]);
  });

  it("makes circling the bonus tile optimal only for a far-sighted discount", () => {
    for (const gamma of [0.5, 0.85, 0.88]) {
      const config = { gamma, slip: 0, bonus: true };
      expect(rollout(valueIteration(config)[30].values, config, 3).outcome).toBe("goal");
    }
    for (const gamma of [0.89, 0.95, 0.99]) {
      const config = { gamma, slip: 0, bonus: true };
      const episode = rollout(valueIteration(config)[30].values, config, 3);
      expect(episode.outcome).toBe("cap");
      expect(episode.cells.filter((cell) => cell === BONUS).length).toBeGreaterThan(5);
    }
    const converged = valueIteration({ gamma: 0.95, slip: 0, bonus: true }, 600)[600].values;
    expect(converged[BONUS]).toBeCloseTo((0.3 * 0.95 - 0.04) / (1 - 0.95 ** 2), 6);
  });
});

describe("ε-greedy bandit", () => {
  it("locks onto one arm with no exploration", () => {
    for (const seed of [1, 2, 3, 4]) {
      const run = runBandit(0, seed);
      expect(run.pulls.filter((count) => count > 0).length).toBeLessThanOrEqual(2);
    }
  });

  it("finds the best arm more often with a little exploration than with none or a lot", () => {
    const none = banditAverages(0).finalBestShare;
    const some = banditAverages(0.1).finalBestShare;
    const lots = banditAverages(0.5).finalBestShare;
    expect(none).toBeCloseTo(0.326, 2);
    expect(some).toBeCloseTo(0.717, 2);
    expect(lots).toBeCloseTo(0.565, 2);
    expect(some).toBeGreaterThan(lots);
    expect(lots).toBeGreaterThan(none);
  });
});

describe("rl-foundations state", () => {
  it("migrates a version 1 payload", () => {
    const migrated = migrateRlState({ policy: "greedy", hack: "on", step: 5, seed: 7 });
    expect(migrated).not.toHaveProperty("policy");
    expect(migrated.hack).toBe("on");
    expect(migrated.seed).toBe(7);
    expect(migrated.step).toBe(5);
    expect(migrated.gamma).toBe(0.95);
    expect(definition.hydrateState(JSON.stringify({ policy: "explore", hack: true, step: 99, seed: 50 }))).toMatchObject({
      hack: "on",
      step: 24,
      seed: 20,
    });
  });
});
