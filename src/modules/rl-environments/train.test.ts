import { describe, expect, it } from "vitest";
import { tinyRandom } from "@app/module-sdk";
import { CLOSE, MAX_TURNS, READ, UPDATE, features, outcomeOf, startState, strictGrader, transitions } from "./env";
import { klDivergence, referenceWeights, toolProbabilities, type Weights } from "./policy";
import {
  GROUP_RANGE,
  UPDATES,
  acrossSeeds,
  applyGroupUpdate,
  evaluateClass,
  evaluatePolicy,
  isFlat,
  sampleEpisode,
  sanitizeOptions,
  train,
  type Episode,
  type EpisodeStep,
  type TrainOptions,
} from "./train";

const DEFAULTS: TrainOptions = {
  reward: "outcome",
  checker: "strict",
  tickets: "six",
  group: 8,
  kl: 0,
  stepCost: 0,
  scale: "std",
  seed: 1,
};
const referenceAt = (values: ReadonlyArray<number>) => toolProbabilities(referenceWeights(), values);

/** An episode made by hand from a list of calls, with every known write saving. */
function scripted(weights: Weights, given: boolean, calls: number[]): Episode {
  let state = startState(given);
  const steps: EpisodeStep[] = [];
  for (let turn = 1; turn <= calls.length; turn += 1) {
    const before = state;
    const transition = transitions(state, calls[turn - 1])[0];
    state = transition.next;
    steps.push({
      turn,
      features: features(before),
      probabilities: toolProbabilities(weights, features(before)),
      tool: calls[turn - 1],
      saved: transition.saved,
      before,
      after: state,
    });
    if (transition.closed) return { steps, outcome: outcomeOf(state, true, turn) };
  }
  return { steps, outcome: outcomeOf(state, false, calls.length) };
}

describe("one group-relative update", () => {
  it("leaves all 48 weights exactly where they were when the group is flat and β is 0", () => {
    for (const scale of ["std", "none"] as const) {
      const weights = referenceWeights();
      const before = Float64Array.from(weights);
      const random = tinyRandom(3);
      const episodes = Array.from({ length: 4 }, () => sampleEpisode(weights, true, random));
      for (const rewards of [
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [0.1, 0.1, 0.1, 0.1],
        [-0.15, -0.15, -0.15, -0.15],
      ]) {
        expect(isFlat(rewards)).toBe(true);
        expect(applyGroupUpdate(weights, episodes, rewards, { scale, beta: 0, rate: 0.6, referenceAt })).toBe(true);
        expect(weights).toEqual(before);
      }
    }
  });

  it("still pulls a drifted policy toward the reference on a flat group when β is above 0", () => {
    const drifted = Float64Array.from(referenceWeights(), (value, index) => value + (index % 5 === 0 ? 1.5 : -0.5));
    const random = tinyRandom(4);
    const episodes = Array.from({ length: 8 }, () => sampleEpisode(drifted, true, random));
    const start = [1, 0, 1, 0, 0, 0, 0];
    const before = klDivergence(toolProbabilities(drifted, start), referenceAt(start));
    applyGroupUpdate(drifted, episodes, new Array(8).fill(0), { scale: "std", beta: 0.5, rate: 0.6, referenceAt });
    expect(klDivergence(toolProbabilities(drifted, start), referenceAt(start))).toBeLessThan(before);
  });

  it("makes the calls of the episode that beat its group likelier", () => {
    const weights = referenceWeights();
    const quit = scripted(weights, true, [READ, CLOSE]);
    const fixed = scripted(weights, true, [READ, UPDATE, CLOSE]);
    expect(strictGrader(fixed.outcome)).toBe(1);
    const afterRead = [1, 0, 1, 0, 0, 0, 0];
    const before = toolProbabilities(weights, afterRead)[UPDATE];
    expect(applyGroupUpdate(weights, [quit, fixed], [0, 1], { scale: "std", beta: 0, rate: 0.6, referenceAt })).toBe(false);
    expect(toolProbabilities(weights, afterRead)[UPDATE]).toBeGreaterThan(before);
  });
});

describe("training is deterministic in the seed", () => {
  it("returns the identical run for the same controls and seed, and a different path for another seed", () => {
    const a = train({ ...DEFAULTS, seed: 7 });
    const b = train({ ...DEFAULTS, seed: 7 });
    expect(b.points).toEqual(a.points);
    expect(b.weights).toEqual(a.weights);
    const c = train({ ...DEFAULTS, seed: 8 });
    expect(c.points[50].trueSuccess).not.toBe(a.points[50].trueSuccess);
  });

  it("records the reference plus one exact point per update", () => {
    const run = train(DEFAULTS);
    expect(run.points).toHaveLength(UPDATES + 1);
    expect(run.weights).toHaveLength(UPDATES + 1);
    expect(run.weights[0]).toEqual(referenceWeights());
    let flat = 0;
    for (const point of run.points) {
      for (const share of [point.trueSuccess, point.hackRate, point.closedNotFixed, point.needlessLookups, point.quitFirst]) {
        expect(share).toBeGreaterThanOrEqual(0);
        expect(share).toBeLessThanOrEqual(1 + 1e-12);
      }
      expect(point.meanTurns).toBeGreaterThanOrEqual(1);
      expect(point.meanTurns).toBeLessThanOrEqual(MAX_TURNS);
      expect(point.flatSoFar).toBeGreaterThanOrEqual(flat);
      flat = point.flatSoFar;
    }
    expect(run.points[UPDATES].flatSoFar).toBe(run.flatGroups);
  });

  it("evaluates only the final policy, with the same result, when detail is off", () => {
    const full = train({ ...DEFAULTS, seed: 5 });
    const quick = train({ ...DEFAULTS, seed: 5 }, false);
    expect(quick.points).toHaveLength(1);
    expect(quick.points[0]).toEqual(full.points[UPDATES]);
  });
});

describe("the exact evaluator", () => {
  /**
   * Samples many episodes and compares the sample means with the dynamic-programming expectations.
   * Each readout must fall within two standard errors at this fixed sampling seed.
   */
  it("agrees with Monte Carlo within two standard errors", () => {
    const sandboxRun = train({ ...DEFAULTS, checker: "sandbox" });
    const cases: Array<[string, Weights]> = [
      ["reference", referenceWeights()],
      ["sandbox run at update 5, mid-hack", sandboxRun.weights[5]],
      ["strict run at update 200", train(DEFAULTS).weights[UPDATES]],
    ];
    const design = { reward: "outcome" as const, checker: "strict" as const, stepCost: 0 };
    const samples = 4000;
    for (const [label, weights] of cases) {
      for (const given of [true, false]) {
        const exact = evaluateClass(weights, given, design);
        const random = tinyRandom(given ? 11 : 12);
        let strict = 0;
        let edited = 0;
        let turns = 0;
        let turnsSquared = 0;
        for (let index = 0; index < samples; index += 1) {
          const { outcome } = sampleEpisode(weights, given, random);
          strict += strictGrader(outcome);
          edited += outcome.edited ? 1 : 0;
          turns += outcome.turns;
          turnsSquared += outcome.turns ** 2;
        }
        const share = (count: number, p: number) => {
          const error = Math.sqrt(Math.max(p * (1 - p), 1e-12) / samples);
          return Math.abs(count / samples - p) <= 2 * error + 1e-9;
        };
        const meanTurns = turns / samples;
        const turnsError = Math.sqrt((turnsSquared / samples - meanTurns ** 2) / samples);
        expect(share(strict, exact.strict), `${label} given=${given} strict`).toBe(true);
        expect(share(edited, exact.edited), `${label} given=${given} edited`).toBe(true);
        expect(Math.abs(meanTurns - exact.turns) <= 2 * turnsError + 1e-9, `${label} given=${given} turns`).toBe(true);
      }
    }
  });

  it("starts from a reference that solves about one held-out ticket in five", () => {
    const start = evaluatePolicy(referenceWeights(), DEFAULTS);
    expect(start.trueSuccess).toBeGreaterThan(0.15);
    expect(start.trueSuccess).toBeLessThan(0.25);
    expect(start.klPerTurn).toBe(0);
  });
});

describe("bounded cost", () => {
  it("clamps every option to its control's range", () => {
    const hostile = sanitizeOptions({ ...DEFAULTS, group: 1e9, kl: Infinity, stepCost: -4, seed: NaN, updates: 1e9 });
    expect(hostile).toMatchObject({ group: GROUP_RANGE.max, kl: 0, stepCost: 0, seed: 1, updates: UPDATES });
    expect(sanitizeOptions({ ...DEFAULTS, kl: 9 }).kl).toBe(0.5);
  });

  it("finishes the largest run the controls allow, and its 10-seed strip, quickly", () => {
    const options: TrainOptions = { ...DEFAULTS, group: GROUP_RANGE.max, kl: 0.5, stepCost: 0.1, checker: "sandbox" };
    let started = performance.now();
    train(options);
    expect(performance.now() - started).toBeLessThan(1000);
    started = performance.now();
    acrossSeeds(options);
    expect(performance.now() - started).toBeLessThan(3000);
    // The strip is memoised on the controls, so a second request costs nothing.
    started = performance.now();
    expect(acrossSeeds({ ...options })).toBe(acrossSeeds(options));
    expect(performance.now() - started).toBeLessThan(50);
  });
});
