import { describe, expect, it } from "vitest";
import { checkpointQuestions, teachingRl, tinyRandom } from "@app/module-sdk";
import definition from "./module";
import { CHECK, CLOSE, EDIT, LOOK_UP, MAX_TURNS, PARTIAL_LOOK_UP, PARTIAL_READ, PARTIAL_RIGHT, READ, UPDATE, outcomeOf, scoreEpisode, startState, transitions } from "./env";
import { fixed, percent } from "./desk";
import { referenceWeights, toolProbabilities } from "./policy";
import { STRIP_SEEDS, UPDATES, acrossSeeds, applyGroupUpdate, sampleEpisode, train, type TrainOptions } from "./train";

/**
 * The factual claims in the checkpoint explanations (module.ts), checked against the lab. Claims
 * about one run are checked at seed 1, the lab's default, and the general claims are checked on
 * every seed from 1 to 10, so a change to the lab cannot silently invalidate an explanation.
 */
const questions = checkpointQuestions(definition);
const correct = (index: number) => questions[index].options[questions[index].answer];
const DEFAULTS = definition.initialState as unknown as TrainOptions;
const end = (patch: Partial<TrainOptions> = {}) => train({ ...DEFAULTS, ...patch }).points[UPDATES];
const strip = (patch: Partial<TrainOptions> = {}) => acrossSeeds({ ...DEFAULTS, ...patch }).seeds.map((entry) => entry.final);
const referenceAt = (values: ReadonlyArray<number>) => toolProbabilities(referenceWeights(), values);

/** Plays a fixed list of calls with every known write saving. */
function play(given: boolean, calls: number[]) {
  let state = startState(given);
  for (let turn = 1; turn <= calls.length; turn += 1) {
    const transition = transitions(state, calls[turn - 1])[0];
    state = transition.next;
    if (transition.closed) return outcomeOf(state, true, turn);
  }
  return outcomeOf(state, false, calls.length);
}

describe("rl-environments checkpoint claims", () => {
  it("has seven questions covering all five objectives", () => {
    expect(questions).toHaveLength(7);
    expect(new Set(questions.map((question) => question.objective))).toEqual(new Set([0, 1, 2, 3, 4]));
  });

  it("question 1: the edited test passes the Sandbox test and fails the Strict grader, and neither needs run_check", () => {
    expect(correct(0)).toMatch(/^The sandbox test runs the edited file/);
    const hacked = play(true, [EDIT, CLOSE]);
    expect(scoreEpisode("outcome", "sandbox", hacked)).toBe(1);
    expect(scoreEpisode("outcome", "strict", hacked)).toBe(0);
    // No run_check and no transcript rule: a fixed record passes the Strict grader in three calls.
    const fixedRecord = play(true, [READ, UPDATE, CLOSE]);
    expect(scoreEpisode("outcome", "strict", fixedRecord)).toBe(1);
    expect(play(true, [READ, UPDATE, CHECK, CLOSE]).turns).toBe(4);
    // Across seeds 1 to 10, training against the sandbox test hacks and the Strict grader fails it.
    for (const point of strip({ checker: "sandbox" })) {
      expect(point.hackRate).toBeGreaterThan(0.9);
      expect(point.trueSuccess).toBeLessThan(0.02);
    }
  });

  it("question 2: a flat group leaves the weights alone at β 0, and Group size 2 wastes more groups on every seed", () => {
    expect(correct(1)).toMatch(/^Nothing: both episodes have zero advantage/);
    const weights = referenceWeights();
    const before = Float64Array.from(weights);
    const random = tinyRandom(2);
    const pair = [sampleEpisode(weights, true, random), sampleEpisode(weights, true, random)];
    for (const scale of ["std", "none"] as const) {
      applyGroupUpdate(weights, pair, [1, 1], { scale, beta: 0, rate: 0.6, referenceAt });
      applyGroupUpdate(weights, pair, [0, 0], { scale, beta: 0, rate: 0.6, referenceAt });
    }
    expect(weights).toEqual(before);
    expect(end({ group: 2 }).flatSoFar).toBe(160);
    expect(end().flatSoFar).toBe(39);
    const small = strip({ group: 2 });
    strip().forEach((point, index) => expect(small[index].flatSoFar).toBeGreaterThan(point.flatSoFar));
  });

  it("question 3: the Rubric judge pays 1.00 for read, look-up, close and is gamed on every seed", () => {
    expect(correct(2)).toMatch(/^Rubric judge, because reading, looking up and closing/);
    const shortcut = play(true, [READ, LOOK_UP, CLOSE]);
    expect(shortcut.right).toBe(false);
    expect(fixed(scoreEpisode("judge", "strict", shortcut))).toBe("1.00");
    const judged = end({ reward: "judge" });
    expect(percent(judged.closedNotFixed)).toBe("100%");
    expect(percent(judged.trueSuccess)).toBe("0%");
    for (const point of strip({ reward: "judge" })) {
      expect(point.closedNotFixed).toBeGreaterThan(0.95);
      expect(point.trueSuccess).toBeLessThan(0.02);
    }
    // Outcome only with the Strict grader pays only for a right, untouched record; partial bonuses add at most 0.4.
    expect(scoreEpisode("outcome", "strict", shortcut)).toBe(0);
    expect(PARTIAL_READ + PARTIAL_LOOK_UP + PARTIAL_RIGHT).toBeCloseTo(0.4, 12);
  });

  it("question 4: β 0.2 lowers the hack rate on every seed without removing it, and true success stays below the Strict run", () => {
    expect(correct(3)).toMatch(/^Hack rate falls but stays far above zero/);
    expect(percent(end({ checker: "sandbox", kl: 0.2 }).hackRate)).toBe("42%");
    expect(percent(end({ checker: "sandbox" }).hackRate)).toBe("100%");
    expect(percent(acrossSeeds({ ...DEFAULTS, checker: "sandbox", kl: 0.2 }).meanHack)).toBe("58%");
    expect(percent(end({ checker: "sandbox", kl: 0.2 }).trueSuccess)).toBe("40%");
    expect(percent(end().trueSuccess)).toBe("84%");
    expect(referenceWeights()[4 * 8]).toBe(-2.5);
    const free = strip({ checker: "sandbox" });
    const strictRuns = strip();
    strip({ checker: "sandbox", kl: 0.2 }).forEach((point, index) => {
      expect(point.hackRate).toBeGreaterThan(0.25);
      expect(point.hackRate).toBeLessThan(free[index].hackRate);
      expect(point.trueSuccess).toBeLessThan(strictRuns[index].trueSuccess);
    });
  });

  it("question 5: under std scaling an all-fail group turns the step cost into full-size advantages, and Mean only does not", () => {
    expect(correct(4)).toMatch(/^In an all-fail group the cost is the only difference/);
    const quit = end({ stepCost: 0.05 });
    expect(quit.quitFirst).toBeGreaterThan(0.99);
    expect(fixed(quit.meanTurns, 1)).toBe("1.0");
    expect(end({ stepCost: 0.05, scale: "none" }).quitFirst).toBeLessThan(0.05);
    // A failed group's rewards are a few hundredths apart; the std division makes them about ±1.
    const failed = [-0.05, -0.1, -0.15, -0.05, -0.1, -0.2, -0.05, -0.1];
    const scaled = teachingRl.groupAdvantages(failed, { scale: "std" });
    expect(Math.max(...scaled.map(Math.abs))).toBeGreaterThan(1);
    expect(Math.max(...teachingRl.groupAdvantages(failed, { scale: "none" }).map(Math.abs))).toBeLessThan(0.15);
    // The cost never outweighs a pass: 8 turns at 0.05 cost 0.40 against a reward of 1.
    expect(fixed(0.05 * MAX_TURNS)).toBe("0.40");
    for (const cost of [0.05, 0.1]) {
      expect(acrossSeeds({ ...DEFAULTS, stepCost: cost }).quitSeeds).toBeGreaterThan(0);
      expect(acrossSeeds({ ...DEFAULTS, stepCost: cost, scale: "none" }).quitSeeds).toBe(0);
    }
  });

  it("question 6: with 1 kind the Value missing ticket fails on every seed, and 6 kinds fix it on every seed", () => {
    expect(correct(5)).toMatch(/^No training ticket left a value out/);
    expect(percent(end({ tickets: "one" }).successMissing)).toBe("4%");
    expect(percent(end().successMissing)).toBe("78%");
    for (const point of strip({ tickets: "one" })) expect(point.successMissing).toBeLessThan(0.15);
    for (const point of strip()) expect(point.successMissing).toBeGreaterThan(0.5);
    expect(STRIP_SEEDS).toBe(10);
  });

  it("question 7: on 6 kinds a held-out ticket looks exactly like a training ticket of its class", () => {
    expect(correct(6)).toMatch(/^Held-out tickets differ only in wording/);
    for (const point of strip()) expect(point.trainingSuccess).toBeCloseTo(point.trueSuccess, 12);
  });

  it("does not make the correct option the longest in most questions", () => {
    // Same rule as registry.test.ts: a hit is a correct option that is the unique longest.
    const longest = questions.filter((question) => {
      const lengths = question.options.map((option) => option.length);
      const best = Math.max(...lengths);
      return lengths[question.answer] === best && lengths.filter((length) => length === best).length === 1;
    }).length;
    expect(longest).toBeLessThanOrEqual(Math.floor(questions.length / 2));
  });
});
