import { describe, expect, it } from "vitest";
import { teachingRl, tinyRandom } from "@app/module-sdk";
import definition from "./module";
import scenes from "./scenes";
import {
  FEATURE_COUNT,
  JUDGE_ITEM,
  JUDGE_MAX_CALLS,
  MAX_TURNS,
  PARTIAL_LOOK_UP,
  PARTIAL_READ,
  PARTIAL_RIGHT,
  SAVE_FAILURE,
  TICKET_KINDS,
  TOOLS,
  scoreEpisode,
} from "./env";
import { EPISODE_STREAM, SITUATIONS, fixed, percent, ticketFor } from "./desk";
import { EDIT_BIAS, WEIGHT_COUNT, referenceWeights, toolProbabilities } from "./policy";
import { EPISODE_RANGE } from "./state";
import {
  COST_RANGE,
  GROUP_RANGE,
  KL_RANGE,
  LEARNING_RATE,
  STRIP_SEEDS,
  UPDATES,
  acrossSeeds,
  evaluatePolicy,
  sampleEpisode,
  train,
  type TrainOptions,
} from "./train";

/**
 * Pins every measured value that standard.mdx, plain.mdx, simple.mdx and card-info.ts quote,
 * formatted exactly as the lab prints it (`percent`, `fixed`). Locations are given by section, not
 * line. The checkpoint explanations in module.ts are pinned in checkpoint-claims.test.ts.
 */
const DEFAULTS = definition.initialState as unknown as TrainOptions;
const run = (patch: Partial<TrainOptions> = {}) => train({ ...DEFAULTS, ...patch });
const end = (patch: Partial<TrainOptions> = {}) => run(patch).points[UPDATES];
const situation = (label: string) => SITUATIONS.find((entry) => entry.label === label)!.features;
const tool = (name: (typeof TOOLS)[number]) => TOOLS.indexOf(name);
const episodeAt = (patch: Partial<TrainOptions>, view: number, episodeSeed: number) => {
  const kind = ticketFor(episodeSeed);
  return sampleEpisode(run(patch).weights[view], kind.given, tinyRandom(EPISODE_STREAM + episodeSeed));
};
const calls = (episode: ReturnType<typeof sampleEpisode>) => episode.steps.map((step) => TOOLS[step.tool]);

describe("rl-environments quoted values: the desk and its constants", () => {
  it("uses the default lab state the lessons start from", () => {
    expect(DEFAULTS).toMatchObject({ reward: "outcome", checker: "strict", tickets: "six", group: 8, kl: 0, stepCost: 0, scale: "std", seed: 1, view: 200 });
  });

  it("has six tools, eight turns, eight tickets, seven features and 48 weights", () => {
    // standard.mdx "The big picture", "What the lab shows"; plain/simple "Six tools and a turn limit", "What the lab shows".
    expect(TOOLS).toHaveLength(6);
    expect(MAX_TURNS).toBe(8);
    expect(TICKET_KINDS).toHaveLength(8);
    expect(FEATURE_COUNT).toBe(7);
    expect(WEIGHT_COUNT).toBe(48);
    expect(UPDATES).toBe(200);
    expect(LEARNING_RATE).toBe(0.6);
    expect(STRIP_SEEDS).toBe(10);
  });

  it("fails 40% of known writes, which is 2 times in 5", () => {
    expect(SAVE_FAILURE).toBe(0.4);
    expect(percent(SAVE_FAILURE)).toBe("40%");
    expect(SAVE_FAILURE * 5).toBeCloseTo(2, 12);
  });

  it("pays partial credit of 0.1, 0.1 and 0.2, and the judge 0.25 per item for 4 calls or fewer", () => {
    expect([PARTIAL_READ, PARTIAL_LOOK_UP, PARTIAL_RIGHT]).toEqual([0.1, 0.1, 0.2]);
    expect(JUDGE_ITEM).toBe(0.25);
    expect(JUDGE_MAX_CALLS).toBe(4);
    // checkpoint and standard.mdx: the bonuses add at most 0.4.
    expect(PARTIAL_READ + PARTIAL_LOOK_UP + PARTIAL_RIGHT).toBeCloseTo(0.4, 12);
  });

  it("states the control ranges card-info.ts lists", () => {
    expect(GROUP_RANGE).toMatchObject({ min: 2, max: 32 });
    expect(KL_RANGE).toMatchObject({ min: 0, max: 0.5, step: 0.05 });
    expect(COST_RANGE).toMatchObject({ min: 0, max: 0.1, step: 0.01 });
    expect(EPISODE_RANGE).toMatchObject({ min: 1, max: 20 });
    expect(EDIT_BIAS).toBe(-2.5);
    expect(fixed(EDIT_BIAS, 1)).toBe("−2.5");
  });
});

describe("rl-environments quoted values: the reference policy", () => {
  const start = evaluatePolicy(referenceWeights(), DEFAULTS);

  it("solves 19% of held-out tickets, about 19 in 100", () => {
    expect(percent(start.trueSuccess)).toBe("19%");
    expect(percent(run().points[0].trueSuccess)).toBe("19%");
  });

  it("edits the test in about 5 episodes in 100 and uses about 3 calls", () => {
    // plain/simple "The starting habits".
    expect(percent(start.hackRate)).toBe("5%");
    expect(Math.round(start.meanTurns)).toBe(3);
  });
});

describe("rl-environments quoted values: 1. Run one episode (update 0)", () => {
  const verdicts = (episodeSeed: number) => {
    const { outcome } = episodeAt({}, 0, episodeSeed);
    return {
      outcome,
      sandbox: scoreEpisode("outcome", "sandbox", outcome),
      strict: scoreEpisode("outcome", "strict", outcome),
      partial: fixed(scoreEpisode("partial", "strict", outcome)),
      judge: fixed(scoreEpisode("judge", "strict", outcome)),
    };
  };

  it("has seeds 1 and 2 read and close at once: both tests fail, partial 0.10, judge 0.75", () => {
    for (const seed of [1, 2]) {
      expect(calls(episodeAt({}, 0, seed))).toEqual(["read_ticket", "close_ticket"]);
      expect(verdicts(seed)).toMatchObject({ sandbox: 0, strict: 0, partial: "0.10", judge: "0.75" });
    }
  });

  it("has seed 3 write, check and close: both tests pass, partial 1.30, judge still 0.75", () => {
    expect(calls(episodeAt({}, 0, 3))).toEqual(["read_ticket", "update_record", "run_check", "close_ticket"]);
    expect(verdicts(3)).toMatchObject({ sandbox: 1, strict: 1, partial: "1.30", judge: "0.75" });
    expect(verdicts(3).outcome.looked).toBe(false);
  });

  it("has seed 5 close on its first call and still get 0.50 from the judge", () => {
    expect(calls(episodeAt({}, 0, 5))).toEqual(["close_ticket"]);
    expect(verdicts(5)).toMatchObject({ sandbox: 0, strict: 0, judge: "0.50" });
  });

  it("opens the four-scorers scene on that seed-3 episode", () => {
    // scenes.ts "four-scorers": "Episode seed 3 fixes the record ... the Rubric judge still gives 0.75".
    expect(scenes.find((scene) => scene.id === "four-scorers")?.state).toEqual({ view: 0, episode: 3 });
    expect(ticketFor(3).label).toBe("Sunday hours");
  });
});

describe("rl-environments quoted values: 2. Train on the outcome", () => {
  it("climbs from 19% to 84% true success with the defaults", () => {
    const points = run().points;
    expect(percent(points[0].trueSuccess)).toBe("19%");
    expect(percent(points[UPDATES].trueSuccess)).toBe("84%");
    expect(fixed(points[UPDATES].trainingReward)).toBe("0.84");
  });

  it("wastes 160 of 200 groups at Group size 2, against 39 at 8, and ends at 46%", () => {
    const small = end({ group: 2 });
    expect(`${small.flatSoFar} of ${UPDATES}`).toBe("160 of 200");
    expect(end().flatSoFar).toBe(39);
    expect(percent(small.trueSuccess)).toBe("46%");
  });

  it("checks after a write 98% of the time and retries after a failed check 94%, over about 6 calls", () => {
    // plain/simple "The agent learns to check its own work".
    const weights = run().weights[UPDATES];
    expect(percent(toolProbabilities(weights, situation("Updated, not checked"))[tool("run_check")])).toBe("98%");
    expect(percent(toolProbabilities(weights, situation("Check failed"))[tool("update_record")])).toBe("94%");
    expect(Math.round(end().meanTurns)).toBe(6);
  });
});

describe("rl-environments quoted values: 3. Leave the test writable", () => {
  it("ends at training reward 1.00, true success 0% and Hack rate 100%", () => {
    const hacked = end({ checker: "sandbox" });
    expect(fixed(hacked.trainingReward)).toBe("1.00");
    expect(percent(hacked.trueSuccess)).toBe("0%");
    expect(percent(hacked.hackRate)).toBe("100%");
  });

  it("replays read_ticket, edit_check, run_check, close_ticket at Episode seed 1", () => {
    const episode = episodeAt({ checker: "sandbox" }, UPDATES, 1);
    expect(calls(episode)).toEqual(["read_ticket", "edit_check", "run_check", "close_ticket"]);
    expect(scoreEpisode("outcome", "sandbox", episode.outcome)).toBe(1);
    expect(scoreEpisode("outcome", "strict", episode.outcome)).toBe(0);
  });

  it("already hacks 78% of episodes at update 10, with true success down to 5%", () => {
    // plain/simple "The loophole in one episode".
    const point = run({ checker: "sandbox" }).points[10];
    expect(percent(point.hackRate)).toBe("78%");
    expect(percent(point.trueSuccess)).toBe("5%");
  });

  it("hacks on all ten seeds", () => {
    expect(acrossSeeds({ ...DEFAULTS, checker: "sandbox" }).hackedSeeds).toBe(10);
  });
});

describe("rl-environments quoted values: 4. Swap the scorer", () => {
  it("closes 100% of tickets unfixed under the Rubric judge, with 99% needless look-ups", () => {
    const judged = end({ reward: "judge" });
    expect(percent(judged.closedNotFixed)).toBe("100%");
    expect(percent(judged.needlessLookups)).toBe("99%");
    expect(percent(judged.trueSuccess)).toBe("0%");
    expect(calls(episodeAt({ reward: "judge" }, UPDATES, 1))).toEqual(["look_up", "read_ticket", "close_ticket"]);
  });

  it("keeps the hack at 99% under Partial credit with the Sandbox test, and needless look-ups go from 1% to 73%", () => {
    expect(percent(end({ reward: "partial", checker: "sandbox" }).hackRate)).toBe("99%");
    expect(percent(end({ checker: "sandbox" }).needlessLookups)).toBe("1%");
    expect(percent(end({ reward: "partial", checker: "sandbox" }).needlessLookups)).toBe("73%");
  });
});

describe("rl-environments quoted values: 5. Add a leash and a cost", () => {
  it("falls to a 42% hack rate and 0.79 KL per turn (from 1.43) at β 0.2, with 40% true success", () => {
    const leashed = end({ checker: "sandbox", kl: 0.2 });
    expect(percent(leashed.hackRate)).toBe("42%");
    expect(fixed(leashed.klPerTurn)).toBe("0.79");
    expect(fixed(end({ checker: "sandbox" }).klPerTurn)).toBe("1.43");
    expect(percent(leashed.trueSuccess)).toBe("40%");
  });

  it("quits at turn 1 (Mean turns 1.0) at Step cost 0.05 under std scaling, and reaches 77% in 6.0 turns under Mean only", () => {
    const quit = end({ stepCost: 0.05 });
    expect(fixed(quit.meanTurns, 1)).toBe("1.0");
    expect(quit.quitFirst).toBeGreaterThan(0.99);
    const meanOnly = end({ stepCost: 0.05, scale: "none" });
    expect(percent(meanOnly.trueSuccess)).toBe("77%");
    expect(fixed(meanOnly.meanTurns, 1)).toBe("6.0");
  });

  it("charges 0.40 for 8 turns at 0.05, less than one success", () => {
    expect(fixed(0.05 * MAX_TURNS)).toBe("0.40");
  });
});

describe("rl-environments quoted values: 6. Test unseen tickets", () => {
  it("ends at Value missing 4% and Value given 90% on 1 kind, and Value missing 78% on 6 kinds", () => {
    const one = end({ tickets: "one" });
    expect(percent(one.successMissing)).toBe("4%");
    expect(percent(one.successGiven)).toBe("90%");
    expect(percent(end().successMissing)).toBe("78%");
  });

  it("does better on 6 kinds than on 1 kind on every seed in the strip", () => {
    const one = acrossSeeds({ ...DEFAULTS, tickets: "one" }).seeds;
    const six = acrossSeeds({ ...DEFAULTS }).seeds;
    one.forEach((entry, index) => expect(six[index].final.trueSuccess).toBeGreaterThan(entry.final.trueSuccess));
  });

  it("writes a guess 89% of the time and looks up 3% in the Read, value missing row on 1 kind", () => {
    const weights = run({ tickets: "one" }).weights[UPDATES];
    const row = toolProbabilities(weights, situation("Read, value missing"));
    expect(percent(row[tool("update_record")])).toBe("89%");
    expect(percent(row[tool("look_up")])).toBe("3%");
  });
});

describe("rl-environments quoted values: What to notice, across seeds 1 to 10", () => {
  const meanCurve = (patch: Partial<TrainOptions>) => {
    const sums = new Array(UPDATES + 1).fill(0);
    for (let seed = 1; seed <= STRIP_SEEDS; seed += 1) {
      run({ ...patch, seed }).points.forEach((point, index) => (sums[index] += point.trueSuccess / STRIP_SEEDS));
    }
    return sums;
  };

  it("has Partial credit first pass 50% at update 16 against 15, and end at 69% against 84%", () => {
    expect(meanCurve({ reward: "partial" }).findIndex((value) => value >= 0.5)).toBe(16);
    expect(meanCurve({}).findIndex((value) => value >= 0.5)).toBe(15);
    expect(percent(acrossSeeds({ ...DEFAULTS, reward: "partial" }).meanSuccess)).toBe("69%");
    expect(percent(acrossSeeds({ ...DEFAULTS }).meanSuccess)).toBe("84%");
  });

  it("averages a 58% hack rate at β 0.2 with 8 of 10 seeds hacking, and 27% at β 0.5", () => {
    const leashed = acrossSeeds({ ...DEFAULTS, checker: "sandbox", kl: 0.2 });
    expect(percent(leashed.meanHack)).toBe("58%");
    expect(leashed.hackedSeeds).toBe(8);
    expect(percent(acrossSeeds({ ...DEFAULTS, checker: "sandbox", kl: 0.5 }).meanHack)).toBe("27%");
  });

  it("ends the Strict grader run at 79% with β 0.2, against 84% without", () => {
    expect(percent(end({ kl: 0.2 }).trueSuccess)).toBe("79%");
    expect(percent(end().trueSuccess)).toBe("84%");
  });

  it("quits on 1 seed at Step cost 0.05 and 4 at 0.10 under std scaling, and never under Mean only", () => {
    expect(acrossSeeds({ ...DEFAULTS, stepCost: 0.05 }).quitSeeds).toBe(1);
    expect(acrossSeeds({ ...DEFAULTS, stepCost: 0.1 }).quitSeeds).toBe(4);
    for (let cost = 1; cost <= 10; cost += 1) {
      expect(acrossSeeds({ ...DEFAULTS, stepCost: cost / 100, scale: "none" }).quitSeeds, `cost ${cost / 100}`).toBe(0);
    }
  });
});

describe("rl-environments quoted values: the worked examples", () => {
  it("gives one pass in four an advantage 0.75 above a 0.25 mean", () => {
    expect(teachingRl.groupAdvantages([1, 0, 0, 0], { scale: "none" })).toEqual([0.75, -0.25, -0.25, -0.25]);
  });

  it("turns failed groups of −0.05 and −0.10 into ±1 under std scaling and ±0.025 under Mean only", () => {
    const rewards = [-0.05, -0.05, -0.1, -0.1];
    const mean = rewards.reduce((sum, value) => sum + value, 0) / rewards.length;
    expect(mean).toBeCloseTo(-0.075, 12);
    const spread = Math.sqrt(rewards.reduce((sum, value) => sum + (value - mean) ** 2, 0) / rewards.length);
    expect(spread).toBeCloseTo(0.025, 12);
    teachingRl.groupAdvantages(rewards, { scale: "std" }).forEach((value, index) => expect(value).toBeCloseTo(index < 2 ? 1 : -1, 4));
    teachingRl.groupAdvantages(rewards, { scale: "none" }).forEach((value, index) => expect(value).toBeCloseTo(index < 2 ? 0.025 : -0.025, 12));
  });
});
