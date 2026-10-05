import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { teachingEval, type ModuleState } from "@app/module-sdk";
import definition from "./module";
import { parseItems } from "./eval";
import {
  ATTEMPTS_RANGE,
  BUDGETS,
  DEFAULT_ITEMS,
  LEAKED_TEXT,
  MAX_ITEMS,
  MAX_LINE_LENGTH,
  RULES,
  SAMPLES_PER_ITEM,
  SAMPLE_SEED,
  TRAINING_TEXT,
  drawCorrect,
  eloUpdate,
  itemsNeeded,
  percent,
  rankCheckpoints,
  rateCheckpoints,
  trainCheckpoints,
  type Rule,
} from "./scoring";

/**
 * The scoring-rule figures in standard.mdx and plain.mdx ("How to play with it", "What to notice", "Where it breaks",
 * "Going deeper"), card-info.ts ("Leaderboard" howItWorks, controls and notice) and module.ts (checkpoint questions 1-3
 * and the glossary). The default items are scored on the clean 3, 10, 30 and 80 epoch checkpoints, formatted the way the
 * lab's table and forest plot format them.
 */
const items = parseItems(DEFAULT_ITEMS.join("\n"));
const clean = trainCheckpoints(TRAINING_TEXT, items, 1);
const leaked = trainCheckpoints(LEAKED_TEXT, items, 1);
const board = (rule: Rule, attempts = 1) => rateCheckpoints(clean, rule, attempts);
const order = (rows: ReturnType<typeof board>) => rankCheckpoints(rows).map((row) => row.epochs);
const at = (rows: ReturnType<typeof board>, epochs: number) => rows.find((row) => row.epochs === epochs)!;

describe("evaluation quoted values: scoring rules", () => {
  it("keeps the length-normalised board the lesson already quoted", () => {
    const rows = board("normalized");
    expect(rows.map((row) => row.epochs)).toEqual([...BUDGETS]);
    expect(rows.map((row) => row.passes)).toEqual([2, 2, 3, 3]);
    expect(order(rows).slice(0, 2)).toEqual([80, 30]);
    expect(at(rows, 80).side.mean).toBeCloseTo(0.012, 3);
    expect(at(rows, 30).side.mean).toBeCloseTo(0.011, 3);
    expect(at(rows, 80).interval.map(percent)).toEqual(["19%", "81%"]);
    expect(itemsNeeded(rankCheckpoints(rows)[0], 0.05)).toBe(385);
  });

  it("gives every checkpoint two more passes under raw log-probability (step 3, question 1)", () => {
    const normalized = board("normalized");
    const raw = board("raw");
    for (const epochs of BUDGETS) expect(at(raw, epochs).passes - at(normalized, epochs).passes, `${epochs} epochs`).toBe(2);
    expect(at(normalized, 80).passes).toBe(3);
    expect(at(raw, 80).passes).toBe(5);
    // "the 30-epoch checkpoint edges ahead of it".
    expect(order(raw).slice(0, 2)).toEqual([30, 80]);
  });

  it("attributes the two extra raw passes to the two items whose correct option is shorter (question 1)", () => {
    const shorter = items.filter((item) => item.correct.length < item.distractor.length).map((item) => item.correct);
    expect(shorter).toEqual(["sails", "repeats"]);
    const normalized = at(board("normalized"), 80).items;
    const raw = at(board("raw"), 80).items;
    const gained = raw.filter((item, index) => item.passed && !normalized[index].passed).map((item) => item.correct);
    expect(gained.sort()).toEqual(["repeats", "sails"]);
  });

  it("puts the 10-epoch checkpoint first and the 80-epoch one third at k = 1", () => {
    const rows = board("sample", 1);
    expect(order(rows)).toEqual([10, 30, 80, 3]);
    expect(rankCheckpoints(rows)[0].epochs).toBe(10);
  });

  it("puts the 3-epoch checkpoint first at 94% and the 80-epoch one last at 83% at k = 5 (step 3, question 2)", () => {
    const rows = board("sample", 5);
    const ranked = rankCheckpoints(rows);
    expect(ranked[0].epochs).toBe(3);
    expect(ranked[ranked.length - 1].epochs).toBe(80);
    expect(percent(at(rows, 3).score)).toBe("94%");
    expect(percent(at(rows, 80).score)).toBe("83%");
    // At k = 1 the least-trained checkpoint is last: the climb the question describes.
    expect(order(board("sample", 1)).pop()).toBe(3);
  });

  it("is confidently wrong on port at 80 epochs, so extra attempts never help on that item", () => {
    const port = (rows: ReturnType<typeof board>, epochs: number) => at(rows, epochs).items.find((item) => item.correct === "port")!;
    expect(port(board("sample", 5), 80).correctSamples).toBe(0);
    expect(port(board("sample", 10), 80).score).toBe(0);
    // The 3-epoch checkpoint spreads probability over both options, so it does hit.
    expect(port(board("sample", 5), 3).correctSamples).toBeGreaterThan(0);
    expect(port(board("sample", 5), 3).score).toBeGreaterThan(0.5);
  });

  it("leaves every interval overlapping the leader's under all three rules", () => {
    for (const [rule, attempts] of [["normalized", 1], ["raw", 1], ["sample", 1], ["sample", 5]] as const) {
      const ranked = rankCheckpoints(board(rule, attempts));
      const overlapping = ranked.slice(1).filter((row) => row.interval[1] >= ranked[0].interval[0]);
      expect(overlapping, `${rule} k=${attempts}`).toHaveLength(3);
    }
  });
});

describe("sample and check uses real, repeatable draws", () => {
  it("scores each item with the unbiased pass@k of its own seeded draws", () => {
    const row = at(board("sample", 5), 30);
    row.items.forEach((item, index) => {
      const share = 1 / (1 + Math.exp(item.distractorTotal - item.correctTotal));
      expect(item.share).toBeCloseTo(share, 12);
      const drawn = drawCorrect(share, SAMPLES_PER_ITEM, SAMPLE_SEED + index);
      expect(item.correctSamples).toBe(drawn);
      expect(item.score).toBeCloseTo(teachingEval.passAtK(SAMPLES_PER_ITEM, drawn, 5), 12);
      expect(item.side).toBe(drawn / SAMPLES_PER_ITEM);
    });
  });

  it("is deterministic, and reuses the same dice for every checkpoint and for the leaked run", () => {
    expect(board("sample", 5)).toEqual(board("sample", 5));
    // A higher share can only turn draws into passes, never the reverse, because the uniform numbers are shared.
    for (let index = 0; index < items.length; index += 1) {
      const counts = BUDGETS.map((epochs) => ({
        share: at(board("sample", 1), epochs).items[index].share,
        count: at(board("sample", 1), epochs).items[index].correctSamples,
      })).sort((a, b) => a.share - b.share);
      for (let i = 1; i < counts.length; i += 1) expect(counts[i].count).toBeGreaterThanOrEqual(counts[i - 1].count);
    }
  });

  it("draws close to the share over many samples, and gives pass@1 equal to the share of correct draws", () => {
    expect(drawCorrect(0.3, 20000, 5) / 20000).toBeCloseTo(0.3, 1);
    expect(drawCorrect(0, 50, 1)).toBe(0);
    expect(drawCorrect(1, 50, 1)).toBe(50);
    const row = at(board("sample", 1), 80);
    row.items.forEach((item) => expect(item.score).toBeCloseTo(item.correctSamples / SAMPLES_PER_ITEM, 12));
  });

  it("keeps pass@k non-decreasing in k for every checkpoint", () => {
    for (const epochs of BUDGETS) {
      let previous = -1;
      for (let attempts = ATTEMPTS_RANGE.min; attempts <= ATTEMPTS_RANGE.max; attempts += 1) {
        const score = at(board("sample", attempts), epochs).score;
        expect(score).toBeGreaterThanOrEqual(previous);
        previous = score;
      }
    }
  });
});

describe("contamination under every rule", () => {
  it("raises the leaked run's headline score under all three rules at the largest checkpoint", () => {
    for (const rule of RULES) {
      const cleanRow = at(rateCheckpoints(clean, rule, 5), 80);
      const leakedRow = at(rateCheckpoints(leaked, rule, 5), 80);
      expect(leakedRow.score, rule).toBeGreaterThanOrEqual(cleanRow.score);
    }
  });
});

describe("evaluation quoted values: applications section", () => {
  it("matches the golden-set worked example: 27 of 30 is 90% with an interval of 74% to 97%", () => {
    expect(27 / 30).toBeCloseTo(0.9, 12);
    expect(teachingEval.wilsonInterval(27, 30).map(percent)).toEqual(["74%", "97%"]);
  });

  it("matches the Elo worked example: equal ratings at K = 32 move 16 points, a 400-point favourite under 3", () => {
    const equal = eloUpdate(1000, 1000, true);
    expect(equal.a - 1000).toBe(16);
    expect(1000 - equal.b).toBe(16);
    const favourite = eloUpdate(1400, 1000, true);
    expect(favourite.a - 1400).toBeGreaterThan(2.9);
    expect(favourite.a - 1400).toBeLessThan(3);
    // An upset is worth the rest of K: the loser's loss equals the winner's gain.
    const upset = eloUpdate(1000, 1400, true);
    expect(upset.a - 1000).toBeCloseTo(32 - (favourite.a - 1400), 10);
    expect(upset.a + upset.b).toBeCloseTo(2400, 10);
  });
});

describe("evaluation state", () => {
  const hydrate = (value: string) => definition.hydrateState(value);

  it("round-trips every state the controls can produce, and keeps the shipped defaults", () => {
    const states: ModuleState[] = [
      definition.initialState,
      { ...definition.initialState, rule: "raw" },
      { ...definition.initialState, rule: "sample", attempts: 10, contaminated: true, epochsScale: 3 },
      { ...definition.initialState, rule: "sample", attempts: 1, epochsScale: 0.25 },
    ];
    for (const state of states) expect(hydrate(definition.serializeState(state))).toEqual(state);
    expect(definition.initialState).toMatchObject({ rule: "normalized", attempts: 1, contaminated: false, epochsScale: 1 });
    expect(definition.stateVersion).toBe(2);
  });

  it("translates a version 1 payload that predates the scoring rule", () => {
    const old = JSON.stringify({ items: "a b | c | d", contaminated: true, epochsScale: 2 });
    expect(hydrate(old)).toEqual({ items: "a b | c | d", contaminated: true, epochsScale: 2, rule: "normalized", attempts: 1 });
  });

  it("clamps and validates hostile payloads", () => {
    const hostile = hydrate(
      JSON.stringify({ rule: "bogus", attempts: 1e9, epochsScale: 1e9, contaminated: "yes", items: 42 }),
    );
    expect(hostile.rule).toBe("normalized");
    expect(hostile.attempts).toBe(ATTEMPTS_RANGE.max);
    expect(hostile.epochsScale).toBe(3);
    expect(hostile.contaminated).toBe(false);
    expect(hostile.items).toBe(definition.initialState.items);
    expect(hydrate('{"attempts": -4}').attempts).toBe(ATTEMPTS_RANGE.min);
    expect(hydrate('{"attempts": 2.6}').attempts).toBe(3);
    for (const value of ["", "not json", "null", "[]", "42"]) expect(hydrate(value)).toEqual(definition.initialState);
  });

  it("bounds a pasted benchmark so every item is scored at every budget in milliseconds", () => {
    const huge = Array.from({ length: 5000 }, (_, index) => `context ${index} | ${"x".repeat(500)} | y`).join("\n");
    const state = hydrate(JSON.stringify({ items: huge }));
    const lines = (state.items as string).split("\n");
    expect(lines).toHaveLength(MAX_ITEMS);
    for (const line of lines) expect(line.length).toBeLessThanOrEqual(MAX_LINE_LENGTH);
    const start = performance.now();
    const bounded = parseItems(state.items as string);
    trainCheckpoints(TRAINING_TEXT, bounded, 3);
    expect(performance.now() - start).toBeLessThan(2500);
  });

  it("renders every rule at every step without a broken value, even from a hostile state", () => {
    for (const rule of RULES) {
      const state = hydrate(JSON.stringify({ rule, attempts: 5, contaminated: true, items: "" }));
      for (const full of [definition.initialState, state, hydrate(JSON.stringify({ rule, attempts: 7 }))]) {
        for (let currentStep = 0; currentStep < definition.steps.length; currentStep += 1) {
          const html = renderToStaticMarkup(
            createElement(definition.Explore, {
              state: full,
              setState: () => undefined,
              currentStep,
              mode: "standard" as const,
              narrate: () => undefined,
            }),
          );
          expect(html).not.toMatch(/\bNaN\b|\bundefined\b|\bInfinity\b/);
          expect(html).toContain("Scoring rule");
        }
      }
    }
  });
});
