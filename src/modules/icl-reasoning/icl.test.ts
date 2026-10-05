import { describe, expect, it } from "vitest";
import definition, { migrateIclState } from "./module";
import { EXAMPLES, RULES, answerDistribution, firstMiss, inductionHead, promptTokens } from "./icl";

const probabilityOf = (shown: number, instruction: boolean, query: string, answer: number) =>
  answerDistribution(shown, instruction, query).find((entry) => entry.answer === answer)?.probability ?? 0;

describe("example pairs", () => {
  it("are true letter counts", () => {
    for (const example of EXAMPLES) expect(example.word.length).toBe(example.answer);
  });
});

describe("rule elimination", () => {
  it("narrows florin to 6 as examples rule out alternatives", () => {
    expect(probabilityOf(0, false, "florin", 6)).toBeCloseTo(1 / 3, 12);
    expect(probabilityOf(1, false, "florin", 6)).toBeCloseTo(1 / 2, 12);
    expect(probabilityOf(2, false, "florin", 6)).toBeCloseTo(2 / 3, 12);
    expect(probabilityOf(3, false, "florin", 6)).toBe(1);
    expect(probabilityOf(0, true, "florin", 6)).toBe(1);
  });

  it("keeps letter ambiguous until an example repeats a letter", () => {
    expect(probabilityOf(4, false, "letter", 6)).toBeCloseTo(0.5, 12);
    expect(probabilityOf(4, false, "letter", 4)).toBeCloseTo(0.5, 12);
    expect(probabilityOf(5, false, "letter", 6)).toBe(1);
    const distinct = RULES.find((rule) => rule.id === "distinct");
    expect(distinct && firstMiss(distinct, 4)).toBe(-1);
    expect(distinct && firstMiss(distinct, 5)).toBe(4);
  });
});

describe("induction head", () => {
  it("retrieves a seen word's answer", () => {
    const head = inductionHead(promptTokens(4, false, "soravel"));
    expect(head.output[0].token).toBe("7");
    expect(head.output[0].probability).toBeGreaterThan(0.9);
  });

  it("spreads over earlier answers for a new word", () => {
    const head = inductionHead(promptTokens(4, false, "florin"));
    const numbers = head.output.filter((entry) => /^\d$/.test(entry.token));
    expect(numbers.map((entry) => entry.token).sort()).toEqual(["4", "5", "6", "7"]);
    for (const entry of numbers) expect(entry.probability).toBeCloseTo(0.236, 3);
    const five = inductionHead(promptTokens(5, false, "florin"));
    expect(five.output[0]).toMatchObject({ token: "6" });
    expect(five.output[0].probability).toBeCloseTo(0.378, 3);
  });
});

describe("icl-reasoning state", () => {
  it("migrates a version 1 payload", () => {
    const migrated = migrateIclState({ examples: 4, budget: 80, strategy: "rule" });
    expect(migrated).toMatchObject({ examples: 4, strategy: "rule", query: "florin", budget: 3 });
    expect(definition.hydrateState(JSON.stringify({ examples: 99 }))).toMatchObject({ examples: 5 });
  });

  it("drops the voting card's keys from a version 2 payload and keeps the shared ones", () => {
    const v2 = { examples: 4, strategy: "rule", query: "letter", accuracy: 0.4, samples: 9, spread: "same", drawSeed: 5 };
    const migrated = migrateIclState(v2);
    expect(migrated).toEqual({
      examples: 4,
      strategy: "rule",
      query: "letter",
      hops: 6,
      budget: 3,
      scratchpad: "off",
    });
    for (const key of ["accuracy", "samples", "spread", "drawSeed"]) expect(migrated).not.toHaveProperty(key);
  });

  it("validates and clamps every scratchpad field", () => {
    expect(
      definition.hydrateState(JSON.stringify({ hops: 99, budget: -4, scratchpad: "yes", query: "nope", strategy: 7 })),
    ).toMatchObject({ hops: 8, budget: 1, scratchpad: "off", query: "florin", strategy: "analogy" });
    expect(definition.hydrateState(JSON.stringify({ hops: 0, budget: 99, scratchpad: "on" }))).toMatchObject({
      hops: 1,
      budget: 8,
      scratchpad: "on",
    });
    expect(definition.hydrateState(JSON.stringify({ hops: "six", budget: null }))).toMatchObject({ hops: 6, budget: 3 });
  });
});
