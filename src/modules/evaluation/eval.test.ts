import { describe, expect, it } from "vitest";
import { tinyNextDistribution, tinySequenceLogProb, trainTinyModel, TINY_CORPORA } from "@app/module-sdk";
import {
  itemsForHalfWidth,
  meanInterval,
  pairedDifference,
  parseItems,
  passAtK,
  proportionStandardError,
  scoreContinuation,
  wilsonInterval,
} from "./eval";

describe("scoring", () => {
  it("parses context | correct | distractor lines and drops malformed ones", () => {
    expect(parseItems("A | b | c\nbroken line\n | x | y\nD|E|F|G")).toEqual([
      { context: "a", correct: "b", distractor: "c" },
      { context: "d", correct: "e", distractor: "f" },
    ]);
  });

  it("scores the space and every candidate character, averaged per transition", () => {
    const weights = trainTinyModel({ text: TINY_CORPORA.harbor.text, epochs: 5, seed: 1 }).weights;
    const score = scoreContinuation(weights, "the boats stay in", "port");
    const space = Math.log(tinyNextDistribution(weights, "the boats stay in")[0]);
    const body = tinySequenceLogProb(weights, "the boats stay in ", "port");
    expect(score.transitions).toBe(5);
    expect(score.total).toBeCloseTo(space + body, 10);
    expect(score.mean).toBeCloseTo((space + body) / 5, 10);
  });
});

describe("uncertainty", () => {
  it("gives the textbook standard error", () => {
    expect(proportionStandardError(0.5, 6)).toBeCloseTo(Math.sqrt(0.25 / 6), 12);
    expect(proportionStandardError(1, 6)).toBe(0);
  });

  it("matches known Wilson intervals and stays wide at 6 of 6", () => {
    const [low, high] = wilsonInterval(3, 6);
    expect(low).toBeCloseTo(0.1876, 3);
    expect(high).toBeCloseTo(0.8124, 3);
    const [allLow, allHigh] = wilsonInterval(6, 6);
    expect(allLow).toBeCloseTo(0.6097, 3);
    expect(allHigh).toBe(1);
  });

  it("uses Student's t for small-sample mean intervals", () => {
    const interval = meanInterval([1, 2, 3, 4, 5, 6]);
    const se = Math.sqrt(3.5 / 6);
    expect(interval.mean).toBe(3.5);
    expect(interval.high - interval.mean).toBeCloseTo(2.571 * se, 6);
    expect(pairedDifference([2, 3, 4], [1, 1, 1]).mean).toBeCloseTo(2, 12);
  });

  it("needs 385 items for a ±5 point interval at 50%", () => {
    expect(itemsForHalfWidth(0.5, 0.05)).toBe(385);
  });
});

describe("pass@k", () => {
  it("equals 1 − C(n−c,k)/C(n,k)", () => {
    // C(7,5)/C(10,5) = 21/252
    expect(passAtK(10, 3, 5)).toBeCloseTo(1 - 21 / 252, 12);
    expect(passAtK(10, 3, 1)).toBeCloseTo(0.3, 12);
    expect(passAtK(10, 8, 3)).toBe(1);
    expect(passAtK(10, 0, 3)).toBe(0);
  });
});
