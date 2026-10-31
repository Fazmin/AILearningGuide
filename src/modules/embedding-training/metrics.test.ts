import { describe, expect, it } from "vitest";
import { teachingRetrieval } from "@app/module-sdk";
import { DUPLICATE_PAIRS, HELD_OUT_QUERIES, PASSAGES, passageIndex, relevantFor } from "./data";
import { dot, gaussian, initialWeights } from "./encoder";
import {
  alignment,
  BM25_RANKINGS,
  buildSpace,
  duplicateCosine,
  firstRelevantRank,
  keywordTerms,
  meanPairCosine,
  principalAxes,
  rankOf,
  rankScores,
  scoreRankings,
  trainCosines,
  uniformity,
} from "./metrics";
import { HELD_OUT_INFO, passesPerQuery, RERANK_DEPTH, rerankRanking, slotMatch, trainReranker, coverage, trigramJaccard } from "./rerank";

const unit = (values: number[]) => {
  const length = Math.hypot(...values);
  return Float64Array.from(values.map((value) => value / length));
};

describe("recall and MRR on hand-built rankings", () => {
  it("counts a hit when any relevant passage is in the top k, and averages 1 / first relevant rank", () => {
    const rankings = [
      [3, 1, 0, 2], // relevant 1 at rank 2
      [0, 2, 1, 3], // a twin pair (2, 3): 2 at rank 2
      [2, 0, 1, 3], // relevant 2 at rank 1
      [], // BM25 with no shared word ranks nothing
    ];
    const relevant = [[1], [2, 3], [2], [0]];
    const score = scoreRankings(rankings, relevant);
    expect(score.recall1).toBe(0.25);
    expect(score.recall5).toBe(0.75);
    expect(score.mrr).toBeCloseTo((0.5 + 0.5 + 1 + 0) / 4, 12);
    expect(firstRelevantRank(rankings[1], relevant[1])).toBe(2);
    expect(firstRelevantRank(rankings[3], relevant[3])).toBeNull();
  });

  it("scores a twin at rank 1 as a full hit, which the fraction-style recall would halve", () => {
    const ranked = [5, 2, 9, 4];
    const twins = [4, 5];
    expect(scoreRankings([ranked], [twins])).toEqual({ recall1: 1, recall5: 1, mrr: 1 });
    expect(teachingRetrieval.recallAtK(ranked, twins, 1)).toBe(0.5);
  });

  it("drops passages that share no keyword from BM25 rankings", () => {
    expect(rankScores([0, 2, 0, 1], true)).toEqual([1, 3]);
    expect(rankScores([0, 2, 0, 1])).toEqual([1, 3, 0, 2]);
    expect(keywordTerms("The docks closes on Sundays")).toEqual(["dock", "close", "sunday"]);
    const unseen = HELD_OUT_QUERIES.findIndex((entry) => entry.id === "h27");
    expect(BM25_RANKINGS[unseen]).toEqual([]);
  });

  it("makes both members of each duplicate pair relevant", () => {
    for (const [a, b] of DUPLICATE_PAIRS) {
      expect(relevantFor({ id: "x", text: "", answer: a }).sort()).toEqual([passageIndex(a), passageIndex(b)].sort());
    }
    expect(relevantFor({ id: "x", text: "", answer: "north-hours" })).toEqual([passageIndex("north-hours")]);
  });
});

describe("geometry readouts", () => {
  it("finds orthonormal principal axes, with the first along a shared lean", () => {
    for (const dim of [8, 16]) {
      const draw = gaussian(dim);
      const lean = unit(Array.from({ length: dim }, (_, k) => (k === 2 ? 1 : 0)));
      const vectors = Array.from({ length: 41 }, () => unit(Array.from({ length: dim }, (_, k) => 0.3 * draw() + 2 * lean[k])));
      const { axes, captured } = principalAxes(vectors, dim);
      expect(dot(axes[0], axes[0])).toBeCloseTo(1, 9);
      expect(dot(axes[1], axes[1])).toBeCloseTo(1, 9);
      expect(dot(axes[0], axes[1])).toBeCloseTo(0, 9);
      expect(Math.abs(dot(axes[0], lean))).toBeGreaterThan(0.95);
      expect(captured[0]).toBeGreaterThan(captured[1]);
      expect(captured[0] + captured[1]).toBeLessThanOrEqual(1 + 1e-9);
    }
  });

  it("computes alignment, uniformity and mean cosine from their definitions", () => {
    const same = [unit([1, 0]), unit([1, 0]), unit([1, 0])];
    expect(uniformity(same)).toBeCloseTo(0, 12);
    expect(meanPairCosine(same)).toBeCloseTo(1, 12);
    const opposite = [unit([1, 0]), unit([-1, 0])];
    // Squared distance 4, so log exp(−2 × 4) = −8.
    expect(uniformity(opposite)).toBeCloseTo(-8, 12);
    expect(meanPairCosine(opposite)).toBeCloseTo(-1, 12);
    const space = buildSpace(initialWeights(1, 16), 16);
    expect(alignment(space)).toBeGreaterThanOrEqual(0);
    expect(alignment(space)).toBeLessThanOrEqual(4);
    expect(duplicateCosine(space.passages)).toBeLessThanOrEqual(1);
  });

  it("ranks every passage for every held-out query", () => {
    const space = buildSpace(initialWeights(2, 8), 8);
    expect(space.rankings).toHaveLength(HELD_OUT_QUERIES.length);
    for (const ranking of space.rankings) expect([...ranking].sort((a, b) => a - b)).toEqual(PASSAGES.map((_, index) => index));
    expect(rankOf(space.rankings[0], space.rankings[0][0])).toBe(1);
  });
});

describe("the reranker stand-in", () => {
  const held = (id: string) => HELD_OUT_INFO[HELD_OUT_QUERIES.findIndex((entry) => entry.id === id)];

  it("scores slot words, coverage and trigram overlap from both texts", () => {
    const east = held("h02");
    expect(slotMatch(east, passageIndex("east-hours"))).toBe(1);
    // South fills the same slot with another word; Crab pots names no dock at all.
    expect(slotMatch(east, passageIndex("south-hours"))).toBe(-1);
    expect(slotMatch(east, passageIndex("crab-pots"))).toBe(0);
    const sunday = held("h01");
    // North and Sunday both repeated; the weekday notice contradicts the day slot.
    expect(slotMatch(sunday, passageIndex("sunday-hours"))).toBe(2);
    expect(slotMatch(sunday, passageIndex("north-hours"))).toBe(0);
    expect(slotMatch(sunday, passageIndex("south-hours"))).toBe(-1);
    // how, late, use, east, berth: only east is in the notice.
    expect(coverage(east, passageIndex("east-hours"))).toBeCloseTo(1 / 5, 12);
    expect(trigramJaccard(east, passageIndex("east-hours"))).toBeGreaterThan(trigramJaccard(east, passageIndex("crab-pots")));
  });

  it("reorders only the top 5 and costs one pass per reranked passage", () => {
    const weights = initialWeights(1, 16);
    const space = buildSpace(weights, 16);
    const model = trainReranker(trainCosines(weights, space));
    space.rankings.forEach((ranking, index) => {
      const reranked = rerankRanking(model, HELD_OUT_INFO[index], ranking, space.cosine[index]);
      expect(new Set(reranked.slice(0, RERANK_DEPTH))).toEqual(new Set(ranking.slice(0, RERANK_DEPTH)));
      expect(reranked.slice(RERANK_DEPTH)).toEqual(ranking.slice(RERANK_DEPTH));
    });
    expect(passesPerQuery(false)).toBe(1);
    expect(passesPerQuery(true)).toBe(6);
  });
});
