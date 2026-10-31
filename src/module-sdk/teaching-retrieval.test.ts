import { describe, expect, it } from "vitest";
import { analyze, bm25, buildChunks } from "../modules/retrieval-augmented-generation/rag";
import { bm25Scores, hitAtK, rankByScore, recallAtK, reciprocalRank } from "./teaching-retrieval";

describe("BM25 over pre-split terms", () => {
  it("scores the RAG lab's chunks exactly as that lab does", () => {
    const chunks = buildChunks(40, 10, true);
    for (const query of ["when does the ferry leave on sunday", "mooring fee for a small boat", "zzz"]) {
      const expected = new Map(bm25(chunks, query).map((chunk) => [chunk.id, chunk.score]));
      // Query analysis belongs to the caller, so reuse that lab's analyzer.
      const scores = bm25Scores(chunks.map((chunk) => chunk.terms), analyze(query));
      chunks.forEach((chunk, index) => expect(scores[index]).toBeCloseTo(expected.get(chunk.id) ?? 0, 12));
    }
  });

  it("gives a document with no query term a score of zero", () => {
    expect(bm25Scores([["dock"], ["fee"]], ["ferry"])).toEqual([0, 0]);
  });
});

describe("relevance metrics", () => {
  const ranked = [3, 0, 2, 1];
  it("reads hits, recall and reciprocal rank from one ranking", () => {
    expect(hitAtK(ranked, [2], 1)).toBe(0);
    expect(hitAtK(ranked, [2], 3)).toBe(1);
    expect(recallAtK(ranked, [0, 1], 2)).toBe(0.5);
    expect(reciprocalRank(ranked, [2])).toBeCloseTo(1 / 3, 12);
    expect(reciprocalRank(ranked, [9])).toBe(0);
  });

  it("ranks by score with ties kept in document order", () => {
    expect(rankByScore([0.5, 2, 0.5, 1])).toEqual([1, 3, 0, 2]);
  });
});
