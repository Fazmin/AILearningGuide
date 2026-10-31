import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { teachingRetrieval } from "@app/module-sdk";
import { DUPLICATE_OF, DUPLICATE_PAIRS, HELD_OUT_QUERIES, PASSAGES, passageIndex, TOPICS, TRAIN_QUERIES, relevantFor } from "./data";
import { BUCKETS, encodeAll, bucketsOf, dot } from "./encoder";
import {
  alignment,
  BM25_RANKINGS,
  BM25_RESULT,
  buildSpace,
  duplicateCosine,
  firstRelevantRank,
  HELD_OUT_RELEVANT,
  meanPairCosine,
  rankOf,
  scoreRankings,
  trainCosines,
  truncatedRecallAt1,
  uniformity,
} from "./metrics";
import definition from "./module";
import { CROSS_ENCODER_ONLY_PASSES, passesPerQuery, rerankHeldOut, trainReranker } from "./rerank";
import { bm25Negatives, LEARNING_RATE, REFRESH_EVERY, STEPS, train, TRAIN_ANSWERS, type TrainConfig, type TrainRun } from "./train";

/**
 * Pins every measured number that content/standard.mdx, plain.mdx, simple.mdx and card-info.ts quote.
 * `measure` mirrors Explore.tsx: train with the controls, read the snapshot at the step, build the space,
 * score the held-out rankings, and rerank with a model trained at that step. Defaults: batch 8, τ 0.10,
 * In-batch only, Skip known duplicates off, width 16, seed 1, step 300.
 */
const DEFAULT: TrainConfig = { batch: 8, temperature: 0.1, negatives: "in-batch", skipDuplicates: false, dim: 16, seed: 1 };
const runs = new Map<string, TrainRun>();
const runOf = (patch: Partial<TrainConfig> = {}) => {
  const config = { ...DEFAULT, ...patch };
  const key = JSON.stringify(config);
  if (!runs.has(key)) runs.set(key, train(config));
  return runs.get(key)!;
};
function measure(patch: Partial<TrainConfig> = {}, step = STEPS) {
  const run = runOf(patch);
  const snapshot = run.snapshots[step / 10];
  const space = buildSpace(snapshot.weights, run.config.dim);
  const score = scoreRankings(space.rankings, HELD_OUT_RELEVANT);
  const reranked = rerankHeldOut(trainReranker(trainCosines(snapshot.weights, space)), space.rankings, space.cosine, HELD_OUT_RELEVANT);
  return { run, snapshot, space, score, reranked };
}
const pct = (value: number) => `${Math.round(value * 100)}%`;
const f2 = (value: number) => value.toFixed(2);
const f3 = (value: number) => value.toFixed(3);
const heldIndex = (id: string) => HELD_OUT_QUERIES.findIndex((query) => query.id === id);
const id = (index: number) => PASSAGES[index].id;
const hitsByKind = (rankings: readonly (readonly number[])[]) =>
  (["paraphrase", "typo", "unseen"] as const).map(
    (kind) => HELD_OUT_QUERIES.filter((query, index) => query.kind === kind && firstRelevantRank(rankings[index], HELD_OUT_RELEVANT[index]) === 1).length,
  );
/** Rows, across the 31 batches the card can show, that face their own answer's twin as a candidate. */
const rowsFacingTwin = (run: TrainRun) =>
  run.snapshots.reduce(
    (total, snapshot) =>
      total +
      snapshot.batch.rows.filter((row) => DUPLICATE_OF[TRAIN_ANSWERS[row]] >= 0 && snapshot.batch.candidates.includes(DUPLICATE_OF[TRAIN_ANSWERS[row]])).length,
    0,
  );

describe("the world the lessons describe", () => {
  it("has 41 passages in ten topics, three duplicate pairs, 90 training and 30 held-out queries", () => {
    expect(PASSAGES).toHaveLength(41);
    expect(TOPICS).toHaveLength(10);
    expect(DUPLICATE_PAIRS).toHaveLength(3);
    expect(TRAIN_QUERIES).toHaveLength(90);
    expect(HELD_OUT_QUERIES).toHaveLength(30);
    expect(TRAIN_QUERIES.length + HELD_OUT_QUERIES.length).toBe(120);
    expect(["paraphrase", "typo", "unseen"].map((kind) => HELD_OUT_QUERIES.filter((query) => query.kind === kind).length)).toEqual([20, 5, 5]);
    // 512 × 16 = 8,192 weights; 300 steps; learning rate 8; Nearest wrong refreshes every 10 steps.
    expect(BUCKETS * 16).toBe(8192);
    expect([STEPS, LEARNING_RATE, REFRESH_EVERY]).toEqual([300, 8, 10]);
    expect([teachingRetrieval.BM25_K1, teachingRetrieval.BM25_B]).toEqual([1.2, 0.75]);
    // Embedding map averages all 820 passage pairs.
    expect((41 * 40) / 2).toBe(820);
    expect(definition.initialState.query).toBe("h01");
    expect(HELD_OUT_QUERIES[heldIndex("h01")].text).toBe("When does the north jetty shut on Sundays?");
  });
});

describe("the headline: 7% to 63% against BM25's 43%", () => {
  it("matches step 0, step 300 and BM25 at the defaults", () => {
    const start = measure({}, 0);
    const end = measure();
    expect([pct(start.score.recall1), pct(start.score.recall5), f2(start.score.mrr)]).toEqual(["7%", "23%", "0.16"]);
    expect([pct(end.score.recall1), pct(end.score.recall5), f2(end.score.mrr)]).toEqual(["63%", "83%", "0.72"]);
    expect([pct(BM25_RESULT.recall1), pct(BM25_RESULT.recall5), f2(BM25_RESULT.mrr)]).toEqual(["43%", "60%", "0.51"]);
    expect([f2(meanPairCosine(start.space.passages)), f2(meanPairCosine(end.space.passages))]).toEqual(["0.94", "-0.01"]);
    // Recall@1 reaches 43% by step 10 and 53% by step 20.
    expect([pct(measure({}, 10).score.recall1), pct(measure({}, 20).score.recall1)]).toEqual(["43%", "53%"]);
    // Plain and Simple: still 37% wrong at Recall@1, and the training loss at step 300 is 0.008.
    expect(pct(1 - end.score.recall1)).toBe("37%");
    expect(f3(end.run.losses[STEPS])).toBe("0.008");
  });

  it("matches the hits by kind, the alignment and uniformity, and the first 4 numbers", () => {
    const start = measure({}, 0);
    const end = measure();
    expect(hitsByKind(BM25_RANKINGS)).toEqual([12, 1, 0]);
    expect(hitsByKind(end.space.rankings)).toEqual([16, 3, 0]);
    expect([f2(uniformity(start.space.passages)), f2(uniformity(end.space.passages))]).toEqual(["-0.24", "-3.59"]);
    expect([f2(alignment(start.space)), f2(alignment(end.space))]).toEqual(["0.16", "0.88"]);
    expect(pct(truncatedRecallAt1(end.space, 4))).toBe("33%");
    // One query is 3.3 points of Recall@1.
    expect((100 / 30).toFixed(1)).toBe("3.3");
  });

  it("memorizes the training queries: every one ranks a relevant passage first, and the shown batch gives answers about 99%", () => {
    const end = measure();
    const trainUnits = encodeAll(end.snapshot.weights, 16, TRAIN_QUERIES.map((query) => bucketsOf(query.text)));
    const rankings = trainUnits.map((query) => teachingRetrieval.rankByScore(end.space.passages.map((passage) => dot(query, passage))));
    expect(scoreRankings(rankings, TRAIN_QUERIES.map(relevantFor)).recall1).toBe(1);
    const { batch, scores } = end.snapshot;
    // About 99% in every row: 98.7% at the least, and 99% or 100% once rounded on screen.
    batch.rows.forEach((_, r) => {
      expect(scores.shares[r][batch.answerColumn[r]]).toBeGreaterThan(0.985);
      expect(["99%", "100%"]).toContain(pct(scores.shares[r][batch.answerColumn[r]]));
    });
  });

  it("spreads seeds 1 to 20 from 63% (seed 1, the lowest) to 87% (seed 12)", () => {
    const results = Array.from({ length: 20 }, (_, index) => measure({ seed: index + 1 }).score.recall1);
    expect(pct(Math.min(...results))).toBe("63%");
    expect(pct(Math.max(...results))).toBe("87%");
    expect(results.indexOf(Math.min(...results))).toBe(0);
    expect(results.filter((value) => value === results[0])).toHaveLength(1);
    expect(pct(results[11])).toBe("87%");
    // The unseen-word queries: none first at the defaults; 4 hits in 100 tries over seeds 1 to 20, near a random pick's 2.9.
    let unseenHits = 0;
    for (let seed = 1; seed <= 20; seed += 1) unseenHits += hitsByKind(measure({ seed }).space.rankings)[2];
    expect(unseenHits).toBe(4);
    const randomPick = HELD_OUT_QUERIES.filter((query) => query.kind === "unseen").reduce((sum, query) => sum + relevantFor(query).length / 41, 0) * 20;
    expect(randomPick.toFixed(1)).toBe("2.9");
  });
});

describe("1. Read the untrained space and 2. Train with in-batch negatives", () => {
  it("matches the starting losses and the measured loss at step 0", () => {
    expect([f2(Math.log(2)), f2(Math.log(8)), f2(Math.log(32))]).toEqual(["0.69", "2.08", "3.47"]);
    expect([f2(runOf({ batch: 2 }).losses[0]), f2(runOf().losses[0]), f2(runOf({ batch: 32 }).losses[0])]).toEqual(["0.47", "1.72", "3.10"]);
    // The first batch of 32 holds only 25 distinct answers; the first batch of 8 holds 8.
    expect(runOf({ batch: 32 }).snapshots[0].batch.candidates).toHaveLength(25);
    expect(runOf().snapshots[0].batch.candidates).toHaveLength(8);
    // The untrained encoder already gives every answer in the first batch of 8 more than an even share.
    const first = runOf().snapshots[0];
    first.batch.rows.forEach((_, r) => expect(first.scores.shares[r][first.batch.answerColumn[r]]).toBeGreaterThan(1 / 8));
    // Plain and Simple: an even share of 8 is 12.5%.
    expect(((1 / 8) * 100).toFixed(1)).toBe("12.5");
  });

  it("matches batch 2 against batch 32 at step 300", () => {
    const small = measure({ batch: 2 });
    const large = measure({ batch: 32 });
    expect([f2(small.score.mrr), f2(large.score.mrr)]).toEqual(["0.69", "0.71"]);
    expect([f2(duplicateCosine(small.space.passages)), f2(duplicateCosine(large.space.passages))]).toEqual(["0.73", "0.27"]);
    expect([rowsFacingTwin(small.run), rowsFacingTwin(large.run)]).toEqual([0, 89]);
    expect(small.run.snapshots).toHaveLength(31);
  });
});

describe("3. Turn the temperature", () => {
  it("matches τ = 0.03 and τ = 1.00 at step 300", () => {
    const cold = measure({ temperature: 0.03 });
    const hot = measure({ temperature: 1 });
    expect(f2(meanPairCosine(cold.space.passages))).toBe("0.52");
    expect(pct(cold.score.recall1)).toBe("70%");
    expect(f3(cold.run.losses[STEPS])).toBe("0.001");
    // A cosine lead of 0.2 at τ = 0.03 is a logit lead of almost 7.
    expect((0.2 / 0.03).toFixed(2)).toBe("6.67");
    expect(f2(meanPairCosine(hot.space.passages))).toBe("-0.02");
    expect(f2(hot.run.losses[STEPS])).toBe("1.28");
    expect(pct(hot.run.meanAnswerShare)).toBe("23%");
    // Two queries: 21 of 30 at τ = 0.03 against 19 of 30 at τ = 0.10.
    expect(Math.round(cold.score.recall1 * 30) - Math.round(measure().score.recall1 * 30)).toBe(2);
  });

  it("fits the 90 training pairs within a few dozen steps at τ = 0.03", () => {
    const losses = runOf({ temperature: 0.03 }).losses;
    const mean = (from: number, to: number) => losses.slice(from, to).reduce((sum, value) => sum + value, 0) / (to - from);
    expect(mean(0, 10)).toBeGreaterThan(1);
    expect(mean(40, 60)).toBeLessThan(0.06);
  });
});

describe("4. Mine hard negatives", () => {
  it("matches batch 2: closest wrong 0.39 to 0.68, answer share 89% to 74%, Recall@1 63% to 77%", () => {
    const plain = measure({ batch: 2 });
    const mined = measure({ batch: 2, negatives: "nearest" });
    expect([f2(plain.run.meanClosestWrong), f2(mined.run.meanClosestWrong)]).toEqual(["0.39", "0.68"]);
    expect([pct(plain.run.meanAnswerShare), pct(mined.run.meanAnswerShare)]).toEqual(["89%", "74%"]);
    expect([pct(plain.score.recall1), pct(mined.score.recall1)]).toEqual(["63%", "77%"]);
  });

  it("matches batch 8: Recall@1 63% either way, closest wrong 0.39 to 0.46", () => {
    const plain = measure();
    const mined = measure({ negatives: "nearest" });
    expect([pct(plain.score.recall1), pct(mined.score.recall1)]).toEqual(["63%", "63%"]);
    expect([f2(plain.run.meanClosestWrong), f2(mined.run.meanClosestWrong)]).toEqual(["0.39", "0.46"]);
  });

  it("gives 22 of the 90 training queries no BM25 negative", () => {
    expect(bm25Negatives(false).filter((negative) => negative < 0)).toHaveLength(22);
  });
});

describe("5. Spot a false negative", () => {
  it("matches the Sunday question with Nearest wrong at batch 8, then with Skip known duplicates", () => {
    const query = heldIndex("h01");
    const mined = measure({ negatives: "nearest" });
    const ranks = (space: typeof mined.space) => ["north-hours", "sunday-hours", "sunday-notice"].map((passage) => rankOf(space.rankings[query], passageIndex(passage)));
    expect(ranks(mined.space)).toEqual([1, 2, 3]);
    expect(f2(duplicateCosine(mined.space.passages))).toBe("0.33");
    const skipped = measure({ negatives: "nearest", skipDuplicates: true });
    expect(rankOf(skipped.space.rankings[query], passageIndex("sunday-notice"))).toBe(1);
    expect(f2(duplicateCosine(skipped.space.passages))).toBe("0.59");
    expect([pct(mined.score.recall1), pct(skipped.score.recall1)]).toEqual(["63%", "60%"]);
  });

  it("matches the default inspector: Sunday notice first and North dock hours the closest wrong passage", () => {
    const { space } = measure();
    const query = heldIndex("h01");
    expect(id(space.rankings[query][0])).toBe("sunday-notice");
    const relevant = HELD_OUT_RELEVANT[query];
    const wrong = space.rankings[query].find((passage) => !relevant.includes(passage))!;
    expect(id(wrong)).toBe("north-hours");
  });
});

describe("6. Rerank the shortlist", () => {
  it("matches Recall@1 63% to 67%, MRR 0.72 to 0.75, and 1 to 6 passes against 41", () => {
    const { score, reranked } = measure();
    expect([pct(score.recall1), pct(reranked.score.recall1)]).toEqual(["63%", "67%"]);
    expect([f2(score.mrr), f2(reranked.score.mrr)]).toEqual(["0.72", "0.75"]);
    expect([passesPerQuery(false), passesPerQuery(true), CROSS_ENCODER_ONLY_PASSES]).toEqual([1, 6, 41]);
  });

  it("fixes the east berth question and breaks the cat question", () => {
    const { space, reranked } = measure();
    const east = heldIndex("h02");
    expect(id(space.rankings[east][0])).toBe("crab-pots");
    expect(id(reranked.rankings[east][0])).toBe("east-hours");
    const cat = heldIndex("h12");
    expect(id(space.rankings[cat][0])).toBe("pets-board");
    expect(id(reranked.rankings[cat][0])).toBe("ferry-crossing");
  });
});

describe("the page shows what the lessons quote", () => {
  const page = (state: Record<string, unknown>) =>
    renderToStaticMarkup(
      createElement(definition.Explore, {
        state: definition.hydrateState(JSON.stringify(state)),
        setState: () => undefined,
        currentStep: 0,
        mode: "standard" as const,
        narrate: () => undefined,
      }),
    )
      .replace(/<[^>]*>/g, " ")
      .replace(/&#x27;/g, "'")
      .replace(/\s+/g, " ");

  it("prints the default readouts with the labels the lessons name", () => {
    const text = page({});
    for (const quoted of [
      "Recall@1 63%",
      "Recall@5 83%",
      "MRR 0.72",
      "Starting loss (ln B) 2.08",
      "Loss at step 0 1.72",
      "Mean cosine between passages -0.01",
      "Recall@1, first 4 of 16 numbers 33%",
      "Closest wrong passage North dock hours",
      "Scoring passes per query 1",
      "BM25 (keywords) 43% 60% 0.51 12 1 0",
    ]) {
      expect(text).toContain(quoted);
    }
    expect(page({ step: 0 })).toContain("Mean cosine between passages 0.94");
    expect(page({ rerank: true })).toContain("Scoring passes per query 6");
    expect(page({ batch: 2, negatives: "nearest" })).toContain("Closest wrong cosine, run average 0.68");
    expect(page({ temperature: 1 })).toContain("Loss at this step 1.28");
    expect(page({ temperature: 0.03 })).toContain("Loss at this step 0.001");
    expect(text).toContain("Loss at this step 0.008");
  });
});
