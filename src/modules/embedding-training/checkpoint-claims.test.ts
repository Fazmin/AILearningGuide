import { describe, expect, it } from "vitest";
import { checkpointQuestions } from "@app/module-sdk";
import { DUPLICATE_OF, HELD_OUT_QUERIES, PASSAGES, passageIndex, TRAIN_QUERIES } from "./data";
import {
  BM25_RANKINGS,
  buildSpace,
  duplicateCosine,
  firstRelevantRank,
  HELD_OUT_RELEVANT,
  meanPairCosine,
  rankOf,
  scoreRankings,
  trainCosines,
} from "./metrics";
import definition from "./module";
import { CROSS_ENCODER_ONLY_PASSES, passesPerQuery, rerankHeldOut, trainReranker } from "./rerank";
import { nearestNegatives, STEPS, train, TRAIN_ANSWERS, type TrainConfig } from "./train";

/**
 * Pins the factual claims in the checkpoint explanations (module.ts), one block per question, at the
 * lab's defaults unless the question changes a control: batch 8, τ 0.10, In-batch only, seed 1, step 300.
 */
const DEFAULT: TrainConfig = { batch: 8, temperature: 0.1, negatives: "in-batch", skipDuplicates: false, dim: 16, seed: 1 };
const end = (patch: Partial<TrainConfig> = {}) => {
  const run = train({ ...DEFAULT, ...patch });
  const snapshot = run.snapshots.at(-1)!;
  const space = buildSpace(snapshot.weights, run.config.dim);
  return { run, snapshot, space, score: scoreRankings(space.rankings, HELD_OUT_RELEVANT) };
};
const f2 = (value: number) => value.toFixed(2);
const pct = (value: number) => `${Math.round(value * 100)}%`;
const held = (id: string) => HELD_OUT_QUERIES.findIndex((query) => query.id === id);

describe("embedding-training checkpoint claims", () => {
  it("has eight questions covering all five objectives", () => {
    const questions = checkpointQuestions(definition);
    expect(questions).toHaveLength(8);
    expect(new Set(questions.map((question) => question.objective))).toEqual(new Set([0, 1, 2, 3, 4]));
  });

  it("question 1: ln 8 = 2.08, the first batch measures 1.72, every answer starts above an even share", () => {
    const run = train(DEFAULT, { steps: 0 });
    expect(f2(Math.log(8))).toBe("2.08");
    expect(f2(run.losses[0])).toBe("1.72");
    const { batch, scores } = run.snapshots[0];
    batch.rows.forEach((_, r) => expect(scores.shares[r][batch.answerColumn[r]]).toBeGreaterThan(1 / 8));
  });

  it("question 2: batch 2 to 32 moves ln B from 0.69 to 3.47, MRR 0.69 to 0.71, twins 0 to 89 rows, duplicate cosine 0.73 to 0.27", () => {
    const small = end({ batch: 2 });
    const large = end({ batch: 32 });
    expect([f2(Math.log(2)), f2(Math.log(32))]).toEqual(["0.69", "3.47"]);
    expect([f2(small.score.mrr), f2(large.score.mrr)]).toEqual(["0.69", "0.71"]);
    const facing = (run: typeof small.run) =>
      run.snapshots.reduce(
        (total, snapshot) =>
          total + snapshot.batch.rows.filter((row) => snapshot.batch.candidates.includes(DUPLICATE_OF[TRAIN_ANSWERS[row]])).length,
        0,
      );
    expect([facing(small.run), facing(large.run)]).toEqual([0, 89]);
    expect([f2(duplicateCosine(small.space.passages)), f2(duplicateCosine(large.space.passages))]).toEqual(["0.73", "0.27"]);
  });

  it("question 3: τ 0.03 lifts mean cosine from −0.01 to 0.52, the loss falls to 0.001, and τ scales the gradient", () => {
    const warm = end();
    const cold = end({ temperature: 0.03 });
    expect([f2(meanPairCosine(warm.space.passages)), f2(meanPairCosine(cold.space.passages))]).toEqual(["-0.01", "0.52"]);
    expect((0.2 / 0.03).toFixed(1)).toBe("6.7");
    expect(cold.run.losses[STEPS].toFixed(3)).toBe("0.001");
    const early = cold.run.losses.slice(40, 60).reduce((sum, value) => sum + value, 0) / 20;
    expect(early).toBeLessThan(0.06);
  });

  it("question 4: at batch 2 Nearest wrong lifts closest wrong 0.39 to 0.68, cuts the answer's share 89% to 74%, lifts Recall@1 63% to 77%; batch 8 stays at 63%", () => {
    const plain = end({ batch: 2 });
    const mined = end({ batch: 2, negatives: "nearest" });
    expect([f2(plain.run.meanClosestWrong), f2(mined.run.meanClosestWrong)]).toEqual(["0.39", "0.68"]);
    expect([pct(plain.run.meanAnswerShare), pct(mined.run.meanAnswerShare)]).toEqual(["89%", "74%"]);
    expect([pct(plain.score.recall1), pct(mined.score.recall1)]).toEqual(["63%", "77%"]);
    expect([pct(end().score.recall1), pct(end({ negatives: "nearest" }).score.recall1)]).toEqual(["63%", "63%"]);
    // Mined negatives are never the labelled answer, and held-out queries are never batch rows.
    for (const snapshot of mined.run.snapshots) {
      snapshot.batch.rows.forEach((row, r) => {
        const column = snapshot.batch.minedColumn[r];
        if (column >= 0) expect(snapshot.batch.candidates[column]).not.toBe(TRAIN_ANSWERS[row]);
        expect(row).toBeLessThan(TRAIN_QUERIES.length);
      });
    }
  });

  it("question 5: the Sunday twins were mined against each other 81 times in 124, duplicate cosine 0.33, and back to 0.59 when skipped", () => {
    const mined = end({ negatives: "nearest" });
    const sunday = TRAIN_QUERIES.map((query, index) => ({ query, index })).filter(({ query }) => query.answer.startsWith("sunday"));
    expect(sunday).toHaveLength(4);
    let twins = 0;
    for (const snapshot of mined.run.snapshots) {
      const negatives = nearestNegatives(snapshot.weights, 16, false);
      for (const { index } of sunday) if (negatives[index] === DUPLICATE_OF[TRAIN_ANSWERS[index]]) twins += 1;
    }
    expect([twins, sunday.length * mined.run.snapshots.length]).toEqual([81, 124]);
    const query = held("h01");
    expect(PASSAGES[mined.space.rankings[query][0]].id).toBe("north-hours");
    expect(rankOf(mined.space.rankings[query], passageIndex("sunday-notice"))).toBe(3);
    expect(f2(duplicateCosine(mined.space.passages))).toBe("0.33");
    const skipped = end({ negatives: "nearest", skipDuplicates: true });
    expect(rankOf(skipped.space.rankings[query], passageIndex("sunday-notice"))).toBe(1);
    expect(f2(duplicateCosine(skipped.space.passages))).toBe("0.59");
    expect(definition.initialState.rerank).toBe(false);
  });

  it("question 6: 120 queries, BM25 finds 12 of 20 paraphrases first, and at the defaults no ranker finds an unseen-word query first", () => {
    expect(TRAIN_QUERIES.length + HELD_OUT_QUERIES.length).toBe(120);
    const firstHits = (rankings: readonly (readonly number[])[], kind: string) =>
      HELD_OUT_QUERIES.filter((query, index) => query.kind === kind && firstRelevantRank(rankings[index], HELD_OUT_RELEVANT[index]) === 1).length;
    expect(firstHits(BM25_RANKINGS, "paraphrase")).toBe(12);
    const start = buildSpace(train(DEFAULT, { steps: 0 }).snapshots[0].weights, 16);
    const trained = end();
    const reranked = rerankHeldOut(
      trainReranker(trainCosines(trained.snapshot.weights, trained.space)),
      trained.space.rankings,
      trained.space.cosine,
      HELD_OUT_RELEVANT,
    );
    for (const rankings of [BM25_RANKINGS, start.rankings, trained.space.rankings, reranked.rankings]) {
      expect(firstHits(rankings, "unseen")).toBe(0);
    }
    expect([pct(trained.score.recall1)]).toEqual(["63%"]);
  });

  it("question 7: a twin at rank 1 is a full hit and a reciprocal rank of 1; the fraction-style recall would give 0.5", () => {
    const twin = passageIndex("sunday-notice");
    const labelled = passageIndex("sunday-hours");
    const ranking = [twin, 4, 7, labelled, 9];
    const relevant = HELD_OUT_RELEVANT[held("h01")];
    expect(relevant.sort()).toEqual([labelled, twin].sort());
    expect(scoreRankings([ranking], [relevant])).toEqual({ recall1: 1, recall5: 1, mrr: 1 });
    expect(relevant.filter((passage) => ranking.slice(0, 1).includes(passage)).length / relevant.length).toBe(0.5);
  });

  it("question 8: 1 pass against 41, Recall@5 83%, and reranking moves three queries to first and knocks two off", () => {
    expect([passesPerQuery(false), passesPerQuery(true), CROSS_ENCODER_ONLY_PASSES]).toEqual([1, 6, 41]);
    const trained = end();
    expect(pct(trained.score.recall5)).toBe("83%");
    const reranked = rerankHeldOut(
      trainReranker(trainCosines(trained.snapshot.weights, trained.space)),
      trained.space.rankings,
      trained.space.cosine,
      HELD_OUT_RELEVANT,
    );
    let gained = 0;
    let lost = 0;
    HELD_OUT_QUERIES.forEach((_, index) => {
      const before = firstRelevantRank(trained.space.rankings[index], HELD_OUT_RELEVANT[index]) === 1;
      const after = firstRelevantRank(reranked.rankings[index], HELD_OUT_RELEVANT[index]) === 1;
      if (!before && after) gained += 1;
      if (before && !after) lost += 1;
    });
    expect([gained, lost]).toEqual([3, 2]);
    expect([pct(trained.score.recall1), pct(reranked.score.recall1)]).toEqual(["63%", "67%"]);
  });
});
