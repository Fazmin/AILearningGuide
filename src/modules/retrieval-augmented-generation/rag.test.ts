import { describe, expect, it } from "vitest";
import { teachingEmbeddings } from "@app/module-sdk";
import definition, { hydrateRagState } from "./module";
import {
  ABSTAIN_KEY,
  ABSTAIN_SCALE,
  analyze,
  bm25,
  buildChunks,
  chunkDoc,
  clampThreshold,
  DOCS,
  encoderView,
  ground,
  groundingSweep,
  groundingTotals,
  pieces,
  OUTCOME_LABEL,
  QUESTIONS,
  rankChunks,
  read,
  RETRIEVERS,
  RRF_K,
  runPipeline,
  scorePipeline,
  scoreTable,
  scoreTotals,
  splitWords,
  type QuestionId,
} from "./rag";

const handbook = DOCS.find((doc) => doc.id === "handbook")!;
const NO_CHECK = { abstainBm25: 0, abstainDense: 0, abstainHybrid: 0 };

describe("chunking", () => {
  it("starts windows every size − overlap words and stops at the end", () => {
    const chunks = chunkDoc(handbook, 12, 0);
    expect(chunks.map((chunk) => chunk.start)).toEqual([0, 12, 24, 36, 48]);
    expect(chunks.at(-1)!.end).toBe(splitWords(handbook.text).length);
    const overlapped = chunkDoc(handbook, 12, 7);
    expect(overlapped.slice(0, 3).map((chunk) => [chunk.start, chunk.end])).toEqual([
      [0, 12],
      [5, 17],
      [10, 22],
    ]);
  });

  it("authors answer spans that point at the intended words", () => {
    for (const question of QUESTIONS) {
      if (!question.span) continue;
      const words = splitWords(DOCS.find((doc) => doc.id === question.span!.doc)!.text);
      const text = words.slice(question.span.start, question.span.end).join(" ");
      expect(text).toContain(question.correct!);
    }
    const sunday = QUESTIONS[0].span!;
    expect(splitWords(handbook.text).slice(sunday.start, sunday.end).join(" ")).toBe(
      "north dock closes at 18:00 on weekdays and at 16:00 on Sundays.",
    );
  });
});

describe("BM25", () => {
  it("drops stop-words and strips one plural s", () => {
    expect(analyze("When does the north dock close on Sundays?")).toEqual(["north", "dock", "close", "sunday"]);
  });

  it("gives zero to a chunk with no query term and weighs rare terms more", () => {
    const chunks = chunkDoc(handbook, 20, 0);
    const ranked = bm25(chunks, "When does the north dock close on Sunday?");
    const top = ranked[0];
    expect(top.start).toBe(0);
    const north = top.contributions.find((entry) => entry.term === "north")!;
    const dock = top.contributions.find((entry) => entry.term === "dock")!;
    expect(north.idf).toBeGreaterThan(dock.idf);
    expect(ranked.some((chunk) => chunk.score === 0)).toBe(true);
  });

  it("matches a hand computation for one term", () => {
    const chunks = [
      { id: "a", doc: "x", docTitle: "x", ordinal: 0, start: 0, end: 2, text: "ferry ferry", terms: ["ferry", "ferry"] },
      { id: "b", doc: "x", docTitle: "x", ordinal: 1, start: 2, end: 4, text: "dock pier", terms: ["dock", "pier"] },
    ];
    const [first] = bm25(chunks, "ferry");
    const idf = Math.log(1 + (2 - 1 + 0.5) / (1 + 0.5));
    const expected = (idf * 2 * 2.2) / (2 + 1.2 * (1 - 0.75 + 0.75 * (2 / 2)));
    expect(first.score).toBeCloseTo(expected, 12);
  });
});

describe("the reader", () => {
  it("splits sentence pieces and quotes the one with the most query terms", () => {
    expect(pieces("A b. C d? e")).toEqual(["A b.", "C d?", "e"]);
    const answer = read("north dock sunday", chunkDoc(handbook, 30, 0));
    expect(answer.sentence).toBe("The north dock closes at 18:00 on weekdays and at 16:00 on Sundays.");
    expect(answer.citation).toBe(1);
  });
});

describe("the claims the lesson and steps make", () => {
  it("default settings answer the Sunday question correctly", () => {
    const run = runPipeline("hours", 20, 0, false, 2);
    expect(run.diagnosis.verdict).toBe("correct");
    expect(run.answer.sentence).toContain("16:00");
    expect(run.chunks.length).toBe(5);
  });

  it("chunk size 12 splits the answer and the reader quotes 18:00", () => {
    const run = runPipeline("hours", 12, 0, false, 2);
    expect(run.diagnosis.spanWholeInIndex).toBe(false);
    expect(run.answer.sentence).toBe("The north dock closes at 18:00 on weekdays");
    expect(run.diagnosis.verdict).toBe("wrong");
    expect(run.chunks.length).toBe(8);
  });

  it("overlap 7 at size 12 restores the answer with 15 chunks", () => {
    const run = runPipeline("hours", 12, 7, false, 2);
    expect(run.diagnosis.verdict).toBe("correct");
    expect(run.diagnosis.spanWholeRetrieved).toBe(true);
    expect(run.chunks.length).toBe(15);
    expect(runPipeline("hours", 12, 6, false, 2).diagnosis.verdict).toBe("wrong");
  });

  it("the ferry answer is stale before the notice and fresh after, with a conflict at k = 2", () => {
    expect(runPipeline("ferry", 20, 0, false, 2).diagnosis.verdict).toBe("stale");
    const one = runPipeline("ferry", 20, 0, true, 1);
    expect(one.diagnosis.verdict).toBe("correct");
    expect(one.retrieved[0].doc).toBe("notice");
    const two = runPipeline("ferry", 20, 0, true, 2);
    expect(two.diagnosis.staleInContext && two.diagnosis.freshInContext).toBe(true);
    expect(two.diagnosis.verdict).toBe("correct");
  });

  it("the dog question never retrieves the pets sentence", () => {
    for (const size of [6, 12, 20, 30]) {
      const run = runPipeline("dog", size, 0, false, 4);
      expect(run.diagnosis.verdict).toBe("missed");
      expect(run.retrieved.every((chunk) => chunk.doc === "timetable")).toBe(true);
    }
  });

  it("the wind question has no answer and the reader quotes the rule anyway", () => {
    const run = runPipeline("wind", 20, 0, false, 2);
    expect(run.diagnosis.verdict).toBe("unsupported");
    expect(run.answer.sentence).toContain("40 knots");
  });
});

describe("state", () => {
  it("migrates a version 1 payload", () => {
    const state = hydrateRagState(JSON.stringify({ question: "fresh", topK: 3, grounded: "off" }));
    expect(state.question).toBe("wind");
    expect(state.topK).toBe(3);
    expect(state).not.toHaveProperty("grounded");
    expect(state.chunkSize).toBe(definition.initialState.chunkSize);
    expect(hydrateRagState(JSON.stringify({ question: "password", grounded: "on" })).question).toBe("hours");
  });

  it("clamps overlap to three quarters of the chunk size", () => {
    expect(hydrateRagState(JSON.stringify({ chunkSize: 8, overlap: 20 })).overlap).toBe(6);
  });

  it("gives a version 2 payload, which had no retriever, the BM25 retriever", () => {
    const old = { question: "dog", chunkSize: 12, overlap: 3, topK: 3, index: "after" };
    expect(hydrateRagState(JSON.stringify(old))).toEqual({ ...old, retriever: "bm25", ...NO_CHECK });
  });

  it("gives a version 3 payload, which had no grounding check, a threshold of 0 for every retriever", () => {
    const old = { question: "wind", chunkSize: 20, overlap: 0, topK: 3, index: "after", retriever: "dense" };
    expect(hydrateRagState(JSON.stringify(old))).toEqual({ ...old, ...NO_CHECK });
    expect(definition.stateVersion).toBe(4);
  });

  it("validates and clamps every threshold", () => {
    const hostile = hydrateRagState(
      JSON.stringify({ abstainBm25: 1e9, abstainDense: -1e9, abstainHybrid: "2", extra: 1 }),
    );
    expect([hostile.abstainBm25, hostile.abstainDense, hostile.abstainHybrid]).toEqual([8, 0, 0]);
    const infinite = hydrateRagState('{"abstainBm25": 1e999, "abstainDense": 0.456, "abstainHybrid": 1.99}');
    expect([infinite.abstainBm25, infinite.abstainDense, infinite.abstainHybrid]).toEqual([0, 0.46, 2]);
  });

  it("keeps a known retriever and replaces anything else with BM25", () => {
    for (const retriever of RETRIEVERS) {
      expect(hydrateRagState(JSON.stringify({ retriever })).retriever).toBe(retriever);
    }
    for (const bad of ["Dense", "", null, 3, ["dense"], {}]) {
      expect(hydrateRagState(JSON.stringify({ retriever: bad })).retriever).toBe("bm25");
    }
  });

  it("accepts the June question and rejects unknown ones", () => {
    expect(hydrateRagState(JSON.stringify({ question: "june" })).question).toBe("june");
    expect(hydrateRagState(JSON.stringify({ question: "nope" })).question).toBe("hours");
  });
});

const defaults = { size: 20, overlap: 0 };
const ids = (run: ReturnType<typeof runPipeline>) => run.retrieved.map((chunk) => chunk.id);

describe("dense retrieval", () => {
  it("scores each chunk with the embeddings lab's embed and cosine", () => {
    const chunks = buildChunks(20, 0, false);
    const question = QUESTIONS.find((item) => item.id === "dog")!;
    const ranked = rankChunks(chunks, question.text, "dense");
    const query = teachingEmbeddings.embed(question.text);
    for (const row of ranked) {
      expect(row.cosine).toBe(teachingEmbeddings.cosine(query, teachingEmbeddings.embed(row.text)));
      expect(row.score).toBe(row.cosine);
    }
    expect(ranked.slice(0, 2).map((row) => [row.id, Number(row.cosine.toFixed(3))])).toEqual([
      ["timetable-0", 0.541],
      ["handbook-1", 0.498],
    ]);
  });

  it("recovers the pets chunk that BM25 scores 0, once Top-k reaches 2", () => {
    const bm = runPipeline("dog", 20, 0, false, 2, "bm25");
    expect(ids(bm)).toEqual(["timetable-0"]);
    expect(bm.ranked.find((row) => row.id === "handbook-1")!.bm25Score).toBe(0);
    expect(bm.diagnosis.spanCoverage).toBe(0);

    expect(ids(runPipeline("dog", 20, 0, false, 1, "dense"))).toEqual(["timetable-0"]);
    for (const retriever of ["dense", "hybrid"] as const) {
      const run = runPipeline("dog", 20, 0, false, 2, retriever);
      expect(ids(run)).toEqual(["timetable-0", "handbook-1"]);
      expect(run.diagnosis.spanWholeRetrieved).toBe(true);
      expect(run.diagnosis.spanCoverage).toBe(1);
    }
  });

  it("still answers wrong, because the reader matches words and the pets sentence shares none", () => {
    const run = runPipeline("dog", 20, 0, false, 2, "dense");
    expect(run.answer.sentence).toBe("The island ferry leaves the harbor at 07:15 in the morning and at 15:40 in the afternoon.");
    expect(run.diagnosis.verdict).toBe("wrong");
    expect(scorePipeline("dog", 20, 0, false, 2, "dense").site).toBe("reader");
    expect(scorePipeline("dog", 20, 0, false, 2, "bm25").site).toBe("retrieval");
  });

  it("encoder view lists the words the encoder knows and the ones it ignores", () => {
    expect(encoderView("Can I bring my dog on the ferry?")).toEqual({
      known: ["bring", "dog", "ferry"],
      ignored: ["can", "i", "my", "on", "the"],
    });
    expect(encoderView("What changed in June?").known).toEqual([]);
  });
});

describe("where dense retrieval has nothing to match", () => {
  it("gives the June question a zero vector, so every cosine is 0 and nothing is retrieved", () => {
    const chunks = buildChunks(20, 0, true);
    const ranked = rankChunks(chunks, "What changed in June?", "dense");
    expect(ranked.every((row) => row.cosine === 0 && row.denseRank === null)).toBe(true);
    const dense = runPipeline("june", 20, 0, true, 4, "dense");
    expect(dense.retrieved).toEqual([]);
    expect(dense.diagnosis.verdict).toBe("missed");
    expect(dense.answer.sentence).toBe("");
    expect(dense.diagnosis.quoteSupported).toBeNull();
  });

  it("lets BM25 and Hybrid find the notice by its rare word", () => {
    for (const retriever of ["bm25", "hybrid"] as const) {
      const run = runPipeline("june", 20, 0, true, 2, retriever);
      expect(ids(run)).toEqual(["notice-0"]);
      if (retriever === "hybrid") expect(run.ranked[0].rrf).toBeCloseTo(1 / 61, 12);
      expect(run.diagnosis.verdict).toBe("correct");
      expect(run.answer.sentence).toContain("16:10");
    }
  });

  it("finds nothing under any retriever before the notice is indexed", () => {
    for (const retriever of RETRIEVERS) {
      const run = runPipeline("june", 20, 0, false, 4, retriever);
      expect(run.retrieved).toEqual([]);
      expect(run.diagnosis.verdict).toBe("missed");
      expect(run.diagnosis.spanCoverage).toBe(0);
    }
  });
});

describe("hybrid retrieval by reciprocal-rank fusion", () => {
  it("uses the standard constant 60", () => {
    expect(RRF_K).toBe(60);
  });

  it("adds 1/(60 + rank) for every list that holds the chunk", () => {
    const chunks = buildChunks(20, 0, false);
    const question = QUESTIONS.find((item) => item.id === "hours")!;
    const ranked = rankChunks(chunks, question.text, "hybrid");
    const top = ranked[0];
    expect(top.bm25Rank).toBe(1);
    expect(top.denseRank).toBe(1);
    expect(top.rrf).toBeCloseTo(2 / 61, 12);
    expect(Number(top.rrf.toFixed(4))).toBe(0.0328);
    for (const row of ranked) {
      const expected =
        (row.bm25Rank ? 1 / (RRF_K + row.bm25Rank) : 0) + (row.denseRank ? 1 / (RRF_K + row.denseRank) : 0);
      expect(row.rrf).toBeCloseTo(expected, 12);
      expect(row.score).toBe(row.rrf);
    }
  });

  it("ranks a chunk found by both retrievers above one found by a single retriever", () => {
    const chunks = buildChunks(20, 0, false);
    const ranked = rankChunks(chunks, "When does the north dock close on Sunday?", "hybrid");
    const both = ranked.filter((row) => row.bm25Rank && row.denseRank);
    const single = ranked.filter((row) => Boolean(row.bm25Rank) !== Boolean(row.denseRank));
    expect(both.length).toBeGreaterThan(0);
    expect(single.length).toBeGreaterThan(0);
    expect(Math.min(...both.map((row) => row.rrf))).toBeGreaterThan(Math.max(...single.map((row) => row.rrf)));
  });

  it("fuses ranks, so it keeps BM25's hit that dense lacks and dense's hit that BM25 lacks", () => {
    expect(ids(runPipeline("june", 20, 0, true, 1, "hybrid"))).toEqual(["notice-0"]);
    expect(ids(runPipeline("dog", 20, 0, false, 2, "hybrid"))).toContain("handbook-1");
  });

  it("keeps the BM25 default so every earlier claim about the lab still holds", () => {
    const explicit = runPipeline("hours", 12, 7, false, 2, "bm25");
    const implicit = runPipeline("hours", 12, 7, false, 2);
    expect(ids(explicit)).toEqual(ids(implicit));
    expect(explicit.ranked.map((row) => row.score)).toEqual(bm25(explicit.chunks, explicit.question.text).map((row) => row.score));
  });
});

describe("scoring the pipeline", () => {
  const cell = (id: QuestionId, size: number, overlap: number, after: boolean, k: number, retriever: (typeof RETRIEVERS)[number]) =>
    scorePipeline(id, size, overlap, after, k, retriever);

  it("measures context recall as the share of answer-span words inside a retrieved chunk", () => {
    // Chunk size 12, no overlap, Top-k 2: only the first 7 of the 12 span words are retrieved.
    expect(cell("hours", 12, 0, false, 2, "bm25").contextRecall).toBeCloseTo(7 / 12, 12);
    expect(cell("hours", 12, 0, false, 2, "bm25").site).toBe("retrieval");
    // Top-k 3 brings the second half too: full recall, and the reader still cannot join them.
    const three = cell("hours", 12, 0, false, 3, "bm25");
    expect(three.contextRecall).toBe(1);
    expect(three.correct).toBe(false);
    expect(three.site).toBe("reader");
    expect(runPipeline("hours", 12, 0, false, 3).diagnosis.spanWholeRetrieved).toBe(false);
  });

  it("calls a wrong answer a retrieval failure below full recall and a reader failure at full recall", () => {
    for (const retriever of RETRIEVERS) {
      for (const row of scoreTable(defaults.size, defaults.overlap, false, 2)) {
        const score = row.scores[retriever];
        if (score.correct === null) expect(score.site).toBe("no-evidence");
        else if (score.correct) expect(score.site).toBe("none");
        else expect(score.site).toBe((score.contextRecall ?? 0) < 1 ? "retrieval" : "reader");
      }
    }
  });

  it("marks the wind question unanswerable: no recall, no correctness, and the reader quotes a rule", () => {
    const score = cell("wind", 20, 0, false, 2, "bm25");
    expect(score.contextRecall).toBeNull();
    expect(score.correct).toBeNull();
    expect(score.faithful).toBe(true);
    expect(score.site).toBe("no-evidence");
  });

  it("passes the support check whenever the reader quotes, and reports nothing when it quotes nothing", () => {
    for (const size of [6, 8, 12, 16, 20, 30]) {
      for (const after of [false, true]) {
        for (const k of [1, 2, 3, 4]) {
          for (const row of scoreTable(size, 0, after, k)) {
            for (const retriever of RETRIEVERS) {
              const run = runPipeline(row.question.id, size, 0, after, k, retriever);
              expect(row.scores[retriever].faithful).toBe(run.answer.sentence ? true : null);
            }
          }
        }
      }
    }
  });

  it("pins the table at the default controls (size 20, overlap 0, index before, Top-k 2)", () => {
    const rows = scoreTable(20, 0, false, 2);
    const totals = Object.fromEntries(RETRIEVERS.map((retriever) => [retriever, scoreTotals(rows, retriever)]));
    expect(RETRIEVERS.map((retriever) => totals[retriever].correct)).toEqual([1, 1, 1]);
    expect(RETRIEVERS.map((retriever) => totals[retriever].answerable)).toEqual([4, 4, 4]);
    expect(RETRIEVERS.map((retriever) => totals[retriever].meanRecall)).toEqual([0.25, 0.5, 0.5]);
    expect(RETRIEVERS.map((retriever) => totals[retriever].retrievalFailures)).toEqual([3, 2, 2]);
    expect(RETRIEVERS.map((retriever) => totals[retriever].readerFailures)).toEqual([0, 1, 1]);
  });

  it("pins the table the last step reads (size 20, overlap 0, index after, Top-k 3)", () => {
    const rows = scoreTable(20, 0, true, 3);
    const byRetriever = Object.fromEntries(RETRIEVERS.map((retriever) => [retriever, scoreTotals(rows, retriever)]));
    expect(RETRIEVERS.map((retriever) => byRetriever[retriever].meanRecall)).toEqual([0.75, 0.75, 1]);
    expect(RETRIEVERS.map((retriever) => byRetriever[retriever].correct)).toEqual([3, 2, 3]);
    expect(RETRIEVERS.map((retriever) => byRetriever[retriever].retrievalFailures)).toEqual([1, 1, 0]);
    expect(RETRIEVERS.map((retriever) => byRetriever[retriever].readerFailures)).toEqual([0, 1, 1]);
    expect(RETRIEVERS.map((retriever) => `${byRetriever[retriever].faithful}/${byRetriever[retriever].quoted}`)).toEqual([
      "5/5",
      "4/4",
      "5/5",
    ]);
    const dog = rows.find((row) => row.question.id === "dog")!;
    expect(dog.scores.bm25.site).toBe("retrieval");
    expect(dog.scores.dense.site).toBe("reader");
    expect(dog.scores.hybrid.site).toBe("reader");
    const june = rows.find((row) => row.question.id === "june")!;
    expect(june.scores.dense.site).toBe("retrieval");
    expect(june.scores.hybrid.correct).toBe(true);
  });

  it("counts supported quotes that are still wrong", () => {
    const totals = scoreTotals(scoreTable(20, 0, true, 3), "hybrid");
    // Wind (cannot abstain) and Dog (reader failure) are both supported and not correct.
    expect(totals.faithfulButWrong).toBe(2);
  });
});

describe("the grounding check", () => {
  // The Score the pipeline step's settings: chunk size 20, overlap 0, index after the notice, Top-k 3.
  const at = (id: QuestionId, retriever: (typeof RETRIEVERS)[number], threshold: number, after = true, k = 3) =>
    ground(runPipeline(id, 20, 0, after, k, retriever), retriever, threshold);
  const ids = QUESTIONS.map((question) => question.id);
  const outcomes = (retriever: (typeof RETRIEVERS)[number], threshold: number) =>
    Object.fromEntries(ids.map((id) => [id, at(id, retriever, threshold).outcome]));

  it("abstains at threshold 0 only when nothing was retrieved", () => {
    for (const retriever of RETRIEVERS) {
      for (const id of ids) {
        const grounding = at(id, retriever, 0);
        expect(grounding.abstained).toBe(grounding.topScore === null);
        expect(grounding.reason).toBe(grounding.abstained ? "nothing-retrieved" : null);
      }
    }
    expect(at("june", "dense", 0).outcome).toBe("abstained-wrong");
  });

  it("pins each question's best chunk score on the retriever's own scale", () => {
    const top = (retriever: (typeof RETRIEVERS)[number]) => ids.map((id) => at(id, retriever, 0).topScore);
    // hours, ferry, wind, dog, june
    expect(top("bm25")).toEqual([5.24, 3.75, 2.28, 1.25, 1.87]);
    expect(top("dense")).toEqual([0.981, 1, 0.843, 0.542, null]);
    expect(top("hybrid")).toEqual([2, 2, 1.984, 2, 1]);
  });

  it("classifies the four outcomes against what the pipeline should have done", () => {
    expect(outcomes("bm25", 0)).toEqual({
      hours: "answered-right",
      ferry: "answered-right",
      wind: "answered-wrong",
      dog: "answered-wrong",
      june: "answered-right",
    });
    expect(outcomes("bm25", 2.4)).toEqual({
      hours: "answered-right",
      ferry: "answered-right",
      wind: "abstained-right",
      dog: "abstained-wrong",
      june: "abstained-wrong",
    });
    expect(Object.values(OUTCOME_LABEL)).toEqual([
      "answered correctly",
      "answered wrongly with confidence",
      "correctly abstained",
      "wrongly abstained",
    ]);
  });

  it("cannot drop the wind answer under BM25 without also dropping Changed in June, which scores lower", () => {
    const wind = at("wind", "bm25", 0).topScore!;
    const june = at("june", "bm25", 0).topScore!;
    expect(june).toBeLessThan(wind);
    // The threshold that silences wind (anything above 2.28) is above June's 1.87 too.
    expect(at("wind", "bm25", 2.3).abstained).toBe(true);
    expect(at("june", "bm25", 2.3).abstained).toBe(true);
    expect(at("wind", "bm25", 2.2).abstained).toBe(false);
    expect(at("june", "bm25", 1.9).abstained).toBe(true);
    expect(at("june", "bm25", 1.8).abstained).toBe(false);
  });

  it("names the words behind the wind and June scores that the checkpoint quotes", () => {
    const wind = runPipeline("wind", 20, 0, true, 3, "bm25").retrieved[0];
    expect(wind.contributions.map((entry) => entry.term)).toEqual(["harbor", "wind"]);
    const june = runPipeline("june", 20, 0, true, 3, "bm25").retrieved[0];
    expect(june.doc).toBe("notice");
    expect(june.contributions.map((entry) => entry.term)).toEqual(["june"]);
  });

  it("separates wind from the right answers under Dense on these five questions", () => {
    expect(at("wind", "dense", 0.85).abstained).toBe(true);
    expect(at("hours", "dense", 0.85).abstained).toBe(false);
    expect(at("ferry", "dense", 0.85).abstained).toBe(false);
    const totals = (threshold: number) => groundingTotals(ids.map((id) => at(id, "dense", threshold).outcome));
    expect(totals(0).handled).toBe(2);
    expect(totals(0.9).handled).toBe(3);
  });

  it("makes the Hybrid threshold a knife edge, because a fused score reads ranks and not match quality", () => {
    expect(at("wind", "hybrid", 1.98).abstained).toBe(false);
    expect(at("wind", "hybrid", 2).abstained).toBe(true);
    // A first place in both lists is exactly 2 on the ×61 scale, and still passes at 2.
    expect(at("hours", "hybrid", 2).abstained).toBe(false);
  });

  it("reports coverage that only falls, and precision that does not rise steadily (BM25, Top-k 3, index after)", () => {
    const sweep = groundingSweep(20, 0, true, 3, "bm25");
    expect(sweep).toHaveLength(81);
    for (let index = 1; index < sweep.length; index += 1) {
      expect(sweep[index].coverage).toBeLessThanOrEqual(sweep[index - 1].coverage);
    }
    const point = (threshold: number) => sweep.find((item) => item.threshold === threshold)!;
    // 0: all five answered, 3 right. 1.3: dog dropped. 1.9: June dropped. 2.4: wind dropped.
    expect([0, 1.3, 1.9, 2.4].map((threshold) => point(threshold).coverage)).toEqual([1, 0.8, 0.6, 0.4]);
    expect([0, 1.3, 1.9, 2.4].map((threshold) => Number(point(threshold).precision!.toFixed(3)))).toEqual([
      0.6, 0.75, 0.667, 1,
    ]);
    expect([0, 1.3, 1.9, 2.4].map((threshold) => point(threshold).handled)).toEqual([3, 3, 2, 3]);
    expect(point(8).precision).toBeNull();
    expect(point(8).coverage).toBe(0);
  });

  it("cannot catch the stale ferry time: its chunk matches as well as a right answer would", () => {
    const stale = runPipeline("ferry", 20, 0, false, 2, "bm25");
    expect(stale.diagnosis.verdict).toBe("stale");
    const ferry = at("ferry", "bm25", 0, false, 2).topScore!;
    const hours = at("hours", "bm25", 0, false, 2).topScore!;
    const wind = at("wind", "bm25", 0, false, 2).topScore!;
    expect([wind, ferry, hours]).toEqual([2.03, 4.38, 4.76]);
    expect(ferry).toBeGreaterThan(wind);
  });

  it("scores the wind question in Score the pipeline: wrong by default, a correct abstention with the check", () => {
    const off = scoreTable(20, 0, true, 3);
    for (const retriever of RETRIEVERS) {
      const wind = off.find((row) => row.question.id === "wind")!.scores[retriever];
      expect(wind.outcome).toBe("answered-wrong");
      expect(wind.correct).toBeNull();
      expect(wind.site).toBe("no-evidence");
    }
    const on = scoreTable(20, 0, true, 3, { bm25: 2.4, dense: 0.9 });
    expect(on.find((row) => row.question.id === "wind")!.scores.bm25.outcome).toBe("abstained-right");
    expect(on.find((row) => row.question.id === "wind")!.scores.hybrid.outcome).toBe("answered-wrong");
    const totals = Object.fromEntries(RETRIEVERS.map((retriever) => [retriever, scoreTotals(on, retriever)]));
    expect(RETRIEVERS.map((retriever) => totals[retriever].handled)).toEqual([3, 3, 3]);
    expect(RETRIEVERS.map((retriever) => totals[retriever].confidentErrors)).toEqual([0, 0, 2]);
    expect(RETRIEVERS.map((retriever) => totals[retriever].abstainedRight)).toEqual([1, 1, 0]);
    const base = Object.fromEntries(RETRIEVERS.map((retriever) => [retriever, scoreTotals(off, retriever)]));
    expect(RETRIEVERS.map((retriever) => base[retriever].handled)).toEqual([3, 2, 3]);
    expect(RETRIEVERS.map((retriever) => base[retriever].confidentErrors)).toEqual([2, 2, 2]);
    // An abstention is not a quote: it is not counted as supported or as correct.
    expect(on.find((row) => row.question.id === "june")!.scores.bm25.faithful).toBeNull();
    expect(on.find((row) => row.question.id === "june")!.scores.bm25.correct).toBe(false);
  });

  it("keeps every earlier Score the pipeline figure when the check is off", () => {
    const off = scoreTable(20, 0, true, 3);
    const explicit = scoreTable(20, 0, true, 3, { bm25: 0, dense: 0, hybrid: 0 });
    expect(explicit).toEqual(off);
  });

  it("clamps a threshold to its scale and to whole steps, and treats anything non-finite as off", () => {
    for (const retriever of RETRIEVERS) {
      const { max } = ABSTAIN_SCALE[retriever];
      expect(clampThreshold(retriever, 1e9)).toBe(max);
      expect(clampThreshold(retriever, -1e9)).toBe(0);
      for (const bad of [Infinity, -Infinity, NaN, "3", null, undefined, {}]) {
        expect(clampThreshold(retriever, bad)).toBe(0);
      }
    }
    expect(clampThreshold("bm25", 2.43)).toBe(2.4);
    expect(clampThreshold("dense", 0.456)).toBe(0.46);
    expect(clampThreshold("hybrid", 1.99)).toBe(2);
    expect(Object.values(ABSTAIN_KEY)).toEqual(["abstainBm25", "abstainDense", "abstainHybrid"]);
  });
});
