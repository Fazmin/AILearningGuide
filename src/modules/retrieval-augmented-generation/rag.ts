/**
 * A small, fully computed retrieve-then-read pipeline:
 * word-window chunking with overlap, three retrievers over the chunks (BM25, the toy dense encoder
 * from the embeddings lab, and reciprocal-rank fusion of the two), prompt assembly, and an
 * extractive reader that quotes the best-matching retrieved sentence, and a grounding check that makes the
 * pipeline abstain when the best retrieved chunk scores below a threshold.
 * Nothing here is a language model; every number is computed from the text.
 */

import { teachingEmbeddings } from "@app/module-sdk";

export interface Doc {
  id: string;
  title: string;
  text: string;
  /** Only indexed when the index is rebuilt after the June notice. */
  fresh?: boolean;
}

export const DOCS: readonly Doc[] = [
  {
    id: "handbook",
    title: "Harbor handbook (2025)",
    text:
      "Welcome to Gull Harbor. The north dock closes at 18:00 on weekdays and at 16:00 on Sundays. The south dock stays open until 21:00 every day. Pets may travel on a leash with a tag. Flares are stored in shed B. If wind exceeds 40 knots the harbor master closes both docks.",
  },
  {
    id: "timetable",
    title: "Ferry timetable (2025)",
    text:
      "The island ferry leaves the harbor at 07:15 in the morning and at 15:40 in the afternoon. A single ticket costs 12 tokens and children ride free. The crossing takes forty minutes in calm weather.",
  },
  {
    id: "notice",
    title: "Ferry notice (June 2026)",
    text: "From 1 June the afternoon ferry leaves at 16:10.",
    fresh: true,
  },
];

export type QuestionId = "hours" | "ferry" | "wind" | "dog" | "june";

export interface Question {
  id: QuestionId;
  label: string;
  text: string;
  /** The words that must sit together in one chunk for the answer to be retrievable whole. */
  span: { doc: string; start: number; end: number } | null;
  /** The fact a correct answer must contain, and facts that make it wrong or stale. */
  correct: string | null;
  stale?: string;
  wrong?: string[];
}

export const QUESTIONS: readonly Question[] = [
  {
    id: "hours",
    label: "North dock, Sunday",
    text: "When does the north dock close on Sunday?",
    span: { doc: "handbook", start: 5, end: 17 },
    correct: "16:00",
    wrong: ["18:00", "21:00"],
  },
  {
    id: "ferry",
    label: "Afternoon ferry",
    text: "What time does the afternoon ferry leave?",
    span: { doc: "notice", start: 4, end: 9 },
    correct: "16:10",
    stale: "15:40",
  },
  {
    id: "wind",
    label: "Closed for wind now?",
    text: "Is the harbor closed for wind right now?",
    span: null,
    correct: null,
  },
  {
    id: "dog",
    label: "Dog on the ferry",
    text: "Can I bring my dog on the ferry?",
    span: { doc: "handbook", start: 26, end: 35 },
    correct: "leash",
  },
  {
    id: "june",
    label: "Changed in June",
    text: "What changed in June?",
    span: { doc: "notice", start: 0, end: 9 },
    correct: "16:10",
  },
];

const STOPWORDS = new Set(
  "a an and are as at be by can could do does for from has have how i if in is it its me may must my of on or our so that the their them there these this to was we were what when where which who will with would you your".split(
    " ",
  ),
);

/** Lowercase, drop punctuation and stopwords, strip one plural or verb "s". */
export function analyze(text: string): string[] {
  const raw = text.toLowerCase().match(/\d{1,2}:\d{2}|[a-z0-9]+/g) ?? [];
  return raw
    .filter((word) => !STOPWORDS.has(word))
    .map((word) => (word.length > 3 && word.endsWith("s") && !word.endsWith("ss") ? word.slice(0, -1) : word));
}

export const splitWords = (text: string) => text.split(/\s+/).filter(Boolean);

export interface Chunk {
  id: string;
  doc: string;
  docTitle: string;
  /** Index of this chunk within its document. */
  ordinal: number;
  /** Word range [start, end). */
  start: number;
  end: number;
  text: string;
  terms: string[];
}

/** Fixed word windows: stride = size − overlap. The last window stops at the end of the document. */
export const CHUNK_MIN = 6;
export const CHUNK_MAX = 30;
/** Overlap is capped at three quarters of the window so the stride never falls below a quarter of it. */
export const maxOverlap = (size: number) => Math.floor(0.75 * size);

export function chunkDoc(doc: Doc, size: number, overlap: number): Chunk[] {
  const words = splitWords(doc.text);
  const width = Math.max(1, Math.round(size));
  const stride = Math.max(1, width - Math.max(0, Math.min(Math.round(overlap), maxOverlap(width))));
  const chunks: Chunk[] = [];
  for (let start = 0; start < words.length; start += stride) {
    const end = Math.min(words.length, start + width);
    const text = words.slice(start, end).join(" ");
    chunks.push({
      id: `${doc.id}-${chunks.length}`,
      doc: doc.id,
      docTitle: doc.title,
      ordinal: chunks.length,
      start,
      end,
      text,
      terms: analyze(text),
    });
    if (end >= words.length) break;
  }
  return chunks;
}

export function indexedDocs(afterNotice: boolean) {
  return DOCS.filter((doc) => afterNotice || !doc.fresh);
}

export function buildChunks(size: number, overlap: number, afterNotice: boolean): Chunk[] {
  return indexedDocs(afterNotice).flatMap((doc) => chunkDoc(doc, size, overlap));
}

export const BM25_K1 = 1.2;
export const BM25_B = 0.75;

export interface TermContribution {
  term: string;
  tf: number;
  idf: number;
  weight: number;
}

export interface ScoredChunk extends Chunk {
  score: number;
  contributions: TermContribution[];
  rank: number;
}

/**
 * BM25 with the non-negative IDF ln(1 + (N − n + 0.5)/(n + 0.5)).
 * Each distinct query term adds idf × tf(k1+1) / (tf + k1(1 − b + b·len/avgLen)).
 */
export function bm25(chunks: readonly Chunk[], query: string): ScoredChunk[] {
  const queryTerms = [...new Set(analyze(query))];
  const count = chunks.length;
  const avgLength = count ? chunks.reduce((sum, chunk) => sum + chunk.terms.length, 0) / count : 1;
  const docFrequency = new Map<string, number>();
  for (const term of queryTerms) {
    docFrequency.set(term, chunks.filter((chunk) => chunk.terms.includes(term)).length);
  }
  const scored = chunks.map((chunk, position) => {
    const length = chunk.terms.length;
    const contributions: TermContribution[] = [];
    for (const term of queryTerms) {
      const tf = chunk.terms.filter((item) => item === term).length;
      if (tf === 0) continue;
      const n = docFrequency.get(term) ?? 0;
      const idf = Math.log(1 + (count - n + 0.5) / (n + 0.5));
      const weight =
        (idf * tf * (BM25_K1 + 1)) / (tf + BM25_K1 * (1 - BM25_B + (BM25_B * length) / Math.max(1e-9, avgLength)));
      contributions.push({ term, tf, idf, weight });
    }
    const score = contributions.reduce((sum, entry) => sum + entry.weight, 0);
    return { ...chunk, score, contributions, rank: 0, position };
  });
  const ordered = [...scored].sort((a, b) => b.score - a.score || a.position - b.position);
  ordered.forEach((chunk, index) => (chunk.rank = index + 1));
  return ordered.map(({ position: _position, ...rest }) => rest);
}

/** Top-k chunks with a positive score. A chunk that matches no query term is never retrieved. */
export function retrieve<T extends { score: number }>(ranked: readonly T[], k: number): T[] {
  return ranked.filter((chunk) => chunk.score > 0).slice(0, Math.max(1, k));
}

export type Retriever = "bm25" | "dense" | "hybrid";
export const RETRIEVERS: readonly Retriever[] = ["bm25", "dense", "hybrid"];
export const RETRIEVER_LABEL: Record<Retriever, string> = { bm25: "BM25", dense: "Dense", hybrid: "Hybrid (RRF)" };

/** Reciprocal-rank fusion constant: a chunk at rank r in a list adds 1 / (RRF_K + r). */
export const RRF_K = 60;

export interface RankedChunk extends ScoredChunk {
  /** BM25 score and 1-based rank among chunks with a positive score (null when the score is 0). */
  bm25Score: number;
  bm25Rank: number | null;
  /** Cosine between the toy-encoder vectors of the question and the chunk, and its rank among positive cosines. */
  cosine: number;
  denseRank: number | null;
  /** Reciprocal-rank fusion of the two ranks; 0 when the chunk is in neither list. */
  rrf: number;
}

/** Dense score: cosine between mean-pooled toy-encoder vectors, with the embeddings lab's own `embed` and `cosine`. */
export function denseScores(chunks: readonly Chunk[], query: string): number[] {
  const queryVector = teachingEmbeddings.embed(query);
  return chunks.map((chunk) => teachingEmbeddings.cosine(queryVector, teachingEmbeddings.embed(chunk.text)));
}

/**
 * Rank every chunk with the chosen retriever, best first.
 * - bm25: the keyword score above.
 * - dense: cosine of the toy encoder's vectors; a chunk with cosine 0 (no shared topic, or no known word) is never retrieved.
 * - hybrid: reciprocal-rank fusion, rrf(c) = Σ over the two lists that contain c of 1 / (RRF_K + rank in that list),
 *   where a list holds only chunks with a positive score. Ties keep document order.
 * The BM25 and dense scores and ranks stay on every row so a table can show where each retriever put it.
 */
export function rankChunks(chunks: readonly Chunk[], query: string, retriever: Retriever): RankedChunk[] {
  const keyword = bm25(chunks, query);
  const bm25Rank = new Map<string, number>();
  let place = 0;
  for (const row of keyword) if (row.score > 0) bm25Rank.set(row.id, ++place);
  const cosines = denseScores(chunks, query);
  const cosineById = new Map(chunks.map((chunk, index) => [chunk.id, cosines[index]]));
  const denseOrder = chunks
    .map((chunk, position) => ({ id: chunk.id, cosine: cosineById.get(chunk.id) ?? 0, position }))
    .sort((a, b) => b.cosine - a.cosine || a.position - b.position);
  const denseRank = new Map<string, number>();
  place = 0;
  for (const row of denseOrder) if (row.cosine > 0) denseRank.set(row.id, ++place);

  const rows = chunks.map((chunk, position) => {
    const scored = keyword.find((row) => row.id === chunk.id)!;
    const bm25R = bm25Rank.get(chunk.id) ?? null;
    const denseR = denseRank.get(chunk.id) ?? null;
    const rrf = (bm25R ? 1 / (RRF_K + bm25R) : 0) + (denseR ? 1 / (RRF_K + denseR) : 0);
    const cosine = cosineById.get(chunk.id) ?? 0;
    const score = retriever === "bm25" ? scored.score : retriever === "dense" ? cosine : rrf;
    return {
      ...scored,
      score,
      rank: 0,
      bm25Score: scored.score,
      bm25Rank: bm25R,
      cosine,
      denseRank: denseR,
      rrf,
      position,
    };
  });
  const ordered = [...rows].sort((a, b) => b.score - a.score || a.position - b.position);
  ordered.forEach((row, index) => (row.rank = index + 1));
  return ordered.map(({ position: _position, ...rest }) => rest);
}

export const SYSTEM_PROMPT =
  "Answer using only the numbered context. Cite the passage as [n]. If the context does not contain the answer, say so.";

export function assemblePrompt(question: string, retrieved: readonly Chunk[]): string {
  const context = retrieved.length
    ? retrieved
        .map((chunk, index) => `[${index + 1}] ${chunk.docTitle}, words ${chunk.start + 1}–${chunk.end}: ${chunk.text}`)
        .join("\n")
    : "(no passage scored above zero for this question)";
  return `System: ${SYSTEM_PROMPT}\n\nContext:\n${context}\n\nQuestion: ${question}\nAnswer:`;
}

export const wordCount = (text: string) => splitWords(text).length;

export interface ReaderAnswer {
  sentence: string;
  /** 1-based position of the cited chunk in the prompt, or 0 when nothing was quotable. */
  citation: number;
  matched: string[];
}

/** Split chunk text into sentence pieces. A chunk that starts or ends mid-sentence yields a fragment. */
export function pieces(text: string): string[] {
  return (text.match(/[^.?!]+[.?!]?/g) ?? []).map((piece) => piece.trim()).filter(Boolean);
}

/**
 * Extractive reader: quote the retrieved sentence piece that shares the most
 * distinct query terms. Ties go to the higher-ranked chunk, then the earlier piece.
 */
export function read(question: string, retrieved: readonly Chunk[]): ReaderAnswer {
  const terms = new Set(analyze(question));
  let best: ReaderAnswer = { sentence: "", citation: 0, matched: [] };
  retrieved.forEach((chunk, chunkIndex) => {
    for (const piece of pieces(chunk.text)) {
      const matched = [...new Set(analyze(piece))].filter((term) => terms.has(term));
      if (matched.length > best.matched.length) {
        best = { sentence: piece, citation: chunkIndex + 1, matched };
      }
    }
  });
  return best;
}

export type Verdict = "correct" | "wrong" | "stale" | "unsupported" | "missed";

export interface Diagnosis {
  verdict: Verdict;
  /** Some chunk in the index holds the whole answer span. */
  spanWholeInIndex: boolean;
  /** Some retrieved chunk holds the whole answer span. */
  spanWholeRetrieved: boolean;
  /** Some retrieved chunk overlaps the answer span at all. */
  spanTouched: boolean;
  staleInContext: boolean;
  freshInContext: boolean;
  /** Share of the answer-span words that sit inside at least one retrieved chunk; null when the question has no answer in the corpus. */
  spanCoverage: number | null;
  /** The quoted sentence appears verbatim inside the retrieved passage it cites; null when nothing was quoted. */
  quoteSupported: boolean | null;
}

const covers = (chunk: Chunk, span: NonNullable<Question["span"]>) =>
  chunk.doc === span.doc && chunk.start <= span.start && chunk.end >= span.end;
const touches = (chunk: Chunk, span: NonNullable<Question["span"]>) =>
  chunk.doc === span.doc && chunk.start < span.end && chunk.end > span.start;

export function diagnose(
  question: Question,
  chunks: readonly Chunk[],
  retrieved: readonly Chunk[],
  answer: ReaderAnswer,
): Diagnosis {
  const span = question.span;
  const spanWholeInIndex = span ? chunks.some((chunk) => covers(chunk, span)) : false;
  const spanWholeRetrieved = span ? retrieved.some((chunk) => covers(chunk, span)) : false;
  const spanTouched = span ? retrieved.some((chunk) => touches(chunk, span)) : false;
  let spanCoverage: number | null = null;
  if (span) {
    let inContext = 0;
    for (let word = span.start; word < span.end; word += 1) {
      if (retrieved.some((chunk) => chunk.doc === span.doc && chunk.start <= word && word < chunk.end)) inContext += 1;
    }
    spanCoverage = inContext / (span.end - span.start);
  }
  const cited = answer.citation > 0 ? retrieved[answer.citation - 1] : undefined;
  const quoteSupported = answer.sentence ? Boolean(cited && cited.text.includes(answer.sentence)) : null;
  const contextText = retrieved.map((chunk) => chunk.text).join(" ");
  const staleInContext = question.stale ? contextText.includes(question.stale) : false;
  const freshInContext = question.correct ? contextText.includes(question.correct) : false;

  let verdict: Verdict;
  if (!question.correct) verdict = "unsupported";
  else if (answer.sentence.includes(question.correct)) verdict = "correct";
  else if (question.stale && answer.sentence.includes(question.stale)) verdict = "stale";
  else if (question.wrong?.some((fact) => answer.sentence.includes(fact))) verdict = "wrong";
  else if (!spanTouched) verdict = "missed";
  else verdict = "wrong";

  return {
    verdict,
    spanWholeInIndex,
    spanWholeRetrieved,
    spanTouched,
    staleInContext,
    freshInContext,
    spanCoverage,
    quoteSupported,
  };
}

/**
 * Where a wrong answer went wrong, read from the diagnosis:
 * - retrieval: some of the answer-span words never reached the prompt, so no reader could have used them;
 * - reader: every answer-span word was in the prompt and the reader still quoted something else;
 * - no-evidence: the corpus holds no answer, so the right output was to say so. The bare reader never does; only the grounding check can make the pipeline abstain.
 */
export type FailureSite = "none" | "retrieval" | "reader" | "no-evidence";

export function failureSite(question: Question, diagnosis: Diagnosis): FailureSite {
  if (!question.span) return "no-evidence";
  if (diagnosis.verdict === "correct") return "none";
  return (diagnosis.spanCoverage ?? 0) < 1 ? "retrieval" : "reader";
}

export function runPipeline(
  questionId: QuestionId,
  size: number,
  overlap: number,
  afterNotice: boolean,
  k: number,
  retriever: Retriever = "bm25",
) {
  const question = QUESTIONS.find((item) => item.id === questionId) ?? QUESTIONS[0];
  const chunks = buildChunks(size, overlap, afterNotice);
  const ranked = rankChunks(chunks, question.text, retriever);
  const retrieved = retrieve(ranked, k);
  const prompt = assemblePrompt(question.text, retrieved);
  const answer = read(question.text, retrieved);
  const diagnosis = diagnose(question, chunks, retrieved, answer);
  return { question, chunks, ranked, retrieved, prompt, answer, diagnosis };
}

/* ------------------------------------------------------------------------------------------------
 * Grounding check: abstain when the best retrieved chunk scores below a threshold.
 * ---------------------------------------------------------------------------------------------- */

export interface AbstainScale {
  /** Slider range and step on the retriever's own score scale. */
  max: number;
  step: number;
  /** Decimals shown for the threshold, and for a score (a score is rounded to this before it is compared). */
  decimals: number;
  scoreDecimals: number;
  unit: string;
}

/**
 * The threshold lives on each retriever's own scale because the scales are not comparable: a BM25 score is a sum of
 * idf-weighted term scores, a cosine sits between 0 and 1, and a fused rank score depends only on ranks. Hybrid is
 * shown in units of 1/(RRF_K + 1): 1 is first place in one list, 2 is first place in both.
 */
export const ABSTAIN_SCALE: Record<Retriever, AbstainScale> = {
  bm25: { max: 8, step: 0.1, decimals: 1, scoreDecimals: 2, unit: "BM25 score" },
  dense: { max: 1, step: 0.01, decimals: 2, scoreDecimals: 3, unit: "cosine" },
  hybrid: { max: 2, step: 0.02, decimals: 2, scoreDecimals: 3, unit: `fused score × ${RRF_K + 1}` },
};

/** The state key that holds each retriever's threshold. */
export const ABSTAIN_KEY: Record<Retriever, "abstainBm25" | "abstainDense" | "abstainHybrid"> = {
  bm25: "abstainBm25",
  dense: "abstainDense",
  hybrid: "abstainHybrid",
};

const roundTo = (value: number, decimals: number) => {
  const scale = 10 ** decimals;
  return Math.round(value * scale) / scale;
};

/** A threshold from untrusted input: finite, inside [0, max] and a whole number of steps. Anything else is 0 (the check is off). */
export function clampThreshold(retriever: Retriever, value: unknown): number {
  const { max, step, decimals } = ABSTAIN_SCALE[retriever];
  if (typeof value !== "number" || !Number.isFinite(value)) return 0;
  const clamped = Math.min(max, Math.max(0, value));
  return roundTo(Math.round(clamped / step) * step, decimals);
}

/** The score the check reads for one chunk, on the retriever's own scale and rounded to the precision shown. */
export function groundingScore(retriever: Retriever, chunk: Pick<RankedChunk, "bm25Score" | "cosine" | "rrf">): number {
  const raw = retriever === "bm25" ? chunk.bm25Score : retriever === "dense" ? chunk.cosine : chunk.rrf * (RRF_K + 1);
  return roundTo(raw, ABSTAIN_SCALE[retriever].scoreDecimals);
}

/**
 * What the pipeline did against what it should have done:
 * - answered-right: it answered and the quote carries the gold fact;
 * - answered-wrong: it answered and the quote is wrong, stale, or answers a question the corpus cannot (stated flatly, with a citation);
 * - abstained-right: it said the sources do not say, and the corpus has no answer;
 * - abstained-wrong: it said the sources do not say, and the corpus has the answer.
 */
export type GroundingOutcome = "answered-right" | "answered-wrong" | "abstained-right" | "abstained-wrong";

export const OUTCOME_LABEL: Record<GroundingOutcome, string> = {
  "answered-right": "answered correctly",
  "answered-wrong": "answered wrongly with confidence",
  "abstained-right": "correctly abstained",
  "abstained-wrong": "wrongly abstained",
};

export type PipelineRun = ReturnType<typeof runPipeline>;

export interface Grounding {
  /** The threshold in force, after clamping. 0 means the check is off. */
  threshold: number;
  /** The best retrieved chunk's score on the retriever's scale; null when nothing was retrieved. */
  topScore: number | null;
  abstained: boolean;
  reason: "nothing-retrieved" | "below-threshold" | null;
  /** What the reader would have done with no check. */
  ungated: Verdict;
  outcome: GroundingOutcome;
}

/**
 * The grounding check. The pipeline abstains when nothing was retrieved (there is no passage to quote) or when the
 * best retrieved chunk's score is below the threshold. A threshold of 0 therefore never abstains on score.
 */
export function ground(run: PipelineRun, retriever: Retriever, threshold: number): Grounding {
  const limit = clampThreshold(retriever, threshold);
  const best = run.retrieved[0];
  const topScore = best ? groundingScore(retriever, best) : null;
  const reason = topScore === null ? "nothing-retrieved" : topScore < limit ? "below-threshold" : null;
  const abstained = reason !== null;
  const answerable = Boolean(run.question.correct);
  const outcome: GroundingOutcome = abstained
    ? answerable
      ? "abstained-wrong"
      : "abstained-right"
    : run.diagnosis.verdict === "correct"
      ? "answered-right"
      : "answered-wrong";
  return { threshold: limit, topScore, abstained, reason, ungated: run.diagnosis.verdict, outcome };
}

export interface GroundingTotals {
  questions: number;
  answered: number;
  right: number;
  confidentWrong: number;
  abstainedRight: number;
  abstainedWrong: number;
  /** Answered ÷ all questions. */
  coverage: number;
  /** Right ÷ answered; null when nothing was answered. */
  precision: number | null;
  /** Answered correctly plus correctly abstained: the right behaviour for the question. */
  handled: number;
}

export function groundingTotals(outcomes: readonly GroundingOutcome[]): GroundingTotals {
  const count = (outcome: GroundingOutcome) => outcomes.filter((item) => item === outcome).length;
  const right = count("answered-right");
  const confidentWrong = count("answered-wrong");
  const abstainedRight = count("abstained-right");
  const abstainedWrong = count("abstained-wrong");
  const answered = right + confidentWrong;
  return {
    questions: outcomes.length,
    answered,
    right,
    confidentWrong,
    abstainedRight,
    abstainedWrong,
    coverage: outcomes.length ? answered / outcomes.length : 0,
    precision: answered ? right / answered : null,
    handled: right + abstainedRight,
  };
}

export interface SweepPoint {
  threshold: number;
  coverage: number;
  precision: number | null;
  handled: number;
}

/** Coverage and precision at every step of the slider, for one retriever and one set of pipeline controls. */
export function groundingSweep(
  size: number,
  overlap: number,
  afterNotice: boolean,
  k: number,
  retriever: Retriever,
): SweepPoint[] {
  const runs = QUESTIONS.map((question) => runPipeline(question.id, size, overlap, afterNotice, k, retriever));
  const { max, step } = ABSTAIN_SCALE[retriever];
  const steps = Math.round(max / step);
  return Array.from({ length: steps + 1 }, (_, index) => {
    const threshold = clampThreshold(retriever, index * step);
    const totals = groundingTotals(runs.map((run) => ground(run, retriever, threshold).outcome));
    return { threshold, coverage: totals.coverage, precision: totals.precision, handled: totals.handled };
  });
}

export interface PipelineScore {
  retriever: Retriever;
  /** Context recall: share of the gold-evidence words present in the retrieved chunks. Null when the corpus has no answer. */
  contextRecall: number | null;
  /** Faithfulness (support check): the quote sits verbatim in the passage it cites. Null when nothing was quoted, including when the pipeline abstained. */
  faithful: boolean | null;
  /** Correctness: the pipeline answered and the quote carries the gold fact. Null when the corpus has no answer (see `outcome`). */
  correct: boolean | null;
  site: FailureSite;
  /** The grounding check's view of this question: the best chunk's score, whether it abstained, and what that amounts to. */
  topScore: number | null;
  abstained: boolean;
  outcome: GroundingOutcome;
  /** What the reader would have done with no check. */
  ungated: Verdict;
}

export interface ScoreRow {
  question: Question;
  scores: Record<Retriever, PipelineScore>;
}

export interface ScoreTotals {
  retriever: Retriever;
  answerable: number;
  /** Mean context recall over the answerable questions. */
  meanRecall: number;
  /** Questions where the support check passed, out of all questions that produced a quote. */
  faithful: number;
  quoted: number;
  /** Questions answered correctly, out of the answerable ones. */
  correct: number;
  /** Wrong answers split by where they went wrong. */
  retrievalFailures: number;
  readerFailures: number;
  /** Answers that passed the support check and were still wrong or unanswerable. */
  faithfulButWrong: number;
  /** All questions, including the one the corpus cannot answer. */
  questions: number;
  /** Quotes that were wrong, or that answered a question the corpus cannot. */
  confidentErrors: number;
  abstainedRight: number;
  abstainedWrong: number;
  /** Answered correctly plus correctly abstained, out of all questions. */
  handled: number;
}

/** Score one question under one retriever by reusing `runPipeline` and its diagnosis. A threshold above 0 adds the grounding check. */
export function scorePipeline(
  questionId: QuestionId,
  size: number,
  overlap: number,
  afterNotice: boolean,
  k: number,
  retriever: Retriever,
  threshold = 0,
): PipelineScore {
  const run = runPipeline(questionId, size, overlap, afterNotice, k, retriever);
  const { question, diagnosis } = run;
  const grounding = ground(run, retriever, threshold);
  return {
    retriever,
    contextRecall: diagnosis.spanCoverage,
    faithful: grounding.abstained ? null : diagnosis.quoteSupported,
    correct: question.span ? !grounding.abstained && diagnosis.verdict === "correct" : null,
    site: failureSite(question, diagnosis),
    topScore: grounding.topScore,
    abstained: grounding.abstained,
    outcome: grounding.outcome,
    ungated: grounding.ungated,
  };
}

export type Thresholds = Partial<Record<Retriever, number>>;

export function scoreTable(
  size: number,
  overlap: number,
  afterNotice: boolean,
  k: number,
  thresholds: Thresholds = {},
): ScoreRow[] {
  return QUESTIONS.map((question) => ({
    question,
    scores: Object.fromEntries(
      RETRIEVERS.map((retriever) => [
        retriever,
        scorePipeline(question.id, size, overlap, afterNotice, k, retriever, thresholds[retriever] ?? 0),
      ]),
    ) as Record<Retriever, PipelineScore>,
  }));
}

export function scoreTotals(rows: readonly ScoreRow[], retriever: Retriever): ScoreTotals {
  const scores = rows.map((row) => row.scores[retriever]);
  const answerable = scores.filter((score) => score.correct !== null);
  const recalls = answerable.map((score) => score.contextRecall ?? 0);
  const grounded = groundingTotals(scores.map((score) => score.outcome));
  return {
    retriever,
    answerable: answerable.length,
    meanRecall: recalls.length ? recalls.reduce((sum, value) => sum + value, 0) / recalls.length : 0,
    faithful: scores.filter((score) => score.faithful === true).length,
    quoted: scores.filter((score) => score.faithful !== null).length,
    correct: answerable.filter((score) => score.correct).length,
    retrievalFailures: scores.filter((score) => score.site === "retrieval").length,
    readerFailures: scores.filter((score) => score.site === "reader").length,
    faithfulButWrong: scores.filter((score) => score.faithful === true && score.correct !== true).length,
    questions: grounded.questions,
    confidentErrors: grounded.confidentWrong,
    abstainedRight: grounded.abstainedRight,
    abstainedWrong: grounded.abstainedWrong,
    handled: grounded.handled,
  };
}

export interface EncoderView {
  /** Query words the toy encoder has a vector for, in order. */
  known: string[];
  /** Query words it has no vector for. They contribute nothing to a dense score. */
  ignored: string[];
}

/** Which words of a text the hand-authored encoder knows, using the embeddings lab's own lookup. */
export function encoderView(text: string): EncoderView {
  const known: string[] = [];
  const ignored: string[] = [];
  for (const token of teachingEmbeddings.tokens(text)) (token.entry === null ? ignored : known).push(token.word);
  return { known, ignored };
}
