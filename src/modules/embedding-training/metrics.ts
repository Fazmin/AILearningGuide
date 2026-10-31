import { teachingRetrieval } from "@app/module-sdk";
import { DUPLICATE_PAIRS, HELD_OUT_QUERIES, PASSAGES, passageIndex, relevantFor, TRAIN_QUERIES } from "./data";
import { bucketsOf, dot, encode, encodeAll, STOPWORDS, words } from "./encoder";

/**
 * How the lab grades a ranking, and the geometry readouts of the embedding space.
 *
 * Recall@k on screen is the hit rate: the share of held-out queries with at least one relevant passage
 * in the top k (what DPR calls top-k retrieval accuracy, `hitAtK`). A query whose answer has a duplicate
 * has two relevant passages, and finding either counts. MRR averages 1 / rank of the first relevant one.
 */

/** BM25's terms: the encoder's words with one trailing plural or verb s removed. */
export function keywordTerms(text: string): string[] {
  return words(text)
    .filter((word) => !STOPWORDS.has(word))
    .map((word) => (word.length > 3 && word.endsWith("s") && !word.endsWith("ss") ? word.slice(0, -1) : word));
}

export const bm25Scores = (documents: readonly (readonly string[])[], query: readonly string[]) =>
  teachingRetrieval.bm25Scores(documents, query);

const PASSAGE_TERMS = PASSAGES.map((passage) => keywordTerms(passage.text));

/** BM25 score of every passage for one query text. */
export const bm25For = (text: string) => bm25Scores(PASSAGE_TERMS, keywordTerms(text));

/**
 * Passage indices best first. Ties keep corpus order. For BM25, a passage that shares no term with the
 * query is not retrieved at all, so a query with no shared word gets an empty ranking instead of a
 * lucky corpus-order guess.
 */
export function rankScores(scores: readonly number[], dropZeros = false): number[] {
  const ranked = teachingRetrieval.rankByScore(scores);
  return dropZeros ? ranked.filter((index) => scores[index] > 0) : ranked;
}

export interface RetrievalScore {
  recall1: number;
  recall5: number;
  mrr: number;
}

/** Hit rate at 1 and 5 and mean reciprocal rank over a set of rankings. */
export function scoreRankings(rankings: readonly (readonly number[])[], relevant: readonly (readonly number[])[]): RetrievalScore {
  const count = Math.max(1, rankings.length);
  let recall1 = 0;
  let recall5 = 0;
  let mrr = 0;
  rankings.forEach((ranked, index) => {
    recall1 += teachingRetrieval.hitAtK(ranked, relevant[index], 1);
    recall5 += teachingRetrieval.hitAtK(ranked, relevant[index], 5);
    mrr += teachingRetrieval.reciprocalRank(ranked, relevant[index]);
  });
  return { recall1: recall1 / count, recall5: recall5 / count, mrr: mrr / count };
}

/** 1-based rank of the first relevant passage, or null when none is ranked. */
export function firstRelevantRank(ranked: readonly number[], relevant: readonly number[]): number | null {
  const position = ranked.findIndex((index) => relevant.includes(index));
  return position < 0 ? null : position + 1;
}

/** 1-based rank of one passage, or null when it is not ranked. */
export function rankOf(ranked: readonly number[], passage: number): number | null {
  const position = ranked.indexOf(passage);
  return position < 0 ? null : position + 1;
}

export const HELD_OUT_RELEVANT: readonly number[][] = HELD_OUT_QUERIES.map(relevantFor);
export const HELD_OUT_BUCKETS: readonly Int32Array[] = HELD_OUT_QUERIES.map((query) => bucketsOf(query.text));
export const PASSAGE_BUCKETS_ALL: readonly Int32Array[] = PASSAGES.map((passage) => bucketsOf(passage.text));

/** BM25 over the held-out queries: fixed, because nothing in it is trained. */
export const BM25_SCORES: readonly number[][] = HELD_OUT_QUERIES.map((query) => bm25For(query.text));
export const BM25_RANKINGS: readonly number[][] = BM25_SCORES.map((scores) => rankScores(scores, true));
export const BM25_RESULT: RetrievalScore = scoreRankings(BM25_RANKINGS, HELD_OUT_RELEVANT);

export interface Space {
  dim: number;
  passages: Float64Array[];
  queries: Float64Array[];
  /** cosine[q][p] for every held-out query and passage. */
  cosine: number[][];
  rankings: number[][];
}

/** Encode every passage and held-out query with the given weights and rank by cosine. */
export function buildSpace(weights: ArrayLike<number>, dim: number): Space {
  const passages = encodeAll(weights, dim, PASSAGE_BUCKETS_ALL);
  const queries = encodeAll(weights, dim, HELD_OUT_BUCKETS);
  const cosine = queries.map((query) => passages.map((passage) => dot(query, passage)));
  return { dim, passages, queries, cosine, rankings: cosine.map((row) => rankScores(row)) };
}

/** Mean cosine over every pair of distinct passages: near 1 when all vectors lean the same way. */
export function meanPairCosine(vectors: readonly ArrayLike<number>[]): number {
  let total = 0;
  let pairs = 0;
  for (let i = 0; i < vectors.length; i += 1) {
    for (let j = i + 1; j < vectors.length; j += 1) {
      total += dot(vectors[i], vectors[j]);
      pairs += 1;
    }
  }
  return pairs ? total / pairs : 0;
}

/** Mean cosine within the three duplicate pairs. */
export function duplicateCosine(passages: readonly ArrayLike<number>[]): number {
  const values = DUPLICATE_PAIRS.map(([a, b]) => dot(passages[passageIndex(a)], passages[passageIndex(b)]));
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

const squaredDistance = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let total = 0;
  for (let k = 0; k < a.length; k += 1) total += (a[k] - b[k]) ** 2;
  return total;
};

/**
 * Wang and Isola's alignment: the mean squared distance between each held-out query and its labelled
 * answer. 0 means every pair coincides; lower is better.
 */
export function alignment(space: Space): number {
  const values = HELD_OUT_QUERIES.map((query, index) => squaredDistance(space.queries[index], space.passages[passageIndex(query.answer)]));
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

/**
 * Wang and Isola's uniformity with t = 2: log of the mean of exp(−2 × squared distance) over pairs of
 * distinct passages. 0 when every passage sits on one point; more negative means more spread out.
 */
export function uniformity(vectors: readonly ArrayLike<number>[]): number {
  let total = 0;
  let pairs = 0;
  for (let i = 0; i < vectors.length; i += 1) {
    for (let j = i + 1; j < vectors.length; j += 1) {
      total += Math.exp(-2 * squaredDistance(vectors[i], vectors[j]));
      pairs += 1;
    }
  }
  return pairs ? Math.log(total / pairs) : 0;
}

/** Recall@1 when only the first `keep` numbers of every vector are used, renormalised. */
export function truncatedRecallAt1(space: Space, keep: number): number {
  const cut = (vector: Float64Array) => {
    const head = Array.from(vector.slice(0, keep));
    const length = Math.hypot(...head) || 1;
    return head.map((value) => value / length);
  };
  const passages = space.passages.map(cut);
  const rankings = space.queries.map((query) => {
    const short = cut(query);
    return rankScores(passages.map((passage) => dot(short, passage)));
  });
  return scoreRankings(rankings, HELD_OUT_RELEVANT).recall1;
}

export interface Projection {
  /** Two orthonormal directions in the d-dimensional space. */
  axes: [Float64Array, Float64Array];
  /** Share of the total squared length each axis captures. */
  captured: [number, number];
}

/**
 * The two directions that capture the most squared length of the passage vectors, by power iteration
 * on the d × d matrix of averaged outer products. The vectors are not centred first, so a lean shared
 * by every vector shows up as the first axis instead of being subtracted away.
 */
export function principalAxes(vectors: readonly ArrayLike<number>[], dim: number, iterations = 60): Projection {
  const moment = Array.from({ length: dim }, () => new Float64Array(dim));
  for (const vector of vectors) {
    for (let i = 0; i < dim; i += 1) for (let j = 0; j < dim; j += 1) moment[i][j] += (vector[i] * vector[j]) / vectors.length;
  }
  const multiply = (input: Float64Array) => {
    const output = new Float64Array(dim);
    for (let i = 0; i < dim; i += 1) output[i] = dot(moment[i], input);
    return output;
  };
  const normalise = (vector: Float64Array) => {
    const length = Math.hypot(...vector) || 1;
    return vector.map((value) => value / length);
  };
  const removeAlong = (vector: Float64Array, axis: Float64Array) => {
    const along = dot(vector, axis);
    return vector.map((value, k) => value - along * axis[k]);
  };
  const start = (offset: number) => normalise(Float64Array.from({ length: dim }, (_, k) => 1 + ((k * 7 + offset) % 5) * 0.1));
  let first = start(0);
  for (let round = 0; round < iterations; round += 1) first = normalise(multiply(first));
  let second = normalise(removeAlong(start(3), first));
  for (let round = 0; round < iterations; round += 1) second = normalise(removeAlong(multiply(second), first));
  const trace = moment.reduce((sum, row, i) => sum + row[i], 0) || 1;
  return {
    axes: [first, second],
    captured: [dot(first, multiply(first)) / trace, dot(second, multiply(second)) / trace],
  };
}

export const project = (vector: ArrayLike<number>, projection: Projection): [number, number] => [
  dot(vector, projection.axes[0]),
  dot(vector, projection.axes[1]),
];

/** The unit vector of one arbitrary text under the given weights (for the inspector's own lookups). */
export const embedText = (weights: ArrayLike<number>, dim: number, text: string) => encode(weights, dim, bucketsOf(text)).unit;

const TRAIN_QUERY_BUCKETS: readonly Int32Array[] = TRAIN_QUERIES.map((query) => bucketsOf(query.text));

/** Cosine of every training query with every passage in a space: what the reranker learns from. */
export function trainCosines(weights: ArrayLike<number>, space: Space): number[][] {
  return encodeAll(weights, space.dim, TRAIN_QUERY_BUCKETS).map((query) => space.passages.map((passage) => dot(query, passage)));
}
