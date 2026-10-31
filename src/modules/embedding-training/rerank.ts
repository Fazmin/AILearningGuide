import { HELD_OUT_QUERIES, PASSAGES, relevantFor, TRAIN_QUERIES } from "./data";
import { textTrigrams } from "./encoder";
import { bm25For, keywordTerms, rankScores, scoreRankings, type RetrievalScore } from "./metrics";

/**
 * A cross-encoder stand-in: a logistic regression over five features of a (query, passage) pair,
 * trained live on the training queries. Four of the features are computed from both texts at once,
 * which is the one thing a real cross-encoder does that a bi-encoder cannot: it reads the query and
 * the passage together. A real cross-encoder is a transformer that does this over every token; this is
 * five hand-picked numbers and five learned weights, so it shows the shape and the cost, not the power.
 *
 *   1. cosine     the bi-encoder's score (the first stage)
 *   2. bm25       the keyword score of the passage for this query
 *   3. slots      exact slot words: +1 for each north/south/east, day, time of day, month or child/adult
 *                 word the passage repeats, −1 for each the passage contradicts with another word of
 *                 the same kind
 *   4. coverage   share of the query's keyword terms that appear in the passage
 *   5. jaccard    overlap of the two texts' character-trigram sets
 *
 * Reranking reorders only the bi-encoder's top 5; everything below keeps its place.
 */

export const RERANK_DEPTH = 5;
export const FEATURE_NAMES = ["cosine", "bm25", "slots", "coverage", "jaccard"] as const;

/** Words that fill the same slot in different notices. Two different words from one group contradict each other. */
export const SLOT_GROUPS: readonly (readonly string[])[] = [
  ["north", "south", "east"],
  ["sunday", "saturday", "weekday", "weekend", "monday", "friday"],
  ["morning", "afternoon", "evening", "night", "midnight", "dawn"],
  ["november", "march", "january", "winter"],
  ["child", "children", "kid", "adult", "under"],
];

const SLOT_OF = new Map<string, number>();
SLOT_GROUPS.forEach((group, index) => group.forEach((word) => SLOT_OF.set(word, index)));
const CANONICAL: Record<string, string> = { children: "child", kid: "child", under: "child" };
const canonical = (word: string) => CANONICAL[word] ?? word;

const slotValues = (text: string) => new Set(keywordTerms(text).filter((word) => SLOT_OF.has(word)).map(canonical));
const groupOf = (value: string) => SLOT_OF.get(value) ?? -1;

interface PassageInfo {
  terms: Set<string>;
  trigrams: Set<string>;
  slots: Set<string>;
}

const PASSAGE_INFO: readonly PassageInfo[] = PASSAGES.map((passage) => ({
  terms: new Set(keywordTerms(passage.text)),
  trigrams: new Set(textTrigrams(passage.text)),
  slots: slotValues(passage.text),
}));

export interface QueryInfo {
  text: string;
  terms: string[];
  trigrams: Set<string>;
  slots: string[];
  bm25: number[];
}

export function queryInfo(text: string): QueryInfo {
  return {
    text,
    terms: [...new Set(keywordTerms(text))],
    trigrams: new Set(textTrigrams(text)),
    slots: [...slotValues(text)],
    bm25: bm25For(text),
  };
}

export const TRAIN_INFO: readonly QueryInfo[] = TRAIN_QUERIES.map((query) => queryInfo(query.text));
export const HELD_OUT_INFO: readonly QueryInfo[] = HELD_OUT_QUERIES.map((query) => queryInfo(query.text));

/** +1 per query slot word the passage repeats, −1 per query slot the passage fills with a different word. */
export function slotMatch(query: QueryInfo, passage: number): number {
  const slots = PASSAGE_INFO[passage].slots;
  let score = 0;
  for (const value of query.slots) {
    if (slots.has(value)) score += 1;
    else if ([...slots].some((other) => groupOf(other) === groupOf(value))) score -= 1;
  }
  return score;
}

export function coverage(query: QueryInfo, passage: number): number {
  if (query.terms.length === 0) return 0;
  const terms = PASSAGE_INFO[passage].terms;
  return query.terms.filter((term) => terms.has(term)).length / query.terms.length;
}

export function trigramJaccard(query: QueryInfo, passage: number): number {
  const other = PASSAGE_INFO[passage].trigrams;
  if (query.trigrams.size === 0 && other.size === 0) return 0;
  let shared = 0;
  for (const piece of query.trigrams) if (other.has(piece)) shared += 1;
  return shared / (query.trigrams.size + other.size - shared);
}

export const pairFeatures = (query: QueryInfo, passage: number, cosine: number): number[] => [
  cosine,
  query.bm25[passage],
  slotMatch(query, passage),
  coverage(query, passage),
  trigramJaccard(query, passage),
];

export interface Reranker {
  /** Weights on the standardised features. */
  weights: number[];
  bias: number;
  mean: number[];
  scale: number[];
  examples: number;
}

const sigmoid = (value: number) => 1 / (1 + Math.exp(-value));

/** Full-batch gradient descent on the logistic loss over standardised features, with a small L2 penalty. */
export function fitLogistic(rows: readonly number[][], labels: readonly number[], rounds = 300, rate = 0.5, l2 = 0.01): Reranker {
  const width = FEATURE_NAMES.length;
  const count = Math.max(1, rows.length);
  const mean = Array.from({ length: width }, (_, k) => rows.reduce((sum, row) => sum + row[k], 0) / count);
  const scale = Array.from({ length: width }, (_, k) => {
    const variance = rows.reduce((sum, row) => sum + (row[k] - mean[k]) ** 2, 0) / count;
    return Math.sqrt(variance) || 1;
  });
  const standard = rows.map((row) => row.map((value, k) => (value - mean[k]) / scale[k]));
  const weights = new Array<number>(width).fill(0);
  let bias = 0;
  const grad = new Array<number>(width);
  for (let round = 0; round < rounds; round += 1) {
    grad.fill(0);
    let gradBias = 0;
    for (let index = 0; index < standard.length; index += 1) {
      const row = standard[index];
      let logit = bias;
      for (let k = 0; k < width; k += 1) logit += row[k] * weights[k];
      const error = sigmoid(logit) - labels[index];
      for (let k = 0; k < width; k += 1) grad[k] += error * row[k];
      gradBias += error;
    }
    for (let k = 0; k < width; k += 1) weights[k] -= rate * (grad[k] / count + l2 * weights[k]);
    bias -= rate * (gradBias / count);
  }
  return { weights, bias, mean, scale, examples: rows.length };
}

export const rerankScore = (model: Reranker, features: readonly number[]) =>
  features.reduce((sum, value, k) => sum + ((value - model.mean[k]) / model.scale[k]) * model.weights[k], model.bias);

/**
 * Training examples: each training query's bi-encoder top 5 under the current weights, with its labelled
 * answer added when the shortlist missed it. Label 1 for the answer or its duplicate, 0 otherwise.
 */
export function trainReranker(trainCosine: readonly (readonly number[])[]): Reranker {
  const rows: number[][] = [];
  const labels: number[] = [];
  TRAIN_QUERIES.forEach((query, index) => {
    const cosines = trainCosine[index];
    const relevant = relevantFor(query);
    const shortlist = rankScores([...cosines]).slice(0, RERANK_DEPTH);
    if (!shortlist.includes(relevant[0])) shortlist.push(relevant[0]);
    for (const passage of shortlist) {
      rows.push(pairFeatures(TRAIN_INFO[index], passage, cosines[passage]));
      labels.push(relevant.includes(passage) ? 1 : 0);
    }
  });
  return fitLogistic(rows, labels);
}

/** Reorder the top 5 of a bi-encoder ranking by the reranker's score; the rest keep their places. */
export function rerankRanking(model: Reranker, query: QueryInfo, ranking: readonly number[], cosines: readonly number[]): number[] {
  const head = ranking.slice(0, RERANK_DEPTH).map((passage, position) => ({
    passage,
    position,
    score: rerankScore(model, pairFeatures(query, passage, cosines[passage])),
  }));
  head.sort((a, b) => b.score - a.score || a.position - b.position);
  return [...head.map((entry) => entry.passage), ...ranking.slice(RERANK_DEPTH)];
}

/** Rerank every held-out query's bi-encoder ranking and score the result. */
export function rerankHeldOut(
  model: Reranker,
  rankings: readonly (readonly number[])[],
  cosines: readonly (readonly number[])[],
  relevant: readonly (readonly number[])[],
): { rankings: number[][]; score: RetrievalScore } {
  const reranked = HELD_OUT_INFO.map((query, index) => rerankRanking(model, query, rankings[index], cosines[index]));
  return { rankings: reranked, score: scoreRankings(reranked, relevant) };
}

/**
 * Model passes needed to score one query. The bi-encoder encodes the query once and compares it with
 * passage vectors stored in advance; each reranked passage costs one more pass over the joined pair.
 */
export const passesPerQuery = (rerank: boolean) => 1 + (rerank ? RERANK_DEPTH : 0);
export const CROSS_ENCODER_ONLY_PASSES = PASSAGES.length;
