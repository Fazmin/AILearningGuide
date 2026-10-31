import { tinyRandom } from "@app/module-sdk";
import { DUPLICATE_OF, PASSAGES, passageIndex, TRAIN_QUERIES } from "./data";
import { bucketsOf, encode, encodeAll, initialWeights } from "./encoder";
import { bm25Scores, keywordTerms } from "./metrics";

/**
 * Contrastive training of the bi-encoder in encoder.ts, run for real inside the page.
 *
 * Each step draws B training queries (a shuffled pass through the 90, reshuffled when it runs out).
 * The candidates are the batch's labelled answers, each counted once, plus one mined negative per
 * query when mining is on; every query is scored against every candidate, as in DPR. The loss is
 * query→passage InfoNCE:
 *
 *   loss_i = −log softmax_j( cos(q_i, p_j) / τ )[answer_i]       averaged over the B rows
 *
 * Its exact gradient flows back through the cosine, the L2 normalisation and the mean pooling into the
 * rows of W, and plain SGD takes one step. With Skip known duplicates on, a row's softmax leaves out the
 * labelled duplicate of its own answer, and the miners never return it.
 *
 * Mining: "bm25" takes, once, the highest-scoring BM25 passage that is not the answer; "nearest" takes
 * the passage nearest the query under the current encoder that is not the answer, from an index that is
 * rebuilt every 10 steps, the way ANCE refreshes its index while training continues.
 */

export const STEPS = 300;
export const SNAPSHOT_EVERY = 10;
export const REFRESH_EVERY = 10;
export const LEARNING_RATE = 8;

export const BATCH_SIZES = [2, 4, 8, 16, 32] as const;
export const NEGATIVE_MODES = ["in-batch", "bm25", "nearest"] as const;
export type NegativeMode = (typeof NEGATIVE_MODES)[number];

export const NEGATIVE_LABEL: Record<NegativeMode, string> = {
  "in-batch": "In-batch only",
  bm25: "BM25-mined",
  nearest: "Nearest wrong",
};

export interface TrainConfig {
  batch: number;
  temperature: number;
  negatives: NegativeMode;
  skipDuplicates: boolean;
  dim: number;
  seed: number;
}

export const PASSAGE_BUCKETS: readonly Int32Array[] = PASSAGES.map((passage) => bucketsOf(passage.text));
export const TRAIN_BUCKETS: readonly Int32Array[] = TRAIN_QUERIES.map((query) => bucketsOf(query.text));
export const TRAIN_ANSWERS: readonly number[] = TRAIN_QUERIES.map((query) => passageIndex(query.answer));

/** One step's batch, laid out as the loss sees it. */
export interface Batch {
  /** Training-query index of each row. */
  rows: number[];
  /** Passage index of every candidate column: distinct answers first, then distinct mined negatives. */
  candidates: number[];
  /** For each row, the column of its labelled answer. */
  answerColumn: number[];
  /** For each row, the column of its own mined negative, or −1. */
  minedColumn: number[];
  /** masked[r][c]: the column is left out of row r's softmax (the known duplicate of its answer). */
  masked: boolean[][];
}

/** The passage a miner hands one query: never its labelled answer, and never that answer's duplicate when skipping. */
export function pickNegative(
  scores: ArrayLike<number>,
  answer: number,
  skipDuplicates: boolean,
  requirePositive = false,
): number {
  let best = -1;
  let bestScore = -Infinity;
  for (let passage = 0; passage < scores.length; passage += 1) {
    if (passage === answer) continue;
    if (skipDuplicates && passage === DUPLICATE_OF[answer]) continue;
    if (requirePositive && !(scores[passage] > 0)) continue;
    if (scores[passage] > bestScore) {
      bestScore = scores[passage];
      best = passage;
    }
  }
  return best;
}

/** BM25's hard negative for every training query: fixed for the whole run. −1 when BM25 shares no word with any wrong passage. */
export function bm25Negatives(skipDuplicates: boolean): number[] {
  const documents = PASSAGES.map((passage) => keywordTerms(passage.text));
  return TRAIN_QUERIES.map((query, index) =>
    pickNegative(bm25Scores(documents, keywordTerms(query.text)), TRAIN_ANSWERS[index], skipDuplicates, true),
  );
}

/** The nearest wrong passage for every training query under the given weights. */
export function nearestNegatives(weights: ArrayLike<number>, dim: number, skipDuplicates: boolean): number[] {
  const passages = encodeAll(weights, dim, PASSAGE_BUCKETS);
  return TRAIN_BUCKETS.map((buckets, index) => {
    const query = encode(weights, dim, buckets).unit;
    const scores = passages.map((passage) => {
      let total = 0;
      for (let k = 0; k < dim; k += 1) total += query[k] * passage[k];
      return total;
    });
    return pickNegative(scores, TRAIN_ANSWERS[index], skipDuplicates);
  });
}

/** Lay out one batch: answers once each, mined negatives once each, and the duplicate mask. */
export function buildBatch(rows: readonly number[], mined: readonly number[] | null, skipDuplicates: boolean): Batch {
  const candidates: number[] = [];
  const columnOf = new Map<number, number>();
  const add = (passage: number) => {
    if (!columnOf.has(passage)) {
      columnOf.set(passage, candidates.length);
      candidates.push(passage);
    }
    return columnOf.get(passage)!;
  };
  const answerColumn = rows.map((row) => add(TRAIN_ANSWERS[row]));
  const minedColumn = rows.map((row) => {
    const negative = mined ? mined[row] : -1;
    return negative >= 0 ? add(negative) : -1;
  });
  const masked = rows.map((row) =>
    candidates.map((passage) => skipDuplicates && passage === DUPLICATE_OF[TRAIN_ANSWERS[row]]),
  );
  return { rows: [...rows], candidates, answerColumn, minedColumn, masked };
}

export interface BatchScores {
  /** Cosine of every row with every candidate. */
  cosines: number[][];
  /** Softmax share of every candidate in each row (0 where masked). */
  shares: number[][];
  /** −log share of the answer, per row. */
  rowLoss: number[];
  loss: number;
}

function softmaxRow(logits: readonly number[], masked: readonly boolean[]) {
  let high = -Infinity;
  logits.forEach((value, column) => {
    if (!masked[column] && value > high) high = value;
  });
  let total = 0;
  const shares = logits.map((value, column) => {
    if (masked[column]) return 0;
    const weight = Math.exp(value - high);
    total += weight;
    return weight;
  });
  return { shares: shares.map((value) => value / total), logSum: high + Math.log(total) };
}

/** The forward pass the loss sees, from already-encoded unit vectors. */
export function scoreBatch(
  batch: Batch,
  queryUnits: readonly ArrayLike<number>[],
  candidateUnits: readonly ArrayLike<number>[],
  temperature: number,
): BatchScores {
  const cosines = queryUnits.map((query) =>
    candidateUnits.map((passage) => {
      let total = 0;
      for (let k = 0; k < query.length; k += 1) total += query[k] * passage[k];
      return total;
    }),
  );
  const rowLoss: number[] = [];
  const shares = cosines.map((row, r) => {
    const logits = row.map((value) => value / temperature);
    const { shares: rowShares, logSum } = softmaxRow(logits, batch.masked[r]);
    rowLoss.push(logSum - logits[batch.answerColumn[r]]);
    return rowShares;
  });
  return { cosines, shares, rowLoss, loss: rowLoss.reduce((sum, value) => sum + value, 0) / Math.max(1, rowLoss.length) };
}

/** Loss of a batch under the given weights: encode, score, average. */
export function batchLoss(weights: ArrayLike<number>, dim: number, batch: Batch, temperature: number): number {
  const queries = batch.rows.map((row) => encode(weights, dim, TRAIN_BUCKETS[row]).unit);
  const passages = batch.candidates.map((passage) => encode(weights, dim, PASSAGE_BUCKETS[passage]).unit);
  return scoreBatch(batch, queries, passages, temperature).loss;
}

/**
 * The loss and its exact gradient with respect to W.
 *
 *   dL/dlogit_rc = (share_rc − [c is the answer]) / B,  dL/dcos = dL/dlogit / τ
 *   dL/dq̂_r = Σ_c dL/dcos_rc p̂_c,  dL/dp̂_c = Σ_r dL/dcos_rc q̂_r
 *   through q̂ = u / |u|:  dL/du = (dL/dq̂ − (dL/dq̂ · q̂) q̂) / |u|
 *   through u = mean of n rows: each row receives dL/du / n, once per occurrence.
 */
export function batchGradient(
  weights: ArrayLike<number>,
  dim: number,
  batch: Batch,
  temperature: number,
): { loss: number; scores: BatchScores; gradient: Float64Array } {
  const queries = batch.rows.map((row) => encode(weights, dim, TRAIN_BUCKETS[row]));
  const passages = batch.candidates.map((passage) => encode(weights, dim, PASSAGE_BUCKETS[passage]));
  const scores = scoreBatch(
    batch,
    queries.map((entry) => entry.unit),
    passages.map((entry) => entry.unit),
    temperature,
  );
  const rows = batch.rows.length;
  const columns = batch.candidates.length;
  const queryGrad = queries.map(() => new Float64Array(dim));
  const passageGrad = passages.map(() => new Float64Array(dim));
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < columns; c += 1) {
      if (batch.masked[r][c]) continue;
      const target = c === batch.answerColumn[r] ? 1 : 0;
      const coefficient = (scores.shares[r][c] - target) / (rows * temperature);
      if (coefficient === 0) continue;
      const q = queries[r].unit;
      const p = passages[c].unit;
      for (let k = 0; k < dim; k += 1) {
        queryGrad[r][k] += coefficient * p[k];
        passageGrad[c][k] += coefficient * q[k];
      }
    }
  }
  const gradient = new Float64Array(weights.length);
  const scatter = (encoded: { unit: Float64Array; norm: number }, unitGrad: Float64Array, buckets: Int32Array) => {
    if (encoded.norm === 0 || buckets.length === 0) return;
    let along = 0;
    for (let k = 0; k < dim; k += 1) along += unitGrad[k] * encoded.unit[k];
    const scale = 1 / (encoded.norm * buckets.length);
    for (let position = 0; position < buckets.length; position += 1) {
      const row = buckets[position] * dim;
      for (let k = 0; k < dim; k += 1) gradient[row + k] += (unitGrad[k] - along * encoded.unit[k]) * scale;
    }
  };
  batch.rows.forEach((row, r) => scatter(queries[r], queryGrad[r], TRAIN_BUCKETS[row]));
  batch.candidates.forEach((passage, c) => scatter(passages[c], passageGrad[c], PASSAGE_BUCKETS[passage]));
  return { loss: scores.loss, scores, gradient };
}

export interface BatchStats {
  /** Mean over rows of the labelled answer's softmax share. */
  answerShare: number;
  /** Mean over rows of the highest cosine among the row's wrong candidates (rows with none are skipped). */
  closestWrong: number;
  /** Rows that had at least one wrong candidate. */
  rowsWithNegatives: number;
}

/** How hard a batch is: what the answers already win, and how close the closest wrong candidate sits. */
export function batchStats(batch: Batch, scores: BatchScores): BatchStats {
  let share = 0;
  let closest = 0;
  let counted = 0;
  batch.rows.forEach((_, r) => {
    share += scores.shares[r][batch.answerColumn[r]];
    let best = -Infinity;
    scores.cosines[r].forEach((value, c) => {
      if (c !== batch.answerColumn[r] && !batch.masked[r][c] && value > best) best = value;
    });
    if (best > -Infinity) {
      closest += best;
      counted += 1;
    }
  });
  const rows = Math.max(1, batch.rows.length);
  return { answerShare: share / rows, closestWrong: counted ? closest / counted : 0, rowsWithNegatives: counted };
}

export interface Snapshot {
  step: number;
  weights: Float32Array;
  /** The batch the next update draws, scored with these weights. */
  batch: Batch;
  scores: BatchScores;
}

export interface TrainRun {
  config: TrainConfig;
  /** Weights and the next batch every SNAPSHOT_EVERY steps: steps 0, 10, …, 300. */
  snapshots: Snapshot[];
  /** Loss of the batch at each step 0…STEPS, before that step's update. */
  losses: number[];
  /** The answer's share and the closest wrong cosine of the batch at each step 0…STEPS. */
  stats: BatchStats[];
  /** Run averages of the two, over every step's batch. */
  meanAnswerShare: number;
  meanClosestWrong: number;
  /** Distinct answers per batch, averaged over the run: below B when rows share an answer. */
  meanDistinctAnswers: number;
}

/** Shuffled passes through the training queries, B at a time. */
function batchSampler(seed: number, batch: number) {
  const random = tinyRandom(seed * 104729 + 3);
  let order: number[] = [];
  let cursor = 0;
  const reshuffle = () => {
    order = TRAIN_QUERIES.map((_, index) => index);
    for (let index = order.length - 1; index > 0; index -= 1) {
      const swap = Math.floor(random() * (index + 1));
      [order[index], order[swap]] = [order[swap], order[index]];
    }
    cursor = 0;
  };
  reshuffle();
  return () => {
    const rows: number[] = [];
    while (rows.length < batch) {
      if (cursor >= order.length) reshuffle();
      const next = order[cursor];
      cursor += 1;
      if (!rows.includes(next)) rows.push(next);
    }
    return rows;
  };
}

export interface TrainOptions {
  steps?: number;
  learningRate?: number;
  initScale?: number;
  initOffset?: number;
}

export function train(config: TrainConfig, options: TrainOptions = {}): TrainRun {
  const { batch: batchSize, temperature, negatives, skipDuplicates, dim, seed } = config;
  const steps = options.steps ?? STEPS;
  const learningRate = options.learningRate ?? LEARNING_RATE;
  const weights = initialWeights(seed, dim, options.initScale, options.initOffset);
  const nextRows = batchSampler(seed, batchSize);
  const fixedMined = negatives === "bm25" ? bm25Negatives(skipDuplicates) : null;
  let mined: number[] | null = fixedMined;
  const snapshots: Snapshot[] = [];
  const losses: number[] = [];
  const stats: BatchStats[] = [];
  let answersTotal = 0;
  for (let step = 0; step <= steps; step += 1) {
    if (negatives === "nearest" && step % REFRESH_EVERY === 0) mined = nearestNegatives(weights, dim, skipDuplicates);
    const batch = buildBatch(nextRows(), mined, skipDuplicates);
    const { loss, scores, gradient } = batchGradient(weights, dim, batch, temperature);
    losses.push(loss);
    stats.push(batchStats(batch, scores));
    answersTotal += new Set(batch.answerColumn).size;
    if (step % SNAPSHOT_EVERY === 0) snapshots.push({ step, weights: Float32Array.from(weights), batch, scores });
    if (step === steps) break;
    for (let index = 0; index < weights.length; index += 1) weights[index] -= learningRate * gradient[index];
  }
  const mean = (values: number[]) => values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
  return {
    config,
    snapshots,
    losses,
    stats,
    meanAnswerShare: mean(stats.map((entry) => entry.answerShare)),
    meanClosestWrong: mean(stats.filter((entry) => entry.rowsWithNegatives > 0).map((entry) => entry.closestWrong)),
    meanDistinctAnswers: answersTotal / (steps + 1),
  };
}
