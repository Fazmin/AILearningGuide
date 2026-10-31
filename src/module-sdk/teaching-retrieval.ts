/**
 * Retrieval scoring shared by labs that rank passages: BM25 over pre-split terms, and the relevance
 * metrics that read a ranking against the passages a query should find.
 *
 * BM25 follows Robertson and Zaragoza (2009) with the non-negative IDF ln(1 + (N − n + 0.5)/(n + 0.5)),
 * the same formula Retrieval-augmented generation applies to its chunks.
 */
export const BM25_K1 = 1.2;
export const BM25_B = 0.75;

/** BM25 score of every document, in document order. Each distinct query term counts once. */
export function bm25Scores(
  documents: ReadonlyArray<ReadonlyArray<string>>,
  query: ReadonlyArray<string>,
  { k1 = BM25_K1, b = BM25_B }: { k1?: number; b?: number } = {},
): number[] {
  const terms = [...new Set(query)];
  const count = documents.length;
  const avgLength = count ? documents.reduce((sum, document) => sum + document.length, 0) / count : 1;
  const idf = new Map<string, number>();
  for (const term of terms) {
    const n = documents.filter((document) => document.includes(term)).length;
    idf.set(term, Math.log(1 + (count - n + 0.5) / (n + 0.5)));
  }
  return documents.map((document) => {
    let score = 0;
    for (const term of terms) {
      const tf = document.filter((item) => item === term).length;
      if (tf === 0) continue;
      score +=
        ((idf.get(term) ?? 0) * tf * (k1 + 1)) /
        (tf + k1 * (1 - b + (b * document.length) / Math.max(1e-9, avgLength)));
    }
    return score;
  });
}

/** Document indices from highest score to lowest; ties keep document order. */
export function rankByScore(scores: ReadonlyArray<number>): number[] {
  return scores
    .map((score, index) => ({ score, index }))
    .sort((a, b) => b.score - a.score || a.index - b.index)
    .map((entry) => entry.index);
}

/** 1 when any relevant document is in the top k, else 0: DPR's top-k retrieval accuracy. */
export function hitAtK(ranked: ReadonlyArray<number>, relevant: ReadonlyArray<number>, k: number): number {
  const top = ranked.slice(0, Math.max(0, k));
  return relevant.some((index) => top.includes(index)) ? 1 : 0;
}

/** Share of the relevant documents that appear in the top k. */
export function recallAtK(ranked: ReadonlyArray<number>, relevant: ReadonlyArray<number>, k: number): number {
  if (relevant.length === 0) return 0;
  const top = new Set(ranked.slice(0, Math.max(0, k)));
  return relevant.filter((index) => top.has(index)).length / relevant.length;
}

/** 1 / rank of the first relevant document, 0 when none is ranked. Averaged over queries it is MRR. */
export function reciprocalRank(ranked: ReadonlyArray<number>, relevant: ReadonlyArray<number>): number {
  const position = ranked.findIndex((index) => relevant.includes(index));
  return position < 0 ? 0 : 1 / (position + 1);
}
