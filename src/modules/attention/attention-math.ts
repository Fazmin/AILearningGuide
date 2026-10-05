/**
 * Module-local helpers for the attention lab. The ONNX model attends over
 * characters; these functions read its tensors, recompute one attention row
 * from Q and K, and aggregate character weights into word weights without
 * losing any probability mass.
 */
import { dot, mixVectors, scaledDotProductAttention, stableSoftmax } from "@app/module-sdk";

export const MAX_WORDS = 12;

export interface WordSpan {
  /** The word without its whitespace. */
  text: string;
  /** First character this word owns, including the whitespace before it. */
  start: number;
  /** First non-whitespace character. */
  wordStart: number;
  /** One past the last character of the word. */
  end: number;
  /** The last character of the word, which has seen the whole word under the causal mask. */
  representative: number;
}

const isSpace = (character: string) => /\s/.test(character);

/**
 * Split characters into words. Each word owns the whitespace in front of it,
 * the way GPT-style tokenizers attach a leading space to the next word. With
 * that convention, a causal row read at a word's last character only covers
 * characters owned by that word and earlier words, so summing a row per word
 * conserves its total of one.
 */
export function wordSpans(
  characters: ReadonlyArray<string>,
  maxWords = MAX_WORDS,
): WordSpan[] {
  const spans: WordSpan[] = [];
  let owner = 0;
  let wordStart = -1;

  const close = (end: number) => {
    spans.push({
      text: characters.slice(wordStart, end).join(""),
      start: owner,
      wordStart,
      end,
      representative: end - 1,
    });
    owner = end;
    wordStart = -1;
  };

  characters.forEach((character, index) => {
    if (isSpace(character)) {
      if (wordStart >= 0) close(index);
      return;
    }
    if (wordStart < 0) wordStart = index;
  });
  if (wordStart >= 0) close(characters.length);

  return spans.slice(0, maxWords);
}

/** Sum a character-level attention row inside each word's span. */
export function aggregateByWord(
  row: ReadonlyArray<number>,
  spans: ReadonlyArray<WordSpan>,
) {
  return spans.map((span) => {
    let total = 0;
    for (let index = span.start; index < span.end; index += 1) {
      total += row[index] ?? 0;
    }
    return total;
  });
}

/** Share of a row that lands on whitespace characters. */
export function whitespaceShare(
  row: ReadonlyArray<number>,
  characters: ReadonlyArray<string>,
) {
  return characters.reduce(
    (total, character, index) =>
      total + (isSpace(character) ? (row[index] ?? 0) : 0),
    0,
  );
}

/**
 * Read a [layers, batch, heads, sequence, width] tensor as one head's
 * sequence × width matrix (batch index 0).
 */
export function headMatrix(
  tensor: ArrayLike<number> | undefined,
  shape: ReadonlyArray<number>,
  layer: number,
  head: number,
): number[][] | undefined {
  if (!tensor || shape.length !== 5) return undefined;
  const [layers, batches, heads, sequence, width] = shape;
  if (batches < 1 || layer < 0 || layer >= layers || head < 0 || head >= heads) {
    return undefined;
  }
  const base = ((layer * batches) * heads + head) * sequence * width;
  if (tensor.length < base + sequence * width) return undefined;
  return Array.from({ length: sequence }, (_, position) =>
    Array.from({ length: width }, (_, dimension) =>
      Number(tensor[base + position * width + dimension]),
    ),
  );
}

/**
 * One causal attention row, softmax(q·kⱼ / √d) over positions j ≤ position.
 * Later positions get zero weight.
 */
export function causalRow(
  query: ReadonlyArray<number>,
  keys: ReadonlyArray<ReadonlyArray<number>>,
  position: number,
) {
  return scaledDotProductAttention(query, keys, { causalIndex: position });
}

/** Full causal attention matrix from per-position queries and keys. */
export function causalMatrix(
  queries: ReadonlyArray<ReadonlyArray<number>>,
  keys: ReadonlyArray<ReadonlyArray<number>>,
) {
  return queries.map((query, position) => causalRow(query, keys, position).weights);
}

export interface RowOptions {
  /**
   * Hide positions after the query behind the causal mask, as the shipped model does. Pass
   * false to lift the mask: every position becomes readable, including the ones after it.
   */
  causal?: boolean;
  /** Divide scores by the square root of the head width, as the shipped model does. */
  scaled?: boolean;
}

/**
 * One attention row from the model's own query and key vectors, with the two switches the lab
 * exposes. `causal` and `scaled` both on is exactly `causalRow`. `scores` is always the raw dot
 * product; `scaledScores` is what softmax sees (raw when scaling is off, minus infinity where the
 * mask hides a key).
 */
export function attentionRow(
  query: ReadonlyArray<number>,
  keys: ReadonlyArray<ReadonlyArray<number>>,
  position: number,
  { causal = true, scaled = true }: RowOptions = {},
) {
  const scale = scaled ? Math.sqrt(Math.max(1, query.length)) : 1;
  const scores = keys.map((key) => dot(query, key));
  const scaledScores = scores.map((score, index) =>
    causal && index > position ? Number.NEGATIVE_INFINITY : score / scale,
  );
  return { scores, scaledScores, weights: stableSoftmax(scaledScores), scale };
}

/** Full attention matrix with the lab's two switches. */
export function attentionMatrix(
  queries: ReadonlyArray<ReadonlyArray<number>>,
  keys: ReadonlyArray<ReadonlyArray<number>>,
  options: RowOptions = {},
) {
  return queries.map((query, position) => attentionRow(query, keys, position, options).weights);
}

/**
 * The matrix the same head would produce with the causal mask lifted, from the same query and key
 * vectors. The shipped model was trained causal, so this is what a bidirectional layer would be
 * allowed to see here, not what a trained bidirectional model would compute.
 */
export function bidirectionalMatrix(
  queries: ReadonlyArray<ReadonlyArray<number>>,
  keys: ReadonlyArray<ReadonlyArray<number>>,
) {
  return attentionMatrix(queries, keys, { causal: false });
}

/** Share of a row that sits on positions after `position`. Zero for any causal row. */
export function laterShare(row: ReadonlyArray<number>, position: number) {
  let total = 0;
  for (let index = position + 1; index < row.length; index += 1) total += row[index] ?? 0;
  return total;
}

/** Row weight that falls outside every word span (trailing spaces, words past the cap). */
export function unownedShare(row: ReadonlyArray<number>, spans: ReadonlyArray<WordSpan>) {
  const owned = aggregateByWord(row, spans).reduce((sum, value) => sum + value, 0);
  const total = row.reduce((sum, value) => sum + value, 0);
  return Math.max(0, total - owned);
}

/** Rows whose largest weight is at least `threshold`: the saturated rows of a head. */
export function saturatedRows(
  matrix: ReadonlyArray<ReadonlyArray<number>>,
  causal: boolean,
  threshold = 0.99,
) {
  // A causal first row has one key, so its weight is 1 whatever the scores are.
  const rows = causal ? matrix.slice(1) : matrix;
  return {
    saturated: rows.filter((row) => Math.max(0, ...row) >= threshold).length,
    total: rows.length,
  };
}

/** Copy of the value vectors with positions `from` up to (not including) `to` set to zero. */
export function zeroValues(
  values: ReadonlyArray<ReadonlyArray<number>>,
  from: number,
  to: number,
) {
  return values.map((vector, position) =>
    position >= from && position < to ? vector.map(() => 0) : [...vector],
  );
}

export function vectorLength(vector: ReadonlyArray<number>) {
  return Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
}

/** The head's output at one position: the attention-weighted sum of value vectors. */
export function mixRow(
  weights: ReadonlyArray<number>,
  values: ReadonlyArray<ReadonlyArray<number>>,
) {
  return mixVectors(weights, values);
}

export function largestAbsoluteDifference(
  left: ReadonlyArray<number>,
  right: ReadonlyArray<number>,
) {
  let largest = 0;
  const length = Math.max(left.length, right.length);
  for (let index = 0; index < length; index += 1) {
    largest = Math.max(largest, Math.abs((left[index] ?? 0) - (right[index] ?? 0)));
  }
  return largest;
}

/**
 * Deterministic stand-in vectors used only while the model is loading or if it
 * fails. They carry no learned meaning.
 */
export function fallbackVectors(
  characters: ReadonlyArray<string>,
  layer: number,
  head: number,
  kind: number,
  width = 64,
) {
  return characters.map((character, position) => {
    const code = character.charCodeAt(0) || 17;
    return Array.from({ length: width }, (_, dimension) =>
      Math.sin(
        code * 0.17 +
          position * 0.73 +
          layer * 1.91 +
          head * 2.37 +
          kind * 3.11 +
          dimension * 0.41,
      ) * 0.62,
    );
  });
}

/** Printable label for one character in an axis or table. */
export function characterLabel(character: string) {
  if (character === " ") return "␣";
  if (character === "\n") return "↵";
  if (isSpace(character)) return "·";
  return character;
}
