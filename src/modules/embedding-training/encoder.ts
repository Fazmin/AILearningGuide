import { tinyRandom } from "@app/module-sdk";

/**
 * The bi-encoder this lab trains. One function maps any text, query or passage, to a unit vector:
 *
 *   1. lowercase, split into words, drop a short list of function words;
 *   2. pad each word as #word# and cut it into character trigrams (#do, doc, ock, ck#), as DSSM's
 *      letter trigrams do, so a typo or a plural still shares most of its pieces with the clean word;
 *   3. hash each trigram with 32-bit FNV-1a into one of 512 buckets;
 *   4. average the rows of a 512 × d weight table W picked out by those buckets (mean pooling);
 *   5. divide by the length (L2 normalisation), so cosine similarity is a plain dot product.
 *
 * W is the only thing training changes. It starts as seeded Gaussian noise plus one shared offset
 * added to every row, so every text's vector leans the same way before training: a constructed
 * stand-in for the anisotropy measured in real pretrained encoders, not a copy of one.
 */

export const BUCKETS = 512;
export const WIDTHS = [8, 16] as const;
export type Width = (typeof WIDTHS)[number];

/** Spread of the Gaussian part of each row, and the length of the offset shared by every row. */
export const INIT_SCALE = 0.42;
export const INIT_OFFSET = 1;

/** Function words dropped before trigrams are cut. Question words such as when, where and how stay. */
export const STOPWORDS = new Set(
  "a an and are as at be been by can could did do does for from had has have i if in into is it its me my of on or our so that the their them there these this to was we were will with would you your am".split(
    " ",
  ),
);

/** Lowercase words and clock times, apostrophes dropped, function words removed. */
export function words(text: string): string[] {
  const raw = text.toLowerCase().replace(/['’]/g, "").match(/\d{1,2}:\d{2}|[a-z0-9]+/g) ?? [];
  return raw.filter((word) => !STOPWORDS.has(word));
}

/** Character trigrams of one word, padded with # at both ends: dock → #do, doc, ock, ck#. */
export function trigramsOf(word: string): string[] {
  const padded = `#${word}#`;
  const pieces: string[] = [];
  for (let start = 0; start + 3 <= padded.length; start += 1) pieces.push(padded.slice(start, start + 3));
  return pieces;
}

/** Every trigram of a text, repeats kept, in reading order. */
export function textTrigrams(text: string): string[] {
  return words(text).flatMap(trigramsOf);
}

/** 32-bit FNV-1a over the UTF-16 code units of a string. */
export function fnv1a(text: string): number {
  let hash = 0x811c9dc5;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash >>> 0;
}

export const bucketOf = (trigram: string) => fnv1a(trigram) % BUCKETS;

/** The bucket of every trigram in a text, repeats kept. Mean pooling averages these rows. */
export function bucketsOf(text: string): Int32Array {
  return Int32Array.from(textTrigrams(text).map(bucketOf));
}

/** Seeded standard normal draws (Box–Muller over the SDK's tinyRandom). */
export function gaussian(seed: number) {
  const random = tinyRandom(seed);
  let spare: number | null = null;
  return () => {
    if (spare !== null) {
      const value = spare;
      spare = null;
      return value;
    }
    const u = Math.max(1e-12, random());
    const v = random();
    const radius = Math.sqrt(-2 * Math.log(u));
    spare = radius * Math.sin(2 * Math.PI * v);
    return radius * Math.cos(2 * Math.PI * v);
  };
}

/** A unit-length direction drawn from the seed: the shared lean every row starts with. */
export function offsetDirection(seed: number, dim: number): Float64Array {
  const draw = gaussian(seed * 7919 + 17);
  const direction = new Float64Array(dim);
  for (let k = 0; k < dim; k += 1) direction[k] = draw();
  const length = Math.hypot(...direction) || 1;
  for (let k = 0; k < dim; k += 1) direction[k] /= length;
  return direction;
}

/** W at step 0: row b = INIT_SCALE × Gaussian noise + INIT_OFFSET × the shared direction. */
export function initialWeights(seed: number, dim: number, scale = INIT_SCALE, offset = INIT_OFFSET): Float64Array {
  const draw = gaussian(seed);
  const lean = offsetDirection(seed, dim);
  const weights = new Float64Array(BUCKETS * dim);
  for (let bucket = 0; bucket < BUCKETS; bucket += 1) {
    for (let k = 0; k < dim; k += 1) weights[bucket * dim + k] = scale * draw() + offset * lean[k];
  }
  return weights;
}

export interface Encoded {
  /** The mean-pooled vector before normalisation. */
  pooled: Float64Array;
  /** Its length; the unit vector is pooled / norm. */
  norm: number;
  unit: Float64Array;
}

/** Mean of the selected rows, then L2 normalisation. A text with no trigrams encodes to the zero vector. */
export function encode(weights: ArrayLike<number>, dim: number, buckets: ArrayLike<number>): Encoded {
  const pooled = new Float64Array(dim);
  const count = buckets.length;
  for (let position = 0; position < count; position += 1) {
    const row = buckets[position] * dim;
    for (let k = 0; k < dim; k += 1) pooled[k] += weights[row + k];
  }
  if (count > 0) for (let k = 0; k < dim; k += 1) pooled[k] /= count;
  let squared = 0;
  for (let k = 0; k < dim; k += 1) squared += pooled[k] * pooled[k];
  const norm = Math.sqrt(squared);
  const unit = new Float64Array(dim);
  if (norm > 0) for (let k = 0; k < dim; k += 1) unit[k] = pooled[k] / norm;
  return { pooled, norm, unit };
}

export const dot = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let total = 0;
  for (let k = 0; k < a.length; k += 1) total += a[k] * b[k];
  return total;
};

/** Unit vectors for many texts at once. */
export function encodeAll(weights: ArrayLike<number>, dim: number, texts: readonly Int32Array[]): Float64Array[] {
  return texts.map((buckets) => encode(weights, dim, buckets).unit);
}
