/**
 * The contrastive objective on a batch of three image–caption pairs, computed for real over
 * HAND-AUTHORED vectors.
 *
 * Nothing here is an encoder's output. Each image and each caption gets one short vector that we chose,
 * with four named topics so a reader can see why two items are close. A real CLIP encoder pair learns
 * anonymous vectors of 512 or 768 numbers until the matched pairs win. What is computed from these
 * vectors is the objective itself: cosine similarities, a softmax over each row and each column after
 * dividing by a temperature, and the symmetric cross-entropy (Radford et al., 2021).
 *
 *   logits[i][j] = cos(image_i, caption_j) / τ
 *   image→caption loss = mean over images of  −log softmax_row(logits)[i][positive(i)]
 *   caption→image loss = mean over captions of −log softmax_column(logits)[positive⁻¹(j)][j]
 *   loss               = (image→caption + caption→image) / 2
 */

export const TOPICS = ["vehicle", "animal", "sky", "structure"] as const;

export interface Item {
  id: string;
  /** What the picture shows, or the caption text. */
  label: string;
  vector: readonly number[];
}

export const IMAGES: readonly Item[] = [
  { id: "A", label: "a ferry at a dock", vector: [0.9, 0.1, 0.3, 0.7] },
  { id: "B", label: "a dog on a pier", vector: [0.2, 0.9, 0.1, 0.8] },
  { id: "C", label: "a ship under storm clouds", vector: [0.7, 0.0, 0.8, 0.4] },
];

export const CAPTIONS: readonly Item[] = [
  { id: "1", label: "a ferry tied up at the dock", vector: [0.8, 0.0, 0.1, 0.8] },
  { id: "2", label: "a dog waiting on the pier", vector: [0.0, 1.0, 0.0, 0.7] },
  { id: "3", label: "dark clouds over the water", vector: [0.4, 0.1, 1.0, 0.2] },
];

export const TEMPERATURE_RANGE = { min: 0.01, max: 1 } as const;
export const DEFAULT_TEMPERATURE = 0.2;

export const PAIRINGS = ["true", "shuffled"] as const;
export type Pairing = (typeof PAIRINGS)[number];

/** For each image, the index of the caption the labels call its match. */
export const POSITIVES: Record<Pairing, readonly number[]> = {
  true: [0, 1, 2],
  shuffled: [1, 2, 0],
};

export const cosine = (a: readonly number[], b: readonly number[]) => {
  const dot = a.reduce((sum, value, index) => sum + value * b[index], 0);
  const lengths = Math.hypot(...a) * Math.hypot(...b);
  return lengths === 0 ? 0 : dot / lengths;
};

/** Cosine similarity of every image with every caption: rows are images, columns are captions. */
export const SIMILARITY: readonly (readonly number[])[] = IMAGES.map((image) =>
  CAPTIONS.map((caption) => cosine(image.vector, caption.vector)),
);

/** −log of the softmax probability, computed as logsumexp − logit so a tiny share cannot overflow. */
function logSumExp(values: readonly number[]) {
  const high = Math.max(...values);
  return high + Math.log(values.reduce((sum, value) => sum + Math.exp(value - high), 0));
}

const softmax = (values: readonly number[]) => {
  const total = logSumExp(values);
  return values.map((value) => Math.exp(value - total));
};

export const boundedTemperature = (value: unknown) =>
  typeof value === "number" && Number.isFinite(value)
    ? Math.round(Math.min(TEMPERATURE_RANGE.max, Math.max(TEMPERATURE_RANGE.min, value)) * 100) / 100
    : DEFAULT_TEMPERATURE;

export const pairingOf = (value: unknown): Pairing => (value === "shuffled" ? "shuffled" : "true");

export interface Contrast {
  temperature: number;
  positives: readonly number[];
  /** Row i: the share each caption gets for image i, summing to 1. */
  rowShares: number[][];
  /** Column j: the share each image gets for caption j, summing to 1. */
  columnShares: number[][];
  imageToCaption: number;
  captionToImage: number;
  loss: number;
  /** Images whose highest-scoring caption is the one the labels call its match. */
  imagesRight: number;
  captionsRight: number;
}

/** The objective for any square similarity matrix, so the formulas can be tested on matrices other than ours. */
export function contrastFor(
  similarity: readonly (readonly number[])[],
  temperature: number,
  positives: readonly number[],
): Contrast {
  const tau = boundedTemperature(temperature);
  const size = similarity.length;
  const logits = similarity.map((row) => row.map((value) => value / tau));
  const rowShares = logits.map((row) => softmax(row));
  const columns = logits[0].map((_, j) => logits.map((row) => row[j]));
  const columnSharesByCaption = columns.map((column) => softmax(column));
  // columnShares[i][j]: the share image i gets in caption j's column, laid out like the matrix.
  const columnShares = logits.map((_, i) => columns.map((__, j) => columnSharesByCaption[j][i]));
  const imageToCaption = logits.reduce((sum, row, i) => sum + (logSumExp(row) - row[positives[i]]), 0) / size;
  const captionToImage =
    columns.reduce((sum, column, j) => sum + (logSumExp(column) - column[positives.indexOf(j)]), 0) / size;
  const argmax = (values: readonly number[]) => values.indexOf(Math.max(...values));
  return {
    temperature: tau,
    positives,
    rowShares,
    columnShares,
    imageToCaption,
    captionToImage,
    loss: (imageToCaption + captionToImage) / 2,
    imagesRight: similarity.filter((row, i) => argmax(row) === positives[i]).length,
    captionsRight: columns.filter((column, j) => argmax(column) === positives.indexOf(j)).length,
  };
}

export const contrast = (temperature: number, pairing: Pairing): Contrast =>
  contrastFor(SIMILARITY, temperature, POSITIVES[pairing]);

/** What a guess spread evenly over the three candidates scores: ln 3. */
export const CHANCE_LOSS = Math.log(IMAGES.length);
