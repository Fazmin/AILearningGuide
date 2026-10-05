import { teachingTransformer } from "@app/module-sdk";

/**
 * The logit lens on the shipped two-layer character transformer.
 *
 * The model ends with a final LayerNorm and a linear "unembedding" that turns the last
 * position's residual stream into one logit per character. Normally that readout is applied
 * once, after layer 2. The logit lens applies the same readout to the residual stream at
 * earlier points as well, so each stage reports the next-character distribution it would
 * predict if the network stopped there. Early stages were never trained to be read this way,
 * so the readout shows what is linearly readable from the state, not what a layer "decides".
 */

type Weights = teachingTransformer.TransformerWeights;

/** The final LayerNorm uses PyTorch's default epsilon, the one train_language_models.py exports. */
const LAYER_NORM_EPSILON = 1e-5;

export const LENS_STAGES = [
  { id: "embeddings", name: "After embeddings", detail: "token + position" },
  { id: "layer-1", name: "After layer 1", detail: "attention + MLP" },
  { id: "layer-2", name: "After layer 2", detail: "the model's own output" },
] as const;

export interface LensCandidate {
  index: number;
  p: number;
}

export interface LensStage {
  id: (typeof LENS_STAGES)[number]["id"];
  name: string;
  /** Raw unembedding output, one logit per vocabulary entry. */
  logits: Float64Array;
  probabilities: Float64Array;
  entropyBits: number;
  /** Most likely characters first. */
  top: LensCandidate[];
  /** Where the character the finished model picks ranks at this stage (1 = most likely). */
  finalRank: number;
  /** The probability this stage gives that character. */
  finalProbability: number;
}

export interface LensResult {
  /** Number of characters the model read (at most its 64-character context). */
  length: number;
  stages: LensStage[];
  /** The character the finished model ranks first. */
  finalIndex: number;
  /**
   * Index into `stages` of the first stage from which the finished model's first choice is also
   * this stage's first choice, and stays so at every later stage.
   */
  firstStage: number;
  /** Largest gap between the readout of the last stage and the model's logits from the full forward pass. */
  readoutError: number;
}

/** Final LayerNorm and unembedding applied to one residual-stream vector of `weights.width` numbers. */
export function readout(weights: Weights, hidden: ArrayLike<number>): Float64Array {
  const { width, vocabulary, finalNormWeight, finalNormBias, unembedding } = weights;
  let mean = 0;
  for (let column = 0; column < width; column += 1) mean += hidden[column];
  mean /= width;
  let variance = 0;
  for (let column = 0; column < width; column += 1) {
    const gap = hidden[column] - mean;
    variance += gap * gap;
  }
  const scale = 1 / Math.sqrt(variance / width + LAYER_NORM_EPSILON);
  const normed = new Float64Array(width);
  for (let column = 0; column < width; column += 1) {
    normed[column] = (hidden[column] - mean) * scale * finalNormWeight[column] + finalNormBias[column];
  }
  // unembedding is [width, vocabulary], input-major, and has no bias.
  const logits = new Float64Array(vocabulary);
  for (let inner = 0; inner < width; inner += 1) {
    const value = normed[inner];
    const offset = inner * vocabulary;
    for (let column = 0; column < vocabulary; column += 1) logits[column] += value * unembedding[offset + column];
  }
  return logits;
}

function entropyOf(probabilities: ArrayLike<number>) {
  let bits = 0;
  for (let index = 0; index < probabilities.length; index += 1) {
    const p = probabilities[index];
    if (p > 0) bits -= p * Math.log2(p);
  }
  return bits;
}

const TOP_COUNT = 5;

/**
 * Runs the model on `ids` (the last 64 are used) and reads the last position's residual stream
 * after the embeddings, after layer 1 and after layer 2 through the model's own final norm and
 * unembedding.
 */
export function logitLens(weights: Weights, ids: ReadonlyArray<number>): LensResult {
  const context = ids.slice(-weights.context);
  if (context.length === 0) throw new Error("The logit lens needs at least one character.");
  const { width, vocabulary } = weights;
  const run = teachingTransformer.runTransformer(weights, context);
  const last = context.length - 1;

  const embedded = new Float64Array(width);
  for (let column = 0; column < width; column += 1) {
    embedded[column] = weights.tokenEmbedding[context[last] * width + column] + weights.positionEmbedding[last * width + column];
  }
  const hiddenStates: ArrayLike<number>[] = [
    embedded,
    ...run.residuals.map((residual) => teachingTransformer.rowOf(residual, last, width)),
  ];
  const logitsByStage = hiddenStates.map((hidden) => readout(weights, hidden));
  const finalLogits = logitsByStage[logitsByStage.length - 1];
  const finalIndex = finalLogits.indexOf(Math.max(...finalLogits));

  const stages: LensStage[] = logitsByStage.map((logits, stage) => {
    const probabilities = teachingTransformer.softmax(logits);
    const order = Array.from(probabilities, (p, index) => ({ index, p })).sort((a, b) => b.p - a.p || a.index - b.index);
    return {
      id: LENS_STAGES[stage].id,
      name: LENS_STAGES[stage].name,
      logits,
      probabilities,
      entropyBits: entropyOf(probabilities),
      top: order.slice(0, TOP_COUNT),
      finalRank: order.findIndex((entry) => entry.index === finalIndex) + 1,
      finalProbability: probabilities[finalIndex],
    };
  });

  let firstStage = stages.length - 1;
  while (firstStage > 0 && stages[firstStage - 1].top[0].index === finalIndex) firstStage -= 1;

  const reference = teachingTransformer.rowOf(run.logits, last, vocabulary);
  let readoutError = 0;
  for (let column = 0; column < vocabulary; column += 1) {
    readoutError = Math.max(readoutError, Math.abs(finalLogits[column] - reference[column]));
  }
  return { length: context.length, stages, finalIndex, firstStage, readoutError };
}
