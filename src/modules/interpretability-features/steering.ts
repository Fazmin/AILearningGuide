/**
 * Steering with a decoder direction, using the SDK's TypeScript forward pass of the
 * shipped transformer (the same weights the attention lab runs through ONNX).
 *
 * The SAE reads the residual stream after the transformer's last block. Adding
 * `strength x direction` to that stream at one position changes only what the
 * final normalization and the output matrix make of that position, so the whole
 * effect is on the next-character logits at that position: there is no later
 * layer to carry it elsewhere. `teachingTransformer.finalLogitsWithPatch` is the
 * SDK's patch intervention; `runTransformer(..., { patch })` gives the same logits
 * from a full forward pass (a test checks that the two agree).
 *
 * Units: the SAE sees x = (r - mean) / scale and each decoder direction has unit
 * length in x-space, so a strength of s adds s x scale x direction to the raw
 * residual r. It is the amount by which the feature's activation is raised.
 */

import { teachingTransformer, tinyRandom } from "@app/module-sdk";
import { WIDTH, type SaeWeights } from "./sae";
import { ITOS, UNKNOWN, encodeText } from "./vocabulary";

/** Index into the transformer's list of residual streams that the SAE reads: after block 2. */
export const SAE_LAYER = 1;
/** Random directions drawn for the null control. */
export const NULL_DIRECTIONS = 20;
export const MAX_STRENGTH = 15;

export type TransformerWeights = teachingTransformer.TransformerWeights;

/**
 * Features whose label was written by measurement and is shown in the steering picker.
 * `character` and `threshold` define the rule "fires on this character": the activation
 * exceeds `threshold`. sae.test.ts re-measures each rule on a fixed passage and
 * requires the feature to be unflagged in the cache.
 */
export interface LabelledFeature {
  feature: number;
  character: string;
  threshold: number;
}

export const LABELLED_FEATURES: ReadonlyArray<LabelledFeature> = [
  { feature: 160, character: "-", threshold: 2 },
  { feature: 683, character: "y", threshold: 2 },
  { feature: 305, character: "T", threshold: 2 },
  { feature: 121, character: " ", threshold: 5 },
  { feature: 122, character: "m", threshold: 1 },
];

export const labelledFeature = (feature: number) => LABELLED_FEATURES.find((entry) => entry.feature === feature);

/** The label as the lab words it, rule included. */
export const labelText = (entry: LabelledFeature) =>
  `fires on ${entry.character === " " ? "spaces" : entry.character === "-" ? "hyphens" : `“${entry.character}”`} (above ${entry.threshold.toFixed(1)})`;

/** One feature's decoder direction: a unit vector in the SAE's normalized space. */
export function decoderDirection(sae: SaeWeights, feature: number) {
  return sae.decoder.slice(feature * WIDTH, (feature + 1) * WIDTH);
}

/** A unit-length direction drawn from a seeded Gaussian, the "null" a real feature is compared with. */
export function randomDirection(seed: number) {
  const random = tinyRandom(seed);
  const out = new Float32Array(WIDTH);
  for (let index = 0; index < WIDTH; index += 2) {
    const radius = Math.sqrt(-2 * Math.log(Math.max(random(), 1e-12)));
    const angle = 2 * Math.PI * random();
    out[index] = radius * Math.cos(angle);
    if (index + 1 < WIDTH) out[index + 1] = radius * Math.sin(angle);
  }
  let norm = 0;
  for (const value of out) norm += value * value;
  norm = Math.sqrt(norm);
  for (let index = 0; index < WIDTH; index += 1) out[index] /= norm;
  return out;
}

let nullCache: Float32Array[] | null = null;
/** The NULL_DIRECTIONS fixed random directions, seeds 1 to NULL_DIRECTIONS. */
export function nullDirections() {
  nullCache ??= Array.from({ length: NULL_DIRECTIONS }, (_, index) => randomDirection(index + 1));
  return nullCache;
}

/** The raw residual with `strength x scale x direction` added. */
export function steeredResidual(raw: ArrayLike<number>, direction: ArrayLike<number>, strength: number, scale: number) {
  const out = new Float32Array(WIDTH);
  for (let index = 0; index < WIDTH; index += 1) out[index] = raw[index] + strength * scale * direction[index];
  return out;
}

/** The residual rows of positions 0 to `position`: all the next-character logits at `position` depend on. */
export function prefixRows(residual: Float32Array, position: number) {
  return residual.slice(0, (position + 1) * WIDTH);
}

/**
 * Next-character logits after the characters up to the last row of `prefix`, with that row's
 * residual replaced by `vector` (default: unchanged). Layer 2 is the last block, so the patch
 * touches nothing downstream but the final normalization and the output matrix.
 */
export function nextLogits(weights: TransformerWeights, prefix: Float32Array, vector?: ArrayLike<number>) {
  const position = prefix.length / WIDTH - 1;
  const stream = [prefix, prefix];
  return teachingTransformer.finalLogitsWithPatch(
    weights,
    stream,
    SAE_LAYER,
    position,
    vector ?? teachingTransformer.rowOf(prefix, position, WIDTH),
  );
}

export interface Distribution {
  probabilities: Float64Array;
  logProbabilities: Float64Array;
}

export function distributionOf(logits: ArrayLike<number>): Distribution {
  const probabilities = teachingTransformer.softmax(logits);
  const logProbabilities = Float64Array.from(probabilities, (value) => Math.log(Math.max(value, 1e-300)));
  return { probabilities, logProbabilities };
}

/** The most likely real characters (never <unk>), largest first. */
export function topTokens(distribution: Distribution, count: number) {
  return Array.from(distribution.probabilities, (probability, id) => ({ id, probability }))
    .filter((entry) => entry.id !== UNKNOWN)
    .sort((a, b) => b.probability - a.probability)
    .slice(0, count);
}

export interface Shift {
  id: number;
  /** Change in log probability, in nats. */
  delta: number;
}

/** The characters whose log probability rose and fell the most. */
export function largestShifts(before: Distribution, after: Distribution, count: number) {
  const shifts: Shift[] = Array.from(before.logProbabilities, (value, id) => ({
    id,
    delta: after.logProbabilities[id] - value,
  })).filter((entry) => entry.id !== UNKNOWN);
  const sorted = [...shifts].sort((a, b) => b.delta - a.delta);
  return { gains: sorted.slice(0, count), losses: sorted.slice(-count).reverse() };
}

export interface SweepRow {
  strength: number;
  /** Change in log probability of the target character when the feature's direction is added. */
  feature: number;
  nullMin: number;
  nullMedian: number;
  nullMax: number;
  nullMean: number;
  /** Sample standard deviation of the random directions' changes. */
  nullSd: number;
  /** How many random-direction standard deviations the feature's change is from their mean. */
  score: number;
}

const median = (values: number[]) => {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = sorted.length >> 1;
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
};

/**
 * Change in the target character's log probability at each strength, for the feature's direction
 * and for the same-length random directions. `raw` is the unsteered layer-2 residual at the
 * position, `prefix` the rows up to it.
 */
export function strengthSweep(
  weights: TransformerWeights,
  prefix: Float32Array,
  direction: ArrayLike<number>,
  target: number,
  scale: number,
  strengths: ReadonlyArray<number>,
  nulls: ReadonlyArray<ArrayLike<number>> = nullDirections(),
): SweepRow[] {
  const position = prefix.length / WIDTH - 1;
  const raw = teachingTransformer.rowOf(prefix, position, WIDTH);
  const baseline = distributionOf(nextLogits(weights, prefix)).logProbabilities[target];
  const change = (vector: ArrayLike<number>) =>
    distributionOf(nextLogits(weights, prefix, vector)).logProbabilities[target] - baseline;
  return strengths.map((strength) => {
    const spread = nulls.map((entry) => change(steeredResidual(raw, entry, strength, scale)));
    const effect = change(steeredResidual(raw, direction, strength, scale));
    const mean = spread.reduce((sum, value) => sum + value, 0) / spread.length;
    const sd = Math.sqrt(spread.reduce((sum, value) => sum + (value - mean) ** 2, 0) / Math.max(1, spread.length - 1));
    return {
      strength,
      feature: effect,
      nullMin: Math.min(...spread),
      nullMedian: median(spread),
      nullMax: Math.max(...spread),
      nullMean: mean,
      nullSd: sd,
      score: sd > 1e-12 ? (effect - mean) / sd : 0,
    };
  });
}

/** Layer-2 residual rows ([ids.length, WIDTH], flattened) for a list of character ids. */
export type ResidualProvider = (ids: ReadonlyArray<number>) => Promise<Float32Array> | Float32Array;

export interface Continuation {
  text: string;
  /** Logit gap between the chosen character and the runner-up at each step. */
  margins: number[];
}

/**
 * Greedy continuation: at each step take the most likely real character. With `steer`, the
 * direction is added at the last position every step (the only position that matters at the
 * last layer). `residuals` supplies the unsteered layer-2 residual; the lab passes ONNX
 * Runtime, the tests the TypeScript forward pass, and a test checks the margins are wide enough
 * that the difference between the two cannot change a choice.
 */
export async function greedyContinuation(options: {
  weights: TransformerWeights;
  prompt: string;
  steps: number;
  residuals: ResidualProvider;
  steer?: { direction: ArrayLike<number>; strength: number; scale: number };
}): Promise<Continuation> {
  const { weights, prompt, residuals, steer } = options;
  const ids = encodeText(prompt);
  const steps = Math.max(0, Math.min(options.steps, weights.context - ids.length));
  const margins: number[] = [];
  let generated = "";
  for (let step = 0; step < steps; step += 1) {
    const rows = await residuals(ids);
    const position = ids.length - 1;
    const prefix = rows.slice(0, (position + 1) * WIDTH);
    const raw = teachingTransformer.rowOf(prefix, position, WIDTH);
    const vector = steer ? steeredResidual(raw, steer.direction, steer.strength, steer.scale) : raw;
    const logits = nextLogits(weights, prefix, vector);
    let best = -1;
    let second = -1;
    for (let id = 0; id < logits.length; id += 1) {
      if (id === UNKNOWN) continue;
      if (best < 0 || logits[id] > logits[best]) {
        second = best;
        best = id;
      } else if (second < 0 || logits[id] > logits[second]) second = id;
    }
    margins.push(logits[best] - logits[second]);
    ids.push(best);
    generated += ITOS[best];
  }
  return { text: generated, margins };
}
