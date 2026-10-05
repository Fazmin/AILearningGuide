/**
 * The shipped sparse autoencoder (models/train_sparse_autoencoder.py).
 *
 * Input: the transformer's residual stream after its second (last) block,
 * centred by a per-dimension corpus mean and divided by one global RMS scale.
 * Encoder: ReLU(W_enc x + b_enc), then keep only the 32 largest of 1,024
 * values (TopK). Decoder: x-hat = W_dec f, no bias, with every feature's
 * decoder direction renormalized to unit length after each training step.
 * Loss: MSE(x-hat, x) + 0.002 x mean |f|, so sparsity comes mainly from TopK.
 *
 * Window position: the transformer reads at most 64 characters, so the
 * character at position p of a window has seen only p characters before it.
 * Activations at positions 0 to 3 are unlike running text (see EARLY_POSITIONS).
 */

import { readOnnxGraph } from "./onnx-weights";

export const FEATURES = 1024;
export const WIDTH = 256;
export const ACTIVE = 32;
export const L1_COEFFICIENT = 2e-3;
/** Window positions 0 to 3: the cache flags features whose top examples come from here. */
export const EARLY_POSITIONS = 4;

export interface SaeWeights {
  /** [FEATURES x WIDTH]: row i is feature i's encoder direction. */
  encoderWeight: Float32Array;
  encoderBias: Float32Array;
  /** [FEATURES x WIDTH]: row i is feature i's (unit) decoder direction. */
  decoder: Float32Array;
}

export function saeFromOnnx(bytes: ArrayBuffer | Uint8Array): SaeWeights {
  const graph = readOnnxGraph(bytes);
  const encoderWeight = graph.initializers.get("encoder.weight");
  const encoderBias = graph.initializers.get("encoder.bias");
  const decoderNode = graph.nodes.find((node) => node.name === "/decoder/MatMul");
  const decoder = decoderNode?.inputs.map((name) => graph.initializers.get(name)).find(Boolean);
  if (!encoderWeight || !encoderBias || !decoder) throw new Error("residual-sae.onnx is missing a weight.");
  if (encoderWeight.data.length !== FEATURES * WIDTH || decoder.data.length !== FEATURES * WIDTH) {
    throw new Error("residual-sae.onnx has an unexpected shape.");
  }
  return { encoderWeight: encoderWeight.data, encoderBias: encoderBias.data, decoder: decoder.data };
}

/** (residual - mean) / scale, as the SAE saw its training activations. */
export function normalizeResidual(residual: ArrayLike<number>, mean: ArrayLike<number>, scale: number) {
  return Float32Array.from({ length: residual.length }, (_, index) => (residual[index] - mean[index]) / scale);
}

export interface ActiveFeature {
  feature: number;
  value: number;
}

/** Nonzero entries of one token's feature vector, largest first. */
export function activeFeatures(features: ArrayLike<number>): ActiveFeature[] {
  const out: ActiveFeature[] = [];
  for (let feature = 0; feature < features.length; feature += 1) {
    if (features[feature] > 0) out.push({ feature, value: features[feature] });
  }
  return out.sort((a, b) => b.value - a.value);
}

/** The encoder in plain TypeScript: ReLU, then keep the ACTIVE largest values. */
export function encode(sae: SaeWeights, x: ArrayLike<number>) {
  const dense = new Float32Array(FEATURES);
  for (let feature = 0; feature < FEATURES; feature += 1) {
    let total = sae.encoderBias[feature];
    const offset = feature * WIDTH;
    for (let dimension = 0; dimension < WIDTH; dimension += 1) total += sae.encoderWeight[offset + dimension] * x[dimension];
    dense[feature] = Math.max(0, total);
  }
  const keep = Array.from(dense, (value, feature) => ({ value, feature }))
    .sort((a, b) => b.value - a.value)
    .slice(0, ACTIVE);
  const out = new Float32Array(FEATURES);
  for (const { feature, value } of keep) out[feature] = value;
  return out;
}

/** x-hat from a list of (feature, value) pairs: the sum of value x decoder direction. */
export function reconstruct(sae: SaeWeights, active: ReadonlyArray<ActiveFeature>) {
  const out = new Float32Array(WIDTH);
  for (const { feature, value } of active) {
    const offset = feature * WIDTH;
    for (let dimension = 0; dimension < WIDTH; dimension += 1) out[dimension] += value * sae.decoder[offset + dimension];
  }
  return out;
}

/** Fraction of variance unexplained: ||x - x-hat||^2 / ||x||^2 (x is already centred). */
export function unexplainedFraction(x: ArrayLike<number>, reconstruction: ArrayLike<number>) {
  let error = 0;
  let total = 0;
  for (let index = 0; index < x.length; index += 1) {
    error += (x[index] - reconstruction[index]) ** 2;
    total += x[index] ** 2;
  }
  return total > 0 ? error / total : 0;
}

export function cosine(a: ArrayLike<number>, b: ArrayLike<number>) {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let index = 0; index < a.length; index += 1) {
    dot += a[index] * b[index];
    na += a[index] ** 2;
    nb += b[index] ** 2;
  }
  return na > 0 && nb > 0 ? dot / Math.sqrt(na * nb) : 0;
}

/** Unexplained fraction when only the k largest active features are kept, for k = 1 … active.length. */
export function topKCurve(sae: SaeWeights, x: ArrayLike<number>, active: ReadonlyArray<ActiveFeature>) {
  const running = new Float32Array(WIDTH);
  return active.map(({ feature, value }) => {
    const offset = feature * WIDTH;
    for (let dimension = 0; dimension < WIDTH; dimension += 1) running[dimension] += value * sae.decoder[offset + dimension];
    return unexplainedFraction(x, running);
  });
}

function decoderDot(sae: SaeWeights, a: number, b: number) {
  let dot = 0;
  const oa = a * WIDTH;
  const ob = b * WIDTH;
  for (let dimension = 0; dimension < WIDTH; dimension += 1) dot += sae.decoder[oa + dimension] * sae.decoder[ob + dimension];
  return dot;
}

export interface OverlapStats {
  pairs: number;
  meanAbs: number;
  maxAbs: number;
  /** Share of pairs with |cos| above 0.3. */
  above: number;
  /** Number of pairs with |cos| above 0.9: a direction that two features share almost exactly. */
  nearIdentical: number;
  /** The pair with the largest |cos| (first one found on a tie), with the sign of its cosine. */
  strongest: { a: number; b: number; cosine: number } | null;
}

/** |cosine| between decoder directions over every pair in `features` (all 1,024 when omitted). */
export function decoderOverlap(sae: SaeWeights, features?: ReadonlyArray<number>): OverlapStats {
  const list = features ?? Array.from({ length: FEATURES }, (_, index) => index);
  let pairs = 0;
  let total = 0;
  let maxAbs = 0;
  let above = 0;
  let nearIdentical = 0;
  let strongest: OverlapStats["strongest"] = null;
  for (let i = 0; i < list.length; i += 1) {
    for (let j = i + 1; j < list.length; j += 1) {
      const signed = decoderDot(sae, list[i], list[j]);
      const value = Math.abs(signed);
      pairs += 1;
      total += value;
      if (value > maxAbs) {
        maxAbs = value;
        strongest = { a: list[i], b: list[j], cosine: signed };
      }
      if (value > 0.3) above += 1;
      if (value > 0.9) nearIdentical += 1;
    }
  }
  return { pairs, meanAbs: pairs ? total / pairs : 0, maxAbs, above: pairs ? above / pairs : 0, nearIdentical, strongest };
}

/** Cosine between two features' decoder directions (unit rows, so a dot product). */
export function decoderCosine(sae: SaeWeights, a: number, b: number) {
  return decoderDot(sae, a, b);
}

export type Hypothesis = "char" | "upper" | "space" | "newline" | "start";

export const HYPOTHESES: ReadonlyArray<{ value: Hypothesis; label: string }> = [
  { value: "char", label: "Same character" },
  { value: "upper", label: "Capital letter" },
  { value: "space", label: "Space" },
  { value: "newline", label: "Line break" },
  { value: "start", label: "First 4 characters" },
];

/**
 * Does character `character` at `position` satisfy the hypothesis? `position` is the position
 * in the model's window (a probe text is its own window, so its index; a cached example's
 * `window_position`), never the offset of the character inside a displayed excerpt.
 */
export function matchesHypothesis(hypothesis: Hypothesis, character: string, position: number, focus: string) {
  if (hypothesis === "char") return character === focus;
  if (hypothesis === "upper") return /^[A-Z]$/.test(character);
  if (hypothesis === "space") return character === " ";
  if (hypothesis === "newline") return character === "\n";
  return position < EARLY_POSITIONS;
}

export interface LabelTest {
  truePositive: number;
  falsePositive: number;
  falseNegative: number;
  precision: number | null;
  recall: number | null;
}

/** Score a label as a classifier: fires (activation > threshold) against matches the hypothesis. */
export function testLabel(
  text: string,
  activations: ReadonlyArray<number>,
  threshold: number,
  hypothesis: Hypothesis,
  focus: string,
): LabelTest {
  let truePositive = 0;
  let falsePositive = 0;
  let falseNegative = 0;
  Array.from(text).forEach((character, position) => {
    const fires = (activations[position] ?? 0) > threshold;
    const matches = matchesHypothesis(hypothesis, character, position, focus);
    if (fires && matches) truePositive += 1;
    else if (fires) falsePositive += 1;
    else if (matches) falseNegative += 1;
  });
  return {
    truePositive,
    falsePositive,
    falseNegative,
    precision: truePositive + falsePositive > 0 ? truePositive / (truePositive + falsePositive) : null,
    recall: truePositive + falseNegative > 0 ? truePositive / (truePositive + falseNegative) : null,
  };
}

/** On how many of the rows (one per character) both features, and at least one of them, are active. */
export function pairActivity(rowsOfFeatures: ReadonlyArray<ArrayLike<number>>, a: number, b: number) {
  let both = 0;
  let either = 0;
  for (const row of rowsOfFeatures) {
    const first = row[a] > 0;
    const second = row[b] > 0;
    if (first && second) both += 1;
    if (first || second) either += 1;
  }
  return { both, either, total: rowsOfFeatures.length };
}

export interface CachedExample {
  activation: number;
  /** About 32 corpus characters either side of the focus, with line breaks shown as spaces. */
  text: string;
  /** Index of the focus character inside `text` (the excerpt), not its position in the model's window. */
  focus_offset: number;
  /** Position of the focus character in the 64-character window the transformer read: 0 is the first character. */
  window_position: number;
}

export interface CachedRecord {
  feature: number;
  maximum: number;
  maximum_after_position_zero: number;
  /** Tokens, of the scanned windows, on which the feature is active. */
  fires: number;
  fires_at_position_zero: number;
  position_zero_share: number;
  position_zero_only: boolean;
  /** More than 20 times its chance share (1/64) of its activations are at position 0. */
  position_zero_dominated: boolean;
  examples_at_position_zero: number;
  /** At least half of the top examples are the first character of a window. */
  position_zero_artefact: boolean;
  examples_at_early_positions: number;
  /** At least half of the top examples are within the first EARLY_POSITIONS positions. */
  window_start_artefact: boolean;
  examples: CachedExample[];
  /** The top examples from window position 4 on; present when any top example is earlier. */
  examples_in_context?: CachedExample[];
}

/** Short reasons a feature's top examples may describe the start of a window rather than language. */
export function featureFlags(record: CachedRecord): string[] {
  const flags: string[] = [];
  if (record.position_zero_artefact) flags.push("position-0 artefact");
  else if (record.window_start_artefact) flags.push("window-start artefact");
  if (record.position_zero_dominated) flags.push("mostly fires at position 0");
  if (record.examples.length > 1 && record.examples.every((example) => example.activation === record.examples[0].activation)) {
    flags.push("same activation in every example");
  }
  return flags;
}

/** True when a cached feature's top examples are not a safe basis for a label. */
export const isStartArtefact = (record: CachedRecord) => record.window_start_artefact || record.position_zero_artefact;

/**
 * Which examples to show: the cache's top eight by activation, or (when the feature has
 * early-window examples) the top eight from window position 4 on.
 */
export function examplesFor(record: CachedRecord, view: "top" | "context"): CachedExample[] {
  return view === "context" && record.examples_in_context ? record.examples_in_context : record.examples;
}

/**
 * How many of an excerpt's characters before the focus the model actually read. An excerpt
 * carries about 32 corpus characters of left context, but the transformer's window began
 * `window_position` characters before the focus.
 */
export function seenBefore(example: CachedExample) {
  return Math.min(example.window_position, example.focus_offset);
}

/** The most common focus character among a feature's cached top examples, with its count. */
export function dominantFocus(examples: ReadonlyArray<CachedExample>) {
  const counts = new Map<string, number>();
  for (const example of examples) {
    const character = example.text[example.focus_offset] ?? "";
    counts.set(character, (counts.get(character) ?? 0) + 1);
  }
  let best = "";
  let count = 0;
  for (const [character, n] of counts) {
    if (n > count) {
      best = character;
      count = n;
    }
  }
  return { character: best, count, distinct: counts.size };
}
