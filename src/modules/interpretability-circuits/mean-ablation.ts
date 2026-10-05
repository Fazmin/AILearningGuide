/**
 * Mean ablation for the shipped transformer, built on the SDK runner's own interventions.
 *
 * The runner can scale a head's output (scale 0 is zero-ablation) but cannot substitute another
 * value. Mean ablation needs exactly that: replace the head's output, before the output
 * projection, with its average over a reference set. Because the replacement is one constant
 * vector per head, it can be expressed with what the runner already offers: scale the head to
 * zero and add the constant's contribution, mean · W_O, to that block's output bias. The result is
 * the same forward pass the runner would compute if it accepted a replacement value.
 *
 * The reference set is a fixed, seeded batch of random-capital repeats with the same period and
 * length as the prompt being tested (different letters), averaged over the positions where
 * copying is scored. Nothing about the prompt under test leaks into its own average.
 */

import { runTransformer, type RunResult, type TransformerWeights } from "./engine";
import { encodeText } from "./vocabulary";

/** Number of random repeated strings averaged to form each head's reference output. */
export const REFERENCE_STRINGS = 8;

const CAPITALS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

/** A small seeded generator (mulberry32) so the reference batch is the same on every run. */
function seededRandom(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Random distinct capitals repeating every `period` letters, `length` letters long. The default
 * seeds give the reference batch for mean ablation; a different `seedOffset` gives an independent
 * batch (tests use one to check results on strings the averages never saw).
 */
export function referenceStrings(period: number, length: number, count = REFERENCE_STRINGS, seedOffset = 0) {
  return Array.from({ length: count }, (_, index) => {
    const random = seededRandom(7919 * (index + 1) + period + seedOffset);
    const letters = Array.from(CAPITALS);
    for (let at = letters.length - 1; at > 0; at -= 1) {
      const swap = Math.floor(random() * (at + 1));
      [letters[at], letters[swap]] = [letters[swap], letters[at]];
    }
    const base = letters.slice(0, period).join("");
    let text = "";
    while (text.length < length) text += base;
    return text.slice(0, length);
  });
}

function layerNorm(input: Float32Array, rows: number, width: number, weight: Float32Array, bias: Float32Array) {
  const out = new Float32Array(rows * width);
  for (let row = 0; row < rows; row += 1) {
    const offset = row * width;
    let mean = 0;
    for (let column = 0; column < width; column += 1) mean += input[offset + column];
    mean /= width;
    let variance = 0;
    for (let column = 0; column < width; column += 1) variance += (input[offset + column] - mean) ** 2;
    const scale = 1 / Math.sqrt(variance / width + 1e-5);
    for (let column = 0; column < width; column += 1) {
      out[offset + column] = (input[offset + column] - mean) * scale * weight[column] + bias[column];
    }
  }
  return out;
}

/**
 * Every head's output before the output projection (attention-weighted values), as
 * [layer][head] arrays of [sequence, headWidth]. Attention weights come from the runner;
 * values are recomputed from the block's input, which is the embeddings for block 1 and the
 * previous block's residual stream for block 2.
 */
export function headOutputs(weights: TransformerWeights, ids: ReadonlyArray<number>, run: RunResult) {
  const sequence = ids.length;
  const { width, headWidth, heads } = weights;
  return weights.blocks.map((block, layer) => {
    let input: Float32Array;
    if (layer === 0) {
      input = new Float32Array(sequence * width);
      ids.forEach((id, position) => {
        for (let column = 0; column < width; column += 1) {
          input[position * width + column] =
            weights.tokenEmbedding[id * width + column] + weights.positionEmbedding[position * width + column];
        }
      });
    } else {
      input = run.residuals[layer - 1];
    }
    const normed = layerNorm(input, sequence, width, block.attentionNormWeight, block.attentionNormBias);
    const values = new Float32Array(sequence * width);
    for (let position = 0; position < sequence; position += 1) {
      for (let column = 0; column < width; column += 1) {
        let total = block.qkvBias[2 * width + column];
        for (let inner = 0; inner < width; inner += 1) {
          total += normed[position * width + inner] * block.qkvWeight[inner * 3 * width + 2 * width + column];
        }
        values[position * width + column] = total;
      }
    }
    return Array.from({ length: heads }, (_, head) => {
      const out = new Float32Array(sequence * headWidth);
      for (let query = 0; query < sequence; query += 1) {
        for (let key = 0; key <= query; key += 1) {
          const weight = run.attention[layer][(head * sequence + query) * sequence + key];
          for (let dimension = 0; dimension < headWidth; dimension += 1) {
            out[query * headWidth + dimension] += weight * values[key * width + head * headWidth + dimension];
          }
        }
      }
      return out;
    });
  });
}

/** Each head's pre-projection output averaged over positions >= `fromPosition` of every reference text. */
export type HeadMeans = Float32Array[][];

export function meanHeadOutputs(
  weights: TransformerWeights,
  texts: ReadonlyArray<string>,
  fromPosition: number,
): HeadMeans {
  const { headWidth, heads } = weights;
  const sums = weights.blocks.map(() => Array.from({ length: heads }, () => new Float64Array(headWidth)));
  let count = 0;
  for (const text of texts) {
    const ids = encodeText(text);
    const outputs = headOutputs(weights, ids, runTransformer(weights, ids));
    outputs.forEach((layer, layerIndex) =>
      layer.forEach((head, headIndex) => {
        for (let position = fromPosition; position < ids.length; position += 1) {
          for (let dimension = 0; dimension < headWidth; dimension += 1) {
            sums[layerIndex][headIndex][dimension] += head[position * headWidth + dimension];
          }
        }
      }),
    );
    count += Math.max(0, ids.length - fromPosition);
  }
  return sums.map((layer) => layer.map((sum) => Float32Array.from(sum, (value) => value / Math.max(1, count))));
}

/**
 * Weights in which every head whose `headScale` entry is 0 contributes its mean output instead
 * of zero: the constant mean · W_O is added to that block's output bias. Run the result with the
 * same `headScale` so the live head output is dropped.
 */
export function meanAblatedWeights(
  weights: TransformerWeights,
  means: HeadMeans,
  headScale: ReadonlyArray<ReadonlyArray<number>>,
): TransformerWeights {
  const { width, headWidth } = weights;
  const blocks = weights.blocks.map((block, layer) => {
    const ablated = (headScale[layer] ?? []).map((scale, head) => (scale === 0 ? head : -1)).filter((head) => head >= 0);
    if (ablated.length === 0) return block;
    const outputBias = new Float32Array(block.outputBias);
    for (const head of ablated) {
      for (let column = 0; column < width; column += 1) {
        let shift = 0;
        for (let dimension = 0; dimension < headWidth; dimension += 1) {
          shift += means[layer][head][dimension] * block.outputWeight[(head * headWidth + dimension) * width + column];
        }
        outputBias[column] += shift;
      }
    }
    return { ...block, outputBias };
  });
  return { ...weights, blocks };
}
