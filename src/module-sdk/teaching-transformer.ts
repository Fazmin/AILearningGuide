/**
 * A plain TypeScript forward pass of the shipped two-layer character
 * transformer (models/train_language_models.py: pre-LayerNorm blocks, four
 * heads of width 64, exact-erf GELU MLP, learned positions, no bias on the
 * unembedding). The weights are read from tiny-transformer.onnx itself, so
 * this is the same model the attention lab runs through ONNX Runtime; running
 * it here is what allows heads to be switched off and activations patched.
 */

import { readOnnxGraph, type OnnxGraph } from "./teaching-onnx";

export interface BlockWeights {
  attentionNormWeight: Float32Array;
  attentionNormBias: Float32Array;
  /** [width, 3 * width], input-major: q | k | v columns. */
  qkvWeight: Float32Array;
  qkvBias: Float32Array;
  /** [width, width]. */
  outputWeight: Float32Array;
  outputBias: Float32Array;
  mlpNormWeight: Float32Array;
  mlpNormBias: Float32Array;
  /** [width, feedForward]. */
  upWeight: Float32Array;
  upBias: Float32Array;
  /** [feedForward, width]. */
  downWeight: Float32Array;
  downBias: Float32Array;
}

export interface TransformerWeights {
  vocabulary: number;
  width: number;
  heads: number;
  headWidth: number;
  context: number;
  feedForward: number;
  tokenEmbedding: Float32Array;
  positionEmbedding: Float32Array;
  blocks: BlockWeights[];
  finalNormWeight: Float32Array;
  finalNormBias: Float32Array;
  /** [width, vocabulary]. */
  unembedding: Float32Array;
}

const LAYER_NORM_EPSILON = 1e-5;

/** Map each MatMul node's weight initializer by the node name torch.onnx gave it. */
function matmulWeights(graph: OnnxGraph) {
  const byNode = new Map<string, Float32Array>();
  for (const node of graph.nodes) {
    if (node.opType !== "MatMul") continue;
    for (const input of node.inputs) {
      const tensor = graph.initializers.get(input);
      if (tensor) byNode.set(node.name, tensor.data);
    }
  }
  return byNode;
}

export function weightsFromOnnx(bytes: ArrayBuffer | Uint8Array): TransformerWeights {
  const graph = readOnnxGraph(bytes);
  const matmuls = matmulWeights(graph);
  const tensor = (name: string) => {
    const found = graph.initializers.get(name);
    if (!found) throw new Error(`tiny-transformer.onnx has no initializer ${name}.`);
    return found;
  };
  const matrix = (node: string) => {
    const found = matmuls.get(node);
    if (!found) throw new Error(`tiny-transformer.onnx has no MatMul node ${node}.`);
    return found;
  };

  const tokenEmbedding = tensor("token_embedding.weight");
  const positionEmbedding = tensor("position_embedding.weight");
  const [vocabulary, width] = tokenEmbedding.dims;
  const context = positionEmbedding.dims[0];
  const feedForward = tensor("blocks.0.mlp.0.bias").dims[0];
  const heads = 4;
  const blocks: BlockWeights[] = [];
  for (let layer = 0; graph.initializers.has(`blocks.${layer}.attention_norm.weight`); layer += 1) {
    const prefix = `blocks.${layer}`;
    blocks.push({
      attentionNormWeight: tensor(`${prefix}.attention_norm.weight`).data,
      attentionNormBias: tensor(`${prefix}.attention_norm.bias`).data,
      qkvWeight: matrix(`/${prefix}/attention/qkv/MatMul`),
      qkvBias: tensor(`${prefix}.attention.qkv.bias`).data,
      outputWeight: matrix(`/${prefix}/attention/output/MatMul`),
      outputBias: tensor(`${prefix}.attention.output.bias`).data,
      mlpNormWeight: tensor(`${prefix}.mlp_norm.weight`).data,
      mlpNormBias: tensor(`${prefix}.mlp_norm.bias`).data,
      upWeight: matrix(`/${prefix}/mlp/mlp.0/MatMul`),
      upBias: tensor(`${prefix}.mlp.0.bias`).data,
      downWeight: matrix(`/${prefix}/mlp/mlp.2/MatMul`),
      downBias: tensor(`${prefix}.mlp.2.bias`).data,
    });
  }
  const weights: TransformerWeights = {
    vocabulary,
    width,
    heads,
    headWidth: width / heads,
    context,
    feedForward,
    tokenEmbedding: tokenEmbedding.data,
    positionEmbedding: positionEmbedding.data,
    blocks,
    finalNormWeight: tensor("final_norm.weight").data,
    finalNormBias: tensor("final_norm.bias").data,
    unembedding: matrix("/lm_head/MatMul"),
  };
  if (weights.unembedding.length !== width * vocabulary) {
    throw new Error("Unexpected unembedding shape in tiny-transformer.onnx.");
  }
  return weights;
}

/** erf with maximum absolute error 1.2e-7 (Numerical Recipes erfc approximation). */
export function erf(x: number) {
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  const tail =
    t *
    Math.exp(
      -z * z -
        1.26551223 +
        t *
          (1.00002368 +
            t *
              (0.37409196 +
                t *
                  (0.09678418 +
                    t *
                      (-0.18628806 +
                        t *
                          (0.27886807 +
                            t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))),
    );
  return x >= 0 ? 1 - tail : tail - 1;
}

const gelu = (x: number) => 0.5 * x * (1 + erf(x / Math.SQRT2));

function layerNorm(input: Float32Array, rows: number, width: number, weight: Float32Array, bias: Float32Array) {
  const out = new Float32Array(rows * width);
  for (let row = 0; row < rows; row += 1) {
    const offset = row * width;
    let mean = 0;
    for (let column = 0; column < width; column += 1) mean += input[offset + column];
    mean /= width;
    let variance = 0;
    for (let column = 0; column < width; column += 1) {
      const gap = input[offset + column] - mean;
      variance += gap * gap;
    }
    const scale = 1 / Math.sqrt(variance / width + LAYER_NORM_EPSILON);
    for (let column = 0; column < width; column += 1) {
      out[offset + column] = (input[offset + column] - mean) * scale * weight[column] + bias[column];
    }
  }
  return out;
}

/** out[rows x n] = input[rows x k] . weight[k x n] (+ bias), accumulated in double precision. */
function matmul(
  input: Float32Array,
  rows: number,
  k: number,
  weight: Float32Array,
  n: number,
  bias?: Float32Array,
  rowList?: ReadonlyArray<number>,
) {
  const out = new Float32Array(rows * n);
  const accumulator = new Float64Array(n);
  const selected = rowList ?? Array.from({ length: rows }, (_, row) => row);
  for (const row of selected) {
    if (bias) for (let column = 0; column < n; column += 1) accumulator[column] = bias[column];
    else accumulator.fill(0);
    const inputOffset = row * k;
    for (let inner = 0; inner < k; inner += 1) {
      const value = input[inputOffset + inner];
      if (value === 0) continue;
      const weightOffset = inner * n;
      for (let column = 0; column < n; column += 1) accumulator[column] += value * weight[weightOffset + column];
    }
    const outOffset = row * n;
    for (let column = 0; column < n; column += 1) out[outOffset + column] = accumulator[column];
  }
  return out;
}

export interface Intervention {
  /** headScale[layer][head] multiplies that head's output before the output projection: 1 keeps it, 0 zero-ablates it. */
  headScale?: ReadonlyArray<ReadonlyArray<number>>;
  /** Overwrite the residual stream after block `layer` at `position` with `vector`. */
  patch?: { layer: number; position: number; vector: ArrayLike<number> };
}

export interface RunResult {
  sequence: number;
  /** [sequence, vocabulary] */
  logits: Float32Array;
  /** One [heads, sequence, sequence] array per layer (query-major). */
  attention: Float32Array[];
  /** Residual stream after each block, [sequence, width] each. */
  residuals: Float32Array[];
}

interface BlockOptions {
  headScale?: ReadonlyArray<number>;
  /** Only the last position's output is needed (keys and values are still computed everywhere). */
  lastOnly?: boolean;
}

function runBlock(
  weights: TransformerWeights,
  block: BlockWeights,
  hidden: Float32Array,
  sequence: number,
  options: BlockOptions,
) {
  const { width, heads, headWidth, feedForward } = weights;
  const rows = options.lastOnly ? [sequence - 1] : undefined;
  const normed = layerNorm(hidden, sequence, width, block.attentionNormWeight, block.attentionNormBias);
  const qkv = matmul(normed, sequence, width, block.qkvWeight, 3 * width, block.qkvBias);
  const attention = new Float32Array(heads * sequence * sequence);
  const mixed = new Float32Array(sequence * width);
  const scale = 1 / Math.sqrt(headWidth);
  const queries = rows ?? Array.from({ length: sequence }, (_, index) => index);
  const scores = new Float64Array(sequence);

  for (let head = 0; head < heads; head += 1) {
    const headScale = options.headScale?.[head] ?? 1;
    const qOffset = head * headWidth;
    const kOffset = width + head * headWidth;
    const vOffset = 2 * width + head * headWidth;
    for (const query of queries) {
      let peak = -Infinity;
      for (let key = 0; key <= query; key += 1) {
        let dot = 0;
        for (let dimension = 0; dimension < headWidth; dimension += 1) {
          dot += qkv[query * 3 * width + qOffset + dimension] * qkv[key * 3 * width + kOffset + dimension];
        }
        scores[key] = dot * scale;
        if (scores[key] > peak) peak = scores[key];
      }
      let total = 0;
      for (let key = 0; key <= query; key += 1) {
        scores[key] = Math.exp(scores[key] - peak);
        total += scores[key];
      }
      const rowOffset = (head * sequence + query) * sequence;
      for (let key = 0; key <= query; key += 1) {
        const weight = scores[key] / total;
        attention[rowOffset + key] = weight;
        if (headScale === 0) continue;
        for (let dimension = 0; dimension < headWidth; dimension += 1) {
          mixed[query * width + qOffset + dimension] +=
            headScale * weight * qkv[key * 3 * width + vOffset + dimension];
        }
      }
    }
  }

  const update = matmul(mixed, sequence, width, block.outputWeight, width, block.outputBias, rows);
  const afterAttention = new Float32Array(hidden);
  for (const row of queries) {
    for (let column = 0; column < width; column += 1) {
      afterAttention[row * width + column] += update[row * width + column];
    }
  }
  const mlpInput = layerNorm(afterAttention, sequence, width, block.mlpNormWeight, block.mlpNormBias);
  const up = matmul(mlpInput, sequence, width, block.upWeight, feedForward, block.upBias, rows);
  for (const row of queries) {
    for (let column = 0; column < feedForward; column += 1) {
      up[row * feedForward + column] = gelu(up[row * feedForward + column]);
    }
  }
  const down = matmul(up, sequence, feedForward, block.downWeight, width, block.downBias, rows);
  const out = afterAttention;
  for (const row of queries) {
    for (let column = 0; column < width; column += 1) {
      out[row * width + column] += down[row * width + column];
    }
  }
  return { hidden: out, attention };
}

function unembed(weights: TransformerWeights, hidden: Float32Array, sequence: number, rows?: ReadonlyArray<number>) {
  const normed = layerNorm(hidden, sequence, weights.width, weights.finalNormWeight, weights.finalNormBias);
  return matmul(normed, sequence, weights.width, weights.unembedding, weights.vocabulary, undefined, rows);
}

/** Full forward pass with optional head scaling and one residual patch. */
export function runTransformer(
  weights: TransformerWeights,
  inputIds: ReadonlyArray<number>,
  intervention: Intervention = {},
): RunResult {
  const sequence = inputIds.length;
  if (sequence === 0 || sequence > weights.context) {
    throw new Error(`The model takes 1 to ${weights.context} characters.`);
  }
  const { width } = weights;
  let hidden = new Float32Array(sequence * width);
  inputIds.forEach((id, position) => {
    for (let column = 0; column < width; column += 1) {
      hidden[position * width + column] =
        weights.tokenEmbedding[id * width + column] + weights.positionEmbedding[position * width + column];
    }
  });
  const attention: Float32Array[] = [];
  const residuals: Float32Array[] = [];
  weights.blocks.forEach((block, layer) => {
    const result = runBlock(weights, block, hidden, sequence, { headScale: intervention.headScale?.[layer] });
    hidden = result.hidden;
    const patch = intervention.patch;
    if (patch && patch.layer === layer) {
      for (let column = 0; column < width; column += 1) {
        hidden[patch.position * width + column] = patch.vector[column];
      }
    }
    attention.push(result.attention);
    residuals.push(new Float32Array(hidden));
  });
  return { sequence, logits: unembed(weights, hidden, sequence), attention, residuals };
}

/**
 * Final-position logits after overwriting the residual after block `layer` at
 * `position`, resuming from a stored residual stream. Later blocks only
 * compute the last position's output, which is all the metric reads.
 */
export function finalLogitsWithPatch(
  weights: TransformerWeights,
  baseResiduals: ReadonlyArray<Float32Array>,
  layer: number,
  position: number,
  vector: ArrayLike<number>,
) {
  const { width, vocabulary } = weights;
  const sequence = baseResiduals[layer].length / width;
  let hidden = new Float32Array(baseResiduals[layer]);
  for (let column = 0; column < width; column += 1) hidden[position * width + column] = vector[column];
  for (let next = layer + 1; next < weights.blocks.length; next += 1) {
    hidden = runBlock(weights, weights.blocks[next], hidden, sequence, { lastOnly: true }).hidden;
  }
  const last = sequence - 1;
  const logits = unembed(weights, hidden, sequence, [last]);
  return logits.slice(last * vocabulary, (last + 1) * vocabulary);
}

export function rowOf(values: Float32Array, row: number, width: number) {
  return values.subarray(row * width, (row + 1) * width);
}

export function softmax(logits: ArrayLike<number>) {
  let peak = -Infinity;
  for (let index = 0; index < logits.length; index += 1) peak = Math.max(peak, logits[index]);
  const out = new Float64Array(logits.length);
  let total = 0;
  for (let index = 0; index < logits.length; index += 1) {
    out[index] = Math.exp(logits[index] - peak);
    total += out[index];
  }
  for (let index = 0; index < logits.length; index += 1) out[index] /= total;
  return out;
}
