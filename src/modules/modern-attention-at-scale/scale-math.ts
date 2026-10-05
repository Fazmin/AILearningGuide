/**
 * Closed-form memory and FLOP counts for one stated configuration, plus two
 * small exact computations: tiled online softmax and rotary position
 * embedding. Every number the lab prints comes from here.
 */

/** Llama 3 8B's shape: the configuration every count in the lab uses. */
export const CONFIG = {
  name: "Llama 3 8B shape",
  layers: 32,
  queryHeads: 32,
  headDim: 128,
  modelWidth: 4096,
  mlpHidden: 14336,
  cacheBytes: 2,
  window: 4096,
} as const;

export const KV_HEAD_OPTIONS = [32, 16, 8, 4, 2, 1] as const;
export const GIB = 1024 ** 3;

export function kvLabel(kvHeads: number) {
  if (kvHeads === CONFIG.queryHeads) return "MHA";
  if (kvHeads === 1) return "MQA";
  return "GQA";
}

/** Tokens whose keys and values must be kept: all of them, or the last `window`. */
export function cachedTokens(tokens: number, window?: number) {
  return window ? Math.min(tokens, window) : tokens;
}

/** Bytes of K and V cached per token across all layers. */
export function kvBytesPerToken(kvHeads: number) {
  return 2 * CONFIG.layers * kvHeads * CONFIG.headDim * CONFIG.cacheBytes;
}

export function kvCacheBytes(kvHeads: number, tokens: number, window?: number) {
  return kvBytesPerToken(kvHeads) * cachedTokens(tokens, window);
}

/** Query–key pairs a causal layer scores for a prompt of n tokens, optionally windowed. */
export function causalPairs(tokens: number, window?: number) {
  if (!window || tokens <= window) return (tokens * (tokens + 1)) / 2;
  return (window * (window + 1)) / 2 + (tokens - window) * window;
}

/** Weights in one layer: Q and O are d×d, K and V are d×(kvHeads·headDim), SwiGLU MLP has three matrices. */
export function layerWeights(kvHeads: number) {
  const d = CONFIG.modelWidth;
  const kv = kvHeads * CONFIG.headDim;
  return 2 * d * d + 2 * d * kv + 3 * d * CONFIG.mlpHidden;
}

/**
 * Prefill FLOPs for one layer over n tokens. Weight matmuls cost 2 FLOPs per
 * weight per token. Attention costs 2·headDim for the score and 2·headDim for
 * the value mix, per query head, per query–key pair.
 */
export function prefillFlops(tokens: number, kvHeads: number, window?: number) {
  const weights = 2 * tokens * layerWeights(kvHeads);
  const attention = 4 * CONFIG.queryHeads * CONFIG.headDim * causalPairs(tokens, window);
  return { weights, attention };
}

/** Sequence length where causal attention FLOPs equal the weight-matmul FLOPs. */
export function crossoverTokens(kvHeads: number) {
  // 2·n·W = 4·H·dh·n(n+1)/2  →  n + 1 = W / (H·dh)
  return layerWeights(kvHeads) / (CONFIG.queryHeads * CONFIG.headDim) - 1;
}

/** Bytes to store one layer's full score matrix at 16-bit precision. */
export function scoreMatrixBytes(tokens: number) {
  return CONFIG.queryHeads * tokens * tokens * 2;
}

/** Bytes FlashAttention keeps per layer instead: one float32 log-sum-exp per row per head. */
export function rowStatisticBytes(tokens: number) {
  return CONFIG.queryHeads * tokens * 4;
}

/* Online softmax */

export interface OnlineStep {
  tile: number;
  from: number;
  to: number;
  tileMax: number;
  runningMax: number;
  rescale: number;
  runningSum: number;
  accumulator: number[];
}

/**
 * Process scores and value vectors in tiles, keeping only a running max m, a
 * running sum ℓ of exp(s − m), and an unnormalised output accumulator. When a
 * tile raises the max, the old ℓ and accumulator are rescaled by exp(m_old − m_new).
 */
export function onlineSoftmax(
  scores: ReadonlyArray<number>,
  values: ReadonlyArray<ReadonlyArray<number>>,
  tileSize: number,
) {
  const width = values[0]?.length ?? 0;
  let runningMax = Number.NEGATIVE_INFINITY;
  let runningSum = 0;
  let accumulator = Array.from({ length: width }, () => 0);
  const steps: OnlineStep[] = [];

  for (let from = 0, tile = 0; from < scores.length; from += tileSize, tile += 1) {
    const to = Math.min(scores.length, from + tileSize);
    const block = scores.slice(from, to);
    const tileMax = Math.max(...block);
    const nextMax = Math.max(runningMax, tileMax);
    const rescale = Number.isFinite(runningMax) ? Math.exp(runningMax - nextMax) : 0;
    runningSum *= rescale;
    accumulator = accumulator.map((value) => value * rescale);
    block.forEach((score, offset) => {
      const weight = Math.exp(score - nextMax);
      runningSum += weight;
      accumulator = accumulator.map((value, index) => value + weight * values[from + offset][index]);
    });
    runningMax = nextMax;
    steps.push({ tile, from, to, tileMax, runningMax, rescale, runningSum, accumulator: [...accumulator] });
  }

  const output = accumulator.map((value) => value / runningSum);
  return { steps, output, runningMax, runningSum };
}

export function fullSoftmaxAttention(
  scores: ReadonlyArray<number>,
  values: ReadonlyArray<ReadonlyArray<number>>,
) {
  const peak = Math.max(...scores);
  const exps = scores.map((score) => Math.exp(score - peak));
  const total = exps.reduce((sum, value) => sum + value, 0);
  const weights = exps.map((value) => value / total);
  const width = values[0]?.length ?? 0;
  const output = Array.from({ length: width }, (_, index) =>
    weights.reduce((sum, weight, key) => sum + weight * values[key][index], 0),
  );
  return { weights, output };
}

/* Rotary position embedding */

/** θ_i = base^(−2i/d) for pair i of a d-dimensional head. */
export function ropeFrequency(pair: number, headDim: number, base: number) {
  return base ** ((-2 * pair) / headDim);
}

/** Rotate each pair (2i, 2i+1) of a vector by position · θ_i. */
export function ropeRotate(vector: ReadonlyArray<number>, position: number, base: number) {
  const headDim = vector.length;
  const out = [...vector];
  for (let pair = 0; pair < headDim / 2; pair += 1) {
    const angle = position * ropeFrequency(pair, headDim, base);
    const x = vector[2 * pair];
    const y = vector[2 * pair + 1];
    out[2 * pair] = x * Math.cos(angle) - y * Math.sin(angle);
    out[2 * pair + 1] = x * Math.sin(angle) + y * Math.cos(angle);
  }
  return out;
}

export function dot(left: ReadonlyArray<number>, right: ReadonlyArray<number>) {
  return left.reduce((sum, value, index) => sum + value * (right[index] ?? 0), 0);
}

/** Score between a query at position m and a key at position n after RoPE. */
export function ropeScore(
  query: ReadonlyArray<number>,
  key: ReadonlyArray<number>,
  queryPosition: number,
  keyPosition: number,
  base: number,
) {
  return dot(ropeRotate(query, queryPosition, base), ropeRotate(key, keyPosition, base));
}
