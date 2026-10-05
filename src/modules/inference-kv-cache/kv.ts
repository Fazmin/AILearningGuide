/**
 * Arithmetic for the inference-kv-cache lab.
 *
 * Two scales on purpose:
 * - a toy causal sequence (a few dozen positions) whose attention triangle can be drawn cell by cell;
 * - a named reference model and accelerator, where memory and time are computed from published
 *   shapes and spec-sheet peaks. The time figures are a roofline model, not a measurement.
 */

export interface ModelShape {
  name: string;
  layers: number;
  dModel: number;
  queryHeads: number;
  kvHeads: number;
  headDim: number;
  vocab: number;
  parameters: number;
  bytesPerValue: number;
}

/** Meta Llama 3 8B, bf16: 32 layers, 32 query heads, 8 key/value heads (GQA), head dim 128. */
export const LLAMA3_8B: ModelShape = {
  name: "Llama 3 8B",
  layers: 32,
  dModel: 4096,
  queryHeads: 32,
  kvHeads: 8,
  headDim: 128,
  vocab: 128256,
  parameters: 8.03e9,
  bytesPerValue: 2,
};

/** NVIDIA H100 SXM spec-sheet peaks: dense BF16 tensor throughput and HBM3 bandwidth. */
export const H100_SXM = {
  name: "H100 SXM",
  peakFlops: 989e12,
  bandwidth: 3.35e12,
  memoryBytes: 80e9,
};

export type Hardware = typeof H100_SXM;

/** Bytes of keys plus values stored per token: 2 × layers × kv_heads × head_dim × bytes. */
export function kvBytesPerToken(model: ModelShape, kvHeads = model.kvHeads) {
  return 2 * model.layers * kvHeads * model.headDim * model.bytesPerValue;
}

export function weightBytes(model: ModelShape) {
  return model.parameters * model.bytesPerValue;
}

/**
 * Parameters that take part in a matrix multiply for every token. The input embedding is a
 * table lookup, so it is excluded; the untied output head is included.
 */
export function matmulParameters(model: ModelShape) {
  return model.parameters - model.vocab * model.dModel;
}

/** Attention FLOPs for one query over `context` keys, summed over layers: QKᵀ plus weights × V. */
export function attentionFlopsPerQuery(model: ModelShape, context: number) {
  return 4 * model.layers * model.queryHeads * model.headDim * context;
}

export interface PassCost {
  flops: number;
  bytes: number;
  computeSeconds: number;
  memorySeconds: number;
  seconds: number;
  bound: "compute" | "memory";
  intensity: number;
}

function roofline(flops: number, bytes: number, hardware: Hardware): PassCost {
  const computeSeconds = flops / hardware.peakFlops;
  const memorySeconds = bytes / hardware.bandwidth;
  return {
    flops,
    bytes,
    computeSeconds,
    memorySeconds,
    seconds: Math.max(computeSeconds, memorySeconds),
    bound: computeSeconds >= memorySeconds ? "compute" : "memory",
    intensity: flops / bytes,
  };
}

/** Prefill of one prompt: every position in one pass, causal attention over earlier positions. */
export function prefillCost(model: ModelShape, promptTokens: number, hardware: Hardware = H100_SXM) {
  const matmul = 2 * matmulParameters(model) * promptTokens;
  // Σ_{i=1..T} 4·L·H·d·i = 2·L·H·d·T(T+1)
  const attention = 2 * model.layers * model.queryHeads * model.headDim * promptTokens * (promptTokens + 1);
  const bytes = matmulParameters(model) * model.bytesPerValue + promptTokens * kvBytesPerToken(model);
  return roofline(matmul + attention, bytes, hardware);
}

/**
 * One decode step for `batch` sequences that each already hold `context` tokens.
 * With the cache: one new position per sequence attends over its stored keys and values, so the
 * step reads every weight once plus every sequence's cache.
 * Without the cache: every sequence reruns the whole prefix, which is a prefill per step.
 */
export function decodeStepCost(
  model: ModelShape,
  context: number,
  batch: number,
  useCache: boolean,
  hardware: Hardware = H100_SXM,
) {
  const weights = matmulParameters(model) * model.bytesPerValue;
  if (useCache) {
    const flops = batch * (2 * matmulParameters(model) + attentionFlopsPerQuery(model, context));
    const bytes = weights + batch * context * kvBytesPerToken(model);
    return roofline(flops, bytes, hardware);
  }
  const flops =
    batch *
    (2 * matmulParameters(model) * context +
      2 * model.layers * model.queryHeads * model.headDim * context * (context + 1));
  return roofline(flops, weights, hardware);
}

/** Arithmetic intensity where a kernel stops being bandwidth-bound: peak FLOP/s ÷ bytes/s. */
export function ridgePoint(hardware: Hardware = H100_SXM) {
  return hardware.peakFlops / hardware.bandwidth;
}

export function memoryBudget(
  model: ModelShape,
  context: number,
  batch: number,
  hardware: Hardware = H100_SXM,
) {
  const weights = weightBytes(model);
  const cache = batch * context * kvBytesPerToken(model);
  const free = hardware.memoryBytes - weights - cache;
  const maxSequences = Math.max(
    0,
    Math.floor((hardware.memoryBytes - weights) / (context * kvBytesPerToken(model))),
  );
  return { weights, cache, free, fits: free >= 0, maxSequences };
}

/* ---------- toy causal sequence ---------- */

/**
 * The pass shown for a sequence of `prompt` prompt tokens and `generated` tokens already produced.
 * generated = 0 is the prefill pass, which also produces the first new token.
 * generated = g ≥ 1 is decode step g: it feeds token +g at position prompt + g.
 */
export function toyPass(prompt: number, generated: number, useCache: boolean) {
  const length = prompt + generated;
  const isPrefill = generated === 0;
  const triangle = (n: number) => (n * (n + 1)) / 2;
  let runRecompute = triangle(prompt);
  for (let step = 1; step <= generated; step += 1) runRecompute += triangle(prompt + step);
  const fullPass = isPrefill || !useCache;
  return {
    length,
    isPrefill,
    /** Positions whose query, key and value are projected in this pass. */
    projectedPositions: fullPass ? length : 1,
    /** Query-key dot products in this pass, per head per layer. */
    dotProducts: fullPass ? triangle(length) : length,
    /** Dot products this pass that repeat work an earlier pass already did. */
    repeatedDotProducts: !isPrefill && !useCache ? triangle(length - 1) : 0,
    readFromCache: useCache && !isPrefill ? length - 1 : 0,
    cachedAfter: useCache ? length : 0,
    /** Dot products summed over prefill and every decode step so far. */
    runTotalCached: triangle(length),
    runTotalRecompute: runRecompute,
  };
}

/* ---------- formatting ---------- */

export function formatBytes(bytes: number) {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(bytes >= 1e11 ? 0 : 1)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(bytes >= 1e8 ? 0 : 1)} MB`;
  if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(0)} kB`;
  return `${bytes} B`;
}

export function formatSeconds(seconds: number) {
  if (seconds >= 1) return `${seconds.toFixed(2)} s`;
  if (seconds >= 0.01) return `${(seconds * 1e3).toFixed(0)} ms`;
  if (seconds >= 0.001) return `${(seconds * 1e3).toFixed(1)} ms`;
  return `${(seconds * 1e3).toFixed(2)} ms`;
}

export function formatFlopCount(flops: number) {
  if (flops >= 1e15) return `${(flops / 1e15).toFixed(1)} PFLOP`;
  if (flops >= 1e12) return `${(flops / 1e12).toFixed(1)} TFLOP`;
  return `${(flops / 1e9).toFixed(1)} GFLOP`;
}
