/**
 * First-order serving arithmetic: a roofline-style account of memory and
 * decode speed under stated assumptions. Nothing here is measured.
 */

export const GIB = 1024 ** 3;

export const HARDWARE = {
  name: "one 24 GiB accelerator",
  vramBytes: 24 * GIB,
  bandwidthBytesPerSecond: 400e9,
  computeFlops: 120e12,
  overheadBytes: 1 * GIB,
};

/** Shapes follow the published configs these presets are named after. */
export const MODELS = [
  { id: "small", name: "1.5B", shape: "Qwen2.5-1.5B", params: 1.54e9, layers: 28, kvHeads: 2, headDim: 128 },
  { id: "medium", name: "3B", shape: "Qwen2.5-3B", params: 3.09e9, layers: 36, kvHeads: 2, headDim: 128 },
  { id: "large", name: "8B", shape: "Llama-3.1-8B", params: 8.03e9, layers: 32, kvHeads: 8, headDim: 128 },
] as const;

export type ServingModel = (typeof MODELS)[number];

/**
 * Whole-file bits per weight. F16 is exact. The others are llama.cpp's own
 * reported file sizes for Llama-3-8B (8.03B parameters), which include the
 * tensors a mix keeps at higher precision, so they sit above the block's
 * nominal rate (Q4_0 blocks are 4.5 bpw; the file is 4.64).
 */
export const SERVE_FORMATS = [
  { name: "F16", bitsPerWeight: 16, basis: "2 bytes per weight" },
  { name: "Q8_0", bitsPerWeight: fileBits(7.96), basis: "7.96 GiB Llama-3-8B file" },
  { name: "Q5_K_M", bitsPerWeight: fileBits(5.33), basis: "5.33 GiB Llama-3-8B file" },
  { name: "Q4_K_M", bitsPerWeight: fileBits(4.58), basis: "4.58 GiB Llama-3-8B file" },
  { name: "Q4_0", bitsPerWeight: fileBits(4.34), basis: "4.34 GiB Llama-3-8B file" },
] as const;

export type ServeFormat = (typeof SERVE_FORMATS)[number];

function fileBits(gibibytes: number) {
  return (gibibytes * GIB * 8) / 8.03e9;
}

/** Block-level bits per weight for the toy export, from the llama.cpp block structs. */
export const BLOCK_TYPE: Record<string, { block: string; bitsPerWeight: number }> = {
  F16: { block: "F16", bitsPerWeight: 16 },
  Q8_0: { block: "Q8_0", bitsPerWeight: 8.5 },
  Q5_K_M: { block: "Q5_K", bitsPerWeight: 5.5 },
  Q4_K_M: { block: "Q4_K", bitsPerWeight: 4.5 },
  Q4_0: { block: "Q4_0", bitsPerWeight: 4.5 },
};

/** Keys and values, every layer, every KV head, fp16. */
export function kvBytesPerToken(model: Pick<ServingModel, "layers" | "kvHeads" | "headDim">, bytesPerValue = 2) {
  return 2 * model.layers * model.kvHeads * model.headDim * bytesPerValue;
}

export interface ServingPoint {
  weightBytes: number;
  kvBytesPerToken: number;
  kvBytesPerStream: number;
  kvBytes: number;
  totalBytes: number;
  fits: boolean;
  maxStreams: number;
  stepSeconds: number;
  totalTokensPerSecond: number;
  perUserTokensPerSecond: number;
  timePerOutputToken: number;
  computeCeiling: number;
  computeBound: boolean;
  timeToFirstToken: number;
}

export function servingPoint(
  model: ServingModel,
  format: Pick<ServeFormat, "bitsPerWeight">,
  users: number,
  context: number,
  batching: boolean,
  hardware = HARDWARE,
): ServingPoint {
  const weightBytes = (model.params * format.bitsPerWeight) / 8;
  const perToken = kvBytesPerToken(model);
  const kvBytesPerStream = perToken * context;
  const kvBytes = kvBytesPerStream * users;
  const totalBytes = weightBytes + kvBytes + hardware.overheadBytes;
  const free = hardware.vramBytes - weightBytes - hardware.overheadBytes;
  const maxStreams = free <= 0 ? 0 : Math.floor(free / kvBytesPerStream);
  const computeCeiling = hardware.computeFlops / (2 * model.params);

  let stepSeconds: number;
  let totalTokensPerSecond: number;
  let perUserTokensPerSecond: number;
  let timePerOutputToken: number;
  if (batching) {
    // One step reads every weight once and each stream's cache once, and emits one token per stream.
    const memorySeconds = (weightBytes + kvBytes) / hardware.bandwidthBytesPerSecond;
    const computeSeconds = users / computeCeiling;
    stepSeconds = Math.max(memorySeconds, computeSeconds);
    totalTokensPerSecond = users / stepSeconds;
    perUserTokensPerSecond = 1 / stepSeconds;
    timePerOutputToken = stepSeconds;
  } else {
    // Round-robin: each step serves one stream, so each user waits for everyone else's step.
    stepSeconds = Math.max(
      (weightBytes + kvBytesPerStream) / hardware.bandwidthBytesPerSecond,
      1 / computeCeiling,
    );
    totalTokensPerSecond = 1 / stepSeconds;
    perUserTokensPerSecond = totalTokensPerSecond / users;
    timePerOutputToken = stepSeconds * users;
  }

  // Prefill of a prompt that fills the context: 2 FLOPs per parameter per token, at least one weight read.
  const timeToFirstToken = Math.max(
    (2 * model.params * context) / hardware.computeFlops,
    weightBytes / hardware.bandwidthBytesPerSecond,
  );

  return {
    weightBytes,
    kvBytesPerToken: perToken,
    kvBytesPerStream,
    kvBytes,
    totalBytes,
    fits: totalBytes <= hardware.vramBytes,
    maxStreams,
    stepSeconds,
    totalTokensPerSecond,
    perUserTokensPerSecond,
    timePerOutputToken,
    computeCeiling,
    computeBound: batching ? users / computeCeiling >= (weightBytes + kvBytes) / hardware.bandwidthBytesPerSecond : false,
    timeToFirstToken,
  };
}

/**
 * Expected tokens produced per target-model verification pass in speculative
 * decoding, with draft length gamma and per-token acceptance rate alpha
 * (Leviathan et al., 2023): (1 − α^(γ+1)) / (1 − α).
 */
export function speculativeTokensPerPass(alpha: number, gamma: number) {
  if (alpha >= 1) return gamma + 1;
  return (1 - alpha ** (gamma + 1)) / (1 - alpha);
}

export const formatBytes = (bytes: number) =>
  bytes >= GIB
    ? `${(bytes / GIB).toFixed(2)} GiB`
    : bytes >= 1024 ** 2
      ? `${(bytes / 1024 ** 2).toFixed(0)} MiB`
      : bytes >= 1024
        ? `${(bytes / 1024).toFixed(1)} KiB`
        : `${Math.round(bytes)} B`;
