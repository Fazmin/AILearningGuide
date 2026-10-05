/**
 * Module-local quantizers for the quantization lab.
 *
 * `quantizeWeights` is the teaching quantizer: one block per tensor, per row, or
 * per group, with either an absmax (symmetric) or a min/max (asymmetric) grid.
 * Its storage account charges one fp16 scale per symmetric block and an fp16
 * scale plus an fp16 offset per asymmetric block, which is exactly what
 * llama.cpp's Q4_0 and Q4_1 blocks store.
 *
 * `GGUF_BLOCK_FORMATS` emulates llama.cpp's block layouts on a flat weight
 * array. The bits-per-weight figures are the struct sizes in ggml-common.h,
 * not estimates. The K-quant emulation keeps the real structure (256-weight
 * super-blocks, sub-block scales quantized to 4, 6, or 8 bits against an fp16
 * super-scale) and skips llama.cpp's iterative scale search, so its error is
 * an upper bound on what the real quantizer achieves.
 */

export type QuantScope = "tensor" | "row" | "group";
export type ZeroPoint = "asymmetric" | "symmetric";

export const SCALE_BITS = 16;

export interface QuantBlock {
  start: number;
  end: number;
  /** Width of one integer step. */
  scale: number;
  /** Value that code 0 decodes to. Zero for symmetric blocks. */
  offset: number;
  minCode: number;
  maxCode: number;
}

export interface QuantResult {
  weights: Float32Array;
  codes: Int32Array;
  blocks: QuantBlock[];
  rmsError: number;
  maxError: number;
  /** Representable levels per block. */
  levels: number;
  metadataBitsPerBlock: number;
  bitsPerWeight: number;
  totalBytes: number;
}

export interface QuantOptions {
  bits: number;
  scope: QuantScope;
  groupSize?: number;
  zeroPoint?: ZeroPoint;
  rowLength?: number;
}

export function blockLength(scope: QuantScope, total: number, groupSize: number, rowLength: number) {
  if (scope === "tensor") return Math.max(1, total);
  if (scope === "row") return Math.max(1, rowLength);
  return Math.max(1, Math.round(groupSize));
}

export function quantizeWeights(source: Float32Array, options: QuantOptions): QuantResult {
  const bits = Math.max(1, Math.min(16, Math.round(options.bits)));
  const zeroPoint = options.zeroPoint ?? "asymmetric";
  const size = blockLength(options.scope, source.length, options.groupSize ?? 8, options.rowLength ?? 30);
  const count = Math.ceil(source.length / size);
  const weights = new Float32Array(source.length);
  const codes = new Int32Array(source.length);
  const blocks: QuantBlock[] = [];
  const symmetricMax = Math.max(1, 2 ** (bits - 1) - 1);
  const asymmetricMax = 2 ** bits - 1;

  for (let block = 0; block < count; block += 1) {
    const start = block * size;
    const end = Math.min(source.length, start + size);
    if (zeroPoint === "symmetric") {
      let amax = 0;
      for (let index = start; index < end; index += 1) amax = Math.max(amax, Math.abs(source[index]));
      const scale = amax === 0 ? 1 : amax / symmetricMax;
      for (let index = start; index < end; index += 1) {
        const code = Math.max(-symmetricMax, Math.min(symmetricMax, Math.round(source[index] / scale)));
        codes[index] = code;
        weights[index] = code * scale;
      }
      blocks.push({ start, end, scale, offset: 0, minCode: -symmetricMax, maxCode: symmetricMax });
    } else {
      let minimum = Infinity;
      let maximum = -Infinity;
      for (let index = start; index < end; index += 1) {
        minimum = Math.min(minimum, source[index]);
        maximum = Math.max(maximum, source[index]);
      }
      const span = maximum - minimum;
      const scale = span === 0 ? 1 : span / asymmetricMax;
      for (let index = start; index < end; index += 1) {
        const code = Math.max(0, Math.min(asymmetricMax, Math.round((source[index] - minimum) / scale)));
        codes[index] = code;
        weights[index] = minimum + code * scale;
      }
      blocks.push({ start, end, scale, offset: minimum, minCode: 0, maxCode: asymmetricMax });
    }
  }

  const { rmsError, maxError } = errorStats(source, weights);
  const metadataBitsPerBlock = zeroPoint === "symmetric" ? SCALE_BITS : 2 * SCALE_BITS;
  const payloadBits = source.length * bits + count * metadataBitsPerBlock;
  return {
    weights,
    codes,
    blocks,
    rmsError,
    maxError,
    levels: zeroPoint === "symmetric" ? 2 * symmetricMax + 1 : asymmetricMax + 1,
    metadataBitsPerBlock,
    bitsPerWeight: source.length === 0 ? bits : payloadBits / source.length,
    totalBytes: Math.ceil(payloadBits / 8),
  };
}

export function errorStats(source: Float32Array, rounded: Float32Array) {
  let squared = 0;
  let maxError = 0;
  for (let index = 0; index < source.length; index += 1) {
    const error = rounded[index] - source[index];
    squared += error * error;
    maxError = Math.max(maxError, Math.abs(error));
  }
  return { rmsError: source.length ? Math.sqrt(squared / source.length) : 0, maxError };
}

/** Distinct codes that at least one weight in the block actually landed on. */
export function codesUsed(result: QuantResult, block: QuantBlock) {
  const seen = new Set<number>();
  for (let index = block.start; index < block.end; index += 1) seen.add(result.codes[index]);
  return seen.size;
}

/** Round to the nearest IEEE half-precision value (round half to even). */
export function toHalf(value: number): number {
  if (value === 0 || !Number.isFinite(value)) return value;
  const magnitude = Math.abs(value);
  if (magnitude >= 65520) return Math.sign(value) * Infinity;
  let exponent = Math.floor(Math.log2(magnitude));
  if (2 ** exponent > magnitude) exponent -= 1;
  if (2 ** (exponent + 1) <= magnitude) exponent += 1;
  const spacing = 2 ** (Math.max(exponent, -14) - 10);
  const steps = magnitude / spacing;
  let rounded = Math.round(steps);
  if (steps - Math.floor(steps) === 0.5 && rounded % 2 === 1) rounded -= 1;
  return Math.sign(value) * rounded * spacing;
}

/* -------------------------------------------------------------------------- */
/* llama.cpp block layouts                                                     */
/* -------------------------------------------------------------------------- */

const signedMaxAbs = (source: Float32Array, start: number, end: number) => {
  let value = 0;
  for (let index = start; index < end; index += 1) {
    if (Math.abs(source[index]) > Math.abs(value)) value = source[index];
  }
  return value;
};

/** Q8_0: 32 weights, fp16 d = amax / 127, int8 codes. */
function quantizeQ8_0(source: Float32Array, into: Float32Array) {
  for (let start = 0; start < source.length; start += 32) {
    const end = Math.min(source.length, start + 32);
    let amax = 0;
    for (let index = start; index < end; index += 1) amax = Math.max(amax, Math.abs(source[index]));
    const d = toHalf(amax / 127);
    for (let index = start; index < end; index += 1) {
      into[index] = d === 0 ? 0 : Math.max(-127, Math.min(127, Math.round(source[index] / d))) * d;
    }
  }
}

/** Q4_0: 32 weights, fp16 d = signed-max / -8, codes -8..7 (the extreme maps to -8 exactly). */
function quantizeQ4_0(source: Float32Array, into: Float32Array) {
  for (let start = 0; start < source.length; start += 32) {
    const end = Math.min(source.length, start + 32);
    const d = toHalf(signedMaxAbs(source, start, end) / -8);
    const inverse = d === 0 ? 0 : 1 / d;
    for (let index = start; index < end; index += 1) {
      const code = Math.min(15, Math.max(0, Math.floor(source[index] * inverse + 8.5))) - 8;
      into[index] = code * d;
    }
  }
}

/** Q4_1: 32 weights, fp16 d and fp16 min, codes 0..15. */
function quantizeQ4_1(source: Float32Array, into: Float32Array) {
  for (let start = 0; start < source.length; start += 32) {
    const end = Math.min(source.length, start + 32);
    let minimum = Infinity;
    let maximum = -Infinity;
    for (let index = start; index < end; index += 1) {
      minimum = Math.min(minimum, source[index]);
      maximum = Math.max(maximum, source[index]);
    }
    const m = toHalf(minimum);
    const d = toHalf((maximum - minimum) / 15);
    const inverse = d === 0 ? 0 : 1 / d;
    for (let index = start; index < end; index += 1) {
      const code = Math.min(15, Math.max(0, Math.floor((source[index] - m) * inverse + 0.5)));
      into[index] = code * d + m;
    }
  }
}

/**
 * Asymmetric K-quant (Q2_K, Q4_K, Q5_K): each sub-block has a scale and a
 * non-negative min, both quantized to `scaleCodeMax` levels against fp16 d and
 * dmin for the 256-weight super-block. x ≈ d·sc·q − dmin·m.
 */
function quantizeKAffine(
  source: Float32Array,
  into: Float32Array,
  subBlock: number,
  codeMax: number,
  scaleCodeMax: number,
) {
  for (let superStart = 0; superStart < source.length; superStart += 256) {
    const superEnd = Math.min(source.length, superStart + 256);
    const scales: number[] = [];
    const mins: number[] = [];
    for (let start = superStart; start < superEnd; start += subBlock) {
      const end = Math.min(superEnd, start + subBlock);
      let minimum = 0;
      let maximum = -Infinity;
      for (let index = start; index < end; index += 1) {
        minimum = Math.min(minimum, source[index]);
        maximum = Math.max(maximum, source[index]);
      }
      scales.push(Math.max(0, maximum - minimum) / codeMax);
      mins.push(-minimum);
    }
    const d = toHalf(Math.max(...scales) / scaleCodeMax);
    const dmin = toHalf(Math.max(...mins) / scaleCodeMax);
    let sub = 0;
    for (let start = superStart; start < superEnd; start += subBlock, sub += 1) {
      const end = Math.min(superEnd, start + subBlock);
      const scale = d === 0 ? 0 : d * Math.min(scaleCodeMax, Math.round(scales[sub] / d));
      const offset = dmin === 0 ? 0 : dmin * Math.min(scaleCodeMax, Math.round(mins[sub] / dmin));
      for (let index = start; index < end; index += 1) {
        const code = scale === 0 ? 0 : Math.max(0, Math.min(codeMax, Math.round((source[index] + offset) / scale)));
        into[index] = scale * code - offset;
      }
    }
  }
}

/**
 * Symmetric K-quant (Q3_K, Q6_K): 16-weight sub-blocks with a signed scale,
 * quantized to signed `scaleCodeMax`-bit integers against one fp16 d.
 */
function quantizeKSymmetric(source: Float32Array, into: Float32Array, negativeCode: number, scaleNegative: number) {
  for (let superStart = 0; superStart < source.length; superStart += 256) {
    const superEnd = Math.min(source.length, superStart + 256);
    const scales: number[] = [];
    for (let start = superStart; start < superEnd; start += 16) {
      const end = Math.min(superEnd, start + 16);
      scales.push(signedMaxAbs(source, start, end) / -negativeCode);
    }
    let maxScale = 0;
    for (const scale of scales) if (Math.abs(scale) > Math.abs(maxScale)) maxScale = scale;
    const d = toHalf(maxScale / -scaleNegative);
    let sub = 0;
    for (let start = superStart; start < superEnd; start += 16, sub += 1) {
      const end = Math.min(superEnd, start + 16);
      const code = d === 0 ? 0 : Math.max(-scaleNegative, Math.min(scaleNegative - 1, Math.round(scales[sub] / d)));
      const scale = d * code;
      for (let index = start; index < end; index += 1) {
        const q =
          scale === 0 ? 0 : Math.max(-negativeCode, Math.min(negativeCode - 1, Math.round(source[index] / scale)));
        into[index] = q * scale;
      }
    }
  }
}

export interface BlockFormat {
  name: string;
  /** Bits in each weight's integer code. */
  codeBits: number;
  /** Exact bits per weight of one block, from the struct in ggml-common.h. */
  bitsPerWeight: number;
  layout: string;
  note: string;
  /** llama.cpp's own size for the whole-file mixes built on this block, where one exists. */
  files?: string;
  quantize: (source: Float32Array) => Float32Array;
}

const run = (fill: (source: Float32Array, into: Float32Array) => void) => (source: Float32Array) => {
  const into = new Float32Array(source.length);
  fill(source, into);
  return into;
};

export const GGUF_BLOCK_FORMATS: ReadonlyArray<BlockFormat> = [
  {
    name: "F16",
    codeBits: 16,
    bitsPerWeight: 16,
    layout: "IEEE half precision",
    note: "Not integer quantization: every weight keeps its own exponent.",
    files: "F16: 16 bpw",
    quantize: (source) => Float32Array.from(source, toHalf),
  },
  {
    name: "Q8_0",
    codeBits: 8,
    bitsPerWeight: 8.5,
    layout: "32 × int8 + fp16 scale = 34 B",
    note: "Symmetric, d = amax / 127.",
    files: "Q8_0: 7.96 GiB on Llama-3-8B",
    quantize: run(quantizeQ8_0),
  },
  {
    name: "Q6_K",
    codeBits: 6,
    bitsPerWeight: 6.5625,
    layout: "256 × 6-bit, 16 int8 sub-scales, fp16 d = 210 B",
    note: "Symmetric 16-weight sub-blocks.",
    files: "Q6_K: 6.14 GiB",
    quantize: run((source, into) => quantizeKSymmetric(source, into, 32, 128)),
  },
  {
    name: "Q5_K",
    codeBits: 5,
    bitsPerWeight: 5.5,
    layout: "256 × 5-bit, 8 × (6-bit scale, 6-bit min), fp16 d, dmin = 176 B",
    note: "Asymmetric 32-weight sub-blocks.",
    files: "Q5_K_S / Q5_K_M: 5.21 / 5.33 GiB",
    quantize: run((source, into) => quantizeKAffine(source, into, 32, 31, 63)),
  },
  {
    name: "Q4_K",
    codeBits: 4,
    bitsPerWeight: 4.5,
    layout: "256 × 4-bit, 8 × (6-bit scale, 6-bit min), fp16 d, dmin = 144 B",
    note: "Asymmetric 32-weight sub-blocks; scales cost 0.5 bpw, not 1.",
    files: "Q4_K_S / Q4_K_M: 4.37 / 4.58 GiB",
    quantize: run((source, into) => quantizeKAffine(source, into, 32, 15, 63)),
  },
  {
    name: "Q4_1",
    codeBits: 4,
    bitsPerWeight: 5,
    layout: "32 × 4-bit + fp16 scale + fp16 min = 20 B",
    note: "Asymmetric legacy block.",
    files: "Q4_1: 4.78 GiB",
    quantize: run(quantizeQ4_1),
  },
  {
    name: "Q4_0",
    codeBits: 4,
    bitsPerWeight: 4.5,
    layout: "32 × 4-bit + fp16 scale = 18 B",
    note: "Symmetric legacy block, codes −8…7.",
    files: "Q4_0: 4.34 GiB",
    quantize: run(quantizeQ4_0),
  },
  {
    name: "Q3_K",
    codeBits: 3,
    bitsPerWeight: 3.4375,
    layout: "256 × 3-bit, 16 × 6-bit scales, fp16 d = 110 B",
    note: "Symmetric 16-weight sub-blocks.",
    files: "Q3_K_M: 3.74 GiB",
    quantize: run((source, into) => quantizeKSymmetric(source, into, 4, 32)),
  },
  {
    name: "Q2_K",
    codeBits: 2,
    bitsPerWeight: 2.625,
    layout: "256 × 2-bit, 16 × (4-bit scale, 4-bit min), fp16 d, dmin = 84 B",
    note: "Asymmetric 16-weight sub-blocks.",
    files: "Q2_K: 2.96 GiB",
    quantize: run((source, into) => quantizeKAffine(source, into, 16, 3, 15)),
  },
];

/** Llama-3-8B parameter count, used to turn llama.cpp's reported file sizes into bits per weight. */
export const LLAMA3_8B_PARAMETERS = 8.03e9;
export const fileBitsPerWeight = (gibibytes: number) => (gibibytes * 1024 ** 3 * 8) / LLAMA3_8B_PARAMETERS;

/* -------------------------------------------------------------------------- */
/* The file this app downloads                                                 */
/* -------------------------------------------------------------------------- */

export type FileRole = "embedding" | "ffn" | "linear" | "attention" | "norms";

export interface FileTensorGroup {
  role: FileRole;
  /** ggml tensor type as recorded in the file's own tensor table. */
  type: "Q4_K" | "Q5_K" | "Q6_K" | "Q8_0" | "F32";
  tensors: number;
  parameters: number;
  bytes: number;
}

/**
 * The model file the desktop app downloads (src-tauri/src/local_ai.rs, MODEL_URL and MODEL_BYTES), so the lab can
 * ask the question the lesson is about: what does a "4-bit" file really cost per weight?
 *
 * Provenance. Read on 2026-09-30 from the GGUF header of the pinned revision (HTTP range request for the first
 * 20 MB, parsed per the GGUF v3 layout, then each tensor's bytes computed from its ggml block size and type size):
 * https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/blob/f6d5376be1edb4d416d56da11e5397a961aca8ae/Qwen3.5-2B-Q4_K_M.gguf
 * The same total, 1,881,825,088 parameters, is what the Hugging Face API reports as `gguf.total` for that repository.
 * Tensor data starts at byte 10,961,920 (header, metadata, the 248,320-token vocabulary and the chat template), and
 * 10,961,920 + the tensor bytes below is exactly the file size, which is how the table is checked in quoted.test.ts.
 * The per-group rows are sums over the 320 tensors (24 blocks, 6 of them full attention, 18 Gated DeltaNet).
 */
export const DOWNLOADED_FILE = {
  name: "Qwen3.5 2B · Q4_K_M",
  file: "Qwen3.5-2B-Q4_K_M.gguf",
  repository: "unsloth/Qwen3.5-2B-GGUF",
  revision: "f6d5376be1edb4d416d56da11e5397a961aca8ae",
  checked: "30 September 2026",
  bytes: 1_280_835_840,
  headerBytes: 10_961_920,
  parameters: 1_881_825_088,
  tensorCount: 320,
  layers: 24,
  fullAttentionLayers: 6,
  vocabulary: 248_320,
  hiddenSize: 2048,
  /** Recorded in the file's metadata: the quantizer was guided by an importance matrix computed on calibration text. */
  calibration: { dataset: "unsloth_calibration_Qwen3.5-2B.txt", chunks: 80, entries: 186 },
  groups: [
    { role: "embedding", type: "Q6_K", tensors: 1, parameters: 508_559_360, bytes: 417_177_600 },
    { role: "ffn", type: "Q4_K", tensors: 60, parameters: 754_974_720, bytes: 424_673_280 },
    { role: "ffn", type: "Q6_K", tensors: 12, parameters: 150_994_944, bytes: 123_863_040 },
    { role: "linear", type: "Q5_K", tensors: 36, parameters: 301_989_888, bytes: 207_618_048 },
    { role: "linear", type: "Q4_K", tensors: 18, parameters: 75_497_472, bytes: 42_467_328 },
    { role: "linear", type: "Q8_0", tensors: 36, parameters: 1_179_648, bytes: 1_253_376 },
    { role: "linear", type: "F32", tensors: 72, parameters: 445_248, bytes: 1_780_992 },
    { role: "attention", type: "Q4_K", tensors: 20, parameters: 83_886_080, bytes: 47_185_920 },
    { role: "attention", type: "Q6_K", tensors: 4, parameters: 4_194_304, bytes: 3_440_640 },
    { role: "norms", type: "F32", tensors: 61, parameters: 103_424, bytes: 413_696 },
  ] as ReadonlyArray<FileTensorGroup>,
} as const;

export const FILE_ROLE_LABELS: Record<FileRole, string> = {
  embedding: "Token embedding, shared with the output",
  ffn: "Feed-forward layers",
  linear: "Gated DeltaNet layers",
  attention: "Full-attention layers",
  norms: "Norm weights",
};

export interface FileSlice {
  key: string;
  label: string;
  tensors: number;
  parameters: number;
  bytes: number;
  bitsPerWeight: number;
  parameterShare: number;
  byteShare: number;
}

export const bitsPerWeightOf = (bytes: number, parameters: number) => (parameters === 0 ? 0 : (bytes * 8) / parameters);

/** Tensor groups of the downloaded file, summed by block type or by role in the network, largest bytes first. */
export function fileSlices(by: "type" | "role"): FileSlice[] {
  const tensorBytes = DOWNLOADED_FILE.groups.reduce((total, group) => total + group.bytes, 0);
  const merged = new Map<string, { tensors: number; parameters: number; bytes: number }>();
  for (const group of DOWNLOADED_FILE.groups) {
    const key = by === "type" ? group.type : group.role;
    const entry = merged.get(key) ?? { tensors: 0, parameters: 0, bytes: 0 };
    entry.tensors += group.tensors;
    entry.parameters += group.parameters;
    entry.bytes += group.bytes;
    merged.set(key, entry);
  }
  return [...merged.entries()]
    .map(([key, entry]) => ({
      key,
      label: by === "type" ? key : FILE_ROLE_LABELS[key as FileRole],
      ...entry,
      bitsPerWeight: bitsPerWeightOf(entry.bytes, entry.parameters),
      parameterShare: entry.parameters / DOWNLOADED_FILE.parameters,
      byteShare: entry.bytes / tensorBytes,
    }))
    .sort((a, b) => b.bytes - a.bytes);
}

/** Whole-file effective bits per weight: every byte of the file over the parameters in its tensor table. */
export const fileEffectiveBitsPerWeight = () => bitsPerWeightOf(DOWNLOADED_FILE.bytes, DOWNLOADED_FILE.parameters);

/* -------------------------------------------------------------------------- */
/* Floating-point formats                                                      */
/* -------------------------------------------------------------------------- */

export interface FloatFormat {
  id: "fp32" | "fp16" | "bf16" | "e4m3" | "e5m2";
  name: string;
  exponentBits: number;
  mantissaBits: number;
  bias: number;
  /**
   * IEEE-style formats give the whole top exponent to infinity and NaN. E4M3 (Micikevicius et al., 2022) keeps
   * infinities out and spends only one mantissa pattern on NaN, which buys one more binade of range.
   */
  topExponent: "special" | "number";
}

export const FLOAT_FORMATS: ReadonlyArray<FloatFormat> = [
  { id: "fp32", name: "fp32", exponentBits: 8, mantissaBits: 23, bias: 127, topExponent: "special" },
  { id: "fp16", name: "fp16", exponentBits: 5, mantissaBits: 10, bias: 15, topExponent: "special" },
  { id: "bf16", name: "bf16", exponentBits: 8, mantissaBits: 7, bias: 127, topExponent: "special" },
  { id: "e4m3", name: "fp8 E4M3", exponentBits: 4, mantissaBits: 3, bias: 7, topExponent: "number" },
  { id: "e5m2", name: "fp8 E5M2", exponentBits: 5, mantissaBits: 2, bias: 15, topExponent: "special" },
];

export interface FloatFormatStats {
  /** Sign, exponent, and mantissa bits together. */
  bits: number;
  maxFinite: number;
  minNormal: number;
  minSubnormal: number;
  /** Gap between 1 and the next value up: 2 to the power of minus the mantissa bits. */
  spacingAtOne: number;
  /** Largest relative error of rounding to nearest inside the normal range: half of the spacing at one. */
  maxRelativeError: number;
  /** Distinct finite values; +0 and -0 count once. */
  finiteValues: number;
}

export function floatFormatStats(format: FloatFormat): FloatFormatStats {
  const { exponentBits, mantissaBits, bias, topExponent } = format;
  const bits = 1 + exponentBits + mantissaBits;
  const topCode = 2 ** exponentBits - 1;
  // The biggest exponent field that still holds numbers, and the biggest mantissa that goes with it.
  const maxExponent = (topExponent === "special" ? topCode - 1 : topCode) - bias;
  const maxMantissa = topExponent === "special" ? 2 ** mantissaBits - 1 : 2 ** mantissaBits - 2;
  const specialCodes = topExponent === "special" ? 2 * 2 ** mantissaBits : 2;
  return {
    bits,
    maxFinite: (1 + maxMantissa / 2 ** mantissaBits) * 2 ** maxExponent,
    minNormal: 2 ** (1 - bias),
    minSubnormal: 2 ** (1 - bias - mantissaBits),
    spacingAtOne: 2 ** -mantissaBits,
    maxRelativeError: 2 ** -(mantissaBits + 1),
    finiteValues: 2 ** bits - specialCodes - 1,
  };
}

export interface FloatRounding {
  /** The nearest representable value, ties to even. For an overflow it is the first grid point past the largest finite value. */
  value: number;
  /** The magnitude rounds past the largest finite value, so the format cannot hold it. */
  overflow: boolean;
  /** The result lives below the smallest normal value, where the spacing stops shrinking. */
  subnormal: boolean;
}

/** Round to the nearest value a format can store, ties to even. Pure arithmetic on the bit layout, no bit tricks. */
export function roundToFloatFormat(input: number, format: FloatFormat): FloatRounding {
  if (!Number.isFinite(input) || input === 0) return { value: input, overflow: false, subnormal: false };
  const stats = floatFormatStats(format);
  const magnitude = Math.abs(input);
  let exponent = Math.floor(Math.log2(magnitude));
  if (2 ** exponent > magnitude) exponent -= 1;
  if (2 ** (exponent + 1) <= magnitude) exponent += 1;
  const spacing = 2 ** (Math.max(exponent, 1 - format.bias) - format.mantissaBits);
  const steps = magnitude / spacing;
  const floor = Math.floor(steps);
  const fraction = steps - floor;
  const rounded = fraction > 0.5 ? floor + 1 : fraction < 0.5 ? floor : floor % 2 === 0 ? floor : floor + 1;
  const result = rounded * spacing;
  return {
    value: Math.sign(input) * result,
    overflow: result > stats.maxFinite,
    subnormal: result > 0 && result < stats.minNormal,
  };
}

/** Round every weight of a table through a float format; the weights were float32 to begin with. */
export function roundTableToFloatFormat(source: Float32Array, format: FloatFormat): Float32Array {
  return Float32Array.from(source, (weight) => roundToFloatFormat(weight, format).value);
}

/** Histogram counts over fixed edges, so bins never move when the grid changes. */
export function histogram(values: ArrayLike<number>, minimum: number, maximum: number, bins: number) {
  const counts = new Array<number>(bins).fill(0);
  const width = (maximum - minimum) / bins || 1;
  for (let index = 0; index < values.length; index += 1) {
    const bin = Math.min(bins - 1, Math.max(0, Math.floor((values[index] - minimum) / width)));
    counts[bin] += 1;
  }
  return counts;
}
