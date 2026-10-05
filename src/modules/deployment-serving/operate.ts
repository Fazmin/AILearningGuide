import {
  GIB,
  HARDWARE,
  MODELS,
  SERVE_FORMATS,
  speculativeTokensPerPass,
  type ServeFormat,
  type ServingPoint,
} from "./serving";

/**
 * Closed-form serving arithmetic for the Deployment & serving lab beyond the SDK's `servingPoint`: a hybrid-attention
 * model preset (the one this app runs), hardware presets with editable inputs, paged attention, KV-cache
 * quantization, speculative decoding, and a cost per million tokens. Nothing here is measured.
 *
 * With every lever off, no recurrent state, and the SDK's own hardware, `operatingPoint` reproduces
 * `servingPoint` for the three SDK presets (serving.test.ts pins that), so the cards that existed before keep their
 * numbers.
 */

/** A preset whose KV cache is charged only on some layers, with a fixed recurrent state on the others. */
export interface OperatingModel {
  id: string;
  name: string;
  shape: string;
  params: number;
  /** Layers that keep a KV cache growing with context. For a hybrid model that is only its full-attention layers. */
  layers: number;
  kvHeads: number;
  headDim: number;
  /** All layers, cache-holding or not, for the note that explains the preset. */
  totalLayers: number;
  /** Fixed per-stream state of the recurrent (linear-attention) layers, in bytes. Zero for plain transformers. */
  stateBytesPerStream: number;
  /** How the shape was read, in one sentence. */
  basis: string;
}

/**
 * Recurrent state of one stream in llama.cpp (b10991, the engine this app pins), per recurrent layer:
 *   state  = ssm_d_state x ssm_d_inner            (llama-hparams.cpp, n_embd_s)
 *   conv   = (ssm_d_conv - 1) x (ssm_d_inner + 2 x ssm_n_group x ssm_d_state)   (n_embd_r)
 * both held as float32 whatever the KV cache type (llama-model.cpp passes GGML_TYPE_F32 for the recurrent memory).
 */
export function recurrentStateBytesPerStream(shape: {
  recurrentLayers: number;
  stateSize: number;
  innerSize: number;
  convKernel: number;
  groupCount: number;
  bytesPerElement?: number;
}) {
  const state = shape.stateSize * shape.innerSize;
  const conv = (shape.convKernel - 1) * (shape.innerSize + 2 * shape.groupCount * shape.stateSize);
  return shape.recurrentLayers * (state + conv) * (shape.bytesPerElement ?? 4);
}

/**
 * Qwen3.5 2B, the model this app downloads. Numbers read from the GGUF header of the pinned file on 2026-09-30
 * (qwen35.block_count 24, full_attention_interval 4, attention.head_count_kv 2, attention.key_length 256,
 * ssm.state_size 128, ssm.inner_size 2048, ssm.conv_kernel 4, ssm.group_count 16) and from the Qwen/Qwen3.5-2B model
 * card, which lists 6 x (3 x Gated DeltaNet + 1 x gated attention). Parameters are the file's tensor table total.
 */
export const LOCAL_ARCHITECTURE = {
  totalLayers: 24,
  fullAttentionLayers: 6,
  recurrentLayers: 18,
  kvHeads: 2,
  headDim: 256,
  stateSize: 128,
  innerSize: 2048,
  convKernel: 4,
  groupCount: 16,
  parameters: 1_881_825_088,
  fileBytes: 1_280_835_840,
} as const;

export const LOCAL_MODEL: OperatingModel = {
  id: "local",
  name: "2B · this app",
  shape: "Qwen3.5-2B",
  params: LOCAL_ARCHITECTURE.parameters,
  layers: LOCAL_ARCHITECTURE.fullAttentionLayers,
  kvHeads: LOCAL_ARCHITECTURE.kvHeads,
  headDim: LOCAL_ARCHITECTURE.headDim,
  totalLayers: LOCAL_ARCHITECTURE.totalLayers,
  stateBytesPerStream: recurrentStateBytesPerStream({
    recurrentLayers: LOCAL_ARCHITECTURE.recurrentLayers,
    stateSize: LOCAL_ARCHITECTURE.stateSize,
    innerSize: LOCAL_ARCHITECTURE.innerSize,
    convKernel: LOCAL_ARCHITECTURE.convKernel,
    groupCount: LOCAL_ARCHITECTURE.groupCount,
  }),
  basis:
    "24 layers, of which only every fourth is full attention: 6 layers with 2 key/value heads of 256 dimensions, which keep a cache that grows with context. The other 18 are Gated DeltaNet layers that keep a fixed recurrent state per stream.",
};

export const OPERATING_MODELS: ReadonlyArray<OperatingModel> = [
  ...MODELS.map(
    (model): OperatingModel => ({
      id: model.id,
      name: model.name,
      shape: model.shape,
      params: model.params,
      layers: model.layers,
      kvHeads: model.kvHeads,
      headDim: model.headDim,
      totalLayers: model.layers,
      stateBytesPerStream: 0,
      basis: `${model.layers} layers, every one full attention, ${model.kvHeads} key/value heads of ${model.headDim} dimensions.`,
    }),
  ),
  LOCAL_MODEL,
];

export interface OperatingFormat {
  name: string;
  bitsPerWeight: number;
  basis: string;
}

/** Formats for the local preset: the real file, and half precision as the reference. */
export const LOCAL_FORMATS: ReadonlyArray<OperatingFormat> = [
  { name: "F16", bitsPerWeight: 16, basis: "2 bytes per weight" },
  {
    name: "Q4_K_M",
    bitsPerWeight: (LOCAL_ARCHITECTURE.fileBytes * 8) / LOCAL_ARCHITECTURE.parameters,
    basis: "the 1,280,835,840-byte file this app downloads over its 1,881,825,088 parameters",
  },
];

export const formatsFor = (model: OperatingModel): ReadonlyArray<OperatingFormat> =>
  model.id === LOCAL_MODEL.id ? LOCAL_FORMATS : SERVE_FORMATS;

/** Bytes of one key or value for the cache types llama.cpp offers: fp16, and the q8_0 and q4_0 block layouts. */
export const KV_TYPES = [
  { id: "f16", name: "fp16", bitsPerValue: 16, basis: "2 bytes per value" },
  { id: "q8_0", name: "q8_0", bitsPerValue: 8.5, basis: "34 bytes per 32 values" },
  { id: "q4_0", name: "q4_0", bitsPerValue: 4.5, basis: "18 bytes per 32 values" },
] as const;

export type KvTypeId = (typeof KV_TYPES)[number]["id"];

/** vLLM's default block size, in tokens (Kwon et al., 2023, section 7.2). */
export const PAGE_TOKENS = 16;

export interface HardwareSpec {
  id: string;
  /** Short button label. */
  label: string;
  name: string;
  memoryGiB: number;
  bandwidthGBs: number;
  /** Peak throughput of the matrix hardware at fp16, in TFLOP/s, dense. */
  computeTflops: number;
  pricePerHour: number;
  /** Where each figure comes from and when it was read; assumptions are named as such. */
  basis: string;
}

/**
 * Presets. Every figure marked as published was read on 2026-09-30 from the page named in its basis; the others are
 * stated as assumptions. All four numbers stay editable in the lab: specs change, prices change faster.
 */
export const HARDWARE_PRESETS: ReadonlyArray<HardwareSpec> = [
  {
    id: "lab",
    label: "Lab default",
    name: "one 24 GiB accelerator",
    memoryGiB: HARDWARE.vramBytes / GIB,
    bandwidthGBs: HARDWARE.bandwidthBytesPerSecond / 1e9,
    computeTflops: HARDWARE.computeFlops / 1e12,
    pricePerHour: 1,
    basis:
      "This lab's original assumptions: 24 GiB, 400 GB/s, 120 TFLOP/s. It is not a product, and the $1.00 an hour is a placeholder to replace, not a quote.",
  },
  {
    id: "consumer",
    label: "RTX 4090",
    name: "consumer GPU, GeForce RTX 4090",
    memoryGiB: 24,
    bandwidthGBs: 1008,
    computeTflops: 165.2,
    pricePerHour: 0.74,
    basis:
      "NVIDIA Ada GPU Architecture whitepaper v2.02, Appendix A: 24 GB GDDR6X (read here as GiB), 1008 GB/s, and 165.2 TFLOP/s of FP16 tensor throughput with FP32 accumulate, dense, at boost clock. Price: $0.74 an hour for an RTX 4090 pod on runpod.io/pricing, page dated 27 September 2026. All read 30 September 2026.",
  },
  {
    id: "datacenter",
    label: "H100 SXM",
    name: "data-centre GPU, H100 SXM",
    memoryGiB: 80,
    bandwidthGBs: 3350,
    computeTflops: 989.5,
    pricePerHour: 3.49,
    basis:
      "nvidia.com/en-us/data-center/h100 and NVIDIA's H100 datasheet: 80 GB (read here as GiB), 3.35 TB/s, and 1,979 TFLOP/s of FP16 tensor throughput with sparsity, which the datasheet footnote says is one half without: 989.5 dense. Price: $3.49 an hour for an H100 SXM pod on runpod.io/pricing, page dated 27 September 2026; lambda.ai/pricing lists $4.29 for one H100 SXM instance. All read 30 September 2026.",
  },
  {
    id: "laptop",
    label: "M4 Max laptop",
    name: "Apple-silicon laptop, M4 Max",
    memoryGiB: 64,
    bandwidthGBs: 546,
    computeTflops: 18,
    pricePerHour: 0,
    basis:
      "Apple, M4 Max with a 40-core GPU: up to 128 GB of unified memory (Apple lists 48, 64, and 128 GB; the lab uses 64, read as GiB) and 546 GB/s, from the October 2024 newsroom release and the 14-inch MacBook Pro tech specs, read 30 September 2026. Apple publishes no GPU FLOP/s figure: 18 TFLOP/s is the FP32 peak the third-party database flopper.io lists, an assumption. Unified memory is shared with the rest of the system. Price 0: you already own it.",
  },
  {
    id: "cpu",
    label: "CPU box",
    name: "CPU box, Core i9-14900K with DDR5-5600",
    memoryGiB: 64,
    bandwidthGBs: 89.6,
    computeTflops: 1,
    pricePerHour: 0,
    basis:
      "Intel's specification page for the Core i9-14900K, read 30 September 2026: up to DDR5 5600 MT/s, two memory channels, and 89.6 GB/s of maximum memory bandwidth. The 64 GiB of memory and 1 TFLOP/s of compute are assumptions, not figures from that page. Price 0: you already own it.",
  },
];

export interface Levers {
  paged: boolean;
  /** Average share of the reserved context that a stream has actually filled, 0 to 1. Only used when `paged`. */
  pagedFill: number;
  /** Bits per cached key or value. 16 is fp16. */
  kvBits: number;
  speculative: boolean;
  /** Per-token acceptance rate alpha. */
  acceptance: number;
  /** Draft length gamma. */
  draftLength: number;
  /** Cost coefficient c: one draft step as a share of one target step (Leviathan et al., 2023, Definition 3.7). */
  draftCost: number;
}

export const NO_LEVERS: Levers = {
  paged: false,
  pagedFill: 1,
  kvBits: 16,
  speculative: false,
  acceptance: 0,
  draftLength: 0,
  draftCost: 0,
};

export interface OperatingHardware {
  vramBytes: number;
  bandwidthBytesPerSecond: number;
  computeFlops: number;
  overheadBytes: number;
  /** Dollars per hour for the whole accelerator. */
  pricePerHour: number;
  /** Share of each hour spent generating tokens at this load, 0 to 1. */
  utilization: number;
}

export const hardwareFrom = (
  spec: Pick<HardwareSpec, "memoryGiB" | "bandwidthGBs" | "computeTflops" | "pricePerHour">,
  utilization = 1,
): OperatingHardware => ({
  vramBytes: spec.memoryGiB * GIB,
  bandwidthBytesPerSecond: spec.bandwidthGBs * 1e9,
  computeFlops: spec.computeTflops * 1e12,
  overheadBytes: HARDWARE.overheadBytes,
  pricePerHour: spec.pricePerHour,
  utilization,
});

export interface OperatingPoint extends ServingPoint {
  /** Tokens each stream holds in the cache: its whole reserved context, or the filled part rounded up to whole pages. */
  tokensHeldPerStream: number;
  kvBytesPerValue: number;
  stateBytesPerStream: number;
  /** Fixed recurrent state of every stream, in bytes. */
  stateBytes: number;
  /** Tokens one stream produces per pass: 1, or the speculative expectation. */
  tokensPerPass: number;
  /** Seconds one pass takes: a decode step, or a draft-and-verify pass. */
  passSeconds: number;
  /**
   * Dollars to generate a million tokens at this load: the hourly price over the tokens produced in an hour of
   * `utilization`. Infinite when the load does not fit in memory.
   */
  costPerMillionTokens: number;
}

/**
 * The SDK's `servingPoint` with the levers added. KV bytes per token are 2 x cache layers x KV heads x head dim x
 * bytes per value; recurrent state is added per stream and read once per step; paged attention charges (and reads)
 * only the filled tokens, rounded up to 16-token pages; a speculative pass costs one verification step plus
 * `draftLength` draft steps at `draftCost` of a target step and yields (1 - a^(g+1)) / (1 - a) tokens.
 */
export function operatingPoint(
  model: OperatingModel,
  format: Pick<ServeFormat, "bitsPerWeight">,
  users: number,
  context: number,
  batching: boolean,
  hardware: OperatingHardware,
  levers: Levers = NO_LEVERS,
): OperatingPoint {
  const weightBytes = (model.params * format.bitsPerWeight) / 8;
  const kvBytesPerValue = levers.kvBits / 8;
  const perToken = 2 * model.layers * model.kvHeads * model.headDim * kvBytesPerValue;
  const tokensHeld = levers.paged
    ? Math.min(context, Math.ceil((context * levers.pagedFill) / PAGE_TOKENS) * PAGE_TOKENS)
    : context;
  const kvBytesPerStream = perToken * tokensHeld;
  const stateBytesPerStream = model.stateBytesPerStream;
  const perStream = kvBytesPerStream + stateBytesPerStream;
  const kvBytes = kvBytesPerStream * users;
  const stateBytes = stateBytesPerStream * users;
  const totalBytes = weightBytes + kvBytes + stateBytes + hardware.overheadBytes;
  const free = hardware.vramBytes - weightBytes - hardware.overheadBytes;
  const maxStreams = free <= 0 ? 0 : Math.floor(free / perStream);
  const computeCeiling = hardware.computeFlops / (2 * model.params);
  const bandwidth = hardware.bandwidthBytesPerSecond;

  const speculative = levers.speculative;
  const draftLength = speculative ? levers.draftLength : 0;
  const tokensPerPass = speculative ? speculativeTokensPerPass(levers.acceptance, levers.draftLength) : 1;
  const verified = draftLength + 1;

  let stepSeconds: number;
  let passSeconds: number;
  let totalTokensPerSecond: number;
  let perUserTokensPerSecond: number;
  let timePerOutputToken: number;
  let computeBound: boolean;
  if (batching) {
    // One step reads every weight once and each stream's cache and state once, and emits one token per stream.
    const memorySeconds = (weightBytes + kvBytes + stateBytes) / bandwidth;
    stepSeconds = Math.max(memorySeconds, users / computeCeiling);
    const verifySeconds = Math.max(memorySeconds, (users * verified) / computeCeiling);
    passSeconds = verifySeconds + draftLength * levers.draftCost * stepSeconds;
    totalTokensPerSecond = (users * tokensPerPass) / passSeconds;
    perUserTokensPerSecond = tokensPerPass / passSeconds;
    timePerOutputToken = passSeconds / tokensPerPass;
    computeBound = (users * verified) / computeCeiling >= memorySeconds;
  } else {
    // Round-robin: each step serves one stream, so each user waits for everyone else's step.
    const memorySeconds = (weightBytes + perStream) / bandwidth;
    stepSeconds = Math.max(memorySeconds, 1 / computeCeiling);
    const verifySeconds = Math.max(memorySeconds, verified / computeCeiling);
    passSeconds = verifySeconds + draftLength * levers.draftCost * stepSeconds;
    totalTokensPerSecond = tokensPerPass / passSeconds;
    perUserTokensPerSecond = totalTokensPerSecond / users;
    timePerOutputToken = (passSeconds * users) / tokensPerPass;
    computeBound = false;
  }

  // Prefill of a prompt that fills the context: 2 FLOPs per parameter per token, at least one weight read.
  const timeToFirstToken = Math.max((2 * model.params * context) / hardware.computeFlops, weightBytes / bandwidth);
  const fits = totalBytes <= hardware.vramBytes;
  const tokensPerHour = totalTokensPerSecond * 3600 * hardware.utilization;
  return {
    weightBytes,
    kvBytesPerToken: perToken,
    kvBytesPerStream,
    kvBytes,
    totalBytes,
    fits,
    maxStreams,
    stepSeconds,
    totalTokensPerSecond,
    perUserTokensPerSecond,
    timePerOutputToken,
    computeCeiling,
    computeBound,
    timeToFirstToken,
    tokensHeldPerStream: tokensHeld,
    kvBytesPerValue,
    stateBytesPerStream,
    stateBytes,
    tokensPerPass,
    passSeconds,
    costPerMillionTokens: fits && tokensPerHour > 0 ? (hardware.pricePerHour / tokensPerHour) * 1e6 : Number.POSITIVE_INFINITY,
  };
}

/* -------------------------------------------------------------------------- */
/* Lab state                                                                   */
/* -------------------------------------------------------------------------- */

export const LIMITS = {
  users: { min: 1, max: 64 },
  context: { min: 512, max: 32768 },
  memoryGiB: { min: 4, max: 256 },
  bandwidthGBs: { min: 10, max: 4000 },
  computeTflops: { min: 0.5, max: 2000 },
  pricePerHour: { min: 0, max: 20 },
  utilization: { min: 0.1, max: 1 },
  pagedFill: { min: 0.1, max: 1 },
  acceptance: { min: 0, max: 0.95 },
  draftLength: { min: 1, max: 8 },
  draftCost: { min: 0.02, max: 0.5 },
} as const;

/** Everything the lab stores: controls only, never a result. */
export interface ServingConfig {
  modelId: string;
  formatName: string;
  users: number;
  context: number;
  batching: boolean;
  memoryGiB: number;
  bandwidthGBs: number;
  computeTflops: number;
  pricePerHour: number;
  utilization: number;
  paged: boolean;
  pagedFill: number;
  kvType: KvTypeId;
  speculative: boolean;
  acceptance: number;
  draftLength: number;
  draftCost: number;
}

export const DEFAULT_CONFIG: ServingConfig = {
  modelId: "large",
  formatName: "Q4_K_M",
  users: 8,
  context: 4096,
  batching: true,
  memoryGiB: 24,
  bandwidthGBs: 400,
  computeTflops: 120,
  pricePerHour: 1,
  utilization: 1,
  paged: false,
  pagedFill: 0.5,
  kvType: "f16",
  speculative: false,
  acceptance: 0.8,
  draftLength: 4,
  draftCost: 0.1,
};

const FORMAT_NAMES: ReadonlyArray<string> = [...new Set([...SERVE_FORMATS, ...LOCAL_FORMATS].map((entry) => entry.name))];

/**
 * Validate and clamp every key. State arrives from share links, snapshots, and hand-edited storage, and every control
 * here feeds closed-form arithmetic, so an out-of-range value must land on the nearest value the control could set.
 * Version 1 stored only modelId, formatName, users, context, and batching; the rest take their defaults.
 */
export function normalizeConfig(raw: Record<string, unknown>): ServingConfig {
  const number = (key: keyof ServingConfig, step: number, limits: { min: number; max: number }) => {
    const value = raw[key];
    const base = typeof value === "number" && Number.isFinite(value) ? value : (DEFAULT_CONFIG[key] as number);
    return Math.min(limits.max, Math.max(limits.min, Math.round(base / step) * step));
  };
  // Round to the step's own decimals so 0.1 + 0.2 style noise never reaches a stored value or a slider.
  const decimals = (value: number, places: number) => Number(value.toFixed(places));
  const flag = (key: keyof ServingConfig) => (typeof raw[key] === "boolean" ? (raw[key] as boolean) : (DEFAULT_CONFIG[key] as boolean));
  const modelId = typeof raw.modelId === "string" && OPERATING_MODELS.some((model) => model.id === raw.modelId) ? raw.modelId : DEFAULT_CONFIG.modelId;
  const formatName =
    typeof raw.formatName === "string" && FORMAT_NAMES.includes(raw.formatName) ? raw.formatName : DEFAULT_CONFIG.formatName;
  const kvType = KV_TYPES.some((entry) => entry.id === raw.kvType) ? (raw.kvType as KvTypeId) : DEFAULT_CONFIG.kvType;
  return {
    modelId,
    formatName,
    users: number("users", 1, LIMITS.users),
    context: number("context", 512, LIMITS.context),
    batching: flag("batching"),
    memoryGiB: number("memoryGiB", 1, LIMITS.memoryGiB),
    bandwidthGBs: decimals(number("bandwidthGBs", 0.1, LIMITS.bandwidthGBs), 1),
    computeTflops: decimals(number("computeTflops", 0.1, LIMITS.computeTflops), 1),
    pricePerHour: decimals(number("pricePerHour", 0.01, LIMITS.pricePerHour), 2),
    utilization: decimals(number("utilization", 0.05, LIMITS.utilization), 2),
    paged: flag("paged"),
    pagedFill: decimals(number("pagedFill", 0.05, LIMITS.pagedFill), 2),
    kvType,
    speculative: flag("speculative"),
    acceptance: decimals(number("acceptance", 0.05, LIMITS.acceptance), 2),
    draftLength: number("draftLength", 1, LIMITS.draftLength),
    draftCost: decimals(number("draftCost", 0.01, LIMITS.draftCost), 2),
  };
}

/** The preset whose four editable numbers match the current ones exactly, if any. */
export const activePreset = (config: Pick<ServingConfig, "memoryGiB" | "bandwidthGBs" | "computeTflops" | "pricePerHour">) =>
  HARDWARE_PRESETS.find(
    (preset) =>
      preset.memoryGiB === config.memoryGiB &&
      preset.bandwidthGBs === config.bandwidthGBs &&
      preset.computeTflops === config.computeTflops &&
      preset.pricePerHour === config.pricePerHour,
  );

export const leversFrom = (
  config: Pick<ServingConfig, "paged" | "pagedFill" | "kvType" | "speculative" | "acceptance" | "draftLength" | "draftCost">,
): Levers => ({
  paged: config.paged,
  pagedFill: config.pagedFill,
  kvBits: (KV_TYPES.find((entry) => entry.id === config.kvType) ?? KV_TYPES[0]).bitsPerValue,
  speculative: config.speculative,
  acceptance: config.acceptance,
  draftLength: config.draftLength,
  draftCost: config.draftCost,
});
