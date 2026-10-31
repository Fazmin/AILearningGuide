import { teachingMemory } from "@app/module-sdk";

/**
 * Closed-form arithmetic for splitting one training step across many GPUs: what each GPU holds,
 * how long a step takes, and how often a job of that size is interrupted. Nothing here is measured
 * and nothing runs on a GPU.
 *
 * Every formula is first order and comes from a named source:
 * - parameter counts from each model's published shape (hidden size, layers, heads, FFN size, vocabulary);
 * - model state per GPU from the ZeRO paper's accounting (Rajbhandari et al., 2020, section 3.1 and Figure 1):
 *   16 bytes per parameter for mixed-precision Adam, split by ZeRO stage;
 * - activations per layer from Korthikanti et al. (2023), Table 2, with sequence parallelism whenever t > 1;
 * - model FLOPs per token, 6 × active parameters + 12 × layers × hidden × sequence, from PaLM (Chowdhery
 *   et al., 2023, Appendix B), which also defines MFU;
 * - collective times from the bandwidth terms of ring collectives, with the four all-reduces per layer of
 *   Megatron-LM (Shoeybi et al., 2019) and the pipeline bubble of Narayanan et al. (2021);
 * - checkpoint intervals from Young (1974) and Daly (2006), with the failure rate derived from the
 *   Llama 3 paper's 54-day snapshot (419 unexpected interruptions on up to 16,384 GPUs).
 */

/** Tokens per sequence in every layout this lab prices (Llama 3 pre-trained at 8,192 before long context). */
export const SEQUENCE = 8192;

/**
 * One H100 SXM, as stated assumptions. 989.5 TFLOP/s is half of the 1,979 teraFLOPS "with sparsity" that
 * NVIDIA's H100 page lists for BF16 (the H100 whitepaper prints 989.4 dense). 900 GB/s of NVLink is the total
 * of both directions, so 450 GB/s each way. 50 GB/s is one 400 Gb/s InfiniBand port per GPU, as in a DGX H100
 * and in the Llama 3 clusters. The kernel efficiency is the share of peak that the matrix arithmetic is
 * assumed to reach; it is a stated constant, not a measurement.
 */
export const HARDWARE = {
  name: "H100 SXM",
  peakFlops: 989.5e12,
  memoryBytes: 80e9,
  nvlinkBytesPerSecond: 450e9,
  networkBytesPerSecond: 50e9,
  gpusPerNode: 8,
  efficiency: 0.75,
} as const;

export type ModelId = "8b" | "70b" | "405b" | "mixtral";

export interface ModelShape {
  id: ModelId;
  /** Button label. */
  label: string;
  layers: number;
  hidden: number;
  heads: number;
  kvHeads: number;
  /** Width of the feed-forward layer, per expert. */
  ffn: number;
  vocab: number;
  /** Experts per layer: 1 for a dense model. */
  experts: number;
  /** Experts each token is sent to. */
  topK: number;
}

/**
 * Published shapes: the Llama 3 paper's Table 3 and the models' config.json files (vocabulary 128,256), and the
 * Mixtral paper's Table 1 (8 experts, 2 per token, vocabulary 32,000). Every model here has untied input and
 * output embeddings.
 */
export const MODELS: ReadonlyArray<ModelShape> = [
  { id: "8b", label: "Llama 3.1 8B", layers: 32, hidden: 4096, heads: 32, kvHeads: 8, ffn: 14336, vocab: 128256, experts: 1, topK: 1 },
  { id: "70b", label: "Llama 3.1 70B", layers: 80, hidden: 8192, heads: 64, kvHeads: 8, ffn: 28672, vocab: 128256, experts: 1, topK: 1 },
  { id: "405b", label: "Llama 3.1 405B", layers: 126, hidden: 16384, heads: 128, kvHeads: 8, ffn: 53248, vocab: 128256, experts: 1, topK: 1 },
  { id: "mixtral", label: "Mixtral 8x7B", layers: 32, hidden: 4096, heads: 32, kvHeads: 8, ffn: 14336, vocab: 32000, experts: 8, topK: 2 },
];

export const modelById = (id: string): ModelShape => MODELS.find((model) => model.id === id) ?? MODELS[0];

export const isMoe = (model: ModelShape) => model.experts > 1;

export interface ParameterCounts {
  /** Every parameter the model stores. */
  total: number;
  /** Parameters one token passes through: the dense part plus topK experts per layer. */
  active: number;
  /** Parameters every GPU in a data-parallel group replicates: attention, embeddings, and a dense FFN. */
  dense: number;
  /** Expert FFN parameters, split further by expert parallelism. Zero for a dense model. */
  expert: number;
}

/**
 * Ψ = L·(2h² + 2h·kv·h/a + 3h·f·E) + 2Vh. Query and output projections are h × h; key and value are
 * h × (kv heads × head size); a SwiGLU FFN has three h × f matrices per expert; the input embedding and the
 * output head are V × h each. Norms and the router (a few hundred thousand parameters) are left out.
 */
export function parameterCounts(model: ModelShape): ParameterCounts {
  const { layers: L, hidden: h, heads: a, kvHeads: kv, ffn: f, vocab: V, experts: E, topK } = model;
  const attention = 2 * h * h + 2 * h * kv * (h / a);
  const ffnPerExpert = 3 * h * f;
  const embeddings = 2 * V * h;
  const moe = E > 1;
  const dense = L * attention + embeddings + (moe ? 0 : L * ffnPerExpert);
  const expert = moe ? L * ffnPerExpert * E : 0;
  return {
    total: dense + expert,
    active: dense + (moe ? L * ffnPerExpert * topK : 0),
    dense,
    expert,
  };
}

/* -------------------------------------------------------------------------- */
/* Controls                                                                    */
/* -------------------------------------------------------------------------- */

export const TENSOR_LADDER = [1, 2, 4, 8, 16, 32] as const;
export const PIPELINE_LADDER = [1, 2, 4, 8, 16] as const;
export const DATA_LADDER = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048] as const;
export const EXPERT_LADDER = [1, 2, 4, 8] as const;
export const MICRO_BATCH_LADDER = [1, 2, 4, 8] as const;
export const MICROBATCHES_LADDER = [1, 2, 4, 8, 16, 32, 64, 128] as const;
export const ZERO_STAGES = [0, 1, 2, 3] as const;
export const RECOMPUTE = ["none", "selective", "full"] as const;

export type ZeroStage = (typeof ZERO_STAGES)[number];
export type Recompute = (typeof RECOMPUTE)[number];

export const RECOMPUTE_LABELS: Record<Recompute, string> = { none: "None", selective: "Selective", full: "Full" };

/** Minutes between checkpoints, seconds to save one, and minutes to restart after a failure. */
export const CHECKPOINT_MINUTES = { min: 5, max: 120, step: 1 } as const;
export const SAVE_SECONDS = { min: 5, max: 600, step: 5 } as const;
export const RESTART_MINUTES = { min: 1, max: 60, step: 1 } as const;

/** The largest value on a ladder that is not above `value`, so a hand-edited 300 becomes 256. */
export function snapToLadder(value: number, ladder: ReadonlyArray<number>): number {
  if (!Number.isFinite(value)) return ladder[0];
  let chosen = ladder[0];
  for (const step of ladder) if (step <= value) chosen = step;
  return chosen;
}

/* -------------------------------------------------------------------------- */
/* Layout                                                                      */
/* -------------------------------------------------------------------------- */

export interface Layout {
  model: ModelShape;
  /** Tensor-parallel degree t: each layer's matrices split over t GPUs. */
  tensor: number;
  /** Pipeline-parallel degree p: the layers split into p stages. */
  pipeline: number;
  /** Data-parallel degree d: d copies of the whole t × p arrangement, each on its own data. */
  data: number;
  /** Expert-parallel degree e, carved out of the data-parallel group. Only an MoE model uses it. */
  expert: number;
  zero: ZeroStage;
  /** Sequences per microbatch, b. */
  microBatch: number;
  /** Microbatches per step on each pipeline, m (gradient accumulation steps). */
  microbatches: number;
  recompute: Recompute;
  /** Overlap the data-parallel gradient sync with the last microbatch's backward pass. */
  overlap: boolean;
}

export const gpuCount = (layout: Pick<Layout, "tensor" | "pipeline" | "data">) => layout.tensor * layout.pipeline * layout.data;

/** The expert-parallel degree actually used: 1 for a dense model, and never more than the data-parallel degree. */
export const effectiveExpert = (layout: Pick<Layout, "model" | "expert" | "data">) =>
  isMoe(layout.model) ? Math.max(1, Math.min(layout.expert, layout.data)) : 1;

export type LinkId = "nvlink" | "network";

export interface Link {
  id: LinkId;
  label: string;
  bytesPerSecond: number;
}

export const LINKS: Record<LinkId, Link> = {
  nvlink: { id: "nvlink", label: "NVLink", bytesPerSecond: HARDWARE.nvlinkBytesPerSecond },
  network: { id: "network", label: "InfiniBand", bytesPerSecond: HARDWARE.networkBytesPerSecond },
};

/**
 * Ranks are numbered tensor-fastest: a tensor group is t neighbouring GPUs, the expert and data groups step by t,
 * and pipeline stages step by t × d. A group of n GPUs placed `stride` apart stays inside one 8-GPU node when
 * stride × n ≤ 8, and then talks over NVLink; otherwise its slowest hop crosses the network.
 */
export function linkFor(stride: number, size: number): Link {
  return stride * size <= HARDWARE.gpusPerNode ? LINKS.nvlink : LINKS.network;
}

/* -------------------------------------------------------------------------- */
/* Collectives                                                                 */
/* -------------------------------------------------------------------------- */

/** Ring all-reduce: each GPU sends and receives 2(n − 1)/n of the buffer. */
export const allReduceSeconds = (bytes: number, size: number, bytesPerSecond: number) =>
  size <= 1 ? 0 : ((2 * (size - 1)) / size) * (bytes / bytesPerSecond);

/** Reduce-scatter, all-gather, and all-to-all: each GPU moves (n − 1)/n of the buffer. */
export const reduceScatterSeconds = (bytes: number, size: number, bytesPerSecond: number) =>
  size <= 1 ? 0 : ((size - 1) / size) * (bytes / bytesPerSecond);
export const allGatherSeconds = reduceScatterSeconds;
export const allToAllSeconds = reduceScatterSeconds;

/* -------------------------------------------------------------------------- */
/* Memory per GPU                                                              */
/* -------------------------------------------------------------------------- */

/** Bytes of each activation element and the per-element counts are Korthikanti et al.'s (16-bit activations). */
export function activationBytesPerLayer(model: ModelShape, microBatch: number, tensor: number, recompute: Recompute): number {
  const s = SEQUENCE;
  const h = model.hidden;
  const sbh = s * microBatch * h;
  if (recompute === "full") return 2 * sbh; // only each layer's input is kept (Table 2: sbh·2, not divided by t)
  if (recompute === "selective") return (34 * sbh) / tensor; // the attention core is recomputed
  return (sbh * (34 + (5 * model.heads * s) / h)) / tensor; // tensor + sequence parallel: sbh(34/t + 5as/(ht))
}

export interface MemoryPerGpu {
  /** Parameters this GPU's tensor-and-pipeline slice owns before ZeRO: dense and expert parts. */
  denseShare: number;
  expertShare: number;
  /** Ψg = Ψdense/(t·p) + Ψexperts/(t·p·e). */
  sliceParameters: number;
  weights: number;
  gradients: number;
  optimizer: number;
  modelState: number;
  activations: number;
  /** Layers on the busiest (first) stage, rounded up. */
  layersPerStage: number;
  /** Microbatches whose activations the first stage holds at once under 1F1B: min(m, p). */
  inFlight: number;
  activationPerLayer: number;
  total: number;
  fits: boolean;
  headroom: number;
}

export function memoryPerGpu(layout: Layout): MemoryPerGpu {
  const { model, tensor: t, pipeline: p, data: d, zero } = layout;
  const e = effectiveExpert(layout);
  const counts = parameterCounts(model);
  const denseShare = counts.dense / (t * p);
  const expertShare = counts.expert / (t * p * e);
  // ZeRO-0 bytes per parameter come from the shared training-memory arithmetic: 2 + 2 + 12 for mixed-precision Adam.
  const bytes = teachingMemory.memoryBreakdown(1, "mixed", "adam");
  // The dense part is replicated d times; each expert is replicated d / e times (its expert-data-parallel group).
  const shard = (sharded: boolean) => denseShare / (sharded ? d : 1) + expertShare / (sharded ? d / e : 1);
  const weights = bytes.weightBytesPerParameter * shard(zero >= 3);
  const gradients = bytes.gradientBytesPerParameter * shard(zero >= 2);
  const optimizer = bytes.stateBytesPerParameter * shard(zero >= 1);
  const layersPerStage = Math.ceil(model.layers / p);
  const inFlight = Math.min(layout.microbatches, p);
  const activationPerLayer = activationBytesPerLayer(model, layout.microBatch, t, layout.recompute);
  const activations = layersPerStage * inFlight * activationPerLayer;
  const modelState = weights + gradients + optimizer;
  const total = modelState + activations;
  return {
    denseShare,
    expertShare,
    sliceParameters: denseShare + expertShare,
    weights,
    gradients,
    optimizer,
    modelState,
    activations,
    layersPerStage,
    inFlight,
    activationPerLayer,
    total,
    fits: total <= HARDWARE.memoryBytes,
    headroom: HARDWARE.memoryBytes - total,
  };
}

/** ZeRO model state for a dense model, as in the ZeRO paper's Figure 1: Ψ parameters on d data-parallel GPUs. */
export function zeroModelState(parameters: number, data: number, zero: ZeroStage): number {
  if (zero === 0) return 16 * parameters;
  if (zero === 1) return 4 * parameters + (12 * parameters) / data;
  if (zero === 2) return 2 * parameters + (14 * parameters) / data;
  return (16 * parameters) / data;
}

/* -------------------------------------------------------------------------- */
/* Step time                                                                   */
/* -------------------------------------------------------------------------- */

/** Hardware FLOPs over model FLOPs: full recomputation runs the forward pass twice, 4/3 of the work. */
export const recomputeFactor = (recompute: Recompute) => (recompute === "full" ? 4 / 3 : 1);

/** Model FLOPs per token: 6 per active parameter plus 12·L·h·s for attention scores (forward and backward). */
export const flopsPerToken = (model: ModelShape) => 6 * parameterCounts(model).active + 12 * model.layers * model.hidden * SEQUENCE;

/** All-reduces per layer per microbatch for tensor parallelism: 2 forward and 2 backward, plus 2 to recompute. */
export const tensorAllReducesPerLayer = (recompute: Recompute) => (recompute === "full" ? 6 : 4);

/** All-to-alls per MoE layer per microbatch: dispatch and combine, forward and backward. */
export const ALL_TO_ALLS_PER_LAYER = 4;

export interface CollectiveCost {
  /** GPUs in the group. */
  size: number;
  link: Link;
  /** Bytes in one call's buffer. */
  bytesPerCall: number;
  /** Calls per microbatch on one stage (per step for the data-parallel sync). */
  calls: number;
  /** Seconds per microbatch (per step for the data-parallel sync). */
  seconds: number;
}

export interface StepTime {
  gpus: number;
  /** Seconds of matrix arithmetic per microbatch on one stage, including any recomputation. */
  computePerMicrobatch: number;
  tensor: CollectiveCost;
  pipelineSend: CollectiveCost;
  expertAllToAll: CollectiveCost;
  dataSync: CollectiveCost;
  /** Compute plus tensor, pipeline, and expert communication for one microbatch on one stage. */
  microbatchTime: number;
  /** (m + p − 1) microbatch slots: m doing work, p − 1 waiting. */
  pipelineTime: number;
  bubbleFraction: number;
  /** Seconds the data-parallel sync adds after the last backward pass. */
  exposedDataSync: number;
  step: number;
  tokensPerStep: number;
  tokensPerSecond: number;
  mfu: number;
  hfu: number;
  /** The step split into what the GPUs were doing: these add up to `step`. */
  parts: {
    compute: number;
    recompute: number;
    communication: number;
    bubble: number;
    exposedDataSync: number;
  };
}

export function stepTime(layout: Layout): StepTime {
  const { model, tensor: t, pipeline: p, data: d, microbatches: m, microBatch: b } = layout;
  const e = effectiveExpert(layout);
  const gpus = gpuCount(layout);
  const s = SEQUENCE;
  const h = model.hidden;
  const r = recomputeFactor(layout.recompute);
  const layersPerStage = model.layers / p; // the average stage; stage imbalance is ignored
  const tokensPerMicrobatch = b * s;

  const computePerMicrobatch =
    (flopsPerToken(model) * tokensPerMicrobatch * r) / (t * p * HARDWARE.efficiency * HARDWARE.peakFlops);

  const tensorLink = linkFor(1, t);
  const tensorBytes = 2 * b * s * h;
  const tensorCalls = t > 1 ? tensorAllReducesPerLayer(layout.recompute) * layersPerStage : 0;
  const tensor: CollectiveCost = {
    size: t,
    link: tensorLink,
    bytesPerCall: tensorBytes,
    calls: tensorCalls,
    seconds: tensorCalls * allReduceSeconds(tensorBytes, t, tensorLink.bytesPerSecond),
  };

  // One activation forward and one gradient back across each stage boundary, sharded by sequence parallelism.
  const pipelineLink = linkFor(t * d, p);
  const pipelineBytes = (2 * b * s * h) / t;
  const pipelineCalls = p > 1 ? 2 : 0;
  const pipelineSend: CollectiveCost = {
    size: p,
    link: pipelineLink,
    bytesPerCall: pipelineBytes,
    calls: pipelineCalls,
    seconds: (pipelineCalls * pipelineBytes) / pipelineLink.bytesPerSecond,
  };

  const expertLink = linkFor(t, e);
  const expertBytes = 2 * b * s * model.topK * h;
  const expertCalls = isMoe(model) && e > 1 ? ALL_TO_ALLS_PER_LAYER * layersPerStage : 0;
  const expertAllToAll: CollectiveCost = {
    size: e,
    link: expertLink,
    bytesPerCall: expertBytes,
    calls: expertCalls,
    seconds: expertCalls * allToAllSeconds(expertBytes, e, expertLink.bytesPerSecond),
  };

  // Gradient sync: bf16 gradients of this GPU's slice, all-reduced (or reduce-scattered and all-gathered, the same
  // volume) over the data-parallel group; ZeRO-3 also all-gathers the parameters in the backward pass, 1.5 times.
  const memory = memoryPerGpu(layout);
  const dataLink = linkFor(t, d);
  const zeroFactor = layout.zero === 3 ? 1.5 : 1;
  const denseSync = allReduceSeconds(2 * memory.denseShare, d, dataLink.bytesPerSecond);
  const expertSync = allReduceSeconds(2 * memory.expertShare, d / e, dataLink.bytesPerSecond);
  const dataSeconds = (denseSync + expertSync) * zeroFactor;
  const dataSync: CollectiveCost = {
    size: d,
    link: dataLink,
    bytesPerCall: 2 * (memory.denseShare + memory.expertShare),
    calls: d > 1 ? 1 : 0,
    seconds: dataSeconds,
  };

  const communicationPerMicrobatch = tensor.seconds + pipelineSend.seconds + expertAllToAll.seconds;
  const microbatchTime = computePerMicrobatch + communicationPerMicrobatch;
  const pipelineTime = (m + p - 1) * microbatchTime;
  const bubbleFraction = (p - 1) / (m + p - 1);
  // With overlap on, the sync runs under the last microbatch's backward pass, two thirds of its compute.
  const exposedDataSync = layout.overlap ? Math.max(0, dataSeconds - (2 / 3) * computePerMicrobatch) : dataSeconds;
  const step = pipelineTime + exposedDataSync;
  const tokensPerStep = d * m * b * s;
  const tokensPerSecond = tokensPerStep / step;
  const mfu = (flopsPerToken(model) * tokensPerStep) / (step * gpus * HARDWARE.peakFlops);

  const usefulCompute = (m * computePerMicrobatch) / r;
  return {
    gpus,
    computePerMicrobatch,
    tensor,
    pipelineSend,
    expertAllToAll,
    dataSync,
    microbatchTime,
    pipelineTime,
    bubbleFraction,
    exposedDataSync,
    step,
    tokensPerStep,
    tokensPerSecond,
    mfu,
    hfu: mfu * r,
    parts: {
      compute: usefulCompute,
      recompute: m * computePerMicrobatch - usefulCompute,
      communication: m * communicationPerMicrobatch,
      bubble: (p - 1) * microbatchTime,
      exposedDataSync,
    },
  };
}

/* -------------------------------------------------------------------------- */
/* Pipeline schedule (1F1B)                                                    */
/* -------------------------------------------------------------------------- */

export interface ScheduledOp {
  stage: number;
  microbatch: number;
  kind: "forward" | "backward";
  /** In units of one forward pass: a forward takes 1, a backward 2. */
  start: number;
  end: number;
}

/**
 * The one-forward-one-backward schedule (PipeDream-Flush, as Narayanan et al. use it): stage s first runs
 * min(p − s − 1, m) forwards, then alternates one forward and one backward, then drains its backwards. A forward
 * waits for the previous stage's forward of the same microbatch; a backward waits for the next stage's backward.
 * With a forward of 1 and a backward of 2 the whole schedule takes 3(m + p − 1) units.
 */
export function oneForwardOneBackward(stages: number, microbatches: number): ScheduledOp[] {
  const p = Math.max(1, Math.floor(stages));
  const m = Math.max(1, Math.floor(microbatches));
  const orders: Array<Array<{ kind: "forward" | "backward"; microbatch: number }>> = [];
  for (let stage = 0; stage < p; stage += 1) {
    const warmup = Math.min(p - stage - 1, m);
    const order: Array<{ kind: "forward" | "backward"; microbatch: number }> = [];
    for (let j = 0; j < warmup; j += 1) order.push({ kind: "forward", microbatch: j });
    for (let k = 0; k < m - warmup; k += 1) {
      order.push({ kind: "forward", microbatch: warmup + k });
      order.push({ kind: "backward", microbatch: k });
    }
    for (let k = m - warmup; k < m; k += 1) order.push({ kind: "backward", microbatch: k });
    orders.push(order);
  }
  const forwardEnd = Array.from({ length: p }, () => new Array<number>(m).fill(Number.NaN));
  const backwardEnd = Array.from({ length: p }, () => new Array<number>(m).fill(Number.NaN));
  const next = new Array<number>(p).fill(0);
  const free = new Array<number>(p).fill(0);
  const ops: ScheduledOp[] = [];
  let remaining = orders.reduce((sum, order) => sum + order.length, 0);
  while (remaining > 0) {
    let progressed = false;
    for (let stage = 0; stage < p; stage += 1) {
      const op = orders[stage][next[stage]];
      if (!op) continue;
      const ready =
        op.kind === "forward"
          ? stage === 0
            ? 0
            : forwardEnd[stage - 1][op.microbatch]
          : stage === p - 1
            ? forwardEnd[stage][op.microbatch]
            : backwardEnd[stage + 1][op.microbatch];
      if (Number.isNaN(ready)) continue;
      const start = Math.max(free[stage], ready);
      const end = start + (op.kind === "forward" ? 1 : 2);
      (op.kind === "forward" ? forwardEnd : backwardEnd)[stage][op.microbatch] = end;
      free[stage] = end;
      ops.push({ stage, microbatch: op.microbatch, kind: op.kind, start, end });
      next[stage] += 1;
      remaining -= 1;
      progressed = true;
    }
    if (!progressed) break; // cannot happen for this schedule; guards against an infinite loop
  }
  return ops;
}

/* -------------------------------------------------------------------------- */
/* Failures and checkpoints                                                    */
/* -------------------------------------------------------------------------- */

/**
 * GPU-hours per unexpected interruption, from the Llama 3 paper's 54-day snapshot: 419 unexpected interruptions,
 * assuming all 16,384 GPUs ran for the whole period. 16,384 × 54 × 24 / 419 ≈ 50,677.
 */
export const LLAMA3_SNAPSHOT = { gpus: 16384, days: 54, unexpected: 419, total: 466, planned: 47 } as const;
export const GPU_HOURS_PER_INTERRUPTION = (LLAMA3_SNAPSHOT.gpus * LLAMA3_SNAPSHOT.days * 24) / LLAMA3_SNAPSHOT.unexpected;

/** Mean time between interruptions of the whole job, in hours: failures are independent, so the rate adds up. */
export const jobMtbfHours = (gpus: number) => GPU_HOURS_PER_INTERRUPTION / Math.max(1, gpus);

/** Young (1974): the first-order optimum interval √(2δM), with δ the save time and M the job's MTBF. */
export const youngInterval = (saveSeconds: number, mtbfSeconds: number) => Math.sqrt(2 * saveSeconds * mtbfSeconds);

/** Daly (2006): √(2δM)·[1 + ⅓√(δ/2M) + (δ/2M)/9] − δ while δ < 2M, and M beyond that. */
export function dalyInterval(saveSeconds: number, mtbfSeconds: number): number {
  if (saveSeconds >= 2 * mtbfSeconds) return mtbfSeconds;
  const x = saveSeconds / (2 * mtbfSeconds);
  return Math.sqrt(2 * saveSeconds * mtbfSeconds) * (1 + Math.sqrt(x) / 3 + x / 9) - saveSeconds;
}

/**
 * Share of wall-clock time lost, to first order: δ/τ spent saving, and per failure (one every M) half an interval
 * of work redone plus the restart. Its minimum over τ is Young's √(2δM).
 */
export const timeLost = (intervalSeconds: number, saveSeconds: number, mtbfSeconds: number, restartSeconds: number) =>
  saveSeconds / intervalSeconds + (intervalSeconds / 2 + restartSeconds) / mtbfSeconds;

/** A checkpoint that can resume a run: bf16 weights (2 bytes) and the fp32 master copy, momentum, and variance (12). */
export const CHECKPOINT_BYTES_PER_PARAMETER = 14;
export const checkpointBytes = (model: ModelShape) => CHECKPOINT_BYTES_PER_PARAMETER * parameterCounts(model).total;

export interface FailurePlan {
  mtbfHours: number;
  mtbfSeconds: number;
  youngSeconds: number;
  dalySeconds: number;
  lost: number;
  lostAtYoung: number;
  checkpointBytes: number;
  /** Bytes per second the whole job must write to save a checkpoint in the save time. */
  writeBytesPerSecond: number;
}

export function failurePlan(
  model: ModelShape,
  gpus: number,
  intervalMinutes: number,
  saveSeconds: number,
  restartMinutes: number,
): FailurePlan {
  const mtbfHours = jobMtbfHours(gpus);
  const mtbfSeconds = mtbfHours * 3600;
  const young = youngInterval(saveSeconds, mtbfSeconds);
  const restart = restartMinutes * 60;
  const bytes = checkpointBytes(model);
  return {
    mtbfHours,
    mtbfSeconds,
    youngSeconds: young,
    dalySeconds: dalyInterval(saveSeconds, mtbfSeconds),
    lost: timeLost(intervalMinutes * 60, saveSeconds, mtbfSeconds, restart),
    lostAtYoung: timeLost(young, saveSeconds, mtbfSeconds, restart),
    checkpointBytes: bytes,
    writeBytesPerSecond: bytes / saveSeconds,
  };
}

/* -------------------------------------------------------------------------- */
/* Formatting                                                                  */
/* -------------------------------------------------------------------------- */

/** Decimal units, as the ZeRO paper and NVIDIA's 80 GB use: 1 GB is 10⁹ bytes. */
export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes)) return "—";
  const units: ReadonlyArray<[number, string]> = [
    [1e12, "TB"],
    [1e9, "GB"],
    [1e6, "MB"],
    [1e3, "KB"],
  ];
  for (const [size, unit] of units) {
    if (bytes >= size) {
      const value = bytes / size;
      // One decimal throughout, so 128.5 GB and 2.0 GB read alike; a whole hundred drops its ".0" (165 GB).
      const text = value.toFixed(1);
      return `${value >= 100 ? text.replace(/\.0$/, "") : text} ${unit}`;
    }
  }
  return `${Math.round(bytes)} B`;
}

/** The memory limit as people say it: 80 GB, not 80.0 GB. */
export const MEMORY_LABEL = `${HARDWARE.memoryBytes / 1e9} GB`;

export function formatSeconds(seconds: number): string {
  if (!Number.isFinite(seconds)) return "—";
  if (seconds <= 0) return "0 s";
  if (seconds >= 100) return `${seconds.toFixed(0)} s`;
  if (seconds >= 10) return `${seconds.toFixed(1)} s`;
  if (seconds >= 1) return `${seconds.toFixed(2)} s`;
  if (seconds >= 0.01) return `${seconds.toFixed(3)} s`;
  return `${(seconds * 1000).toFixed(seconds >= 0.001 ? 1 : 2)} ms`;
}

export const formatPercent = (fraction: number, digits = 1) =>
  Number.isFinite(fraction) ? `${(fraction * 100).toFixed(digits)}%` : "—";

export function formatCount(value: number): string {
  if (!Number.isFinite(value)) return "—";
  if (value >= 1e12) return `${(value / 1e12).toFixed(2)} T`;
  if (value >= 1e9) return `${(value / 1e9).toFixed(value >= 100e9 ? 1 : 2)} B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(value >= 100e6 ? 0 : 1)} M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(value >= 100e3 ? 0 : 1)} k`;
  return `${Math.round(value)}`;
}

export function formatHours(hours: number): string {
  if (!Number.isFinite(hours)) return "—";
  if (hours >= 1000) return `${(hours / 24).toFixed(0)} days`;
  if (hours >= 100) return `${hours.toFixed(0)} h`;
  if (hours >= 1) return `${hours.toFixed(hours >= 10 ? 1 : 2)} h`;
  return `${(hours * 60).toFixed(hours * 60 >= 10 ? 0 : 1)} min`;
}

export function formatMinutes(seconds: number): string {
  if (!Number.isFinite(seconds)) return "—";
  const minutes = seconds / 60;
  if (minutes >= 600) return `${(minutes / 60).toFixed(0)} h`;
  if (minutes >= 100) return `${minutes.toFixed(0)} min`;
  return `${minutes.toFixed(1)} min`;
}
