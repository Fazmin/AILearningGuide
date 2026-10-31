import { describe, expect, it } from "vitest";
import definition from "./module";
import {
  ALL_TO_ALLS_PER_LAYER,
  checkpointBytes,
  CHECKPOINT_BYTES_PER_PARAMETER,
  CHECKPOINT_MINUTES,
  DATA_LADDER,
  EXPERT_LADDER,
  failurePlan,
  formatBytes,
  formatCount,
  formatHours,
  formatMinutes,
  formatPercent,
  formatSeconds,
  GPU_HOURS_PER_INTERRUPTION,
  HARDWARE,
  jobMtbfHours,
  LLAMA3_SNAPSHOT,
  memoryPerGpu,
  MEMORY_LABEL,
  MICRO_BATCH_LADDER,
  MICROBATCHES_LADDER,
  modelById,
  parameterCounts,
  PIPELINE_LADDER,
  RESTART_MINUTES,
  SAVE_SECONDS,
  SEQUENCE,
  stepTime,
  TENSOR_LADDER,
  tensorAllReducesPerLayer,
  timeLost,
  youngInterval,
  zeroModelState,
  type Layout,
  type ZeroStage,
} from "./parallel";
import { defaultState, LLAMA3_LAYOUT } from "./state";

/**
 * Pins every figure the parallel-training lessons (content/standard.mdx, plain.mdx, simple.mdx), card-info.ts, the
 * glossary, and the reference notes quote, computed the way Explore.tsx prints them. Everything is closed-form
 * arithmetic; nothing is timed.
 *
 * Figures the lessons take from sources rather than compute (Llama 3's 41% MFU, 400 TFLOP/s per GPU, 466 interruptions
 * and above-90% effective training time; Megatron-LM's 502 petaFLOP/s on 3,072 GPUs at 52%; MegaScale's 55.2% on
 * 12,288 GPUs; PaLM's 46.2% MFU and 57.8% HFU; Korthikanti et al.'s 54.2% and 42.1%; GPipe's four micro-batches per
 * stage; Llama 3's 2 and 7 TB/s of storage and its 131,072-token row at 38% MFU) are quoted from the papers and only
 * cross-checked here where the lab's own constants depend on them.
 */
const layout = (overrides: Partial<Layout> & { id?: string } = {}): Layout => {
  const { id, ...rest } = overrides;
  return {
    model: modelById(id ?? defaultState.modelId),
    tensor: defaultState.tensor,
    pipeline: defaultState.pipeline,
    data: defaultState.data,
    expert: defaultState.expert,
    zero: defaultState.zero,
    microBatch: defaultState.microBatch,
    microbatches: defaultState.microbatches,
    recompute: defaultState.recompute,
    overlap: defaultState.overlap,
    ...rest,
  };
};
const memory = (overrides: Partial<Layout> & { id?: string } = {}) => memoryPerGpu(layout(overrides));
const step = (overrides: Partial<Layout> & { id?: string } = {}) => stepTime(layout(overrides));
/** The "Split each layer" setup: 70B, pipeline 4, data 16, 16 microbatches of one sequence, ZeRO 1, Selective. */
const split = (tensor: number) => ({ id: "70b", tensor, pipeline: 4, data: 16, microbatches: 16, zero: 1 as ZeroStage });
/** The "Fill the pipeline" and "Pay for the sync" setup: 70B, tensor 8, pipeline 8, data 16, ZeRO 1. */
const deep = (microbatches: number) => ({ id: "70b", tensor: 8, pipeline: 8, data: 16, microbatches, zero: 1 as ZeroStage });
const preset = { ...LLAMA3_LAYOUT, id: LLAMA3_LAYOUT.modelId, zero: LLAMA3_LAYOUT.zero as ZeroStage };
const percent = (value: number) => formatPercent(value);

describe("parallel-training quoted values: the hardware and the shapes", () => {
  it("states the H100 assumptions the lesson and the reference notes quote", () => {
    // standard.mdx "What the lab shows", card-info.ts, NVIDIA reference note: 989.5 is half of 1,979; 450 is half of 900.
    expect(HARDWARE.peakFlops).toBe(1979e12 / 2);
    expect(HARDWARE.nvlinkBytesPerSecond).toBe(900e9 / 2);
    // 50 GB/s is one 400 Gb/s link per GPU (Llama 3 paper, DGX H100); "nine times slower" (glossary: "ninefold gap").
    expect(HARDWARE.networkBytesPerSecond).toBe(400e9 / 8);
    expect(HARDWARE.nvlinkBytesPerSecond / HARDWARE.networkBytesPerSecond).toBe(9);
    expect(HARDWARE.memoryBytes).toBe(80e9);
    expect(MEMORY_LABEL).toBe("80 GB");
    expect(HARDWARE.gpusPerNode).toBe(8);
    expect(HARDWARE.efficiency).toBe(0.75);
    expect(SEQUENCE).toBe(8192);
  });

  it("matches the parameter counts and Mixtral's active share", () => {
    // card-info.ts "Parallel layout": 8.03 B, 70.55 B, 405.85 B; 46.70 B with 12.88 B active, 2 of 8 experts.
    expect(["8b", "70b", "405b", "mixtral"].map((id) => (parameterCounts(modelById(id)).total / 1e9).toFixed(2))).toEqual([
      "8.03",
      "70.55",
      "405.85",
      "46.70",
    ]);
    expect((parameterCounts(modelById("mixtral")).active / 1e9).toFixed(2)).toBe("12.88");
    expect([modelById("mixtral").experts, modelById("mixtral").topK]).toEqual([8, 2]);
    // Mixtral reference note: "about 13 billion of 47 billion parameters".
    expect(Math.round(parameterCounts(modelById("mixtral")).active / 1e9)).toBe(13);
    expect(Math.round(parameterCounts(modelById("mixtral")).total / 1e9)).toBe(47);
    // standard.mdx "The big picture": a 405-billion-parameter model needs 6.5 TB for its model state.
    expect(formatBytes(16 * parameterCounts(modelById("405b")).total)).toBe("6.5 TB");
  });

  it("matches the control ranges card-info.ts lists", () => {
    expect([TENSOR_LADDER[0], TENSOR_LADDER.at(-1)]).toEqual([1, 32]);
    expect([PIPELINE_LADDER[0], PIPELINE_LADDER.at(-1)]).toEqual([1, 16]);
    expect([DATA_LADDER[0], DATA_LADDER.at(-1)]).toEqual([1, 2048]);
    expect([EXPERT_LADDER[0], EXPERT_LADDER.at(-1)]).toEqual([1, 8]);
    expect([MICRO_BATCH_LADDER[0], MICRO_BATCH_LADDER.at(-1)]).toEqual([1, 8]);
    expect([MICROBATCHES_LADDER[0], MICROBATCHES_LADDER.at(-1)]).toEqual([1, 128]);
    expect([CHECKPOINT_MINUTES.min, CHECKPOINT_MINUTES.max]).toEqual([5, 120]);
    expect([SAVE_SECONDS.min, SAVE_SECONDS.max]).toEqual([5, 600]);
    expect([RESTART_MINUTES.min, RESTART_MINUTES.max]).toEqual([1, 60]);
  });

  it("states the per-layer and per-step counts the lesson and the glossary use", () => {
    // Four all-reduces per layer per microbatch, six with full recomputation; four all-to-alls per MoE layer.
    expect([tensorAllReducesPerLayer("selective"), tensorAllReducesPerLayer("full")]).toEqual([4, 6]);
    expect(ALL_TO_ALLS_PER_LAYER).toBe(4);
    // 14 bytes per parameter in a checkpoint: 2 of bf16 weights and 12 of fp32 optimizer state.
    expect(CHECKPOINT_BYTES_PER_PARAMETER).toBe(2 + 12);
  });
});

describe("parallel-training quoted values: Shard the state", () => {
  it("matches the default layout at every ZeRO stage", () => {
    // standard.mdx / plain.mdx / simple.mdx / card-info.ts / checkpoint question 2.
    const z0 = memory({ zero: 0 });
    expect([z0.weights, z0.gradients, z0.optimizer, z0.modelState, z0.activations, z0.total].map(formatBytes)).toEqual([
      "16.1 GB",
      "16.1 GB",
      "96.4 GB",
      "128.5 GB",
      "36.5 GB",
      "165 GB",
    ]);
    expect(z0.fits).toBe(false);
    const z1 = memory({ zero: 1 });
    expect([z1.optimizer, z1.modelState, z1.total].map(formatBytes)).toEqual(["1.5 GB", "33.6 GB", "70.1 GB"]);
    expect(z1.fits).toBe(true);
    const z2 = memory({ zero: 2 });
    expect([z2.gradients, z2.modelState].map(formatBytes)).toEqual(["250.9 MB", "17.8 GB"]);
    const z3 = memory({ zero: 3 });
    expect([z3.modelState, z3.total].map(formatBytes)).toEqual(["2.0 GB", "38.5 GB"]);
    // plain.mdx / simple.mdx worked example: 4 × 8.03 B = 32.1 GB, plus 12 × 8.03 B ÷ 64 = 1.5 GB, is 33.6 GB.
    const psi = parameterCounts(modelById("8b")).total;
    expect(formatBytes(4 * psi)).toBe("32.1 GB");
    expect(formatBytes((12 * psi) / 64)).toBe("1.5 GB");
    // "more than twice the 80 GB it has".
    expect(z0.total).toBeGreaterThan(2 * HARDWARE.memoryBytes);
  });

  it("reproduces the ZeRO paper's Figure 1 quoted in the lesson, card-info.ts, and the reference note", () => {
    expect(([0, 1, 2, 3] as ZeroStage[]).map((stage) => Number((zeroModelState(7.5e9, 64, stage) / 1e9).toFixed(1)))).toEqual([
      120, 31.4, 16.6, 1.9,
    ]);
  });
});

describe("parallel-training quoted values: Count the activations", () => {
  it("matches the activation sizes at ZeRO stage 3", () => {
    // standard.mdx / plain.mdx / simple.mdx / checkpoint question 3.
    expect(formatBytes(memory({ zero: 3 }).activations)).toBe("36.5 GB");
    const four = memory({ zero: 3, microBatch: 4 });
    expect([four.activations, four.total].map(formatBytes)).toEqual(["146 GB", "148 GB"]);
    expect(four.fits).toBe(false);
    expect(formatBytes(memory({ zero: 3, microBatch: 4, recompute: "none" }).activations)).toBe("1.5 TB");
    const full = memory({ zero: 3, microBatch: 4, recompute: "full" });
    expect([full.activations, full.total].map(formatBytes)).toEqual(["8.6 GB", "10.6 GB"]);
    expect(full.fits).toBe(true);
    // "the attention-score term 5as/h is 320 against 34" for the 8B model at 8,192 tokens, so None is ten times Selective.
    const model = modelById("8b");
    expect((5 * model.heads * SEQUENCE) / model.hidden).toBe(320);
    expect(Math.round(memory({ recompute: "none" }).activations / memory({ recompute: "selective" }).activations)).toBe(10);
  });

  it("matches the price of full recomputation in Step time and MFU", () => {
    const selective = step({ zero: 3, microBatch: 4 });
    const full = step({ zero: 3, microBatch: 4, recompute: "full" });
    expect(formatSeconds(full.parts.compute)).toBe("21.6 s");
    expect(formatSeconds(full.parts.recompute)).toBe("7.19 s");
    expect([selective.mfu, full.mfu, full.hfu].map((value) => percent(value))).toEqual(["75.0%", "56.3%", "75.0%"]);
    // "a quarter of their arithmetic redoes the forward pass".
    expect(full.parts.recompute / (full.parts.compute + full.parts.recompute)).toBeCloseTo(0.25, 12);
  });

  it("matches the MFU and HFU of card-info.ts and checkpoint question 7 at one sequence per microbatch", () => {
    const selective = step({ zero: 3 });
    const full = step({ zero: 3, recompute: "full" });
    expect([selective.mfu, full.mfu, full.hfu].map((value) => percent(value))).toEqual(["68.6%", "53.6%", "71.5%"]);
  });
});

describe("parallel-training quoted values: Split each layer", () => {
  it("matches memory, TP time, and MFU at tensor 1, 8, and 16", () => {
    expect(formatBytes(memory(split(1)).total)).toBe("266.3 GB");
    expect(memory(split(1)).fits).toBe(false);
    const eight = step(split(8));
    const sixteen = step(split(16));
    expect(formatBytes(memory(split(8)).total)).toBe("33.3 GB");
    expect(formatSeconds(eight.tensor.seconds)).toBe("0.042 s");
    expect(formatSeconds(eight.computePerMicrobatch)).toBe("0.168 s");
    expect(eight.tensor.link.label).toBe("NVLink");
    expect(percent(eight.mfu)).toBe("49.8%");
    expect(formatBytes(memory(split(16)).total)).toBe("16.6 GB");
    expect(formatSeconds(sixteen.tensor.seconds)).toBe("0.403 s");
    expect(sixteen.tensor.link.label).toBe("InfiniBand");
    expect((sixteen.tensor.seconds / eight.tensor.seconds).toFixed(1)).toBe("9.6");
    expect(percent(sixteen.mfu)).toBe("10.9%");
    // "halves its memory per GPU" (standard.mdx "Why it matters") and "Parameters per GPU halves" (card-info.ts).
    expect(memory(split(16)).total / memory(split(8)).total).toBeCloseTo(0.5, 9);
    expect(memory(split(16)).sliceParameters / memory(split(8)).sliceParameters).toBeCloseTo(0.5, 12);
    // plain.mdx / simple.mdx "Why it matters": doubling the GPUs makes each step more than twice as slow.
    expect(sixteen.gpus).toBe(2 * eight.gpus);
    expect(sixteen.step / eight.step).toBeGreaterThan(2);
    // standard.mdx "What to notice" / plain.mdx / simple.mdx: MFU 56.0% at tensor 4.
    expect(percent(step(split(4)).mfu)).toBe("56.0%");
    // The 15.8% bubble of this layout, 3 / 19.
    expect(percent(eight.bubbleFraction)).toBe("15.8%");
  });

  it("matches the worked all-reduce in plain.mdx and simple.mdx", () => {
    const eight = step(split(8)).tensor;
    // A 134 MB block; 2 × 7 ÷ 8 of it, about 235 MB, in about half a thousandth of a second; 80 per microbatch.
    expect(formatBytes(eight.bytesPerCall)).toBe("134.2 MB");
    expect(Math.round(((2 * 7) / 8) * eight.bytesPerCall / 1e6)).toBe(235);
    expect((eight.seconds / eight.calls) * 1000).toBeCloseTo(0.5, 1);
    expect(eight.calls).toBe(80);
    // With 16 GPUs: 2 × 15 ÷ 16 of the block at 50 GB/s, about 5 thousandths of a second each.
    const sixteen = step(split(16)).tensor;
    expect(Math.round((sixteen.seconds / sixteen.calls) * 1000)).toBe(5);
  });
});

describe("parallel-training quoted values: Fill the pipeline", () => {
  it("matches the bubble and MFU as microbatches grow", () => {
    expect([1, 2, 8, 32, 128].map((m) => percent(step(deep(m)).bubbleFraction))).toEqual(["87.5%", "77.8%", "46.7%", "17.9%", "5.2%"]);
    expect(percent(step(deep(1)).mfu)).toBe("7.2%");
    expect(percent(step(deep(32)).mfu)).toBe("48.7%");
    expect(formatCount(step(deep(32)).tokensPerStep)).toBe("4.2 M");
    // plain.mdx: 7 ÷ 8 and 7 ÷ 39.
    expect(step(deep(1)).bubbleFraction).toBe(7 / 8);
    expect(step(deep(32)).bubbleFraction).toBe(7 / 39);
  });
});

describe("parallel-training quoted values: Pay for the sync", () => {
  it("matches the sync, its exposed part, MFU, and tokens per second", () => {
    const start = step(deep(32));
    expect(formatSeconds(start.dataSync.seconds)).toBe("0.083 s");
    expect(formatSeconds(start.exposedDataSync)).toBe("0.027 s");
    expect(formatCount(start.tokensPerSecond)).toBe("1.0 M");
    const wide = step({ ...deep(32), data: 256 });
    expect(wide.gpus).toBe(16384);
    expect(formatCount(wide.tokensPerSecond)).toBe("16.2 M");
    expect([start.mfu, wide.mfu].map((value) => percent(value))).toEqual(["48.7%", "48.6%"]);
    expect(formatSeconds(wide.dataSync.seconds)).toBe("0.088 s");
    expect(formatSeconds(wide.exposedDataSync)).toBe("0.032 s");
    const off = step({ ...deep(32), data: 256, overlap: false });
    expect(formatSeconds(off.exposedDataSync)).toBe("0.088 s");
    expect(percent(off.mfu)).toBe("48.0%");
    const zero3 = step({ ...deep(32), data: 256, overlap: false, zero: 3 });
    expect(formatSeconds(zero3.exposedDataSync)).toBe("0.132 s");
    expect(percent(zero3.mfu)).toBe("47.5%");
  });

  it("matches the node-boundary example on the 8B model at ZeRO stage 2 with one microbatch", () => {
    // standard.mdx / plain.mdx / simple.mdx "What to notice".
    const inside = step({ data: 8, microbatches: 1, zero: 2 });
    const across = step({ data: 16, microbatches: 1, zero: 2 });
    expect(inside.dataSync.link.label).toBe("NVLink");
    expect(formatSeconds(inside.dataSync.seconds)).toBe("0.062 s");
    expect(inside.exposedDataSync).toBe(0);
    expect(across.dataSync.link.label).toBe("InfiniBand");
    expect(formatSeconds(across.dataSync.seconds)).toBe("0.602 s");
    expect(formatSeconds(across.exposedDataSync)).toBe("0.153 s");
    expect([inside.mfu, across.mfu].map((value) => percent(value))).toEqual(["75.0%", "61.1%"]);
    // Both layouts fit, so the example is one the lab can show.
    expect(memory({ data: 8, microbatches: 1, zero: 2 }).fits).toBe(true);
  });
});

describe("parallel-training quoted values: the experts", () => {
  it("matches Mixtral's memory and all-to-all figures", () => {
    // standard.mdx / plain.mdx / simple.mdx "What to notice".
    expect(formatBytes(memory({ id: "mixtral", zero: 1, expert: 1 }).modelState)).toBe("195.6 GB");
    expect(formatBytes(memory({ id: "mixtral", zero: 1, expert: 8 }).modelState)).toBe("37.7 GB");
    const eight = step({ id: "mixtral", zero: 1, expert: 8 });
    expect(formatSeconds(eight.expertAllToAll.seconds)).toBe("0.033 s");
    expect(eight.expertAllToAll.link.label).toBe("NVLink");
    const crossed = step({ id: "mixtral", zero: 1, expert: 8, tensor: 2 });
    expect(formatSeconds(crossed.expertAllToAll.seconds)).toBe("0.301 s");
    expect(crossed.expertAllToAll.link.label).toBe("InfiniBand");
  });
});

describe("parallel-training quoted values: the 16,384-GPU layout", () => {
  it("matches its memory, bubble, and MFU", () => {
    const fits = memory(preset);
    expect(formatBytes(fits.total)).toBe("79.7 GB");
    expect(fits.fits).toBe(true);
    // "fits with 0.3 GB to spare".
    expect(formatBytes(fits.headroom)).toBe("297.4 MB");
    expect(Number((fits.headroom / 1e9).toFixed(1))).toBe(0.3);
    const timing = step(preset);
    expect(percent(timing.bubbleFraction)).toBe("48.4%");
    expect(timing.bubbleFraction).toBe(15 / 31);
    expect(percent(timing.mfu)).toBe("33.2%");
    expect(timing.gpus).toBe(16384);
    // "None would charge 760 GB of activations per GPU".
    expect(Math.round(memory({ ...preset, recompute: "none" }).activations / 1e9)).toBe(760);
    // "a 5.7 TB checkpoint".
    expect(formatBytes(checkpointBytes(modelById("405b")))).toBe("5.7 TB");
    // Llama 3, Table 4: tensor 8, pipeline 16, data 128, 16 sequences of 8,192 per data-parallel rank: 16.8 M tokens.
    expect([LLAMA3_LAYOUT.tensor, LLAMA3_LAYOUT.pipeline, LLAMA3_LAYOUT.data]).toEqual([8, 16, 128]);
    expect(formatCount(timing.tokensPerStep)).toBe("16.8 M");
    // standard.mdx "Context parallelism": 8 × 16 × 16 × 8 still makes 16,384 GPUs.
    expect(8 * 16 * 16 * 8).toBe(16384);
  });
});

describe("parallel-training quoted values: failures and checkpoints", () => {
  const plan = (gpus: number, minutes: number) => failurePlan(modelById("405b"), gpus, minutes, 60, 10);

  it("matches the failure rate derived from the Llama 3 snapshot", () => {
    expect(LLAMA3_SNAPSHOT.planned + LLAMA3_SNAPSHOT.unexpected).toBe(LLAMA3_SNAPSHOT.total);
    expect([LLAMA3_SNAPSHOT.total, LLAMA3_SNAPSHOT.unexpected, LLAMA3_SNAPSHOT.days]).toEqual([466, 419, 54]);
    expect(16384 * 54 * 24).toBe(21_233_664);
    expect(Math.round(GPU_HOURS_PER_INTERRUPTION)).toBe(50_677);
    expect(formatHours(jobMtbfHours(16384))).toBe("3.09 h");
    expect(formatHours(jobMtbfHours(512))).toBe("99.0 h");
    expect(jobMtbfHours(512) / jobMtbfHours(16384)).toBeCloseTo(32, 9);
  });

  it("matches the optima and the time lost at 16,384 and 512 GPUs", () => {
    const big = plan(16384, 60);
    expect([formatMinutes(big.youngSeconds), formatMinutes(big.dalySeconds)]).toEqual(["19.3 min", "18.6 min"]);
    expect([5, 19, 120].map((minutes) => percent(plan(16384, minutes).lost))).toEqual(["26.7%", "15.8%", "38.6%"]);
    // The lowest whole-minute Time lost on the slider is at 19 minutes.
    let best = CHECKPOINT_MINUTES.min;
    for (let minutes = CHECKPOINT_MINUTES.min; minutes <= CHECKPOINT_MINUTES.max; minutes += 1) {
      if (plan(16384, minutes).lost < plan(16384, best).lost) best = minutes;
    }
    expect(best).toBe(19);
    expect(formatMinutes(plan(512, 60).youngSeconds)).toBe("109 min");
    // The default interval, 60 minutes, reads 23.2% on the preset.
    expect(percent(plan(16384, defaultState.checkpointMinutes).lost)).toBe("23.2%");
  });

  it("matches the worked Young interval and time lost in plain.mdx and simple.mdx", () => {
    const mtbf = jobMtbfHours(16384) * 3600;
    expect(Math.round(mtbf)).toBe(11_135);
    expect(Math.round((2 * 60 * mtbf) / 1000) * 1000).toBe(1_336_000);
    expect(Math.round(youngInterval(60, mtbf))).toBe(1_156);
    // At 19 minutes: 60 of every 1,140 seconds saving (5.3%), and (9.5 + 10) minutes every 185.6 minutes (10.5%).
    expect(formatPercent(60 / 1140)).toBe("5.3%");
    expect((mtbf / 60).toFixed(1)).toBe("185.6");
    expect(formatPercent((9.5 + 10) / (mtbf / 60))).toBe("10.5%");
    expect(formatPercent(timeLost(19 * 60, 60, mtbf, 600))).toBe("15.8%");
  });
});

describe("parallel-training quoted values: going deeper", () => {
  it("matches the reconciliation of the two activation counts", () => {
    // ZeRO counts about 12 two-byte values per token per hidden unit per layer: 24 bytes. Korthikanti counts 34
    // before the attention scores, and 2 with full recomputation. Scaled from ZeRO's 60 GB: about 85 GB and 5 GB.
    expect(12 * 2).toBe(24);
    expect(Math.round((60 * 34) / 24)).toBe(85);
    expect((60 * 2) / 24).toBe(5);
  });

  it("matches the glossary's per-job MTBF and the interleaved bubble", () => {
    // Glossary "Mean time between failures": about 3 hours on 16,384 H100s.
    expect(Math.round(jobMtbfHours(16384))).toBe(3);
    // Narayanan et al.: (p − 1)/m against the ideal time is the same idle time as the lab's (p − 1)/(m + p − 1).
    const p = 8;
    const m = 32;
    const againstIdeal = (p - 1) / m;
    expect(againstIdeal / (1 + againstIdeal)).toBeCloseTo((p - 1) / (m + p - 1), 12);
  });

  it("keeps the module's own text in step with the pinned figures", () => {
    const questions = JSON.stringify(definition.checkpoint);
    for (const figure of ["250.9 MB", "128.5 GB", "146 GB", "8.6 GB", "1.5 TB", "0.042 s", "0.403 s", "87.5%", "17.9%", "33.2%", "68.6%", "53.6%", "71.5%", "0.032 s", "0.088 s", "3.09 hours", "99.0 hours", "19.3 minutes"]) {
      expect(questions, figure).toContain(figure);
    }
  });
});
