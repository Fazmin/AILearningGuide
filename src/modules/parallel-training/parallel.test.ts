import { describe, expect, it } from "vitest";
import { teachingMemory, teachingServing } from "@app/module-sdk";
import {
  activationBytesPerLayer,
  allGatherSeconds,
  allReduceSeconds,
  allToAllSeconds,
  checkpointBytes,
  dalyInterval,
  effectiveExpert,
  flopsPerToken,
  GPU_HOURS_PER_INTERRUPTION,
  HARDWARE,
  jobMtbfHours,
  linkFor,
  memoryPerGpu,
  modelById,
  MODELS,
  oneForwardOneBackward,
  parameterCounts,
  reduceScatterSeconds,
  SEQUENCE,
  snapToLadder,
  stepTime,
  timeLost,
  youngInterval,
  zeroModelState,
  type Layout,
  type ZeroStage,
} from "./parallel";

const layout = (overrides: Partial<Layout> & { id?: string } = {}): Layout => {
  const { id, ...rest } = overrides;
  return {
    model: modelById(id ?? "8b"),
    tensor: 1,
    pipeline: 1,
    data: 64,
    expert: 1,
    zero: 0,
    microBatch: 1,
    microbatches: 8,
    recompute: "selective",
    overlap: true,
    ...rest,
  };
};
const GB = 1e9;

describe("parameter counts from the published shapes", () => {
  it("reproduces the Llama 3.1 sizes exactly from Ψ = L(2h² + 2h·kv·h/a + 3hf) + 2Vh", () => {
    expect(parameterCounts(modelById("8b")).total).toBe(8_029_995_008);
    expect(parameterCounts(modelById("70b")).total).toBe(70_552_387_584);
    expect(parameterCounts(modelById("405b")).total).toBe(405_849_243_648);
    // Rounded the way model cards round them: 8.030 B, 70.552 B, 405.849 B.
    expect(MODELS.slice(0, 3).map((model) => Number((parameterCounts(model).total / 1e9).toFixed(3)))).toEqual([8.03, 70.552, 405.849]);
  });

  it("counts Mixtral's 8 experts for memory and only 2 per token for compute", () => {
    const mixtral = parameterCounts(modelById("mixtral"));
    expect(mixtral.total).toBe(46_701_477_888);
    expect(mixtral.active).toBe(12_878_610_432);
    expect(Number((mixtral.total / 1e9).toFixed(2))).toBe(46.7);
    expect(Number((mixtral.active / 1e9).toFixed(2))).toBe(12.88);
    expect(mixtral.dense + mixtral.expert).toBe(mixtral.total);
    // Dense models put the FFN in the dense part, so active = total and nothing is expert.
    for (const id of ["8b", "70b", "405b"]) {
      const counts = parameterCounts(modelById(id));
      expect(counts.active).toBe(counts.total);
      expect(counts.expert).toBe(0);
    }
  });

  it("agrees with the serving lab's Llama-3.1-8B preset in the shared SDK", () => {
    const serving = teachingServing.MODELS.find((model) => model.shape === "Llama-3.1-8B")!;
    const ours = modelById("8b");
    expect(serving).toBeDefined();
    expect(Number((parameterCounts(ours).total / 1e9).toFixed(2))).toBe(serving.params / 1e9);
    expect(ours.layers).toBe(serving.layers);
    expect(ours.kvHeads).toBe(serving.kvHeads);
    expect(ours.hidden / ours.heads).toBe(serving.headDim);
    // The same shape gives the serving lab's 128 KiB of KV cache per token: keys and values, every layer, fp16.
    expect(2 * ours.layers * ours.kvHeads * (ours.hidden / ours.heads) * 2).toBe(teachingServing.kvBytesPerToken(serving));
    expect(teachingServing.kvBytesPerToken(serving)).toBe(128 * 1024);
  });
});

describe("model state per GPU", () => {
  it("reproduces the ZeRO paper's Figure 1: 7.5 B parameters on 64 GPUs", () => {
    const figure = ([0, 1, 2, 3] as ZeroStage[]).map((stage) => Number((zeroModelState(7.5e9, 64, stage) / GB).toFixed(1)));
    expect(figure).toEqual([120, 31.4, 16.6, 1.9]);
  });

  it("uses the shared training-memory arithmetic for the ZeRO-0 bytes: 2 + 2 + 12 = 16 per parameter", () => {
    const shared = teachingMemory.memoryBreakdown(1, "mixed", "adam");
    expect([shared.weightBytesPerParameter, shared.gradientBytesPerParameter, shared.stateBytesPerParameter]).toEqual([2, 2, 12]);
    const memory = memoryPerGpu(layout({ zero: 0 }));
    const psi = parameterCounts(modelById("8b")).total;
    expect(memory.weights).toBe(2 * psi);
    expect(memory.gradients).toBe(2 * psi);
    expect(memory.optimizer).toBe(12 * psi);
    expect(memory.modelState).toBe(teachingMemory.memoryBreakdown(psi, "mixed", "adam").total);
  });

  it("matches the closed-form stage formulas for a dense model on any layout", () => {
    for (const id of ["8b", "70b", "405b"]) {
      for (const [tensor, pipeline, data] of [
        [1, 1, 64],
        [8, 4, 16],
        [8, 16, 128],
      ]) {
        for (const zero of [0, 1, 2, 3] as ZeroStage[]) {
          const memory = memoryPerGpu(layout({ id, tensor, pipeline, data, zero }));
          const slice = parameterCounts(modelById(id)).total / (tensor * pipeline);
          expect(memory.modelState).toBeCloseTo(zeroModelState(slice, data, zero), 0);
        }
      }
    }
  });

  it("shards each stage's part and nothing else", () => {
    const at = (zero: ZeroStage) => memoryPerGpu(layout({ zero }));
    expect(at(1).optimizer).toBeCloseTo(at(0).optimizer / 64, 0);
    expect(at(1).gradients).toBe(at(0).gradients);
    expect(at(2).gradients).toBeCloseTo(at(0).gradients / 64, 0);
    expect(at(2).weights).toBe(at(0).weights);
    expect(at(3).weights).toBeCloseTo(at(0).weights / 64, 0);
    for (const zero of [0, 1, 2, 3] as ZeroStage[]) expect(at(zero).activations).toBe(at(0).activations);
  });

  it("shards Mixtral's experts over the d ÷ e GPUs that hold the same expert", () => {
    const counts = parameterCounts(modelById("mixtral"));
    // At ZeRO 3 every byte of state is spread over all N GPUs exactly once.
    const z3 = layout({ id: "mixtral", data: 64, expert: 8, zero: 3 });
    expect(memoryPerGpu(z3).modelState).toBeCloseTo((16 * counts.total) / 64, 0);
    // With no sharding each GPU holds the dense part and 1/e of the experts.
    const z0 = memoryPerGpu(layout({ id: "mixtral", data: 64, expert: 8, zero: 0 }));
    expect(z0.sliceParameters).toBeCloseTo(counts.dense + counts.expert / 8, 0);
    // Expert parallel is capped by data parallel and means nothing for a dense model.
    expect(effectiveExpert(layout({ id: "mixtral", data: 4, expert: 8 }))).toBe(4);
    expect(effectiveExpert(layout({ id: "8b", expert: 8 }))).toBe(1);
  });
});

describe("activations (Korthikanti et al., Table 2)", () => {
  const s = SEQUENCE;

  it("charges sbh(34 + 5as/h)/t with none, 34sbh/t with selective, and 2sbh with full recomputation", () => {
    const model = modelById("8b");
    const sbh = s * 1 * model.hidden;
    expect(activationBytesPerLayer(model, 1, 1, "none")).toBe(sbh * (34 + (5 * model.heads * s) / model.hidden));
    expect(activationBytesPerLayer(model, 1, 1, "none")).toBe(11_878_268_928);
    expect(activationBytesPerLayer(model, 1, 1, "selective")).toBe(34 * sbh);
    expect(activationBytesPerLayer(model, 1, 1, "full")).toBe(2 * sbh);
    const big = modelById("70b");
    expect(activationBytesPerLayer(big, 1, 8, "selective")).toBe((34 * s * big.hidden) / 8);
    expect(activationBytesPerLayer(big, 1, 8, "none")).toBe((s * big.hidden * (34 + (5 * big.heads * s) / big.hidden)) / 8);
    // Full recomputation keeps each layer's whole input: the table's sbh·2 is not divided by t.
    expect(activationBytesPerLayer(big, 1, 8, "full")).toBe(2 * s * big.hidden);
    expect(activationBytesPerLayer(big, 1, 1, "full")).toBe(activationBytesPerLayer(big, 1, 8, "full"));
  });

  it("holds ⌈L/p⌉ layers for min(m, p) microbatches on the first stage", () => {
    const memory = memoryPerGpu(layout({ id: "405b", tensor: 8, pipeline: 16, data: 128, microbatches: 16, zero: 2 }));
    expect(memory.layersPerStage).toBe(8); // 126 / 16 = 7.875
    expect(memory.inFlight).toBe(16);
    expect(memory.activations).toBe(8 * 16 * activationBytesPerLayer(modelById("405b"), 1, 8, "selective"));
    expect(memoryPerGpu(layout({ pipeline: 8, microbatches: 3 })).inFlight).toBe(3);
  });

  it("grows with the micro-batch size and shrinks with each recompute mode", () => {
    for (const model of MODELS) {
      for (const t of [1, 8]) {
        const none = activationBytesPerLayer(model, 2, t, "none");
        const selective = activationBytesPerLayer(model, 2, t, "selective");
        const full = activationBytesPerLayer(model, 2, t, "full");
        expect(none).toBeGreaterThan(selective);
        expect(selective).toBeGreaterThan(full);
        expect(activationBytesPerLayer(model, 4, t, "selective")).toBe(2 * selective);
      }
    }
  });
});

describe("collectives and links", () => {
  it("prices ring collectives by their bandwidth terms", () => {
    const bytes = 1e9;
    const bandwidth = 50e9;
    expect(allReduceSeconds(bytes, 8, bandwidth)).toBeCloseTo((2 * 7) / 8 / 50, 12);
    for (const collective of [reduceScatterSeconds, allGatherSeconds, allToAllSeconds]) {
      expect(collective(bytes, 8, bandwidth)).toBeCloseTo(7 / 8 / 50, 12);
      expect(collective(bytes, 1, bandwidth)).toBe(0);
    }
    expect(allReduceSeconds(bytes, 1, bandwidth)).toBe(0);
    // A reduce-scatter plus an all-gather moves exactly what an all-reduce moves.
    expect(reduceScatterSeconds(bytes, 64, bandwidth) + allGatherSeconds(bytes, 64, bandwidth)).toBeCloseTo(allReduceSeconds(bytes, 64, bandwidth), 12);
  });

  it("keeps a group on NVLink only while it fits in one 8-GPU node", () => {
    expect(linkFor(1, 8).id).toBe("nvlink");
    expect(linkFor(1, 16).id).toBe("network");
    expect(linkFor(2, 4).id).toBe("nvlink");
    expect(linkFor(2, 8).id).toBe("network");
    expect(linkFor(1, 8).bytesPerSecond / linkFor(1, 16).bytesPerSecond).toBe(9);
  });

  it("makes the tensor all-reduce about ten times slower when the group leaves the node at t = 16", () => {
    const at = (tensor: number) => stepTime(layout({ id: "70b", tensor, pipeline: 4, data: 16, microbatches: 16, zero: 1 })).tensor;
    expect(at(8).link.id).toBe("nvlink");
    expect(at(16).link.id).toBe("network");
    // The buffer is 2bsh at every t: tensor parallelism does not shrink what it all-reduces.
    expect(at(16).bytesPerCall).toBe(at(8).bytesPerCall);
    expect(at(16).seconds / at(8).seconds).toBeCloseTo((450 / 50) * ((15 / 16) / (7 / 8)), 9);
    // Four all-reduces per layer, six with full recomputation.
    expect(at(8).calls).toBe(4 * 20);
    const full = stepTime(layout({ id: "70b", tensor: 8, pipeline: 4, data: 16, microbatches: 16, zero: 1, recompute: "full" })).tensor;
    expect(full.calls).toBe(6 * 20);
  });

  it("moves 1.5 times the data-parallel bytes at ZeRO stage 3 and the same at stages 0 to 2", () => {
    const sync = (zero: ZeroStage) => stepTime(layout({ zero })).dataSync.seconds;
    expect(sync(1)).toBe(sync(0));
    expect(sync(2)).toBe(sync(0));
    expect(sync(3)).toBeCloseTo(1.5 * sync(0), 12);
    // An all-reduce of the bf16 gradients of this GPU's slice over the d replicas.
    const psi = parameterCounts(modelById("8b")).total;
    expect(sync(0)).toBeCloseTo(allReduceSeconds(2 * psi, 64, HARDWARE.networkBytesPerSecond), 12);
  });

  it("crosses the node for expert all-to-alls once tensor × expert passes 8", () => {
    const at = (tensor: number) => stepTime(layout({ id: "mixtral", tensor, data: 64, expert: 8, zero: 1 })).expertAllToAll;
    expect(at(1).link.id).toBe("nvlink");
    expect(at(2).link.id).toBe("network");
    expect(at(1).calls).toBe(4 * 32);
    expect(at(1).bytesPerCall).toBe(2 * SEQUENCE * 2 * 4096);
    expect(stepTime(layout({ id: "8b", expert: 8 })).expertAllToAll.seconds).toBe(0);
  });
});

describe("step time", () => {
  it("adds compute, traffic, bubble, and exposed sync up to the step", () => {
    for (const config of [
      layout(),
      layout({ id: "70b", tensor: 8, pipeline: 8, data: 16, microbatches: 32, zero: 1 }),
      layout({ id: "405b", tensor: 8, pipeline: 16, data: 128, microbatches: 16, zero: 2 }),
      layout({ id: "mixtral", tensor: 2, pipeline: 2, data: 32, expert: 8, zero: 3, recompute: "full", overlap: false }),
    ]) {
      const step = stepTime(config);
      const sum = Object.values(step.parts).reduce((total, part) => total + part, 0);
      expect(sum).toBeCloseTo(step.step, 9);
      expect(step.step).toBeCloseTo((config.microbatches + config.pipeline - 1) * step.microbatchTime + step.exposedDataSync, 9);
      expect(step.tokensPerStep).toBe(config.data * config.microbatches * config.microBatch * SEQUENCE);
      expect(step.mfu).toBeLessThanOrEqual(HARDWARE.efficiency + 1e-12);
    }
  });

  it("computes t_c = (6Ψactive + 12Lhs)·b·s·r / (t·p·η·F) and MFU against the peak", () => {
    const config = layout({ id: "70b", tensor: 8, pipeline: 4, data: 16, microbatches: 16, zero: 1 });
    const step = stepTime(config);
    const model = modelById("70b");
    const perToken = 6 * parameterCounts(model).active + 12 * model.layers * model.hidden * SEQUENCE;
    expect(flopsPerToken(model)).toBe(perToken);
    expect(step.computePerMicrobatch).toBeCloseTo((perToken * SEQUENCE) / (8 * 4 * 0.75 * 989.5e12), 12);
    expect(step.mfu).toBeCloseTo((perToken * step.tokensPerSecond) / (512 * 989.5e12), 12);
  });

  it("counts recomputation in HFU but not in MFU", () => {
    const selective = stepTime(layout({ zero: 3 }));
    const full = stepTime(layout({ zero: 3, recompute: "full" }));
    expect(full.hfu).toBeCloseTo((4 / 3) * full.mfu, 12);
    expect(selective.hfu).toBe(selective.mfu);
    expect(full.mfu).toBeLessThan(selective.mfu);
    expect(full.parts.recompute).toBeCloseTo(full.parts.compute / 3, 9);
  });

  it("hides the sync under two thirds of one microbatch's compute when overlap is on", () => {
    const on = stepTime(layout({ zero: 3 }));
    const off = stepTime(layout({ zero: 3, overlap: false }));
    expect(on.exposedDataSync).toBeCloseTo(Math.max(0, on.dataSync.seconds - (2 / 3) * on.computePerMicrobatch), 12);
    expect(off.exposedDataSync).toBe(off.dataSync.seconds);
    expect(stepTime(layout({ data: 1 })).dataSync.seconds).toBe(0);
  });

  it("gives the pipeline bubble (p − 1)/(m + p − 1)", () => {
    const bubble = (pipeline: number, microbatches: number) =>
      stepTime(layout({ id: "70b", tensor: 8, pipeline, data: 16, microbatches, zero: 1 })).bubbleFraction;
    expect(bubble(8, 1)).toBe(7 / 8);
    expect(bubble(8, 32)).toBe(7 / 39);
    expect(bubble(4, 16)).toBe(3 / 19);
    expect(bubble(1, 1)).toBe(0);
    expect(bubble(16, 16)).toBe(15 / 31);
  });

  it("moves the way the lesson says when one control moves", () => {
    // More microbatches: a smaller bubble and a higher MFU on a deep pipeline.
    let previous = stepTime(layout({ id: "70b", tensor: 8, pipeline: 8, data: 16, microbatches: 1, zero: 1 }));
    for (const microbatches of [2, 4, 8, 16, 32, 64, 128]) {
      const next = stepTime(layout({ id: "70b", tensor: 8, pipeline: 8, data: 16, microbatches, zero: 1 }));
      expect(next.bubbleFraction).toBeLessThan(previous.bubbleFraction);
      expect(next.mfu).toBeGreaterThan(previous.mfu);
      previous = next;
    }
    // More data parallelism: more tokens per second, and never less sync time per GPU.
    let last = stepTime(layout({ id: "70b", tensor: 8, pipeline: 8, data: 2, microbatches: 32, zero: 1 }));
    for (const data of [4, 8, 16, 64, 256, 2048]) {
      const next = stepTime(layout({ id: "70b", tensor: 8, pipeline: 8, data, microbatches: 32, zero: 1 }));
      expect(next.tokensPerSecond).toBeGreaterThan(last.tokensPerSecond);
      expect(next.dataSync.seconds).toBeGreaterThanOrEqual(last.dataSync.seconds);
      last = next;
    }
    // More tensor parallelism: less memory on every GPU.
    let memory = memoryPerGpu(layout({ id: "70b", tensor: 1, pipeline: 4, data: 16, zero: 1 })).total;
    for (const tensor of [2, 4, 8, 16, 32]) {
      const next = memoryPerGpu(layout({ id: "70b", tensor, pipeline: 4, data: 16, zero: 1 })).total;
      expect(next).toBeLessThan(memory);
      memory = next;
    }
    // A higher ZeRO stage never adds model state.
    for (const zero of [1, 2, 3] as ZeroStage[]) {
      expect(memoryPerGpu(layout({ zero })).modelState).toBeLessThan(memoryPerGpu(layout({ zero: (zero - 1) as ZeroStage })).modelState);
    }
  });
});

describe("the 1F1B schedule", () => {
  it("runs every forward and backward once and takes 3(m + p − 1) units", () => {
    for (const p of [1, 2, 4, 8, 16]) {
      for (const m of [1, 2, 3, 8, 32]) {
        const ops = oneForwardOneBackward(p, m);
        expect(ops.length).toBe(2 * p * m);
        expect(Math.max(...ops.map((op) => op.end))).toBe(3 * (m + p - 1));
        for (let stage = 0; stage < p; stage += 1) {
          const busy = ops.filter((op) => op.stage === stage).reduce((total, op) => total + op.end - op.start, 0);
          expect(busy).toBe(3 * m);
        }
      }
    }
  });

  it("respects the dependencies between stages and never runs two ops at once on a stage", () => {
    const ops = oneForwardOneBackward(4, 6);
    const find = (stage: number, microbatch: number, kind: "forward" | "backward") =>
      ops.find((op) => op.stage === stage && op.microbatch === microbatch && op.kind === kind)!;
    for (const op of ops) {
      if (op.kind === "forward" && op.stage > 0) expect(op.start).toBeGreaterThanOrEqual(find(op.stage - 1, op.microbatch, "forward").end);
      if (op.kind === "backward" && op.stage < 3) expect(op.start).toBeGreaterThanOrEqual(find(op.stage + 1, op.microbatch, "backward").end);
      if (op.kind === "backward") expect(op.start).toBeGreaterThanOrEqual(find(op.stage, op.microbatch, "forward").end);
    }
    for (let stage = 0; stage < 4; stage += 1) {
      const mine = ops.filter((op) => op.stage === stage).sort((a, b) => a.start - b.start);
      for (let index = 1; index < mine.length; index += 1) expect(mine[index].start).toBeGreaterThanOrEqual(mine[index - 1].end);
    }
  });

  it("keeps at most min(m, p) microbatches in flight on the first stage", () => {
    for (const [p, m] of [
      [8, 32],
      [8, 3],
      [4, 4],
    ]) {
      const first = oneForwardOneBackward(p, m).filter((op) => op.stage === 0);
      let inFlight = 0;
      let peak = 0;
      for (const op of [...first].sort((a, b) => a.start - b.start)) {
        inFlight += op.kind === "forward" ? 1 : -1;
        peak = Math.max(peak, inFlight);
      }
      expect(peak).toBe(Math.min(m, p));
    }
  });
});

describe("failures and checkpoints", () => {
  it("derives 50,677 GPU-hours per interruption from Llama 3's 54-day snapshot", () => {
    expect(GPU_HOURS_PER_INTERRUPTION).toBeCloseTo((16384 * 54 * 24) / 419, 9);
    expect(Math.round(GPU_HOURS_PER_INTERRUPTION)).toBe(50_677);
    expect(jobMtbfHours(16384)).toBeCloseTo((54 * 24) / 419, 12);
    expect(jobMtbfHours(512) / jobMtbfHours(16384)).toBeCloseTo(32, 12);
  });

  it("gives Young's interval √(2δM) as the minimum of the first-order time lost", () => {
    const save = 60;
    const mtbf = jobMtbfHours(16384) * 3600;
    const young = youngInterval(save, mtbf);
    expect(young).toBeCloseTo(Math.sqrt(2 * save * mtbf), 9);
    for (const factor of [0.5, 0.9, 0.99, 1.01, 1.1, 2]) {
      expect(timeLost(young * factor, save, mtbf, 600)).toBeGreaterThan(timeLost(young, save, mtbf, 600));
    }
  });

  it("gives Daly's higher-order interval, which tends to Young's as δ/M → 0", () => {
    const mtbf = 100_000;
    const daly = (save: number) => dalyInterval(save, mtbf);
    const x = 60 / (2 * mtbf);
    expect(daly(60)).toBeCloseTo(Math.sqrt(2 * 60 * mtbf) * (1 + Math.sqrt(x) / 3 + x / 9) - 60, 9);
    let gap = Number.POSITIVE_INFINITY;
    for (const save of [1000, 100, 10, 1, 0.1, 0.01]) {
      const relative = Math.abs(daly(save) / youngInterval(save, mtbf) - 1);
      expect(relative).toBeLessThan(gap);
      gap = relative;
    }
    expect(gap).toBeLessThan(1e-3);
    // Once the save takes at least twice the MTBF, Daly's estimate is the MTBF itself.
    expect(dalyInterval(300, 100)).toBe(100);
    expect(daly(60)).toBeLessThan(youngInterval(60, mtbf));
  });

  it("saves 14 bytes per parameter: bf16 weights and the fp32 optimizer state, not the gradients", () => {
    for (const model of MODELS) expect(checkpointBytes(model)).toBe(14 * parameterCounts(model).total);
  });
});

describe("ladders", () => {
  it("snaps a value down to the nearest rung and clamps both ends", () => {
    const ladder = [1, 2, 4, 8];
    expect(snapToLadder(3, ladder)).toBe(2);
    expect(snapToLadder(8, ladder)).toBe(8);
    expect(snapToLadder(1e9, ladder)).toBe(8);
    expect(snapToLadder(-5, ladder)).toBe(1);
    expect(snapToLadder(Number.NaN, ladder)).toBe(1);
  });
});
