import { describe, expect, it } from "vitest";
import { checkpointQuestions } from "@app/module-sdk";
import definition from "./module";
import {
  formatBytes,
  formatHours,
  formatMinutes,
  formatPercent,
  formatSeconds,
  HARDWARE,
  jobMtbfHours,
  memoryPerGpu,
  modelById,
  parameterCounts,
  stepTime,
  timeLost,
  youngInterval,
  type Layout,
  type ZeroStage,
} from "./parallel";
import { defaultState, LLAMA3_LAYOUT } from "./state";

/**
 * Pins the factual claims in each checkpoint question's explanation, and which option is correct, so a change to the
 * arithmetic or to the question text cannot leave an explanation saying something the lab no longer shows.
 */
const questions = checkpointQuestions(definition);
const correct = (index: number) => questions[index].options[questions[index].answer];
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

describe("parallel-training checkpoint claims", () => {
  it("has ten questions that cover all five objectives", () => {
    expect(questions).toHaveLength(10);
    expect(new Set(questions.map((question) => question.objective))).toEqual(new Set([0, 1, 2, 3, 4]));
  });

  it("question 1: ZeRO stage 2 shards the gradients, not the weights or the activations", () => {
    expect(correct(0)).toMatch(/^Gradients, because each GPU now reduce-scatters them/);
    const one = memory({ zero: 1 });
    const two = memory({ zero: 2 });
    expect([formatBytes(one.gradients), formatBytes(two.gradients)]).toEqual(["16.1 GB", "250.9 MB"]);
    expect(two.weights).toBe(one.weights);
    expect(two.optimizer).toBe(one.optimizer);
    expect(two.activations).toBe(one.activations);
    // The weights are sharded only at stage 3.
    expect(memory({ zero: 3 }).weights).toBeLessThan(two.weights);
  });

  it("question 2: data parallelism alone leaves the whole 128.5 GB of model state on every GPU", () => {
    expect(correct(1)).toMatch(/^Each replica holds the whole model state/);
    for (const data of [1, 8, 64, 2048]) expect(formatBytes(memory({ zero: 0, data }).modelState)).toBe("128.5 GB");
    expect(memory({ zero: 0 }).modelState).toBe(16 * parameterCounts(modelById("8b")).total);
    // Tensor and pipeline parallelism divide it; ZeRO shards it over the replicas.
    expect(memory({ zero: 0, tensor: 2 }).modelState).toBeCloseTo(memory({ zero: 0 }).modelState / 2, 0);
    expect(memory({ zero: 0, pipeline: 2 }).modelState).toBeCloseTo(memory({ zero: 0 }).modelState / 2, 0);
    expect(memory({ zero: 3 }).modelState).toBeLessThan(memory({ zero: 0 }).modelState);
  });

  it("question 3: only recomputation brings four sequences per microbatch back under 80 GB", () => {
    expect(correct(2)).toMatch(/^Recompute Full, which keeps only each layer's input/);
    const selective = memory({ zero: 3, microBatch: 4 });
    expect(formatBytes(selective.activations)).toBe("146 GB");
    expect(selective.fits).toBe(false);
    expect(memory({ zero: 3, microBatch: 4 }).layersPerStage).toBe(32);
    // ZeRO and data parallelism leave the activations alone.
    expect(memory({ zero: 0, microBatch: 4 }).activations).toBe(selective.activations);
    expect(memory({ zero: 3, microBatch: 4, data: 2048 }).activations).toBe(selective.activations);
    const full = memory({ zero: 3, microBatch: 4, recompute: "full" });
    expect(formatBytes(full.activations)).toBe("8.6 GB");
    expect(full.fits).toBe(true);
    expect(formatBytes(memory({ zero: 3, microBatch: 4, recompute: "none" }).activations)).toBe("1.5 TB");
    // The Recompute row is a third the size of Compute.
    const timing = step({ zero: 3, microBatch: 4, recompute: "full" });
    expect(timing.parts.recompute / timing.parts.compute).toBeCloseTo(1 / 3, 12);
  });

  it("question 4: tensor 16 leaves the node, so the same all-reduces move over the slower network", () => {
    expect(correct(3)).toMatch(/^Sixteen GPUs span two 8-GPU nodes/);
    const at = (tensor: number) => step({ id: "70b", tensor, pipeline: 4, data: 16, microbatches: 16, zero: 1 }).tensor;
    expect(at(16).bytesPerCall).toBe(at(8).bytesPerCall);
    expect(at(16).bytesPerCall).toBe(2 * 1 * 8192 * modelById("70b").hidden);
    expect(at(8).calls).toBe(at(16).calls);
    expect(at(8).calls / 20).toBe(4);
    expect([at(8).link.bytesPerSecond, at(16).link.bytesPerSecond]).toEqual([450e9, 50e9]);
    expect([formatSeconds(at(8).seconds), formatSeconds(at(16).seconds)]).toEqual(["0.042 s", "0.403 s"]);
    expect(Math.round(at(16).seconds / at(8).seconds)).toBe(10);
  });

  it("question 5: 32 microbatches cut the bubble from 87.5% to 17.9% and make the batch 32 times larger", () => {
    expect(correct(4)).toMatch(/^It falls from 87\.5% to 17\.9%, and each step processes 32 times as many tokens/);
    const one = step({ id: "70b", tensor: 8, pipeline: 8, data: 16, microbatches: 1, zero: 1 });
    const many = step({ id: "70b", tensor: 8, pipeline: 8, data: 16, microbatches: 32, zero: 1 });
    expect([formatPercent(one.bubbleFraction), formatPercent(many.bubbleFraction)]).toEqual(["87.5%", "17.9%"]);
    expect([one.bubbleFraction, many.bubbleFraction]).toEqual([7 / 8, 7 / 39]);
    expect(many.tokensPerStep).toBe(32 * one.tokensPerStep);
    // The idle slots stay the same in seconds while the step gets longer, not shorter.
    expect(many.parts.bubble).toBeCloseTo(one.parts.bubble, 12);
    expect(many.step).toBeGreaterThan(one.step);
  });

  it("question 6: the lab charges plain 1F1B's 48.4% bubble on the layout Llama 3 interleaved", () => {
    expect(correct(5)).toMatch(/^An interleaved schedule/);
    const preset = step({ ...LLAMA3_LAYOUT, id: LLAMA3_LAYOUT.modelId, zero: LLAMA3_LAYOUT.zero as ZeroStage });
    expect(formatPercent(preset.mfu)).toBe("33.2%");
    expect(preset.bubbleFraction).toBe(15 / 31);
    expect(formatPercent(preset.bubbleFraction)).toBe("48.4%");
    expect(HARDWARE.efficiency).toBe(0.75);
  });

  it("question 7: Full recompute lowers MFU and raises HFU by the rerun forward pass", () => {
    expect(correct(6)).toMatch(/^MFU falls and HFU rises/);
    const selective = step({ zero: 3 });
    const full = step({ zero: 3, recompute: "full" });
    expect([formatPercent(selective.mfu), formatPercent(full.mfu), formatPercent(full.hfu)]).toEqual(["68.6%", "53.6%", "71.5%"]);
    expect(full.hfu / full.mfu).toBeCloseTo(4 / 3, 12);
    expect(full.hfu).toBeGreaterThan(selective.hfu);
  });

  it("question 8: overlap hides the sync under the last backward pass without changing its bytes or link", () => {
    expect(correct(7)).toMatch(/^The sync runs under the last microbatch's backward pass/);
    const base = { id: "70b", tensor: 8, pipeline: 8, data: 256, microbatches: 32, zero: 1 as ZeroStage };
    const on = step(base);
    const off = step({ ...base, overlap: false });
    expect([formatSeconds(on.exposedDataSync), formatSeconds(off.exposedDataSync)]).toEqual(["0.032 s", "0.088 s"]);
    expect(on.dataSync.seconds).toBe(off.dataSync.seconds);
    expect(on.dataSync.bytesPerCall).toBe(off.dataSync.bytesPerCall);
    expect(on.dataSync.link).toBe(off.dataSync.link);
    expect(on.exposedDataSync).toBeCloseTo(on.dataSync.seconds - (2 / 3) * on.computePerMicrobatch, 12);
  });

  it("question 9: failure rates add up, so 32 times the GPUs means a 32 times shorter MTBF", () => {
    expect(correct(8)).toMatch(/^Each GPU fails no more often/);
    expect([formatHours(jobMtbfHours(16384)), formatHours(jobMtbfHours(512))]).toEqual(["3.09 h", "99.0 h"]);
    expect(jobMtbfHours(512) / jobMtbfHours(16384)).toBeCloseTo(32, 9);
    const young = (gpus: number) => youngInterval(60, jobMtbfHours(gpus) * 3600);
    expect([formatMinutes(young(512)), formatMinutes(young(16384))]).toEqual(["109 min", "19.3 min"]);
    expect(young(512) / young(16384)).toBeCloseTo(Math.sqrt(32), 9);
  });

  it("question 10: the curve that bottoms out near 19 minutes assumes a steady, memoryless failure rate", () => {
    expect(correct(9)).toMatch(/^Failures arrive independently at a steady rate/);
    const mtbf = jobMtbfHours(16384) * 3600;
    const lost = (minutes: number) => timeLost(minutes * 60, 60, mtbf, 600);
    expect(lost(19)).toBeLessThan(lost(18));
    expect(lost(19)).toBeLessThan(lost(20));
    // One fixed rate: the time lost depends only on the interval, the save, the restart, and the mean.
    expect(timeLost(1140, 60, mtbf, 600)).toBe(60 / 1140 + (570 + 600) / mtbf);
  });
});
