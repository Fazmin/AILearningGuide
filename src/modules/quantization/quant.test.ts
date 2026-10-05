import { describe, expect, it } from "vitest";
import { TINY_CORPORA, tinyPerplexity, trainTinyModel } from "@app/module-sdk";
import { hydrateQuantizationState } from "./module";
import { codesUsed, errorStats, fileBitsPerWeight, GGUF_BLOCK_FORMATS, quantizeWeights, toHalf } from "./quant";

const sample = Float32Array.from([-1, -0.25, 0, 0.4, 2, 0.9, -0.6, 1.3]);

describe("teaching quantizer", () => {
  it("symmetric: scale = max|w| / (2^(b-1) - 1), zero stays exact, one fp16 scale per block", () => {
    const result = quantizeWeights(sample, { bits: 4, scope: "tensor", zeroPoint: "symmetric" });
    expect(result.blocks[0].scale).toBeCloseTo(2 / 7, 10);
    expect(result.weights[2]).toBe(0);
    expect(result.weights[4]).toBeCloseTo(2, 6);
    expect(result.levels).toBe(15);
    expect(result.bitsPerWeight).toBeCloseTo(4 + 16 / 8, 10);
  });

  it("asymmetric: min and max are exact levels, and a scale plus offset cost 32 bits", () => {
    const result = quantizeWeights(sample, { bits: 3, scope: "tensor", zeroPoint: "asymmetric" });
    expect(result.blocks[0].scale).toBeCloseTo(3 / 7, 10);
    expect(result.weights[0]).toBeCloseTo(-1, 6);
    expect(result.weights[4]).toBeCloseTo(2, 6);
    expect(result.levels).toBe(8);
    expect(result.bitsPerWeight).toBeCloseTo(3 + 32 / 8, 10);
  });

  it("keeps every in-range error within half a step", () => {
    const result = quantizeWeights(sample, { bits: 2, scope: "group", groupSize: 4 });
    for (const block of result.blocks) {
      for (let index = block.start; index < block.end; index += 1) {
        expect(Math.abs(result.weights[index] - sample[index])).toBeLessThanOrEqual(block.scale / 2 + 1e-9);
      }
    }
  });

  it("charges Q4_0-style and Q4_1-style blocks 4.5 and 5.0 bits per weight at 32 weights", () => {
    const long = Float32Array.from({ length: 64 }, (_, index) => Math.sin(index));
    expect(quantizeWeights(long, { bits: 4, scope: "group", groupSize: 32, zeroPoint: "symmetric" }).bitsPerWeight).toBe(4.5);
    expect(quantizeWeights(long, { bits: 4, scope: "group", groupSize: 32 }).bitsPerWeight).toBe(5);
  });

  it("stores a pair exactly with an asymmetric group of 2, at 20 bits per weight", () => {
    const result = quantizeWeights(sample, { bits: 4, scope: "group", groupSize: 2 });
    expect(result.rmsError).toBeLessThan(1e-7);
    expect(result.bitsPerWeight).toBe(20);
  });

  it("counts distinct codes used in a block", () => {
    const result = quantizeWeights(Float32Array.from([0, 0, 0, 5]), { bits: 4, scope: "tensor" });
    expect(codesUsed(result, result.blocks[0])).toBe(2);
  });
});

describe("half precision and llama.cpp layouts", () => {
  it("rounds to IEEE half precision", () => {
    expect(toHalf(65504)).toBe(65504);
    expect(toHalf(1 / 3)).toBeCloseTo(0.333251953125, 12);
    expect(toHalf(1e-8)).toBe(0);
    expect(toHalf(-2.5)).toBe(-2.5);
  });

  it("uses the struct sizes from ggml-common.h for bits per weight", () => {
    const bpw = Object.fromEntries(GGUF_BLOCK_FORMATS.map((format) => [format.name, format.bitsPerWeight]));
    expect(bpw.Q8_0).toBe(((2 + 32) * 8) / 32);
    expect(bpw.Q4_0).toBe(((2 + 16) * 8) / 32);
    expect(bpw.Q4_1).toBe(((4 + 16) * 8) / 32);
    expect(bpw.Q4_K).toBe(((4 + 12 + 128) * 8) / 256);
    expect(bpw.Q5_K).toBe(((4 + 12 + 32 + 128) * 8) / 256);
    expect(bpw.Q6_K).toBe(((128 + 64 + 16 + 2) * 8) / 256);
    expect(bpw.Q3_K).toBe(((32 + 64 + 12 + 2) * 8) / 256);
    expect(bpw.Q2_K).toBe(((16 + 64 + 4) * 8) / 256);
  });

  it("reconstructs Q4_0's largest-magnitude weight exactly and orders formats sensibly", () => {
    const weights = trainTinyModel({ text: TINY_CORPORA.harbor.text, epochs: 80, seed: 1 }).weights;
    const byName = Object.fromEntries(GGUF_BLOCK_FORMATS.map((format) => [format.name, format.quantize(weights)]));
    const block = weights.slice(0, 32);
    const peak = block.reduce((best, value) => (Math.abs(value) > Math.abs(best) ? value : best), 0);
    const index = block.indexOf(peak);
    expect(byName.Q4_0[index]).toBeCloseTo(toHalf(peak / -8) * -8, 6);
    const rms = (name: string) => errorStats(weights, byName[name]).rmsError;
    expect(rms("F16")).toBeLessThan(rms("Q8_0"));
    expect(rms("Q8_0")).toBeLessThan(rms("Q4_K"));
    expect(rms("Q4_K")).toBeLessThan(rms("Q4_0"));
    const text = TINY_CORPORA.harbor.text;
    expect(tinyPerplexity(byName.Q2_K, text)).toBeGreaterThan(tinyPerplexity(byName.Q8_0, text));
  });

  it("converts llama.cpp's Q4_K_M Llama-3-8B size to about 4.9 bits per weight", () => {
    expect(fileBitsPerWeight(4.58)).toBeCloseTo(4.9, 1);
    expect(fileBitsPerWeight(7.96)).toBeCloseTo(8.52, 2);
  });
});

describe("quantization state", () => {
  it("hydrates a version 1 payload with the new defaults", () => {
    const state = hydrateQuantizationState(JSON.stringify({ bits: 3, scope: "row", groupSize: 9 }));
    expect(state).toMatchObject({ bits: 3, scope: "row", groupSize: 10, zeroPoint: "asymmetric", inspect: -1 });
  });

  it("clamps malformed values", () => {
    expect(hydrateQuantizationState(JSON.stringify({ bits: 99, scope: "nope", zeroPoint: 1 }))).toMatchObject({
      bits: 8,
      scope: "group",
      zeroPoint: "asymmetric",
    });
    expect(hydrateQuantizationState("{")).toMatchObject({ bits: 4 });
  });
});
