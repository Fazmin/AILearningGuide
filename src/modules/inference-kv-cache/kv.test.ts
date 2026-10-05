import { describe, expect, it } from "vitest";
import definition from "./module";
import {
  decodeStepCost,
  H100_SXM,
  kvBytesPerToken,
  LLAMA3_8B,
  matmulParameters,
  memoryBudget,
  prefillCost,
  ridgePoint,
  toyPass,
  weightBytes,
} from "./kv";

describe("reference-model memory", () => {
  it("stores 128 KiB of keys and values per token for Llama 3 8B in bf16", () => {
    // 2 (K and V) × 32 layers × 8 KV heads × 128 dims × 2 bytes
    expect(kvBytesPerToken(LLAMA3_8B)).toBe(131072);
    expect(kvBytesPerToken(LLAMA3_8B) / 1024).toBe(128);
    // One key/value head per query head would be four times larger.
    expect(kvBytesPerToken(LLAMA3_8B, 32)).toBe(4 * 131072);
  });

  it("holds 4 GiB of cache for a 32k-token context", () => {
    expect((32768 * kvBytesPerToken(LLAMA3_8B)) / 2 ** 30).toBe(4);
  });

  it("counts bf16 weights and excludes the embedding lookup from per-token matmuls", () => {
    expect(weightBytes(LLAMA3_8B) / 1e9).toBeCloseTo(16.06, 2);
    expect(matmulParameters(LLAMA3_8B) / 1e9).toBeCloseTo(7.505, 3);
  });

  it("reports the cache overtaking the weights at the default settings", () => {
    const budget = memoryBudget(LLAMA3_8B, 8192, 16);
    expect(budget.cache / 1e9).toBeCloseTo(17.18, 2);
    expect(budget.cache).toBeGreaterThan(budget.weights);
    expect(budget.maxSequences).toBe(59);
    expect(memoryBudget(LLAMA3_8B, 8192, 64).fits).toBe(false);
  });
});

describe("roofline time model", () => {
  it("puts the H100 ridge near 295 FLOP per byte", () => {
    expect(ridgePoint(H100_SXM)).toBeCloseTo(295.2, 1);
  });

  it("makes long prefill compute bound and cached decode memory bound", () => {
    const prefill = prefillCost(LLAMA3_8B, 8192);
    expect(prefill.bound).toBe("compute");
    expect(prefill.seconds * 1e3).toBeCloseTo(142.1, 1);
    const decode = decodeStepCost(LLAMA3_8B, 8192, 1, true);
    expect(decode.bound).toBe("memory");
    expect(decode.seconds * 1e3).toBeCloseTo(4.8, 1);
    expect(decode.intensity).toBeLessThan(2);
  });

  it("makes a recompute step cost a full prefill", () => {
    const recompute = decodeStepCost(LLAMA3_8B, 8192, 1, false);
    expect(recompute.flops).toBeCloseTo(prefillCost(LLAMA3_8B, 8192).flops, -6);
    expect(recompute.seconds / decodeStepCost(LLAMA3_8B, 8192, 1, true).seconds).toBeGreaterThan(25);
  });

  it("raises total throughput but lowers each sequence's rate as the batch grows (checkpoint question 6)", () => {
    const one = decodeStepCost(LLAMA3_8B, 8192, 1, true);
    const sixteen = decodeStepCost(LLAMA3_8B, 8192, 16, true);
    expect(16 / sixteen.seconds).toBeGreaterThan(1 / one.seconds);
    expect(1 / sixteen.seconds).toBeLessThan(1 / one.seconds);
    expect(sixteen.bound).toBe("memory");
  });

  it("doubles the cache total, not the bytes per token or the sequences that fit, when sequences double (checkpoint question 5)", () => {
    const sixteen = memoryBudget(LLAMA3_8B, 8192, 16);
    const thirtyTwo = memoryBudget(LLAMA3_8B, 8192, 32);
    expect(thirtyTwo.cache).toBe(2 * sixteen.cache);
    expect(thirtyTwo.maxSequences).toBe(sixteen.maxSequences);
    expect(kvBytesPerToken(LLAMA3_8B)).toBe(131072);
  });

  it("amortizes weights across a batch but not caches", () => {
    const one = decodeStepCost(LLAMA3_8B, 8192, 1, true);
    const sixteen = decodeStepCost(LLAMA3_8B, 8192, 16, true);
    const cacheBytes = 8192 * kvBytesPerToken(LLAMA3_8B);
    expect(sixteen.bytes - one.bytes).toBeCloseTo(15 * cacheBytes, -3);
    expect(16 / sixteen.seconds).toBeGreaterThan(7 / one.seconds);
  });
});

describe("toy causal pass", () => {
  it("computes the whole triangle in prefill", () => {
    const pass = toyPass(7, 0, true);
    expect(pass).toMatchObject({ isPrefill: true, projectedPositions: 7, dotProducts: 28, cachedAfter: 7 });
  });

  it("computes one row per decode step with the cache", () => {
    const pass = toyPass(7, 4, true);
    expect(pass).toMatchObject({
      length: 11,
      projectedPositions: 1,
      dotProducts: 11,
      readFromCache: 10,
      runTotalCached: 66,
      runTotalRecompute: 230,
    });
  });

  it("scores the full triangle again when the cache is off, and reads nothing from it (checkpoint question 4)", () => {
    const cached = toyPass(7, 4, true);
    const recompute = toyPass(7, 4, false);
    expect(cached.dotProducts).toBe(11);
    expect(recompute.dotProducts).toBe(66);
    expect(recompute.projectedPositions).toBe(11);
    expect(recompute.readFromCache).toBe(0);
  });

  it("repeats the previous triangle without the cache", () => {
    const pass = toyPass(14, 10, false);
    expect(pass).toMatchObject({ projectedPositions: 24, dotProducts: 300, repeatedDotProducts: 276, cachedAfter: 0 });
    expect(pass.runTotalRecompute).toBe(2145);
  });
});

describe("inference-kv-cache state", () => {
  it("keeps old payloads and clamps new keys", () => {
    const old = definition.hydrateState(JSON.stringify({ context: 7, generated: 4, useCache: false }));
    expect(old).toMatchObject({ context: 7, generated: 4, useCache: false, realContext: 8192, sequences: 16 });
    const wild = definition.hydrateState(JSON.stringify({ context: 99, generated: -3, realContext: 1e9, sequences: 0 }));
    expect(wild).toMatchObject({ context: 14, generated: 0, realContext: 131072, sequences: 1 });
  });

  it("reads a version 1 payload, which had no speculative-decoding controls, and fills in their defaults", () => {
    const version1 = JSON.stringify({ context: 9, generated: 2, useCache: true, realContext: 4096, sequences: 8 });
    expect(definition.stateVersion).toBe(2);
    expect(definition.hydrateState(version1)).toEqual({
      context: 9,
      generated: 2,
      useCache: true,
      realContext: 4096,
      sequences: 8,
      specGamma: 4,
      specQuality: 0.7,
      specSeed: 7,
    });
  });

  it("clamps the speculative-decoding controls, so a crafted link cannot ask for a long simulation", () => {
    const wild = definition.hydrateState(JSON.stringify({ specGamma: 1e9, specQuality: -1e9, specSeed: 1e9 }));
    expect(wild).toMatchObject({ specGamma: 8, specQuality: 0.2, specSeed: 999 });
    const low = definition.hydrateState(JSON.stringify({ specGamma: -1e9, specQuality: 1e9, specSeed: -1e9 }));
    expect(low).toMatchObject({ specGamma: 1, specQuality: 1, specSeed: 1 });
    // "1e999" parses to Infinity, which is how a hand-edited link can carry one.
    const infinite = definition.hydrateState('{"specGamma":1e999,"specQuality":1e999,"specSeed":1e999}');
    expect(infinite).toMatchObject({ specGamma: 4, specQuality: 0.7, specSeed: 7 });
    expect(definition.hydrateState(JSON.stringify({ specQuality: "good" })).specQuality).toBe(0.7);
    expect(definition.hydrateState(JSON.stringify({ specQuality: 0.72 })).specQuality).toBe(0.7);
  });
});
