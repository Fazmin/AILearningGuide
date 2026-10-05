import { describe, expect, it } from "vitest";
import {
  causalPairs,
  crossoverTokens,
  fullSoftmaxAttention,
  kvBytesPerToken,
  kvCacheBytes,
  layerWeights,
  onlineSoftmax,
  prefillFlops,
  ropeFrequency,
  ropeRotate,
  ropeScore,
  scoreMatrixBytes,
} from "./scale-math";

describe("KV cache", () => {
  it("matches Llama 3 8B's 128 KiB per token with 8 KV heads", () => {
    expect(kvBytesPerToken(8)).toBe(128 * 1024);
    expect(kvBytesPerToken(32)).toBe(512 * 1024);
    expect(kvBytesPerToken(1)).toBe(16 * 1024);
  });

  it("scales linearly with KV heads and tokens, and caps at the window", () => {
    expect(kvCacheBytes(8, 8192)).toBe(1024 ** 3);
    expect(kvCacheBytes(32, 8192) / kvCacheBytes(8, 8192)).toBe(4);
    expect(kvCacheBytes(8, 131072, 4096)).toBe(kvCacheBytes(8, 4096));
  });
});

describe("attention cost", () => {
  it("counts causal pairs with and without a window", () => {
    expect(causalPairs(4)).toBe(10);
    expect(causalPairs(6, 3)).toBe(6 + 3 * 3);
    expect(causalPairs(3, 8)).toBe(6);
  });

  it("layer weights match Llama 3 8B's 218,103,808 per layer", () => {
    expect(layerWeights(8)).toBe(218_103_808);
  });

  it("puts the causal crossover where the two FLOP terms are equal", () => {
    const n = crossoverTokens(8);
    const flops = prefillFlops(n, 8);
    expect(flops.attention / flops.weights).toBeCloseTo(1, 9);
    expect(Math.round(n)).toBe(53_247);
  });

  it("stores a 4 GiB score matrix per layer at 8K tokens", () => {
    expect(scoreMatrixBytes(8192)).toBe(4 * 1024 ** 3);
  });
});

describe("online softmax", () => {
  const scores = [0.3, -1.2, 2.4, 0.9, -0.5, 3.1, 1.7, -2.2, 0.1, 2.9, -0.8, 1.1, 0.4, -1.6, 2.2, 0.6];
  const values = scores.map((_, index) => [Math.cos(index * 0.7), Math.sin(index * 0.7)]);
  const full = fullSoftmaxAttention(scores, values);

  it.each([1, 2, 4, 8, 16, 5])("tile size %i gives the exact softmax output", (tile) => {
    const online = onlineSoftmax(scores, values, tile);
    online.output.forEach((value, index) => expect(value).toBeCloseTo(full.output[index], 12));
    expect(online.runningMax).toBe(Math.max(...scores));
  });

  it("rescales by exp(old max − new max) only when a tile raises the max", () => {
    const { steps } = onlineSoftmax(scores, values, 4);
    expect(steps[0].rescale).toBe(0);
    steps.slice(1).forEach((step, index) => {
      const previous = steps[index].runningMax;
      expect(step.rescale).toBeCloseTo(Math.exp(previous - step.runningMax), 12);
    });
  });
});

describe("RoPE", () => {
  const q = [1, 0.2, 0.6, -0.4, 0.8, 0.5, -0.3, 0.9];
  const k = [0.9, 0.1, 0.3, -0.7, 0.6, 0.6, 0.2, 0.8];

  it("uses θ_i = base^(−2i/d)", () => {
    expect(ropeFrequency(0, 8, 10000)).toBe(1);
    expect(ropeFrequency(2, 8, 10000)).toBeCloseTo(0.01, 12);
  });

  it("preserves length", () => {
    const rotated = ropeRotate(q, 37, 10000);
    const length = (v: number[]) => Math.hypot(...v);
    expect(length(rotated)).toBeCloseTo(length(q), 12);
  });

  it("makes the score depend only on the relative position", () => {
    for (const base of [10000, 500000]) {
      const reference = ropeScore(q, k, 20, 7, base);
      expect(ropeScore(q, k, 33, 20, base)).toBeCloseTo(reference, 10);
      expect(ropeScore(q, k, 13, 0, base)).toBeCloseTo(reference, 10);
    }
  });
});
