import { describe, expect, it } from "vitest";
import {
  columnMeans,
  columnVectors,
  convolve,
  DEFAULT_PIXELS,
  dependencyMatrix,
  FILTERS,
  moeParameters,
  parsePixels,
  rnnForward,
  routeTokens,
  wiringStats,
} from "./zoo";

const pixels = parsePixels(DEFAULT_PIXELS);

describe("convolution", () => {
  it("responds with the kernel sum over one inked pixel's neighbourhood", () => {
    const single = new Array(64).fill(0);
    single[3 * 8 + 3] = 1;
    const center = convolve(single, FILTERS[2].kernel);
    expect(center[3][3]).toBe(4);
    expect(center[3][4]).toBe(-1);
    expect(center[4][4]).toBe(0);
  });

  it("reuses one kernel everywhere: shifting the ink shifts the response", () => {
    const a = new Array(64).fill(0);
    const b = new Array(64).fill(0);
    a[2 * 8 + 2] = 1;
    b[5 * 8 + 5] = 1;
    const ra = convolve(a, FILTERS[0].kernel);
    const rb = convolve(b, FILTERS[0].kernel);
    expect(ra[2][1]).toBe(rb[5][4]);
    expect(ra[2][3]).toBe(rb[5][6]);
  });

  it("gives the default 3 its strongest responses at stroke edges", () => {
    const vertical = convolve(pixels, FILTERS[0].kernel);
    const horizontal = convolve(pixels, FILTERS[1].kernel);
    // Ink to the left, blank to the right: the right-hand stroke's outer edge.
    expect(vertical[2][7]).toBe(-2);
    // The bottom bar (row index 6): +3 in the row above it, −3 in the row below it.
    expect(horizontal[5][3]).toBe(3);
    expect(horizontal[7][3]).toBe(-3);
  });
});

describe("scalar RNN", () => {
  it("reads the column means of the default digit", () => {
    expect(columnMeans(pixels)).toEqual([0, 0.25, 0.25, 0.25, 0.375, 0.375, 0.5, 0]);
  });

  it("matches a finite-difference gradient of h7 with respect to each input", () => {
    const inputs = columnMeans(pixels);
    const { hidden, influence } = rnnForward(inputs, 0.9);
    const epsilon = 1e-6;
    inputs.forEach((_, step) => {
      const bumped = inputs.slice();
      bumped[step] += epsilon;
      const numeric = (rnnForward(bumped, 0.9).hidden[7] - hidden[7]) / epsilon;
      expect(influence[step]).toBeCloseTo(numeric, 5);
    });
  });

  it("lets the first column reach h7 roughly 60 times more weakly than the last at w = 0.9", () => {
    const { influence } = rnnForward(columnMeans(pixels), 0.9);
    expect(influence[0]).toBeCloseTo(0.0094, 3);
    expect(influence[7]).toBeCloseTo(0.598, 2);
    expect(influence[7] / influence[0]).toBeGreaterThan(50);
  });

  it("explodes on a blank pad when w exceeds one", () => {
    const { influence } = rnnForward(new Array(8).fill(0), 1.5);
    expect(influence[0]).toBeCloseTo(1.5 ** 7, 6);
  });
});

describe("dependency matrices", () => {
  it("widens a convolution's reach by one position per layer", () => {
    const one = dependencyMatrix("conv", 8, 1);
    const three = dependencyMatrix("conv", 8, 3);
    expect(one[4].filter((cell) => cell !== null)).toHaveLength(3);
    expect(three[4].filter((cell) => cell !== null)).toHaveLength(7);
    expect(three[4][1]).toBe(3);
  });

  it("makes attention causal with one hop and recurrence causal with t − s + 1 hops", () => {
    const attention = dependencyMatrix("attn", 8, 1);
    const recurrent = dependencyMatrix("rnn", 8, 1);
    expect(attention[7][0]).toBe(1);
    expect(attention[0][7]).toBeNull();
    expect(recurrent[7][0]).toBe(8);
    expect(recurrent[3][3]).toBe(1);
  });

  it("keeps a per-position MLP on the diagonal", () => {
    const ffn = dependencyMatrix("ffn", 8, 1);
    expect(ffn.flat().filter((cell) => cell !== null)).toHaveLength(8);
  });

  it("counts weights and carried state from the stated formulas", () => {
    expect(wiringStats("mlp", 8, 64).weights).toBe(262144);
    expect(wiringStats("conv", 8, 64).weights).toBe(12288);
    expect(wiringStats("attn", 8, 64).weights).toBe(16384);
    expect(wiringStats("attn", 8, 64).carried(4096)).toBe(524288);
    expect(wiringStats("ssm", 8, 64).carried(4096)).toBe(1024);
    expect(wiringStats("conv", 8, 64).longestPath).toBe(7);
  });
});

describe("mixture of experts", () => {
  it("routes every token to exactly k experts with gates summing to one", () => {
    const routes = routeTokens(columnVectors(pixels), 8, 2);
    for (const route of routes) {
      expect(route.chosen).toHaveLength(2);
      expect(route.gates.reduce((sum, gate) => sum + gate, 0)).toBeCloseTo(1, 10);
      expect(route.gates.filter((gate) => gate > 0)).toHaveLength(2);
    }
  });

  it("separates total from active parameters", () => {
    const params = moeParameters(64, 256, 8, 2);
    expect(params.perExpert).toBe(32768);
    expect(params.total).toBe(262144);
    expect(params.active).toBe(65536);
    expect(moeParameters(64, 256, 16, 2).active).toBe(65536);
  });
});
