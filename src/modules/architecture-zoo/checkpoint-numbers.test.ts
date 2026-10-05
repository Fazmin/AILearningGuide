import { describe, expect, it } from "vitest";
import { checkpointQuestions } from "@app/module-sdk";
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
import definition from "./module";

/**
 * Pins the comparisons the checkpoint questions rely on. The exact figures the lesson
 * quotes are pinned in zoo.test.ts.
 */
const questions = checkpointQuestions(definition);
const correctOption = (index: number) => questions[index].options[questions[index].answer];
const pixels = parsePixels(DEFAULT_PIXELS);

describe("architecture-zoo checkpoint questions", () => {
  it("question 1: shifting the same stroke shifts the response map by the same amount, with 9 weights replacing 4,096", () => {
    const stroke = (row: number, column: number) => {
      const pad = new Array(64).fill(0);
      pad[row * 8 + column] = 1;
      pad[row * 8 + column + 1] = 1;
      return pad;
    };
    for (const filter of FILTERS) {
      const a = convolve(stroke(2, 2), filter.kernel);
      const b = convolve(stroke(5, 4), filter.kernel);
      for (let row = 0; row < 3; row += 1) {
        for (let column = 0; column < 4; column += 1) {
          expect(a[1 + row][1 + column]).toBe(b[4 + row][3 + column]);
        }
      }
      expect(filter.kernel).toHaveLength(9);
    }
    expect(wiringStats("mlp", 8, 8).weights).toBe(4096);
    expect(correctOption(0)).toMatch(/^The same pattern of responses appears at the new location/);
  });

  it("question 2: at w = 0.9 the first column's pull on h7 is far smaller than the last column's; above 1 it grows instead", () => {
    const { influence } = rnnForward(columnMeans(pixels), 0.9);
    expect(influence[0]).toBeGreaterThan(0);
    expect(influence[7]).toBeGreaterThan(30 * influence[0]);
    expect(influence[7]).toBeGreaterThan(0);
    const blank = rnnForward(new Array(8).fill(0), 1.5).influence;
    expect(blank[0]).toBeGreaterThan(blank[7]);
    expect(correctOption(1)).toMatch(/^Column 1 is far smaller/);
  });

  it("question 3: causal attention reaches position 1 from position 8 in one hop and hides later positions; recurrence takes 8 hops; one kernel-3 layer cannot reach", () => {
    const attention = dependencyMatrix("attn", 8, 1);
    const recurrent = dependencyMatrix("rnn", 8, 1);
    const conv = dependencyMatrix("conv", 8, 1);
    expect(attention[7][0]).toBe(1);
    expect(attention[7].slice(0, 8).every((hop) => hop !== null)).toBe(true);
    expect(attention[0].slice(1).every((hop) => hop === null)).toBe(true);
    expect(recurrent[7][0]).toBe(8);
    expect(conv[7][0]).toBeNull();
    expect(correctOption(2)).toMatch(/^Causal self-attention/);
  });

  it("question 4: the per-position MLP touches only the diagonal, while attention mixes across positions", () => {
    const ffn = dependencyMatrix("ffn", 8, 1);
    const offDiagonal = ffn.flatMap((row, t) => row.filter((hop, s) => s !== t && hop !== null));
    expect(offDiagonal).toHaveLength(0);
    const attention = dependencyMatrix("attn", 8, 1);
    expect(attention.flatMap((row, t) => row.filter((hop, s) => s !== t && hop !== null)).length).toBeGreaterThan(0);
    expect(correctOption(3)).toMatch(/^Attention, which lets positions read other allowed positions/);
  });

  it("question 5: doubling the experts at top-2 doubles total weights and leaves active weights unchanged", () => {
    const eight = moeParameters(64, 256, 8, 2);
    const sixteen = moeParameters(64, 256, 16, 2);
    expect(sixteen.total).toBe(2 * eight.total);
    expect(sixteen.active).toBe(eight.active);
    expect(correctOption(4)).toMatch(/^Total expert weights double while Active per token stays put/);
  });

  it("question 6: at the default 8 experts and top-2 the untrained, seeded router already leaves experts with no tokens", () => {
    const routes = routeTokens(columnVectors(pixels), 8, 2);
    const load = Array.from({ length: 8 }, (_, expert) => routes.filter((route) => route.chosen.includes(expert)).length);
    expect(load.some((tokens) => tokens === 0)).toBe(true);
    // The lab's note names the idle experts: stored, never used.
    expect(load.filter((tokens) => tokens === 0).length).toBeGreaterThanOrEqual(1);
    expect(correctOption(5)).toMatch(/^Only that this untrained, seeded router/);
  });
});
