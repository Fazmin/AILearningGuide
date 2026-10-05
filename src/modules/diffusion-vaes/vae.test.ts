import { describe, expect, it } from "vitest";
import metadata from "./assets/mnist-models.metadata.json";
import {
  binaryCrossEntropy,
  gaussianKl,
  inkTotal,
  KL_WEIGHT,
  latentGrid,
  PIXELS,
  PRESET_STROKES,
  priorLogDensity,
  rasterizeStrokes,
} from "./vae";

describe("VAE loss terms", () => {
  it("scores a perfect binary reconstruction near zero and a coin flip at ln 2 per pixel", () => {
    const target = Float32Array.from({ length: PIXELS }, (_, index) => (index % 3 === 0 ? 1 : 0));
    expect(binaryCrossEntropy(target, target)).toBe(0);
    const half = new Float32Array(PIXELS).fill(0.5);
    expect(binaryCrossEntropy(target, half)).toBeCloseTo(PIXELS * Math.log(2), 6);
  });

  it("clamps each log term at -100 like PyTorch", () => {
    expect(binaryCrossEntropy([1], [0])).toBe(100);
  });

  it("gives zero KL for the prior itself and grows with distance and narrowness", () => {
    expect(gaussianKl([0, 0], [0, 0])).toBeCloseTo(0, 12);
    expect(gaussianKl([1, 0], [0, 0])).toBeCloseTo(0.5, 12);
    const narrow = gaussianKl([0, 0], [2 * Math.log(0.1), 2 * Math.log(0.1)]);
    // Each dimension: -1/2 (1 + ln 0.01 - 0.01) = 1.807...
    expect(narrow).toBeCloseTo(2 * -0.5 * (1 + Math.log(0.01) - 0.01), 10);
  });

  it("matches the shipped training setup", () => {
    expect(KL_WEIGHT).toBe(0.35);
    expect(metadata.vae.latent_dimensions).toBe(2);
    const last = metadata.vae.history[metadata.vae.history.length - 1];
    expect(last.loss).toBeCloseTo(last.reconstruction + KL_WEIGHT * last.kl, 0);
  });

  it("evaluates the standard normal log density", () => {
    expect(priorLogDensity([0, 0])).toBeCloseTo(-Math.log(2 * Math.PI), 12);
  });
});

describe("latent mosaic and drawings", () => {
  it("lays the grid out row-major from the top", () => {
    const grid = latentGrid(4, -4, 4);
    expect(grid).toHaveLength(32);
    expect(grid.slice(0, 2)).toEqual([-3, 3]);
    expect(grid.slice(-2)).toEqual([3, -3]);
  });

  it("rasterizes every preset into an MNIST-like amount of ink", () => {
    for (const [name, strokes] of Object.entries(PRESET_STROKES)) {
      const image = rasterizeStrokes(strokes);
      expect(image, name).toHaveLength(PIXELS);
      expect(Math.max(...image), name).toBe(1);
      expect(Math.min(...image), name).toBe(0);
      const ink = inkTotal(image);
      expect(ink, name).toBeGreaterThan(40);
      expect(ink, name).toBeLessThan(200);
    }
  });
});
