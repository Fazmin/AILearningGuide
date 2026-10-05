import { describe, expect, it } from "vitest";
import {
  DEPTH_CLIP_MAX,
  DEPTH_GAIN_MAX,
  DEPTH_GAIN_MIN,
  DEPTH_LAYERS_MAX,
  depthReport,
  formatMagnitude,
} from "./depth";

/** Pins every figure the Gradient through depth card, its card text, and the lesson quote. */
describe("gradient through depth", () => {
  it("multiplies the gradient by the gain at every layer, starting from 1 at the output", () => {
    const report = depthReport(4, 2, 0);
    expect(report.norms).toEqual([8, 4, 2, 1]);
    expect(report.total).toBeCloseTo(Math.sqrt(64 + 16 + 4 + 1), 12);
    expect(report.inputToOutput).toBe(8);
    expect(report.scale).toBe(1);
    expect(report.binding).toBe(false);
    expect(report.clipped).toEqual(report.norms);
  });

  it("explodes at the default setting, and a clip of 1 scales the whole gradient to exactly 1", () => {
    const raw = depthReport(24, 1.2, 0);
    expect(raw.inputToOutput).toBeCloseTo(66.25, 2);
    expect(raw.total).toBeCloseTo(119.84, 2);
    expect(raw.binding).toBe(false);
    const clipped = depthReport(24, 1.2, 1);
    expect(clipped.binding).toBe(true);
    expect(clipped.scale).toBeCloseTo(1 / clipped.total, 12);
    expect(clipped.scale).toBeCloseTo(0.00834, 5);
    const clippedTotal = Math.sqrt(clipped.clipped.reduce((sum, value) => sum + value * value, 0));
    expect(clippedTotal).toBeCloseTo(1, 12);
    // The clip keeps the direction: every layer is scaled by the same factor, so the ratios do not move.
    expect(clipped.clipped[0] / clipped.clipped[23]).toBeCloseTo(raw.inputToOutput, 9);
  });

  it("explodes much harder at 40 layers", () => {
    const report = depthReport(40, 1.2, 0);
    expect(report.inputToOutput).toBeCloseTo(1224.8, 1);
    expect(formatMagnitude(report.total)).toBe("2216");
  });

  it("vanishes at a gain below 1, and no clip can lengthen it", () => {
    const vanishing = depthReport(40, 0.8, 0);
    expect(vanishing.inputToOutput).toBeCloseTo(1.66e-4, 6);
    expect(vanishing.inputToOutput).toBeLessThan(1e-3);
    expect(vanishing.total).toBeCloseTo(1.667, 3);
    for (const clip of [0, 0.5, 1, 2, 5, DEPTH_CLIP_MAX]) {
      const report = depthReport(40, 0.8, clip);
      expect(report.scale).toBeLessThanOrEqual(1);
      expect(report.clipped[0]).toBeLessThanOrEqual(vanishing.norms[0]);
      // The ratio between the first and last layer is untouched whatever the clip.
      expect(report.clipped[0] / report.clipped[39]).toBeCloseTo(vanishing.inputToOutput, 12);
    }
    // A clip of 2 sits above the total of 1.667, so it never binds here.
    expect(depthReport(40, 0.8, 2).binding).toBe(false);
    expect(depthReport(40, 0.8, 1).binding).toBe(true);
  });

  it("keeps every length finite and positive at the largest setting the sliders allow", () => {
    const report = depthReport(DEPTH_LAYERS_MAX, DEPTH_GAIN_MAX, DEPTH_CLIP_MAX);
    expect(report.norms.every((value) => Number.isFinite(value) && value > 0)).toBe(true);
    expect(Number.isFinite(report.total)).toBe(true);
    const smallest = depthReport(DEPTH_LAYERS_MAX, DEPTH_GAIN_MIN, 0);
    expect(smallest.norms.every((value) => Number.isFinite(value) && value > 0)).toBe(true);
    // A gain of 1 gives every layer the same length, so the total grows like the square root of the depth.
    expect(depthReport(16, 1, 0).total).toBe(4);
  });

  it("formats lengths across many orders of magnitude without raw floats", () => {
    expect(formatMagnitude(1)).toBe("1.000");
    expect(formatMagnitude(66.2523)).toBe("66.3");
    expect(formatMagnitude(1224.81)).toBe("1225");
    expect(formatMagnitude(1.66e-4)).toBe("1.66 × 10⁻⁴");
    expect(formatMagnitude(7.5e6)).toBe("7.50 × 10⁶");
    expect(formatMagnitude(0)).toBe("0");
    expect(formatMagnitude(Number.POSITIVE_INFINITY)).toBe("off the chart");
  });
});
