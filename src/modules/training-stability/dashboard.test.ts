import { describe, expect, it } from "vitest";
import { trainTinyExtended } from "@app/module-sdk";
import {
  corpusPoints,
  decadeTicks,
  formatCoefficient,
  formatLoss,
  formatNumber,
  logDomain,
  logPosition,
  normalBand,
  powerOfTen,
  powerOfTwo,
  quantile,
  stepPoints,
  stepRanges,
  summarizeRun,
  superscript,
  values,
} from "./dashboard";
import { runOptions, type FaultSettings } from "./faults";
import { defaultState } from "./state";

const settings = (patch: Partial<FaultSettings>): FaultSettings => ({ ...defaultState, ...patch });
const finite = (value: number | null) => value === null || Number.isFinite(value);

describe("step points", () => {
  it("never passes a value that is not a finite number to the page", () => {
    for (const patch of [{ fault: "rate" }, { fault: "fp16" }, { fault: "fp16", scaling: "dynamic" }, { fault: "logits" }] as const) {
      const run = trainTinyExtended(runOptions(settings(patch)));
      for (const point of stepPoints(run)) {
        for (const value of [point.loss, point.norm, point.maxLogit, point.logPartition, point.ratio]) {
          expect(finite(value), JSON.stringify(patch)).toBe(true);
        }
      }
    }
  });

  it("leaves a gap on the halted step instead of its broken numbers", () => {
    const run = trainTinyExtended(runOptions(settings({ fault: "rate" })));
    expect(run.haltedAt).toBe(154);
    expect(Number.isFinite(run.trace[153].loss)).toBe(false);
    const points = stepPoints(run);
    expect(points).toHaveLength(154);
    expect(points[153]).toMatchObject({ step: 154, loss: null, norm: null, maxLogit: null, ratio: null });
    expect(points[152].norm).toBeGreaterThan(1e18);
  });

  it("leaves a gap on an overflow, whether it halts the run or is skipped", () => {
    const fixed = stepPoints(trainTinyExtended(runOptions(settings({ fault: "fp16" }))));
    expect(fixed.at(-1)).toMatchObject({ step: 33, overflow: true, norm: null });
    const dynamic = stepPoints(trainTinyExtended(runOptions(settings({ fault: "fp16", scaling: "dynamic" }))));
    const skipped = dynamic.filter((point) => point.skipped);
    expect(skipped.map((point) => point.step)).toEqual([33, 62, 184, 430]);
    for (const point of skipped) expect([point.loss, point.norm, point.ratio]).toEqual([null, null, null]);
  });

  it("keeps whole-corpus checkpoints every 4 steps and drops any that are not numbers", () => {
    const healthy = trainTinyExtended(runOptions(settings({})));
    const corpus = corpusPoints(healthy.checkpoints);
    expect(corpus).toHaveLength(121);
    expect(corpus.slice(0, 3).map((point) => point.step)).toEqual([0, 4, 8]);
    const halted = trainTinyExtended(runOptions(settings({ fault: "rate" })));
    expect(corpusPoints(halted.checkpoints).every((point) => Number.isFinite(point.loss))).toBe(true);
    expect(corpusPoints(halted.checkpoints).at(-1)?.step).toBe(152);
  });
});

describe("bands and summaries", () => {
  it("takes nearest-rank quantiles of finite values", () => {
    expect(quantile([5, 1, 3, Number.NaN, 2, 4], 0.5)).toBe(3);
    expect(quantile([1, 2, 3, 4, 5], 0)).toBe(1);
    expect(quantile([1, 2, 3, 4, 5], 1)).toBe(5);
    expect(quantile([], 0.5)).toBeNull();
    expect(quantile([1, 2], 7)).toBe(2);
  });

  it("shades the 5th to 95th percentile of a run as its normal range", () => {
    const points = stepPoints(trainTinyExtended(runOptions(settings({}))));
    const band = normalBand(points, "norm");
    const norms = values(points, "norm");
    const inside = norms.filter((value) => band && value >= band.low && value <= band.high).length;
    expect(inside / norms.length).toBeGreaterThan(0.88);
    expect(inside / norms.length).toBeLessThan(0.92);
    expect(normalBand([], "loss")).toBeNull();
  });

  it("summarizes a halted run without inventing a final loss", () => {
    const summary = summarizeRun(trainTinyExtended(runOptions(settings({ fault: "rate" }))));
    expect(summary.finalLoss).toBeNull();
    expect(summary.haltedAt).toBe(154);
    expect(summary.peakNorm).toBeGreaterThan(1e18);
    expect(Number.isFinite(summary.peakLogit ?? Number.NaN)).toBe(true);
  });

  it("separates dynamic-scaling skips and tracks the loss scale", () => {
    const summary = summarizeRun(trainTinyExtended(runOptions(settings({ fault: "fp16", scaling: "dynamic" }))));
    expect([summary.skipped, summary.overflow, summary.firstOverflow]).toEqual([4, 4, 33]);
    expect(summary.overflowSkips).toEqual([33, 62, 184, 430]);
    expect(summary.endScale).toBe(2 ** 17);
    const rollback = summarizeRun(trainTinyExtended(runOptions(settings({ fault: "bad-batch", rollback: true }))));
    expect([rollback.skipped, rollback.overflowSkips.length, rollback.faulty]).toEqual([5, 0, 5]);
  });

  it("groups consecutive steps into ranges", () => {
    const points = [1, 2, 3, 7, 9, 10].map((step) => ({ step, skipped: true }) as never);
    expect(stepRanges(points, () => true)).toEqual([
      [1, 3],
      [7, 7],
      [9, 10],
    ]);
  });
});

describe("log scales", () => {
  it("covers the data in whole decades within the floor and ceiling", () => {
    expect(logDomain([0.11, 0.95])).toEqual({ low: -1, high: 1 });
    expect(logDomain([0.079, 1.1e19])).toEqual({ low: -2, high: 20 });
    expect(logDomain([1e-9, 5], { floor: -3 })).toEqual({ low: -3, high: 1 });
    expect(logDomain([1e60], { ceiling: 40 })).toEqual({ low: 38, high: 40 });
    expect(logDomain([], { floor: -2 })).toEqual({ low: -2, high: 0 });
    expect(logDomain([0, -5, Number.NaN, Infinity], { floor: -4 })).toEqual({ low: -4, high: -2 });
  });

  it("pins values outside the axis to its edges and keeps gaps as gaps", () => {
    const domain = { low: -2, high: 2 };
    expect(logPosition(1, domain)).toBeCloseTo(0.5, 12);
    expect(logPosition(1e9, domain)).toBe(1);
    expect(logPosition(1e-9, domain)).toBe(0);
    expect(logPosition(0, domain)).toBe(0);
    expect(logPosition(null, domain)).toBeNull();
    expect(logPosition(Number.NaN, domain)).toBeNull();
  });

  it("always labels both ends of the axis and thins the decades between", () => {
    expect(decadeTicks({ low: -1, high: 1 })).toEqual([-1, 0, 1]);
    expect(decadeTicks({ low: -2, high: 20 })).toEqual([-2, 4, 10, 16, 20]);
    expect(decadeTicks({ low: -3, high: 1 })).toEqual([-3, -2, -1, 0, 1]);
    for (const domain of [
      { low: -4, high: 40 },
      { low: 0, high: 1 },
      { low: -2, high: 39 },
    ]) {
      const ticks = decadeTicks(domain);
      expect(ticks[0]).toBe(domain.low);
      expect(ticks.at(-1)).toBe(domain.high);
      expect(ticks.length).toBeLessThanOrEqual(6);
    }
  });
});

describe("formatting", () => {
  it("prints readable numbers and a dash for anything that is not a number", () => {
    expect(formatNumber(1.1e19)).toBe("1.1 × 10¹⁹");
    expect(formatNumber(3.1e38)).toBe("3.1 × 10³⁸");
    expect(formatNumber(0.0264)).toBe("0.026");
    expect(formatNumber(0.31)).toBe("0.310");
    expect(formatNumber(0.00256)).toBe("0.0026");
    expect(formatNumber(395.03)).toBe("395");
    expect(formatNumber(19.66)).toBe("19.7");
    expect(formatNumber(65504)).toBe("65,504");
    expect(formatNumber(2.84e-14, 1)).toBe("2.8 × 10⁻¹⁴");
    expect(formatNumber(65520 / 2 ** 20, 2)).toBe("0.062");
    expect(formatNumber(0)).toBe("0");
    for (const broken of [Number.NaN, Infinity, -Infinity, null, undefined]) expect(formatNumber(broken)).toBe("—");
    expect(formatLoss(2.0394)).toBe("2.039");
    expect(formatLoss(Number.NaN)).toBe("—");
  });

  it("writes exponents as superscripts", () => {
    expect(superscript(-14)).toBe("⁻¹⁴");
    expect(powerOfTen(0)).toBe("1");
    expect(powerOfTen(1)).toBe("10");
    expect(powerOfTen(-3)).toBe("10⁻³");
    expect(powerOfTwo(20)).toBe("2²⁰");
    expect(formatCoefficient(0)).toBe("off");
    expect(formatCoefficient(0.0001)).toBe("0.0001");
  });
});
