import { describe, expect, it } from "vitest";
import { tinyRandom } from "@app/module-sdk";
import {
  completionCurve,
  completionProbability,
  expectedStepsRun,
  gatePosition,
  halfLifeSteps,
  percent,
  simulateCompletion,
} from "./reliability";

const gate = (after: number, catchRate = 1) => ({ after, catchRate });

describe("completion probability, closed form", () => {
  it("is the per-step success raised to the number of steps", () => {
    expect(completionProbability(0.95, 20)).toBeCloseTo(0.95 ** 20, 12);
    expect(percent(completionProbability(0.95, 20))).toBe("35.8%");
    expect(percent(completionProbability(0.99, 20))).toBe("81.8%");
    expect(completionProbability(0.99, 20) / completionProbability(0.95, 20)).toBeGreaterThan(2);
    expect(completionProbability(1, 30)).toBe(1);
    expect(completionProbability(0.5, 1)).toBe(0.5);
  });

  it("gives a perfect reviewer's gate the chance of running the prefix twice", () => {
    const s = 0.95 ** 10;
    expect(completionProbability(0.95, 20, gate(10))).toBeCloseTo((1 - (1 - s) ** 2) * 0.95 ** 10, 12);
    expect(percent(completionProbability(0.95, 20, gate(10)))).toBe("50.2%");
    expect(percent(completionProbability(0.95, 20, gate(1)))).toBe("37.6%");
    expect(percent(completionProbability(0.95, 20, gate(20)))).toBe("58.8%");
    expect(percent(completionProbability(0.95, 20, gate(10, 0.5)))).toBe("43.0%");
  });

  it("is no better than no gate when the reviewer catches nothing, and never worse otherwise", () => {
    expect(completionProbability(0.95, 20, gate(10, 0))).toBeCloseTo(completionProbability(0.95, 20), 12);
    for (const k of [1, 5, 10, 20]) {
      expect(completionProbability(0.9, 20, gate(k, 0.5))).toBeGreaterThanOrEqual(completionProbability(0.9, 20));
    }
  });

  it("protects more steps the later the gate sits, and treats a gate past the end as a gate at the end", () => {
    const at = (k: number) => completionProbability(0.95, 20, gate(k));
    expect(at(1)).toBeLessThan(at(10));
    expect(at(10)).toBeLessThan(at(20));
    expect(at(25)).toBe(at(20));
    expect(gatePosition(20, gate(25))).toBe(20);
    expect(gatePosition(20, gate(0))).toBe(0);
    expect(gatePosition(20, null)).toBe(0);
    expect(completionProbability(0.95, 5, gate(10))).toBe(completionProbability(0.95, 5, gate(5)));
  });

  it("counts the redone steps as the cost of a gate", () => {
    expect(expectedStepsRun(0.95, 20)).toBe(20);
    expect(expectedStepsRun(0.95, 20, gate(10)).toFixed(1)).toBe("24.0");
    expect((expectedStepsRun(0.95, 20, gate(10)) - 20).toFixed(1)).toBe("4.0");
    expect(expectedStepsRun(0.95, 20, gate(10, 0))).toBe(20);
    expect(expectedStepsRun(0.95, 20, gate(1)).toFixed(2)).toBe("20.05");
    expect(expectedStepsRun(0.95, 20, gate(20)).toFixed(1)).toBe("32.8");
    expect(expectedStepsRun(1, 20, gate(10))).toBe(20);
  });

  it("finds the longest run that finishes at least half the time", () => {
    expect([halfLifeSteps(0.95), halfLifeSteps(0.99), halfLifeSteps(0.9), halfLifeSteps(0.5)]).toEqual([13, 68, 6, 1]);
    expect(0.95 ** 13).toBeGreaterThanOrEqual(0.5);
    expect(0.95 ** 14).toBeLessThan(0.5);
    expect(halfLifeSteps(1)).toBeNull();
  });

  it("stays finite for any input the sliders could never produce", () => {
    for (const bad of [Number.NaN, Infinity, -Infinity, 1e9, -1e9]) {
      for (const value of [
        completionProbability(bad, bad, gate(bad, bad)),
        expectedStepsRun(bad, bad, gate(bad, bad)),
        completionProbability(0.95, bad),
      ]) {
        expect(Number.isFinite(value)).toBe(true);
      }
    }
  });

  it("draws one point per run length, with a gated value only when there is a gate", () => {
    const plain = completionCurve(0.95, null);
    expect(plain).toHaveLength(30);
    expect(plain[0]).toEqual({ steps: 1, plain: 0.95, gated: null });
    expect(plain[19].plain).toBeCloseTo(0.95 ** 20, 12);
    const gated = completionCurve(0.95, gate(10));
    expect(gated[19].gated).toBeCloseTo(completionProbability(0.95, 20, gate(10)), 12);
    expect(gated.every((point) => point.gated !== null && point.gated >= point.plain)).toBe(true);
  });
});

describe("closed form against a seeded simulation", () => {
  const runs = 50000;
  it.each([
    ["no gate", 0.95, 20, null],
    ["gate after step 10, perfect reviewer", 0.95, 20, gate(10)],
    ["gate after step 10, reviewer catches half", 0.95, 20, gate(10, 0.5)],
    ["gate after step 1", 0.9, 12, gate(1)],
    ["gate past the end", 0.97, 8, gate(30, 0.8)],
  ])("%s agrees within one percentage point", (_, p, steps, g) => {
    const exact = completionProbability(p, steps, g);
    const simulated = simulateCompletion(p, steps, g, runs, tinyRandom(7));
    expect(Math.abs(simulated - exact)).toBeLessThan(0.01);
  });

  it("is reproducible from its seed", () => {
    expect(simulateCompletion(0.95, 20, gate(10), 2000, tinyRandom(3))).toBe(
      simulateCompletion(0.95, 20, gate(10), 2000, tinyRandom(3)),
    );
  });
});
