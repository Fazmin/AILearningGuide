import { describe, expect, it } from "vitest";
import { hydrateReasoningState } from "./module";
import { binomialPmf, majorityAccuracy, verifierAccuracy, winGivenCorrect } from "./voting";

describe("sampling arithmetic", () => {
  it("builds a normalized binomial", () => {
    const pmf = binomialPmf(4, 0.5);
    expect(pmf).toHaveLength(5);
    expect(pmf[2]).toBeCloseTo(0.375, 12);
    expect(pmf.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 12);
  });

  it("gives pass@n for a perfect verifier", () => {
    expect(verifierAccuracy(8, 0.4)).toBeCloseTo(1 - 0.6 ** 8, 12);
  });

  it("reduces to the binomial majority with one wrong answer", () => {
    // P(Bin(9, 0.6) >= 5)
    expect(majorityAccuracy(9, 0.6, 1)).toBeCloseTo(0.73343, 4);
    expect(majorityAccuracy(16, 0.4, 1)).toBeLessThan(0.4);
  });

  it("never beats a single sample at n = 2", () => {
    for (const wrong of [1, 3, "unique"] as const) expect(majorityAccuracy(2, 0.3, wrong)).toBeCloseTo(0.3, 12);
  });

  it("breaks ties at random and counts exactly", () => {
    // two correct against two copies of one wrong answer: a two-way tie
    expect(winGivenCorrect(4, 2, 1)).toBeCloseTo(0.5, 12);
    expect(winGivenCorrect(3, 1, "unique")).toBeCloseTo(1 / 3, 12);
    expect(winGivenCorrect(3, 2, 3)).toBe(1);
    // one correct vote against two wrong votes over 2 answers: win only when they split, then a 3-way tie
    expect(winGivenCorrect(3, 1, 2)).toBeCloseTo(0.5 * (1 / 3), 12);
  });

  it("matches the lesson's quoted values", () => {
    expect(majorityAccuracy(16, 0.4, 3)).toBeCloseTo(0.708, 3);
    expect(majorityAccuracy(16, 0.4, 1)).toBeCloseTo(0.213, 3);
    expect(majorityAccuracy(8, 0.4, 3)).toBeCloseTo(0.58, 2);
  });
});

describe("reasoning state", () => {
  it("hydrates a version 1 payload", () => {
    expect(hydrateReasoningState(JSON.stringify({ problem: "add", budget: 9 }))).toMatchObject({
      problem: "add",
      budget: 6,
      samples: 8,
      accuracy: 0.4,
      spread: "three",
    });
  });
});
