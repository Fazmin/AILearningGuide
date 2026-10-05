import { describe, expect, it } from "vitest";
import { checkpointQuestions } from "@app/module-sdk";
import definition, { migrateEthicsState } from "./module";
import {
  DEPLOYMENTS,
  GROUP_A,
  GROUP_B,
  HOLDOUT_SHARE_B,
  accuracyOf,
  breakEvenDays,
  equalOpportunityGap,
  falsePositiveGap,
  falsePositiveRate,
  megawattHours,
  mixOutcome,
  servingFlopsPerDay,
  trainingFlops,
  truePositiveRate,
} from "./impact";

describe("deployment claims", () => {
  it("has four deployments, five claims each, and at most one fully measured", () => {
    expect(DEPLOYMENTS.map((deployment) => deployment.value)).toEqual(["school", "news", "clinic", "image"]);
    for (const deployment of DEPLOYMENTS) {
      expect(deployment.claims).toHaveLength(5);
      expect(deployment.claims.filter((claim) => claim.measured === "yes").length).toBeLessThanOrEqual(1);
    }
  });

  it("gives the image generator consent, likeness, and provenance claims, and one benchmark claim answered by an Against line", () => {
    const image = DEPLOYMENTS.find((deployment) => deployment.value === "image")!;
    expect(image.label).toBe("Image generator");
    const themes = image.claims.map((claim) => claim.theme);
    expect(themes).toEqual(expect.arrayContaining(["consent", "likeness", "provenance"]));
    expect(image.claims.filter((claim) => claim.measured === "yes")).toHaveLength(1);
    expect(image.claims.filter((claim) => claim.side === "For")).toHaveLength(2);
    // The lesson says an Against line already answers the benchmark claim in three of the four deployments.
    const answered = DEPLOYMENTS.filter((deployment) => {
      const benchmark = deployment.claims.findIndex((claim) => claim.measured === "yes");
      return (
        benchmark >= 0 && deployment.claims.some((claim) => claim.side === "Against" && claim.measured === "partly")
      );
    });
    expect(answered.map((deployment) => deployment.value)).toEqual(["school", "clinic", "image"]);
  });
});

describe("disaggregated evaluation", () => {
  it("computes the group accuracies the card quotes", () => {
    expect(accuracyOf(GROUP_A)).toBeCloseTo(0.94, 12);
    expect(accuracyOf(GROUP_B)).toBeCloseTo(0.854, 12);
  });

  it("barely moves accuracy while missed urgent cases double", () => {
    const holdout = mixOutcome(HOLDOUT_SHARE_B);
    const half = mixOutcome(0.5);
    expect(holdout.accuracy).toBeCloseTo(0.9314, 4);
    expect(holdout.misses).toBeCloseTo(27, 9);
    expect(half.accuracy).toBeCloseTo(0.897, 4);
    expect(half.misses).toBeCloseTo(55, 9);
    expect(half.shareOfMissesB).toBeCloseTo(0.818, 3);
    const standard = mixOutcome(0.3);
    expect(standard.accuracy).toBeCloseTo(0.9142, 4);
    expect(Math.round(standard.misses)).toBe(41);
    expect(Math.round(standard.shareOfMissesB * 100)).toBe(66);
  });
});

describe("fairness criteria the triage card names", () => {
  it("reads the true-positive and false-positive rates off the invented group rates", () => {
    expect(truePositiveRate(GROUP_A)).toBeCloseTo(0.9, 12);
    expect(truePositiveRate(GROUP_B)).toBeCloseTo(0.55, 12);
    expect(falsePositiveRate(GROUP_A)).toBeCloseTo(0.05, 12);
    expect(falsePositiveRate(GROUP_B)).toBeCloseTo(0.07, 12);
  });

  it("puts the equal-opportunity gap at 35 points and the false-positive gap at 2, whatever the group mix", () => {
    expect(equalOpportunityGap()).toBeCloseTo(0.35, 12);
    expect(falsePositiveGap()).toBeCloseTo(0.02, 12);
    // Neither gap depends on the share of group B, unlike accuracy and the miss counts.
    expect(mixOutcome(0.1).accuracy).not.toBeCloseTo(mixOutcome(0.5).accuracy, 3);
    expect(Math.round(equalOpportunityGap() * 100)).toBe(35);
    expect(Math.round(falsePositiveGap() * 100)).toBe(2);
    // The model is much further from equal opportunity than from equal false-positive rates.
    expect(equalOpportunityGap()).toBeGreaterThan(10 * falsePositiveGap());
  });

  it("makes the accuracies the question quotes: within nine points of each other", () => {
    expect(Math.abs(accuracyOf(GROUP_A) - accuracyOf(GROUP_B))).toBeCloseTo(0.086, 12);
    expect(Math.abs(accuracyOf(GROUP_A) - accuracyOf(GROUP_B))).toBeLessThan(0.09);
  });
});

describe("ethics checkpoint questions that rest on those numbers", () => {
  const questions = checkpointQuestions(definition);
  const correct = (index: number) => questions[index].options[questions[index].answer];

  it("question 1 now speaks of four deployments", () => {
    expect(questions[0].prompt).toContain("four deployments");
  });

  it("question 5 states the rates the card prints", () => {
    expect(questions[4].prompt).toContain("90% of urgent cases in group A and 55% in group B");
    expect(questions[4].prompt).toContain("5% and 7%");
    expect(correct(4)).toMatch(/^It breaks equal opportunity, which asks for equal true-positive rates/);
    expect(questions[4].explanation).toContain("35 points");
    expect(questions[4].explanation).toContain("2 points");
  });

  it("question 6 rests on Annex III listing emergency patient triage, and carries no date", () => {
    expect(correct(5)).toMatch(/^It sets duties by use/);
    expect(questions[5].explanation).not.toMatch(/\b20\d\d\b/);
  });
});

describe("training versus serving", () => {
  it("breaks even after 3D over tokens per day, independent of size", () => {
    expect(breakEvenDays(2e12, 1e10)).toBeCloseTo(600, 9);
    expect(breakEvenDays(2e12, 1e12)).toBeCloseTo(6, 9);
    for (const n of [1e9, 7e9, 405e9]) {
      expect(trainingFlops(n, 2e12) / servingFlopsPerDay(n, 1e10)).toBeCloseTo(600, 9);
    }
  });

  it("reproduces the order of magnitude Meta reported for Llama 2 7B", () => {
    const flops = trainingFlops(7e9, 2e12);
    expect(Math.round(megawattHours(flops))).toBe(75);
    const gpuHours = flops / (312e12 * 0.4) / 3600;
    expect(Math.abs(gpuHours - 184320) / 184320).toBeLessThan(0.02);
  });
});

describe("ethics-and-impact state", () => {
  it("accepts the new image-generator deployment and still reads older payloads", () => {
    expect(definition.hydrateState(JSON.stringify({ deployment: "image", claim: 4 }))).toMatchObject({
      deployment: "image",
      claim: 4,
    });
    expect(migrateEthicsState({ deployment: "news", claim: 1, shareB: 0.2, paramIndex: 3, trainTokens: 2, servedLog: 9 })).toMatchObject({
      deployment: "news",
      claim: 1,
    });
  });

  it("keeps version 1 keys and adds defaults", () => {
    expect(migrateEthicsState({ deployment: "clinic", claim: 3 })).toMatchObject({
      deployment: "clinic",
      claim: 3,
      shareB: 0.3,
    });
    expect(definition.hydrateState(JSON.stringify({ deployment: "mars", claim: 99 }))).toMatchObject({
      deployment: "school",
      claim: 4,
    });
  });
});
