import { describe, expect, it } from "vitest";
import {
  acceptanceOf,
  countTransitions,
  fitCounts,
  realWordShare,
  runChain,
  runFamily,
  sampleIds,
  SEED_COUNTS,
  SEED_IDS,
  SEED_WORDS,
  seedCoverage,
  seedWeightedEntropy,
  supportSize,
  tailLoss,
  verifyWords,
} from "./collapse";
import { hydrateSyntheticState } from "./module";
import { tinyRandom } from "@app/module-sdk";

const defaults = { temperature: 1, filter: "none", sampleSize: 500 } as const;

describe("the exact bigram fit", () => {
  it("normalizes every row and reproduces the seed's transitions", () => {
    const model = fitCounts(SEED_COUNTS);
    for (let row = 0; row < 30; row += 1) {
      let total = 0;
      for (let next = 0; next < 30; next += 1) total += model[row * 30 + next];
      expect(total).toBeCloseTo(1, 12);
    }
    expect(SEED_IDS.length).toBe(499);
    expect(supportSize(SEED_COUNTS)).toBe(157);
    expect(seedCoverage(SEED_COUNTS)).toBe(1);
    expect(seedWeightedEntropy(model)).toBeCloseTo(1.787, 3);
    expect(SEED_WORDS.size).toBe(57);
  });

  it("samples only transitions the fit has seen, at any temperature", () => {
    const model = fitCounts(SEED_COUNTS);
    for (const temperature of [0.5, 1, 1.2]) {
      const counts = countTransitions(sampleIds(model, 3000, temperature, tinyRandom(3)));
      counts.forEach((count, index) => {
        if (count > 0) expect(SEED_COUNTS[index]).toBeGreaterThan(0);
      });
    }
  });
});

describe("recursive training on synthetic text only", () => {
  const family = runFamily({ regime: "replace", ...defaults });

  it("can never gain a transition, so support only shrinks", () => {
    for (const run of family.runs) {
      for (let g = 1; g < run.stats.length; g += 1) {
        expect(run.stats[g].support).toBeLessThanOrEqual(run.stats[g - 1].support);
      }
      run.counts.forEach((counts) =>
        counts.forEach((count, index) => {
          if (count > 0) expect(SEED_COUNTS[index]).toBeGreaterThan(0);
        }),
      );
    }
  });

  it("matches the values the lesson quotes at the default settings", () => {
    expect(family.mean[1].support).toBeCloseTo(125.1, 1);
    expect(family.mean[5].support).toBeCloseTo(75.3, 1);
    expect(family.mean[10].support).toBeCloseTo(51.1, 1);
    expect(family.mean[10].coverage).toBeCloseTo(0.589, 3);
    expect(family.mean[10].entropy).toBeCloseTo(0.912, 3);
    const final = family.runs.map((run) => run.stats[10].support);
    expect(Math.min(...final)).toBe(46);
    expect(Math.max(...final)).toBe(58);
  });

  it("loses rare transitions first", () => {
    const [once, few, common] = tailLoss(family.runs[0].counts[10]);
    expect([once.lost, once.total]).toEqual([63, 72]);
    expect([few.lost, few.total]).toEqual([35, 58]);
    expect([common.lost, common.total]).toEqual([8, 27]);
    expect(once.lost / once.total).toBeGreaterThan(common.lost / common.total);
  });

  it("collapses faster when sampling is sharpened or samples are small", () => {
    const sharp = runFamily({ regime: "replace", ...defaults, temperature: 0.8 });
    const tiny = runFamily({ regime: "replace", ...defaults, sampleSize: 250 });
    const large = runFamily({ regime: "replace", ...defaults, sampleSize: 2000 });
    expect(sharp.mean[10].support).toBeCloseTo(10.6, 1);
    expect(tiny.mean[10].support).toBeCloseTo(34.1, 1);
    expect(large.mean[10].support).toBeCloseTo(109.4, 1);
    const coldest = runFamily({ regime: "replace", ...defaults, temperature: 0.5 });
    expect(coldest.mean[5].support).toBe(4);
    expect(coldest.mean[5].entropy).toBeLessThan(0.001);
  });
});

describe("accumulating synthetic text on top of the seed", () => {
  it("keeps every seed transition and all of its coverage", () => {
    const family = runFamily({ regime: "accumulate", ...defaults });
    for (const point of family.mean) {
      expect(point.support).toBe(157);
      expect(point.coverage).toBe(1);
    }
    expect(family.mean[10].entropy).toBeCloseTo(1.713, 3);
  });

  it("still narrows when sampling is sharpened, just without losing support", () => {
    const family = runFamily({ regime: "accumulate", ...defaults, temperature: 0.8 });
    expect(family.mean[10].support).toBe(157);
    expect(family.mean[10].entropy).toBeCloseTo(1.341, 3);
  });
});

describe("the real-words verifier", () => {
  it("keeps only seed words", () => {
    expect(verifyWords("the fog zqx harbor. thar")).toBe("the fog harbor.");
    expect(realWordShare("the fog zqx")).toBeCloseTo(2 / 3, 10);
    expect(acceptanceOf("")).toBe(0);
  });

  it("raises quality while narrowing diversity, unless the seed is kept", () => {
    const replace = runFamily({ regime: "replace", ...defaults, filter: "words" });
    const accumulate = runFamily({ regime: "accumulate", ...defaults, filter: "words" });
    expect(replace.mean[1].keptShare).toBeLessThan(0.07);
    expect(replace.mean[10].realWords).toBeGreaterThan(0.99);
    expect(replace.mean[10].support).toBeCloseTo(16.0, 1);
    expect(accumulate.mean[10].support).toBe(157);
    expect(accumulate.mean[10].realWords).toBeCloseTo(0.82, 2);
  });
});

it("is reproducible from its seed", () => {
  const first = runChain({ regime: "replace", ...defaults, seed: 3 });
  const again = runChain({ regime: "replace", ...defaults, seed: 3 });
  expect(first.stats).toEqual(again.stats);
  expect(first.probes[4]).toBe(again.probes[4]);
});

describe("state hydration", () => {
  it("migrates the version 1 payload", () => {
    const legacy = JSON.stringify({ epochs: 20, strict: 0.4, mix: "mixed" });
    expect(hydrateSyntheticState(legacy)).toEqual({
      generation: 5,
      regime: "accumulate",
      measure: "support",
      temperature: 1,
      filter: "none",
      sampleSize: 500,
      generator: "self",
    });
    expect(hydrateSyntheticState(JSON.stringify({ mix: "synthetic" })).regime).toBe("replace");
  });

  it("migrates a version 2 payload: every old key keeps its meaning and the generator is the model itself", () => {
    const version2 = JSON.stringify({
      generation: 7,
      regime: "accumulate",
      measure: "entropy",
      temperature: 0.8,
      filter: "words",
      sampleSize: 2000,
    });
    expect(hydrateSyntheticState(version2)).toEqual({
      generation: 7,
      regime: "accumulate",
      measure: "entropy",
      temperature: 0.8,
      filter: "words",
      sampleSize: 2000,
      generator: "self",
    });
  });

  it("accepts only known generators and measures", () => {
    expect(hydrateSyntheticState(JSON.stringify({ generator: "teacher" })).generator).toBe("teacher");
    expect(hydrateSyntheticState(JSON.stringify({ generator: "oracle" })).generator).toBe("self");
    expect(hydrateSyntheticState(JSON.stringify({ generator: 3 })).generator).toBe("self");
    expect(hydrateSyntheticState(JSON.stringify({ measure: "heldOut" })).measure).toBe("heldOut");
    expect(hydrateSyntheticState(JSON.stringify({ measure: "accuracy" })).measure).toBe("support");
  });

  it("clamps malformed values", () => {
    const state = hydrateSyntheticState(
      JSON.stringify({ generation: 99, temperature: 9, sampleSize: 7, filter: "x" }),
    );
    expect(state.generation).toBe(10);
    expect(state.temperature).toBe(1.2);
    expect(state.sampleSize).toBe(500);
    expect(state.filter).toBe("none");
    expect(hydrateSyntheticState("not json").regime).toBe("replace");
  });
});
