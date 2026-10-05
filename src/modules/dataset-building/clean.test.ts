import { describe, expect, it } from "vitest";
import { encodeTinyText, tinyCrossEntropy, trainTinyModel } from "@app/module-sdk";
import {
  funnel,
  jaccard,
  LEAKED_LINE,
  longestSharedSpan,
  minhashEstimate,
  overlapsHeldOut,
  RAW_LINES,
  runPipeline,
  scrapeLines,
  trigramStats,
  UNLEAKED_TEXT,
  VALIDATION_TEXT,
  VARIANT_PAIR,
  type PipelineOptions,
} from "./clean";
import { hydrateDatasetState } from "./module";

const defaults: PipelineOptions = {
  dropBoilerplate: true,
  dropExact: true,
  dropNear: false,
  decontaminate: false,
  leak: false,
  minLength: 0,
};
const training = { epochs: 25, batchSize: 16, learningRate: 0.6, seed: 4 } as const;

const kept = (options: PipelineOptions) =>
  runPipeline(options)
    .filter((entry) => entry.verdict === "kept")
    .map((entry) => entry.line);
const heldOut = (text: string, scored = VALIDATION_TEXT) =>
  tinyCrossEntropy(trainTinyModel({ text, ...training }).weights, scored);

describe("the cleaning pipeline", () => {
  it("assigns the verdicts the lesson describes at the defaults", () => {
    const results = runPipeline(defaults);
    const count = (verdict: string) => results.filter((entry) => entry.verdict === verdict).length;
    expect(results).toHaveLength(20);
    expect(count("boilerplate")).toBe(9);
    expect(count("duplicate")).toBe(1);
    expect(count("kept")).toBe(10);
    expect(encodeTinyText(kept(defaults).join(" ")).length).toBe(516);
  });

  it("lets exact dedup absorb the boilerplate copies when boilerplate is off", () => {
    const results = runPipeline({ ...defaults, dropBoilerplate: false });
    expect(results.filter((entry) => entry.verdict === "duplicate")).toHaveLength(7);
    expect(encodeTinyText(kept({ ...defaults, dropBoilerplate: false }).join(" ")).length).toBe(623);
  });

  it("catches the harbour spelling variant only as a near-duplicate", () => {
    const results = runPipeline({ ...defaults, dropNear: true });
    const near = results.filter((entry) => entry.verdict === "near");
    expect(near).toHaveLength(1);
    expect(near[0].line).toContain("harbour");
    expect(near[0].match?.similarity).toBeCloseTo(0.915, 3);
  });

  it("starts eating real sentences only above a 56-character floor", () => {
    expect(kept({ ...defaults, minLength: 12 })).toHaveLength(8);
    expect(kept({ ...defaults, minLength: 56 })).toHaveLength(8);
    expect(kept({ ...defaults, minLength: 57 })).toHaveLength(7);
  });

  it("builds a funnel whose rows account for every removed line", () => {
    const rows = funnel(runPipeline(defaults), defaults);
    expect(rows.map((row) => row.lines)).toEqual([20, 11, 11, 10, 10, 10]);
    expect(rows[0].chars).toBe(899);
    expect(rows[1].chars).toBe(580);
    expect(rows.at(-1)?.chars).toBe(516);
    const removed = rows.slice(1).reduce((sum, row) => sum + row.removedLines, 0);
    expect(removed).toBe(20 - 10);
  });
});

describe("trigram statistics", () => {
  it("matches the quoted diversity and repetition", () => {
    const raw = trigramStats(RAW_LINES.join(" "));
    const clean = trigramStats(kept(defaults).join(" "));
    expect(raw.distinct).toBe(366);
    expect(clean.distinct).toBe(285);
    expect(Math.round((1 - raw.ratio) * 100)).toBe(60);
    expect(Math.round((1 - clean.ratio) * 100)).toBe(45);
  });
});

describe("MinHash", () => {
  it("estimates the exact Jaccard of the spelling-variant pair", () => {
    const exact = jaccard(VARIANT_PAIR[0], VARIANT_PAIR[1]);
    expect(exact).toBeCloseTo(0.915, 3);
    expect(Math.abs(minhashEstimate(VARIANT_PAIR[0], VARIANT_PAIR[1], 64) - exact)).toBeLessThan(0.02);
    expect(minhashEstimate(VARIANT_PAIR[0], VARIANT_PAIR[0], 64)).toBe(1);
  });

  it("gets closer on average with more hash functions", () => {
    const lines = [...new Set(RAW_LINES)];
    const error = (count: number) => {
      let total = 0;
      let pairs = 0;
      for (let i = 0; i < lines.length; i += 1) {
        for (let j = 0; j < i; j += 1) {
          total += Math.abs(minhashEstimate(lines[i], lines[j], count) - jaccard(lines[i], lines[j]));
          pairs += 1;
        }
      }
      return total / pairs;
    };
    expect(error(256)).toBeLessThan(error(16));
  });
});

describe("leakage and decontamination", () => {
  it("finds no long overlap between the clean scrape and the held-out text", () => {
    expect(longestSharedSpan(kept(defaults).join(" "), VALIDATION_TEXT).length).toBe(2);
    expect(RAW_LINES.some((line) => overlapsHeldOut(line))).toBe(false);
  });

  it("flags the leaked sentence and nothing else", () => {
    const leaked = { ...defaults, leak: true };
    expect(scrapeLines(true)).toHaveLength(21);
    expect(longestSharedSpan(kept(leaked).join(" "), VALIDATION_TEXT).length).toBe(10);
    const results = runPipeline({ ...leaked, decontaminate: true });
    const flagged = results.filter((entry) => entry.verdict === "contaminated");
    expect(flagged.map((entry) => entry.line)).toEqual([LEAKED_LINE]);
    expect(kept({ ...leaked, decontaminate: true })).toEqual(kept(defaults));
  });

  it("improves the held-out score while the untouched sentences get worse", () => {
    const clean = kept(defaults).join(" ");
    const leakedText = kept({ ...defaults, leak: true }).join(" ");
    expect(heldOut(clean)).toBeCloseTo(2.3, 3);
    expect(heldOut(leakedText)).toBeCloseTo(2.245, 3);
    expect(heldOut(clean, UNLEAKED_TEXT)).toBeCloseTo(2.397, 3);
    expect(heldOut(leakedText, UNLEAKED_TEXT)).toBeCloseTo(2.422, 3);
  });
});

describe("held-out scores quoted in the lesson", () => {
  it("reproduces the walk through the switches", () => {
    const score = (options: PipelineOptions) => heldOut(kept(options).join(" "));
    expect(heldOut(RAW_LINES.join(" "))).toBeCloseTo(2.365, 3);
    expect(score({ ...defaults, dropExact: false })).toBeCloseTo(2.306, 3);
    expect(score(defaults)).toBeCloseTo(2.3, 3);
    expect(score({ ...defaults, dropNear: true })).toBeCloseTo(2.312, 3);
    expect(score({ ...defaults, dropNear: true, minLength: 20 })).toBeCloseTo(2.316, 3);
    expect(score({ ...defaults, minLength: 20 })).toBeCloseTo(2.296, 3);
    expect(score({ ...defaults, dropBoilerplate: false })).toBeCloseTo(2.292, 3);
  });
});

describe("state hydration", () => {
  it("migrates a version 1 payload with the new switches off", () => {
    const legacy = JSON.stringify({
      dropBoilerplate: false,
      dropExact: true,
      dropNear: true,
      minLength: 20,
      view: "clean",
    });
    expect(hydrateDatasetState(legacy)).toEqual({
      dropBoilerplate: false,
      dropExact: true,
      dropNear: true,
      decontaminate: false,
      leak: false,
      minLength: 20,
      view: "clean",
      context: 3,
    });
    expect(hydrateDatasetState("{").leak).toBe(false);
  });
});
