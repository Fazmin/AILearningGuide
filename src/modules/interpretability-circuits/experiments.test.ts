/// <reference types="node" />
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { runTransformer, weightsFromOnnx } from "./engine";
import {
  cachedPatchSweep,
  controlPosition,
  copyPatchSweep,
  copyPositions,
  copyPrompt,
  copyPromptFromText,
  copyScore,
  expectedNext,
  headId,
  headScores,
  HELD_OUT_PERIODS,
  inductionStripe,
  patchEndpoints,
  PERIODS,
  referenceMeans,
  runWithAblation,
  singleHeadEffects,
  STABLE_MARGIN,
  topAt,
  type CachedPatchRecord,
} from "./experiments";
import { referenceStrings } from "./mean-ablation";
import cache from "./assets/interpretability-cache.json";

/**
 * Every figure the lesson, the card explanations and the checkpoint quote about the shipped model is
 * measured here with the repo's own forward pass, from the shipped ONNX file and cache. The percentages
 * are pinned to the one decimal the lesson prints. A retrained model changes these on purpose.
 */
const weights = weightsFromOnnx(
  readFileSync(resolve(process.cwd(), "src/modules/attention/assets/tiny-transformer.onnx")),
);
const LAYER_ONE = ["L1H1", "L1H2", "L1H3", "L1H4"];
const LAYER_TWO = ["L2H1", "L2H2", "L2H3", "L2H4"];
const pct = (value: number) => (value * 100).toFixed(1);
const two = (value: number) => value.toFixed(2);
const mean = (values: number[]) => values.reduce((sum, value) => sum + value, 0) / values.length;
const byId = (effects: { id: string; mean: number }[]) => Object.fromEntries(effects.map((entry) => [entry.id, entry.mean]));

/** Measurements shared by several tests, computed once. */
const cacheByPeriod = new Map<number, ReturnType<typeof measure>>();
function measure(period: number) {
  const prompt = copyPrompt(period);
  const run = runTransformer(weights, prompt.ids);
  return { prompt, run, score: copyScore(run, prompt), heads: headScores(run, prompt) };
}
const at = (period: number) => {
  if (!cacheByPeriod.has(period)) cacheByPeriod.set(period, measure(period));
  return cacheByPeriod.get(period)!;
};
const head = (period: number, id: string) => at(period).heads.find((entry) => headId(entry.layer, entry.head) === id)!;

describe("copy prompts", () => {
  it("repeats the pool and names the next letter and its previous occurrence", () => {
    const prompt = copyPrompt(8);
    expect(prompt.text).toBe("QXZRKWMPQXZRKWMPQXZRKWMP");
    expect(prompt.target).toBe("Q");
    expect(prompt.source).toBe(16);
    expect(prompt.text[prompt.source]).toBe(prompt.target);
    for (const period of PERIODS) {
      const other = copyPrompt(period);
      expect(other.text[other.source]).toBe(other.target);
      expect(other.text).toHaveLength(24);
    }
  });

  it("scores the second repeat onward, where an earlier occurrence exists", () => {
    const prompt = copyPrompt(8);
    expect(copyPositions(prompt)).toHaveLength(16);
    expect(copyPositions(prompt)[0]).toBe(8);
    expect(copyPositions(copyPrompt(13))).toHaveLength(11);
  });

  it("names the expected next letter for any periodic text, not only the lab's pool", () => {
    const prompt = copyPromptFromText("HGFEDCBAHGFEDCBAHGFEDCBA", 8);
    expect(prompt.target).toBe("H");
    expect(prompt.source).toBe(16);
    expect(expectedNext(prompt, 7)).toBe("H");
    expect(expectedNext(prompt, 9)).toBe("F");
    expect(expectedNext(prompt, 23)).toBe("H");
  });

  it("marks 11 and 13 as the periods the training windows never used", () => {
    expect([...HELD_OUT_PERIODS]).toEqual([11, 13]);
    for (const period of HELD_OUT_PERIODS) expect(PERIODS as readonly number[]).toContain(period);
  });

  it("puts the control letter in the first repeat, half a period from the letter being copied", () => {
    for (const period of PERIODS) {
      const prompt = copyPrompt(period);
      const position = controlPosition(prompt);
      expect(position).toBeGreaterThanOrEqual(0);
      expect(position).toBeLessThan(period);
      const distance = Math.abs(((position - prompt.source) % period + period) % period);
      expect(Math.min(distance, period - distance)).toBe(Math.floor(period / 2));
    }
  });
});

describe("copying (quoted in the lesson)", () => {
  it("copies at every period from 6 to 13, unseen ones included", () => {
    const accuracy = PERIODS.map((period) => at(period).score.mean);
    expect(pct(at(8).score.mean)).toBe("94.5");
    expect(pct(at(11).score.mean)).toBe("88.5");
    expect(pct(at(13).score.mean)).toBe("88.0");
    expect(pct(Math.min(...accuracy))).toBe("88.0");
    expect(pct(Math.max(...accuracy))).toBe("97.2");
    for (const period of PERIODS) {
      expect(at(period).score.finalTarget, `period ${period}`).toBeGreaterThan(0.97);
      expect(at(period).score.top[0].token).toBe(at(period).prompt.target);
    }
  });

  it("is weakest at the first letter of the second repeat, and at least 95% from the second letter on", () => {
    for (const period of PERIODS) {
      const { score } = at(period);
      expect(mean(score.perPosition.slice(1)), `period ${period}`).toBeGreaterThanOrEqual(0.95);
      expect(score.perPosition[0], `period ${period}`).toBeLessThan(score.mean);
    }
  });
});

describe("attention: the two halves of an induction circuit", () => {
  it("puts about two thirds of L2 H2 and L2 H3 on the induction key at period 8, and almost nothing elsewhere", () => {
    expect(two(head(8, "L2H2").induction)).toBe("0.66");
    expect(two(head(8, "L2H3").induction)).toBe("0.67");
    for (const entry of at(8).heads) {
      if (["L2H2", "L2H3"].includes(headId(entry.layer, entry.head))) continue;
      expect(entry.induction, headId(entry.layer, entry.head)).toBeLessThanOrEqual(0.06);
    }
  });

  it.each(PERIODS.map((period) => [period]))("keeps that split at period %i", (period) => {
    for (const entry of at(period).heads) {
      const id = headId(entry.layer, entry.head);
      if (id === "L2H2" || id === "L2H3") {
        expect(entry.induction, id).toBeGreaterThan(0.55);
        expect(entry.induction, id).toBeLessThan(0.85);
        // The stripe follows the period: the strongest single offset is period - 1.
        expect(entry.strongestOffset, `${id} strongest offset`).toBe(period - 1);
      } else {
        expect(entry.induction, id).toBeLessThanOrEqual(0.1);
      }
    }
    // The layer-1 half: previous-letter heads.
    expect(head(period, "L1H3").previous).toBeGreaterThanOrEqual(0.98);
    expect(head(period, "L1H4").previous).toBeGreaterThanOrEqual(0.99);
    for (const id of ["L1H1", "L1H2", "L2H2", "L2H3"]) expect(head(period, id).previous, id).toBeLessThan(0.05);
    for (const id of ["L2H1", "L2H4"]) expect(head(period, id).previous, id).toBeLessThan(0.27);
  });

  it("reports the induction share range the lesson quotes: 0.55 to 0.84", () => {
    const shares = PERIODS.flatMap((period) => ["L2H2", "L2H3"].map((id) => head(period, id).induction));
    expect(two(Math.min(...shares))).toBe("0.55");
    expect(two(Math.max(...shares))).toBe("0.84");
  });

  it("is unmoved by the period test: the stripe sits at period - 1 for the unseen 11 and 13 too", () => {
    for (const period of HELD_OUT_PERIODS) {
      expect(head(period, "L2H3").strongestOffset).toBe(period - 1);
      expect(head(period, "L2H3").induction).toBeGreaterThan(0.75);
    }
  });

  it("spreads over several earlier copies at short periods", () => {
    const { run, prompt } = at(6);
    const sequence = prompt.ids.length;
    for (const [id, nearest, all] of [
      ["L2H2", "0.55", "0.89"],
      ["L2H3", "0.56", "0.93"],
    ] as const) {
      const entry = at(6).heads.find((candidate) => headId(candidate.layer, candidate.head) === id)!;
      let total = 0;
      let count = 0;
      for (let query = prompt.period; query < sequence; query += 1) {
        const row = (entry.head * sequence + query) * sequence;
        for (let copy = 1; query - copy * prompt.period + 1 >= 0; copy += 1) {
          total += run.attention[entry.layer][row + query - copy * prompt.period + 1];
        }
        count += 1;
      }
      expect(two(entry.induction)).toBe(nearest);
      expect(two(total / count)).toBe(all);
    }
  });

  it("finds two heads that look at fixed offsets, which neither table column shows", () => {
    for (const period of PERIODS) {
      expect(head(period, "L1H2").strongestOffset).toBe(2);
      expect(head(period, "L1H2").strongestShare).toBeGreaterThanOrEqual(0.98);
      expect(head(period, "L1H1").strongestOffset).toBe(3);
      expect(head(period, "L1H1").strongestShare).toBeGreaterThan(0.5);
      expect(head(period, "L1H1").strongestShare).toBeLessThan(0.65);
    }
    expect(two(head(8, "L1H4").strongestShare)).toBe("1.00");
    expect(two(head(8, "L1H2").strongestShare)).toBe("0.99");
  });
});

describe("zero-ablation at period 8 (quoted)", () => {
  const prompt = copyPrompt(8);
  it("makes single heads necessary, unlike the first release", () => {
    const base = at(8).score.mean;
    const single = byId(singleHeadEffects(weights, prompt));
    expect(
      Object.fromEntries(Object.entries(single).map(([id, value]) => [id, pct(value)])),
    ).toEqual({
      L1H1: "89.2",
      L1H2: "5.1",
      L1H3: "56.6",
      L1H4: "9.9",
      L2H1: "95.8",
      L2H2: "50.2",
      L2H3: "6.0",
      L2H4: "99.6",
    });
    for (const id of ["L1H2", "L1H4", "L2H3"]) expect(base - single[id], id).toBeGreaterThan(0.8);
    for (const id of ["L1H3", "L2H2"]) expect(base - single[id], id).toBeGreaterThan(0.35);
    for (const id of ["L1H1", "L2H1"]) expect(Math.abs(base - single[id]), id).toBeLessThan(0.06);
  });

  it("raises accuracy when L2 H4 is removed, at every period", () => {
    for (const period of PERIODS) {
      const single = byId(singleHeadEffects(weights, copyPrompt(period)).filter((entry) => entry.id === "L2H4"));
      expect(single.L2H4 - at(period).score.mean, `period ${period}`).toBeGreaterThan(0.02);
    }
  }, 60000);

  it("collapses with a whole layer off", () => {
    expect(pct(copyScore(runWithAblation(weights, prompt, LAYER_ONE), prompt).mean)).toBe("1.9");
    expect(pct(copyScore(runWithAblation(weights, prompt, LAYER_TWO), prompt).mean)).toBe("0.4");
    expect(pct(copyScore(runWithAblation(weights, prompt, ["L2H2", "L2H3"]), prompt).mean)).toBe("1.3");
  });
});

describe("mean ablation at period 8 (quoted)", () => {
  const prompt = copyPrompt(8);
  const means = referenceMeans(weights, prompt);

  it("leaves more of the copying than zero does for the two induction heads, and keeps layer 1 costly", () => {
    const zero = byId(singleHeadEffects(weights, prompt));
    const mean = byId(singleHeadEffects(weights, prompt, "mean", means));
    expect(Object.fromEntries(Object.entries(mean).map(([id, value]) => [id, pct(value)]))).toEqual({
      L1H1: "95.1",
      L1H2: "27.4",
      L1H3: "48.8",
      L1H4: "24.3",
      L2H1: "94.8",
      L2H2: "75.9",
      L2H3: "47.9",
      L2H4: "95.3",
    });
    // The conclusion that survives: the same layer-1 heads stay costly, L1 H1 / L2 H1 / L2 H4 stay free.
    for (const id of ["L1H2", "L1H3", "L1H4"]) {
      expect(zero[id], id).toBeLessThan(0.6);
      expect(mean[id], id).toBeLessThan(0.5);
    }
    for (const id of ["L1H1", "L2H1", "L2H4"]) expect(mean[id], id).toBeGreaterThan(0.9);
    // What does not: the induction heads' solo roles shrink a lot.
    expect(mean.L2H3 - zero.L2H3).toBeGreaterThan(0.35);
    expect(mean.L2H2 - zero.L2H2).toBeGreaterThan(0.2);
  });

  it("still collapses with both induction heads, or a whole layer, off", () => {
    expect(pct(copyScore(runWithAblation(weights, prompt, ["L2H2", "L2H3"], "mean", means), prompt).mean)).toBe("2.4");
    expect(pct(copyScore(runWithAblation(weights, prompt, LAYER_ONE, "mean", means), prompt).mean)).toBe("1.9");
    expect(pct(copyScore(runWithAblation(weights, prompt, LAYER_TWO, "mean", means), prompt).mean)).toBe("2.8");
  });

  it("needs the reference means", () => {
    expect(() => runWithAblation(weights, prompt, ["L1H1"], "mean")).toThrow();
  });
});

describe("do the layer-1 heads feed the layer-2 induction heads? (quoted)", () => {
  const prompt = copyPrompt(8);
  const means = referenceMeans(weights, prompt);
  const stripe = (ablated: string[], mode: "zero" | "mean" = "zero") =>
    inductionStripe(runWithAblation(weights, prompt, ablated, mode, mode === "mean" ? means : undefined), prompt).map((entry) =>
      two(entry.induction),
    );

  it("cuts the induction heads' attention on the induction key when previous-letter heads go", () => {
    expect(stripe([])).toEqual(["0.66", "0.67"]);
    expect(stripe(["L1H4"])).toEqual(["0.19", "0.24"]);
    expect(stripe(["L1H3", "L1H4"])[0]).toBe("0.01");
    expect(stripe(["L1H3", "L1H4"])[1]).toBe("0.03");
    expect(stripe(["L1H3", "L1H4"], "mean").map(Number).every((value) => value < 0.06)).toBe(true);
    expect(stripe(["L1H4"], "mean").map(Number).every((value) => value > 0.2 && value < 0.35)).toBe(true);
  });

  it("is not only the previous-letter heads: L1 H2 cuts it too, and L1 H1 barely does", () => {
    expect(stripe(["L1H2"])).toEqual(["0.30", "0.21"]);
    expect(Number(stripe(["L1H1"])[0])).toBeGreaterThan(0.6);
  });

  it("leaves the rest of layer 2 alone: heads in one layer do not read each other", () => {
    expect(stripe(["L2H3"])).toEqual(stripe([]));
  });

  it("does not move the induction heads' attention to the previous letter when L1 H4 is off", () => {
    const run = runWithAblation(weights, prompt, ["L1H4"]);
    const heads = headScores(run, prompt);
    for (const id of ["L2H2", "L2H3"]) {
      const entry = heads.find((candidate) => headId(candidate.layer, candidate.head) === id)!;
      expect(entry.previous, id).toBeLessThan(0.05);
      expect(entry.strongestOffset, id).toBe(7);
    }
  });
});

describe("the ablation split on random strings the lab never shows", () => {
  /** Seeded random-letter repeats, from an independent batch of seeds (the reference means use the default ones). */
  const randomPrompts = (period: number) =>
    referenceStrings(period, 24, 3, 100000).map((text) => copyPromptFromText(text, period));

  it.each([9, 13])("holds at period %i", (period) => {
    const prompts = randomPrompts(period);
    expect(new Set(prompts.map((prompt) => prompt.text)).size).toBe(prompts.length);
    const means = referenceMeans(weights, prompts[0]);
    for (const prompt of prompts) {
      const base = copyScore(runTransformer(weights, prompt.ids), prompt).mean;
      expect(base).toBeGreaterThan(0.8);
      const zero = byId(singleHeadEffects(weights, prompt));
      const meanAblated = byId(singleHeadEffects(weights, prompt, "mean", means));
      for (const id of ["L1H2", "L1H4", "L2H3"]) expect(zero[id], `zero ${id}`).toBeLessThan(0.4);
      for (const id of ["L1H1", "L2H1", "L2H4"]) expect(zero[id], `zero ${id}`).toBeGreaterThan(0.65);
      for (const id of ["L1H2", "L1H3", "L1H4"]) expect(meanAblated[id], `mean ${id}`).toBeLessThan(0.55);
      for (const id of ["L1H1", "L2H1", "L2H4"]) expect(meanAblated[id], `mean ${id}`).toBeGreaterThan(0.7);
      // Zero ablation overstates the cost of L2 H3 on every string.
      expect(meanAblated.L2H3 - zero.L2H3, "L2 H3").toBeGreaterThan(0.25);
      // The layer-2 stripe needs the previous-letter heads on every string.
      const stripe = inductionStripe(runWithAblation(weights, prompt, ["L1H3", "L1H4"]), prompt);
      for (const entry of stripe) expect(entry.induction, entry.id).toBeLessThan(0.05);
    }
  }, 120000);
});

describe("activation patching", () => {
  const eight = copyPrompt(8);

  it("recovers the letter at its own position after block 1 and nowhere else (period 8)", () => {
    const sweep = copyPatchSweep(weights, eight, "source");
    const last = eight.ids.length - 1;
    expect(sweep.differing).toEqual([16]);
    expect(sweep.cleanScore.toFixed(2)).toBe("10.69");
    expect(sweep.corruptScore.toFixed(2)).toBe("-1.01");
    expect((sweep.cleanScore - sweep.corruptScore).toFixed(2)).toBe("11.70");
    expect(pct(sweep.recovery[0][eight.source])).toBe("98.8");
    expect(pct(sweep.recovery[0][last])).toBe("-0.5");
    for (let position = 0; position <= last; position += 1) {
      if (position !== eight.source) expect(Math.abs(sweep.recovery[0][position]), `position ${position}`).toBeLessThan(0.006);
      if (position < last) expect(Math.abs(sweep.recovery[1][position]), `block 2, position ${position}`).toBeLessThan(1e-4);
    }
    expect(sweep.recovery[1][last]).toBeCloseTo(1, 5);
  }, 60000);

  it("makes the model copy the planted letter", () => {
    const sweep = copyPatchSweep(weights, eight, "source");
    expect(sweep.cleanTop?.token).toBe("Q");
    expect(sweep.corruptTop?.token).toBe("Y");
    expect(Math.round(sweep.corruptTop!.probability * 100)).toBe(48);
    const thirteen = patchEndpoints(weights, copyPrompt(13), "source");
    const top = topAt(thirteen.corrupt, 23, 1)[0];
    expect(top.token).toBe("Y");
    expect((top.probability * 100).toFixed(1)).toBe("99.5");
  });

  it("has a large gap at every period, which the old model did not", () => {
    const gaps = PERIODS.map((period) => {
      const endpoints = patchEndpoints(weights, copyPrompt(period), "source");
      return endpoints.cleanScore - endpoints.corruptScore;
    });
    expect(Math.min(...gaps).toFixed(2)).toBe("6.05");
    expect(Math.max(...gaps).toFixed(2)).toBe("22.35");
    for (const gap of gaps) expect(gap).toBeGreaterThan(STABLE_MARGIN * 5);
    expect(gaps[PERIODS.indexOf(8)].toFixed(2)).toBe("11.70");
    expect(gaps[PERIODS.indexOf(13)].toFixed(2)).toBe("18.43");
  });

  it("holds on random period-13 strings the lab never shows", () => {
    const gaps = referenceStrings(13, 24, 16, 100000).map((text) => {
      const endpoints = patchEndpoints(weights, copyPromptFromText(text, 13), "source");
      return endpoints.cleanScore - endpoints.corruptScore;
    });
    expect(mean(gaps).toFixed(1)).toBe("20.7");
    expect(Math.min(...gaps).toFixed(1)).toBe("15.9");
  }, 60000);

  it("gives the control corruption a gap too small to read recoveries from", () => {
    const gaps = PERIODS.map((period) => {
      const endpoints = patchEndpoints(weights, copyPrompt(period), "control");
      return endpoints.cleanScore - endpoints.corruptScore;
    });
    for (const gap of gaps) expect(Math.abs(gap)).toBeLessThan(0.35);
    expect(Math.abs(gaps[PERIODS.indexOf(8)]).toFixed(2)).toBe("0.16");
    const sweep = copyPatchSweep(weights, eight, "control");
    expect(sweep.differing).toEqual([controlPosition(eight)]);
    expect(sweep.corruptTop?.token).toBe("Q");
    // The grid still prints percentages, which is the trap.
    const biggest = Math.max(...sweep.recovery[0].map(Math.abs));
    expect(Math.round(biggest * 100)).toBe(43);
  }, 60000);
});

describe("cached sweeps", () => {
  const records = cache.activation_patching as CachedPatchRecord[];
  const sweeps = records.map(cachedPatchSweep);

  it("keeps the speaker pair, whose corrupted prompt differs in its last characters", () => {
    const speaker = sweeps.find((entry) => entry.id === "speaker")!;
    expect(speaker.differing).toEqual([27, 28, 29, 30, 31, 32]);
    expect((speaker.cleanScore - speaker.corruptScore).toFixed(2)).toBe("8.43");
    expect(pct(speaker.recovery[0][32])).toBe("96.0");
    expect(speaker.recovery[1][32]).toBeCloseTo(1, 2);
    for (let position = 0; position < 32; position += 1) expect(Math.abs(speaker.recovery[0][position])).toBeLessThan(0.06);
  });

  it("rebuilds the letters pair on a period-13 string with a well-conditioned gap", () => {
    const letters = sweeps.find((entry) => entry.id === "copy")!;
    expect(letters.differing).toEqual([11]);
    expect(letters.clean).toBe(copyPrompt(13).text);
    expect((letters.cleanScore - letters.corruptScore).toFixed(2)).toBe("11.99");
    expect(Math.abs(letters.cleanScore - letters.corruptScore)).toBeGreaterThan(STABLE_MARGIN);
    expect(pct(letters.recovery[0][11])).toBe("99.5");
    expect(letters.recovery[1][23]).toBeCloseTo(1, 2);
  });

  it("is the same prompt pair as the live sweep at period 13, under a different metric", () => {
    const letters = sweeps.find((entry) => entry.id === "copy")!;
    const live = copyPatchSweep(weights, copyPrompt(13), "source");
    expect(live.clean).toBe(letters.clean);
    expect(live.corrupt).toBe(letters.corrupt);
    expect((live.cleanScore - live.corruptScore).toFixed(2)).toBe("18.43");
    let worst = 0;
    for (let layer = 0; layer < 2; layer += 1) {
      for (let position = 0; position < live.clean.length; position += 1) {
        worst = Math.max(worst, Math.abs(live.recovery[layer][position] - letters.recovery[layer][position]));
      }
    }
    expect(worst).toBeLessThan(0.01);
    expect(worst.toFixed(3)).toBe("0.006");
  }, 60000);

  it("uses a canned attention example that is the live period-13 prompt", () => {
    const record = cache.attention.find((entry) => entry.id === "induction-random")!;
    expect(record.text).toBe(copyPrompt(13).text);
  });
});
