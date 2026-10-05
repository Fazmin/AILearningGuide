import { beforeAll, describe, expect, it } from "vitest";
import { teachingTransformer } from "@app/module-sdk";
import cache from "./assets/sae-top-activations.json";
import { dominantFocus, isStartArtefact, testLabel, type CachedRecord } from "./sae";
import {
  decoderDirection,
  distributionOf,
  greedyContinuation,
  LABELLED_FEATURES,
  labelText,
  largestShifts,
  MAX_STRENGTH,
  nextLogits,
  NULL_DIRECTIONS,
  nullDirections,
  prefixRows,
  randomDirection,
  SAE_LAYER,
  steeredResidual,
  strengthSweep,
  topTokens,
} from "./steering";
import definition from "./module";
import { DEFAULT_TEXT, fixtureRuns, runText, SCALE, sae, transformer, type TextRun } from "./test-support";
import { ITOS, tokenId, UNKNOWN } from "./vocabulary";

const records = cache.top_activations as unknown as CachedRecord[];
const record = (feature: number) => records.find((entry) => entry.feature === feature)!;
const POSITION = 43;

let probe: TextRun;
beforeAll(() => {
  probe = runText(DEFAULT_TEXT);
  fixtureRuns();
}, 120_000);

const prefix = () => prefixRows(probe.residual, POSITION);
const raw = () => teachingTransformer.rowOf(prefix(), POSITION, 256);
const label = (feature: number) => LABELLED_FEATURES.find((entry) => entry.feature === feature)!;
const sweepAt = (feature: number, strengths: number[]) =>
  strengthSweep(transformer, prefix(), decoderDirection(sae, feature), tokenId(label(feature).character), SCALE, strengths);
const shown = (id: number) => ITOS[id];

describe("the patch intervention", () => {
  it("is the SDK's final-logits patch, and agrees with a full forward pass that overwrites the residual", () => {
    const ids = probe.text;
    for (const position of [4, POSITION, probe.text.length - 1]) {
      const rows = prefixRows(probe.residual, position);
      const vector = steeredResidual(teachingTransformer.rowOf(rows, position, 256), decoderDirection(sae, 160), 7, SCALE);
      const patched = nextLogits(transformer, rows, vector);
      const full = teachingTransformer.runTransformer(
        transformer,
        Array.from(ids, (character) => tokenId(character)),
        { patch: { layer: SAE_LAYER, position, vector } },
      );
      const row = teachingTransformer.rowOf(full.logits, position, transformer.vocabulary);
      let worst = 0;
      for (let id = 0; id < patched.length; id += 1) worst = Math.max(worst, Math.abs(patched[id] - row[id]));
      expect(worst, `position ${position}`).toBeLessThan(1e-5);
    }
  });

  it("changes nothing at strength 0 and nothing before the patched position", () => {
    const before = distributionOf(nextLogits(transformer, prefix()));
    const zero = distributionOf(nextLogits(transformer, prefix(), steeredResidual(raw(), decoderDirection(sae, 160), 0, SCALE)));
    expect(Array.from(zero.logProbabilities)).toEqual(Array.from(before.logProbabilities));
    // The next-character distribution after position 43 depends only on characters 0 to 43: the rest of the text is irrelevant.
    const shorter = runText(DEFAULT_TEXT.slice(0, POSITION + 1));
    const again = distributionOf(nextLogits(transformer, prefixRows(shorter.residual, POSITION)));
    for (let id = 0; id < before.probabilities.length; id += 1) expect(again.probabilities[id]).toBeCloseTo(before.probabilities[id], 6);
  });

  it("adds exactly strength x scale x direction, a vector of length 3.578 per unit of strength", () => {
    const direction = decoderDirection(sae, 683);
    const moved = steeredResidual(raw(), direction, 10, SCALE);
    let length = 0;
    for (let index = 0; index < 256; index += 1) length += (moved[index] - raw()[index]) ** 2;
    expect(Math.sqrt(length)).toBeCloseTo(10 * SCALE, 3);
    let norm = 0;
    for (const value of raw()) norm += value * value;
    // At position 43 the residual is 90.1 long, so strength 10 adds a vector about 40% of its length.
    expect(Math.sqrt(norm)).toBeCloseTo(90.12, 1);
    expect((10 * SCALE) / Math.sqrt(norm)).toBeCloseTo(0.397, 3);
  });

  it("gives a probability distribution that never lists <unk>", () => {
    const before = distributionOf(nextLogits(transformer, prefix()));
    expect(Array.from(before.probabilities).reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 6);
    expect(topTokens(before, 66).some((entry) => entry.id === UNKNOWN)).toBe(false);
    expect(topTokens(before, 3).map((entry) => shown(entry.id))).toEqual([" ", "\n", "h"]);
    expect(topTokens(before, 3)[0].probability).toBeCloseTo(0.5778, 4);
    expect(topTokens(before, 3)[1].probability).toBeCloseTo(0.1992, 4);
  });
});

describe("the random control", () => {
  it("draws twenty fixed unit-length directions, reproducibly", () => {
    expect(NULL_DIRECTIONS).toBe(20);
    expect(nullDirections()).toHaveLength(20);
    for (const direction of nullDirections()) {
      let norm = 0;
      for (const value of direction) norm += value * value;
      expect(Math.sqrt(norm)).toBeCloseTo(1, 5);
    }
    expect(Array.from(randomDirection(1))).toEqual(Array.from(nullDirections()[0]));
    expect(Array.from(randomDirection(1))).not.toEqual(Array.from(randomDirection(2)));
  });

  it("draws directions that are nearly orthogonal to each other (no pair above |cos| 0.22)", () => {
    const directions = nullDirections();
    let worst = 0;
    for (let a = 0; a < directions.length; a += 1) {
      for (let b = a + 1; b < directions.length; b += 1) {
        let dot = 0;
        for (let index = 0; index < 256; index += 1) dot += directions[a][index] * directions[b][index];
        worst = Math.max(worst, Math.abs(dot));
      }
    }
    expect(worst).toBeLessThan(0.22);
  });
});

describe("the features the steering card offers carry labels measured on held-out text", () => {
  it("offers exactly the features that module.ts hydrates, and none that the cache flags", () => {
    expect(LABELLED_FEATURES.map((entry) => entry.feature)).toEqual([160, 683, 305, 121, 122]);
    for (const entry of LABELLED_FEATURES) {
      expect(isStartArtefact(record(entry.feature)), `#${entry.feature}`).toBe(false);
      expect(record(entry.feature).position_zero_dominated).toBe(false);
      expect(dominantFocus(record(entry.feature).examples).character, `#${entry.feature}`).toBe(entry.character);
    }
    const hydrated = definition.hydrateState(JSON.stringify({ sae: 2, steerFeature: 305 }));
    expect(hydrated.steerFeature).toBe(305);
    expect(definition.hydrateState(JSON.stringify({ sae: 2, steerFeature: 124 })).steerFeature).toBe(160);
    for (const entry of LABELLED_FEATURES) {
      expect(definition.hydrateState(JSON.stringify({ sae: 2, steerFeature: entry.feature })).steerFeature).toBe(entry.feature);
    }
    expect(labelText(label(160))).toBe("fires on hyphens (above 2.0)");
    expect(labelText(label(683))).toBe("fires on “y” (above 2.0)");
    expect(labelText(label(121))).toBe("fires on spaces (above 5.0)");
  });

  it("scores each label on twelve held-out windows: perfect for the letters and the hyphen, 96% precision for the space", () => {
    const counts = (feature: number) => {
      const entry = label(feature);
      let truePositive = 0;
      let falsePositive = 0;
      let falseNegative = 0;
      for (const run of fixtureRuns()) {
        const result = testLabel(run.text, run.features.map((row) => row[feature]), entry.threshold, "char", entry.character);
        truePositive += result.truePositive;
        falsePositive += result.falsePositive;
        falseNegative += result.falseNegative;
      }
      return { truePositive, falsePositive, falseNegative };
    };
    expect(counts(160)).toEqual({ truePositive: 12, falsePositive: 0, falseNegative: 0 });
    expect(counts(683)).toEqual({ truePositive: 10, falsePositive: 0, falseNegative: 0 });
    expect(counts(305)).toEqual({ truePositive: 10, falsePositive: 0, falseNegative: 0 });
    expect(counts(122)).toEqual({ truePositive: 20, falsePositive: 0, falseNegative: 0 });
    expect(counts(121)).toEqual({ truePositive: 106, falsePositive: 4, falseNegative: 11 });
  });

  it("scores the same labels on the default text the lab starts with", () => {
    const onProbe = (feature: number) => {
      const entry = label(feature);
      return testLabel(probe.text, probe.features.map((row) => row[feature]), entry.threshold, "char", entry.character);
    };
    expect(onProbe(683)).toMatchObject({ truePositive: 1, falsePositive: 0, falseNegative: 0 });
    expect(onProbe(160)).toMatchObject({ truePositive: 0, falsePositive: 0, falseNegative: 0, precision: null, recall: null });
    expect(onProbe(121)).toMatchObject({ truePositive: 7, falsePositive: 0, falseNegative: 3 });
  });
});

describe("steering at the comma after aye (position 43) of the default text", () => {
  it("raises a hyphen's probability from 1.3% to 10.3% at strength 5 and 43.0% at strength 10, far outside the random spread", () => {
    expect(probe.text[POSITION]).toBe(",");
    const before = distributionOf(nextLogits(transformer, prefix()));
    const dash = tokenId("-");
    const at = (strength: number) =>
      distributionOf(nextLogits(transformer, prefix(), steeredResidual(raw(), decoderDirection(sae, 160), strength, SCALE)));
    expect(before.probabilities[dash]).toBeCloseTo(0.01267, 5);
    expect(at(5).probabilities[dash]).toBeCloseTo(0.10268, 5);
    expect(at(10).probabilities[dash]).toBeCloseTo(0.43036, 5);
    expect(topTokens(at(10), 3).map((entry) => shown(entry.id))).toEqual(["-", "\n", " "]);
    expect(topTokens(at(5), 3).map((entry) => shown(entry.id))).toEqual([" ", "\n", "-"]);

    const [five] = sweepAt(160, [5]);
    expect(five.feature).toBeCloseTo(2.092, 3);
    expect([five.nullMin, five.nullMedian, five.nullMax].map((value) => Number(value.toFixed(2)))).toEqual([-0.27, 0.13, 0.7]);
    expect(five.score).toBeCloseTo(7.9, 1);
    const [ten] = sweepAt(160, [10]);
    expect(ten.feature).toBeCloseTo(3.525, 3);
    expect(ten.score).toBeCloseTo(6.9, 1);
    expect([ten.nullMin, ten.nullMax].map((value) => Number(value.toFixed(2)))).toEqual([-0.53, 1.36]);

    const shifts = largestShifts(before, at(5), 3);
    expect(shifts.gains.map((entry) => shown(entry.id))).toEqual(["-", "b", "h"]);
    expect(shifts.gains[0].delta).toBeCloseTo(2.092, 3);
    expect(shifts.losses.map((entry) => shown(entry.id))).toEqual(["'", " ", "x"]);
  });

  it("ranks the hyphen feature first of all 1,024 decoder directions for raising a hyphen", () => {
    const dash = tokenId("-");
    const before = distributionOf(nextLogits(transformer, prefix()));
    const change = (feature: number) =>
      distributionOf(nextLogits(transformer, prefix(), steeredResidual(raw(), decoderDirection(sae, feature), 5, SCALE))).logProbabilities[dash] -
      before.logProbabilities[dash];
    const all = Array.from({ length: 1024 }, (_, feature) => ({ feature, change: change(feature) })).sort((a, b) => b.change - a.change);
    expect(all[0].feature).toBe(160);
    expect(all[0].change).toBeCloseTo(2.092, 3);
    expect(all[1].feature).toBe(563);
    expect(all[1].change).toBeCloseTo(1.68, 2);
    // The 'y' feature goes the other way: it is 1,011th of 1,024 for raising a y.
    const why = tokenId("y");
    const changeY = (feature: number) =>
      distributionOf(nextLogits(transformer, prefix(), steeredResidual(raw(), decoderDirection(sae, feature), 5, SCALE))).logProbabilities[why] -
      distributionOf(nextLogits(transformer, prefix())).logProbabilities[why];
    const ranked = Array.from({ length: 1024 }, (_, feature) => ({ feature, change: changeY(feature) })).sort((a, b) => b.change - a.change);
    expect(ranked.findIndex((entry) => entry.feature === 683) + 1).toBe(1011);
    // What does raise a y most: features whose cached examples are the letters b, m and t.
    expect(ranked.slice(0, 3).map((entry) => entry.feature)).toEqual([167, 902, 737]);
    expect(ranked.slice(0, 3).map((entry) => dominantFocus(record(entry.feature).examples).character)).toEqual(["b", "m", "t"]);
  }, 60_000);

  it("lowers the probability of a y when the y feature is added, and raises an o most: the label predicts the wrong output", () => {
    const before = distributionOf(nextLogits(transformer, prefix()));
    const after = (strength: number) =>
      distributionOf(nextLogits(transformer, prefix(), steeredResidual(raw(), decoderDirection(sae, 683), strength, SCALE)));
    const why = tokenId("y");
    expect(before.probabilities[why]).toBeCloseTo(0.00546, 5);
    expect(after(5).probabilities[why]).toBeCloseTo(0.00137, 5);
    expect(after(10).probabilities[why]).toBeCloseTo(0.00041, 5);
    const shifts = largestShifts(before, after(5), 3);
    expect(shifts.gains.map((entry) => shown(entry.id))).toEqual(["o", "i", "M"]);
    expect(shifts.gains[0].delta).toBeCloseTo(1.36, 2);
    expect(shifts.losses.map((entry) => shown(entry.id))).toEqual(["y", "u", "a"]);
    const [five] = sweepAt(683, [5]);
    expect(five.feature).toBeCloseTo(-1.382, 3);
    expect([five.nullMin, five.nullMax].map((value) => Number(value.toFixed(2)))).toEqual([-0.62, 0.85]);
    expect(five.score).toBeCloseTo(-3.8, 1);
  });

  it("shows a T feature that also lowers its own letter, and a space feature whose effect is elsewhere", () => {
    const [capital] = sweepAt(305, [5]);
    expect(capital.feature).toBeCloseTo(-0.779, 3);
    expect(capital.score).toBeCloseTo(-2.2, 1);
    const before = distributionOf(nextLogits(transformer, prefix()));
    const after = (feature: number, strength: number) =>
      distributionOf(nextLogits(transformer, prefix(), steeredResidual(raw(), decoderDirection(sae, feature), strength, SCALE)));
    expect(largestShifts(before, after(305, 5), 2).gains.map((entry) => shown(entry.id))).toEqual(["H", "h"]);

    const [space] = sweepAt(121, [5]);
    expect(space.feature).toBeCloseTo(-0.101, 3);
    expect(space.score).toBeCloseTo(-0.9, 1);
    const shifts = largestShifts(before, after(121, 5), 3);
    expect(shifts.gains.map((entry) => shown(entry.id))).toEqual(["c", "p", "v"]);
    expect(shifts.losses.map((entry) => shown(entry.id))).toEqual(["\n", ".", "?"]);
    expect(shifts.losses[0].delta).toBeCloseTo(-1.87, 2);
  });

  it("shows a feature whose effect cannot be told from a random direction: the m feature stays inside the spread", () => {
    const rows = sweepAt(122, [1, 5, 10, 15]);
    for (const row of rows) expect(Math.abs(row.score), `strength ${row.strength}`).toBeLessThan(2);
    expect(rows[1].feature).toBeCloseTo(0.393, 3);
    expect(rows[1].nullMax).toBeCloseTo(0.366, 3);
    expect(rows[1].score).toBeCloseTo(1.6, 1);
    expect(rows[2].feature).toBeCloseTo(0.787, 3);
  });

  it("sweeps strength 0 to 15: zero at zero, rising and flattening, and a table's worth of rows", () => {
    expect(MAX_STRENGTH).toBe(15);
    const rows = sweepAt(160, Array.from({ length: 16 }, (_, strength) => strength));
    expect(rows).toHaveLength(16);
    expect(rows[0]).toMatchObject({ strength: 0, feature: 0, nullMin: 0, nullMedian: 0, nullMax: 0 });
    const effects = rows.map((row) => row.feature);
    for (let index = 1; index < effects.length; index += 1) expect(effects[index]).toBeGreaterThan(effects[index - 1]);
    // Diminishing: the final normalization compresses a very large push.
    expect(effects[15] - effects[14]).toBeLessThan(effects[1] - effects[0]);
    expect(effects[15]).toBeCloseTo(4.10, 2);
    expect(rows.every((row) => row.strength === 0 || row.feature > row.nullMax)).toBe(true);
    // Random pushes get larger, not smaller, with strength.
    expect(rows[15].nullMax).toBeGreaterThan(rows[5].nullMax);
    expect(rows[15].nullMax).toBeCloseTo(1.90, 2);
    expect(rows[15].nullMin).toBeCloseTo(-0.85, 2);
  });
});

describe("greedy continuation from the comma after aye", () => {
  const cache2 = new Map<string, Float32Array>();
  const provider = (ids: ReadonlyArray<number>) => {
    const key = ids.join(",");
    let rows = cache2.get(key);
    if (!rows) {
      rows = teachingTransformer.runTransformer(transformer, [...ids]).residuals[SAE_LAYER];
      cache2.set(key, rows);
    }
    return rows;
  };
  const prompt = DEFAULT_TEXT.slice(0, POSITION + 1);
  const run = (feature: number | null, strength: number, seed?: number) =>
    greedyContinuation({
      weights: transformer,
      prompt,
      steps: 12,
      residuals: provider,
      steer:
        feature === null
          ? seed === undefined
            ? undefined
            : { direction: nullDirections()[seed - 1], strength, scale: SCALE }
          : { direction: decoderDirection(sae, feature), strength, scale: SCALE },
    });
  /** The ONNX runner and this TypeScript pass agree to about 1e-4; every choice must be wider than that. */
  const SAFE_MARGIN = 0.02;

  it("writes the unsteered continuation, and strength 0 is the same text", async () => {
    expect(prompt).toBe("hath the king a good horse? KING HENRY: aye,");
    const plain = await run(null, 0);
    expect(plain.text).toBe(" my lord,\nTh");
    expect(Math.min(...plain.margins)).toBeGreaterThan(SAFE_MARGIN);
    const zero = await run(160, 0);
    expect(zero.text).toBe(plain.text);
  }, 120_000);

  it("turns the hyphen feature at strength 10 into a row of hyphens, which a random direction of the same length does not", async () => {
    const steered = await run(160, 10);
    expect(steered.text).toBe("------------");
    expect(Math.min(...steered.margins)).toBeGreaterThan(SAFE_MARGIN);
    const control = await run(null, 10, 1);
    expect(control.text).toBe(" where is no");
    expect(Math.min(...control.margins)).toBeGreaterThan(SAFE_MARGIN);
  }, 120_000);

  it("rewrites the text at strength 5 even though a hyphen is not yet the top choice, and a random direction does too", async () => {
    const steered = await run(160, 5);
    expect(steered.text).toBe(" horse!\n\nKIN");
    expect(Math.min(...steered.margins)).toBeGreaterThan(SAFE_MARGIN);
    const control = await run(null, 5, 1);
    expect(control.text).toBe(" what we wil");
    expect(Math.min(...control.margins)).toBeGreaterThan(SAFE_MARGIN);
  }, 120_000);

  it("keeps writing o after the y feature is added at strength 10: the label's own letter is not what comes out", async () => {
    const steered = await run(683, 10);
    expect(steered.text).toBe(" o'er o'er o");
    expect(steered.text.includes("y")).toBe(false);
    expect(Math.min(...steered.margins)).toBeGreaterThan(SAFE_MARGIN);
  }, 120_000);

  it("stops at the end of the window", async () => {
    const full = DEFAULT_TEXT.padEnd(62, "a");
    const result = await greedyContinuation({
      weights: transformer,
      prompt: full,
      steps: 12,
      residuals: (ids) => teachingTransformer.runTransformer(transformer, [...ids]).residuals[SAE_LAYER],
    });
    expect(result.text).toHaveLength(2);
  }, 60_000);
});
