import { describe, expect, it } from "vitest";
import { teachingServing, TINY_RESERVED, TINY_VOCAB, TINY_VOCAB_SIZE } from "@app/module-sdk";
import definition from "./module";
import {
  clampQuality,
  closedForm,
  draftRows,
  exactTokensPerPass,
  GAMMA_RANGE,
  meanAcceptance,
  PROMPT,
  QUALITY_RANGE,
  rowAcceptance,
  SEED_RANGE,
  simulate,
  SIM_PASSES,
  stationary,
  TRANSCRIPT_PASSES,
  trainTargetRows,
} from "./speculative";

/**
 * Pins the numbers the Speculative decoding card, inference-kv-cache standard.mdx and plain.mdx, and card-info.ts quote.
 * The closed form is the SDK's teachingServing.speculativeTokensPerPass, which Deployment & serving also uses.
 */
const V = TINY_VOCAB_SIZE;
const target = trainTargetRows();
const defaults = {
  gamma: definition.initialState.specGamma as number,
  quality: definition.initialState.specQuality as number,
  seed: definition.initialState.specSeed as number,
};
const lab = (gamma = defaults.gamma, quality = defaults.quality, seed = defaults.seed) => {
  const draft = draftRows(target, quality);
  const alpha = meanAcceptance(target, draft);
  return { draft, alpha, closed: closedForm(alpha, gamma), run: simulate({ p: target, q: draft, gamma, seed }) };
};

describe("toy distributions", () => {
  it("makes every row of the target and of every draft a probability distribution with no marker characters", () => {
    for (const rows of [target, draftRows(target, 0.2), draftRows(target, 0.7), draftRows(target, 1)]) {
      for (let row = 0; row < V; row += 1) {
        let total = 0;
        for (let k = 0; k < V; k += 1) {
          expect(rows[row * V + k]).toBeGreaterThanOrEqual(0);
          total += rows[row * V + k];
        }
        expect(total).toBeCloseTo(1, 9);
        for (const marker of TINY_RESERVED) expect(rows[row * V + TINY_VOCAB.indexOf(marker)]).toBe(0);
      }
    }
  });

  it("flattens every row of the draft as quality falls, and leaves the target alone at quality 1", () => {
    const same = draftRows(target, 1);
    for (let index = 0; index < target.length; index += 1) expect(same[index]).toBeCloseTo(target[index], 12);
    const sharp = (rows: Float64Array) => {
      const row = TINY_VOCAB.indexOf("t");
      return Math.max(...Array.from(rows.slice(row * V, row * V + V)));
    };
    expect(sharp(draftRows(target, 0.5))).toBeLessThan(sharp(draftRows(target, 0.9)));
    expect(sharp(draftRows(target, 0.9))).toBeLessThan(sharp(target));
  });

  it("clamps the quality control to steps of 0.05 inside its range", () => {
    expect(clampQuality(1e9)).toBe(QUALITY_RANGE.max);
    expect(clampQuality(-1e9)).toBe(QUALITY_RANGE.min);
    expect(clampQuality(0.72)).toBe(0.7);
    expect(clampQuality(Number.NaN)).toBe(0.7);
    expect(clampQuality(Infinity)).toBe(0.7);
  });

  it("measures the acceptance rate as the sum of min(p, q), averaged over the contexts the target visits", () => {
    expect(meanAcceptance(target, draftRows(target, 1))).toBeCloseTo(1, 12);
    const draft = draftRows(target, 0.7);
    const byRow = rowAcceptance(target, draft);
    byRow.forEach((value) => expect(value).toBeLessThanOrEqual(1 + 1e-12));
    const share = stationary(target);
    expect(share.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 9);
    expect(meanAcceptance(target, draft)).toBeCloseTo(
      Array.from(byRow).reduce((sum, value, row) => sum + value * share[row], 0),
      12,
    );
    // A flatter draft is accepted less often.
    expect(meanAcceptance(target, draftRows(target, 0.4))).toBeLessThan(meanAcceptance(target, draft));
  });
});

describe("the closed form", () => {
  it("is the SDK formula that Deployment & serving also quotes", () => {
    expect(closedForm(0.8, 4)).toBe(teachingServing.speculativeTokensPerPass(0.8, 4));
    // inference-kv-cache standard.mdx "Going deeper": Deployment & serving states 3.36 tokens per pass at alpha 0.8 and gamma 4.
    expect(closedForm(0.8, 4)).toBeCloseTo(3.36, 2);
  });

  it("gives gamma plus one when every proposal is kept, and one when none is", () => {
    for (let gamma = GAMMA_RANGE.min; gamma <= GAMMA_RANGE.max; gamma += 1) expect(closedForm(1, gamma)).toBe(gamma + 1);
    expect(closedForm(0, 4)).toBe(1);
  });

  it("rises with draft length toward 1 / (1 - alpha) and with draft quality (checkpoint question 8)", () => {
    const { alpha } = lab();
    const values = Array.from({ length: GAMMA_RANGE.max }, (_, index) => closedForm(alpha, index + 1));
    for (let index = 1; index < values.length; index += 1) {
      expect(values[index]).toBeGreaterThan(values[index - 1]);
      // Each added proposal is worth less than the one before it.
      if (index > 1) expect(values[index] - values[index - 1]).toBeLessThan(values[index - 1] - values[index - 2]);
    }
    expect(values[values.length - 1]).toBeLessThan(1 / (1 - alpha));
    // card-info.ts "Speculative decoding" notice: from gamma 1 to 4 the gain is larger than from 5 to 8.
    expect(values[3] - values[0]).toBeGreaterThan(values[7] - values[4]);
    expect(lab(4, 0.9).closed).toBeGreaterThan(lab(4, 0.5).closed);
  });
});

describe("the seeded simulation", () => {
  it("is deterministic in its seed", () => {
    expect(lab().run).toEqual(lab().run);
    expect(lab(4, 0.7, 8).run.meanTokens).not.toBe(lab().run.meanTokens);
  });

  it("writes whole passes: each yields its accepted tokens plus one, and a pass that keeps every proposal says so", () => {
    const { run } = lab();
    expect(run.transcript).toHaveLength(TRANSCRIPT_PASSES);
    for (const pass of run.transcript) {
      expect(pass.proposed).toHaveLength(defaults.gamma);
      expect(pass.accepted).toBeGreaterThanOrEqual(0);
      expect(pass.accepted).toBeLessThanOrEqual(defaults.gamma);
      expect(pass.produced).toBe(pass.accepted + 1);
      expect(pass.rejected === null).toBe(pass.accepted === defaults.gamma);
      expect(pass.bonus).toBe(pass.accepted === defaults.gamma);
      expect(pass.fromTarget).toHaveLength(1);
      if (pass.rejected !== null) expect(pass.rejected).toBe(pass.proposed[pass.accepted]);
    }
    expect(run.text).toHaveLength(run.transcript.reduce((sum, pass) => sum + pass.produced, 0));
  });

  it("yields exactly gamma plus one tokens per pass when the draft is the target (checkpoint question 7)", () => {
    for (const gamma of [1, 4, 8]) {
      const { run, alpha } = lab(gamma, 1, 3);
      expect(alpha).toBeCloseTo(1, 12);
      expect(run.meanTokens).toBe(gamma + 1);
      expect(run.standardError).toBe(0);
      expect(run.acceptedShare).toBe(1);
      expect(run.transcript.every((pass) => pass.rejected === null)).toBe(true);
    }
  });

  it("agrees with the closed form within sampling error, at every draft length, quality and several seeds", () => {
    let worst = 0;
    for (const quality of [0.3, 0.5, 0.7, 0.9]) {
      for (const gamma of [1, 3, 5, 8]) {
        for (const seed of [1, 2, 3, 7, 11]) {
          const { run, closed } = lab(gamma, quality, seed);
          const z = Math.abs(run.meanTokens - closed) / run.standardError;
          worst = Math.max(worst, z);
          expect(z, `quality ${quality}, gamma ${gamma}, seed ${seed}`).toBeLessThan(4);
        }
      }
    }
    expect(worst).toBeGreaterThan(0);
  });

  it("matches the exact expectation for this Markov toy, which the closed form only approximates", () => {
    for (const quality of [0.3, 0.7, 0.95]) {
      const draft = draftRows(target, quality);
      for (const gamma of [1, 4, 8]) {
        const exact = exactTokensPerPass(target, draft, gamma);
        const approximate = closedForm(meanAcceptance(target, draft), gamma);
        // card-info.ts "Speculative decoding" limits: the independence assumption costs less than 0.03 tokens here.
        expect(Math.abs(approximate - exact)).toBeLessThan(0.03);
      }
    }
    // card-info.ts "Speculative decoding" limits: across every quality and draft length the controls offer, at most about 0.02.
    let worst = 0;
    for (let step = Math.round(QUALITY_RANGE.min * 20); step <= Math.round(QUALITY_RANGE.max * 20); step += 1) {
      const swept = draftRows(target, step / 20);
      for (let gamma = GAMMA_RANGE.min; gamma <= GAMMA_RANGE.max; gamma += 1) {
        worst = Math.max(worst, Math.abs(closedForm(meanAcceptance(target, swept), gamma) - exactTokensPerPass(target, swept, gamma)));
      }
    }
    expect(worst).toBeLessThan(0.025);
    expect(worst).toBeGreaterThan(0.015);
    // One proposal has no later position to correlate with, so the closed form is exact there.
    const draft = draftRows(target, 0.7);
    expect(closedForm(meanAcceptance(target, draft), 1)).toBeCloseTo(exactTokensPerPass(target, draft, 1), 9);
  });

  it("writes text distributed like the target's, not the draft's (checkpoint question 9)", () => {
    const quality = 0.3;
    const draft = draftRows(target, quality);
    const passes = 30000;
    const { text } = simulate({ p: target, q: draft, gamma: 4, seed: 5, passes, transcriptPasses: passes });
    // Count what follows a space in the speculative text.
    const space = TINY_VOCAB.indexOf(" ");
    const counts = new Float64Array(V);
    let total = 0;
    for (let index = 0; index + 1 < text.length; index += 1) {
      if (text[index] !== " ") continue;
      counts[TINY_VOCAB.indexOf(text[index + 1])] += 1;
      total += 1;
    }
    expect(total).toBeGreaterThan(5000);
    let distanceToTarget = 0;
    let distanceToDraft = 0;
    for (let k = 0; k < V; k += 1) {
      const empirical = counts[k] / total;
      distanceToTarget += Math.abs(empirical - target[space * V + k]);
      distanceToDraft += Math.abs(empirical - draft[space * V + k]);
    }
    expect(distanceToTarget).toBeLessThan(0.05);
    expect(distanceToTarget).toBeLessThan(distanceToDraft / 3);
  });
});

describe("the default lab values quoted in the lesson", () => {
  it("starts at draft length 4, quality 0.70, seed 7 and reads 0.831, 3.57 and 3.60", () => {
    expect(defaults).toEqual({ gamma: 4, quality: 0.7, seed: 7 });
    const { alpha, closed, run } = lab();
    // card-info.ts "Speculative decoding" controls; standard.mdx and plain.mdx "How to play with it".
    expect(alpha).toBeCloseTo(0.831, 3);
    expect(closed).toBeCloseTo(3.57, 2);
    expect(run.meanTokens).toBeCloseTo(3.6, 2);
    expect(run.standardError).toBeCloseTo(0.02, 2);
    expect(SIM_PASSES).toBe(4000);
    // The simulation lands within about one standard error of the closed form at the default seed.
    expect(Math.abs(run.meanTokens - closed) / run.standardError).toBeLessThan(2);
  });

  it("shows the first pass refusing every proposal and the second keeping three, at the default seed", () => {
    // card-info.ts "Speculative decoding" whatYouSee: the transcript names refused and replacement characters.
    const [first, second] = lab().run.transcript;
    expect(first).toMatchObject({ accepted: 0, rejected: "o", fromTarget: "t", produced: 1 });
    expect(second).toMatchObject({ accepted: 3, produced: 4 });
    expect((PROMPT + lab().run.text).startsWith("the t the")).toBe(true);
    expect(PROMPT).toBe("the ");
  });

  it("falls to 2.21 tokens per pass at quality 0.30, and reaches 5 at quality 1.00 (card controls)", () => {
    expect(lab(4, 0.3).closed).toBeCloseTo(2.21, 2);
    expect(lab(4, 1).closed).toBe(5);
    // gamma 1 to 4 against 5 to 8 at the default quality: 1.83 to 3.57, then 3.97 to 4.79.
    const { alpha } = lab();
    expect([1, 4, 5, 8].map((gamma) => Number(closedForm(alpha, gamma).toFixed(2)))).toEqual([1.83, 3.57, 3.97, 4.79]);
    // card-info.ts "Speculative decoding" controls: the ceiling 1 / (1 - alpha) is 5.91.
    expect((1 / (1 - alpha)).toFixed(2)).toBe("5.91");
  });

  it("accepts seeds only inside the control's range", () => {
    expect(SEED_RANGE).toEqual({ min: 1, max: 999 });
  });
});
