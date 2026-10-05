import { describe, expect, it } from "vitest";
import { encodeTinyText } from "@app/module-sdk";
import {
  countTransitions,
  HELD_OUT_IDS,
  HELD_OUT_TEXT,
  heldOutCoverage,
  runFamily,
  SEED_COUNTS,
  SEED_IDS,
  SEED_TEXT,
  supportSize,
  TEACHER_EXTRA_TEXT,
  teacherModel,
  teacherUnseenMass,
  type Generator,
} from "./collapse";

/**
 * Pins every figure the Generator control, its card text, and the lesson quote. Means are over the lab's
 * eight sampling seeds, at temperature 1 and 500 characters per generation unless a test says otherwise.
 */
const family = (regime: "replace" | "accumulate", generator: Generator, extra: { sampleSize?: 250 | 500 | 2000; filter?: "none" | "words" } = {}) =>
  runFamily({ regime, temperature: 1, filter: extra.filter ?? "none", sampleSize: extra.sampleSize ?? 500, generator });
const heldOut = (run: ReturnType<typeof family>, generation: number) => run.mean[generation].heldOut;

describe("the stronger teacher", () => {
  it("is trained on about four times the seed's text and scored on text nobody trained on", () => {
    const extra = encodeTinyText(TEACHER_EXTRA_TEXT);
    expect(SEED_IDS).toHaveLength(499);
    expect(extra).toHaveLength(1515);
    expect(HELD_OUT_IDS).toHaveLength(586);
    expect(SEED_IDS.length + extra.length).toBe(2014);
    expect((extra.length + SEED_IDS.length) / SEED_IDS.length).toBeCloseTo(4.04, 2);
    // No held-out sentence appears in either training text.
    const trained = `${SEED_TEXT} ${TEACHER_EXTRA_TEXT}`;
    for (const line of HELD_OUT_TEXT.split("\n")) expect(trained.includes(line.trim()), line).toBe(false);
    // The seed has 157 distinct pairs and the teacher's data 258.
    expect(supportSize(countTransitions(SEED_IDS))).toBe(157);
    expect(supportSize(countTransitions(encodeTinyText(trained)))).toBe(258);
  });

  it("rows sum to one, never emit the reserved symbols, and keep about one percent on unseen pairs", () => {
    const model = teacherModel();
    for (let row = 0; row < 30; row += 1) {
      let total = 0;
      for (let next = 0; next < 30; next += 1) total += model[row * 30 + next];
      expect(total).toBeCloseTo(1, 9);
      expect(model[row * 30 + 28]).toBe(0);
      expect(model[row * 30 + 29]).toBe(0);
    }
    expect(teacherUnseenMass()).toBeCloseTo(0.01, 3);
    // A second call returns the cached table, so the SGD fit runs once per session.
    expect(teacherModel()).toBe(model);
  });

  it("covers the unseen text better than the seed does, which is the room a student can gain", () => {
    expect(heldOutCoverage(SEED_COUNTS)).toBeCloseTo(0.9265, 4);
    // The teacher's whole training text covers 99.5 percent of the unseen pairs, the ceiling its data sets.
    expect(heldOutCoverage(countTransitions(encodeTinyText(`${SEED_TEXT} ${TEACHER_EXTRA_TEXT}`)))).toBeCloseTo(0.995, 3);
    expect(heldOutCoverage(new Float64Array(900))).toBe(0);
  });

  it("lifts Unseen text and Transitions under Real + synthetic, where the model itself changes nothing", () => {
    const teacher = family("accumulate", "teacher");
    const self = family("accumulate", "self");
    expect(heldOut(teacher, 0)).toBeCloseTo(0.926, 3);
    expect(heldOut(teacher, 1)).toBeCloseTo(0.962, 3);
    expect(heldOut(teacher, 3)).toBeCloseTo(0.981, 3);
    expect(heldOut(teacher, 5)).toBeCloseTo(0.988, 3);
    expect(heldOut(teacher, 10)).toBeCloseTo(0.994, 3);
    for (const generation of [1, 5, 10]) expect(heldOut(self, generation)).toBeCloseTo(0.926, 3);
    expect(teacher.mean.map((point) => Math.round(point.support)).filter((_, index) => [1, 3, 5, 10].includes(index))).toEqual([197, 231, 251, 275]);
    // The seed stays in every round, so no seed pair is ever lost.
    for (const point of teacher.mean) expect(point.coverage).toBe(1);
  });

  it("stops the collapse under Synthetic only, but a single 500-character sample still leaves gaps", () => {
    const self = family("replace", "self");
    const teacher = family("replace", "teacher");
    expect(heldOut(self, 10)).toBeCloseTo(0.556, 3);
    expect(Math.round(self.mean[10].support)).toBe(51);
    // The teacher's level does not decay with the generation count...
    const levels = [1, 3, 5, 10].map((generation) => heldOut(teacher, generation));
    expect(Math.max(...levels) - Math.min(...levels)).toBeLessThan(0.02);
    expect(levels[0]).toBeCloseTo(0.878, 3);
    expect(levels[3]).toBeCloseTo(0.865, 3);
    // ...but at 500 characters it sits below the seed-fitted model's 92.6 percent.
    for (const level of levels) expect(level).toBeLessThan(0.926);
  });

  it("beats the seed with no human data in the loop once the sample is large", () => {
    const teacher = family("replace", "teacher", { sampleSize: 2000 });
    const self = family("replace", "self", { sampleSize: 2000 });
    expect(heldOut(teacher, 1)).toBeCloseTo(0.975, 3);
    expect(heldOut(teacher, 5)).toBeCloseTo(0.979, 3);
    expect(heldOut(teacher, 10)).toBeCloseTo(0.975, 3);
    expect(heldOut(self, 10)).toBeCloseTo(0.807, 3);
    for (const generation of [1, 5, 10]) expect(heldOut(teacher, generation)).toBeGreaterThan(heldOutCoverage(SEED_COUNTS));
  });

  it("is thrown away by a verifier built from the seed's words", () => {
    const replace = family("replace", "teacher", { filter: "words" });
    const accumulate = family("accumulate", "teacher", { filter: "words" });
    expect(heldOut(replace, 1)).toBeCloseTo(0.422, 3);
    expect(heldOut(accumulate, 1)).toBeCloseTo(0.928, 3);
    expect(heldOut(accumulate, 10)).toBeCloseTo(0.93, 3);
    // Unfiltered, the same teacher and regime reach 99.4 percent.
    expect(heldOut(accumulate, 10)).toBeLessThan(heldOut(family("accumulate", "teacher"), 10) - 0.05);
  }, 30_000);

  it("keeps the model-itself runs exactly as they were before the generator existed", () => {
    // These are the lesson's long-standing figures: Transitions at generation 10 under Synthetic only and at every generation under Real + synthetic.
    expect(Math.round(family("replace", "self").mean[10].support)).toBe(51);
    expect(family("accumulate", "self").mean[10].support).toBe(157);
    expect(runFamily({ regime: "replace", temperature: 1, filter: "none", sampleSize: 500 }).mean[10].support).toBe(
      family("replace", "self").mean[10].support,
    );
  });
});
