import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { checkpointQuestions } from "@app/module-sdk";
import definition from "./module";
import { COMMON_EXAMPLES, EXAMPLES, FRESH_ROWS, countMatrix, leakReport, precisionAtPrevalence, sweep } from "./metrics";

const { hydrateState, initialState, serializeState } = definition;

const render = (patch: Record<string, string | number>, currentStep = 0) =>
  renderToStaticMarkup(
    createElement(definition.Explore, {
      state: { ...initialState, ...patch },
      setState: () => undefined,
      currentStep,
      mode: "standard" as const,
      narrate: () => undefined,
    }),
  );
const visible = (html: string) => html.replace(/<[^>]*>/g, "\u0001");

describe("how-models-are-evaluated state", () => {
  it("round-trips the defaults and keeps the rare population as the default", () => {
    expect(hydrateState(serializeState(initialState))).toEqual(initialState);
    expect(initialState.population).toBe("rare");
    expect(definition.stateVersion).toBe(3);
  });

  it("reads a version 2 payload, which has no population, as the rare one-positive set", () => {
    const version2 = JSON.stringify({
      threshold: 0.42,
      baseline: "learned",
      split: "test",
      tuneSplit: "test",
      goal: "recall",
      curve: "good",
      fold: 2,
      outlier: 9,
    });
    expect(hydrateState(version2)).toEqual({
      threshold: 0.42,
      baseline: "learned",
      split: "test",
      tuneSplit: "test",
      goal: "recall",
      curve: "good",
      fold: 2,
      outlier: 9,
      population: "rare",
    });
  });

  it("validates the population and the goal, and clamps the numbers", () => {
    expect(hydrateState(JSON.stringify({ population: "common", goal: "f1" }))).toMatchObject({ population: "common", goal: "f1" });
    expect(hydrateState(JSON.stringify({ population: "huge", goal: "auc", threshold: 9, fold: -3, outlier: "far" }))).toMatchObject({
      population: "rare",
      goal: "accuracy",
      threshold: 1,
      fold: 0,
      outlier: 12,
    });
    expect(hydrateState(JSON.stringify({ extra: "ignored" }))).toEqual(initialState);
  });
});

describe("the lab renders both populations without a broken value", () => {
  const cases: [string, Record<string, string | number>][] = [
    ["rare, defaults", {}],
    ["rare, learned at 0.42", { baseline: "learned", threshold: 0.42 }],
    ["common, defaults", { population: "common" }],
    ["common, learned at 0.5", { population: "common", baseline: "learned", threshold: 0.5 }],
    ["common, leaked F1", { population: "common", baseline: "learned", goal: "f1", tuneSplit: "test" }],
    ["common, intact recall on train", { population: "common", goal: "recall", tuneSplit: "train", split: "train" }],
    ["rare, F1 tuned on val", { goal: "f1", tuneSplit: "val", baseline: "learned" }],
    ["common, threshold 0 and 1", { population: "common", baseline: "learned", threshold: 0 }],
    ["common, threshold 1", { population: "common", baseline: "learned", threshold: 1 }],
  ];
  it.each(cases)("%s", (_name, patch) => {
    for (let step = 0; step < definition.steps.length; step += 1) {
      const html = render(patch, step);
      expect(html).not.toMatch(/\bNaN\b|\bundefined\b|\bInfinity\b/);
      expect(visible(html)).not.toMatch(/\d\.\d{6,}/);
    }
  });

  it("prints the worked numbers the lesson and checkpoint quote", () => {
    const common = visible(render({ population: "common", baseline: "learned", threshold: 0.5 }));
    expect(common).toContain("16/23");
    expect(common).toContain("69.6%");
    expect(common).toContain("8.5%");
    expect(common).toContain("16/20");
    expect(common).toContain("Population");
    const never = visible(render({ population: "common" }));
    expect(never).toContain("80% accuracy, zero recall");
    const rare = visible(render({}));
    expect(rare).toContain("99% accuracy, zero recall");
  });

  it("shows the leak table for the 1-in-5 population with fresh rows", () => {
    const html = visible(render({ population: "common", goal: "f1" }));
    expect(html).toContain("Intact · t from val");
    expect(html).toContain("Leaked · t from test");
    expect(html).toContain("+19.0 pts");
    expect(html).toContain("85.7%");
    expect(html).toContain("68.8%");
    expect(html).toContain("+16.9 pts");
    const rareAccuracy = visible(render({}));
    expect(rareAccuracy).toContain("+6.7 pts");
  });
});

describe("the checkpoint's quoted numbers", () => {
  const all = COMMON_EXAMPLES;
  it("matches the 0.40 versus 0.60 comparison in the threshold question", () => {
    const low = countMatrix(all, "learned", 0.4);
    const high = countMatrix(all, "learned", 0.6);
    expect([low.fp, high.fp]).toEqual([22, 1]);
    expect([low.tp, high.tp]).toEqual([19, 11]);
    expect([(low.precision! * 100).toFixed(1), (high.precision! * 100).toFixed(1)]).toEqual(["46.3", "91.7"]);
    expect([low.recall, high.recall]).toEqual([0.95, 0.55]);
    const points = sweep(all);
    expect(points[60].precision!).toBeGreaterThan(points[40].precision!);
    expect(points[60].recall!).toBeLessThan(points[40].recall!);
  });

  it("matches the base-rate question: 69.6 percent at 1 in 5 and 8.5 percent at 1 in 100", () => {
    const counted = countMatrix(all, "learned", 0.5);
    expect((precisionAtPrevalence(counted, 0.2)! * 100).toFixed(1)).toBe("69.6");
    expect((precisionAtPrevalence(counted, 0.01)! * 100).toFixed(1)).toBe("8.5");
  });

  it("matches the leakage question: 66.7 and 77.9 for the intact path, 85.7 and 68.8 for the leaked path", () => {
    const report = leakReport(all, "f1", FRESH_ROWS);
    expect([report.intactTest, report.intactFresh, report.leakedTest, report.leakedFresh].map((value) => (value! * 100).toFixed(1))).toEqual([
      "66.7",
      "77.9",
      "85.7",
      "68.8",
    ]);
  });

  it("has four objectives and a question for each", () => {
    expect(definition.objectives).toHaveLength(4);
    const covered = new Set(checkpointQuestions(definition).map((question) => question.objective));
    expect([...covered].sort()).toEqual([0, 1, 2, 3]);
  });

  it("keeps the rare population's quoted numbers", () => {
    expect(countMatrix(EXAMPLES, "always-negative", 0.5)).toMatchObject({ tn: 99, fn: 1 });
  });
});
