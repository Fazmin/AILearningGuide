import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { ModuleState } from "@app/module-sdk";
import definition from "./module";
import { WEIGHT_MAX, WEIGHT_MIN } from "./mixture";
import { BUDGET_MAX, BUDGET_MIN, UNIQUE_MAX, UNIQUE_MIN } from "./repetition";
import { defaultState, hydrateMidTrainingState, initialState, readState, weightsOf } from "./state";

const hydrate = (value: string) => definition.hydrateState(value);
const roundTrip = (state: ModuleState) => hydrate(definition.serializeState(state));

const renderLab = (state: ModuleState, currentStep = 0) =>
  renderToStaticMarkup(
    createElement(definition.Explore, {
      state,
      setState: () => undefined,
      currentStep,
      mode: "standard" as const,
      narrate: () => undefined,
    }),
  );

const KEYS = [
  "arithmeticWeight",
  "harborWeight",
  "latePhase",
  "proverbsWeight",
  "recipesWeight",
  "repeatBudget",
  "schedule",
  "uniqueSentences",
];

describe("mid-training state", () => {
  it("ships General-heavy, no switch, the decay schedule, and all 120 sentences at 8,000 pairs", () => {
    expect(initialState).toEqual({
      harborWeight: 3,
      recipesWeight: 3,
      proverbsWeight: 3,
      arithmeticWeight: 1,
      latePhase: "same",
      schedule: "wsd",
      uniqueSentences: 120,
      repeatBudget: 8000,
    });
    expect(definition.initialState).toEqual(initialState);
    expect(definition.stateVersion).toBe(1);
    expect(weightsOf(defaultState)).toEqual([3, 3, 3, 1]);
  });

  it("round-trips every state the controls can produce unchanged", () => {
    const states: ModuleState[] = [
      initialState,
      { ...initialState, harborWeight: 0, recipesWeight: 8, proverbsWeight: 0, arithmeticWeight: 5 },
      { ...initialState, latePhase: "mix", schedule: "constant", uniqueSentences: 2, repeatBudget: 2000 },
      { ...initialState, latePhase: "only", uniqueSentences: 77, repeatBudget: 16000 },
      { harborWeight: 1, recipesWeight: 1, proverbsWeight: 1, arithmeticWeight: 1, latePhase: "same", schedule: "wsd", uniqueSentences: 120, repeatBudget: 8000 },
    ];
    for (const state of states) expect(roundTrip(state)).toEqual(state);
  });

  it("returns the defaults for an empty object and for garbage", () => {
    expect(hydrate("{}")).toEqual(initialState);
    expect(hydrateMidTrainingState("{}")).toEqual(initialState);
    for (const payload of ["", "{", "not json", "[]", "[1,2]", "null", "42", '"text"', "true", "⛔"]) {
      expect(hydrate(payload), payload).toEqual(initialState);
    }
    expect(readState(undefined)).toEqual(defaultState);
    expect(readState([3, 3, 3, 1])).toEqual(defaultState);
  });

  it("falls back to Uniform when all four weights are zero", () => {
    expect(weightsOf(readState({ harborWeight: 0, recipesWeight: 0, proverbsWeight: 0, arithmeticWeight: 0 }))).toEqual([1, 1, 1, 1]);
    expect(weightsOf(readState({ harborWeight: -1e9, recipesWeight: -5, proverbsWeight: 0, arithmeticWeight: -0.4 }))).toEqual([1, 1, 1, 1]);
    // One weight above zero is a real mixture and is kept.
    expect(weightsOf(readState({ harborWeight: 0, recipesWeight: 0, proverbsWeight: 0, arithmeticWeight: 2 }))).toEqual([0, 0, 0, 2]);
  });

  it("clamps and rounds every number to its control", () => {
    for (const value of [1e9, -1e9]) {
      const state = readState({
        harborWeight: value,
        recipesWeight: value,
        proverbsWeight: value,
        arithmeticWeight: value,
        uniqueSentences: value,
        repeatBudget: value,
      });
      for (const weight of weightsOf(state)) {
        expect(weight).toBeGreaterThanOrEqual(WEIGHT_MIN);
        expect(weight).toBeLessThanOrEqual(WEIGHT_MAX);
      }
      expect(state.uniqueSentences).toBe(value > 0 ? UNIQUE_MAX : UNIQUE_MIN);
      expect(state.repeatBudget).toBe(value > 0 ? BUDGET_MAX : BUDGET_MIN);
    }
    expect(readState({ arithmeticWeight: 2.6 }).arithmeticWeight).toBe(3);
    expect(readState({ uniqueSentences: 7.4 }).uniqueSentences).toBe(7);
    expect(readState({ repeatBudget: 7100 }).repeatBudget).toBe(8000);
    expect(readState({ repeatBudget: 6900 }).repeatBudget).toBe(6000);
  });

  it("replaces non-finite numbers and wrong types with the defaults", () => {
    expect(hydrate('{"harborWeight":1e999,"uniqueSentences":-1e999,"repeatBudget":1e999}')).toEqual(initialState);
    expect(readState({ arithmeticWeight: Number.NaN, uniqueSentences: "60", repeatBudget: [4000] })).toEqual(defaultState);
    expect(readState({ latePhase: "mid", schedule: "cosine" })).toEqual(defaultState);
    expect(readState({ latePhase: 3, schedule: null })).toEqual(defaultState);
    expect(readState({ latePhase: "only", schedule: "constant" })).toMatchObject({ latePhase: "only", schedule: "constant" });
  });

  it("drops unknown keys and never pollutes Object.prototype", () => {
    const hydrated = hydrate(`{"extra":"${"x".repeat(5000)}","__proto__":{"polluted":true},"latePhase":"mix"}`);
    expect(Object.keys(hydrated).sort()).toEqual(KEYS);
    expect(hydrated.latePhase).toBe("mix");
    expect(({} as Record<string, unknown>).polluted).toBeUndefined();
  });

  it("uses the slider bounds the interface offers", () => {
    const html = renderLab(initialState);
    const slider = (label: string) => {
      const match = html.match(new RegExp(`aria-label="${label}" min="([^"]*)" max="([^"]*)" step="([^"]*)"`));
      expect(match, `${label} slider`).not.toBeNull();
      return [Number(match![1]), Number(match![2]), Number(match![3])];
    };
    for (const label of ["Harbor weight", "Recipes weight", "Proverbs weight", "Arithmetic weight"]) {
      expect(slider(label)).toEqual([WEIGHT_MIN, WEIGHT_MAX, 1]);
    }
    expect(slider("Unique sentences")).toEqual([UNIQUE_MIN, UNIQUE_MAX, 1]);
    expect(slider("Budget")).toEqual([BUDGET_MIN, BUDGET_MAX, 2000]);
  });

  it("renders a hostile live state through readState instead of hanging or printing broken values", () => {
    for (const value of [1e9, -1e9, Infinity, Number.NaN]) {
      const state = Object.fromEntries(
        Object.entries(initialState).map(([key, original]) => [key, typeof original === "number" ? value : original]),
      ) as ModuleState;
      const started = performance.now();
      const html = renderLab(state);
      expect(performance.now() - started).toBeLessThan(4000);
      expect(html).not.toMatch(/\bNaN\b|\bundefined\b|\bInfinity\b/);
    }
    const garbage = renderLab({ harborWeight: "x", latePhase: 7, schedule: "linear", uniqueSentences: [] } as unknown as ModuleState);
    expect(garbage).toContain("<output>3 · 30% of each batch</output>");
  });
});
