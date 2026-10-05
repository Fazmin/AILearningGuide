import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { ModuleState } from "@app/module-sdk";
import definition from "./module";
import { GROUP_RANGE, ITERATIONS, K_RANGE, RATE_RANGE, SEED_RANGE } from "./grpo";

const hydrate = (value: string) => definition.hydrateState(value);
const roundTrip = (state: ModuleState) => hydrate(definition.serializeState(state));

const KEYS = [
  "accuracy",
  "budget",
  "grpoCost",
  "grpoGroup",
  "grpoK",
  "grpoRate",
  "grpoSeed",
  "grpoView",
  "problem",
  "samples",
  "spread",
];

function expectBounded(state: ModuleState) {
  expect(Object.keys(state).sort()).toEqual(KEYS);
  const within = (key: string, low: number, high: number, integer = true) => {
    const value = state[key] as number;
    expect(Number.isFinite(value), key).toBe(true);
    expect(value, key).toBeGreaterThanOrEqual(low);
    expect(value, key).toBeLessThanOrEqual(high);
    if (integer) expect(Number.isInteger(value), key).toBe(true);
  };
  within("grpoK", K_RANGE.min, K_RANGE.max);
  within("grpoGroup", GROUP_RANGE.min, GROUP_RANGE.max);
  within("grpoRate", RATE_RANGE.min, RATE_RANGE.max, false);
  within("grpoSeed", SEED_RANGE.min, SEED_RANGE.max);
  within("grpoView", 0, ITERATIONS);
  within("budget", 0, 6);
  within("samples", 1, 32);
  within("accuracy", 0.05, 0.95, false);
  expect(typeof state.grpoCost).toBe("boolean");
}

describe("reasoning-models state", () => {
  it("round-trips every state the controls can produce unchanged", () => {
    const states: ModuleState[] = [
      definition.initialState,
      { ...definition.initialState, grpoK: 1, grpoGroup: 2, grpoRate: 0.1, grpoCost: true, grpoSeed: 20, grpoView: 0 },
      { ...definition.initialState, grpoK: 4, grpoGroup: 32, grpoRate: 1, grpoCost: false, grpoSeed: 1, grpoView: 100 },
      { ...definition.initialState, problem: "add", budget: 6, samples: 32, accuracy: 0.95, spread: "unique", grpoRate: 0.7 },
    ];
    for (const state of states) expect(roundTrip(state)).toEqual(state);
  });

  it("keeps the shipped defaults", () => {
    expect(definition.initialState).toMatchObject({
      grpoK: 3,
      grpoGroup: 16,
      grpoRate: 0.3,
      grpoCost: false,
      grpoSeed: 1,
      grpoView: ITERATIONS,
    });
    expect(hydrate("{}")).toEqual(definition.initialState);
  });

  it("translates a version 2 payload that predates the training card, keeping what it stored", () => {
    const old = JSON.stringify({ problem: "ferry", budget: 4, samples: 16, accuracy: 0.6, spread: "same" });
    expect(hydrate(old)).toEqual({
      ...definition.initialState,
      problem: "ferry",
      budget: 4,
      samples: 16,
      accuracy: 0.6,
      spread: "same",
    });
    expect(definition.stateVersion).toBe(3);
  });

  it("clamps absurd, NaN-like, and mistyped values instead of trusting them", () => {
    const hostile = [
      { grpoK: 1e9, grpoGroup: 1e9, grpoRate: 1e9, grpoSeed: 1e9, grpoView: 1e9, budget: 1e9, samples: 1e9, accuracy: 1e9 },
      { grpoK: -5, grpoGroup: -1e9, grpoRate: -3, grpoSeed: -1, grpoView: -50, budget: -1, samples: -4, accuracy: -2 },
      { grpoK: "3", grpoGroup: null, grpoRate: NaN, grpoCost: "yes", grpoSeed: [1], grpoView: {}, problem: 4, spread: "none" },
      { grpoK: Infinity, grpoGroup: -Infinity, grpoRate: Infinity },
    ];
    for (const payload of hostile) {
      expectBounded(hydrate(JSON.stringify(payload)));
      expectBounded(hydrate(JSON.stringify(payload).replace(/null/g, "0")));
    }
    expect(hydrate('{"grpoGroup": 1e9}').grpoGroup).toBe(GROUP_RANGE.max);
    expect(hydrate('{"grpoK": -5}').grpoK).toBe(K_RANGE.min);
    expect(hydrate('{"grpoCost": "yes"}').grpoCost).toBe(false);
  });

  it("survives malformed JSON and non-object payloads", () => {
    for (const value of ["", "not json", "null", "[]", '"text"', "42"]) expectBounded(hydrate(value));
  });

  it("renders the training card at every step, even from a hostile payload", () => {
    const state = hydrate(JSON.stringify({ grpoGroup: 1e9, grpoView: 1e9, grpoK: 99, grpoCost: true }));
    for (let currentStep = 0; currentStep < definition.steps.length; currentStep += 1) {
      const html = renderToStaticMarkup(
        createElement(definition.Explore, {
          state,
          setState: () => undefined,
          currentStep,
          mode: "standard" as const,
          narrate: () => undefined,
        }),
      );
      expect(html).toContain("Train against a checker");
      expect(html).not.toMatch(/\bNaN\b|\bundefined\b|\bInfinity\b/);
    }
  });
});
