import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { ModuleState } from "@app/module-sdk";
import definition from "./module";
import { EPISODE_RANGE, readState, sanitizeState } from "./state";
import { COST_RANGE, GROUP_RANGE, KL_RANGE, SEED_RANGE, UPDATES } from "./train";

const hydrate = (value: string) => definition.hydrateState(value);
const roundTrip = (state: ModuleState) => hydrate(definition.serializeState(state));

const KEYS = ["checker", "episode", "group", "kl", "reward", "scale", "seed", "stepCost", "tickets", "view"];

function expectBounded(state: ModuleState) {
  expect(Object.keys(state).sort()).toEqual(KEYS);
  const within = (key: string, low: number, high: number, grid = 1) => {
    const value = state[key] as number;
    expect(Number.isFinite(value), key).toBe(true);
    expect(value, key).toBeGreaterThanOrEqual(low);
    expect(value, key).toBeLessThanOrEqual(high);
    expect(Math.abs(value * grid - Math.round(value * grid)), `${key} is on its grid`).toBeLessThan(1e-9);
  };
  within("group", GROUP_RANGE.min, GROUP_RANGE.max);
  within("kl", KL_RANGE.min, KL_RANGE.max, 20);
  within("stepCost", COST_RANGE.min, COST_RANGE.max, 100);
  within("seed", SEED_RANGE.min, SEED_RANGE.max);
  within("view", 0, UPDATES);
  within("episode", EPISODE_RANGE.min, EPISODE_RANGE.max);
  expect(["outcome", "partial", "judge"]).toContain(state.reward);
  expect(["strict", "sandbox"]).toContain(state.checker);
  expect(["one", "three", "six"]).toContain(state.tickets);
  expect(["std", "none"]).toContain(state.scale);
}

describe("rl-environments state", () => {
  it("keeps the shipped defaults", () => {
    expect(definition.initialState).toEqual({
      reward: "outcome",
      checker: "strict",
      tickets: "six",
      group: 8,
      kl: 0,
      stepCost: 0,
      scale: "std",
      seed: 1,
      view: 200,
      episode: 1,
    });
    expect(hydrate("{}")).toEqual(definition.initialState);
    expect(definition.stateVersion).toBe(1);
  });

  it("round-trips every state the controls can produce unchanged", () => {
    const states: ModuleState[] = [
      definition.initialState,
      { ...definition.initialState, reward: "judge", checker: "sandbox", tickets: "one", group: 2, kl: 0.15, stepCost: 0.07 },
      { ...definition.initialState, reward: "partial", tickets: "three", group: 32, kl: 0.5, stepCost: 0.1, scale: "none" },
      { ...definition.initialState, seed: 20, view: 0, episode: 20 },
    ];
    for (const state of states) expect(roundTrip(state)).toEqual(state);
    for (let step = 0; step <= 10; step += 1) {
      expect(roundTrip({ ...definition.initialState, kl: Number((step * 0.05).toFixed(2)) }).kl).toBe(Number((step * 0.05).toFixed(2)));
      expect(roundTrip({ ...definition.initialState, stepCost: step / 100 }).stepCost).toBe(step / 100);
    }
  });

  it("snaps β and the step cost to their slider grids", () => {
    expect(readState({ kl: 0.149 }).kl).toBe(0.15);
    expect(readState({ kl: 0.21 }).kl).toBe(0.2);
    expect(readState({ stepCost: 0.0549 }).stepCost).toBe(0.05);
    expect(readState({ group: 7.6 }).group).toBe(8);
  });

  it("drops unknown keys and rejects unknown choices", () => {
    const state = hydrate(JSON.stringify({ reward: "bribe", checker: "none", tickets: "all", scale: "max", weights: [1, 2], extra: 3 }));
    expect(state).toEqual(definition.initialState);
    expect(sanitizeState({ reward: "partial" }).reward).toBe("partial");
  });

  it("clamps absurd, infinite and mistyped values instead of trusting them", () => {
    const hostile = [
      { group: 1e9, kl: 1e9, stepCost: 1e9, seed: 1e9, view: 1e9, episode: 1e9 },
      { group: -1e9, kl: -3, stepCost: -1, seed: -5, view: -50, episode: 0 },
      { group: "8", kl: null, stepCost: NaN, seed: [1], view: {}, episode: true, reward: 4 },
    ];
    for (const payload of hostile) expectBounded(hydrate(JSON.stringify(payload)));
    expectBounded(hydrate('{"group": 1e999, "kl": -1e999, "view": 1e999}'));
    expect(hydrate('{"group": 1e9}').group).toBe(GROUP_RANGE.max);
    expect(hydrate('{"kl": 1e9}').kl).toBe(KL_RANGE.max);
    expect(hydrate('{"view": -50}').view).toBe(0);
  });

  it("survives malformed JSON and non-object payloads", () => {
    for (const value of ["", "not json", "null", "[]", '"text"', "42", "⛔"]) {
      expect(hydrate(value)).toEqual(definition.initialState);
    }
  });

  it("renders every step from a hostile payload without a broken value", () => {
    const state = hydrate(JSON.stringify({ group: 1e9, kl: 1e9, stepCost: 1e9, seed: 1e9, view: 1e9, episode: 1e9 }));
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
      expect(html).toContain("Train with group-relative updates");
      expect(html).not.toMatch(/\bNaN\b|\bundefined\b|\bInfinity\b/);
    }
  });
});
