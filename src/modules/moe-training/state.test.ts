import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { ModuleState } from "@app/module-sdk";
import definition from "./module";
import {
  AUX_LADDER,
  BIAS_LADDER,
  CAPACITY_LADDER,
  defaultState,
  hydrateMoeState,
  initialState,
  readState,
} from "./state";

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
const text = (html: string) => html.replace(/<[^>]*>/g, "\u0001");
const broken = /\bNaN\b|\bundefined\b|\bInfinity\b/;
const KEYS = ["auxWeight", "balancing", "biasStep", "capacity", "epochs", "experts", "seed", "specialView", "topK"];

function expectBounded(state: ModuleState) {
  expect(Object.keys(state).sort()).toEqual(KEYS);
  expect(["none", "aux", "bias"]).toContain(state.balancing);
  expect(AUX_LADDER as readonly number[]).toContain(state.auxWeight);
  expect(BIAS_LADDER as readonly number[]).toContain(state.biasStep);
  expect(CAPACITY_LADDER as readonly number[]).toContain(state.capacity);
  for (const [key, low, high] of [
    ["experts", 1, 8],
    ["epochs", 1, 40],
    ["seed", 1, 6],
  ] as const) {
    expect(Number.isInteger(state[key]), key).toBe(true);
    expect(state[key] as number, key).toBeGreaterThanOrEqual(low);
    expect(state[key] as number, key).toBeLessThanOrEqual(high);
  }
  expect(Number.isInteger(state.topK)).toBe(true);
  expect(state.topK as number).toBeGreaterThanOrEqual(1);
  expect(state.topK as number).toBeLessThanOrEqual(Math.min(2, state.experts as number));
  expect(["chars", "corpus"]).toContain(state.specialView);
}

describe("moe-training state", () => {
  it("starts with no balancing, 4 experts, top-1, no cap, 30 epochs and seed 1", () => {
    expect(initialState).toEqual({
      balancing: "none",
      auxWeight: 0.01,
      biasStep: 0.001,
      experts: 4,
      topK: 1,
      capacity: 0,
      epochs: 30,
      seed: 1,
      specialView: "chars",
    });
    expect(definition.initialState).toEqual(initialState);
    expect(definition.stateVersion).toBe(1);
  });

  it("round-trips the defaults and falls back to them on empty or broken input", () => {
    expect(roundTrip(initialState)).toEqual(initialState);
    for (const payload of ["{}", "", "not json", "[1,2,3]", "null", "42", '"aux"']) {
      expect(hydrate(payload), payload).toEqual(initialState);
    }
    expect(hydrateMoeState("{}")).toEqual(initialState);
  });

  it("round-trips every state the controls can produce unchanged", () => {
    const states: ModuleState[] = [];
    for (const balancing of ["none", "aux", "bias"]) {
      for (const experts of [1, 2, 8]) {
        for (const topK of [1, 2]) {
          if (topK > experts) continue;
          states.push({
            balancing,
            auxWeight: AUX_LADDER[experts % AUX_LADDER.length],
            biasStep: BIAS_LADDER[topK],
            experts,
            topK,
            capacity: CAPACITY_LADDER[experts % CAPACITY_LADDER.length],
            epochs: experts === 8 ? 40 : 1,
            seed: experts === 8 ? 6 : 1,
            specialView: topK === 2 ? "corpus" : "chars",
          });
        }
      }
    }
    for (const state of states) expect(roundTrip(state)).toEqual(state);
    for (const auxWeight of AUX_LADDER) expect(readState({ auxWeight }).auxWeight).toBe(auxWeight);
    for (const biasStep of BIAS_LADDER) expect(readState({ biasStep }).biasStep).toBe(biasStep);
    for (const capacity of CAPACITY_LADDER) expect(readState({ capacity }).capacity).toBe(capacity);
  });

  it("drops unknown keys, rejects unknown strings and snaps numbers to the controls' notches", () => {
    const wild = readState({
      balancing: "both",
      auxWeight: 0.2,
      biasStep: 0.05,
      experts: 3.6,
      topK: 9,
      capacity: 1.4,
      epochs: 12.4,
      seed: 0,
      specialView: "topics",
      injected: "<script>",
    });
    expect(wild).toEqual({
      balancing: "none",
      auxWeight: 0.3,
      biasStep: 0.03,
      experts: 4,
      topK: 2,
      capacity: 1.5,
      epochs: 12,
      seed: 1,
      specialView: "chars",
    });
    expect(readState({ experts: 1, topK: 2 }).topK).toBe(1);
    expect(readState({ experts: "8", epochs: null, auxWeight: "big", capacity: [2] })).toEqual(defaultState);
  });

  it("stays bounded for 1e9, -1e9 and Infinity in every numeric key", () => {
    const numeric = Object.keys(initialState).filter((key) => typeof initialState[key] === "number");
    for (const value of ["1e9", "-1e9", "1e999", "-1e999"]) {
      const payload = `{${numeric.map((key) => `"${key}":${value}`).join(",")}}`;
      expectBounded(hydrate(payload));
    }
    expect(hydrate('{"experts":1e9,"topK":1e9,"epochs":1e9,"capacity":1e9,"auxWeight":1e9,"biasStep":1e9,"seed":1e9}')).toEqual({
      ...initialState,
      experts: 8,
      topK: 2,
      epochs: 40,
      capacity: 2,
      auxWeight: 1,
      biasStep: 0.1,
      seed: 6,
    });
  });

  it("renders every extreme the controls allow without printing a broken or raw value", { timeout: 60_000 }, () => {
    const extremes: ModuleState[] = [];
    for (const balancing of ["none", "aux", "bias"]) {
      for (const [experts, topK] of [
        [1, 1],
        [8, 1],
        [8, 2],
      ]) {
        for (const capacity of [0, 1, 2]) {
          extremes.push({
            ...initialState,
            balancing,
            experts,
            topK,
            capacity,
            auxWeight: 1,
            biasStep: 0.1,
            epochs: capacity === 2 ? 40 : 1,
            seed: 6,
            specialView: capacity === 1 ? "corpus" : "chars",
          });
        }
      }
    }
    for (const state of extremes) {
      const html = text(renderLab(state));
      expect(html, JSON.stringify(state)).not.toMatch(broken);
      expect(html, JSON.stringify(state)).not.toMatch(/\d\.\d{6,}/);
    }
  });

  it("renders the zeroed and emptied payloads the registry feeds it", () => {
    const zeroed = Object.fromEntries(
      Object.entries(initialState).map(([key, value]) => [key, typeof value === "number" ? 0 : ""]),
    ) as ModuleState;
    expect(text(renderLab(zeroed))).not.toMatch(broken);
  });

  it("names a collapsed expert at the defaults and stops calling it collapsed once balanced", () => {
    expect(text(renderLab(initialState))).toContain("Collapsed: E3 takes 100% of the choices");
    const balanced = text(renderLab({ ...initialState, balancing: "aux", auxWeight: 0.1 }));
    expect(balanced).not.toContain("Collapsed");
    expect(balanced).toContain("Every expert gets work");
  });
});
