import { describe, expect, it } from "vitest";
import type { ModuleState } from "@app/module-sdk";
import definition from "./module";
import scenes from "./scenes";
import { cleanState, readState } from "./state";

const { hydrateState, initialState, serializeState } = definition;

/** What "The math you need" saved at state version 3, before this lab replaced it. */
const VERSION_3_FIXTURE = {
  topic: "probability",
  ax: 1.5,
  ay: -0.4,
  bx: 0.6,
  by: 1.1,
  m00: 2,
  m01: -1.3,
  m10: 0.7,
  m11: 1,
  vx: 1,
  vy: 0.2,
  b0: 0.2,
  b1: 0.1,
  x: 1.2,
  nudge: 0.4,
  pA: 2.4,
  pB: 1.1,
  value: 0.25,
  modelQ: 0.6,
  surface: "nonconvex",
  optW: -1.4,
  stepSize: 0.15,
  probIdea: "bayes",
  prior: 0.1,
  gaussMu: 0,
  gaussVar: 1,
  gaussX: 1,
};

describe("vectors-matrices state", () => {
  it("is at version 4 and round-trips its defaults", () => {
    expect(definition.stateVersion).toBe(4);
    expect(hydrateState(serializeState(initialState))).toEqual(initialState);
    expect(initialState).toMatchObject({ word: "horse", tipX: 0.9, tipY: 1.2, other: "whale", pieces: 2, round: 0 });
  });

  it("opens a version 3 payload of the old lab on this lab's defaults, dropping every old key", () => {
    const hydrated = hydrateState(JSON.stringify(VERSION_3_FIXTURE));
    expect(hydrated).toEqual(initialState);
    for (const key of Object.keys(VERSION_3_FIXTURE)) expect(hydrated).not.toHaveProperty(key);
  });

  it("opens the even older version 2 payload, which had fewer keys", () => {
    expect(hydrateState(JSON.stringify({ topic: "vectors", pA: 3, pB: 0.5, surface: "nonconvex", optW: 1.4 }))).toEqual(
      initialState,
    );
  });

  it("keeps the new keys of a payload that mixes old and new ones", () => {
    const hydrated = hydrateState(JSON.stringify({ ...VERSION_3_FIXTURE, word: "cat", times: -1.7, keep: 0.2 }));
    expect(hydrated).toEqual({ ...initialState, word: "cat", times: -1.7, keep: 0.2 });
  });

  it("falls back to the defaults for anything that is not a JSON object", () => {
    for (const value of ["", "not json at all", "⛔", "null", "[]", "[1,2]", "42", '"horse"', "true", "{"]) {
      expect(hydrateState(value), value).toEqual(initialState);
    }
    expect(hydrateState("{}")).toEqual(initialState);
  });

  it("replaces values of the wrong type and unknown names with the defaults", () => {
    const hydrated = hydrateState(
      JSON.stringify({
        word: "unicorn",
        other: 7,
        push: "sideways",
        first: "squash",
        second: null,
        undo: "maybe",
        tipX: "far",
        times: null,
        keep: [0.5],
        pieces: "3",
        hint: true,
        extra: "ignored",
      }),
    );
    expect(hydrated).toEqual(initialState);
    expect(hydrated).not.toHaveProperty("extra");
  });

  it("ignores keys that try to reach the object's prototype", () => {
    const hydrated = hydrateState('{"__proto__":{"word":"cat"},"constructor":{"prototype":{"word":"cat"}},"word":"dog"}');
    expect(hydrated).toEqual({ ...initialState, word: "dog" });
    expect(({} as Record<string, unknown>).word).toBeUndefined();
  });

  it("clamps every number of a hostile payload into its slider's range", () => {
    for (const hostile of [1e9, -1e9]) {
      const payload = Object.fromEntries(
        Object.entries(initialState).map(([key, value]) => [key, typeof value === "number" ? hostile : value]),
      );
      const hydrated = hydrateState(JSON.stringify(payload));
      const top = hostile > 0;
      expect(hydrated).toMatchObject({
        tipX: top ? 3 : -3,
        tipY: top ? 3 : -3,
        times: top ? 2 : -2,
        turn: top ? 360 : 0,
        stretch: top ? 1.5 : 0.5,
        ma: top ? 2 : -2,
        md: top ? 2 : -2,
        ca: top ? 2 : -2,
        keep: top ? 1 : 0,
        stage: top ? 2 : 0,
        pieces: top ? 4 : 1,
        round: top ? 2 : 0,
        solved: top ? 7 : 0,
        hint: top ? 1 : 0,
        predictDot: top ? 2 : -1,
        predictOrder: top ? 2 : -1,
        predictUndo: top ? 2 : -1,
        predictPieces: top ? 2 : -1,
      });
      expect(hydrateState(serializeState(hydrated))).toEqual(hydrated);
    }
  });

  it("treats infinities and NaN as missing", () => {
    const text = JSON.stringify(initialState).replace(/"tipX":0\.9/, '"tipX":1e999').replace(/"keep":1/, '"keep":-1e999');
    expect(hydrateState(text)).toEqual(initialState);
    expect(cleanState({ times: Number.NaN, turn: Infinity })).toEqual(initialState);
  });

  it("rounds whole-number settings and leaves slider values as they are", () => {
    const hydrated = hydrateState(JSON.stringify({ pieces: 2.6, stage: 0.4, round: 1.5, solved: 5.2, times: 0.33 }));
    expect(hydrated).toMatchObject({ pieces: 3, stage: 0, round: 2, solved: 5, times: 0.33 });
  });

  it("accepts every word, step, sequence move and undo choice", () => {
    for (const word of ["mouse", "cat", "dog", "horse", "whale", "bicycle", "car", "bus"]) {
      expect(hydrateState(JSON.stringify({ word, other: word }))).toMatchObject({ word, other: word });
    }
    for (const push of ["bigger", "alive", "both"]) expect(hydrateState(JSON.stringify({ push })).push).toBe(push);
    for (const move of ["rotate", "stretch", "shear", "flip"]) {
      expect(hydrateState(JSON.stringify({ first: move, second: move }))).toMatchObject({ first: move, second: move });
    }
    for (const undo of ["leave", "stretch"]) expect(hydrateState(JSON.stringify({ undo })).undo).toBe(undo);
  });

  it("reads raw, zeroed state without breaking, as the lab does when no one has hydrated it", () => {
    const zeroed = Object.fromEntries(
      Object.entries(initialState).map(([key, value]) => [key, typeof value === "number" ? 0 : ""]),
    );
    const read = readState(zeroed);
    expect(read).toMatchObject({ word: "horse", other: "whale", push: "bigger", first: "rotate", second: "stretch" });
    expect(read).toMatchObject({ pieces: 1, stretch: 0.5, keep: 0, hint: false, tip: [0, 0], matrix: [0, 0, 0, 0] });
    expect(readState({})).toMatchObject({ word: "horse", tip: [0.9, 1.2], matrix: [1, 0.5, 0, 1], aim: [1, 0, 0, 1] });
  });

  it("keeps every scene's settings exactly as the scene wrote them", () => {
    for (const scene of scenes) {
      const state = { ...initialState, ...scene.state } as ModuleState;
      expect(hydrateState(serializeState(state)), scene.id).toEqual(state);
    }
  });
});
