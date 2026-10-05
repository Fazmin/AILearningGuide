import { describe, expect, it } from "vitest";
import definition from "./module";
import { MAX_CHARACTERS, MODEL_HEADS, MODEL_LAYERS, STAGE_COUNT, readState } from "./state";
import { MAX_WORDS } from "./attention-math";

/** A version 2 payload: the shape this lab saved before the mask, scaling and value-vector controls. */
const VERSION_2 = JSON.stringify({
  text: "The animal did not cross the road because it was tired",
  layer: 0,
  head: 3,
  selected: 8,
  inspected: 4,
  stage: 2,
  editVectors: true,
  matrixView: "characters",
  q0: 0.5,
  k1: -0.25,
});

describe("attention state", () => {
  it("is version 3, which added the mask, scaling and values switches", () => {
    expect(definition.stateVersion).toBe(3);
    expect(definition.initialState).toMatchObject({ mask: "causal", scaling: "scaled", values: "model" });
  });

  it("migrates a version 2 payload: keeps its sentence, head, view and edits, and gets the model's own mask, scaling and values", () => {
    const state = definition.hydrateState(VERSION_2);
    expect(state).toMatchObject({
      text: "The animal did not cross the road because it was tired",
      layer: 0,
      head: 3,
      selected: 8,
      inspected: 4,
      stage: 2,
      editVectors: true,
      matrixView: "characters",
      q0: 0.5,
      k1: -0.25,
      mask: "causal",
      scaling: "scaled",
      values: "model",
    });
  });

  it("round-trips the current state, switches included", () => {
    const state = { ...definition.initialState, mask: "bidirectional", scaling: "raw", values: "all", q2: 1.5 };
    expect(definition.hydrateState(definition.serializeState(state))).toEqual(state);
    expect(definition.hydrateState(definition.serializeState(definition.initialState))).toEqual(definition.initialState);
  });

  it("falls back to the defaults for malformed payloads", () => {
    for (const value of ["", "not json", "null", "3", "[]", '"text"', "{}"]) {
      expect(definition.hydrateState(value), value).toEqual(definition.initialState);
    }
  });

  it("rejects values outside each switch's set", () => {
    const state = definition.hydrateState(JSON.stringify({ mask: "open", scaling: 1, values: ["all"], matrixView: "grid", editVectors: "yes" }));
    expect(state).toMatchObject({ mask: "causal", scaling: "scaled", values: "model", matrixView: "words", editVectors: false });
  });

  it("clamps every number a hand-edited link can carry, and drops edits that are not finite", () => {
    for (const hostile of [1e9, -1e9]) {
      const state = readState({
        layer: hostile,
        head: hostile,
        selected: hostile,
        inspected: hostile,
        stage: hostile,
        q0: hostile,
        k3: hostile,
      });
      expect(state.layer).toBeGreaterThanOrEqual(0);
      expect(state.layer).toBeLessThanOrEqual(MODEL_LAYERS - 1);
      expect(state.head).toBeGreaterThanOrEqual(0);
      expect(state.head).toBeLessThanOrEqual(MODEL_HEADS - 1);
      expect(state.selected).toBeGreaterThanOrEqual(0);
      expect(state.selected).toBeLessThanOrEqual(MAX_WORDS - 1);
      expect(state.inspected).toBeLessThanOrEqual(MAX_WORDS - 1);
      expect(state.stage).toBeLessThanOrEqual(STAGE_COUNT - 1);
      expect(Math.abs(state.edits.q0)).toBeLessThanOrEqual(100);
      expect(Math.abs(state.edits.k3)).toBeLessThanOrEqual(100);
    }
    const infinite = definition.hydrateState('{"layer":1e999,"head":-1e999,"q1":1e999,"k2":"x"}');
    expect(infinite).toMatchObject({ layer: 1, head: 1 });
    expect(infinite).not.toHaveProperty("q1");
    expect(infinite).not.toHaveProperty("k2");
  });

  it("caps the sentence at the model's 64 characters", () => {
    const state = definition.hydrateState(JSON.stringify({ text: "ab".repeat(100) }));
    expect(Array.from(state.text as string)).toHaveLength(MAX_CHARACTERS);
  });
});
