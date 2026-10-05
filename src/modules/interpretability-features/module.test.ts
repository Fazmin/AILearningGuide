import { describe, expect, it } from "vitest";
import definition, { hydrateFeatureState } from "./module";
import { LABELLED_FEATURES, MAX_STRENGTH, NULL_DIRECTIONS } from "./steering";
import { HYPOTHESES } from "./sae";

describe("interpretability-features state", () => {
  it("resets a version-1 payload, whose keys meant something else", () => {
    const state = hydrateFeatureState(JSON.stringify({ feature: 12, threshold: 42, overlap: 35 }));
    expect(state).toEqual(definition.initialState);
    expect(state.overlap).toBeUndefined();
  });

  it("translates a version-2 payload from the first SAE: feature ids are renumbered, so they reset", () => {
    // What the first release saved: feature 686 was a space feature of the old SAE; 624 and 511 had their own stories.
    const old = JSON.stringify({ text: "my own probe, here", token: 7, feature: 624, threshold: 3.5, hypothesis: "upper", keep: 12 });
    const state = hydrateFeatureState(old);
    expect(state.text).toBe("my own probe, here");
    expect(state.token).toBe(7);
    expect(state.keep).toBe(12);
    // Everything bound to what an old feature id showed goes back to the defaults.
    expect(state.feature).toBe(definition.initialState.feature);
    expect(state.threshold).toBe(definition.initialState.threshold);
    expect(state.hypothesis).toBe(definition.initialState.hypothesis);
    expect(state.sae).toBe(2);
    // The steering keys did not exist and take their defaults.
    expect(state.steerFeature).toBe(160);
    expect(state.steerAt).toBe(43);
    expect(state.strength).toBe(10);
    expect(state.seed).toBe(1);
    expect(state.examples).toBe("top");
  });

  it("keeps a current payload's feature, threshold, hypothesis and steering choices", () => {
    const state = hydrateFeatureState(
      JSON.stringify({ sae: 2, text: "abc", token: 2, feature: 511, threshold: 3.2, hypothesis: "space", examples: "context", keep: 5, steerFeature: 683, steerAt: 2, strength: 4, seed: 9 }),
    );
    expect(state).toEqual({
      sae: 2,
      text: "abc",
      token: 2,
      feature: 511,
      threshold: 3.2,
      hypothesis: "space",
      examples: "context",
      keep: 5,
      steerFeature: 683,
      steerAt: 2,
      strength: 4,
      seed: 9,
    });
  });

  it("clamps and validates every key of a current payload", () => {
    const state = hydrateFeatureState(
      JSON.stringify({
        sae: 2,
        text: "x".repeat(90),
        token: 1e9,
        feature: 5000,
        threshold: -3,
        hypothesis: "vibes",
        examples: "everything",
        keep: 99,
        steerFeature: 1e9,
        steerAt: -1e9,
        strength: 77.6,
        seed: 0,
      }),
    );
    expect(state.text).toHaveLength(64);
    expect(state.token).toBe(63);
    expect(state.feature).toBe(1023);
    expect(state.threshold).toBe(0);
    expect(state.hypothesis).toBe("start");
    expect(state.examples).toBe("top");
    expect(state.keep).toBe(32);
    expect(state.steerFeature).toBe(160);
    expect(state.steerAt).toBe(0);
    expect(state.strength).toBe(15);
    expect(state.seed).toBe(1);
    expect(definition.stateVersion).toBe(3);
  });

  it("survives non-object payloads and non-numeric values", () => {
    for (const payload of ["null", "5", '"text"', "[1,2]", "not json", "{}"]) {
      expect(hydrateFeatureState(payload)).toEqual(definition.initialState);
    }
    const state = hydrateFeatureState(JSON.stringify({ sae: 2, token: "3", feature: null, strength: { a: 1 }, seed: Infinity, text: 7 }));
    expect(state.token).toBe(4);
    expect(state.feature).toBe(876);
    expect(state.strength).toBe(10);
    expect(state.seed).toBe(1);
    expect(state.text).toBe(definition.initialState.text);
  });

  it("round-trips its initial state, and its limits match what the lab offers", () => {
    expect(hydrateFeatureState(definition.serializeState(definition.initialState))).toEqual(definition.initialState);
    expect(MAX_STRENGTH).toBe(15);
    expect(NULL_DIRECTIONS).toBe(20);
    expect(HYPOTHESES.map((entry) => entry.value)).toEqual(["char", "upper", "space", "newline", "start"]);
    expect(LABELLED_FEATURES.map((entry) => entry.feature)).toContain(definition.initialState.steerFeature);
  });
});

describe("interpretability-features definition", () => {
  it("has three objectives, each tested by at least one checkpoint question", () => {
    expect(definition.objectives).toHaveLength(3);
    const questions = Array.isArray(definition.checkpoint) ? definition.checkpoint : [definition.checkpoint];
    const covered = new Set(questions.map((question) => question.objective));
    expect([...covered].sort()).toEqual([0, 1, 2]);
    expect(questions.length).toBeGreaterThanOrEqual(6);
  });

  it("has one instruction per step and a glossary of 14 terms", () => {
    expect(definition.steps).toHaveLength(5);
    expect(definition.stepInstructions).toHaveLength(5);
    expect(definition.glossary).toHaveLength(14);
  });
});
