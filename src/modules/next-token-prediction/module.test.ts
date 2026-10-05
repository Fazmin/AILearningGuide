import { describe, expect, it } from "vitest";
import definition from "./module";
import { DEFAULT_PROMPT, TRAINING_LINE } from "./prompts";
import { MIN_P_MAX } from "./sampling";

describe("next-token-prediction state", () => {
  it("is version 3: min-p was added and the default prompt changed", () => {
    expect(definition.stateVersion).toBe(3);
    expect(definition.initialState).toEqual({
      prompt: DEFAULT_PROMPT,
      temperature: 0.8,
      model: "rnn",
      sample: 2,
      topK: 66,
      topP: 1,
      minP: 0,
      generated: "",
      draws: [],
    });
  });

  it("keeps a version 1 payload's prompt, temperature and model, and reads its counter as a seed", () => {
    const v1 = JSON.stringify({ prompt: "The model learns from", temperature: 1.25, model: "bigram", sample: 70 });
    expect(definition.hydrateState(v1)).toEqual({
      prompt: "The model learns from",
      temperature: 1.25,
      model: "bigram",
      sample: 6,
      topK: 66,
      topP: 1,
      minP: 0,
      generated: "",
      draws: [],
    });
  });

  it("keeps a version 2 payload's own prompt, settings and continuation, with min-p off", () => {
    const v2 = JSON.stringify({
      prompt: "My own line",
      temperature: 0.5,
      model: "transformer",
      sample: 9,
      topK: 5,
      topP: 0.9,
      generated: "ab",
      draws: ["transformer|0.1|0.2|0.3", "transformer|0.2|0.2|0.3"],
    });
    expect(definition.hydrateState(v2)).toEqual({
      prompt: "My own line",
      temperature: 0.5,
      model: "transformer",
      sample: 9,
      topK: 5,
      topP: 0.9,
      minP: 0,
      generated: "ab",
      draws: ["transformer|0.1|0.2|0.3", "transformer|0.2|0.2|0.3"],
    });
  });

  it("moves a version 2 payload that still holds the retired default prompt to the new default and drops its continuation", () => {
    // Version 2's default prompt was a line from the training text. A payload with no minP key still
    // holding it was never the learner's choice, and the continuation was drawn from it.
    expect(TRAINING_LINE).toBe("O Romeo, Romeo! wherefore art thou R");
    const v2 = JSON.stringify({
      prompt: TRAINING_LINE,
      temperature: 0.8,
      model: "rnn",
      sample: 2,
      topK: 66,
      topP: 1,
      generated: "o",
      draws: ["rnn|0.5|0.4|0.5"],
    });
    const state = definition.hydrateState(v2);
    expect(state.prompt).toBe(DEFAULT_PROMPT);
    expect(state.generated).toBe("");
    expect(state.draws).toEqual([]);
    expect(state.minP).toBe(0);
  });

  it("keeps the training line when a current payload (with minP) holds it: the learner chose it", () => {
    const current = JSON.stringify({ ...definition.initialState, prompt: TRAINING_LINE, generated: "o", draws: ["rnn|0.5|0.4|0.5"] });
    const state = definition.hydrateState(current);
    expect(state.prompt).toBe(TRAINING_LINE);
    expect(state.generated).toBe("o");
    expect(state.draws).toEqual(["rnn|0.5|0.4|0.5"]);
  });

  it("drops draw records when there is no continuation and trims them to the continuation length", () => {
    expect(definition.hydrateState(JSON.stringify({ draws: ["rnn|0.1|0.2|0.3"] })).draws).toEqual([]);
    const state = definition.hydrateState(
      JSON.stringify({ generated: "ab", draws: ["rnn|0.1|0.2|0.3", "rnn|0.2|0.2|0.3", "bigram|0.3|0.2|0.3", 7] }),
    );
    expect(state.draws).toEqual(["rnn|0.2|0.2|0.3", "bigram|0.3|0.2|0.3"]);
  });

  it("falls back to the defaults for an unknown model and out-of-range sampler settings", () => {
    const state = definition.hydrateState(JSON.stringify({ model: "gpt", topK: 500, topP: 0, temperature: -3, minP: 7 }));
    expect(state.model).toBe("rnn");
    expect(state.topK).toBe(66);
    expect(state.topP).toBe(0.05);
    expect(state.temperature).toBe(0.1);
    expect(state.minP).toBe(MIN_P_MAX);
    expect(definition.hydrateState(JSON.stringify({ minP: -4 })).minP).toBe(0);
    expect(definition.hydrateState(JSON.stringify({ minP: "high" })).minP).toBe(0);
    // JSON has no Infinity; "1e999" parses to it, which is how a hand-edited link can carry one.
    // A non-finite number is not a setting, so the default stands.
    expect(definition.hydrateState('{"minP": 1e999, "topK": 1e999, "temperature": -1e999}')).toMatchObject({
      minP: 0,
      topK: 66,
      temperature: 0.8,
    });
  });

  it("caps the prompt and the continuation at their limits", () => {
    const state = definition.hydrateState(JSON.stringify({ prompt: "x".repeat(5000), generated: "y".repeat(5000), minP: 0 }));
    expect(String(state.prompt)).toHaveLength(120);
    expect(Array.from(String(state.generated))).toHaveLength(120);
  });

  it("keeps the new default and the lesson's objectives, steps and glossary in step", () => {
    expect(definition.steps).toHaveLength(5);
    expect(definition.stepInstructions).toHaveLength(5);
    expect(definition.objectives).toHaveLength(4);
    expect(definition.glossary.length).toBeLessThanOrEqual(14);
    expect(definition.glossary.map((entry) => entry.term)).toEqual(
      expect.arrayContaining(["Min-p sampling", "End-of-sequence token", "Logit lens", "Unembedding", "Residual stream"]),
    );
  });
});
