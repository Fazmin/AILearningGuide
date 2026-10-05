import { describe, expect, it } from "vitest";
import { hydrateTokensState, initialState } from "./state";

describe("tokens-embeddings state", () => {
  it("migrates a version 1 payload (1–5 vocabulary level) to a vocabulary size", () => {
    const legacy = JSON.stringify({
      text: "attention turns context into meaning",
      vocabulary: 1,
      focus: 3,
      word: "king",
    });
    const hydrated = hydrateTokensState(legacy);
    expect(hydrated.vocabSize).toBe(69);
    expect(hydrated.text).toBe("attention turns context into meaning");
    expect(hydrated.focus).toBe(3);
    expect(hydrated.word).toBe("king");
    expect(hydrated).not.toHaveProperty("vocabulary");
    expect(hydrateTokensState(JSON.stringify({ vocabulary: 3 })).vocabSize).toBe(290);
  });

  it("clamps and validates current payloads", () => {
    const hydrated = hydrateTokensState(
      JSON.stringify({ vocabSize: 9000, projection: "tsne", word: "Queen", exclude: "yes" }),
    );
    expect(hydrated.vocabSize).toBe(512);
    expect(hydrated.projection).toBe("umap");
    expect(hydrated.word).toBe("queen");
    expect(hydrated.exclude).toBe(true);
  });

  it("falls back to defaults on malformed JSON", () => {
    expect(hydrateTokensState("{not json")).toEqual(initialState);
    expect(hydrateTokensState("[1,2]")).toEqual(initialState);
  });
});
