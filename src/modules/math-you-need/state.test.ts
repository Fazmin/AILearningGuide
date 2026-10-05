import { describe, expect, it } from "vitest";
import definition from "./module";

describe("math-you-need state", () => {
  const { hydrateState, initialState, serializeState } = definition;

  it("round-trips the defaults, including the probability keys", () => {
    expect(hydrateState(serializeState(initialState))).toEqual(initialState);
    expect(initialState).toMatchObject({ probIdea: "counts", prior: 0.5, gaussMu: 0, gaussVar: 1, gaussX: 1 });
  });

  it("fills the new keys when an older (version 2) payload has none", () => {
    const version2 = JSON.stringify({ topic: "probability", pA: 3, pB: 0.5, surface: "nonconvex", optW: 1.4 });
    expect(hydrateState(version2)).toEqual({
      ...initialState,
      topic: "probability",
      pA: 3,
      pB: 0.5,
      surface: "nonconvex",
      optW: 1.4,
    });
  });

  it("validates and clamps every new field of a malformed payload", () => {
    const hydrated = hydrateState(
      JSON.stringify({
        topic: "calculus",
        probIdea: "tarot",
        prior: 7,
        gaussMu: -50,
        gaussVar: 0,
        gaussX: "far",
        surface: 3,
        extra: "ignored",
      }),
    );
    expect(hydrated).toEqual({ ...initialState, prior: 0.98, gaussMu: -2, gaussVar: 0.1 });
  });

  it("accepts every valid probability idea", () => {
    for (const probIdea of ["counts", "bayes", "gaussian", "logit"]) {
      expect(hydrateState(JSON.stringify({ probIdea })).probIdea).toBe(probIdea);
    }
  });
});
