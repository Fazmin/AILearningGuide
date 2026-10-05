import { describe, expect, it } from "vitest";
import definition from "./module";
import { DEFAULT_RECIPE, RECIPE_BOUNDS, runPipeline } from "./pipeline";
import { normalizeRecipe, recipeToState } from "./state";

describe("normalizeRecipe", () => {
  it("round-trips a valid recipe unchanged", () => {
    const recipe = { ...DEFAULT_RECIPE, replay: 3, format: "Q8_0", prune: 0.35, gateTarget: 0.045 };
    expect(normalizeRecipe(recipeToState(recipe))).toEqual(recipe);
  });

  it("falls back to the defaults for anything that is not a plain object", () => {
    for (const bad of [null, undefined, 3, "text", [], [1, 2, 3], true]) {
      expect(normalizeRecipe(bad), String(bad)).toEqual(DEFAULT_RECIPE);
    }
  });

  it("drops unknown keys and wrong types", () => {
    const recipe = normalizeRecipe({ evil: 1, __proto__: { polluted: true }, replay: "four", format: 7, users: null });
    expect(recipe).toEqual(DEFAULT_RECIPE);
    expect(Object.keys(recipe).sort()).toEqual(Object.keys(DEFAULT_RECIPE).sort());
  });

  it("clamps every number to its slider range and rounds whole-number controls", () => {
    const recipe = normalizeRecipe({ baseEpochs: 1e9, loraRank: -5, loraEpochs: 41.6, replay: 2.4, beta: 99, prune: -1, gateTarget: 5, users: 0, context: 1e12 });
    expect(recipe.baseEpochs).toBe(RECIPE_BOUNDS.baseEpochs[1]);
    expect(recipe.loraRank).toBe(RECIPE_BOUNDS.loraRank[0]);
    expect(recipe.loraEpochs).toBe(RECIPE_BOUNDS.loraEpochs[1]);
    expect(recipe.replay).toBe(2);
    expect(recipe.beta).toBe(RECIPE_BOUNDS.beta[1]);
    expect(recipe.prune).toBe(RECIPE_BOUNDS.prune[0]);
    expect(recipe.gateTarget).toBe(RECIPE_BOUNDS.gateTarget[1]);
    expect(recipe.users).toBe(RECIPE_BOUNDS.users[0]);
    expect(recipe.context).toBe(RECIPE_BOUNDS.context[1]);
  });

  it("ignores NaN and Infinity instead of letting them reach training", () => {
    const recipe = normalizeRecipe({ baseEpochs: Number.NaN, loraEpochs: Infinity, dpoSteps: -Infinity });
    expect(recipe.baseEpochs).toBe(DEFAULT_RECIPE.baseEpochs);
    expect(recipe.loraEpochs).toBe(DEFAULT_RECIPE.loraEpochs);
    expect(recipe.dpoSteps).toBe(DEFAULT_RECIPE.dpoSteps);
  });

  it("only accepts a format the lab offers", () => {
    expect(normalizeRecipe({ format: "Q2_K" }).format).toBe(DEFAULT_RECIPE.format);
    expect(normalizeRecipe({ format: "F16" }).format).toBe("F16");
  });

  it("bounds the work a hostile share link can request", () => {
    const hostile = normalizeRecipe({ baseEpochs: 1e12, loraEpochs: 1e12, dpoSteps: 1e12, loraRank: 1e12, users: 1e12, context: 1e12 });
    const started = performance.now();
    const release = runPipeline(hostile);
    expect(performance.now() - started).toBeLessThan(5000);
    expect(release.stages).toHaveLength(4);
  });
});

describe("definition.hydrateState", () => {
  it("returns the defaults for malformed JSON", () => {
    expect(definition.hydrateState("{not json")).toEqual(definition.initialState);
    expect(definition.hydrateState("null")).toEqual(definition.initialState);
    expect(definition.hydrateState("[1,2]")).toEqual(definition.initialState);
  });

  it("serializes and hydrates the initial state without change", () => {
    expect(definition.hydrateState(definition.serializeState(definition.initialState))).toEqual(definition.initialState);
  });

  it("clamps a crafted payload", () => {
    const state = definition.hydrateState(JSON.stringify({ baseEpochs: 1e9, format: "bogus", replay: 99 }));
    expect(state.baseEpochs).toBe(RECIPE_BOUNDS.baseEpochs[1]);
    expect(state.format).toBe(DEFAULT_RECIPE.format);
    expect(state.replay).toBe(RECIPE_BOUNDS.replay[1]);
  });
});
