import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import definition from "./module";
import { defaultState, hydrateEmbeddingTrainingState, initialState, readState } from "./state";

const render = (state: Record<string, unknown>) =>
  renderToStaticMarkup(
    createElement(definition.Explore, {
      state: state as typeof initialState,
      setState: () => undefined,
      currentStep: 0,
      mode: "standard" as const,
      narrate: () => undefined,
    }),
  );

describe("embedding-training state", () => {
  it("round-trips the defaults and survives garbage", () => {
    expect(initialState).toEqual({
      step: 300,
      batch: 8,
      temperature: 0.1,
      negatives: "in-batch",
      skipDuplicates: false,
      dim: 16,
      seed: 1,
      query: "h01",
      rerank: false,
    });
    expect(hydrateEmbeddingTrainingState(JSON.stringify(initialState))).toEqual(initialState);
    expect(hydrateEmbeddingTrainingState("{}")).toEqual(initialState);
    expect(hydrateEmbeddingTrainingState("not json")).toEqual(initialState);
    expect(hydrateEmbeddingTrainingState("[1]")).toEqual(initialState);
    expect(hydrateEmbeddingTrainingState("null")).toEqual(initialState);
    expect(definition.stateVersion).toBe(1);
  });

  it("clamps and snaps every number a hostile link can carry", () => {
    const high = readState({ step: 1e9, batch: 1e9, temperature: 1e9, dim: 1e9, seed: 1e9 });
    expect([high.step, high.batch, high.temperature, high.dim, high.seed]).toEqual([300, 32, 1, 16, 20]);
    const low = readState({ step: -1e9, batch: -1e9, temperature: -1e9, dim: -1e9, seed: -1e9 });
    expect([low.step, low.batch, low.temperature, low.dim, low.seed]).toEqual([0, 2, 0.02, 8, 1]);
    // JSON cannot spell Infinity, but 1e999 parses to it.
    const infinite = readState(JSON.parse('{"step":1e999,"batch":-1e999,"temperature":1e999,"seed":1e999}'));
    expect([infinite.step, infinite.batch, infinite.temperature, infinite.seed]).toEqual([300, 8, 0.1, 1]);
    expect(readState({ batch: Number.NaN }).batch).toBe(8);
  });

  it("snaps to the values the controls can produce", () => {
    expect(readState({ step: 147 }).step).toBe(150);
    expect(readState({ step: 3 }).step).toBe(0);
    expect(readState({ batch: 5 }).batch).toBe(4);
    expect(readState({ batch: 20 }).batch).toBe(16);
    expect(readState({ dim: 13 }).dim).toBe(16);
    expect(readState({ dim: 9 }).dim).toBe(8);
    expect(readState({ temperature: 0.034999 }).temperature).toBe(0.03);
    expect(readState({ temperature: 0.1000000001 }).temperature).toBe(0.1);
    expect(readState({ seed: 7.6 }).seed).toBe(8);
  });

  it("accepts only known negatives, queries and real booleans, and drops unknown keys", () => {
    expect(readState({ negatives: "nearest" }).negatives).toBe("nearest");
    expect(readState({ negatives: "random" }).negatives).toBe("in-batch");
    expect(readState({ query: "h26" }).query).toBe("h26");
    expect(readState({ query: "t01" }).query).toBe("h01");
    expect(readState({ skipDuplicates: "true", rerank: 1 })).toMatchObject({ skipDuplicates: false, rerank: false });
    expect(readState({ skipDuplicates: true, rerank: true })).toMatchObject({ skipDuplicates: true, rerank: true });
    expect(hydrateEmbeddingTrainingState(JSON.stringify({ extra: 1, step: 100 }))).toEqual({ ...defaultState, step: 100 });
  });

  it("reads state only through readState, so a zeroed or emptied state still renders", () => {
    const html = render({ step: 0, batch: 0, temperature: 0, negatives: "", skipDuplicates: false, dim: 0, seed: 0, query: "", rerank: false });
    expect(html).not.toMatch(/\bNaN\b|\bundefined\b|\bInfinity\b/);
    expect(html).toContain('data-lab-surface="Query inspector"');
  });

  it("renders the heaviest setting, batch 32 with Nearest wrong, in well under a second", () => {
    render({ ...initialState, batch: 4 });
    const started = performance.now();
    const html = render({ ...initialState, batch: 32, negatives: "nearest", temperature: 0.02 });
    const elapsed = performance.now() - started;
    expect(elapsed, `${Math.round(elapsed)} ms`).toBeLessThan(300);
    expect(html).toContain("Nearest wrong");
  });
});
