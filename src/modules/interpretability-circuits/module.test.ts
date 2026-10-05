/// <reference types="node" />
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { checkpointQuestions } from "@app/module-sdk";
import definition, { hydrateCircuitState } from "./module";
import { PERIODS } from "./experiments";

describe("interpretability-circuits state", () => {
  it("is at version 3 with every key in the initial state", () => {
    expect(definition.stateVersion).toBe(3);
    expect(Object.keys(definition.initialState).sort()).toEqual(
      ["ablated", "ablation", "attnHead", "attnLayer", "corruption", "layer", "patch", "period", "sweep"].sort(),
    );
  });

  it("migrates a version-1 payload", () => {
    const state = hydrateCircuitState(JSON.stringify({ layer: 3, patch: 5, steering: 24, ablated: true }));
    expect(state.layer).toBe(1);
    expect(state.patch).toBe(5);
    expect(state.ablated).toEqual([]);
    expect(state.steering).toBeUndefined();
    expect(state.period).toBe(8);
    expect(state.ablation).toBe("zero");
    expect(state.corruption).toBe("source");
  });

  it("migrates a version-2 payload: every old value is kept and the new keys take their defaults", () => {
    // Version 2 had no `ablation` or `corruption`, offered periods 6, 7, 8, 9, 10 and 12, and opened on L1 H1.
    const fixture = {
      period: 12,
      attnLayer: 0,
      attnHead: 0,
      ablated: ["L1H2", "L2H3"],
      sweep: "copy",
      layer: 1,
      patch: 23,
    };
    const state = hydrateCircuitState(JSON.stringify(fixture));
    expect(state).toEqual({ ...fixture, ablation: "zero", corruption: "source" });
    for (const period of [6, 7, 8, 9, 10, 12]) {
      expect(hydrateCircuitState(JSON.stringify({ period })).period).toBe(period);
    }
  });

  it("opens on the induction head L2 H3 for a new learner", () => {
    expect(definition.initialState.attnLayer).toBe(1);
    expect(definition.initialState.attnHead).toBe(2);
  });

  it("keeps valid head ids and periods, including the unseen 11 and 13, and drops the rest", () => {
    const state = hydrateCircuitState(
      JSON.stringify({ ablated: ["L1H2", "L9H9", "L1H2"], period: 11, sweep: "speaker", ablation: "mean", corruption: "control" }),
    );
    expect(state.ablated).toEqual(["L1H2"]);
    expect(state.period).toBe(11);
    expect(state.sweep).toBe("speaker");
    expect(state.ablation).toBe("mean");
    expect(state.corruption).toBe("control");
    expect(hydrateCircuitState(JSON.stringify({ period: 13 })).period).toBe(13);
    expect(hydrateCircuitState("{")).toEqual(definition.initialState);
  });

  it("clamps or rejects every key under hostile values", () => {
    for (const value of [1e9, -1e9, 1e999, Number.NaN, 7.6, "7", null, {}, []]) {
      const state = hydrateCircuitState(
        JSON.stringify({ period: value, attnLayer: value, attnHead: value, layer: value, patch: value, ablation: value, corruption: value, sweep: value, ablated: value }),
      );
      expect(PERIODS as readonly number[], `period for ${String(value)}`).toContain(state.period);
      expect([0, 1]).toContain(state.attnLayer);
      expect([0, 1, 2, 3]).toContain(state.attnHead);
      expect([0, 1]).toContain(state.layer);
      expect(state.patch).toBeGreaterThanOrEqual(0);
      expect(state.patch).toBeLessThanOrEqual(63);
      expect(["zero", "mean"]).toContain(state.ablation);
      expect(["source", "control"]).toContain(state.corruption);
      expect(["live", "speaker", "copy"]).toContain(state.sweep);
      expect(Array.isArray(state.ablated)).toBe(true);
    }
    // "1e999" parses to Infinity, which is how a hand-edited share link can carry one.
    const infinite = hydrateCircuitState('{"period":1e999,"patch":-1e999,"attnHead":1e999,"layer":1e999}');
    // A value that is not finite falls back to the default; a finite one is clamped.
    expect(infinite.period).toBe(8);
    expect(infinite.patch).toBe(16);
    expect(infinite.attnHead).toBe(2);
    expect(infinite.layer).toBe(0);
    expect(hydrateCircuitState(JSON.stringify({ period: 1e9 })).period).toBe(13);
    expect(hydrateCircuitState(JSON.stringify({ period: -1e9 })).period).toBe(6);
    expect(hydrateCircuitState(JSON.stringify({ patch: 1e9 })).patch).toBe(63);
  });

  it("round-trips through serializeState", () => {
    const state = { ...definition.initialState, period: 13, ablation: "mean", corruption: "control", ablated: ["L1H4"] };
    expect(definition.hydrateState(definition.serializeState(state))).toEqual(state);
  });
});

describe("interpretability-circuits checkpoint", () => {
  const questions = checkpointQuestions(definition);

  it("covers all three objectives, two questions each", () => {
    expect(definition.objectives).toHaveLength(3);
    expect(questions).toHaveLength(6);
    for (let objective = 0; objective < definition.objectives.length; objective += 1) {
      expect(questions.filter((question) => question.objective === objective)).toHaveLength(2);
    }
  });

  it("does not let the longest option give the answer away", () => {
    const longest = questions.filter((question) => {
      const lengths = question.options.map((option) => option.length);
      const best = Math.max(...lengths);
      return lengths[question.answer] === best && lengths.filter((length) => length === best).length === 1;
    });
    expect(longest.length).toBeLessThanOrEqual(2);
  });

  it("keeps the glossary at 14 terms and in sync with both lessons", () => {
    const terms = definition.glossary.map((entry) => entry.term);
    expect(terms).toHaveLength(14);
    expect(terms).toContain("Previous-token head");
    for (const mode of ["standard", "plain"]) {
      const file = readFileSync(resolve(process.cwd(), `src/modules/interpretability-circuits/content/${mode}.mdx`), "utf8");
      const listed = JSON.parse(file.match(/^terms:\s*(\[.*\])\s*$/m)![1]) as string[];
      expect(listed).toEqual(terms);
    }
  });
});
