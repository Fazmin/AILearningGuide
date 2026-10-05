import { describe, expect, it } from "vitest";
import definition, { hydrateHyperState } from "./module";
import { defaultState, initialState, readState } from "./state";

describe("hyperparameters state", () => {
  it("round-trips the defaults and survives garbage", () => {
    expect(hydrateHyperState(JSON.stringify(initialState))).toEqual(initialState);
    expect(hydrateHyperState("{}")).toEqual(initialState);
    expect(hydrateHyperState("not json")).toEqual(initialState);
    expect(hydrateHyperState("[1]")).toEqual(initialState);
  });

  it("migrates a version 2 payload: old keys keep their meaning, the seed switch and depth controls take defaults", () => {
    const version2 = JSON.stringify({
      schedule: "one-cycle",
      peakLearningRate: 12,
      warmup: 0.3,
      clipNorm: 0.4,
      weightDecay: 0.02,
      batchSize: 8,
      inspect: 20,
    });
    expect(hydrateHyperState(version2)).toEqual({
      ...defaultState,
      schedule: "one-cycle",
      peakLearningRate: 12,
      warmup: 0.3,
      clipNorm: 0.4,
      weightDecay: 0.02,
      batchSize: 8,
      inspect: 20,
    });
    // Version 1 had no inspected epoch either.
    expect(hydrateHyperState(JSON.stringify({ schedule: "constant", batchSize: 4 })).inspect).toBe(12);
    expect(definition.stateVersion).toBe(3);
  });

  it("clamps every number a hostile link can carry", () => {
    const numeric = ["peakLearningRate", "warmup", "clipNorm", "weightDecay", "batchSize", "inspect", "depthLayers", "depthGain", "depthClip"] as const;
    for (const value of [1e9, -1e9]) {
      const state = readState(Object.fromEntries(numeric.map((key) => [key, value])));
      expect(state.peakLearningRate).toBeGreaterThanOrEqual(0.1);
      expect(state.peakLearningRate).toBeLessThanOrEqual(60);
      expect(state.batchSize).toBeGreaterThanOrEqual(2);
      expect(state.batchSize).toBeLessThanOrEqual(64);
      expect(state.inspect).toBeGreaterThanOrEqual(0);
      expect(state.inspect).toBeLessThanOrEqual(25);
      expect(state.depthLayers).toBeGreaterThanOrEqual(2);
      expect(state.depthLayers).toBeLessThanOrEqual(48);
      expect(state.depthGain).toBeGreaterThanOrEqual(0.7);
      expect(state.depthGain).toBeLessThanOrEqual(1.4);
      expect(state.depthClip).toBeGreaterThanOrEqual(0);
      expect(state.depthClip).toBeLessThanOrEqual(10);
    }
    // JSON cannot spell Infinity, but 1e999 parses to it.
    const infinite = readState(JSON.parse('{"batchSize":1e999,"depthLayers":-1e999,"depthGain":1e999}'));
    expect(infinite.batchSize).toBe(defaultState.batchSize);
    expect(infinite.depthLayers).toBe(defaultState.depthLayers);
    expect(infinite.depthGain).toBe(defaultState.depthGain);
    expect(readState({ batchSize: Number.NaN }).batchSize).toBe(defaultState.batchSize);
  });

  it("accepts only known schedules and seed modes", () => {
    expect(readState({ schedule: "linear" }).schedule).toBe("cosine");
    expect(readState({ schedule: "constant" }).schedule).toBe("constant");
    expect(readState({ seeds: "five" }).seeds).toBe("five");
    expect(readState({ seeds: "ten" }).seeds).toBe("one");
    expect(readState({ seeds: 5 }).seeds).toBe("one");
  });

  it("rounds the sliders to their steps", () => {
    expect(readState({ depthGain: 1.2000000000000002 }).depthGain).toBe(1.2);
    expect(readState({ depthClip: 1.3 }).depthClip).toBe(1.5);
    expect(readState({ depthLayers: 23.6 }).depthLayers).toBe(24);
  });
});
