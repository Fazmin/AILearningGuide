import { describe, expect, it } from "vitest";
import definition from "./module";
import { hydrateTrainingRunState, initialState, readState } from "./state";

describe("training-run state", () => {
  it("round-trips the defaults", () => {
    expect(hydrateTrainingRunState(JSON.stringify(initialState))).toEqual(initialState);
    expect(hydrateTrainingRunState("{}")).toEqual(initialState);
    expect(hydrateTrainingRunState("not json")).toEqual(initialState);
    expect(hydrateTrainingRunState("[1, 2]")).toEqual(initialState);
  });

  it("migrates a version 1 payload: old keys keep their meaning, new keys take their defaults", () => {
    const version1 = JSON.stringify({ stage: 4, epochs: 30, batchSize: 8, checkpoint: 99 });
    expect(hydrateTrainingRunState(version1)).toEqual({
      ...initialState,
      stage: 4,
      epochs: 30,
      batchSize: 8,
      checkpoint: 99,
    });
    expect(definition.stateVersion).toBe(2);
  });

  it("clamps every number a hostile link can carry, including values JSON cannot spell", () => {
    for (const value of ["1e9", "-1e9", "1e999", "-1e999", "null", "\"7\"", "true"]) {
      const hostile = `{"stage":${value},"epochs":${value},"batchSize":${value},"checkpoint":${value},"stepsTaken":${value}}`;
      const state = readState(JSON.parse(hostile));
      expect(state.stage).toBeGreaterThanOrEqual(0);
      expect(state.stage).toBeLessThanOrEqual(6);
      expect(state.epochs).toBeGreaterThanOrEqual(1);
      expect(state.epochs).toBeLessThanOrEqual(60);
      expect(state.batchSize).toBeGreaterThanOrEqual(2);
      expect(state.batchSize).toBeLessThanOrEqual(64);
      expect(state.checkpoint).toBeGreaterThanOrEqual(0);
      expect(state.checkpoint).toBeLessThanOrEqual(99);
      expect(state.stepsTaken).toBeGreaterThanOrEqual(0);
      expect(state.stepsTaken).toBeLessThanOrEqual(24);
    }
    expect(readState({ epochs: 1e9 }).epochs).toBe(60);
    expect(readState({ epochs: -1e9 }).epochs).toBe(1);
    expect(readState({ stepsTaken: 1e9 }).stepsTaken).toBe(24);
    expect(readState({ epochs: Infinity }).epochs).toBe(12);
    expect(readState({ epochs: NaN }).epochs).toBe(12);
    expect(readState({ epochs: 12.6 }).epochs).toBe(13);
  });

  it("accepts only known memory choices", () => {
    expect(readState({ memModel: "70b", memPrecision: "fp32", memOptimizer: "sgd" })).toMatchObject({
      memModel: "70b",
      memPrecision: "fp32",
      memOptimizer: "sgd",
    });
    expect(readState({ memModel: "1e99", memPrecision: "fp4", memOptimizer: 3 })).toMatchObject({
      memModel: "7.5b",
      memPrecision: "mixed",
      memOptimizer: "adam",
    });
  });
});
