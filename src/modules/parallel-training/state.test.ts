import { describe, expect, it } from "vitest";
import definition from "./module";
import { defaultState, hydrateParallelState, initialState, LLAMA3_LAYOUT, readState } from "./state";

describe("parallel-training state", () => {
  it("round-trips the defaults and falls back to them on anything malformed", () => {
    expect(hydrateParallelState(JSON.stringify(initialState))).toEqual(initialState);
    expect(hydrateParallelState("{}")).toEqual(initialState);
    expect(hydrateParallelState("not json")).toEqual(initialState);
    expect(hydrateParallelState("[1, 2]")).toEqual(initialState);
    expect(hydrateParallelState("null")).toEqual(initialState);
    expect(hydrateParallelState('"8b"')).toEqual(initialState);
    expect(definition.stateVersion).toBe(1);
    expect(definition.initialState).toEqual(initialState);
  });

  it("starts where the first step needs it: the 8B model on 64 replicas at ZeRO stage 0", () => {
    expect(defaultState).toMatchObject({ modelId: "8b", tensor: 1, pipeline: 1, data: 64, zero: 0, microBatch: 1, recompute: "selective" });
  });

  it("drops unknown keys and keeps only known choices", () => {
    const state = readState({ ...defaultState, injected: "<script>", users: 9 });
    expect(Object.keys(state).sort()).toEqual(Object.keys(defaultState).sort());
    expect(readState({ modelId: "gpt-5", recompute: "sometimes", overlap: "yes" })).toMatchObject({
      modelId: "8b",
      recompute: "selective",
      overlap: true,
    });
    expect(readState({ modelId: "mixtral", recompute: "full", overlap: false })).toMatchObject({
      modelId: "mixtral",
      recompute: "full",
      overlap: false,
    });
  });

  it("snaps every parallel degree, the micro-batch size, and the microbatch count down to its ladder", () => {
    expect(readState({ tensor: 12, pipeline: 5, data: 300, microBatch: 3, microbatches: 100 })).toMatchObject({
      tensor: 8,
      pipeline: 4,
      data: 256,
      microBatch: 2,
      microbatches: 64,
    });
    expect(readState({ tensor: 0, pipeline: -2, data: 0.5, microBatch: 0, microbatches: 0 })).toMatchObject({
      tensor: 1,
      pipeline: 1,
      data: 1,
      microBatch: 1,
      microbatches: 1,
    });
  });

  it("never lets Expert parallel exceed Data parallel", () => {
    expect(readState({ expert: 8, data: 4 }).expert).toBe(4);
    expect(readState({ expert: 5, data: 64 }).expert).toBe(4);
    expect(readState({ expert: 8, data: 64 }).expert).toBe(8);
    expect(readState({ expert: 8, data: 1 }).expert).toBe(1);
  });

  it("rounds and clamps ZeRO stage and the three failure controls", () => {
    expect(readState({ zero: 2.6 }).zero).toBe(3);
    expect(readState({ zero: 9 }).zero).toBe(3);
    expect(readState({ zero: -1 }).zero).toBe(0);
    expect(readState({ checkpointMinutes: 19.4, saveSeconds: 62, restartMinutes: 0 })).toMatchObject({
      checkpointMinutes: 19,
      saveSeconds: 60,
      restartMinutes: 1,
    });
  });

  it("clamps every number a hostile link can carry, including values JSON cannot spell", () => {
    for (const value of ["1e9", "-1e9", "1e999", "-1e999", "null", '"7"', "true"]) {
      const keys = ["tensor", "pipeline", "data", "expert", "zero", "microBatch", "microbatches", "checkpointMinutes", "saveSeconds", "restartMinutes"];
      const hostile = `{${keys.map((key) => `"${key}":${value}`).join(",")}}`;
      const state = readState(JSON.parse(hostile));
      expect(state.tensor).toBeGreaterThanOrEqual(1);
      expect(state.tensor).toBeLessThanOrEqual(32);
      expect(state.pipeline).toBeLessThanOrEqual(16);
      expect(state.data).toBeLessThanOrEqual(2048);
      expect(state.expert).toBeLessThanOrEqual(Math.min(8, state.data));
      expect(state.zero).toBeGreaterThanOrEqual(0);
      expect(state.zero).toBeLessThanOrEqual(3);
      expect(state.microBatch).toBeLessThanOrEqual(8);
      expect(state.microbatches).toBeLessThanOrEqual(128);
      expect(state.checkpointMinutes).toBeGreaterThanOrEqual(5);
      expect(state.checkpointMinutes).toBeLessThanOrEqual(120);
      expect(state.saveSeconds).toBeGreaterThanOrEqual(5);
      expect(state.saveSeconds).toBeLessThanOrEqual(600);
      expect(state.restartMinutes).toBeGreaterThanOrEqual(1);
      expect(state.restartMinutes).toBeLessThanOrEqual(60);
    }
    expect(readState({ data: 1e9, tensor: 1e9 })).toMatchObject({ data: 2048, tensor: 32 });
    expect(readState({ data: Number.POSITIVE_INFINITY, tensor: Number.NaN })).toMatchObject({ data: 64, tensor: 1 });
  });

  it("keeps the 16,384-GPU layout exactly as the button sets it", () => {
    const state = readState({ ...defaultState, ...LLAMA3_LAYOUT });
    expect(state).toMatchObject(LLAMA3_LAYOUT);
    expect(state.tensor * state.pipeline * state.data).toBe(16384);
  });
});
