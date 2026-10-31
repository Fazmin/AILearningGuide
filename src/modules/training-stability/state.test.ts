import { describe, expect, it } from "vitest";
import definition from "./module";
import { ADAM_RATES, Z_LOSSES } from "./faults";
import { defaultState, hydrateStabilityState, initialState, nearestRung, readState } from "./state";

describe("training-stability state", () => {
  it("round-trips the defaults and survives garbage", () => {
    expect(hydrateStabilityState(JSON.stringify(initialState))).toEqual(initialState);
    expect(hydrateStabilityState("{}")).toEqual(initialState);
    expect(hydrateStabilityState("not json")).toEqual(initialState);
    expect(hydrateStabilityState("[1, 2]")).toEqual(initialState);
    expect(hydrateStabilityState("null")).toEqual(initialState);
    expect(definition.stateVersion).toBe(1);
  });

  it("drops unknown keys and keeps the known ones", () => {
    const state = hydrateStabilityState(JSON.stringify({ fault: "rate", clipNorm: 0.5, injected: "<script>", lr: 900 }));
    expect(state).toEqual({ ...defaultState, fault: "rate", clipNorm: 0.5 });
    expect(Object.keys(state).sort()).toEqual(Object.keys(initialState).sort());
  });

  it("whitelists every choice", () => {
    expect(readState({ fault: "meltdown" }).fault).toBe("none");
    for (const fault of ["none", "rate", "fp16", "bad-batch", "logits"]) expect(readState({ fault }).fault).toBe(fault);
    expect(readState({ schedule: "cosine" }).schedule).toBe("constant");
    expect(readState({ schedule: "warmup-cosine" }).schedule).toBe("warmup-cosine");
    expect(readState({ scaling: "Dynamic" }).scaling).toBe("static");
    expect(readState({ scaling: "dynamic" }).scaling).toBe("dynamic");
    expect(readState({ fault: 3 }).fault).toBe("none");
  });

  it("accepts only a real true for the rollback flag", () => {
    expect(readState({ rollback: true }).rollback).toBe(true);
    for (const value of ["true", 1, "yes", null, {}]) expect(readState({ rollback: value }).rollback).toBe(false);
  });

  it("clamps every number a hostile link can carry", () => {
    const numeric = ["clipNorm", "lossScale", "zLoss", "adamRate"] as const;
    for (const value of [1e9, -1e9]) {
      const state = readState(Object.fromEntries(numeric.map((key) => [key, value])));
      expect(state.clipNorm).toBeGreaterThanOrEqual(0);
      expect(state.clipNorm).toBeLessThanOrEqual(1.5);
      expect(state.lossScale).toBeGreaterThanOrEqual(0);
      expect(state.lossScale).toBeLessThanOrEqual(24);
      expect(Z_LOSSES).toContain(state.zLoss);
      expect(ADAM_RATES).toContain(state.adamRate);
    }
    expect(readState({ lossScale: 1e9 }).lossScale).toBe(24);
    expect(readState({ lossScale: -1e9 }).lossScale).toBe(0);
    expect(readState({ zLoss: 1e9 }).zLoss).toBe(0.01);
    expect(readState({ zLoss: -1e9 }).zLoss).toBe(0);
    expect(readState({ adamRate: 1e9 }).adamRate).toBe(1);
    expect(readState({ adamRate: -1e9 }).adamRate).toBe(0.01);
    // JSON cannot spell Infinity, but 1e999 parses to it.
    const infinite = readState(JSON.parse('{"clipNorm":1e999,"lossScale":-1e999,"zLoss":1e999,"adamRate":1e999}'));
    expect(infinite.clipNorm).toBe(defaultState.clipNorm);
    expect(infinite.lossScale).toBe(defaultState.lossScale);
    expect(infinite.zLoss).toBe(defaultState.zLoss);
    expect(infinite.adamRate).toBe(defaultState.adamRate);
    expect(readState({ clipNorm: Number.NaN }).clipNorm).toBe(defaultState.clipNorm);
  });

  it("snaps the sliders to values a control can produce", () => {
    expect(readState({ clipNorm: 0.52 }).clipNorm).toBe(0.5);
    expect(readState({ clipNorm: 0.15 }).clipNorm).toBe(0.15);
    expect(readState({ clipNorm: 0.149999 }).clipNorm).toBe(0.15);
    expect(readState({ lossScale: 19.6 }).lossScale).toBe(20);
    expect(readState({ zLoss: 0.008 }).zLoss).toBe(0.01);
    expect(readState({ zLoss: 0.0003 }).zLoss).toBe(0.0001);
    expect(readState({ zLoss: 0.000001 }).zLoss).toBe(0);
    expect(readState({ adamRate: 0.12 }).adamRate).toBe(0.1);
    expect(readState({ adamRate: 0.5 }).adamRate).toBe(0.3);
  });

  it("keeps every rung of both ladders exactly", () => {
    for (const rung of Z_LOSSES) expect(nearestRung(rung, Z_LOSSES, -1)).toBe(rung);
    for (const rung of ADAM_RATES) expect(nearestRung(rung, ADAM_RATES, -1)).toBe(rung);
    expect(nearestRung("0.1", ADAM_RATES, 7)).toBe(7);
  });

  it("stores only JSON-safe values", () => {
    const state = hydrateStabilityState(JSON.stringify({ fault: "bad-batch", rollback: true, clipNorm: 1 }));
    expect(JSON.parse(JSON.stringify(state))).toEqual(state);
    expect(state.rollback).toBe(true);
  });
});
