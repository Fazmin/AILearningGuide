import { describe, expect, it } from "vitest";
import definition, { hydrateOptimizersState } from "./module";
import { DECAY_LADDER, rateLadder } from "./race";
import { defaultState, initialState, readState } from "./state";

describe("optimizers state", () => {
  it("round-trips the defaults and survives garbage", () => {
    expect(hydrateOptimizersState(JSON.stringify(initialState))).toEqual(initialState);
    expect(hydrateOptimizersState("{}")).toEqual(initialState);
    expect(hydrateOptimizersState("not json")).toEqual(initialState);
    expect(hydrateOptimizersState("[1]")).toEqual(initialState);
    expect(hydrateOptimizersState("null")).toEqual(initialState);
    expect(definition.stateVersion).toBe(1);
    expect(definition.initialState).toEqual(defaultState);
  });

  it("drops keys it does not know", () => {
    const state = readState({ ...defaultState, schedule: "cosine", peakLearningRate: 3 });
    expect(Object.keys(state).sort()).toEqual(Object.keys(defaultState).sort());
  });

  it("snaps every number a hostile link can carry onto a notch", () => {
    const numeric = ["sgdRate", "momentumRate", "adamRate", "muonRate", "warmup", "weightDecay", "nsSteps"] as const;
    const high = readState(Object.fromEntries(numeric.map((key) => [key, 1e9])));
    expect([high.sgdRate, high.momentumRate, high.adamRate, high.muonRate]).toEqual([10, 1, 0.3, 0.5]);
    expect([high.warmup, high.weightDecay, high.nsSteps]).toEqual([0.3, 0.3, 8]);
    const low = readState(Object.fromEntries(numeric.map((key) => [key, -1e9])));
    expect([low.sgdRate, low.momentumRate, low.adamRate, low.muonRate]).toEqual([0.1, 0.01, 0.003, 0.005]);
    expect([low.warmup, low.weightDecay, low.nsSteps]).toEqual([0, 0, 0]);
    // JSON cannot spell Infinity, but 1e999 parses to it; non-finite values fall back to the defaults.
    const infinite = readState(JSON.parse('{"sgdRate":1e999,"adamRate":-1e999,"warmup":1e999,"nsSteps":-1e999}'));
    expect([infinite.sgdRate, infinite.adamRate, infinite.warmup, infinite.nsSteps]).toEqual([1, 0.03, 0, 5]);
    expect(readState({ weightDecay: Number.NaN }).weightDecay).toBe(0);
  });

  it("rounds values between notches to the nearest one", () => {
    expect(readState({ adamRate: 0.031 }).adamRate).toBe(0.03);
    expect(readState({ adamRate: 0.25 }).adamRate).toBe(0.3);
    expect(readState({ sgdRate: 2.6 }).sgdRate).toBe(3);
    expect(readState({ weightDecay: 0.00004 }).weightDecay).toBe(0);
    expect(readState({ weightDecay: 0.00008 }).weightDecay).toBe(0.0001);
    expect(readState({ weightDecay: 0.09 }).weightDecay).toBe(0.1);
    expect(readState({ warmup: 0.12 }).warmup).toBe(0.1);
    expect(readState({ warmup: 0.15000000000000002 }).warmup).toBe(0.15);
    expect(readState({ nsSteps: 4.6 }).nsSteps).toBe(5);
    for (const id of ["sgd", "momentum", "adamw", "muon"] as const) {
      for (const rate of rateLadder(id)) {
        const key = { sgd: "sgdRate", momentum: "momentumRate", adamw: "adamRate", muon: "muonRate" }[id];
        expect(readState({ [key]: rate })[key as "sgdRate"]).toBe(rate);
      }
    }
    for (const decay of DECAY_LADDER) expect(readState({ weightDecay: decay }).weightDecay).toBe(decay);
  });

  it("accepts only the listed choices", () => {
    expect(readState({ optimizer: "adamw" }).optimizer).toBe("adamw");
    expect(readState({ optimizer: "adam" }).optimizer).toBe("sgd");
    expect(readState({ curves: "held" }).curves).toBe("held");
    expect(readState({ curves: "both" }).curves).toBe("train");
    expect(readState({ biasCorrection: "off" }).biasCorrection).toBe("off");
    expect(readState({ biasCorrection: false }).biasCorrection).toBe("on");
    expect(readState({ decayMode: "l2" }).decayMode).toBe("l2");
    expect(readState({ decayMode: "L2" }).decayMode).toBe("decoupled");
    expect(readState({ capture: "first" }).capture).toBe("first");
    expect(readState({ capture: 300 }).capture).toBe("middle");
  });
});
