import { describe, expect, it } from "vitest";
import definition from "./module";
import { AUTO_MAX, HAND_LIMIT, HAND_START, INPUT_STOPS, MAX_LAYERS, MAX_UNITS, OUTPUT_STOPS, RATE_STOPS, TRAIN_STEPS, WIDTH_STOPS } from "./model";
import { GUESS_OPTIONS, MAX_SEED, PLAY_MAX_WIDTH, handPatch, hydrateState, initialState, readState } from "./state";

describe("lab state", () => {
  it("round-trips the opening state and merges defaults into anything partial", () => {
    expect(hydrateState(JSON.stringify(initialState))).toEqual(initialState);
    expect(hydrateState("{}")).toEqual(initialState);
    expect(hydrateState("not json")).toEqual(initialState);
    expect(hydrateState("[1,2]")).toEqual(initialState);
    expect(hydrateState("null")).toEqual(initialState);
    expect(definition.hydrateState(definition.serializeState(definition.initialState))).toEqual(definition.initialState);
    expect(hydrateState('{"units":4}')).toEqual({ ...initialState, units: 4 });
  });

  it("opens the seven knobs on the first scramble", () => {
    expect(readState(initialState).hand).toEqual(HAND_START);
    expect(HAND_START).toEqual([-1.5, 0.1, 1.4, 1.4, -0.7, 0.3, 0.7]);
  });

  it("clamps every number into its control's range under hostile values", () => {
    for (const value of [1e9, -1e9, 1e308, -1e308]) {
      const hostile = Object.fromEntries(
        Object.entries(initialState).map(([key, original]) => [key, typeof original === "number" ? value : original]),
      );
      const state = readState(hostile);
      for (const knob of [state.lineW, state.lineB, ...state.hand]) expect(Math.abs(knob)).toBeLessThanOrEqual(HAND_LIMIT);
      expect(Math.abs(state.lineX)).toBeLessThanOrEqual(2);
      expect(state.pick).toBeGreaterThanOrEqual(0);
      expect(state.pick).toBeLessThanOrEqual(6);
      expect(state.auto).toBeGreaterThanOrEqual(0);
      expect(state.auto).toBeLessThanOrEqual(AUTO_MAX);
      expect(state.units).toBeGreaterThanOrEqual(1);
      expect(state.units).toBeLessThanOrEqual(MAX_UNITS);
      for (const seed of [state.scramble, state.unitsSeed, state.pgSeed]) {
        expect(seed).toBeGreaterThanOrEqual(1);
        expect(seed).toBeLessThanOrEqual(MAX_SEED);
      }
      expect(INPUT_STOPS).toContain(state.inputs);
      expect(WIDTH_STOPS).toContain(state.width);
      expect(OUTPUT_STOPS).toContain(state.outputs);
      expect(RATE_STOPS).toContain(state.pgRate);
      expect(state.layers).toBeGreaterThanOrEqual(1);
      expect(state.layers).toBeLessThanOrEqual(MAX_LAYERS);
      expect([1, 2]).toContain(state.pgLayers);
      expect(state.pgWidth).toBeGreaterThanOrEqual(1);
      expect(state.pgWidth).toBeLessThanOrEqual(PLAY_MAX_WIDTH);
      expect(state.pgSteps).toBeGreaterThanOrEqual(0);
      expect(state.pgSteps).toBeLessThanOrEqual(TRAIN_STEPS);
      for (const key of Object.keys(GUESS_OPTIONS) as (keyof typeof GUESS_OPTIONS)[]) expect(state[key]).toBe(-1);
    }
  });

  it("treats infinities and NaN as missing", () => {
    expect(readState({ lineW: Infinity, k3: -Infinity, auto: NaN, width: Infinity, pgRate: NaN, pgSteps: Infinity })).toEqual(
      readState(initialState),
    );
    // "1e999" parses to Infinity, which is how a hand-edited link can carry one.
    expect(hydrateState('{"k0":1e999,"pgSteps":-1e999,"inputs":1e999}')).toEqual(initialState);
  });

  it("snaps sizes and step sizes to the nearest stop a slider offers", () => {
    expect(readState({ inputs: 500 }).inputs).toBe(784);
    expect(readState({ inputs: 5 }).inputs).toBe(2);
    expect(readState({ width: 20 }).width).toBe(16);
    expect(readState({ width: 0 }).width).toBe(2);
    expect(readState({ outputs: 7 }).outputs).toBe(10);
    expect(readState({ pgRate: 0.07 }).pgRate).toBe(0.05);
    expect(readState({ pgRate: 9 }).pgRate).toBe(0.5);
  });

  it("keeps a valid guess and discards one that names no option", () => {
    expect(readState({ lineGuess: 0, comboGuess: 2, sameGuess: 1, shapeGuess: 2, wiresGuess: 1 })).toMatchObject({
      lineGuess: 0,
      comboGuess: 2,
      sameGuess: 1,
      shapeGuess: 2,
      wiresGuess: 1,
    });
    expect(readState({ lineGuess: 3, comboGuess: -2, sameGuess: 1.5, shapeGuess: "0", wiresGuess: null })).toMatchObject({
      lineGuess: -1,
      comboGuess: -1,
      sameGuess: -1,
      shapeGuess: -1,
      wiresGuess: -1,
    });
  });

  it("reads zeroed and emptied state without leaving any control out of range", () => {
    const zeroed = Object.fromEntries(Object.entries(initialState).map(([key, value]) => [key, typeof value === "number" ? 0 : ""]));
    expect(readState(zeroed)).toMatchObject({
      units: 1,
      scramble: 1,
      unitsSeed: 1,
      pgSeed: 1,
      inputs: 1,
      layers: 1,
      width: 2,
      outputs: 1,
      pgDots: "wave",
      pgLayers: 1,
      pgWidth: 1,
      pgRate: 0.01,
      pgMomentum: true,
    });
    expect(readState(null)).toEqual(readState(initialState));
    expect(readState(undefined)).toEqual(readState(initialState));
  });

  it("accepts every choice each picker offers", () => {
    for (const pgDots of ["slope", "hill", "wave", "step"]) expect(readState({ pgDots }).pgDots).toBe(pgDots);
    expect(readState({ pgMomentum: "off" }).pgMomentum).toBe(false);
    expect(readState({ pgMomentum: "on" }).pgMomentum).toBe(true);
    expect(readState({ pgMomentum: "sideways" }).pgMomentum).toBe(true);
  });

  it("writes a hand setting back as seven clamped knobs and a fresh count of steps", () => {
    const patch = handPatch([4, -4, 0.12345, -0, 1, 2, 3]);
    expect(patch).toEqual({ auto: 0, k0: 3, k1: -3, k2: 0.123, k3: 0, k4: 1, k5: 2, k6: 3 });
    expect(Object.is(patch.k3, -0)).toBe(false);
    // A setting written by the machine's steps survives the store unchanged.
    const stored = hydrateState(JSON.stringify({ ...initialState, ...patch }));
    expect(hydrateState(JSON.stringify(stored))).toEqual(stored);
  });
});
