import { describe, expect, it } from "vitest";
import definition from "./module";
import { MAX_WALK_STEPS, NUDGES, RACE_BUDGET } from "./curves";
import { GUESS_OPTIONS, LIMITS, hydrateState, initialState, readState } from "./state";

describe("slopes-gradients state", () => {
  it("is at version 1 and round-trips its defaults", () => {
    expect(definition.stateVersion).toBe(1);
    expect(definition.hydrateState(definition.serializeState(initialState))).toEqual(initialState);
    expect(readState(initialState)).toEqual(initialState);
  });

  it("opens every card where the lesson says it does", () => {
    expect(initialState).toMatchObject({
      nudgeW: 4,
      nudge: 1,
      slopeCurve: "valleys",
      slopeW: 1,
      surface: "bowl",
      mapW: 4.5,
      mapB: 4,
      legs: "w",
      chainW: 3,
      walkCurve: "valleys",
      walkStart: 2.6,
      walkStep: 0.2,
      walkSteps: 0,
      raceStep: 0.1,
      raceSteps: 0,
      racePrev: 0,
    });
  });

  it("starts with every predict-first question unanswered", () => {
    for (const key of Object.keys(GUESS_OPTIONS)) expect(initialState[key]).toBe(-1);
  });

  it("falls back to the defaults for input that is not an object of settings", () => {
    for (const value of ["", "not json at all", "⛔", "null", "[]", "42", '"text"', "{}"]) {
      expect(hydrateState(value), value).toEqual(initialState);
    }
  });

  it("merges defaults into a partial payload and drops keys it does not know", () => {
    expect(hydrateState(JSON.stringify({ chainW: 5, surface: "saddle", extra: "ignored", topic: "derivatives" }))).toEqual({
      ...initialState,
      chainW: 5,
      surface: "saddle",
    });
  });

  it("replaces values of the wrong type or outside a choice list", () => {
    const hydrated = hydrateState(
      JSON.stringify({
        nudgeW: "far",
        nudge: null,
        slopeCurve: "bowl",
        surface: 3,
        legs: "all",
        walkCurve: "shelf",
        walkSteps: "many",
        chainW: [1, 2],
        mapW: { x: 1 },
        raceStep: true,
      }),
    );
    expect(hydrated).toEqual(initialState);
  });

  it("clamps every number into its control's range under hostile values", () => {
    for (const value of [1e9, -1e9, 1e308, -1e308]) {
      const hostile = Object.fromEntries(
        Object.entries(initialState).map(([key, original]) => [key, typeof original === "number" ? value : original]),
      );
      const state = readState(hostile);
      expect(state.nudgeW).toBeGreaterThanOrEqual(LIMITS.nudgeW[0]);
      expect(state.nudgeW).toBeLessThanOrEqual(LIMITS.nudgeW[1]);
      for (const key of ["slopeW", "mapW", "mapB", "chainW", "walkStart"] as const) {
        expect(state[key]).toBeGreaterThanOrEqual(LIMITS.dial[0]);
        expect(state[key]).toBeLessThanOrEqual(LIMITS.dial[1]);
      }
      expect(NUDGES).toContain(state.nudge);
      expect(state.walkStep).toBeGreaterThanOrEqual(LIMITS.walkStep[0]);
      expect(state.walkStep).toBeLessThanOrEqual(LIMITS.walkStep[1]);
      expect(state.raceStep).toBeGreaterThanOrEqual(LIMITS.raceStep[0]);
      expect(state.raceStep).toBeLessThanOrEqual(LIMITS.raceStep[1]);
      expect(state.walkSteps).toBeGreaterThanOrEqual(0);
      expect(state.walkSteps).toBeLessThanOrEqual(MAX_WALK_STEPS);
      expect(state.raceSteps).toBeGreaterThanOrEqual(0);
      expect(state.raceSteps).toBeLessThanOrEqual(RACE_BUDGET);
      for (const key of Object.keys(GUESS_OPTIONS) as (keyof typeof GUESS_OPTIONS)[]) expect(state[key]).toBe(-1);
      expect(state.racePrev === 0 || state.racePrev <= LIMITS.raceStep[1]).toBe(true);
    }
  });

  it("treats infinities and NaN as missing", () => {
    const state = readState({ nudgeW: Infinity, slopeW: -Infinity, walkStep: NaN, walkSteps: Infinity, racePrev: NaN, nudge: Infinity });
    expect(state).toEqual(initialState);
    // "1e999" parses to Infinity, which is how a hand-edited link can carry one.
    expect(hydrateState('{"chainW":1e999,"raceSteps":-1e999}')).toEqual(initialState);
  });

  it("snaps the nudge to the nearest size the slider offers", () => {
    expect(readState({ nudge: 0.3 }).nudge).toBe(0.25);
    expect(readState({ nudge: 0.7 }).nudge).toBe(0.5);
    expect(readState({ nudge: 0 }).nudge).toBe(0.01);
    expect(readState({ nudge: 40 }).nudge).toBe(1);
  });

  it("keeps a valid guess and discards one that names no option", () => {
    expect(readState({ nudgeGuess: 2, arrowGuess: 0, chainGuess: 3, walkGuess: 1 })).toMatchObject({
      nudgeGuess: 2,
      arrowGuess: 0,
      chainGuess: 3,
      walkGuess: 1,
    });
    expect(readState({ nudgeGuess: 3, arrowGuess: -2, chainGuess: 1.5, walkGuess: "0" })).toMatchObject({
      nudgeGuess: -1,
      arrowGuess: -1,
      chainGuess: -1,
      walkGuess: -1,
    });
  });

  it("rounds step counts to whole steps and settings to the two decimals the controls show", () => {
    expect(readState({ walkSteps: 3.6, raceSteps: 2.2 })).toMatchObject({ walkSteps: 4, raceSteps: 2 });
    expect(readState({ slopeW: 1.23456, walkStep: 0.333 })).toMatchObject({ slopeW: 1.23, walkStep: 0.33 });
  });

  it("keeps a previous challenge attempt only when it is a real step size", () => {
    expect(readState({ racePrev: 0.6 }).racePrev).toBe(0.6);
    expect(readState({ racePrev: 0.01 }).racePrev).toBe(0);
    expect(readState({ racePrev: -3 }).racePrev).toBe(0);
    expect(readState({ racePrev: 9 }).racePrev).toBe(LIMITS.raceStep[1]);
  });

  it("reads zeroed and emptied state without leaving any control out of range", () => {
    const zeroed = Object.fromEntries(
      Object.entries(initialState).map(([key, value]) => [key, typeof value === "number" ? 0 : ""]),
    );
    const state = readState(zeroed);
    expect(state).toMatchObject({ slopeCurve: "valleys", surface: "bowl", legs: "w", walkCurve: "valleys", nudge: 0.01, walkStep: 0.05, raceStep: 0.05 });
    expect(readState(null)).toEqual(initialState);
    expect(readState(undefined)).toEqual(initialState);
  });

  it("accepts every choice each picker offers", () => {
    for (const slopeCurve of ["valleys", "shelf"]) expect(readState({ slopeCurve }).slopeCurve).toBe(slopeCurve);
    for (const walkCurve of ["valleys", "bowl"]) expect(readState({ walkCurve }).walkCurve).toBe(walkCurve);
    for (const surface of ["bowl", "valley", "saddle"]) expect(readState({ surface }).surface).toBe(surface);
    for (const legs of ["w", "b", "both"]) expect(readState({ legs }).legs).toBe(legs);
  });
});
