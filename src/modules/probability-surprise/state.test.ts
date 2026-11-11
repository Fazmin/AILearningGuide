import { describe, expect, it } from "vitest";
import definition from "./module";
import { MAX_DRAWS, initialState, parseBars, parseScores, readState, storeState } from "./state";
import { EQUAL_BARS, SKY_BARS, SKY_SCORES } from "./chance";
import { STEPS } from "./steps";

const { hydrateState, serializeState } = definition;

describe("probability-surprise state", () => {
  it("round-trips the defaults", () => {
    expect(hydrateState(serializeState(initialState))).toEqual(initialState);
    expect(definition.initialState).toEqual(initialState);
    expect(storeState(readState(initialState))).toEqual(initialState);
  });

  it("starts on the sky bars, no draws, a know-nothing model and unanswered predictions", () => {
    expect(initialState).toMatchObject({
      view: "percent",
      bars: "50,25,12,8,5",
      draws: 0,
      testSize: 25,
      prompt: 0,
      prior: 50,
      shape: 50,
      guess: "20,20,20,20,20",
      scores: "2,1.3,0.6,0.2,-0.3",
      temperature: 1,
      answer: "20,20,20,20,20",
      scored: 0,
      predictDraws: -1,
      predictPrior: -1,
      predictShape: -1,
      predictBill: -1,
    });
  });

  it("round-trips a state the reader could reach", () => {
    const reached = {
      ...initialState,
      view: "counts",
      bars: "80,10,5,3,2",
      draws: 137,
      luck: 21,
      follow: 3,
      predictDraws: 1,
      testSize: 400,
      prompt: 2,
      clue: 4,
      prior: 10,
      predictPrior: 0,
      word: 2,
      first: 4,
      second: 4,
      shape: 100,
      predictShape: 2,
      guess: "1,1,1,1,96",
      predictBill: 2,
      scores: "-6,6,0,1.5,-2.2",
      temperature: 0.2,
      round: 2,
      answer: "8,10,35,5,42",
      scored: 1,
      tries: 3,
      revealed: 1,
    };
    expect(hydrateState(serializeState(reached))).toEqual(reached);
  });

  it("returns the defaults for input that is not a state at all", () => {
    for (const hostile of ["", "not json", "null", "42", "[1,2,3]", '"bars"', "{", "undefined"]) {
      expect(hydrateState(hostile)).toEqual(initialState);
    }
    expect(hydrateState("{}")).toEqual(initialState);
  });

  it("replaces every malformed field with its default and keeps the valid ones", () => {
    const hydrated = hydrateState(
      JSON.stringify({
        view: "pie",
        bars: "50,25,12,8,6",
        draws: -4,
        luck: 0,
        follow: 9,
        predictDraws: 7,
        testSize: 26,
        prompt: "sky",
        clue: 2.5,
        prior: 200,
        predictPrior: null,
        word: -1,
        first: [1],
        second: {},
        shape: 150,
        predictShape: "2",
        guess: "100,0,0,0,0",
        predictBill: 1,
        scores: "1,2,3",
        temperature: "hot",
        round: 3,
        answer: "20,20,20,20",
        scored: 2,
        tries: -1,
        revealed: true,
        extra: "ignored",
      }),
    );
    expect(hydrated).toEqual({ ...initialState, predictBill: 1 });
    expect(hydrated).not.toHaveProperty("extra");
  });

  it("is not polluted by a payload that names __proto__ or constructor", () => {
    const hydrated = hydrateState('{"__proto__":{"polluted":true},"constructor":{"prototype":{"polluted":true}},"bars":"80,10,5,3,2"}');
    expect(hydrated).toEqual({ ...initialState, bars: "80,10,5,3,2" });
    expect(({} as Record<string, unknown>).polluted).toBeUndefined();
    expect(Object.getPrototypeOf(hydrated)).toBe(Object.prototype);
  });

  it("clamps numbers that are only out of range", () => {
    const hydrated = hydrateState(JSON.stringify({ temperature: 99, scores: "50,-50,0.04,1.26,2" }));
    expect(hydrated.temperature).toBe(4);
    expect(hydrated.scores).toBe("6,-6,0,1.3,2");
    expect(hydrateState(JSON.stringify({ temperature: 0 })).temperature).toBe(0.2);
    expect(hydrateState(JSON.stringify({ draws: MAX_DRAWS })).draws).toBe(MAX_DRAWS);
    expect(hydrateState(JSON.stringify({ draws: MAX_DRAWS + 1 })).draws).toBe(0);
  });

  it("rejects bar lists that are hostile rather than merely wrong", () => {
    for (const hostile of ["50,25,12,8,5,", ",50,25,12,8,5", "50;25;12;8;5", "1e2,0,0,0,0", "50,25,12,8,5\n", "NaN,25,12,8,5", "Infinity,1,1,1,1", " 50,25,12,8,5", "50,25,12,8,5".repeat(2000)]) {
      expect(parseBars(hostile, SKY_BARS)).toEqual([...SKY_BARS]);
    }
    expect(parseBars(["50", "25"], EQUAL_BARS)).toEqual([...EQUAL_BARS]);
    expect(parseScores("1,2,3,4,x")).toEqual([...SKY_SCORES]);
    expect(parseScores(undefined)).toEqual([...SKY_SCORES]);
  });

  it("ignores a version it does not know and still reads the fields it does", () => {
    expect(definition.stateVersion).toBe(1);
    const fromLater = hydrateState(JSON.stringify({ stateVersion: 7, bars: "80,10,5,3,2", futureKey: [1, 2] }));
    expect(fromLater).toEqual({ ...initialState, bars: "80,10,5,3,2" });
  });

  it("names the same steps in the step bar and in the lab", () => {
    expect(definition.steps).toEqual([...STEPS]);
    expect(definition.stepInstructions).toHaveLength(STEPS.length);
  });
});
