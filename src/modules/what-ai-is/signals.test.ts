import { describe, expect, it } from "vitest";
import {
  OUTPUT_KINDS,
  SCENARIOS,
  SIGNALS,
  parseSignalPlacements,
  scenarioById,
  scoreSignalPlacements,
} from "./signals";
import definition from "./module";

describe("learning-signal scenarios", () => {
  it("has unique scenario ids and gives every signal at least two scenarios", () => {
    expect(new Set(SCENARIOS.map((item) => item.id)).size).toBe(SCENARIOS.length);
    for (const signal of SIGNALS) {
      expect(SCENARIOS.filter((item) => item.signal === signal.id).length, signal.id).toBeGreaterThanOrEqual(2);
    }
    for (const scenario of SCENARIOS) {
      expect(OUTPUT_KINDS[scenario.output], scenario.id).toBeTruthy();
      expect(scenario.why.length, scenario.id).toBeGreaterThan(40);
    }
  });

  it("keeps the signal and the output type as separate axes", () => {
    // Next-word prediction is self-supervised classification; blanked-photo repair is self-supervised regression.
    expect(scenarioById("text")).toMatchObject({ signal: "self", output: "classification" });
    expect(scenarioById("patch")).toMatchObject({ signal: "self", output: "regression" });
    expect(scenarioById("house")).toMatchObject({ signal: "supervised", output: "regression" });
    expect(scenarioById("spam")).toMatchObject({ signal: "supervised", output: "classification" });
    expect(scenarioById("customers").signal).toBe("unsupervised");
    expect(scenarioById("game").signal).toBe("reward");
    const outputsUnderSelf = new Set(SCENARIOS.filter((item) => item.signal === "self").map((item) => item.output));
    expect(outputsUnderSelf.size).toBe(2);
  });
});

describe("placements", () => {
  it("drops malformed, unknown, and repeated entries and lets a later placement win", () => {
    expect(
      parseSignalPlacements(["house:supervised", "house:self", "nope:self", "spam:nope", "spam", "text:self:extra", 7, null]),
    ).toEqual(["house:self"]);
    expect(parseSignalPlacements("house:supervised")).toEqual([]);
    expect(parseSignalPlacements(undefined)).toEqual([]);
  });

  it("scores placed and matching scenarios", () => {
    expect(scoreSignalPlacements([])).toEqual({ placed: 0, matched: 0, total: 8 });
    expect(scoreSignalPlacements(["house:supervised", "text:supervised", "game:reward"])).toEqual({
      placed: 3,
      matched: 2,
      total: 8,
    });
    expect(scoreSignalPlacements(SCENARIOS.map((item) => `${item.id}:${item.signal}`))).toEqual({
      placed: 8,
      matched: 8,
      total: 8,
    });
  });
});

describe("what-ai-is state", () => {
  const { hydrateState, initialState, serializeState } = definition;

  it("round-trips the defaults, including the learning-signal keys", () => {
    expect(hydrateState(serializeState(initialState))).toEqual(initialState);
    expect(initialState).toMatchObject({ scenario: "house", signal: "supervised", signalPlacements: [] });
  });

  it("fills the new keys when an older (version 2) payload has none", () => {
    const version2 = JSON.stringify({
      ring: "dl",
      era: 4,
      term: "gan",
      destination: "gen",
      placements: ["gan:gen"],
      stage: 2,
      weight: 0.5,
      bias: 1.2,
      distinction: "scope",
    });
    // `era` was an index into seven stops; version 4 stores the stop's year, and index 4 was AlexNet (2012).
    expect(hydrateState(version2)).toEqual({
      ...initialState,
      ring: "dl",
      eraYear: 2012,
      term: "gan",
      destination: "gen",
      placements: ["gan:gen"],
      stage: 2,
      weight: 0.5,
      bias: 1.2,
      distinction: "scope",
    });
  });

  it("validates and clamps every field of a malformed payload", () => {
    const hydrated = hydrateState(
      JSON.stringify({
        ring: "mars",
        era: 99,
        term: 5,
        destination: null,
        placements: ["chess:ai", "chess:nowhere", 3],
        stage: -4,
        weight: "heavy",
        bias: 1e9,
        distinction: {},
        scenario: "unicorn",
        signal: ["self"],
        signalPlacements: ["text:self", "text:reward", "robot:invented"],
        extra: "ignored",
      }),
    );
    expect(hydrated).toEqual({
      ...initialState,
      eraYear: 2022,
      placements: ["chess:ai"],
      stage: 0,
      bias: 2.4,
      signalPlacements: ["text:reward"],
    });
  });
});
