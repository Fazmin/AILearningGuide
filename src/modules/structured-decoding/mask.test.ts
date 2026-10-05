import { describe, expect, it } from "vitest";
import definition from "./module";
import {
  distortion,
  distributionAt,
  EOS,
  expectation,
  GRAMMARS,
  isComplete,
  isLegal,
  matchPrefix,
  softmax,
  VOCAB,
} from "./mask";

const json = GRAMMARS.json;
const call = GRAMMARS.call;

describe("character-level grammar matcher", () => {
  it("accepts exactly the complete strings", () => {
    expect(isComplete(json, '{"ok":true}')).toBe(true);
    expect(isComplete(json, '{"ok":false}')).toBe(true);
    expect(isComplete(json, '{"ok":null}')).toBe(false);
    expect(isComplete(json, '{"ok":true')).toBe(false);
    expect(isComplete(call, 'get_weather(city="Oslo")')).toBe(true);
    expect(isComplete(call, 'get_weather(city="")')).toBe(false);
    expect(matchPrefix(json, '{"ok":tru').length).toBeGreaterThan(0);
    expect(matchPrefix(json, '{"ok":x').length).toBe(0);
  });

  it("lets a token cross a symbol boundary when every character fits", () => {
    expect(isLegal(json, "", '{"')).toBe(true);
    expect(isLegal(json, '{"ok', '":')).toBe(true);
    expect(isLegal(json, '{"ok', ":")).toBe(false);
    expect(isLegal(json, '{"ok', '"}')).toBe(false);
    expect(isLegal(call, "get_weather", "(city")).toBe(true);
  });

  it("enforces the schema, not just JSON syntax", () => {
    const prefix = '{"ok":';
    expect(isLegal(json, prefix, "true")).toBe(true);
    expect(isLegal(json, prefix, "false")).toBe(true);
    expect(isLegal(json, prefix, "null")).toBe(false);
    expect(isLegal(json, prefix, '"')).toBe(false);
  });

  it("allows end of sequence only once the string is complete", () => {
    expect(isLegal(json, '{"ok":true', EOS)).toBe(false);
    expect(isLegal(json, '{"ok":true}', EOS)).toBe(true);
    expect(isLegal(json, '{"ok":true}', "The")).toBe(false);
  });

  it("lets any letters through a free string field", () => {
    const prefix = 'get_weather(city="';
    for (const token of ["Os", "hello", "Sure", "true"] as const) expect(isLegal(call, prefix, token)).toBe(true);
    expect(isLegal(call, prefix, '")')).toBe(false);
    expect(isLegal(call, `${prefix}Os`, '")')).toBe(true);
  });

  it("describes what comes next", () => {
    expect(expectation(json, '{"ok":')).toBe("true or false");
    expect(expectation(json, '{"ok":true}')).toBe("end of output");
    expect(expectation(call, 'get_weather(city="Os')).toBe('more letters or ")');
  });
});

describe("masking and renormalization", () => {
  it("computes a softmax that sums to one", () => {
    const p = softmax([3, 2, -2]);
    expect(p.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 12);
    expect(p[0] / p[1]).toBeCloseTo(Math.E, 10);
  });

  it("renormalizes legal mass and keeps the model's ratios", () => {
    for (const grammar of [json, call]) {
      grammar.walk.forEach((token, step) => {
        const { rows, z, greedy } = distributionAt(grammar, step);
        expect(rows).toHaveLength(VOCAB.length);
        expect(rows.reduce((sum, row) => sum + row.p, 0)).toBeCloseTo(1, 12);
        expect(rows.reduce((sum, row) => sum + row.masked, 0)).toBeCloseTo(1, 12);
        rows.forEach((row) => {
          if (!row.legal) expect(row.masked).toBe(0);
          else expect(row.masked).toBeCloseTo(row.p / z, 12);
        });
        // The authored walk is exactly the greedy constrained path.
        expect(greedy.token).toBe(token);
      });
    }
  });

  it("masks the model's favourite token at the first step", () => {
    const first = distributionAt(json, 0);
    expect(first.rawTop.token).toBe("Sure");
    expect(first.rawTop.legal).toBe(false);
    expect(first.z).toBeCloseTo(0.32, 2);
    expect(first.greedy.masked).toBeCloseTo(0.711, 3);
  });
});

describe("per-step masking versus conditioning", () => {
  it("matches only when both branches keep the same legal share", () => {
    const low = distortion(0.1);
    expect(low.stepwise.oslo).toBeCloseTo(0.55 / 0.85, 12);
    expect(low.conditioned.oslo).toBeCloseTo(0.055 / 0.325, 12);
    const equal = distortion(0.9);
    expect(equal.conditioned.oslo).toBeCloseTo(equal.stepwise.oslo, 12);
  });
});

describe("checkpoint question 5: dragging P(slo\" after \"O)", () => {
  it("leaves the per-step bar fixed while the conditioned bar rises until the two meet", () => {
    const sliderSteps = [0.1, 0.3, 0.5, 0.7, 0.9];
    const results = sliderSteps.map((value) => distortion(value));
    results.forEach((result) => expect(result.stepwise.oslo).toBe(results[0].stepwise.oslo));
    for (let index = 1; index < results.length; index += 1) {
      expect(results[index].conditioned.oslo).toBeGreaterThan(results[index - 1].conditioned.oslo);
    }
    expect(results[0].conditioned.oslo).toBeLessThan(results[0].stepwise.oslo);
    expect(results[4].conditioned.oslo).toBeCloseTo(results[4].stepwise.oslo, 12);
  });
});

describe("structured-decoding state", () => {
  it("clamps the step to the chosen grammar and fills the new key", () => {
    expect(definition.hydrateState(JSON.stringify({ grammar: "json", step: 9 }))).toMatchObject({
      grammar: "json",
      step: 5,
      detour: 0.1,
    });
    expect(definition.hydrateState(JSON.stringify({ grammar: "call", step: 6 }))).toMatchObject({ step: 6 });
    expect(definition.hydrateState(JSON.stringify({ grammar: "xml", detour: 7 }))).toMatchObject({
      grammar: "json",
      detour: 1,
    });
  });
});
