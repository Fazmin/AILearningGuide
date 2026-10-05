import { describe, expect, it } from "vitest";
import definition from "./module";
import { MAX_LINE_LENGTH, MAX_PAIRS } from "./dpo";

describe("preference-optimization state", () => {
  it("reads a version 1 payload, which had no reference, and uses the base model", () => {
    const version1 = JSON.stringify({ beta: 0.6, steps: 20, learningRate: 0.5, pairsText: "a | b. | c." });
    expect(definition.stateVersion).toBe(2);
    expect(definition.hydrateState(version1)).toEqual({
      beta: 0.6,
      steps: 20,
      learningRate: 0.5,
      reference: "base",
      pairsText: "a | b. | c.",
    });
  });

  it("clamps every numeric key and rejects an unknown reference or a wrong type", () => {
    const wild = definition.hydrateState(
      JSON.stringify({ beta: 1e9, steps: -1e9, learningRate: -5, reference: "everything", pairsText: 7 }),
    );
    expect(wild).toEqual({ ...definition.initialState, beta: 1.5, steps: 0, learningRate: 0.05 });
    expect(definition.hydrateState(JSON.stringify({ steps: 12.6 })).steps).toBe(13);
    expect(definition.hydrateState(JSON.stringify({ reference: "sft" })).reference).toBe("sft");
    expect(definition.hydrateState(JSON.stringify({ beta: "high" })).beta).toBe(0.4);
  });

  it("cuts a huge pairs text to the editor's limits", () => {
    const huge = Array.from({ length: 3000 }, () => `${"p".repeat(300)} | ${"c".repeat(300)} | ${"r".repeat(300)}`).join("\n");
    const text = definition.hydrateState(JSON.stringify({ pairsText: huge })).pairsText as string;
    expect(text.split("\n")).toHaveLength(MAX_PAIRS);
    expect(Math.max(...text.split("\n").map((line) => line.length))).toBeLessThanOrEqual(MAX_LINE_LENGTH);
  });
});
