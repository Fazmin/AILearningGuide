import { describe, expect, it } from "vitest";
import definition, { hydrateBackpropState } from "./module";

describe("backpropagation state", () => {
  it("reads a version 2 payload (the one-weight graph only) and adds the gradient-flow defaults", () => {
    const version2 = JSON.stringify({ x: 1.1, weight: -1.4, target: 0.25, phase: 4, learningRate: 1.3 });
    expect(hydrateBackpropState(version2)).toEqual({
      x: 1.1,
      weight: -1.4,
      target: 0.25,
      phase: 4,
      learningRate: 1.3,
      flowDepth: 6,
      flowActivation: "sigmoid",
      flowInit: "glorot",
      flowSkip: "off",
    });
    expect(definition.stateVersion).toBe(3);
  });

  it("keeps valid gradient-flow choices and rejects unknown ones", () => {
    const hydrated = hydrateBackpropState(
      JSON.stringify({ flowDepth: 12, flowActivation: "relu", flowInit: "he", flowSkip: "on" }),
    );
    expect(hydrated).toMatchObject({ flowDepth: 12, flowActivation: "relu", flowInit: "he", flowSkip: "on" });
    const rejected = hydrateBackpropState(
      JSON.stringify({ flowDepth: "deep", flowActivation: "swish", flowInit: "zero", flowSkip: true }),
    );
    expect(rejected).toMatchObject({ flowDepth: 6, flowActivation: "sigmoid", flowInit: "glorot", flowSkip: "off" });
  });

  it("clamps every numeric key into its slider range", () => {
    for (const value of [1e9, -1e9]) {
      const hydrated = hydrateBackpropState(
        JSON.stringify({ x: value, weight: value, target: value, phase: value, learningRate: value, flowDepth: value }),
      );
      expect(hydrated.x).toBeGreaterThanOrEqual(0.1);
      expect(hydrated.x).toBeLessThanOrEqual(1.5);
      expect(Math.abs(hydrated.weight as number)).toBe(3);
      expect(hydrated.target).toBeGreaterThanOrEqual(0);
      expect(hydrated.target).toBeLessThanOrEqual(1);
      expect(hydrated.phase).toBeGreaterThanOrEqual(0);
      expect(hydrated.phase).toBeLessThanOrEqual(5);
      expect(hydrated.learningRate).toBeGreaterThanOrEqual(0.05);
      expect(hydrated.learningRate).toBeLessThanOrEqual(2);
      expect(hydrated.flowDepth).toBeGreaterThanOrEqual(1);
      expect(hydrated.flowDepth).toBeLessThanOrEqual(12);
    }
  });

  it("falls back to the defaults on malformed or non-object JSON", () => {
    expect(hydrateBackpropState("{")).toEqual(definition.initialState);
    expect(hydrateBackpropState("[1,2]")).toEqual(definition.initialState);
    expect(hydrateBackpropState("null")).toEqual(definition.initialState);
  });
});
