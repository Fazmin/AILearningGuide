import { describe, expect, it } from "vitest";
import definition, { hydrateTransformerBlockState } from "./module";

describe("transformer-block state", () => {
  it("migrates a version-1 payload: keeps the toggles, drops the decorative slider", () => {
    const v1 = JSON.stringify({ attention: false, mlp: true, norm: false, strength: 58 });
    const state = hydrateTransformerBlockState(v1);
    expect(state).toMatchObject({ attention: false, mlp: true, norm: false, inputScale: 1, placement: "pre" });
    expect(state).not.toHaveProperty("strength");
  });

  it("clamps out-of-range values and rejects malformed JSON", () => {
    expect(hydrateTransformerBlockState(JSON.stringify({ inputScale: 99, position: -3, placement: "sideways" })))
      .toMatchObject({ inputScale: 8, position: 0, placement: "pre" });
    expect(hydrateTransformerBlockState("not json")).toEqual(definition.initialState);
  });

  it("round-trips its own serialized state", () => {
    const state = { ...definition.initialState, mlp: false, placement: "post", inputScale: 4 };
    expect(hydrateTransformerBlockState(definition.serializeState(state))).toEqual(state);
  });

  it("migrates a version-2 payload: keeps every control and takes the Position card's defaults", () => {
    const v2 = JSON.stringify({
      attention: true,
      mlp: false,
      norm: true,
      placement: "post",
      inputScale: 2,
      position: 3,
      paramPreset: "llama2",
      paramWidth: 4096,
      paramLayers: 32,
      paramMlp: "swiglu",
      paramNorm: "rmsnorm",
    });
    expect(hydrateTransformerBlockState(v2)).toEqual({
      attention: true,
      mlp: false,
      norm: true,
      placement: "post",
      inputScale: 2,
      position: 3,
      paramPreset: "llama2",
      paramWidth: 4096,
      paramLayers: 32,
      paramMlp: "swiglu",
      paramNorm: "rmsnorm",
      order: "swap",
      positionInfo: true,
    });
  });

  it("validates the Position card's controls", () => {
    expect(hydrateTransformerBlockState(JSON.stringify({ order: "shuffle", positionInfo: false })))
      .toMatchObject({ order: "shuffle", positionInfo: false });
    expect(hydrateTransformerBlockState(JSON.stringify({ order: "sideways", positionInfo: "yes" })))
      .toMatchObject({ order: "swap", positionInfo: true });
    expect(hydrateTransformerBlockState(JSON.stringify({ order: 3, positionInfo: null })))
      .toMatchObject({ order: "swap", positionInfo: true });
  });
});
