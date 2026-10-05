import { describe, expect, it } from "vitest";
import definition from "./module";

describe("why-gpus state", () => {
  it("translates a version 1 square-matmul payload into batch and width", () => {
    const v1 = JSON.stringify({ matrixSize: 1024, precision: "int8", bandwidth: 900 });
    expect(definition.hydrateState(v1)).toEqual({
      operation: "matmul",
      batch: 1024,
      layerWidth: 1024,
      precision: "int8",
      bandwidth: 900,
    });
  });

  it("clamps a version 1 size that no longer fits the batch range", () => {
    const v1 = JSON.stringify({ matrixSize: 16384, precision: "fp32", bandwidth: 3000 });
    const state = definition.hydrateState(v1);
    expect(state.batch).toBe(4096);
    expect(state.layerWidth).toBe(16384);
  });

  it("snaps batch and width to powers of two", () => {
    const state = definition.hydrateState(JSON.stringify({ batch: 300, layerWidth: 5000 }));
    expect(state.batch).toBe(256);
    expect(state.layerWidth).toBe(4096);
  });
});
