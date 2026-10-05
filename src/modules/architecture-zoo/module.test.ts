import { describe, expect, it } from "vitest";
import definition from "./module";

describe("architecture-zoo state", () => {
  it("keeps a version 1 payload's drawing and fills the new controls from defaults", () => {
    const pixels = "1".repeat(64);
    const v1 = JSON.stringify({ pixels, filter: "horiz", rnnStep: 5, cell: 12 });
    expect(definition.hydrateState(v1)).toEqual({
      ...definition.initialState,
      pixels,
      filter: "horiz",
      rnnStep: 5,
      cell: 12,
    });
  });

  it("keeps experts per token at or below the expert count", () => {
    const state = definition.hydrateState(JSON.stringify({ experts: 3, expertsPerToken: 4 }));
    expect(state.experts).toBe(3);
    expect(state.expertsPerToken).toBe(3);
  });
});
