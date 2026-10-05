import { describe, expect, it } from "vitest";
import definition from "./module";

describe("model-merging state", () => {
  it("migrates a version 1 snapshot, whose task vectors were LoRA deltas", () => {
    const hydrated = definition.hydrateState(JSON.stringify({ mix: 0.3, epochsA: 30, epochsB: 12 }));
    expect(hydrated).toMatchObject({ source: "lora", mix: 0.3, epochsA: 30, epochsB: 12, method: "linear" });
  });

  it("clamps out-of-range values and rejects unknown methods", () => {
    const hydrated = definition.hydrateState(
      JSON.stringify({ source: "full", method: "average-everything", dropRate: 3, density: 0, epochsA: 99 }),
    );
    expect(hydrated).toMatchObject({ source: "full", method: "linear", dropRate: 0.9, density: 0.05, epochsA: 40 });
  });
});
