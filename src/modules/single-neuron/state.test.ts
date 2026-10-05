import { describe, expect, it } from "vitest";
import definition, { hydrateNeuronState } from "./module";

describe("single-neuron state", () => {
  it("hydrates a version 1 payload (no task key) and adds the default task", () => {
    const legacy = JSON.stringify({ x1: 0.2, x2: 0.9, w1: -1.5, w2: 0.4, bias: -0.1, activation: "relu" });
    expect(hydrateNeuronState(legacy)).toEqual({
      x1: 0.2,
      x2: 0.9,
      w1: -1.5,
      w2: 0.4,
      bias: -0.1,
      activation: "relu",
      task: "off",
    });
  });

  it("clamps out-of-range values and rejects unknown options", () => {
    const hydrated = hydrateNeuronState(JSON.stringify({ x1: 4, w2: -9, activation: "gelu", task: "nand" }));
    expect(hydrated.x1).toBe(1);
    expect(hydrated.w2).toBe(-2);
    expect(hydrated.activation).toBe("sigmoid");
    expect(hydrated.task).toBe("off");
  });

  it("falls back to defaults on malformed JSON", () => {
    expect(hydrateNeuronState("nope")).toEqual(definition.initialState);
  });
});
