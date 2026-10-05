import { describe, expect, it } from "vitest";
import definition, { hydrateStackingState } from "./module";

describe("stacking-neurons state", () => {
  it("clamps a version 1 payload (depth 1–5, width 2–7, epochs 0–80) into the trainable ranges", () => {
    const legacy = JSON.stringify({ depth: 5, width: 7, epochs: 18 });
    expect(hydrateStackingState(legacy)).toEqual({
      dataset: "xor",
      activation: "tanh",
      depth: 3,
      width: 6,
      epochs: 18,
      seed: 2,
      probeX: 0.6,
      probeY: -0.6,
      noise: "on",
      heldCurve: "accuracy",
    });
  });

  it("reads a version 2 payload (no held-out card) and gives the new keys their defaults", () => {
    const version2 = JSON.stringify({
      dataset: "ring",
      activation: "relu",
      depth: 2,
      width: 4,
      epochs: 90,
      seed: 5,
      probeX: -0.2,
      probeY: 0.3,
    });
    expect(hydrateStackingState(version2)).toEqual({
      dataset: "ring",
      activation: "relu",
      depth: 2,
      width: 4,
      epochs: 90,
      seed: 5,
      probeX: -0.2,
      probeY: 0.3,
      noise: "on",
      heldCurve: "accuracy",
    });
    expect(definition.stateVersion).toBe(3);
  });

  it("keeps valid current payloads and rejects unknown options", () => {
    const hydrated = hydrateStackingState(
      JSON.stringify({
        dataset: "checker",
        activation: "sigmoid",
        epochs: 999,
        seed: 7,
        probeX: -3,
        noise: "off",
        heldCurve: "loss",
      }),
    );
    expect(hydrated.dataset).toBe("checker");
    expect(hydrated.activation).toBe("tanh");
    expect(hydrated.epochs).toBe(300);
    expect(hydrated.seed).toBe(7);
    expect(hydrated.probeX).toBe(-1);
    expect(hydrated.noise).toBe("off");
    expect(hydrated.heldCurve).toBe("loss");
    const rejected = hydrateStackingState(JSON.stringify({ dataset: "spiral", noise: "maybe", heldCurve: 3 }));
    expect(rejected.dataset).toBe("xor");
    expect(rejected.noise).toBe("on");
    expect(rejected.heldCurve).toBe("accuracy");
  });

  it("clamps hostile numbers instead of passing them to the trainer", () => {
    for (const value of [1e9, -1e9, 1e999]) {
      const hydrated = hydrateStackingState(
        JSON.stringify({ depth: value, width: value, epochs: value, seed: value, probeX: value, probeY: value }),
      );
      expect(hydrated.depth).toBeGreaterThanOrEqual(1);
      expect(hydrated.depth).toBeLessThanOrEqual(3);
      expect(hydrated.width).toBeGreaterThanOrEqual(1);
      expect(hydrated.width).toBeLessThanOrEqual(6);
      expect(hydrated.epochs).toBeGreaterThanOrEqual(0);
      expect(hydrated.epochs).toBeLessThanOrEqual(300);
      expect(hydrated.seed).toBeGreaterThanOrEqual(1);
      expect(hydrated.seed).toBeLessThanOrEqual(999);
      expect(Math.abs(hydrated.probeX as number)).toBeLessThanOrEqual(1);
    }
  });

  it("falls back to defaults on malformed JSON", () => {
    expect(hydrateStackingState("{")).toEqual(definition.initialState);
  });
});
