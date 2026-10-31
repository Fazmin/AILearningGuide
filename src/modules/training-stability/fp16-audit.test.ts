import { describe, expect, it } from "vitest";
import { trainTinyExtended } from "@app/module-sdk";
import { auditHalfGradients } from "./fp16-audit";
import { healthyOptions, lossScaleFault, TRAINING_TEXT } from "./faults";

const recipe = { text: TRAINING_TEXT, rank: 8, epochs: 20, batchSize: 16, learningRate: 0.5, seed: 9 };

describe("fp16 audit", () => {
  it("replays the 32-bit healthy run bit for bit", () => {
    const audit = auditHalfGradients({ ...recipe, scaleExponent: null });
    const sdk = trainTinyExtended({ ...healthyOptions(), trace: false });
    expect(audit.steps).toBe(480);
    expect(audit.haltedAt).toBeNull();
    expect(Array.from(audit.weights)).toEqual(Array.from(sdk.weights));
    expect(audit.flushed).toBe(0);
  });

  it.each([0, 8, 17])("replays the static fp16 cast at 2^%i bit for bit", (exponent) => {
    const audit = auditHalfGradients({ ...recipe, scaleExponent: exponent });
    const sdk = trainTinyExtended({ ...healthyOptions(), trace: false, faults: [lossScaleFault(exponent, "static")] });
    expect(sdk.haltedAt).toBeNull();
    expect(Array.from(audit.weights)).toEqual(Array.from(sdk.weights));
  });

  it("halts on the same step as the SDK when a static scale overflows", () => {
    for (const [exponent, step] of [
      [18, 184],
      [20, 33],
    ] as const) {
      const audit = auditHalfGradients({ ...recipe, scaleExponent: exponent });
      const sdk = trainTinyExtended({ ...healthyOptions(), trace: false, faults: [lossScaleFault(exponent, "static")] });
      expect([audit.haltedAt, sdk.haltedAt]).toEqual([step, step]);
    }
  });

  it("sees the same nonzero gradient entries with and without the cast, until a run halts", () => {
    const plain = auditHalfGradients({ ...recipe, scaleExponent: null });
    const cast = auditHalfGradients({ ...recipe, scaleExponent: 0 });
    expect(cast.nonzero).toBe(plain.nonzero);
    expect(cast.subnormal).toBeGreaterThan(0);
    expect(auditHalfGradients({ ...recipe, scaleExponent: 20 }).nonzero).toBeLessThan(plain.nonzero);
  });
});
