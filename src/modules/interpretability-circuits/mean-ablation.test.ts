/// <reference types="node" />
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { rowOf, runTransformer, weightsFromOnnx } from "./engine";
import { headScaleFor } from "./experiments";
import {
  headOutputs,
  meanAblatedWeights,
  meanHeadOutputs,
  REFERENCE_STRINGS,
  referenceStrings,
} from "./mean-ablation";
import { encodeText } from "./vocabulary";

const weights = weightsFromOnnx(
  readFileSync(resolve(process.cwd(), "src/modules/attention/assets/tiny-transformer.onnx")),
);
const IDS = ["L1H1", "L1H2", "L1H3", "L1H4", "L2H1", "L2H2", "L2H3", "L2H4"];
const maxDifference = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let worst = 0;
  for (let index = 0; index < a.length; index += 1) worst = Math.max(worst, Math.abs(a[index] - b[index]));
  return worst;
};

describe("the reference batch for mean ablation", () => {
  it("is deterministic, periodic, made of distinct capitals, and independent across seed offsets", () => {
    const first = referenceStrings(9, 24);
    // The lesson and the card explanations say "eight".
    expect(REFERENCE_STRINGS).toBe(8);
    expect(first).toHaveLength(REFERENCE_STRINGS);
    expect(referenceStrings(9, 24)).toEqual(first);
    for (const text of first) {
      expect(text).toHaveLength(24);
      expect(text.slice(0, 9)).toBe(text.slice(9, 18));
      expect(text.slice(0, 6)).toBe(text.slice(18, 24));
      expect(new Set(text.slice(0, 9)).size).toBe(9);
      expect(text).toMatch(/^[A-Z]+$/);
    }
    expect(new Set(first).size).toBe(first.length);
    expect(referenceStrings(9, 24, REFERENCE_STRINGS, 100000)).not.toEqual(first);
    expect(referenceStrings(13, 24)[0]).not.toBe(first[0]);
  });
});

describe("mean ablation, built from the runner's own interventions", () => {
  it("is exactly the identity when a head is replaced by its own output (one-token prompt, every head)", () => {
    const ids = encodeText("Q");
    const intact = runTransformer(weights, ids);
    const means = meanHeadOutputs(weights, ["Q"], 0);
    for (const id of IDS) {
      const headScale = headScaleFor([id]);
      const run = runTransformer(meanAblatedWeights(weights, means, headScale), ids, { headScale });
      expect(maxDifference(run.logits, intact.logits), id).toBeLessThan(1e-3);
      // Zeroing instead of replacing is a real intervention on the same prompt, so the check can fail.
      const zeroed = runTransformer(weights, ids, { headScale });
      expect(maxDifference(zeroed.logits, intact.logits), `${id} zeroed`).toBeGreaterThan(1e-2);
    }
  });

  it("is exactly the identity for a layer-2 head replaced by its own output at the last position", () => {
    const text = "QXZRK";
    const ids = encodeText(text);
    const intact = runTransformer(weights, ids);
    const last = ids.length - 1;
    const means = meanHeadOutputs(weights, [text], last);
    for (const id of IDS.slice(4)) {
      const headScale = headScaleFor([id]);
      const run = runTransformer(meanAblatedWeights(weights, means, headScale), ids, { headScale });
      expect(maxDifference(rowOf(run.logits, last, 66), rowOf(intact.logits, last, 66)), id).toBeLessThan(1e-3);
      const zeroed = runTransformer(weights, ids, { headScale });
      expect(maxDifference(rowOf(zeroed.logits, last, 66), rowOf(intact.logits, last, 66)), `${id} zeroed`).toBeGreaterThan(1e-2);
    }
  });

  it("returns one [position, head width] output per head and layer", () => {
    // Block 1 is checked above at one token (values from the embeddings, attention weight 1) and block 2
    // at five tokens (values from block 1's residual stream, attention mixing several keys).
    const ids = encodeText("QXZRKWMPQXZRKWMP");
    const outputs = headOutputs(weights, ids, runTransformer(weights, ids));
    expect(outputs).toHaveLength(2);
    for (const layer of outputs) {
      expect(layer).toHaveLength(4);
      for (const head of layer) expect(head).toHaveLength(ids.length * 64);
    }
  });

  it("leaves the weights alone and returns them unchanged when nothing is ablated", () => {
    const before = Float32Array.from(weights.blocks[1].outputBias);
    const means = meanHeadOutputs(weights, referenceStrings(8, 24).slice(0, 2), 8);
    const ablated = meanAblatedWeights(weights, means, headScaleFor(["L2H3"]));
    expect(ablated.blocks[1].outputBias).not.toBe(weights.blocks[1].outputBias);
    expect(maxDifference(ablated.blocks[1].outputBias, before)).toBeGreaterThan(0);
    expect(maxDifference(weights.blocks[1].outputBias, before)).toBe(0);
    expect(ablated.blocks[0]).toBe(weights.blocks[0]);
    expect(meanAblatedWeights(weights, means, headScaleFor([])).blocks[1]).toBe(weights.blocks[1]);
  });
});
