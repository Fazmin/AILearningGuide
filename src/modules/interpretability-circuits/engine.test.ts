/// <reference types="node" />
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import attentionVocabulary from "../attention/assets/transformer-vocab.json";
import cache from "./assets/interpretability-cache.json";
import { erf, finalLogitsWithPatch, rowOf, runTransformer, weightsFromOnnx } from "./engine";
import { encodeText, ITOS, tokenId } from "./vocabulary";

const onnxPath = resolve(process.cwd(), "src/modules/attention/assets/tiny-transformer.onnx");
const weights = weightsFromOnnx(readFileSync(onnxPath));

describe("ONNX weight reader", () => {
  it("recovers the shipped architecture", () => {
    expect(weights.vocabulary).toBe(66);
    expect(weights.width).toBe(256);
    expect(weights.context).toBe(64);
    expect(weights.feedForward).toBe(768);
    expect(weights.blocks).toHaveLength(2);
    expect(weights.blocks[0].qkvWeight).toHaveLength(256 * 768);
    expect(weights.unembedding).toHaveLength(256 * 66);
  });

  it("uses the same vocabulary as the attention lab", () => {
    expect(ITOS).toEqual(attentionVocabulary.itos);
    expect(tokenId("A")).toBe(attentionVocabulary.stoi.A);
  });

  it("evaluates erf to 1e-7", () => {
    expect(erf(0)).toBeCloseTo(0, 7);
    expect(erf(1)).toBeCloseTo(0.8427007929, 7);
    expect(erf(-2)).toBeCloseTo(-0.9953222650, 7);
  });
});

describe("forward pass parity with the precomputed PyTorch cache", () => {
  it.each(cache.attention.map((record) => [record.id, record] as const))(
    "reproduces every cached attention weight for %s",
    (_id, record) => {
      const run = runTransformer(weights, encodeText(record.text));
      const [layers, , heads, sequence] = record.shape;
      let worst = 0;
      for (let layer = 0; layer < layers; layer += 1) {
        for (let head = 0; head < heads; head += 1) {
          for (let query = 0; query < sequence; query += 1) {
            for (let key = 0; key < sequence; key += 1) {
              const cached = (record.attention as number[][][][])[layer][head][query][key];
              const mine = run.attention[layer][(head * sequence + query) * sequence + key];
              worst = Math.max(worst, Math.abs(cached - mine));
            }
          }
        }
      }
      expect(worst).toBeLessThan(5e-5);
    },
  );

  it.each(cache.activation_patching.map((record) => [record.id, record] as const))(
    "reproduces the clean and corrupted target logits for %s",
    (_id, record) => {
      const length = record.tokens.length;
      const target = tokenId(record.target);
      const clean = runTransformer(weights, encodeText(record.clean.slice(0, length)));
      const corrupt = runTransformer(weights, encodeText(record.corrupt.slice(0, length)));
      const last = length - 1;
      expect(rowOf(clean.logits, last, 66)[target]).toBeCloseTo(record.clean_logit, 3);
      expect(rowOf(corrupt.logits, last, 66)[target]).toBeCloseTo(record.corrupt_logit, 3);
    },
  );

  it("reproduces the cached speaker patching sweep", () => {
    const record = cache.activation_patching.find((entry) => entry.id === "speaker")!;
    const length = record.tokens.length;
    const target = tokenId(record.target);
    const clean = runTransformer(weights, encodeText(record.clean.slice(0, length)));
    const corrupt = runTransformer(weights, encodeText(record.corrupt.slice(0, length)));
    const cleanScore = rowOf(clean.logits, length - 1, 66)[target];
    const corruptScore = rowOf(corrupt.logits, length - 1, 66)[target];
    for (let layer = 0; layer < 2; layer += 1) {
      for (let position = 0; position < length; position += 1) {
        const patched = finalLogitsWithPatch(
          weights,
          corrupt.residuals,
          layer,
          position,
          rowOf(clean.residuals[layer], position, 256),
        );
        const recovery = (patched[target] - corruptScore) / (cleanScore - corruptScore);
        expect(recovery).toBeCloseTo(record.recovery_by_layer_and_position[layer][position], 2);
      }
    }
  });

  it("gives the same final logits from a full patched run and the resumed shortcut", () => {
    const ids = encodeText("QXZRKWMPQXZRKWMPQXZRKWM");
    const other = encodeText("QXZRKWMPQXZRKWMYQXZRKWM");
    const clean = runTransformer(weights, ids);
    const corrupt = runTransformer(weights, other);
    const vector = rowOf(clean.residuals[0], 15, 256);
    const full = runTransformer(weights, other, { patch: { layer: 0, position: 15, vector } });
    const fast = finalLogitsWithPatch(weights, corrupt.residuals, 0, 15, vector);
    const last = rowOf(full.logits, ids.length - 1, 66);
    for (let index = 0; index < 66; index += 1) expect(fast[index]).toBeCloseTo(last[index], 4);
  });
});
