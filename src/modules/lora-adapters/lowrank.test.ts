import { describe, expect, it } from "vitest";
import {
  TINY_CORPORA,
  TINY_VOCAB_SIZE,
  tinyCrossEntropy,
  trainTinyLora,
  trainTinyModel,
} from "@app/module-sdk";
import {
  centerRows,
  checkMerge,
  energyCaptured,
  numericalRank,
  singularValues,
} from "./lowrank";

const V = TINY_VOCAB_SIZE;

describe("singular values", () => {
  it("recovers a diagonal matrix's entries, largest first", () => {
    const values = singularValues(Float64Array.from([3, 0, 0, 0, -5, 0, 0, 0, 1]), 3, 3);
    expect(values[0]).toBeCloseTo(5, 10);
    expect(values[1]).toBeCloseTo(3, 10);
    expect(values[2]).toBeCloseTo(1, 10);
  });

  it("finds rank one in an outer product and reports its energy", () => {
    // [1, 2]ᵀ · [3, 4] has one singular value, |[1, 2]| · |[3, 4]| = √5 · 5.
    const values = singularValues(Float64Array.from([3, 4, 6, 8]), 2, 2);
    expect(values[0]).toBeCloseTo(Math.sqrt(5) * 5, 10);
    expect(values[1]).toBeCloseTo(0, 6);
    expect(numericalRank(values)).toBe(1);
    expect(energyCaptured(values, 1)).toBeCloseTo(1, 10);
  });

  it("centers each row without changing a softmax", () => {
    const centered = centerRows([1, 2, 3, 10, 10, 10, 0, 0, 3], 3);
    expect(Array.from(centered)).toEqual([-1, 0, 1, 0, 0, 0, -1, -1, 2]);
  });
});

describe("a trained LoRA update", () => {
  const base = trainTinyModel({ text: TINY_CORPORA.harbor.text, epochs: 60, seed: 1 });

  it("has rank at most r, before and after centering", () => {
    for (const rank of [1, 4]) {
      const lora = trainTinyLora({ base: base.weights, text: TINY_CORPORA.recipes.text, rank, alpha: 8, epochs: 8, seed: 5 });
      expect(numericalRank(singularValues(lora.delta, V, V))).toBe(rank);
      expect(numericalRank(singularValues(centerRows(lora.delta, V), V, V))).toBeLessThanOrEqual(rank);
    }
  });

  it("merges to the same logits and the same loss as the unmerged path", () => {
    const lora = trainTinyLora({ base: base.weights, text: TINY_CORPORA.recipes.text, rank: 4, alpha: 8, epochs: 8, seed: 5 });
    const check = checkMerge(lora.merged, base.weights, lora.up, lora.down, 4, lora.scale, V, TINY_CORPORA.recipes.text);
    expect(check.maxLogitGap).toBeLessThan(1e-5);
    expect(check.maxLogitGap).toBeGreaterThan(0);
    expect(check.unmergedLoss).toBeCloseTo(tinyCrossEntropy(lora.merged, TINY_CORPORA.recipes.text), 5);
  });

  it("spreads a full fine-tune's update over many directions", () => {
    const full = trainTinyModel({ text: TINY_CORPORA.recipes.text, epochs: 24, init: base.weights, seed: 5 });
    const delta = full.weights.map((value, index) => value - base.weights[index]);
    const values = singularValues(delta, V, V);
    expect(numericalRank(values)).toBeGreaterThan(12);
    expect(energyCaptured(values, 4)).toBeLessThan(0.75);
    expect(energyCaptured(values, 12)).toBeGreaterThan(0.9);
  });
});
