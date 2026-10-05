import { describe, expect, it } from "vitest";
import { TINY_CORPORA, TINY_VOCAB_SIZE, tinyCrossEntropy, trainTinyModel, UNIFORM_CROSS_ENTROPY } from "@app/module-sdk";
import { BLOCK_IDS, BLOCK_SIZE } from "./state";

/**
 * Pins the claims the checkpoint explanations in module.ts make that no other test in this
 * folder states: directions and equalities rather than quoted numbers (quoted.test.ts pins those).
 */
const source = TINY_CORPORA.harbor;
const target = TINY_CORPORA.recipes;
const base = trainTinyModel({ text: source.text, epochs: 60, seed: 1 });

const rowsOf = (id: string) =>
  Array.from({ length: BLOCK_SIZE }, (_, offset) => BLOCK_IDS.indexOf(id) * BLOCK_SIZE + offset).filter(
    (row) => row < TINY_VOCAB_SIZE,
  );

function tune(frozen: string[]) {
  const trainableRows = Array.from({ length: TINY_VOCAB_SIZE }, () => true);
  for (const id of frozen) for (const row of rowsOf(id)) trainableRows[row] = false;
  const run = trainTinyModel({
    text: target.text,
    epochs: 20,
    learningRate: 0.6,
    init: base.weights,
    trainableRows,
    seed: 2,
    checkpoints: 9,
  });
  return {
    run,
    trainableRows,
    targetLoss: tinyCrossEntropy(run.weights, target.text),
    forgetting: tinyCrossEntropy(run.weights, source.text) - tinyCrossEntropy(base.weights, source.text),
  };
}

describe("fine-tuning-transfer checkpoint claims", () => {
  it("freezing the first two blocks lowers forgetting and raises the target loss", () => {
    const free = tune([]);
    const frozen = tune(["block-0", "block-1"]);
    expect(frozen.forgetting).toBeLessThan(free.forgetting);
    expect(frozen.targetLoss).toBeGreaterThan(free.targetLoss);
    // The pair is the one the question names: the blocks labelled with space to e and f to k.
    expect(rowsOf("block-0").length + rowsOf("block-1").length).toBe(12);
  });

  it("leaves every unfrozen row exactly where it would have gone: nothing compensates", () => {
    const free = tune([]);
    const frozen = tune(["block-0", "block-1"]);
    for (let row = 0; row < TINY_VOCAB_SIZE; row += 1) {
      for (let k = 0; k < TINY_VOCAB_SIZE; k += 1) {
        const index = row * TINY_VOCAB_SIZE + k;
        if (frozen.trainableRows[row]) {
          expect(frozen.run.weights[index]).toBeCloseTo(free.run.weights[index], 6);
        } else {
          // A frozen row keeps its pretrained values.
          expect(frozen.run.weights[index]).toBe(base.weights[index]);
        }
      }
    }
  });

  it("gives the pretrained table a head start on recipes before any recipe step", () => {
    const pretrained = tinyCrossEntropy(base.weights, target.text);
    const zeroed = tinyCrossEntropy(new Float32Array(TINY_VOCAB_SIZE * TINY_VOCAB_SIZE), target.text);
    expect(zeroed).toBeCloseTo(UNIFORM_CROSS_ENTROPY, 6);
    expect(pretrained).toBeLessThan(zeroed);
    // Same model size on both sides: the head start is not extra weights.
    expect(base.weights.length).toBe(TINY_VOCAB_SIZE * TINY_VOCAB_SIZE);
  });
});
