import { describe, expect, it } from "vitest";
import { TINY_CORPORA, TINY_VOCAB, tinyCrossEntropy, trainTinyModel } from "@app/module-sdk";
import { applyBatchStep, stepBatch, stepFromCheckpoint } from "./step";

/** Pins the Step one batch replay against the trainer, and every figure the lesson and card text quote. */
const text = TINY_CORPORA.harbor.text;
const PAIRS = 498;
/** A run that saves a checkpoint and a history point at every step, so each update can be compared. */
const everyStep = (batchSize: number, epochs: number) => {
  const steps = Math.ceil(PAIRS / batchSize) * epochs;
  return trainTinyModel({ text, epochs, batchSize, learningRate: 0.6, seed: 1, checkpoints: steps + 1, historyPoints: steps });
};
const defaults = trainTinyModel({ text, epochs: 12, batchSize: 16, learningRate: 0.6, seed: 1, checkpoints: 9 });

describe("step replay matches the trainer", () => {
  it.each([2, 16, 64])("reproduces every update of a batch-%i run to the last bit", (batchSize) => {
    const run = everyStep(batchSize, 3);
    expect(run.checkpoints).toHaveLength(run.steps + 1);
    for (let index = 0; index < run.steps; index += 1) {
      const from = run.checkpoints[index];
      const to = run.checkpoints[index + 1];
      const result = applyBatchStep(from.weights, stepBatch(text, batchSize, 1, from.step));
      expect(Array.from(result.after), `weights after step ${from.step + 1}`).toEqual(Array.from(to.weights));
      expect(result.lossBefore).toBeCloseTo(run.history[index].loss, 6);
      expect(result.gradientNorm).toBeCloseTo(run.history[index].gradientNorm, 6);
    }
  });

  it("draws the same batch for every step it is asked about, and a full epoch covers each pair once", () => {
    const first = stepBatch(text, 16, 1, 5);
    const again = stepBatch(text, 16, 1, 5);
    expect(again).toEqual(first);
    const seen: string[] = [];
    for (let batch = 0; batch < 32; batch += 1) {
      const drawn = stepBatch(text, 16, 1, batch);
      drawn.inputs.forEach((input, index) => seen.push(`${input}:${drawn.targets[index]}`));
    }
    expect(seen).toHaveLength(PAIRS);
    expect(stepBatch(text, 16, 1, 31).inputs).toHaveLength(2);
  });

  it("is still defined one step past the end of the run, in the next epoch", () => {
    const beyond = stepBatch(text, 16, 1, defaults.steps);
    expect(beyond.epoch).toBe(12);
    expect(beyond.batchInEpoch).toBe(0);
    expect(beyond.inputs).toHaveLength(16);
  });

  it("leaves rows the batch never touched exactly where they were", () => {
    const result = applyBatchStep(defaults.checkpoints[4].weights, stepBatch(text, 16, 1, defaults.checkpoints[4].step));
    const touched = new Set(result.rows.map((row) => row.row));
    for (let row = 0; row < 30; row += 1) {
      for (let k = 0; k < 30; k += 1) {
        if (!touched.has(row)) expect(result.after[row * 30 + k]).toBe(result.before[row * 30 + k]);
      }
    }
    expect(result.weightsChanged).toBe(result.rows.length * 30);
  });

  it("chains steps from a checkpoint and returns the last one", () => {
    const start = defaults.checkpoints[0];
    expect(stepFromCheckpoint(start.weights, start.step, 0, text, 16, 1)).toBeNull();
    const second = stepFromCheckpoint(start.weights, start.step, 2, text, 16, 1)!;
    const first = stepFromCheckpoint(start.weights, start.step, 1, text, 16, 1)!;
    expect(second.batch.step).toBe(1);
    expect(Array.from(second.before)).toEqual(Array.from(first.after));
  });
});

describe("step quoted values", () => {
  const nameOf = (id: number) => TINY_VOCAB[id];

  it("matches the first update from the untrained checkpoint", () => {
    const start = defaults.checkpoints[0];
    const result = stepFromCheckpoint(start.weights, start.step, 1, text, 16, 1)!;
    expect(result.batch.epoch).toBe(0);
    expect(result.batch.batchInEpoch).toBe(0);
    expect(result.batch.inputs).toHaveLength(16);
    expect(result.lossBefore).toBeCloseTo(3.401, 3);
    expect(result.lossAfter).toBeCloseTo(3.362, 3);
    expect(result.lossBefore - result.lossAfter).toBeCloseTo(0.039, 3);
    expect(result.gradientNorm).toBeCloseTo(0.2567, 4);
    expect(tinyCrossEntropy(result.before, text)).toBeCloseTo(3.401, 3);
    expect(tinyCrossEntropy(result.after, text)).toBeCloseTo(3.396, 3);
    expect(tinyCrossEntropy(result.before, text) - tinyCrossEntropy(result.after, text)).toBeCloseTo(0.005, 3);
    expect(result.rows).toHaveLength(9);
    expect(result.weightsChanged).toBe(270);
    expect(nameOf(result.rows[0].row)).toBe("s");
    expect(result.rows[0].pairs).toBe(3);
    expect(result.rows[0].gradientNorm).toBeCloseTo(0.135, 3);
    expect(result.rows[0].logitBefore).toBe(0);
    expect(result.rows[0].logitAfter).toBeCloseTo(0.071, 3);
  });

  it("matches the first update past the final checkpoint", () => {
    const end = defaults.checkpoints[defaults.checkpoints.length - 1];
    expect(end.step).toBe(384);
    const result = stepFromCheckpoint(end.weights, end.step, 1, text, 16, 1)!;
    expect(result.lossBefore).toBeCloseTo(2.274, 3);
    expect(result.lossAfter).toBeCloseTo(2.248, 3);
    expect(result.gradientNorm).toBeCloseTo(0.2104, 4);
    expect(tinyCrossEntropy(result.before, text)).toBeCloseTo(2.277, 3);
    expect(tinyCrossEntropy(result.after, text)).toBeCloseTo(2.277, 3);
  });

  it("never raises the loss on its own batch, but can raise the whole-corpus loss at small batches", () => {
    const tally = (batchSize: number) => {
      const run = everyStep(batchSize, 12);
      let wholeCorpusUp = 0;
      let ownBatchUp = 0;
      for (let index = 0; index < run.steps; index += 1) {
        const from = run.checkpoints[index];
        const result = applyBatchStep(from.weights, stepBatch(text, batchSize, 1, from.step));
        if (run.checkpoints[index + 1].loss > from.loss) wholeCorpusUp += 1;
        if (result.lossAfter > result.lossBefore) ownBatchUp += 1;
      }
      return { steps: run.steps, wholeCorpusUp, ownBatchUp };
    };
    expect(tally(2)).toEqual({ steps: 2988, wholeCorpusUp: 1174, ownBatchUp: 0 });
    expect(tally(16)).toEqual({ steps: 384, wholeCorpusUp: 5, ownBatchUp: 0 });
    expect(tally(64)).toEqual({ steps: 96, wholeCorpusUp: 0, ownBatchUp: 0 });
  }, 90_000);

  it("shows the noise when a learner chains presses from the middle checkpoint at batch size 2", () => {
    const run = trainTinyModel({ text, epochs: 12, batchSize: 2, learningRate: 0.6, seed: 1, checkpoints: 9 });
    const middle = run.checkpoints[4];
    expect(middle.step).toBe(1492);
    let rises = 0;
    let before = tinyCrossEntropy(middle.weights, text);
    for (let taken = 1; taken <= 24; taken += 1) {
      const result = stepFromCheckpoint(middle.weights, middle.step, taken, text, 2, 1)!;
      expect(result.lossAfter).toBeLessThan(result.lossBefore);
      const after = tinyCrossEntropy(result.after, text);
      if (after > before) rises += 1;
      before = after;
    }
    expect(rises).toBe(17);
    // At the default batch size of 16, the same 24 presses from the same relative checkpoint never raise it.
    const steady = defaults.checkpoints[4];
    expect(steady.step).toBe(192);
    let steadyRises = 0;
    before = tinyCrossEntropy(steady.weights, text);
    for (let taken = 1; taken <= 24; taken += 1) {
      const after = tinyCrossEntropy(stepFromCheckpoint(steady.weights, steady.step, taken, text, 16, 1)!.after, text);
      if (after > before) steadyRises += 1;
      before = after;
    }
    expect(steadyRises).toBe(0);
  }, 60_000);
});
